"""
DINOv2-RAE Range View Model  (two-stage pipeline inspired by bytetriper/RAE)

Stage 1 – Reconstruction Autoencoder (RAE):
  Frozen DINOv2 encoder  →  trainable ViT-XL decoder
  Input/output: [B, 5, 64, 2048]  (range, x, y, z, intensity)
  Latent:       [B, 256, 384]     (8×32 DINOv2 patch tokens)

Stage 2 – Latent Diffusion Transformer (forecasting):
  DINOv2 tokens → STT (temporal context) → FluxDiT (rectified flow) → ViT-XL decoder
  Chain-of-forward autoregressive training is preserved from the original pipeline.

Key RAE decoder design follows bytetriper/RAE src/stage1/decoders/decoder.py (ViTXL config):
  decoder_embed  Linear(384 → 1152)
  28 × ViT-XL blocks  (hidden=1152, heads=16, FFN=4096)
  prediction head  Linear(1152 → 5×8×64=2560)
  unpatchify  →  [B, 5, 64, 2048]
"""

import math
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.stt import SpatialTemporalTransformer
from models.flux_dit import FluxParams, FluxDiT
from models.modules.tokenizer import poses_to_indices, yaws_to_indices
from models.modules.sampling import prepare_ids, get_schedule
from utils.preprocess import get_rel_pose
from utils.range_losses import (
    range_view_l1_loss,
    RangeViewProjection, batch_chamfer_distance,
)
from utils.bev_perceptual import BEVPerceptualLoss

# ── ConvStem (DiffLoc/SalsaNext-style, extended to arbitrary in_channels) ──

class _ResContextBlock(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.conv1 = nn.Conv2d(in_f, out_f, 1)
        self.conv2 = nn.Conv2d(out_f, out_f, 3, padding=1)
        self.conv3 = nn.Conv2d(out_f, out_f, 3, dilation=2, padding=2)
        self.bn1 = nn.BatchNorm2d(out_f); self.bn2 = nn.BatchNorm2d(out_f)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        s = self.act(self.conv1(x))
        r = self.bn1(self.act(self.conv2(s)))
        r = self.bn2(self.act(self.conv3(r)))
        return s + r


class _ResBlock(nn.Module):
    def __init__(self, in_f, out_f, drop=0.2):
        super().__init__()
        self.s  = nn.Conv2d(in_f, out_f, 1)
        self.c2 = nn.Conv2d(in_f, out_f, 3, padding=1)
        self.c3 = nn.Conv2d(out_f, out_f, 3, dilation=2, padding=2)
        self.c4 = nn.Conv2d(out_f, out_f, 2, dilation=2, padding=1)
        self.c5 = nn.Conv2d(out_f * 3, out_f, 1)
        self.bn = nn.ModuleList([nn.BatchNorm2d(out_f) for _ in range(4)])
        self.act = nn.LeakyReLU(); self.dropout = nn.Dropout2d(p=drop)

    def forward(self, x):
        s  = self.act(self.s(x))
        r1 = self.bn[0](self.act(self.c2(x)))
        r2 = self.bn[1](self.act(self.c3(r1)))
        r3 = self.bn[2](self.act(self.c4(r2)))
        r  = self.bn[3](self.act(self.c5(torch.cat([r1, r2, r3], 1))))
        return self.dropout(s + r)


class ConvStem(nn.Module):
    """DiffLoc ConvStem: [B, C, H, W] → [B, N_patches, embed_dim].

    patch_stride=(8,64) on 64×2048 images → 8×32 = 256 patch tokens.
    """
    def __init__(self, in_channels=5, base=32, embed_dim=384,
                 hidden=256, patch_stride=(8, 64), img_size=(64, 2048)):
        super().__init__()
        self.grid   = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])
        self.n_patches = self.grid[0] * self.grid[1]
        self.conv_block = nn.Sequential(
            _ResContextBlock(in_channels, base),
            _ResContextBlock(base, base),
            _ResContextBlock(base, base),
            _ResBlock(base, hidden, drop=0.2),
        )
        k = (patch_stride[0]+1, patch_stride[1]+1)
        p = (patch_stride[0]//2, patch_stride[1]//2)
        self.proj = nn.Sequential(
            nn.AvgPool2d(k, stride=patch_stride, padding=p),
            nn.Conv2d(hidden, embed_dim, 1),
        )

    def forward(self, x):
        return self.proj(self.conv_block(x)).flatten(2).transpose(1, 2)


# ── DINOv2 Encoder ─────────────────────────────────────────────────────────

DINO_EMBED_DIM  = 384
DINO_GRID_H, DINO_GRID_W = 8, 32    # 64//8, 2048//64
DINO_N_PATCHES  = DINO_GRID_H * DINO_GRID_W  # 256


class RangeViewDINOv2Encoder(nn.Module):
    """Frozen DINOv2 encoder for 5-channel range views.

    Mirrors DiffLoc ImageFeatureExtractor but returns all patch tokens
    [B, 256, 384] instead of only the CLS token, so they can serve as the
    latent representation for the RAE decoder and DiT.

    ConvStem (randomly initialised; trained in Stage 1 as part of RAE):
      [B, 5, 64, 2048] → [B, 256, 384]
    DINOv2 ViT-S/14 blocks (pretrained, always frozen):
      [B, 256, 384] → [B, 256, 384]
    """
    def __init__(self, in_channels: int = 5, pretrained_path: str = None):
        super().__init__()
        self.conv_stem = ConvStem(in_channels=in_channels, base=32,
                                  embed_dim=DINO_EMBED_DIM, hidden=256,
                                  patch_stride=(8, 64), img_size=(64, 2048))

        # Load DINOv2 ViT-S/14 pretrained weights
        if pretrained_path is not None:
            dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14',
                                  pretrained=False, verbose=False)
            sd = torch.load(pretrained_path, map_location='cpu')
            sd = sd.get('model', sd)
            # Skip patch_embed — ConvStem replaces it (DiffLoc convention)
            dino.load_state_dict({k: v for k, v in sd.items()
                                   if 'patch_embed' not in k}, strict=False)
        else:
            dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14',
                                  pretrained=True, verbose=False)

        self.cls_token = nn.Parameter(dino.cls_token.data.clone())
        self.pos_embed = nn.Parameter(dino.pos_embed.data.clone())  # [1,257,384]
        self.blocks    = dino.blocks   # 12 × pretrained ViT-S/14 blocks
        self.norm      = dino.norm

        # Freeze DINOv2 transformer components (ConvStem stays trainable)
        for p in [self.cls_token, self.pos_embed]:
            p.requires_grad_(False)
        for p in list(self.blocks.parameters()) + list(self.norm.parameters()):
            p.requires_grad_(False)

    def _interp_pos_embed(self, gh, gw):
        """Bilinear-resize pretrained (16×16) pos_embed to (gh×gw)."""
        cls = self.pos_embed[:, :1]
        grid = self.pos_embed[:, 1:]
        gs = int(math.sqrt(grid.shape[1]))  # 16
        grid = grid.reshape(1, gs, gs, -1).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, (gh, gw), mode='bilinear', align_corners=False)
        return torch.cat([cls, grid.permute(0,2,3,1).reshape(1, gh*gw, -1)], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,5,64,2048] → patch tokens: [B,256,384]."""
        patches = self.conv_stem(x)                           # [B,256,384]
        B = patches.shape[0]
        tok = torch.cat([self.cls_token.expand(B,-1,-1), patches], 1)  # [B,257,384]
        tok = tok + self._interp_pos_embed(DINO_GRID_H, DINO_GRID_W)
        for blk in self.blocks:
            tok = blk(tok)
        return self.norm(tok)[:, 1:]   # patch tokens only: [B,256,384]


# ── ViT-XL Decoder (RAE-inspired, bytetriper/RAE configs/decoder/ViTXL) ───

class _ViTLayer(nn.Module):
    """Pre-norm transformer block (from RAE decoder.py ViTMAELayer)."""
    def __init__(self, h, n_heads, ffn):
        super().__init__()
        self.n1 = nn.LayerNorm(h); self.attn = nn.MultiheadAttention(h, n_heads, batch_first=True)
        self.n2 = nn.LayerNorm(h); self.ff = nn.Sequential(nn.Linear(h, ffn), nn.GELU(), nn.Linear(ffn, h))

    def forward(self, x):
        x = x + self.attn(*[self.n1(x)]*3)[0]
        return x + self.ff(self.n2(x))


class ViTXLDecoder(nn.Module):
    """ViT-XL decoder: [B,256,384] → [B,5,64,2048].

    Config (RAE ViTXL): 28 layers, hidden=1152, heads=16, FFN=4096.
    Output: 5 channels (range, x, y, z, intensity) matching encoder input.
    Each patch token reconstructs an 8×64 pixel region (2560 values).
    """
    HIDDEN   = 1152
    N_HEADS  = 16
    FFN_DIM  = 4096
    N_LAYERS = 28
    OUT_CH   = 5       # range, x, y, z, intensity
    PATCH_H, PATCH_W = 8, 64   # ConvStem patch_stride

    def __init__(self, enc_dim: int = DINO_EMBED_DIM):
        super().__init__()
        patch_px = self.OUT_CH * self.PATCH_H * self.PATCH_W   # 2560
        self.embed   = nn.Linear(enc_dim, self.HIDDEN)
        self.pos_emb = nn.Parameter(torch.zeros(1, DINO_N_PATCHES, self.HIDDEN))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.blocks  = nn.ModuleList([
            _ViTLayer(self.HIDDEN, self.N_HEADS, self.FFN_DIM) for _ in range(self.N_LAYERS)
        ])
        self.norm = nn.LayerNorm(self.HIDDEN)
        self.pred = nn.Linear(self.HIDDEN, patch_px)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B,256,384] → [B,5,64,2048]."""
        x = self.embed(z) + self.pos_emb                 # [B,256,1152]
        for blk in self.blocks:
            x = blk(x)
        x = self.pred(self.norm(x))                      # [B,256,2560]
        B = x.shape[0]
        x = x.reshape(B, DINO_GRID_H, DINO_GRID_W, self.OUT_CH, self.PATCH_H, self.PATCH_W)
        return x.permute(0,3,1,4,2,5).reshape(
            B, self.OUT_CH, DINO_GRID_H*self.PATCH_H, DINO_GRID_W*self.PATCH_W
        )   # [B,5,64,2048]


# ── Stage 1: RAE ────────────────────────────────────────────────────────────

class RangeViewRAE(nn.Module):
    """Stage 1: frozen DINOv2 encoder + trainable ViT-XL decoder.

    Losses: per-channel weighted L1 + optional BEV perceptual (VGG16).
    The ConvStem inside the encoder IS trainable here — it adapts to 5-channel
    range views and projects them into DINOv2's embedding space.
    """
    # Default per-channel L1 weights: range gets highest weight, matching RangeLDM convention
    DEFAULT_CH_WEIGHTS = [40., 10., 10., 10., 5.]

    def __init__(self, args, local_rank=-1):
        super().__init__()
        self.encoder = RangeViewDINOv2Encoder(
            in_channels=int(getattr(args, 'range_channels', 5)),
            pretrained_path=getattr(args, 'dino_pretrained_path', None),
        )
        self.decoder = ViTXLDecoder(enc_dim=DINO_EMBED_DIM)
        ch_w = list(getattr(args, 'rae_ch_weights', self.DEFAULT_CH_WEIGHTS))
        self.register_buffer('ch_weights', torch.tensor(ch_w, dtype=torch.float32))

        self.proj_img_mean = list(getattr(args, 'proj_img_mean',
                                          [10.839, 0., 0., 0., 0.]))
        self.proj_img_stds = list(getattr(args, 'proj_img_stds',
                                          [9.314, 10., 10., 2., 1.]))
        self.log_range     = bool(getattr(args, 'log_range', False))

        # Optional BEV perceptual loss (on range channel)
        self.bev_percep_weight = float(getattr(args, 'bev_perceptual_weight', 0.0))
        if self.bev_percep_weight > 0:
            projector = RangeViewProjection(
                fov_up=float(getattr(args, 'fov_up', 3.0)),
                fov_down=float(getattr(args, 'fov_down', -25.0)),
                H=int(getattr(args, 'range_h', 64)),
                W=int(getattr(args, 'range_w', 2048)),
            )
            self.bev_fn = BEVPerceptualLoss(
                projector=projector,
                bev_h=int(getattr(args, 'bev_h', 256)),
                bev_w=int(getattr(args, 'bev_w', 256)),
                x_range=float(getattr(args, 'bev_x_range', 25.6)),
                y_range=float(getattr(args, 'bev_y_range', 25.6)),
            )
            self.log_w_bev = nn.Parameter(
                torch.tensor(math.log(1.0 / self.bev_percep_weight)))
        else:
            self.bev_fn = None; self.log_w_bev = None

    def encode(self, x): return self.encoder(x)    # [B,5,H,W] → [B,256,384]
    def decode(self, z): return self.decoder(z)    # [B,256,384] → [B,5,H,W]

    def forward(self, x: torch.Tensor):
        """x: [B,5,64,2048] → reconstruction losses dict."""
        z   = self.encode(x)
        rec = self.decode(z)

        loss_rec = (((rec - x).abs()) * self.ch_weights[None,:,None,None]).mean()
        loss_all = loss_rec
        loss_bev = torch.zeros(1, device=x.device)

        if self.bev_fn is not None:
            if self.log_range:
                d_pred = torch.exp2(rec[:,0] * 6.) - 1.   # [B, H, W]
                d_gt   = torch.exp2(x[:,0]  * 6.) - 1.
            else:
                mean0, std0 = self.proj_img_mean[0], self.proj_img_stds[0]
                d_pred = rec[:,0] * std0 + mean0           # [B, H, W]
                d_gt   = x[:,0]  * std0 + mean0
            # valid mask: rec[:,0] is already the range channel [B,H,W],
            # so compare directly rather than using make_valid_mask (which
            # expects channel-last features and would index the W axis instead).
            vp = (d_pred > 0.5)   # [B, H, W]
            vg = (d_gt   > 0.5)   # [B, H, W]
            B = d_pred.shape[0]
            loss_bev = self.bev_fn(
                d_pred.reshape(B, -1),
                d_gt.reshape(B, -1),
                vp.reshape(B, -1),
                vg.reshape(B, -1),
            )
            lw = self.log_w_bev.clamp(min=0.)
            loss_all = loss_all + torch.exp(-lw) * loss_bev + lw

        return {'loss_all': loss_all, 'loss_rec': loss_rec,
                'loss_bev': loss_bev, 'x_rec': rec}

    def save_model(self, path, step, rank=0):
        if rank == 0:
            torch.save({'model_state_dict': self.state_dict(), 'step': step},
                       f'{path}/rae_step{step}.pkl')


# ── Stage 2: RangeViewDINODiT ───────────────────────────────────────────────

class RangeViewDINODiT(nn.Module):
    """Stage 2: FluxDiT forecasting in DINOv2 latent space.

    Architecture identical to RangeViewDiT but the VAE tokenizer is replaced
    by the frozen DINOv2 encoder + ViT-XL decoder pair from Stage 1.
    Chain-of-forward autoregressive training is fully preserved.

    Latent space: [B, T, 256, 384]  (256 tokens × 384-d per frame)
    Token grid  : 8×32 (DINOv2 ConvStem patch_stride=(8,64) on 64×2048)
    """
    def __init__(self, args, local_rank=-1, load_path=None):
        super().__init__()
        self.args = args

        # Token geometry
        self.h, self.w      = DINO_GRID_H, DINO_GRID_W
        self.img_token_size = DINO_N_PATCHES       # 256
        self.dino_emb_dim   = DINO_EMBED_DIM       # 384
        self.condition_frames = int(getattr(args, 'condition_frames', 5))

        self.pose_token_size  = 2 * int(getattr(args, 'block_size', 1))
        self.yaw_token_size   = 1 * int(getattr(args, 'block_size', 1))
        self.total_token_size = self.img_token_size + self.pose_token_size + self.yaw_token_size

        # Pose vocab
        self.pose_x_vocab_size = int(getattr(args, 'pose_x_vocab_size', 128))
        self.pose_y_vocab_size = int(getattr(args, 'pose_y_vocab_size', 128))
        self.yaw_vocab_size    = int(getattr(args, 'yaw_vocab_size',    512))
        self.pose_x_bound      = float(getattr(args, 'pose_x_bound',   50.0))
        self.pose_y_bound      = float(getattr(args, 'pose_y_bound',   10.0))
        self.yaw_bound         = float(getattr(args, 'yaw_bound',      12.0))

        tok_dict = {
            'img_tokens_size':   self.img_token_size,
            'pose_tokens_size':  self.pose_token_size,
            'yaw_token_size':    self.yaw_token_size,
            'total_tokens_size': self.total_token_size,
        }

        # ── DINOv2 encoder: always frozen during Stage 2 ────────────────────
        self.dino_encoder = RangeViewDINOv2Encoder(
            in_channels=int(getattr(args, 'range_channels', 5)),
            pretrained_path=getattr(args, 'dino_pretrained_path', None),
        )
        for p in self.dino_encoder.parameters():
            p.requires_grad_(False)

        # ── ViT-XL decoder: load from Stage 1 RAE checkpoint and freeze ────
        self.decoder = ViTXLDecoder(enc_dim=DINO_EMBED_DIM)
        rae_ckpt = getattr(args, 'rae_ckpt', None)
        if rae_ckpt is not None:
            sd = torch.load(rae_ckpt, map_location='cpu').get('model_state_dict', {})
            dec_sd = {k.replace('decoder.', ''): v for k, v in sd.items()
                      if k.startswith('decoder.')}
            self.decoder.load_state_dict(dec_sd, strict=True)
            for p in self.decoder.parameters():
                p.requires_grad_(False)
            print(f"[DINODiT] ViT-XL decoder loaded & frozen from {rae_ckpt}")
        else:
            print("[DINODiT] No rae_ckpt — decoder is randomly initialised. "
                  "Run Stage 1 first for best results.")

        # ── STT (same as RangeViewDiT; vae_emb_dim → 384 instead of 256) ───
        self.model = SpatialTemporalTransformer(
            block_size=self.condition_frames * self.total_token_size,
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
            pose_x_vocab_size=self.pose_x_vocab_size,
            pose_y_vocab_size=self.pose_y_vocab_size,
            yaw_vocab_size=self.yaw_vocab_size,
            latent_size=(self.h, self.w),
            local_rank=local_rank,
            condition_frames=self.condition_frames,
            token_size_dict=tok_dict,
            vae_emb_dim=self.dino_emb_dim,  # 384
            temporal_block=int(getattr(args, 'block_size', 1)),
        )
        self.model.cuda()

        # ── FluxDiT (in_channels = 384 instead of 256) ─────────────────────
        # vec_in_dim = n_embd × (pose+yaw tokens) = 1024×3 = 3072 (unchanged)
        self.dit = FluxDiT(FluxParams(
            in_channels=self.dino_emb_dim,
            out_channels=self.dino_emb_dim,
            vec_in_dim=args.n_embd * (self.total_token_size - self.img_token_size),
            vec_hidden_dim=args.n_embd,
            context_in_dim=args.n_embd,
            hidden_size=args.n_embd_dit,
            mlp_ratio=float(getattr(args, 'mlp_ratio_dit', 4.0)),
            num_heads=args.n_head_dit,
            depth=args.n_layer[1],
            depth_single_blocks=args.n_layer[2],
            axes_dim=args.axes_dim_dit,
            theta=10_000, qkv_bias=True, guidance_embed=False,
            drop_path_rate=float(getattr(args, 'drop_path_rate', 0.0)),
        ))
        self.dit.cuda()

        # ── Latent scale (DINOv2 features are near unit-norm → 1.0 is fine) ─
        self.register_buffer('latent_scale',
                             torch.tensor(float(getattr(args, 'latent_scale', 1.0))))

        # ── Auxiliary loss config (mirrors RangeViewDiT) ────────────────────
        self.proj_img_mean          = list(getattr(args, 'proj_img_mean', [10.839,0.,0.,0.,0.]))
        self.proj_img_stds          = list(getattr(args, 'proj_img_stds', [9.314,10.,10.,2.,1.]))
        self.log_range              = bool(getattr(args, 'log_range', False))
        self.range_view_loss_weight = float(getattr(args, 'range_view_loss_weight', 0.0))
        self.chamfer_loss_weight    = float(getattr(args, 'chamfer_loss_weight',    0.0))
        self.chamfer_max_pts        = int(getattr(args,   'chamfer_max_pts',        2048))
        self.chamfer_start          = int(getattr(args,   'chamfer_start',          0))
        self.bev_perceptual_weight  = float(getattr(args, 'bev_perceptual_weight',  0.0))

        def _log_w_param(w):
            return nn.Parameter(torch.tensor(math.log(1.0 / w))) if w > 0 else None

        self.log_w_l1      = _log_w_param(self.range_view_loss_weight)
        self.log_w_chamfer = _log_w_param(self.chamfer_loss_weight)
        self.log_w_bev     = _log_w_param(self.bev_perceptual_weight)

        self.range_projector = RangeViewProjection(
            fov_up=float(getattr(args, 'fov_up', 3.0)),
            fov_down=float(getattr(args, 'fov_down', -25.0)),
            H=int(getattr(args, 'range_h', 64)),
            W=int(getattr(args, 'range_w', 2048)),
        )
        self.bev_perceptual_fn = (
            BEVPerceptualLoss(
                projector=self.range_projector,
                bev_h=int(getattr(args, 'bev_h', 256)),
                bev_w=int(getattr(args, 'bev_w', 256)),
                x_range=float(getattr(args, 'bev_x_range', 25.6)),
                y_range=float(getattr(args, 'bev_y_range', 25.6)),
            ) if self.bev_perceptual_weight > 0 else None
        )

        # Precomputed RoPE IDs for DiT forward
        _pfx = self.total_token_size - self.img_token_size  # 3
        bs = args.batch_size * self.condition_frames
        self.img_ids, self.cond_ids, _ = prepare_ids(
            bs, self.h, self.w, self.total_token_size, 0, prefix_size=_pfx
        )

        if load_path is not None:
            self._load_ckpt(load_path)

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _load_ckpt(self, path):
        sd = torch.load(path, map_location='cpu').get('model_state_dict', {})
        for attr, pfx in [('model', 'module.model.'), ('dit', 'module.dit.')]:
            obj = getattr(self, attr)
            obj_sd = {k: sd[pfx+k] for k in obj.state_dict() if pfx+k in sd}
            obj.load_state_dict(obj_sd, strict=False)
        print(f"[DINODiT] STT+DiT loaded from {path}")

    @staticmethod
    def _uw(log_w, loss):
        """Kendall et al. (NeurIPS 2018) uncertainty-weighted loss."""
        lw = log_w.clamp(min=0.)
        return torch.exp(-lw) * loss + lw

    @torch.no_grad()
    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """[B,T,5,H,W] → [B,T,256,384] DINOv2 patch tokens."""
        B, T, C, H, W = x.shape
        z = self.dino_encoder(rearrange(x, 'b t c h w -> (b t) c h w'))
        return rearrange(z, '(b t) n d -> b t n d', b=B, t=T)

    def decode_latents(self, z: torch.Tensor) -> torch.Tensor:
        """[B,256,384] → [B,5,64,2048] (denormalised before decoder)."""
        return self.decoder(z * self.latent_scale)

    # ── Training step ────────────────────────────────────────────────────────

    def step_train(
        self, features, rot_matrix, features_gt,
        rel_pose_cond=None, rel_yaw_cond=None,
        features_aug=None, step=0,
        latents_cond_precomputed=None,
    ):
        """One training step.

        features:                 [B, CF, 5, H, W]
        rot_matrix:               [B, T, 4, 4]
        features_gt:              [B, 1, 5, H, W]
        latents_cond_precomputed: [B, CF, 256, 384] or None
          When provided, skip re-encoding the conditioning frames (chain-of-forward).
        """
        self.model.train()

        # ── Encode ──────────────────────────────────────────────────────────
        features_all = torch.cat([features, features_gt], 1)   # [B,CF+1,5,H,W]
        with torch.no_grad():
            if latents_cond_precomputed is not None:
                # Latent-chain: only encode the new GT frame
                lat_gt = self.encode_sequence(features_gt)     # [B,1,256,384]
                lat_all = torch.cat([latents_cond_precomputed, lat_gt], 1)  # [B,CF+1,256,384]
            else:
                lat_all = self.encode_sequence(features_all)   # [B,CF+1,256,384]

        lat_cond   = lat_all[:, :self.condition_frames]        # [B,CF,256,384]
        lat_target = lat_all[:, self.condition_frames]         # [B,256,384]

        # ── Pose / yaw indices ───────────────────────────────────────────────
        with torch.cuda.amp.autocast(enabled=False):
            if rel_pose_cond is not None and rel_yaw_cond is not None:
                rp_gt, ry_gt = get_rel_pose(
                    rot_matrix[:, (self.condition_frames-1)*self.args.block_size:
                                  (self.condition_frames+1)*self.args.block_size]
                )
                rel_pose = torch.cat([rel_pose_cond, rp_gt[:, -1:]], 1)
                rel_yaw  = torch.cat([rel_yaw_cond,  ry_gt[:, -1:]],  1)
            else:
                rel_pose, rel_yaw = get_rel_pose(
                    rot_matrix[:, :(self.condition_frames+1)*self.args.block_size]
                )

        pose_idx = poses_to_indices(rel_pose, self.pose_x_vocab_size, self.pose_y_vocab_size,
                                    x_range=float(self.pose_x_bound),
                                    y_range=float(self.pose_y_bound * 2))
        yaw_idx  = yaws_to_indices(rel_yaw, self.yaw_vocab_size)

        # ── STT ─────────────────────────────────────────────────────────────
        lat_cond_norm = lat_cond / self.latent_scale
        logits        = self.model(lat_cond_norm, pose_idx, yaw_idx,
                                   drop_feature=self.args.drop_feature)
        stt_features  = logits['logits']   # [(B*CF),S,E]
        pose_emb_all  = logits['pose_emb'] # [(B*CF),D]

        B = lat_target.shape[0]; CF = self.condition_frames
        stt_last  = rearrange(stt_features, '(b cf) s e -> b cf s e', b=B)[:, -1]  # [B,S,E]
        pose_last = rearrange(pose_emb_all, '(b cf) d   -> b cf d',   b=B)[:, -1]  # [B,D]

        # ── FluxDiT flow-matching loss ───────────────────────────────────────
        tgt_norm  = lat_target / self.latent_scale                     # [B,256,384]
        t_sample  = torch.rand((B, 1, 1), device=tgt_norm.device)
        loss_terms = self.dit.training_losses(
            img=tgt_norm, img_ids=self.img_ids[:B],
            cond=stt_last, cond_ids=self.cond_ids[:B],
            t=t_sample, y=pose_last,
            return_predict=self.args.return_predict,
        )
        diff_loss     = loss_terms['loss'].mean()
        predict_lats  = loss_terms.get('predict')         # [B,256,384] or None

        # ── Auxiliary losses on ViT-XL decoded predictions ──────────────────
        z_l1 = z_cd = z_bev = torch.zeros(1, device=tgt_norm.device, dtype=tgt_norm.dtype)
        predict_decoded = None

        if predict_lats is not None:
            # Denormalise before decoder
            predict_decoded = self.decode_latents(predict_lats)   # [B,5,H,W]
            gt_img = features_gt[:, 0]                            # [B,5,H,W]
            if self.log_range:
                gt_unnorm = torch.exp2(gt_img[:, 0] * 6.) - 1.
            else:
                range_mean, range_std = self.proj_img_mean[0], self.proj_img_stds[0]
                gt_unnorm = gt_img[:, 0] * range_std + range_mean
            gt_valid   = gt_unnorm > 0.5

            if self.log_w_l1 is not None:
                l1_range = torch.abs(predict_decoded[:,0] - gt_img[:,0])
                # Also supervise x,y,z,intensity with reduced weight (0.25)
                l1_rest  = torch.abs(predict_decoded[:,1:] - gt_img[:,1:]).mean(1)
                l1_map   = l1_range + 0.25 * l1_rest
                z_l1 = ((l1_map * gt_valid.float() * t_sample.squeeze()).sum()
                        / (gt_valid.float().sum() + 1e-8))
                diff_loss = diff_loss + self._uw(self.log_w_l1, z_l1)

            if self.log_w_chamfer is not None and step >= self.chamfer_start:
                if self.log_range:
                    pd = (torch.exp2(predict_decoded[:,0] * 6.) - 1.).reshape(B,-1)
                else:
                    pd = (predict_decoded[:,0] * range_std + range_mean).reshape(B,-1)
                gd = gt_unnorm.reshape(B,-1)
                z_cd = batch_chamfer_distance(
                    pd, gd, pd > 0.5, gd > 0.5,
                    self.range_projector, self.chamfer_max_pts
                ) * t_sample.mean()
                diff_loss = diff_loss + self._uw(self.log_w_chamfer, z_cd)

            if self.log_w_bev is not None and self.bev_perceptual_fn is not None:
                if self.log_range:
                    pd = (torch.exp2(predict_decoded[:,0] * 6.) - 1.).reshape(B,-1)
                else:
                    pd = (predict_decoded[:,0] * range_std + range_mean).reshape(B,-1)
                gd = gt_unnorm.reshape(B,-1)
                z_bev = self.bev_perceptual_fn(pd, gd, pd>0.5, gd>0.5) * t_sample.mean()
                diff_loss = diff_loss + self._uw(self.log_w_bev, z_bev)

        return {
            'loss_all':       diff_loss,
            'loss_diff':      loss_terms['loss'].mean().detach(),
            'loss_range_l1':  z_l1,
            'loss_chamfer':   z_cd,
            'loss_bev_percep': z_bev,
            'predict':         predict_decoded,
            'predict_latents': predict_lats,        # [B,256,384] normalised, for AR chain
            'latents_cond_enc': lat_cond,           # [B,CF,256,384] for sliding window
        }

    # ── Evaluation step ─────────────────────────────────────────────────────

    @torch.no_grad()
    def step_eval(self, features, rel_pose, rel_yaw, sample_last=True):
        """Autoregressively predict next range view frame.

        features: [B, CF, 5, H, W]
        Returns:  [B, 5, H, W]
        """
        self.model.eval(); self.dit.eval()
        lat = self.encode_sequence(features)        # [B,CF,256,384]
        pose_idx = poses_to_indices(rel_pose, self.pose_x_vocab_size, self.pose_y_vocab_size,
                                    x_range=float(self.pose_x_bound),
                                    y_range=float(self.pose_y_bound * 2))
        yaw_idx = yaws_to_indices(rel_yaw, self.yaw_vocab_size)

        stt_feat, pose_emb = self.model.evaluate(lat/self.latent_scale, pose_idx, yaw_idx,
                                                  sample_last=sample_last)
        B = stt_feat.shape[0]
        _pfx = self.total_token_size - self.img_token_size
        img_ids, cond_ids, _ = prepare_ids(B, self.h, self.w,
                                            self.total_token_size, 0, prefix_size=_pfx)

        noise  = torch.randn(B, self.img_token_size, self.dino_emb_dim).to(stt_feat)
        steps  = get_schedule(int(self.args.num_sampling_steps), self.img_token_size)
        pred_z = self.dit.sample(noise, img_ids, stt_feat, cond_ids, pose_emb, steps)
        return self.decode_latents(pred_z)   # [B,5,64,2048]

    def forward(self, features, rot_matrix, features_gt=None,
                rel_pose_cond=None, rel_yaw_cond=None,
                features_aug=None, sample_last=True, step=0,
                latents_cond_precomputed=None, **kwargs):
        if self.training:
            return self.step_train(features, rot_matrix, features_gt,
                                   rel_pose_cond, rel_yaw_cond, features_aug,
                                   step, latents_cond_precomputed)
        with torch.cuda.amp.autocast(enabled=False):
            rel_pose, rel_yaw = get_rel_pose(rot_matrix)
        return self.step_eval(features, rel_pose, rel_yaw, sample_last=sample_last)

    def save_model(self, path, step, rank=0):
        if rank == 0:
            torch.save({'model_state_dict': self.state_dict(), 'step': step},
                       f'{path}/dino_dit_step{step}.pkl')
