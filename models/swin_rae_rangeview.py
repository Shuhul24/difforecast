"""
Swin-RAE Range View Models  (TULIP-inspired two-stage pipeline)

Stage 1 – Reconstruction Autoencoder (RAE):
  TULIPRangeEncoder  →  TULIPRangeDecoder (with skip connections)
  Input/output: [B, 2, 64, 2048]  (range log-normalised + intensity)
  Latent:       [B, 64, 768]      (2×32 Swin bottleneck, 4-stage TULIP design)
  Loss: Berhu on range channel + L1 on intensity + optional BEV perceptual

Stage 2 – Latent Diffusion Transformer (forecasting):
  Frozen TULIPRangeEncoder → STT (temporal) → FluxDiT (flow matching)
  → Frozen TULIPRangeDecoder (with skip connections from last condition frame)
  Latent space: [B, 64, 768] — TULIP 4-stage bottleneck.

Key design:
  - 4-stage hierarchical Swin encoder/decoder matching TULIP-base
  - Skip connections in decoder (U-Net style, as in TULIP)
  - Berhu (inverse Huber) loss promotes sharp depth edges over plain L1
  - For Stage 2 decoding, encoder skip features from the last condition
    frame are fed into the frozen decoder as a structural scene prior
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.swin_rangeview import TULIPRangeEncoder, TULIPRangeDecoder, _init_weights
from models.stt import SpatialTemporalTransformer
from models.flux_dit import FluxParams, FluxDiT
from models.modules.tokenizer import poses_to_indices, yaws_to_indices
from models.modules.sampling import prepare_ids, get_schedule
from utils.preprocess import get_rel_pose
from utils.range_losses import (
    RangeViewProjection,
    batch_chamfer_distance,
)
from utils.bev_perceptual import BEVPerceptualLoss


# ── Berhu (reverse-Huber / inverse-Huber) loss ───────────────────────────────

def berhu_loss(pred, target, threshold_frac=0.2):
    """Berhu loss: L1 for small errors, L2-like for large errors.

    Adaptive threshold c = threshold_frac * max(|pred - target|).
    Mirrors TULIP's inverse_huber_loss but used here as a training objective
    (TULIP only uses it for evaluation).
    """
    absdiff = (pred - target).abs()
    c = (threshold_frac * absdiff.detach().max()).clamp(min=1e-4)
    return torch.where(absdiff <= c, absdiff, (absdiff ** 2 + c ** 2) / (2 * c))


# ── Stage 1: RangeViewSwinRAE ─────────────────────────────────────────────────

class RangeViewSwinRAE(nn.Module):
    """Stage 1: 4-stage Swin encoder + decoder with skip connections (RAE pre-training).

    Losses:
      - MAE on range channel (ch 0): matches TULIP training objective.
      - L1 on intensity channel (ch 1): simple reconstruction.
      - Optional BEV perceptual loss on the decoded range channel.

    Both encoder and decoder are trainable in Stage 1.
    The decoder receives skip features from the encoder (U-Net style).
    """

    def __init__(self, args, local_rank=-1):
        super().__init__()
        n_ch = int(getattr(args, 'range_channels', 2))

        enc_kw = dict(
            in_chans       = n_ch,
            embed_dim      = int(getattr(args, 'swin_embed_dim',      96)),
            depths         = tuple(getattr(args, 'swin_depths',       (2, 6, 2, 2))),
            num_heads      = tuple(getattr(args, 'swin_num_heads',    (3, 6, 12, 24))),
            window_size    = tuple(getattr(args, 'swin_window_size',  (4, 8))),
            mlp_ratio      = float(getattr(args, 'swin_mlp_ratio',    4.0)),
            drop_rate      = float(getattr(args, 'swin_drop_rate',    0.0)),
            attn_drop_rate = float(getattr(args, 'swin_attn_drop',    0.0)),
            drop_path_rate = float(getattr(args, 'swin_drop_path',    0.1)),
            use_v2         = bool(getattr(args, 'swin_v2',            True)),
        )
        self.encoder = TULIPRangeEncoder(**enc_kw)
        self.decoder = TULIPRangeDecoder(
            out_chans=n_ch, **{k: v for k, v in enc_kw.items() if k != 'in_chans'}
        )
        self.encoder.apply(_init_weights)
        self.decoder.apply(_init_weights)

        self.log_range = bool(getattr(args, 'log_range', True))
        self.proj_img_mean = list(getattr(args, 'proj_img_mean', [0.0, 0.0]))
        self.proj_img_stds = list(getattr(args, 'proj_img_stds', [1.0, 1.0]))

        # Per-channel loss weights: [range, intensity]
        ch_w = list(getattr(args, 'rae_ch_weights', [1., 1.]))
        self.register_buffer('ch_weights', torch.tensor(ch_w, dtype=torch.float32))

        # Optional BEV perceptual loss
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
            self.log_w_bev = nn.Parameter(torch.tensor(math.log(1.0 / self.bev_percep_weight)))
        else:
            self.bev_fn = None
            self.log_w_bev = None

    def encode(self, x):
        """[B, C, H, W] → (z [B, 64, 768], skips: list of 3)"""
        return self.encoder(x)

    def decode(self, z, skips=None):
        """(z [B, 64, 768], skips) → [B, C, H, W]"""
        return self.decoder(z, skips)

    def forward(self, x):
        """x: [B, C, 64, 2048] → loss dict."""
        z, skips = self.encode(x)
        rec = self.decode(z, skips)

        # MAE on range (ch 0), L1 on intensity (ch 1+)
        loss_range = (rec[:, 0] - x[:, 0]).abs()
        if x.shape[1] > 1:
            loss_int = (rec[:, 1:] - x[:, 1:]).abs().mean(dim=1)
        else:
            loss_int = torch.zeros_like(loss_range)

        w = self.ch_weights
        loss_rec = w[0] * loss_range.mean() + (w[1] * loss_int.mean() if x.shape[1] > 1 else 0.)
        loss_all = loss_rec
        loss_bev = torch.zeros(1, device=x.device)

        if self.bev_fn is not None:
            if self.log_range:
                d_pred = torch.exp2(rec[:, 0].float() * 6.) - 1.
                d_gt   = torch.exp2(x[:, 0].float()  * 6.) - 1.
            else:
                m0, s0 = self.proj_img_mean[0], self.proj_img_stds[0]
                d_pred = rec[:, 0].float() * s0 + m0
                d_gt   = x[:, 0].float()  * s0 + m0
            B = d_pred.shape[0]
            loss_bev = self.bev_fn(
                d_pred.reshape(B, -1), d_gt.reshape(B, -1),
                (d_pred > 0.5).reshape(B, -1), (d_gt > 0.5).reshape(B, -1),
            )
            lw = self.log_w_bev.clamp(min=0.)
            loss_all = loss_all + torch.exp(-lw) * loss_bev + lw

        return {'loss_all': loss_all, 'loss_rec': loss_rec,
                'loss_bev': loss_bev, 'x_rec': rec}

    def load_from_tulip(self, ckpt_path):
        """Load TULIP-KITTI pre-trained weights into the encoder and decoder.

        TULIP uses SwinV1 with depths=(2,2,2,2) and window=(2,8).  Our model
        must match those settings for shapes to align (see swin_config_rangeview).

        Incompatible keys are skipped with a note:
          - patch_embed.proj  (TULIP: 1-ch; ours: 2-ch, different kernel)
          - decoder_pred / ps_head  (TULIP output heads, no equivalent here)
          - PatchExpand.expand.bias  (TULIP Conv2d has bias; our Linear does not)
          - skip_connection_layers.*.bias  (same reason)
          - encoder.norm  (final encoder LN; not present in TULIP)
          - decoder.expands.*.norm  (extra LN in our PatchExpand; not in TULIP)
          - decoder.final_expand  (our output head; not in TULIP)

        Decoder stage 0 is initialised from TULIP's encoder stage 3 (bottleneck).
        This is sound: both operate on [B, 64, 768] tokens.

        Returns: {'loaded': [...], 'skipped': [...]}
        """
        raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd  = raw.get('model', raw)

        own_sd  = {k: v.clone() for k, v in self.state_dict().items()}
        new_sd  = {}
        loaded, skipped = [], []

        def _try(src_key, dst_key):
            if src_key not in sd:
                return
            src = sd[src_key]
            if dst_key not in own_sd:
                skipped.append(f'SKIP {src_key} → {dst_key}  (dst missing)')
                return
            # Conv2d [Co, Ci, 1, 1] → Linear [Co, Ci]
            if src.dim() == 4 and src.shape[2:] == (1, 1) and own_sd[dst_key].dim() == 2:
                src = src.squeeze(-1).squeeze(-1)
            if src.shape != own_sd[dst_key].shape:
                skipped.append(f'SKIP {src_key} → {dst_key}  '
                               f'({list(src.shape)} vs {list(own_sd[dst_key].shape)})')
                return
            new_sd[dst_key] = src
            loaded.append(dst_key)

        # ── Encoder stages 0-3 ────────────────────────────────────────────────
        for s in range(4):
            for b in range(2):
                ps = f'layers.{s}.blocks.{b}.'
                pd = f'encoder.stages.{s}.{b}.'
                for k in own_sd:
                    if k.startswith(pd):
                        _try(ps + k[len(pd):], k)

        # ── Encoder PatchMerging (stages 0-2) ─────────────────────────────────
        for s in range(3):
            ps = f'layers.{s}.downsample.'
            pd = f'encoder.merges.{s}.'
            for k in own_sd:
                if k.startswith(pd):
                    _try(ps + k[len(pd):], k)

        # ── Encoder patch-embed norm ───────────────────────────────────────────
        _try('patch_embed.norm.weight', 'encoder.patch_embed.norm.weight')
        _try('patch_embed.norm.bias',   'encoder.patch_embed.norm.bias')

        # ── Decoder stage 0  ← TULIP encoder stage 3 (bottleneck) ────────────
        for b in range(2):
            ps = f'layers.3.blocks.{b}.'
            pd = f'decoder.stages.0.{b}.'
            for k in own_sd:
                if k.startswith(pd):
                    _try(ps + k[len(pd):], k)

        # ── Decoder stages 1-3  ← TULIP layers_up.0-2 ────────────────────────
        for i in range(3):
            for b in range(2):
                ps = f'layers_up.{i}.blocks.{b}.'
                pd = f'decoder.stages.{i+1}.{b}.'
                for k in own_sd:
                    if k.startswith(pd):
                        _try(ps + k[len(pd):], k)

        # ── Decoder PatchExpanding ────────────────────────────────────────────
        _try('first_patch_expanding.expand.weight', 'decoder.expands.0.expand.weight')
        for i in range(2):
            _try(f'layers_up.{i}.upsample.expand.weight', f'decoder.expands.{i+1}.expand.weight')

        # ── Skip-connection projections ───────────────────────────────────────
        for i in range(3):
            _try(f'skip_connection_layers.{i}.weight', f'decoder.skip_projs.{i}.weight')
            _try(f'skip_connection_layers.{i}.bias',   f'decoder.skip_projs.{i}.bias')

        # ── Final decoder norm ────────────────────────────────────────────────
        _try('norm_up.weight', 'decoder.norm.weight')
        _try('norm_up.bias',   'decoder.norm.bias')

        own_sd.update(new_sd)
        missing, unexpected = self.load_state_dict(own_sd, strict=False)

        print(f'[load_from_tulip]  loaded {len(loaded)} tensors  '
              f'skipped {len(skipped)}  missing {len(missing)}  '
              f'unexpected {len(unexpected)}')
        if skipped:
            for msg in skipped:
                print(f'  {msg}')
        return {'loaded': loaded, 'skipped': skipped,
                'missing': missing, 'unexpected': unexpected}

    def save_model(self, path, step, rank=0):
        if rank == 0:
            torch.save({'model_state_dict': self.state_dict(), 'step': step},
                       f'{path}/swin_rae_step{step}.pkl')


# ── Stage 2: RangeViewSwinDiT ─────────────────────────────────────────────────

class RangeViewSwinDiT(nn.Module):
    """Stage 2: FluxDiT forecasting in the TULIP 4-stage Swin bottleneck space.

    Frozen TULIPRangeEncoder extracts [B, T, 64, 768] latents from past frames.
    STT provides temporal context; FluxDiT (rectified flow) predicts the future
    latent. Frozen TULIPRangeDecoder (with skip connections from the last
    condition frame's encoder) reconstructs the predicted range view.

    Latent shape: [B, 64, 768] — grid (2×32) at embed_dim×8=768.
    """

    def __init__(self, args, local_rank=-1, load_path=None):
        super().__init__()
        self.args = args

        # Token geometry — TULIP 4-stage bottleneck
        self.h, self.w      = 2, 32              # bottleneck grid
        self.img_token_size = 64                 # 2*32
        self.latent_dim     = 768                # embed_dim*8
        self.condition_frames = int(getattr(args, 'condition_frames', 5))

        self.pose_token_size  = 2 * int(getattr(args, 'block_size', 1))
        self.yaw_token_size   = 1 * int(getattr(args, 'block_size', 1))
        self.total_token_size = self.img_token_size + self.pose_token_size + self.yaw_token_size

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
        n_ch = int(getattr(args, 'range_channels', 2))

        # ── Encoder (frozen after loading Stage-1 checkpoint) ─────────────────
        enc_kw = dict(
            in_chans       = n_ch,
            embed_dim      = int(getattr(args, 'swin_embed_dim',      96)),
            depths         = tuple(getattr(args, 'swin_depths',       (2, 6, 2, 2))),
            num_heads      = tuple(getattr(args, 'swin_num_heads',    (3, 6, 12, 24))),
            window_size    = tuple(getattr(args, 'swin_window_size',  (4, 8))),
            mlp_ratio      = float(getattr(args, 'swin_mlp_ratio',    4.0)),
            drop_rate      = 0.,
            attn_drop_rate = 0.,
            drop_path_rate = 0.,
        )
        self.swin_encoder = TULIPRangeEncoder(**enc_kw)
        for p in self.swin_encoder.parameters():
            p.requires_grad_(False)

        # ── Decoder (loaded from Stage-1 checkpoint and frozen) ───────────────
        self.decoder = TULIPRangeDecoder(
            out_chans=n_ch, **{k: v for k, v in enc_kw.items() if k != 'in_chans'}
        )
        swin_ckpt = getattr(args, 'swin_ckpt', None)
        if swin_ckpt is not None:
            sd = torch.load(swin_ckpt, map_location='cpu').get('model_state_dict', {})
            enc_sd = {k[len('encoder.'):]: v for k, v in sd.items() if k.startswith('encoder.')}
            dec_sd = {k[len('decoder.'):]: v for k, v in sd.items() if k.startswith('decoder.')}
            missing_enc = self.swin_encoder.load_state_dict(enc_sd, strict=True)
            missing_dec = self.decoder.load_state_dict(dec_sd, strict=True)
            print(f"[SwinDiT] Encoder+Decoder loaded from {swin_ckpt}")
        else:
            print("[SwinDiT] No swin_ckpt — encoder+decoder randomly initialised. Run Stage 1 first.")
        for p in self.decoder.parameters():
            p.requires_grad_(False)

        # ── STT (vae_emb_dim=768 for TULIP bottleneck) ───────────────────────
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
            vae_emb_dim=self.latent_dim,          # 768
            temporal_block=int(getattr(args, 'block_size', 1)),
        )
        self.model.cuda()

        # ── FluxDiT (in/out_channels=768 for TULIP bottleneck) ───────────────
        _pfx = self.total_token_size - self.img_token_size
        self.dit = FluxDiT(FluxParams(
            in_channels=self.latent_dim,           # 768
            out_channels=self.latent_dim,          # 768
            vec_in_dim=args.n_embd * _pfx,
            vec_hidden_dim=args.n_embd,
            context_in_dim=args.n_embd,
            hidden_size=args.n_embd_dit,
            mlp_ratio=float(getattr(args, 'mlp_ratio_dit', 4.0)),
            num_heads=args.n_head_dit,
            depth=args.n_layer[1],
            depth_single_blocks=args.n_layer[2],
            axes_dim=args.axes_dim_dit,
            theta=10_000, qkv_bias=True, guidance_embed=True,
            drop_path_rate=float(getattr(args, 'drop_path_rate', 0.0)),
        ))
        self.dit.cuda()

        self.register_buffer('latent_scale',
                             torch.tensor(float(getattr(args, 'latent_scale', 1.0))))

        # Auxiliary loss config
        self.log_range              = bool(getattr(args,  'log_range',              True))
        self.proj_img_mean          = list(getattr(args,  'proj_img_mean',          [0.0, 0.0]))
        self.proj_img_stds          = list(getattr(args,  'proj_img_stds',          [1.0, 1.0]))
        self.chamfer_loss_weight    = float(getattr(args, 'chamfer_loss_weight',    0.0))
        self.chamfer_max_pts        = int(getattr(args,   'chamfer_max_pts',        2048))
        self.chamfer_start          = int(getattr(args,   'chamfer_start',          0))
        self.bev_perceptual_weight  = float(getattr(args, 'bev_perceptual_weight',  0.0))

        def _log_w(w):
            return nn.Parameter(torch.tensor(math.log(1.0 / w))) if w > 0 else None

        self.log_w_chamfer = _log_w(self.chamfer_loss_weight)
        self.log_w_bev     = _log_w(self.bev_perceptual_weight)

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

        _pfx2 = self.total_token_size - self.img_token_size
        bs = args.batch_size * self.condition_frames
        self.img_ids, self.cond_ids, _ = prepare_ids(
            bs, self.h, self.w, self.total_token_size, 0, prefix_size=_pfx2
        )

        if load_path is not None:
            self._load_ckpt(load_path)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_ckpt(self, path):
        sd = torch.load(path, map_location='cpu').get('model_state_dict', {})
        for attr, prefixes in [('model', ['module.model.', 'model.']),
                                ('dit',   ['module.dit.',   'dit.'])]:
            obj = getattr(self, attr)
            for pfx in prefixes:
                obj_sd = {k: sd[pfx + k] for k in obj.state_dict() if pfx + k in sd}
                if obj_sd:
                    obj.load_state_dict(obj_sd, strict=False)
                    break
        print(f"[SwinDiT] STT+DiT loaded from {path}")

    @staticmethod
    def _uw(log_w, loss):
        lw = log_w.clamp(min=0.)
        return torch.exp(-lw) * loss + lw

    @torch.no_grad()
    def encode_sequence(self, x):
        """[B, T, C, H, W] → (latents [B, T, 64, 768], skips_per_frame)

        skips_per_frame: list of 3 tensors, each [B, T, N, D]
          skips_per_frame[0]: [B, T, 4096, 96]
          skips_per_frame[1]: [B, T, 1024, 192]
          skips_per_frame[2]: [B, T, 256,  384]
        """
        B, T, C, H, W = x.shape
        z, sk = self.swin_encoder(rearrange(x, 'b t c h w -> (b t) c h w'))
        latents = rearrange(z, '(b t) n d -> b t n d', b=B, t=T)
        all_skips = []
        for s in sk:
            BT, N, D = s.shape
            all_skips.append(s.view(B, T, N, D))
        return latents, all_skips

    @torch.no_grad()
    def get_frame_skips(self, x_frame):
        """Encode a single frame and return only its skip features.

        x_frame: [B, C, H, W]
        Returns: list of 3 tensors — skips[i] shape [B, N_i, D_i]
        """
        _, skips = self.swin_encoder(x_frame)
        return skips

    def decode_latents(self, z, skips=None):
        """z: [B, 64, 768], skips: list of 3 or None → [B, C, 64, 2048]"""
        return self.decoder(z * self.latent_scale, skips)

    # ── Training step ─────────────────────────────────────────────────────────

    def step_train(
        self, features, rot_matrix, features_gt,
        rel_pose_cond=None, rel_yaw_cond=None,
        features_aug=None, step=0,
        latents_cond_precomputed=None,
    ):
        self.model.train()
        features_all = torch.cat([features, features_gt], 1)  # [B, CF+1, C, H, W]

        with torch.no_grad():
            if latents_cond_precomputed is not None:
                lat_gt, _ = self.encode_sequence(features_gt)
                lat_all = torch.cat([latents_cond_precomputed, lat_gt], 1)
            else:
                lat_all, _ = self.encode_sequence(features_all)

            # Skip features always come from the last condition frame
            # (structural scene prior for the decoder of the predicted frame)
            last_cond_skips = self.get_frame_skips(features[:, -1])

        lat_cond   = lat_all[:, :self.condition_frames]
        lat_target = lat_all[:,  self.condition_frames]

        with torch.cuda.amp.autocast(enabled=False):
            if rel_pose_cond is not None and rel_yaw_cond is not None:
                rp_gt, ry_gt = get_rel_pose(
                    rot_matrix[:, (self.condition_frames - 1) * self.args.block_size:
                                  (self.condition_frames + 1) * self.args.block_size]
                )
                rel_pose = torch.cat([rel_pose_cond, rp_gt[:, -1:]], 1)
                rel_yaw  = torch.cat([rel_yaw_cond,  ry_gt[:, -1:]],  1)
            else:
                rel_pose, rel_yaw = get_rel_pose(
                    rot_matrix[:, :(self.condition_frames + 1) * self.args.block_size]
                )

        pose_idx = poses_to_indices(rel_pose, self.pose_x_vocab_size, self.pose_y_vocab_size,
                                    x_range=float(self.pose_x_bound),
                                    y_range=float(self.pose_y_bound * 2))
        yaw_idx  = yaws_to_indices(rel_yaw, self.yaw_vocab_size)

        lat_cond_norm = lat_cond / self.latent_scale
        lat_target_norm = (lat_target / self.latent_scale).unsqueeze(1)          # [B,1,T,C]
        lat_all_norm  = torch.cat([lat_cond_norm, lat_target_norm], dim=1)       # [B,CF+1,T,C]
        logits        = self.model(lat_all_norm, pose_idx, yaw_idx,
                                   drop_feature=self.args.drop_feature)
        stt_features  = logits['logits']
        pose_emb_all  = logits['pose_emb']

        B = lat_target.shape[0]
        CF = self.condition_frames
        stt_last  = rearrange(stt_features, '(b cf) s e -> b cf s e', b=B)[:, -1]
        pose_last = rearrange(pose_emb_all, '(b cf) d   -> b cf d',   b=B)[:, -1]

        tgt_norm  = lat_target / self.latent_scale
        # Logit-normal timestep sampling: concentrates mass around t≈0.5
        # (harder, intermediate noise levels) rather than uniform [0,1]
        u = torch.randn((B, 1, 1), device=tgt_norm.device)
        t_sample  = torch.sigmoid(u)
        guidance = torch.full((B,), float(getattr(self.args, 'cfg_scale', 3.5)),
                              device=tgt_norm.device, dtype=tgt_norm.dtype)
        loss_terms = self.dit.training_losses(
            img=tgt_norm, img_ids=self.img_ids[:B],
            cond=stt_last, cond_ids=self.cond_ids[:B],
            t=t_sample, y=pose_last, guidance=guidance,
            return_predict=self.args.return_predict,
        )
        diff_loss    = loss_terms['loss'].mean()
        predict_lats = loss_terms.get('predict')

        z_cd = z_bev = torch.zeros(1, device=tgt_norm.device, dtype=tgt_norm.dtype)
        predict_decoded = None

        if predict_lats is not None:
            predict_decoded = self.decode_latents(predict_lats, last_cond_skips)
            gt_img = features_gt[:, 0]

            if self.log_range:
                gt_unnorm = torch.exp2(gt_img[:, 0].float() * 6.) - 1.
            else:
                m0, s0 = self.proj_img_mean[0], self.proj_img_stds[0]
                gt_unnorm = gt_img[:, 0].float() * s0 + m0

            if self.log_w_chamfer is not None and step >= self.chamfer_start:
                if self.log_range:
                    pd = (torch.exp2(predict_decoded[:, 0].float() * 6.) - 1.).reshape(B, -1)
                else:
                    pd = (predict_decoded[:, 0].float() * s0 + m0).reshape(B, -1)
                gd = gt_unnorm.reshape(B, -1)
                z_cd = batch_chamfer_distance(
                    pd, gd, pd > 0.5, gd > 0.5,
                    self.range_projector, self.chamfer_max_pts
                ) * t_sample.mean()
                diff_loss = diff_loss + self._uw(self.log_w_chamfer, z_cd)

            if self.log_w_bev is not None and self.bev_perceptual_fn is not None:
                if self.log_range:
                    pd = (torch.exp2(predict_decoded[:, 0].float() * 6.) - 1.).reshape(B, -1)
                else:
                    pd = (predict_decoded[:, 0].float() * s0 + m0).reshape(B, -1)
                gd = gt_unnorm.reshape(B, -1)
                z_bev = self.bev_perceptual_fn(pd, gd, pd > 0.5, gd > 0.5) * t_sample.mean()
                diff_loss = diff_loss + self._uw(self.log_w_bev, z_bev)

        return {
            'loss_all':        diff_loss,
            'loss_diff':       loss_terms['loss'].mean().detach(),
            'loss_chamfer':    z_cd,
            'loss_bev_percep': z_bev,
            'predict':         predict_decoded,
            'predict_latents': predict_lats,
            'latents_cond_enc': lat_cond,
        }

    # ── Evaluation step ───────────────────────────────────────────────────────

    @torch.no_grad()
    def step_eval(self, features, rel_pose, rel_yaw, sample_last=True):
        """Autoregressively predict next range view frame.

        features: [B, CF, C, H, W]
        Returns:  [B, C, H, W]
        """
        self.model.eval(); self.dit.eval()
        lat, all_skips = self.encode_sequence(features)
        # Use last condition frame's skip features for the decoder
        last_cond_skips = [s[:, -1] for s in all_skips]

        pose_idx = poses_to_indices(rel_pose, self.pose_x_vocab_size, self.pose_y_vocab_size,
                                    x_range=float(self.pose_x_bound),
                                    y_range=float(self.pose_y_bound * 2))
        yaw_idx = yaws_to_indices(rel_yaw, self.yaw_vocab_size)

        stt_feat, pose_emb = self.model.evaluate(lat / self.latent_scale, pose_idx, yaw_idx,
                                                  sample_last=sample_last)
        B = stt_feat.shape[0]
        _pfx = self.total_token_size - self.img_token_size
        img_ids, cond_ids, _ = prepare_ids(B, self.h, self.w,
                                            self.total_token_size, 0, prefix_size=_pfx)
        noise = torch.randn(B, self.img_token_size, self.latent_dim).to(stt_feat)
        steps = get_schedule(int(self.args.num_sampling_steps), self.img_token_size)
        pred_z = self.dit.sample(noise, img_ids, stt_feat, cond_ids, pose_emb, steps)
        return self.decode_latents(pred_z, last_cond_skips)

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
                       f'{path}/swin_dit_step{step}.pkl')
