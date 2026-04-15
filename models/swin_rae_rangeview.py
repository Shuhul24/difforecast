"""
Swin-RAE Range View Models  (TULIP-inspired two-stage pipeline)

Stage 1 – Reconstruction Autoencoder (RAE):
  TULIPRangeEncoder  →  TULIPRangeDecoder (with skip connections)
  Input/output: [B, 1, 64, 2048]  (range log-normalised, 1-channel depth)
  Latent:       [B, 64, 768]      (2×32 Swin bottleneck, 4-stage TULIP design)
  Loss: distance-weighted Berhu on range channel, masked to valid (non-zero) pixels

Stage 2 – Latent Diffusion Transformer (forecasting):
  Frozen TULIPRangeEncoder → STT (temporal) → FluxDiT (flow matching)
  → Frozen TULIPRangeDecoder (with temporally aggregated skip connections)
  Latent space: [B, 64, 768] — TULIP 4-stage bottleneck.

Key design:
  - 4-stage hierarchical Swin encoder/decoder matching TULIP-base
  - Skip connections in decoder (U-Net style, as in TULIP)
  - Berhu (inverse Huber) loss promotes sharp depth edges over plain L1
  - Distance-weighted loss up-weights far pixels to improve high-fidelity detail
  - Temporal skip aggregation: TemporalSkipAggregator cross-attends over all CF
    conditioning frames' skip features instead of using only the last frame,
    preventing copy-frame artefacts at t+2..t+5
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.swin_rangeview import TULIPRangeEncoder, TULIPRangeDecoder, _init_weights
from models.stt import SpatialTemporalTransformer
from models.flux_dit import FluxParams, FluxDiT
from models.traj_dit import TrajDiT, TrajParams
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


# ── Temporal Skip Aggregator ─────────────────────────────────────────────────

class TemporalSkipAggregator(nn.Module):
    """Aggregate encoder skip features across CF conditioning frames via cross-attention.

    For each of the 3 TULIP decoder skip levels, a learnable pool token
    cross-attends over all CF conditioning frames' skip features to produce a
    single aggregated skip tensor [B, N_i, D_i].

    This replaces the static last-frame-only skip injection, which anchors the
    decoder to the last conditioning frame's structure and causes copy-frame
    artefacts at t+2..t+5.  By attending over the full temporal context, the
    decoder can select persistent structure from earlier frames and adapt to
    motion across the conditioning window.

    Only the pool-token parameters and attention projection weights are trainable;
    the encoder producing the skip features remains frozen.

    Skip dims for TULIP 4-stage encoder on 64×2048:
      level 0: [B, CF, 4096,  96]
      level 1: [B, CF, 1024, 192]
      level 2: [B, CF,  256, 384]
    """

    def __init__(self, skip_dims):
        super().__init__()
        self.cross_attn = nn.ModuleList()
        self.norm       = nn.ModuleList()
        self.pool_token = nn.ParameterList()
        for d in skip_dims:
            n_h = d // 32          # head_dim = 32 for all levels
            self.cross_attn.append(
                nn.MultiheadAttention(d, n_h, batch_first=True, dropout=0.0)
            )
            self.norm.append(nn.LayerNorm(d))
            self.pool_token.append(nn.Parameter(torch.zeros(1, 1, d)))

    def forward(self, all_skips):
        """
        all_skips : list of 3 tensors, each [B, CF, N_i, D_i]
        Returns   : list of 3 tensors, each [B, N_i, D_i]
        """
        result = []
        for skip, attn, norm, pt in zip(
            all_skips, self.cross_attn, self.norm, self.pool_token
        ):
            B, CF, N, D = skip.shape
            # [B*N, CF, D]: treat each spatial position as a sequence over time
            kv = skip.permute(0, 2, 1, 3).reshape(B * N, CF, D)
            # Pool token is the query; it aggregates information from all CF frames
            q = pt.expand(B * N, -1, -1)          # [B*N, 1, D]
            out, _ = attn(q, kv, kv)              # [B*N, 1, D]
            pooled = norm(out.squeeze(1))          # [B*N, D]
            result.append(pooled.reshape(B, N, D))
        return result


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

        # Per-channel loss weights: [range, mask/intensity]
        ch_w = list(getattr(args, 'rae_ch_weights', [1., 1.]))
        self.register_buffer('ch_weights', torch.tensor(ch_w, dtype=torch.float32))

        # When True, ch 1 is a binary valid-pixel mask (not intensity).
        # Range loss is then restricted to valid pixels only.
        self.mask_channel = bool(getattr(args, 'mask_channel', False))

        # Distance-weighted loss: upweight far pixels by (1 + log-depth).
        # Only active for the 1-channel path; ignored for multi-channel legacy.
        self.dist_weighted_loss = bool(getattr(args, 'dist_weighted_loss', False))

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

        # KL regularisation (VAE bottleneck)
        self.kl_weight       = float(getattr(args, 'kl_weight',       1e-4))
        self.kl_warmup_steps = int(getattr(args,   'kl_warmup_steps', 10000))

    def encode(self, x, sample=True):
        """[B, C, H, W] → (z, mu, logvar [B, 64, 768], skips)"""
        return self.encoder(x, sample=sample)

    def decode(self, z, skips=None):
        """(z [B, 64, 768], skips) → [B, C, H, W]"""
        return self.decoder(z, skips)

    def forward(self, x, step=0):
        """x: [B, C, 64, 2048] → loss dict."""
        z, mu, logvar, skips = self.encode(x, sample=True)
        rec = self.decode(z, skips)

        w = self.ch_weights
        loss_mask = torch.zeros(1, device=x.device)
        if self.mask_channel and x.shape[1] > 1:
            # ch 1 is the binary valid-pixel mask (1=hit, 0=empty).
            # Range loss: masked MAE — only penalise valid (hit) pixels so the
            # model is not confused by empty-pixel depth values under log2 norm.
            valid = x[:, 1]                                          # [B, H, W]
            n_valid = valid.sum().clamp(min=1.)
            loss_range = ((rec[:, 0] - x[:, 0]).abs() * valid).sum() / n_valid
            # Mask loss: BCE with logits — decoder output ch 1 is treated as a
            # logit for binary pixel-validity classification.  BCE is correct here
            # because L1 does not push predictions toward 0/1 boundaries, while
            # BCEWithLogitsLoss trains the mask head as a proper binary classifier
            # with calibrated confidence.  Class imbalance (sparse LiDAR) is
            # handled implicitly: the gradient magnitude is proportional to the
            # prediction error, so confident correct predictions are cheap.
            loss_mask = F.binary_cross_entropy_with_logits(
                rec[:, 1].float(), x[:, 1].float()
            )
            loss_rec = w[0] * loss_range + w[1] * loss_mask
        elif x.shape[1] == 1:
            # 1-channel range-only: Berhu + distance weighting + validity masking.
            # Validity derived from depth: log2-normed empty pixels are ~0.
            valid_mask = (x[:, 0].abs() > 1e-4).float()          # [B, H, W]
            n_valid    = valid_mask.sum().clamp(min=1.)
            loss_px    = berhu_loss(rec[:, 0].float(), x[:, 0].float())  # [B, H, W]
            if self.dist_weighted_loss:
                # log-depth in [0,1]: near→weight≈1, far→weight≈2 (linear ramp)
                dist_w  = 1.0 + x[:, 0].detach().float().clamp(0., 1.)
                loss_px = loss_px * dist_w
            loss_rec = w[0] * (loss_px * valid_mask).sum() / n_valid
        else:
            # Multi-channel legacy path: MAE on range, L1 on intensity
            loss_range = (rec[:, 0] - x[:, 0]).abs()
            loss_int   = (rec[:, 1:] - x[:, 1:]).abs().mean(dim=1) if x.shape[1] > 1 \
                         else torch.zeros_like(loss_range)
            loss_rec   = w[0] * loss_range.mean() + (w[1] * loss_int.mean() if x.shape[1] > 1 else 0.)
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

        # ── KL regularisation (VAE bottleneck) ───────────────────────────────
        # Free bits (λ=0.5 nats/dim) prevents posterior collapse by only
        # penalising dimensions whose KL exceeds the free-bit floor.
        # Linear β warmup from 0 → kl_weight over kl_warmup_steps so that
        # reconstruction quality stabilises before KL pressure kicks in.
        free_bits  = 0.5
        kl_per_dim = -0.5 * (1. + logvar - mu.pow(2) - logvar.exp())   # [B, 64, 768]
        loss_kl    = kl_per_dim.clamp(min=free_bits).mean()
        beta       = self.kl_weight * min(1.0, float(step) / max(self.kl_warmup_steps, 1))
        loss_all   = loss_all + beta * loss_kl

        return {'loss_all': loss_all, 'loss_rec': loss_rec, 'loss_kl': loss_kl,
                'loss_mask': loss_mask, 'loss_bev': loss_bev, 'x_rec': rec}

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
            # strict=False: tolerate mu_proj/logvar_proj missing in old checkpoints
            # (pre-KL) or extra keys if the checkpoint is newer.
            miss_e, unex_e = self.swin_encoder.load_state_dict(enc_sd, strict=False)
            miss_d, unex_d = self.decoder.load_state_dict(dec_sd, strict=False)
            print(f"[SwinDiT] Encoder loaded from {swin_ckpt} "
                  f"(missing={len(miss_e)}, unexpected={len(unex_e)})")
            print(f"[SwinDiT] Decoder loaded from {swin_ckpt} "
                  f"(missing={len(miss_d)}, unexpected={len(unex_d)})")
        else:
            print("[SwinDiT] No swin_ckpt — encoder+decoder randomly initialised. Run Stage 1 first.")
        for p in self.decoder.parameters():
            p.requires_grad_(False)

        # ── Temporal skip aggregator (trainable) ─────────────────────────────
        # Cross-attends over all CF conditioning frames' skip features to produce
        # a single temporally-aware skip per decoder level.  Only this module's
        # parameters are trained; the frozen encoder produces the raw skips.
        if getattr(args, 'temporal_skip_agg', False):
            _ed = int(getattr(args, 'swin_embed_dim', 96))
            self.skip_aggregator = TemporalSkipAggregator(
                skip_dims=[_ed, _ed * 2, _ed * 4],  # [96, 192, 384]
            )
        else:
            self.skip_aggregator = None

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

        # ── PoseDiT (single-step relative-pose prediction) ───────────────────
        # Predicts the next (x, y, yaw) relative to the last condition frame.
        # Used during AR training to replace GT poses with model predictions.
        self.traj_token_size = 3  # x, y, yaw
        self.lambda_pose     = float(getattr(args, 'lambda_yaw_pose', 0.1))

        self.pose_dit = TrajDiT(TrajParams(
            in_channels=self.traj_token_size,
            out_channels=self.traj_token_size,
            context_in_dim=args.n_embd,
            hidden_size=int(getattr(args, 'n_embd_dit_traj', 512)),
            mlp_ratio=4.0,
            num_heads=int(getattr(args, 'n_head_dit_traj', 8)),
            depth=args.n_layer_traj[0],
            depth_single_blocks=args.n_layer_traj[1],
            axes_dim=list(getattr(args, 'axes_dim_dit_traj', [16, 16, 32])),
            theta=10_000, qkv_bias=True, guidance_embed=False,
        ))
        self.pose_dit.cuda()

        # Auxiliary loss config
        self.log_range              = bool(getattr(args,  'log_range',              True))
        self.proj_img_mean          = list(getattr(args,  'proj_img_mean',          [0.0, 0.0]))
        self.proj_img_stds          = list(getattr(args,  'proj_img_stds',          [1.0, 1.0]))
        self.range_view_loss_weight = float(getattr(args, 'range_view_loss_weight', 0.0))
        self.chamfer_loss_weight    = float(getattr(args, 'chamfer_loss_weight',    0.0))
        self.chamfer_max_pts        = int(getattr(args,   'chamfer_max_pts',        2048))
        self.chamfer_start          = int(getattr(args,   'chamfer_start',          0))
        self.bev_perceptual_weight  = float(getattr(args, 'bev_perceptual_weight',  0.0))

        def _log_w(w):
            return nn.Parameter(torch.tensor(math.log(1.0 / w))) if w > 0 else None

        # BEV perceptual keeps uncertainty weighting; Chamfer uses explicit λ.
        self.log_w_bev = _log_w(self.bev_perceptual_weight)

        # ── REPA: align FluxDiT intermediate features with frozen Swin encoder ──
        # Teacher = lat_target (frozen encoder output for the clean GT frame).
        # Already computed inside step_train — zero extra forward-pass cost.
        # Student = img-stream hidden state after double_blocks[repa_layer].
        # Both are [B, 64, 768] — no spatial pooling or dim projection needed.
        self.repa_weight = float(getattr(args, 'repa_weight',     0.0))
        self.repa_layer  = int(getattr(args,   'repa_layer_idx',  args.n_layer[1] // 2))
        self.repa_start  = int(getattr(args,   'repa_start_step', 0))
        if self.repa_weight > 0:
            # hidden_size == latent_dim == 768 → square projection.
            # Eye init: identity at step 0, learns alignment gradually.
            self.repa_proj = nn.Linear(args.n_embd_dit, self.latent_dim, bias=False)
            nn.init.eye_(self.repa_proj.weight)
        else:
            self.repa_proj = None

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
        # traj_ids for PoseDiT: traj_len=1, so shape [bs, 1, 3] = zeros
        _, _, _pose_traj_ids = prepare_ids(
            bs, self.h, self.w, self.total_token_size, 1, prefix_size=_pfx2
        )
        self.register_buffer('pose_traj_ids', _pose_traj_ids)

        if load_path is not None:
            self._load_ckpt(load_path)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _normalize_pose_continuous(self, rel_pose, rel_yaw):
        """Normalise raw (x, y, yaw) to [0, 1] without quantisation.

        rel_pose: [B, T, 2]  x in [0, pose_x_bound] m, y in [±pose_y_bound] m
        rel_yaw:  [B, T, 1]  yaw in [±yaw_bound] degrees

        Returns pose_norm [B, T, 2], yaw_norm [B, T, 1] both in [0, 1].
        """
        pose_norm = torch.stack([
            (rel_pose[..., 0].float() / self.pose_x_bound).clamp(0.0, 1.0),
            ((rel_pose[..., 1].float() + self.pose_y_bound) / (2.0 * self.pose_y_bound)).clamp(0.0, 1.0),
        ], dim=-1)
        yaw_norm = ((rel_yaw.float() + self.yaw_bound) / (2.0 * self.yaw_bound)).clamp(0.0, 1.0)
        return pose_norm, yaw_norm

    def _apply_azimuth_warp(self, lat_cond_norm, rel_yaw):
        """Warp conditioning latents by azimuth-shifting each frame to align
        to the last conditioning frame's viewpoint.

        Yaw rotation in the real world corresponds to a circular shift along
        the azimuth (W) axis of the 2×32 bottleneck latent grid. Aligning all
        conditioning frames to the same viewpoint before STT attention removes
        the need for the STT to implicitly learn this geometric relationship.

        lat_cond_norm: [B, CF, L, C]  normalised bottleneck latents (L = h*w = 64)
        rel_yaw:       [B, ≥CF-1, 1]  relative yaw in degrees between consecutive frames

        Returns: [B, CF, L, C] azimuth-aligned latents (same shape, in-graph).
        """
        B, CF, L, C = lat_cond_norm.shape
        h, w = self.h, self.w  # 2, 32

        # Cumulative yaw from each frame to frame CF-1 (the reference / last cond frame).
        # rel_yaw[:, t, 0] = yaw from frame t to frame t+1 (degrees).
        ry = rel_yaw[:, :CF - 1, 0].float()          # [B, CF-1]
        cumul_yaw = torch.zeros(B, CF, device=lat_cond_norm.device, dtype=torch.float32)
        for t in range(CF - 2, -1, -1):
            cumul_yaw[:, t] = cumul_yaw[:, t + 1] + ry[:, t]
        # cumul_yaw[:, CF-1] = 0 — reference frame, no shift

        # Convert degrees → azimuth column shift  (32 columns / 360°)
        col_shifts = (cumul_yaw * w / 360.0).round().long()  # [B, CF]

        # Vectorised gather-based roll — avoids Python loops over batch
        lat_grid = lat_cond_norm.view(B, CF, h, w, C)
        w_idx     = torch.arange(w, device=lat_grid.device).view(1, 1, 1, w, 1).expand(B, CF, h, w, C)
        shifts_exp = col_shifts[:, :, None, None, None].expand(B, CF, h, w, C)
        gather_idx = (w_idx - shifts_exp) % w
        warped = torch.gather(lat_grid, dim=3, index=gather_idx)

        return warped.view(B, CF, L, C)

    def _normalize_pose(self, traj: torch.Tensor) -> torch.Tensor:
        """Map (x_m, y_m, yaw_deg) to [-1, 1]. Works on any prefix shape."""
        t = traj.clone().float()
        t[..., 0] = 2.0 * t[..., 0] / self.pose_x_bound - 1.0
        t[..., 1] = t[..., 1] / self.pose_y_bound
        t[..., 2] = t[..., 2] / self.yaw_bound
        return t

    def _denormalize_pose(self, traj: torch.Tensor) -> torch.Tensor:
        """Inverse of _normalize_pose."""
        t = traj.clone().float()
        t[..., 0] = (t[..., 0] + 1.0) * self.pose_x_bound / 2.0
        t[..., 1] = t[..., 1] * self.pose_y_bound
        t[..., 2] = t[..., 2] * self.yaw_bound
        return t

    def _load_ckpt(self, path):
        sd = torch.load(path, map_location='cpu').get('model_state_dict', {})
        for attr, prefixes in [
            ('model',           ['module.model.',           'model.']),
            ('dit',             ['module.dit.',             'dit.']),
            ('pose_dit',        ['module.pose_dit.',        'pose_dit.']),
            ('skip_aggregator', ['module.skip_aggregator.', 'skip_aggregator.']),
            ('repa_proj',       ['module.repa_proj.',       'repa_proj.']),
        ]:
            obj = getattr(self, attr, None)
            if obj is None:
                continue
            for pfx in prefixes:
                obj_sd = {k: sd[pfx + k] for k in obj.state_dict() if pfx + k in sd}
                if obj_sd:
                    obj.load_state_dict(obj_sd, strict=False)
                    break
        print(f"[SwinDiT] STT+DiT+PoseDiT+SkipAgg loaded from {path}")

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
        # sample=False: use mu only — deterministic, KL-normalised latent for Stage 2
        z, _mu, _logvar, sk = self.swin_encoder(
            rearrange(x, 'b t c h w -> (b t) c h w'), sample=False
        )
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
        _, _mu, _logvar, skips = self.swin_encoder(x_frame, sample=False)
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
                # Encode conditioning frames separately to get their skip features
                _, raw_cond_skips = self.encode_sequence(features)
            else:
                lat_all, all_skips_enc = self.encode_sequence(features_all)
                # Conditioning skips: first CF frames from the joint encode
                raw_cond_skips = [s[:, :self.condition_frames] for s in all_skips_enc]
            # raw_cond_skips[i]: [B, CF, N_i, D_i]

        # Temporal skip aggregation (trainable; grads flow through aggregator params,
        # not through the frozen encoder skip tensors)
        if self.skip_aggregator is not None:
            last_cond_skips = self.skip_aggregator(raw_cond_skips)
        else:
            last_cond_skips = [s[:, -1] for s in raw_cond_skips]

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

        # Option 1: continuous sinusoidal embedding — no quantisation loss
        pose_idx, yaw_idx = self._normalize_pose_continuous(rel_pose, rel_yaw)

        lat_cond_norm = lat_cond / self.latent_scale
        # Option 3: azimuth-warp conditioning latents to align to last frame's viewpoint
        lat_cond_norm = self._apply_azimuth_warp(lat_cond_norm, rel_yaw)
        lat_target_norm = (lat_target / self.latent_scale).unsqueeze(1)          # [B,1,T,C]
        lat_all_norm  = torch.cat([lat_cond_norm, lat_target_norm], dim=1)       # [B,CF+1,T,C]
        logits        = self.model(lat_all_norm, pose_idx, yaw_idx,
                                   drop_feature=self.args.drop_feature)
        stt_features  = logits['logits']
        pose_emb_all  = logits['pose_emb']

        B = lat_target.shape[0]
        CF = self.condition_frames
        stt_last = rearrange(stt_features, '(b cf) s e -> b cf s e', b=B)[:, -1]

        tgt_norm = lat_target / self.latent_scale

        # ── Parallel DiT execution (mirrors model.py / train_deepspeed.py) ──────
        # Both PoseDiT and FluxDiT take stt_last as context and run independently.
        # FluxDiT is conditioned on the *last conditioning frame's* relative pose
        # (rel_pose[:, -2:-1]) rather than the current PoseDiT prediction, breaking
        # the sequential data dependency. rel_pose[:, -2:-1] is always the last
        # conditioning transition — no target-frame leakage in either AR path:
        #   j=0 (rel_pose_cond=None): rel_pose is [B, CF, 2] from GT; [-2] = (CF-2)→(CF-1)
        #   j>0 (rel_pose_cond set):  rel_pose is [B, CF+1, 2]; [-2] = last predicted cond pose
        last_cond_xy  = rel_pose[:, -2:-1].float()   # [B, 1, 2]
        last_cond_yaw = rel_yaw[:, -2:-1].float()    # [B, 1, 1]
        lc_pose_norm, lc_yaw_norm = self._normalize_pose_continuous(
            last_cond_xy, last_cond_yaw,
        )
        y_flux = self.model.get_pose_emb(lc_pose_norm, lc_yaw_norm)  # [B, n_embd*3]

        # ── PoseDiT: predict next relative pose, conditioned on stt_last ──────
        pose_target_xy  = rel_pose[:, -1:]           # [B, 1, 2]  — GT target pose
        pose_target_yaw = rel_yaw[:, -1:]            # [B, 1, 1]
        pose_target     = torch.cat([pose_target_xy, pose_target_yaw], dim=-1)  # [B, 1, 3]
        pose_target_norm = self._normalize_pose(pose_target).to(torch.bfloat16)

        u_pose = torch.randn((B, 1, 1), device=tgt_norm.device)
        t_pose = torch.sigmoid(u_pose)
        pose_loss_terms = self.pose_dit.training_losses(
            traj=pose_target_norm,
            traj_ids=self.pose_traj_ids[:B],
            cond=stt_last,
            cond_ids=self.cond_ids[:B],
            t=t_pose,
            return_predict=True,                     # always needed for AR conditioning
        )
        raw_pose_loss = pose_loss_terms['loss'].mean()
        pose_loss     = self.lambda_pose * raw_pose_loss

        # Denormalise predicted pose for AR conditioning and physical-unit outputs
        predict_pose_norm = pose_loss_terms['predict']          # [B, 1, 3] normalised
        predict_pose_raw  = self._denormalize_pose(predict_pose_norm.float())  # [B, 1, 3]
        predict_pose_xy   = predict_pose_raw[:, :, :2]          # [B, 1, 2]
        predict_pose_yaw  = predict_pose_raw[:, :, 2:3]         # [B, 1, 1]

        # ── FluxDiT: predict next latent, conditioned on stt_last + y_flux ───
        # Both losses backprop jointly through stt_last — STT features are
        # optimised for both pose prediction and frame generation simultaneously.
        u = torch.randn((B, 1, 1), device=tgt_norm.device)
        t_sample = torch.sigmoid(u)
        guidance = torch.full((B,), float(getattr(self.args, 'cfg_scale', 3.5)),
                              device=tgt_norm.device, dtype=tgt_norm.dtype)
        _use_repa = (self.repa_weight > 0 and step >= self.repa_start
                     and self.repa_proj is not None)
        loss_terms = self.dit.training_losses(
            img=tgt_norm, img_ids=self.img_ids[:B],
            cond=stt_last, cond_ids=self.cond_ids[:B],
            t=t_sample, y=y_flux, guidance=guidance,
            return_predict=self.args.return_predict,
            return_hidden=_use_repa,
            hidden_after_block=self.repa_layer,
        )
        diff_loss    = loss_terms['loss'].mean()
        predict_lats = loss_terms.get('predict')

        # ── REPA loss ─────────────────────────────────────────────────────────
        # Teacher: tgt_norm [B, 64, 768] — frozen Swin encoder output normalised
        #          by latent_scale (same representation the DiT processes as x_0).
        # Student: img-stream hidden state after double_blocks[repa_layer], float32.
        # Maximise token-wise cosine similarity → align DiT intermediate
        # representations with the domain encoder's semantic space.
        z_repa = torch.zeros(1, device=tgt_norm.device, dtype=torch.float32)
        if _use_repa and loss_terms.get('hidden') is not None:
            dit_h    = loss_terms['hidden']              # [B, 64, hidden_size=768], float32
            # Use the *normalised* clean latent as teacher — identical domain to what
            # the DiT denoises (x_0 = tgt_norm).  Cosine similarity is scale-invariant
            # so this is equivalent to using lat_target, but more principled.
            teacher  = tgt_norm.detach().float()         # [B, 64, 768], no grad
            proj     = self.repa_proj(dit_h)             # [B, 64, 768]
            # Token-wise cosine similarity averaged over spatial tokens and batch
            z_repa   = -torch.nn.functional.cosine_similarity(
                proj, teacher, dim=-1
            ).mean()

        z_cd = z_bev = torch.zeros(1, device=tgt_norm.device, dtype=tgt_norm.dtype)
        predict_decoded = None

        if predict_lats is not None:
            predict_decoded = self.decode_latents(predict_lats, last_cond_skips)
            gt_img = features_gt[:, 0]

            # ── Range-view Berhu loss ─────────────────────────────────────────
            # Provides gradient to TemporalSkipAggregator (flows through the
            # frozen decoder's forward pass into last_cond_skips → aggregator).
            if self.range_view_loss_weight > 0:
                valid = (gt_img[:, 0].abs() > 1e-4).float()
                n_v   = valid.sum().clamp(min=1.)
                rv_px = berhu_loss(predict_decoded[:, 0].float(), gt_img[:, 0].float())
                dist_w = 1.0 + gt_img[:, 0].detach().float().clamp(0., 1.)
                z_rv  = (rv_px * dist_w * valid).sum() / n_v
                diff_loss = diff_loss + self.range_view_loss_weight * z_rv

            if self.log_range:
                gt_unnorm = torch.exp2(gt_img[:, 0].float() * 6.) - 1.
            else:
                m0, s0 = self.proj_img_mean[0], self.proj_img_stds[0]
                gt_unnorm = gt_img[:, 0].float() * s0 + m0

            if self.chamfer_loss_weight > 0 and step >= self.chamfer_start:
                if self.log_range:
                    pd = (torch.exp2(predict_decoded[:, 0].float() * 6.) - 1.).reshape(B, -1)
                else:
                    pd = (predict_decoded[:, 0].float() * s0 + m0).reshape(B, -1)
                gd = gt_unnorm.reshape(B, -1)
                # Scale by mean timestep: high-noise steps produce noisy decoded
                # predictions, so down-weight their Chamfer contribution.
                z_cd = batch_chamfer_distance(
                    pd, gd, pd > 0.5, gd > 0.5,
                    self.range_projector, self.chamfer_max_pts
                ) * t_sample.mean()

            if self.log_w_bev is not None and self.bev_perceptual_fn is not None:
                if self.log_range:
                    pd = (torch.exp2(predict_decoded[:, 0].float() * 6.) - 1.).reshape(B, -1)
                else:
                    pd = (predict_decoded[:, 0].float() * s0 + m0).reshape(B, -1)
                gd = gt_unnorm.reshape(B, -1)
                z_bev = self.bev_perceptual_fn(pd, gd, pd > 0.5, gd > 0.5) * t_sample.mean()
                diff_loss = diff_loss + self._uw(self.log_w_bev, z_bev)

        # ── Combined loss (all terms explicit, individually tunable) ─────────
        # loss_diff  : flow-matching MSE (min-SNR weighted) + range-view Berhu
        # loss_pose  : PoseDiT rectified-flow loss (λ_pose = lambda_yaw_pose)
        # loss_chamfer: 3-D Chamfer distance (λ = chamfer_loss_weight)
        # loss_repa  : REPA cosine-alignment (λ = repa_weight)
        loss_all = (diff_loss
                    + pose_loss
                    + self.chamfer_loss_weight * z_cd
                    + self.repa_weight * z_repa)

        # STT conditioning diagnostics (detached scalars — no memory overhead)
        stt_last_f = stt_last.float()
        stt_last_norm = stt_last_f.norm(dim=-1).mean().detach()
        stt_last_std  = stt_last_f.std(0).mean().detach()

        return {
            'loss_all':         loss_all,
            'loss_diff':        loss_terms['loss'].mean().detach(),
            'loss_pose':        pose_loss.detach(),
            'loss_rv':          z_rv.detach() if predict_decoded is not None and self.range_view_loss_weight > 0
                                else torch.zeros(1, device=tgt_norm.device),
            'loss_chamfer':     z_cd.detach(),
            'loss_repa':        z_repa.detach(),
            'loss_bev_percep':  z_bev,
            'predict':          predict_decoded,
            'predict_latents':  predict_lats,
            'latents_cond_enc': lat_cond,
            # Predicted pose for next AR step (physical units, detached in training loop)
            'predict_pose_xy':  predict_pose_xy,
            'predict_pose_yaw': predict_pose_yaw,
            # STT conditioning diagnostics
            'stt_last_norm':    stt_last_norm,
            'stt_last_std':     stt_last_std,
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
        # all_skips[i]: [B, CF, N_i, D_i] — temporal aggregation or last-frame fallback
        if self.skip_aggregator is not None:
            last_cond_skips = self.skip_aggregator(all_skips)
        else:
            last_cond_skips = [s[:, -1] for s in all_skips]

        pose_idx, yaw_idx = self._normalize_pose_continuous(rel_pose, rel_yaw)
        lat_norm = self._apply_azimuth_warp(lat / self.latent_scale, rel_yaw)

        stt_feat, pose_emb = self.model.evaluate(lat_norm, pose_idx, yaw_idx,
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
