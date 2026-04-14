"""
Calibrate latent_scale for the RangeLDM VAE used in stage-2 DiT training.

latent_scale = std( encode_to_z(x) )  over the training set.

The model divides patchified VAE latents by latent_scale before feeding them
into the DiT (model_rangeview.py:898).  With latent_scale=1.0 (the default)
and a weakly-regularised VAE (kl_weight=1e-6), the actual latent std is
typically 2–4, making the flow-matching noise schedule heavily imbalanced.

Run this script once after Stage 1 finishes, then set the printed value as
``latent_scale`` in configs/dit_config_rangeview.py.

Usage:
    python scripts/calibrate_latent_scale_rangeldm.py \\
        --config  configs/dit_config_rangeview.py \\
        --vae_ckpt /scratch/.../vae_stage1_step401000.pth \\
        --n_batches 200 \\
        --batch_size 8

What is measured
----------------
Two quantities are reported:

  1. Per-channel stats of the raw VAE spatial latent [B, 4, 16, 512]:
     Shows whether individual VAE channels have different scales.
     Use these to decide whether per-channel normalisation is also needed.

  2. Global std of the patchified latent [B*T, L, latent_C]:
     This is exactly what latent_scale should be set to, since the model
     divides encode_to_z() output (post-patchify) by latent_scale.
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from utils.config_utils import Config
from models.modules.rangeldm_vae import RangeLDMVAE
from models.modules.tokenizer import patchify
from dataset.dataset_kitti_rangeview import KITTIRangeViewVAEDataset


def parse_args():
    p = argparse.ArgumentParser(description='Calibrate latent_scale for RangeLDM VAE')
    p.add_argument('--config',     required=True,  help='Path to dit_config_rangeview.py')
    p.add_argument('--vae_ckpt',   required=True,  help='Stage-1 VAE checkpoint (.pth)')
    p.add_argument('--n_batches',  default=200, type=int,
                   help='Batches to sample — 200 × bs=8 ≈ 1600 frames, enough for stable stats')
    p.add_argument('--batch_size', default=8, type=int)
    p.add_argument('--num_workers', default=4, type=int)
    return p.parse_args()


# ── Welford online mean/variance (numerically stable, single pass) ────────────

class WelfordAccumulator:
    """Track mean and variance of a scalar stream without storing all values."""
    def __init__(self):
        self.n    = 0
        self.mean = 0.0
        self.M2   = 0.0

    def update_tensor(self, t: torch.Tensor):
        """Add all scalar values from tensor t."""
        vals = t.float().cpu().numpy().ravel()
        for v in vals:
            self.n   += 1
            delta     = v - self.mean
            self.mean += delta / self.n
            self.M2  += delta * (v - self.mean)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.M2 / (self.n - 1))) if self.n > 1 else 0.0

    @property
    def var(self) -> float:
        return float(self.M2 / (self.n - 1)) if self.n > 1 else 0.0


# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    cfg    = Config.fromfile(args.config)

    # ── Load frozen VAE ───────────────────────────────────────────────────────
    print(f'Loading VAE checkpoint: {args.vae_ckpt}')
    vae = RangeLDMVAE(ckpt_path=None).cuda().eval()
    state = torch.load(args.vae_ckpt, map_location='cpu')
    # Accept both bare state_dict and checkpoint dicts with 'state_dict' key
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    missing, unexpected = vae.load_state_dict(state, strict=False)
    if missing:
        print(f'  WARNING — missing keys: {missing[:5]}{"..." if len(missing) > 5 else ""}')
    if unexpected:
        print(f'  WARNING — unexpected keys: {unexpected[:5]}{"..." if len(unexpected) > 5 else ""}')
    for p in vae.parameters():
        p.requires_grad_(False)

    # Patchification parameters from config
    patch_h = int(getattr(cfg, 'patch_size_h', getattr(cfg, 'patch_size', 8)))
    patch_w = int(getattr(cfg, 'patch_size_w', getattr(cfg, 'patch_size', 8)))
    z_ch    = int(getattr(cfg, 'vae_embed_dim', 4))
    print(f'Patch size: ({patch_h}, {patch_w})  VAE z_channels: {z_ch}')

    # Derived token dimensions (for display)
    h_lat = cfg.range_h // (cfg.downsample_size * patch_h)
    w_lat = cfg.range_w // (cfg.downsample_size * patch_w)
    L     = h_lat * w_lat
    latent_C = z_ch * patch_h * patch_w
    print(f'Token grid: {h_lat}×{w_lat} = {L} tokens,  latent_C = {latent_C}')

    # ── Dataset (single frames — we only need individual range views) ─────────
    ds = KITTIRangeViewVAEDataset(
        sequences_path=cfg.kitti_sequences_path,
        poses_path=cfg.kitti_poses_path,
        sequences=cfg.train_sequences,
        h=cfg.range_h,
        w=cfg.range_w,
        fov_up=cfg.fov_up,
        fov_down=cfg.fov_down,
        fov_left=cfg.fov_left,
        fov_right=cfg.fov_right,
        proj_img_mean=cfg.proj_img_mean,
        proj_img_stds=cfg.proj_img_stds,
        augmentation_config=None,      # no augmentation — measure clean statistics
        pc_extension=cfg.pc_extension,
        pc_dtype=getattr(np, cfg.pc_dtype),
        pc_reshape=tuple(cfg.pc_reshape),
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    print(f'Dataset: {len(ds)} frames  |  collecting {args.n_batches} batches '
          f'(≈ {args.n_batches * args.batch_size} frames)')

    # ── Accumulators ──────────────────────────────────────────────────────────
    # Per-channel stats on the raw spatial VAE latent [B, 4, H_lat, W_lat]
    ch_acc = [WelfordAccumulator() for _ in range(z_ch)]
    # Global stats on the patchified latent [B, L, latent_C]  ← latent_scale target
    global_acc = WelfordAccumulator()

    # ── Collection loop ───────────────────────────────────────────────────────
    with torch.no_grad():
        for i, (frames, _) in enumerate(loader):
            if i >= args.n_batches:
                break

            # frames: [B, C, H, W]  from KITTIRangeViewVAEDataset
            x = frames.cuda()

            # 1. Raw VAE spatial latent  [B, z_ch, H/4, W/4]
            vae_dtype = next(vae.parameters()).dtype
            z_spatial = vae.encode(x.to(vae_dtype))          # [B, z_ch, H_lat, W_lat]

            # Per-channel accumulation
            for c in range(z_ch):
                ch_acc[c].update_tensor(z_spatial[:, c])

            # 2. Patchified latent  [B, L, latent_C]  — same as encode_to_z output
            z_patch = patchify(z_spatial, patch_h, patch_w)  # [B, L, latent_C]
            global_acc.update_tensor(z_patch)

            if (i + 1) % 20 == 0:
                print(f'  batch {i+1:>3}/{args.n_batches}  '
                      f'frames={global_acc.n // latent_C:,}  '
                      f'global_std={global_acc.std:.4f}')

    # ── Report ────────────────────────────────────────────────────────────────
    sep = '=' * 60
    print(f'\n{sep}')
    print(f'  RangeLDM VAE Latent Statistics')
    print(f'  Checkpoint : {args.vae_ckpt}')
    print(f'  Frames     : {global_acc.n // latent_C:,}')
    print(sep)

    print(f'\n  Raw spatial latent [B, {z_ch}, {cfg.range_h//cfg.downsample_size}, '
          f'{cfg.range_w//cfg.downsample_size}] — per channel:')
    print(f'  {"Ch":>4}  {"mean":>10}  {"std":>10}')
    print(f'  {"-"*30}')
    for c, acc in enumerate(ch_acc):
        print(f'  {c:>4}  {acc.mean:>10.5f}  {acc.std:>10.5f}')

    print(f'\n  Patchified latent [B, {L}, {latent_C}] — global:')
    print(f'  mean = {global_acc.mean:.6f}')
    print(f'  std  = {global_acc.std:.6f}')

    # Per-channel std range gives a sense of how much per-channel normalisation
    # would help vs. a single global latent_scale scalar
    ch_stds = [acc.std for acc in ch_acc]
    print(f'\n  Per-channel std range: [{min(ch_stds):.4f}, {max(ch_stds):.4f}]')
    if max(ch_stds) / (min(ch_stds) + 1e-8) > 2.0:
        print(f'  ⚠  Channel std ratio > 2×: consider per-channel normalisation')
        print(f'     in addition to the global latent_scale.')
    else:
        print(f'  ✓  Channel stds are within 2× of each other — '
              f'global latent_scale is sufficient.')

    print(f'\n{sep}')
    print(f'  → Set in configs/dit_config_rangeview.py:')
    print(f'      latent_scale = {global_acc.std:.4f}')
    print(sep)


if __name__ == '__main__':
    main()
