#!/usr/bin/env python3
"""
tsne_latents.py

Visualise the latent space of DINOv2 patch tokens extracted from KITTI
range-view images, and optionally compare them against the RangeLDM VAE
latents used in the current training pipeline.

DINOv2 adaptation follows the DiffLoc approach (liw95/DiffLoc):
  - ViT-S/14 pretrained weights loaded from torch hub
  - Patch embedding first-conv adapted from 3-channel RGB → 2-channel
    [range, intensity] by averaging and rescaling pretrained weights
  - Range images resized from 64×2048 → 56×448 before the ViT, giving a
    4×32 = 128 patch grid — identical to the current VAE token count (L=128)

Outputs (saved to --output_dir):
  dino_tsne_by_frame.png       t-SNE coloured by temporal frame index
  dino_tsne_by_elevation.png   t-SNE coloured by patch elevation row
  dino_tsne_by_azimuth.png     t-SNE coloured by patch azimuth column
  dino_spatial_pca.png         PCA-RGB spatial token map over N example frames
  vae_tsne_by_frame.png        (if --vae_ckpt) VAE latent t-SNE by frame
  vae_tsne_by_elevation.png    (if --vae_ckpt) VAE latent t-SNE by elevation
  vae_spatial_pca.png          (if --vae_ckpt) VAE PCA-RGB spatial token map
  comparison_tsne.png          (if --vae_ckpt) side-by-side DINOv2 vs VAE

Usage:
    python scripts/tsne_latents.py \\
        --config configs/dit_config_rangeview.py \\
        --sequence 0 \\
        --n_frames 64 \\
        --output_dir outputs/tsne_latents \\
        [--vae_ckpt /path/to/vae.pkl]
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from utils.config_utils import Config
from dataset.dataset_kitti_rangeview import KITTIRangeViewDataset

# ─────────────────────────────────────────────────────────────────────────────
# Target resolution for DINOv2 forward pass.
#
# KITTI range images are 64×2048.  DINOv2 ViT-S/14 has patch_size=14, so we
# need H and W to be multiples of 14.
#
#   DINO_H = 4 × 14 = 56   (4 elevation patches)
#   DINO_W = 32 × 14 = 448 (32 azimuth patches)
#   DINO_N_TOKENS = 4 × 32 = 128  ← matches the VAE token count L=128
#
# The positional embeddings are interpolated inside DINOv2's forward_features.
# ─────────────────────────────────────────────────────────────────────────────
DINO_H        = 56     # 4 × patch_size(14)
DINO_W        = 448    # 32 × patch_size(14)
DINO_PATCH_H  = 4      # elevation patch rows
DINO_PATCH_W  = 32     # azimuth  patch columns
DINO_N_TOKENS = DINO_PATCH_H * DINO_PATCH_W   # 128


# ─────────────────────────────────────────────────────────────────────────────
# DINOv2 loading and adaptation
# ─────────────────────────────────────────────────────────────────────────────

def _adapt_patch_embed(model: nn.Module, in_channels: int) -> nn.Module:
    """Replace the 3-channel patch-embed Conv2d with an ``in_channels``-channel
    one, initialised by averaging the pretrained RGB weights and rescaling to
    preserve the expected activation magnitude.

    The scaling factor ``3 / in_channels`` compensates for fewer input channels
    so that the variance of the output feature maps stays consistent with what
    the pretrained transformer layers expect.
    """
    old_proj  = model.patch_embed.proj                  # Conv2d(3, 384, 14, 14)
    old_w     = old_proj.weight.data                    # [384, 3, 14, 14]

    new_proj  = nn.Conv2d(
        in_channels, old_proj.out_channels,
        kernel_size  = old_proj.kernel_size,
        stride       = old_proj.stride,
        padding      = old_proj.padding,
        bias         = old_proj.bias is not None,
    )

    # Average over the 3 RGB channels → [384, 1, 14, 14], then tile to
    # in_channels and rescale.
    avg_w = old_w.mean(dim=1, keepdim=True)             # [384, 1, 14, 14]
    new_w = avg_w.repeat(1, in_channels, 1, 1) * (3.0 / in_channels)
    new_proj.weight = nn.Parameter(new_w)
    if old_proj.bias is not None:
        new_proj.bias = nn.Parameter(old_proj.bias.data.clone())

    model.patch_embed.proj = new_proj
    return model


def build_dinov2_rangeview(device: torch.device, in_channels: int = 2) -> nn.Module:
    """Load DINOv2 ViT-S/14 and adapt the patch embedding for range-view images.

    Args:
        device:      Compute device.
        in_channels: Number of input channels (2 for [range, intensity]).

    Returns:
        Frozen DINOv2 model on ``device``, ready for ``forward_features()``.
    """
    print("Loading DINOv2 ViT-S/14 from torch hub …")
    model = torch.hub.load(
        'facebookresearch/dinov2', 'dinov2_vits14',
        pretrained=True,
        verbose=False,
    )

    model = _adapt_patch_embed(model, in_channels)

    # Freeze — used as a fixed feature extractor
    for p in model.parameters():
        p.requires_grad_(False)

    model.eval().to(device)
    print(
        f"  patch_embed adapted: 3-ch → {in_channels}-ch  |  "
        f"resize target: {DINO_H}×{DINO_W}  |  "
        f"tokens per image: {DINO_N_TOKENS}"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Token extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_dinov2_tokens(
    model   : nn.Module,
    images  : torch.Tensor,   # [B, 2, H, W]  normalised
    device  : torch.device,
) -> torch.Tensor:
    """Resize range images to DINO_H×DINO_W and extract DINOv2 patch tokens.

    DINOv2's ``forward_features`` internally calls ``interpolate_pos_encoding``
    to stretch the pretrained positional embeddings to the new grid size, so no
    manual pos-embed surgery is required.

    Returns:
        tokens: [B, DINO_N_TOKENS, 384]
    """
    x = F.interpolate(
        images.to(device, dtype=torch.float32),
        size=(DINO_H, DINO_W),
        mode='bilinear',
        align_corners=False,
    )   # [B, 2, 56, 448]

    out    = model.forward_features(x)
    tokens = out['x_norm_patchtokens']   # [B, 128, 384] — CLS excluded
    return tokens.cpu()


@torch.no_grad()
def extract_vae_tokens(
    tokenizer,
    images : torch.Tensor,   # [B, 1, 2, H, W]  — T=1 temporal dim
    device : torch.device,
) -> torch.Tensor:
    """Extract RangeLDM VAE latent tokens for comparison.

    Returns:
        tokens: [B, L, latent_C]  e.g. [B, 128, 256]
    """
    b = images.shape[0]
    toks = tokenizer.encode_to_z(images.to(device))    # [B, 1, L, C]
    return toks[:, 0].cpu()                             # [B, L, C]


# ─────────────────────────────────────────────────────────────────────────────
# Dimensionality reduction
# ─────────────────────────────────────────────────────────────────────────────

def run_tsne(
    features     : np.ndarray,   # [N, D]
    perplexity   : int = 30,
    n_iter       : int = 1000,
    random_state : int = 42,
) -> np.ndarray:
    """PCA pre-reduction to 50 dims (standard speed-up), then t-SNE to 2D."""
    n_components_pca = min(50, features.shape[0] - 1, features.shape[1])
    if features.shape[1] > n_components_pca:
        pca      = PCA(n_components=n_components_pca, random_state=random_state)
        features = pca.fit_transform(features)

    tsne = TSNE(
        n_components  = 2,
        perplexity    = perplexity,
        n_iter        = n_iter,
        random_state  = random_state,
        verbose       = 1,
    )
    return tsne.fit_transform(features)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_tsne(
    embedding  : np.ndarray,   # [N, 2]
    labels     : np.ndarray,   # [N]
    title      : str,
    label_name : str,
    save_path  : str,
    cmap       : str   = 'viridis',
    s          : float = 6.0,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=labels, cmap=cmap, s=s, alpha=0.7, linewidths=0,
    )
    fig.colorbar(sc, ax=ax, label=label_name)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_spatial_pca(
    tokens        : np.ndarray,   # [N, patch_h, patch_w, D]
    title         : str,
    save_path     : str,
    n_frames_show : int = 6,
) -> None:
    """Reduce token embeddings to 3 PCA components → display as RGB spatial map.

    Each patch in the H×W grid becomes a pixel whose RGB colour encodes the
    dominant feature directions captured by PCA.  Showing several frames as
    columns lets you see how the scene structure changes over time.

    This is the same visualisation strategy used in DINOv2's official demos
    to show that the model captures semantic structure without supervision.
    """
    N, ph, pw, D = tokens.shape
    n_frames_show = min(n_frames_show, N)

    flat    = tokens.reshape(-1, D)                   # [N*ph*pw, D]
    pca     = PCA(n_components=3)
    rgb_f   = pca.fit_transform(flat)                 # [N*ph*pw, 3]

    # Normalise each PC channel independently to [0, 1]
    for i in range(3):
        v = rgb_f[:, i]
        rgb_f[:, i] = (v - v.min()) / (v.max() - v.min() + 1e-8)

    rgb = rgb_f.reshape(N, ph, pw, 3)                 # [N, ph, pw, 3]

    frame_indices = np.linspace(0, N - 1, n_frames_show, dtype=int)
    fig, axes = plt.subplots(1, n_frames_show, figsize=(3.5 * n_frames_show, 3))
    if n_frames_show == 1:
        axes = [axes]

    for col, idx in enumerate(frame_indices):
        axes[col].imshow(rgb[idx], aspect='auto', interpolation='nearest')
        axes[col].set_title(f'frame {idx}', fontsize=9)
        axes[col].axis('off')

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_comparison_tsne(
    emb_dino  : np.ndarray,   # [N, 2]
    emb_vae   : np.ndarray,   # [N, 2]
    labels    : np.ndarray,   # [N]
    seq_id    : int,
    save_path : str,
) -> None:
    """Side-by-side t-SNE scatter for DINOv2 vs VAE latents."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    names = ['DINOv2 patch tokens (384-d)', 'VAE latent tokens (256-d)']
    for ax, emb, name in zip(axes, [emb_dino, emb_vae], names):
        sc = ax.scatter(
            emb[:, 0], emb[:, 1],
            c=labels, cmap='plasma', s=5, alpha=0.65, linewidths=0,
        )
        fig.colorbar(sc, ax=ax, label='frame index')
        ax.set_title(name, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.axis('off')
    fig.suptitle(f't-SNE latent comparison — KITTI seq {seq_id}', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="t-SNE of DINOv2 patch tokens from KITTI range-view images"
    )
    p.add_argument('--config', type=str,
                   default='configs/dit_config_rangeview.py',
                   help='Path to range-view config file')
    p.add_argument('--sequence', type=int, default=0,
                   help='KITTI sequence index')
    p.add_argument('--n_frames', type=int, default=64,
                   help='Number of frames to embed (more → richer t-SNE)')
    p.add_argument('--batch_size', type=int, default=8,
                   help='Batch size for DINOv2 forward passes')
    p.add_argument('--output_dir', type=str,
                   default='outputs/tsne_latents',
                   help='Directory to write output PNG files')
    p.add_argument('--vae_ckpt', type=str, default=None,
                   help='Optional RangeLDM VAE checkpoint (.pkl) for comparison')
    p.add_argument('--perplexity', type=int, default=30,
                   help='t-SNE perplexity (try 5–50; smaller for few frames)')
    p.add_argument('--n_iter', type=int, default=1000,
                   help='Number of t-SNE optimisation iterations')
    p.add_argument('--n_spatial_frames', type=int, default=6,
                   help='Number of example frames shown in spatial PCA plots')
    p.add_argument('--device', type=str, default='cuda',
                   help='Compute device (cuda / cpu)')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(
        args.device if (args.device == 'cpu' or not torch.cuda.is_available())
        else args.device
    )
    print(f"Device: {device}")

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = Config.fromfile(args.config)

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"\nLoading KITTI sequence {args.sequence} …")
    dataset = KITTIRangeViewDataset(
        sequences_path = cfg.kitti_sequences_path,
        poses_path     = cfg.kitti_poses_path,
        sequences      = [args.sequence],
        condition_frames = 0,
        forward_iter     = 1,
        h              = cfg.range_h,
        w              = cfg.range_w,
        fov_up         = cfg.fov_up,
        fov_down       = cfg.fov_down,
        fov_left       = cfg.fov_left,
        fov_right      = cfg.fov_right,
        proj_img_mean  = cfg.proj_img_mean,
        proj_img_stds  = cfg.proj_img_stds,
        pc_extension   = cfg.pc_extension,
        pc_dtype       = getattr(np, cfg.pc_dtype),
        pc_reshape     = tuple(cfg.pc_reshape),
        is_train       = False,
    )

    n_frames = min(args.n_frames, len(dataset))
    print(f"  Dataset size: {len(dataset)}  |  using {n_frames} frames")
    if n_frames < 10:
        print("  WARNING: very few frames — t-SNE may not be informative")

    # Evenly-spaced frame indices across the sequence
    indices    = np.linspace(0, len(dataset) - 1, n_frames, dtype=int)
    frame_imgs = []
    for idx in indices:
        data, _ = dataset[int(idx)]   # [T=1, 2, H, W]
        frame_imgs.append(data[0])    # [2, H, W]
    frame_imgs = torch.stack(frame_imgs)   # [N, 2, H, W]

    # ── DINOv2 token extraction ────────────────────────────────────────────────
    dino_model = build_dinov2_rangeview(device, in_channels=2)

    print(f"\nExtracting DINOv2 patch tokens for {n_frames} frames …")
    dino_list = []
    for start in range(0, n_frames, args.batch_size):
        batch = frame_imgs[start : start + args.batch_size]
        toks  = extract_dinov2_tokens(dino_model, batch, device)   # [B, 128, 384]
        dino_list.append(toks)
        print(f"  {min(start + args.batch_size, n_frames)}/{n_frames}", end='\r')
    print()

    dino_tokens = torch.cat(dino_list, dim=0).numpy()   # [N, 128, 384]
    print(f"  DINOv2 tokens shape: {dino_tokens.shape}")

    # ── (Optional) VAE token extraction ───────────────────────────────────────
    vae_tokens = None
    if args.vae_ckpt is not None:
        from models.modules.tokenizer import RangeViewVAETokenizer

        class _Cfg:
            pass

        vcfg = _Cfg()
        vcfg.patch_size_h         = cfg.patch_size_h
        vcfg.patch_size_w         = cfg.patch_size_w
        vcfg.vae_embed_dim        = cfg.vae_embed_dim
        vcfg.vae_ckpt             = args.vae_ckpt
        vcfg.add_encoder_temporal = False

        tokenizer = RangeViewVAETokenizer(vcfg, local_rank=device.index or 0)
        tokenizer.eval()

        print(f"\nExtracting VAE latent tokens for {n_frames} frames …")
        vae_list = []
        for start in range(0, n_frames, args.batch_size):
            batch = frame_imgs[start : start + args.batch_size].unsqueeze(1)  # [B,1,2,H,W]
            toks  = extract_vae_tokens(tokenizer, batch, device)              # [B, L, C]
            vae_list.append(toks)
            print(f"  {min(start + args.batch_size, n_frames)}/{n_frames}", end='\r')
        print()

        vae_tokens = torch.cat(vae_list, dim=0).numpy()   # [N, L, C]
        print(f"  VAE tokens shape: {vae_tokens.shape}")

    # ── Label arrays ──────────────────────────────────────────────────────────
    # Attach three scalar labels to every token:
    #   frame_ids  — temporal position (0 … n_frames-1)
    #   elev_ids   — which elevation patch row (0 … DINO_PATCH_H-1)
    #   azim_ids   — which azimuth patch column (0 … DINO_PATCH_W-1)
    frame_ids = np.repeat(np.arange(n_frames), DINO_N_TOKENS)               # [N*L]
    elev_ids  = np.tile(
        np.repeat(np.arange(DINO_PATCH_H), DINO_PATCH_W), n_frames          # [N*L]
    )
    azim_ids  = np.tile(
        np.tile(np.arange(DINO_PATCH_W), DINO_PATCH_H), n_frames            # [N*L]
    )

    # ── DINOv2 t-SNE ──────────────────────────────────────────────────────────
    dino_flat = dino_tokens.reshape(-1, dino_tokens.shape[-1])   # [N*128, 384]
    print(f"\nRunning t-SNE on DINOv2 tokens {dino_flat.shape} …")
    dino_tsne = run_tsne(dino_flat, perplexity=args.perplexity, n_iter=args.n_iter)

    plot_tsne(
        dino_tsne, frame_ids,
        title      = f'DINOv2 patch tokens — by frame (seq {args.sequence})',
        label_name = 'frame index',
        save_path  = os.path.join(args.output_dir, 'dino_tsne_by_frame.png'),
        cmap='plasma',
    )
    plot_tsne(
        dino_tsne, elev_ids,
        title      = f'DINOv2 patch tokens — by elevation patch row (seq {args.sequence})',
        label_name = 'elevation row (0=top)',
        save_path  = os.path.join(args.output_dir, 'dino_tsne_by_elevation.png'),
        cmap='coolwarm', s=4,
    )
    plot_tsne(
        dino_tsne, azim_ids,
        title      = f'DINOv2 patch tokens — by azimuth patch column (seq {args.sequence})',
        label_name = 'azimuth column (0=left)',
        save_path  = os.path.join(args.output_dir, 'dino_tsne_by_azimuth.png'),
        cmap='twilight', s=4,
    )

    # ── DINOv2 spatial PCA map ─────────────────────────────────────────────────
    dino_spatial = dino_tokens.reshape(n_frames, DINO_PATCH_H, DINO_PATCH_W, -1)
    plot_spatial_pca(
        dino_spatial,
        title      = f'DINOv2 PCA-RGB token map — KITTI seq {args.sequence}',
        save_path  = os.path.join(args.output_dir, 'dino_spatial_pca.png'),
        n_frames_show = args.n_spatial_frames,
    )

    # ── VAE comparison ────────────────────────────────────────────────────────
    if vae_tokens is not None:
        L, C      = vae_tokens.shape[1], vae_tokens.shape[2]
        vae_flat  = vae_tokens.reshape(-1, C)            # [N*L, C]

        # If VAE token count differs from DINOv2 (should both be 128, but be safe)
        if L != DINO_N_TOKENS:
            print(
                f"  NOTE: VAE token count ({L}) differs from DINOv2 ({DINO_N_TOKENS}); "
                "frame/elev/azim labels will use DINOv2 grid."
            )
            vae_frame_ids = np.repeat(np.arange(n_frames), L)
            vae_elev_ids  = np.tile(np.repeat(np.arange(4), L // 4), n_frames)
            vae_azim_ids  = np.tile(np.tile(np.arange(L // 4), 4), n_frames)
        else:
            vae_frame_ids, vae_elev_ids, vae_azim_ids = frame_ids, elev_ids, azim_ids

        print(f"\nRunning t-SNE on VAE tokens {vae_flat.shape} …")
        vae_tsne  = run_tsne(vae_flat, perplexity=args.perplexity, n_iter=args.n_iter)

        plot_tsne(
            vae_tsne, vae_frame_ids,
            title      = f'VAE latent tokens — by frame (seq {args.sequence})',
            label_name = 'frame index',
            save_path  = os.path.join(args.output_dir, 'vae_tsne_by_frame.png'),
            cmap='plasma',
        )
        plot_tsne(
            vae_tsne, vae_elev_ids,
            title      = f'VAE latent tokens — by elevation patch row (seq {args.sequence})',
            label_name = 'elevation row',
            save_path  = os.path.join(args.output_dir, 'vae_tsne_by_elevation.png'),
            cmap='coolwarm', s=4,
        )

        vae_spatial = vae_tokens.reshape(n_frames, DINO_PATCH_H, DINO_PATCH_W, -1)
        plot_spatial_pca(
            vae_spatial,
            title      = f'VAE PCA-RGB token map — KITTI seq {args.sequence}',
            save_path  = os.path.join(args.output_dir, 'vae_spatial_pca.png'),
            n_frames_show = args.n_spatial_frames,
        )

        # Side-by-side comparison requires matching token counts
        if L == DINO_N_TOKENS:
            plot_comparison_tsne(
                dino_tsne, vae_tsne, frame_ids,
                seq_id    = args.sequence,
                save_path = os.path.join(args.output_dir, 'comparison_tsne.png'),
            )

    print(f"\nAll outputs written to: {args.output_dir}")


if __name__ == '__main__':
    main()
