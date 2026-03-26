#!/usr/bin/env python3
"""
tsne_latents.py

Visualise DINOv2 features extracted from KITTI range-view images across all
sequences (00-09) using the DiffLoc feature-learning architecture.

Architecture (from liw95/DiffLoc — models/stems.py, models/image_feature_extractor.py):
  ConvStem (SalsaNext-inspired ResBlocks)
    in_channels=2 (range, intensity)  →  256 patch tokens × 384-d
    Input 64×2048, patch_stride=(8,64)  →  grid (8,32) = 256 patches
  +
  Pretrained DINOv2 ViT-S/14 transformer blocks (12 layers)
  →  CLS token extracted as a 384-d global descriptor per frame

A single t-SNE image is saved, coloured by sequence ID, giving a bird's-eye
view of how DINOv2 features vary across scenes.

DINOv2 pretrained weights:
  Download dinov2_vits14_pretrain.pth from:
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
  and pass via --pretrained_path.
  Without it the weights are fetched automatically via torch.hub (needs internet).

Usage:
    python scripts/tsne_latents.py \\
        --config configs/dit_config_rangeview.py \\
        --n_frames_per_seq 50 \\
        --output_dir outputs/tsne_latents \\
        [--pretrained_path /path/to/dinov2_vits14_pretrain.pth]
"""

import os
import sys
import math
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
# ConvStem components — ported verbatim from liw95/DiffLoc (models/stems.py)
# Stem design originally from SalsaNext (github.com/TiagoCortinhal/SalsaNext)
# ─────────────────────────────────────────────────────────────────────────────

class ResContextBlock(nn.Module):
    """Residual context block used in DiffLoc's ConvStem (from SalsaNext)."""
    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1  = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), padding=1)
        self.act2  = nn.LeakyReLU()
        self.bn1   = nn.BatchNorm2d(out_filters)
        self.conv3 = nn.Conv2d(out_filters, out_filters, (3, 3), dilation=2, padding=2)
        self.act3  = nn.LeakyReLU()
        self.bn2   = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        shortcut = self.act1(self.conv1(x))
        resA = self.bn1(self.act2(self.conv2(shortcut)))
        resA = self.bn2(self.act3(self.conv3(resA)))
        return shortcut + resA


class ResBlock(nn.Module):
    """Residual block used in DiffLoc's ConvStem (from SalsaNext)."""
    def __init__(self, in_filters, out_filters, dropout_rate,
                 kernel_size=(3, 3), stride=1, pooling=True, drop_out=True):
        super().__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1  = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1)
        self.act2  = nn.LeakyReLU()
        self.bn1   = nn.BatchNorm2d(out_filters)
        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), dilation=2, padding=2)
        self.act3  = nn.LeakyReLU()
        self.bn2   = nn.BatchNorm2d(out_filters)
        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4  = nn.LeakyReLU()
        self.bn3   = nn.BatchNorm2d(out_filters)
        self.conv5 = nn.Conv2d(out_filters * 3, out_filters, kernel_size=(1, 1))
        self.act5  = nn.LeakyReLU()
        self.bn4   = nn.BatchNorm2d(out_filters)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        if pooling:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)

    def forward(self, x):
        shortcut = self.act1(self.conv1(x))
        resA1 = self.bn1(self.act2(self.conv2(x)))
        resA2 = self.bn2(self.act3(self.conv3(resA1)))
        resA3 = self.bn3(self.act4(self.conv4(resA2)))
        concat = torch.cat((resA1, resA2, resA3), dim=1)
        resA = self.bn4(self.act5(self.conv5(concat)))
        resA = shortcut + resA
        resB = self.dropout(resA) if self.drop_out else resA
        if self.pooling:
            return self.pool(resB), resA
        return resB


class ConvStem(nn.Module):
    """DiffLoc convolutional stem (liw95/DiffLoc, models/stems.py).

    Maps [B, in_channels, H, W] → [B, N_patches, embed_dim] where
    N_patches = (H // patch_stride[0]) × (W // patch_stride[1]).
    """
    def __init__(self, in_channels=2, base_channels=32, img_size=(64, 2048),
                 patch_stride=(8, 64), embed_dim=384, flatten=True, hidden_dim=None):
        super().__init__()
        assert patch_stride[0] % 2 == 0 and patch_stride[1] % 2 == 0
        if hidden_dim is None:
            hidden_dim = 2 * base_channels

        self.dropout_ratio = 0.2
        self.conv_block = nn.Sequential(
            ResContextBlock(in_channels,   base_channels),
            ResContextBlock(base_channels, base_channels),
            ResContextBlock(base_channels, base_channels),
            ResBlock(base_channels, hidden_dim, self.dropout_ratio,
                     pooling=False, drop_out=False),
        )

        kernel_size = (patch_stride[0] + 1, patch_stride[1] + 1)
        padding     = (patch_stride[0] // 2, patch_stride[1] // 2)
        self.proj_block = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size, stride=patch_stride, padding=padding),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=1),
        )

        self.patch_stride = tuple(patch_stride)
        self.patch_size   = self.patch_stride
        self.grid_size    = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])
        self.num_patches  = self.grid_size[0] * self.grid_size[1]
        self.flatten      = flatten

    def forward(self, x):
        x_base = self.conv_block(x)            # [B, hidden_dim, H, W]
        x      = self.proj_block(x_base)       # [B, embed_dim, grid_h, grid_w]
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        return x, x_base


# ─────────────────────────────────────────────────────────────────────────────
# DiffLoc-DINOv2 feature extractor
# Inspired by liw95/DiffLoc models/image_feature_extractor.py:
#   ConvStem replaces the ViT patch_embed; pretrained DINOv2 ViT-S/14 blocks
#   are reused for the transformer layers and the CLS token/pos_embed.
# ─────────────────────────────────────────────────────────────────────────────

# Grid produced by ConvStem from KITTI 64×2048 images with patch_stride=(8,64)
STEM_PATCH_STRIDE  = (8, 64)
STEM_BASE_CHANNELS = 32
STEM_HIDDEN_DIM    = 256        # D_h in DiffLoc
DINO_EMBED_DIM     = 384        # ViT-S/14 hidden dim
DINO_GRID_H        = 8          # 64  // 8
DINO_GRID_W        = 32         # 2048 // 64
DINO_N_PATCHES     = DINO_GRID_H * DINO_GRID_W   # 256


class DiffLocDINOv2(nn.Module):
    """ConvStem (DiffLoc) + pretrained DINOv2 ViT-S/14 blocks.

    Workflow (mirrors DiffLoc's ImageFeatureExtractor):
      1. ConvStem encodes the multi-channel range image to patch tokens.
      2. CLS token prepended; positional embedding added (bilinear-interpolated
         from the DINOv2 pretrained 16×16 grid to our 8×32 grid).
      3. 12 pretrained DINOv2 transformer blocks applied.
      4. CLS token returned as a 384-d global descriptor.
    """

    def __init__(self, in_channels: int = 2, pretrained_path: str = None):
        super().__init__()

        # ── ConvStem (randomly initialised, adapts input channels) ────────────
        self.conv_stem = ConvStem(
            in_channels   = in_channels,
            base_channels = STEM_BASE_CHANNELS,
            img_size      = (64, 2048),
            patch_stride  = STEM_PATCH_STRIDE,
            embed_dim     = DINO_EMBED_DIM,
            flatten       = True,
            hidden_dim    = STEM_HIDDEN_DIM,
        )

        # ── Load DINOv2 ViT-S/14 ─────────────────────────────────────────────
        if pretrained_path is not None:
            print(f"Building DINOv2 ViT-S/14 architecture via torch.hub …")
            dino = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vits14',
                pretrained=False, verbose=False,
            )
            print(f"Loading pretrained weights from {pretrained_path}")
            sd = torch.load(pretrained_path, map_location='cpu')
            if 'model' in sd:
                sd = sd['model']
            # Skip patch_embed — we use ConvStem instead (reuse_patch_emb=False
            # is the DiffLoc default for ConvStem mode)
            sd = {k: v for k, v in sd.items() if 'patch_embed' not in k}
            msg = dino.load_state_dict(sd, strict=False)
            print(f"  DINOv2 weight loading: {msg}")
        else:
            print("Loading DINOv2 ViT-S/14 from torch.hub (pretrained=True) …")
            dino = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vits14',
                pretrained=True, verbose=False,
            )

        # Borrow cls_token, pos_embed, transformer blocks, and final norm
        self.cls_token = nn.Parameter(dino.cls_token.data.clone())
        self.pos_embed = nn.Parameter(dino.pos_embed.data.clone())  # [1, 257, 384]
        self.blocks    = dino.blocks   # 12 × NestedTensorBlock
        self.norm      = dino.norm

        # Freeze pretrained components (visualisation — not fine-tuning)
        for p in self.cls_token, self.pos_embed:
            p.requires_grad_(False)
        for p in list(self.blocks.parameters()) + list(self.norm.parameters()):
            p.requires_grad_(False)

    def _interp_pos_embed(self, grid_h: int, grid_w: int) -> torch.Tensor:
        """Bilinear-resize pretrained pos_embed from (16,16) → (grid_h, grid_w).

        Mirrors DiffLoc's resize_pos_embed (models/model_utils.py) which in turn
        follows the standard ViT pos-embed interpolation recipe.
        """
        pos  = self.pos_embed               # [1, 1+gs_old², 384]
        cls  = pos[:, :1]                   # [1, 1, 384]
        grid = pos[:, 1:]                   # [1, gs_old², 384]
        gs   = int(math.sqrt(grid.shape[1]))  # 16 for ViT-S/14
        grid = grid.reshape(1, gs, gs, -1).permute(0, 3, 1, 2)   # [1, 384, 16, 16]
        grid = F.interpolate(grid, size=(grid_h, grid_w),
                             mode='bilinear', align_corners=False)
        grid = grid.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1)
        return torch.cat([cls, grid], dim=1)  # [1, 1+grid_h*grid_w, 384]

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, in_channels, 64, 2048]  →  [B, 384]  (CLS token)."""
        patches, _ = self.conv_stem(x)          # [B, 256, 384]
        B = patches.shape[0]

        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, 384]
        tok = torch.cat([cls, patches], dim=1)  # [B, 257, 384]

        pos = self._interp_pos_embed(DINO_GRID_H, DINO_GRID_W)
        tok = tok + pos

        for blk in self.blocks:
            tok = blk(tok)
        tok = self.norm(tok)

        return tok[:, 0]   # CLS token: [B, 384]


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction across sequences
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_sequence_features(
    model    : DiffLocDINOv2,
    cfg,
    seq_id   : int,
    n_frames : int,
    batch_sz : int,
    device   : torch.device,
) -> torch.Tensor | None:
    """Load one KITTI sequence, sample n_frames evenly, return [N, 384] CLS tokens."""
    try:
        ds = KITTIRangeViewDataset(
            sequences_path  = cfg.kitti_sequences_path,
            poses_path      = cfg.kitti_poses_path,
            sequences       = [seq_id],
            condition_frames= 0,
            forward_iter    = 1,
            h               = cfg.range_h,
            w               = cfg.range_w,
            fov_up          = cfg.fov_up,
            fov_down        = cfg.fov_down,
            fov_left        = cfg.fov_left,
            fov_right       = cfg.fov_right,
            proj_img_mean   = cfg.proj_img_mean,
            proj_img_stds   = cfg.proj_img_stds,
            pc_extension    = cfg.pc_extension,
            pc_dtype        = getattr(np, cfg.pc_dtype),
            pc_reshape      = tuple(cfg.pc_reshape),
            is_train        = False,
        )
    except Exception as e:
        print(f"  Seq {seq_id:02d}: skipped ({e})")
        return None

    n = min(n_frames, len(ds))
    if n == 0:
        return None
    indices = np.linspace(0, len(ds) - 1, n, dtype=int)

    imgs = []
    for idx in indices:
        data, _ = ds[int(idx)]   # [T=1, 2, H, W]
        imgs.append(data[0])     # [2, H, W]
    imgs = torch.stack(imgs)     # [N, 2, H, W]

    feats = []
    for s in range(0, n, batch_sz):
        batch = imgs[s : s + batch_sz].to(device, dtype=torch.float32)
        feats.append(model(batch).cpu())
    return torch.cat(feats, dim=0)   # [N, 384]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DINOv2 (DiffLoc-style) t-SNE on KITTI sequences 00-09"
    )
    p.add_argument('--config', default='configs/dit_config_rangeview.py')
    p.add_argument('--n_frames_per_seq', type=int, default=50,
                   help='Frames sampled from each sequence (default 50)')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--output_dir', default='outputs/tsne_latents')
    p.add_argument('--pretrained_path', default=None,
                   help='Path to dinov2_vits14_pretrain.pth; falls back to torch.hub')
    p.add_argument('--perplexity', type=int, default=40,
                   help='t-SNE perplexity (default 40 for ~500 points)')
    p.add_argument('--n_iter', type=int, default=1000)
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(
        args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device}")

    cfg = Config.fromfile(args.config)

    # ── Build DiffLoc-DINOv2 model ────────────────────────────────────────────
    print("\nBuilding DiffLoc-DINOv2 feature extractor …")
    model = DiffLocDINOv2(in_channels=2, pretrained_path=args.pretrained_path)
    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Total params: {n_params:.1f} M  |  "
          f"ConvStem grid: {DINO_GRID_H}×{DINO_GRID_W} = {DINO_N_PATCHES} patches")

    # ── Extract features for sequences 00-09 ─────────────────────────────────
    all_feats   = []
    all_seq_ids = []

    for seq in range(10):   # 00 – 09
        print(f"\nSeq {seq:02d} …", end=' ', flush=True)
        feats = extract_sequence_features(
            model, cfg, seq, args.n_frames_per_seq, args.batch_size, device
        )
        if feats is None:
            continue
        all_feats.append(feats)
        all_seq_ids.extend([seq] * feats.shape[0])
        print(f"{feats.shape[0]} frames  →  feats {tuple(feats.shape)}")

    if not all_feats:
        print("No sequences loaded — check kitti_sequences_path in config.")
        return

    X    = torch.cat(all_feats, dim=0).numpy()   # [N_total, 384]
    seqs = np.array(all_seq_ids)
    print(f"\nTotal: {X.shape[0]} frames from {len(set(all_seq_ids))} sequences")

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    n_pca = min(50, X.shape[0] - 1, X.shape[1])
    print(f"PCA {X.shape[1]} → {n_pca} dims …")
    X_pca = PCA(n_components=n_pca, random_state=42).fit_transform(X)

    print(f"t-SNE {X_pca.shape} (perplexity={args.perplexity}) …")
    emb = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        random_state=42,
        verbose=1,
    ).fit_transform(X_pca)

    # ── Plot single image ─────────────────────────────────────────────────────
    seq_ids_present = sorted(set(all_seq_ids))
    cmap   = plt.cm.get_cmap('tab10', 10)

    fig, ax = plt.subplots(figsize=(10, 8))
    for sid in seq_ids_present:
        mask = seqs == sid
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c   = [cmap(sid)],
            s   = 12,
            alpha = 0.75,
            linewidths = 0,
            label = f'seq {sid:02d}',
        )

    ax.legend(
        loc='best', fontsize=8, ncol=2,
        markerscale=1.5, framealpha=0.7,
        title='KITTI sequence',
    )
    ax.set_title(
        'DINOv2 CLS-token t-SNE — KITTI sequences 00-09\n'
        '(DiffLoc ConvStem + pretrained ViT-S/14 blocks, 2-channel range+intensity)',
        fontsize=11,
    )
    ax.set_xlabel('t-SNE dim 1'); ax.set_ylabel('t-SNE dim 2')
    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    save_path = os.path.join(args.output_dir, 'dino_tsne_all_sequences.png')
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved → {save_path}")


if __name__ == '__main__':
    main()
