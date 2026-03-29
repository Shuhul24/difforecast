#!/usr/bin/env python3
"""
tsne_latents.py

Visualise the latent space of a trained Stage 1 encoder across all KITTI
sequences (00-09) using a loaded checkpoint.

Supported architectures (--arch):
  dino   ConvStem + frozen DINOv2 ViT-S/14  →  [B, 256, 384]  (RangeViewRAE)
  swin   TULIP-inspired Swin encoder         →  [B, 256, 384]  (RangeViewSwinRAE)

Both produce the same latent shape so the dimensionality reduction pipeline
(PCA → t-SNE) and Plotly HTML output are identical.

Mean-pooling is applied over the 256 tokens per frame to obtain a
[B, 384] global descriptor before embedding.

Usage:
    # DINOv2 encoder
    python scripts/tsne_latents.py \\
        --arch dino \\
        --config  configs/rae_config_rangeview.py \\
        --ckpt    outputs/rae-s1/rae_stepXXXX.pkl \\
        --output_dir outputs/tsne_latents

    # Swin encoder
    python scripts/tsne_latents.py \\
        --arch swin \\
        --config  configs/swin_config_rangeview.py \\
        --ckpt    outputs/swin-s1/swin_rae_stepXXXX.pkl \\
        --output_dir outputs/tsne_latents

    # 2-D instead of 3-D t-SNE
    python scripts/tsne_latents.py --arch swin ... --n_components 2
"""

import os
import sys
import argparse
import numpy as np
import torch
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from utils.config_utils import Config
from dataset.dataset_kitti_rangeview import KITTIRangeViewDataset


# ── Model loading helpers ─────────────────────────────────────────────────────

def _load_dino_model(cfg, ckpt_path, device):
    from models.dino_rae_rangeview import RangeViewRAE
    model = RangeViewRAE(cfg, local_rank=-1)
    ckpt  = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval().to(device)
    step = ckpt.get('step', '?')
    return model, step, 'ConvStem + frozen DINOv2 ViT-S/14'


def _load_swin_model(cfg, ckpt_path, device):
    from models.swin_rae_rangeview import RangeViewSwinRAE
    model = RangeViewSwinRAE(cfg, local_rank=-1)
    ckpt  = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval().to(device)
    step = ckpt.get('step', '?')
    return model, step, 'TULIP Swin encoder (hierarchical, circular pad)'


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_sequence_features(
    model,
    cfg,
    seq_id   : int,
    n_frames : int,
    batch_sz : int,
    device   : torch.device,
) -> torch.Tensor | None:
    """Encode one KITTI sequence and return mean-pooled latents [N, 384].

    n_frames <= 0  →  use every frame in the sequence.
    n_frames  > 0  →  evenly subsample to that many frames.
    """
    try:
        ds = KITTIRangeViewDataset(
            sequences_path   = cfg.kitti_sequences_path,
            poses_path       = cfg.kitti_poses_path,
            sequences        = [seq_id],
            condition_frames = 0,
            forward_iter     = 1,
            h                = cfg.range_h,
            w                = cfg.range_w,
            fov_up           = cfg.fov_up,
            fov_down         = cfg.fov_down,
            fov_left         = cfg.fov_left,
            fov_right        = cfg.fov_right,
            proj_img_mean    = cfg.proj_img_mean,
            proj_img_stds    = cfg.proj_img_stds,
            pc_extension     = cfg.pc_extension,
            pc_dtype         = getattr(np, cfg.pc_dtype),
            pc_reshape       = tuple(cfg.pc_reshape),
            five_channel     = getattr(cfg, 'five_channel', False),
            log_range        = getattr(cfg, 'log_range', True),
            is_train         = False,
        )
    except Exception as e:
        print(f"  Seq {seq_id:02d}: skipped ({e})")
        return None

    total = len(ds)
    if total == 0:
        return None

    indices = (np.arange(total) if n_frames <= 0
               else np.linspace(0, total - 1, min(n_frames, total), dtype=int))

    feats = []
    for s in range(0, len(indices), batch_sz):
        batch_idx = indices[s: s + batch_sz]
        imgs = torch.stack([ds[int(i)][0][0] for i in batch_idx])   # [B, C, H, W]
        imgs = imgs.to(device, dtype=torch.float32)
        latents = model.encode(imgs)        # [B, 256, 384]
        feats.append(latents.mean(dim=1).cpu())  # mean over tokens → [B, 384]

    return torch.cat(feats, dim=0)          # [N, 384]


# ── Plotting ──────────────────────────────────────────────────────────────────

TAB10 = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]


def _make_3d_figure(emb, seqs, seq_ids_present, step_ckpt, arch, encoder_desc):
    traces = []
    for sid in seq_ids_present:
        mask  = seqs == sid
        color = TAB10[sid % len(TAB10)]
        fi    = np.where(mask)[0]
        traces.append(go.Scatter3d(
            x=emb[mask, 0], y=emb[mask, 1], z=emb[mask, 2],
            mode='markers', name=f'seq {sid:02d}',
            marker=dict(size=4, color=color, opacity=0.80),
            text=[f'seq {sid:02d} frame {i}' for i in fi],
            hovertemplate='%{text}<extra></extra>',
        ))
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(
                f'[{arch.upper()}] encoder latent space — 3-D t-SNE  (ckpt step {step_ckpt})<br>'
                f'<sup>{encoder_desc}  |  mean-pooled patch tokens [256×384] → [384]  |  KITTI seqs 00-09</sup>'
            ),
            x=0.5,
        ),
        scene=dict(
            xaxis_title='t-SNE dim 1', yaxis_title='t-SNE dim 2', zaxis_title='t-SNE dim 3',
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
        ),
        legend=dict(title='KITTI sequence', itemsizing='constant'),
        margin=dict(l=0, r=0, t=80, b=0),
        width=1100, height=800,
    )
    return fig


def _make_2d_figure(emb, seqs, seq_ids_present, step_ckpt, arch, encoder_desc):
    traces = []
    for sid in seq_ids_present:
        mask  = seqs == sid
        color = TAB10[sid % len(TAB10)]
        fi    = np.where(mask)[0]
        traces.append(go.Scatter(
            x=emb[mask, 0], y=emb[mask, 1],
            mode='markers', name=f'seq {sid:02d}',
            marker=dict(size=5, color=color, opacity=0.80),
            text=[f'seq {sid:02d} frame {i}' for i in fi],
            hovertemplate='%{text}<extra></extra>',
        ))
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(
                f'[{arch.upper()}] encoder latent space — 2-D t-SNE  (ckpt step {step_ckpt})<br>'
                f'<sup>{encoder_desc}  |  mean-pooled patch tokens [256×384] → [384]  |  KITTI seqs 00-09</sup>'
            ),
            x=0.5,
        ),
        xaxis=dict(title='t-SNE dim 1', showticklabels=False),
        yaxis=dict(title='t-SNE dim 2', showticklabels=False),
        legend=dict(title='KITTI sequence', itemsizing='constant'),
        margin=dict(l=40, r=20, t=80, b=40),
        width=1100, height=700,
    )
    return fig


# ── CLI + main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="t-SNE of trained RAE/Swin latents on KITTI sequences 00-09"
    )
    p.add_argument('--arch',    default='dino', choices=['dino', 'swin'],
                   help='Encoder architecture: dino (RangeViewRAE) or swin (RangeViewSwinRAE)')
    p.add_argument('--config',  required=True,
                   help='Config file matching the chosen --arch')
    p.add_argument('--ckpt',
                   default='/DATA2/shuhul/exp/rae_ckpt/rae-s1-lr5e-5/rae_step20000.pkl',
                   help='Stage 1 checkpoint (.pkl)')
    p.add_argument('--n_frames_per_seq', type=int, default=-1,
                   help='Frames per sequence; -1 uses every frame')
    p.add_argument('--batch_size',   type=int, default=8)
    p.add_argument('--output_dir',   default='outputs/tsne_latents')
    p.add_argument('--n_components', type=int, default=3, choices=[2, 3],
                   help='t-SNE output dimensionality: 2 or 3')
    p.add_argument('--perplexity',   type=int, default=-1,
                   help='t-SNE perplexity; -1 auto-sets to max(5, min(sqrt(N)/2, 200))')
    p.add_argument('--n_iter',       type=int, default=1000)
    p.add_argument('--device',       default='cuda')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(
        args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device}  |  arch: {args.arch}  |  n_components: {args.n_components}")

    cfg = Config.fromfile(args.config)

    # Load model
    print(f"\nLoading {args.arch.upper()} model from {args.ckpt} …")
    if args.arch == 'dino':
        model, step_ckpt, encoder_desc = _load_dino_model(cfg, args.ckpt, device)
    else:
        model, step_ckpt, encoder_desc = _load_swin_model(cfg, args.ckpt, device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Checkpoint step : {step_ckpt}")
    print(f"  Trainable params: {trainable:.1f} M")
    print(f"  Encoder        : {encoder_desc}")

    # Extract latents for sequences 00-09
    all_feats   = []
    all_seq_ids = []

    for seq in range(10):
        print(f"\nSeq {seq:02d} …", end=' ', flush=True)
        feats = extract_sequence_features(
            model, cfg, seq, args.n_frames_per_seq, args.batch_size, device
        )
        if feats is None:
            continue
        all_feats.append(feats)
        all_seq_ids.extend([seq] * feats.shape[0])
        print(f"{feats.shape[0]} frames  →  latents {tuple(feats.shape)}")

    if not all_feats:
        print("No sequences loaded — check kitti_sequences_path in config.")
        return

    X    = torch.cat(all_feats, dim=0).numpy()   # [N, 384]
    seqs = np.array(all_seq_ids)
    print(f"\nTotal: {X.shape[0]} frames from {len(set(all_seq_ids))} sequences")

    # PCA → t-SNE
    n_pca = min(100, X.shape[0] - 1, X.shape[1])
    print(f"PCA {X.shape[1]} → {n_pca} dims …")
    X_pca = PCA(n_components=n_pca, random_state=42).fit_transform(X)

    N = X_pca.shape[0]
    perplexity = (args.perplexity if args.perplexity > 0
                  else max(5, min(int(N ** 0.5 / 2), 200)))
    nc = args.n_components
    print(f"t-SNE {X_pca.shape}  (N={N}, perplexity={perplexity}, {nc}D) …")
    emb = TSNE(
        n_components=nc,
        perplexity=perplexity,
        max_iter=args.n_iter,
        random_state=42,
        verbose=1,
    ).fit_transform(X_pca)

    # Build interactive Plotly figure
    seq_ids_present = sorted(set(all_seq_ids))
    if nc == 3:
        fig = _make_3d_figure(emb, seqs, seq_ids_present, step_ckpt, args.arch, encoder_desc)
        suffix = '3d'
    else:
        fig = _make_2d_figure(emb, seqs, seq_ids_present, step_ckpt, args.arch, encoder_desc)
        suffix = '2d'

    save_path = os.path.join(
        args.output_dir, f'{args.arch}_tsne_{suffix}_step{step_ckpt}.html'
    )
    fig.write_html(save_path, include_plotlyjs='cdn')
    print(f"\nSaved → {save_path}  (open in any browser)")


if __name__ == '__main__':
    main()
