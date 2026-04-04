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
import matplotlib.cm as cm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Infer input channel count from the patch-embed conv weight in the checkpoint
    # so the model is built correctly regardless of what the config says.
    ckpt_n_ch = ckpt['model_state_dict']['encoder.patch_embed.proj.weight'].shape[1]
    cfg_n_ch  = 5 if getattr(cfg, 'five_channel', False) else getattr(cfg, 'range_channels', 1)
    if ckpt_n_ch != cfg_n_ch:
        print(f"  [info] checkpoint uses {ckpt_n_ch} input channels "
              f"(config says {cfg_n_ch}) — patching cfg")
        cfg.range_channels = ckpt_n_ch
        cfg.five_channel   = False          # five_channel is a separate flag
        # ch_weights must match the channel count
        cfg.rae_ch_weights = list(
            ckpt['model_state_dict']['ch_weights'].cpu().float().numpy()
        )

    model = RangeViewSwinRAE(cfg, local_rank=-1)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval().to(device)
    step = ckpt.get('step', '?')
    return model, step, 'TULIP Swin encoder (hierarchical, circular pad)'


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_sequence_features(
    model,
    cfg,
    seq_id        : int,
    n_frames      : int,
    batch_sz      : int,
    device        : torch.device,
    return_images : bool = False,
):
    """Encode one KITTI sequence and return mean-pooled latents [N, D].

    n_frames <= 0  →  use every frame in the sequence.
    n_frames  > 0  →  evenly subsample to that many frames.

    Returns:
        feats  : Tensor [N, D] of mean-pooled encoder tokens, or None on failure.
        images : list of N cpu Tensors [C, H, W] (raw, un-normalised) when
                 return_images=True, else None.
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
        return None, None

    total = len(ds)
    if total == 0:
        return None, None

    indices = (np.arange(total) if n_frames <= 0
               else np.linspace(0, total - 1, min(n_frames, total), dtype=int))

    feats     = []
    imgs_out  = [] if return_images else None

    for s in range(0, len(indices), batch_sz):
        batch_idx = indices[s: s + batch_sz]
        raw = torch.stack([ds[int(i)][0][0] for i in batch_idx])   # [B, C, H, W]
        if return_images:
            imgs_out.extend(raw.unbind(0))          # list of cpu [C, H, W] tensors
        imgs = raw.to(device, dtype=torch.float32)
        z, _ = model.encode(imgs)                   # z: [B, tokens, D]
        feats.append(z.mean(dim=1).cpu())           # mean over tokens → [B, D]

    return torch.cat(feats, dim=0), imgs_out        # ([N, D], list|None)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _to_rgb_image(img_tensor: torch.Tensor, max_w: int = 640) -> np.ndarray:
    """Convert a [C,H,W] range-view tensor → uint8 RGB [H,W,3] via viridis.

    Uses the range channel (index 0).  Widths above max_w are stride-subsampled
    so the embedded HTML stays compact.
    """
    ch = img_tensor[0].cpu().float().numpy()   # [H, W]
    h, w = ch.shape
    if w > max_w:
        step = max(1, w // max_w)
        ch = ch[:, ::step]
    vmin, vmax = ch.min(), ch.max()
    ch_n = (ch - vmin) / (vmax - vmin + 1e-8)
    return (cm.viridis(ch_n)[:, :, :3] * 255).astype(np.uint8)   # [H,W,3]


TAB10 = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]


def _make_3d_figure(emb, seqs, seq_ids_present, step_ckpt, arch, encoder_desc):
    traces = []
    # per-sequence local scan counters so hover shows the actual .bin index
    seq_local = {sid: 0 for sid in seq_ids_present}
    for i, sid in enumerate(seqs):
        sid = int(sid)
        if sid not in seq_local:
            seq_local[sid] = 0
    # rebuild with correct local indices
    seq_scan_idx = {}   # global_idx → local scan index within its sequence
    counters = {sid: 0 for sid in seq_ids_present}
    for gi in range(len(seqs)):
        sid = int(seqs[gi])
        seq_scan_idx[gi] = counters[sid]
        counters[sid] += 1

    for sid in seq_ids_present:
        mask  = seqs == sid
        color = TAB10[sid % len(TAB10)]
        gi    = np.where(mask)[0]
        local = [seq_scan_idx[g] for g in gi]
        traces.append(go.Scatter3d(
            x=emb[mask, 0], y=emb[mask, 1], z=emb[mask, 2],
            mode='markers', name=f'seq {sid:02d}',
            marker=dict(size=4, color=color, opacity=0.80),
            text=[f'seq {sid:02d}  scan {l:06d}' for l in local],
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
    counters = {sid: 0 for sid in seq_ids_present}
    seq_scan_idx = {}
    for gi in range(len(seqs)):
        sid = int(seqs[gi])
        seq_scan_idx[gi] = counters[sid]
        counters[sid] += 1

    for sid in seq_ids_present:
        mask  = seqs == sid
        color = TAB10[sid % len(TAB10)]
        gi    = np.where(mask)[0]
        local = [seq_scan_idx[g] for g in gi]
        traces.append(go.Scatter(
            x=emb[mask, 0], y=emb[mask, 1],
            mode='markers', name=f'seq {sid:02d}',
            marker=dict(size=5, color=color, opacity=0.80),
            text=[f'seq {sid:02d}  scan {l:06d}' for l in local],
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


# ── Animated 3-D figure (scan-by-scan reveal for one sequence) ────────────────

def _make_animated_3d_figure(
    emb, seqs, seq_ids_present, animate_seq, step_ckpt, arch, encoder_desc,
    images=None,
):
    """3-D t-SNE with a play/slider animation that reveals one sequence scan-by-scan.

    All other sequences are shown as static faded background markers.
    The animated sequence grows point-by-point (cumulative reveal).

    When *images* is provided (a list of [C,H,W] cpu tensors in the same order
    as the animated sequence's rows in *emb*) the figure gains a second panel
    that shows the corresponding range-view image for the most recently revealed
    scan, so the user can see which real-world scene is being projected into the
    latent space.
    """
    # build per-sequence local scan index (same logic as static figures)
    counters = {sid: 0 for sid in seq_ids_present}
    seq_scan_idx = {}
    for gi in range(len(seqs)):
        sid = int(seqs[gi])
        seq_scan_idx[gi] = counters[sid]
        counters[sid] += 1

    anim_mask = seqs == animate_seq
    anim_gi   = np.where(anim_mask)[0]
    n_anim    = len(anim_gi)

    # Static background traces (all sequences except the animated one)
    bg_traces = []
    for sid in seq_ids_present:
        if sid == animate_seq:
            continue
        mask  = seqs == sid
        gi    = np.where(mask)[0]
        local = [seq_scan_idx[g] for g in gi]
        bg_traces.append(go.Scatter3d(
            x=emb[mask, 0], y=emb[mask, 1], z=emb[mask, 2],
            mode='markers', name=f'seq {sid:02d}',
            marker=dict(size=3, color=TAB10[sid % len(TAB10)], opacity=0.25),
            text=[f'seq {sid:02d}  scan {l:06d}' for l in local],
            hovertemplate='%{text}<extra></extra>',
        ))

    anim_color = TAB10[animate_seq % len(TAB10)]
    anim_local = [seq_scan_idx[g] for g in anim_gi]
    n_bg       = len(bg_traces)
    has_images = images is not None and len(images) > 0

    # ── trace builders ────────────────────────────────────────────────────────
    def _anim_scatter(n_shown):
        idx         = anim_gi[:n_shown]
        local_shown = anim_local[:n_shown]
        return go.Scatter3d(
            x=emb[idx, 0], y=emb[idx, 1], z=emb[idx, 2],
            mode='markers', name=f'seq {animate_seq:02d} (animated)',
            marker=dict(size=5, color=anim_color, opacity=0.90),
            text=[f'seq {animate_seq:02d}  scan {l:06d}' for l in local_shown],
            hovertemplate='%{text}<extra></extra>',
            scene='scene',
        )

    def _range_img(n_shown):
        """go.Image trace for the range-view panel (subplot col 2)."""
        if n_shown == 0 or not has_images:
            h, w = images[0].shape[-2], images[0].shape[-1] if has_images else (64, 512)
            rgb  = np.zeros((h, min(w, 640), 3), dtype=np.uint8)
            label = 'Range view'
        else:
            rgb   = _to_rgb_image(images[n_shown - 1])
            label = f'seq {animate_seq:02d}  scan {anim_local[n_shown - 1]:06d}'
        return go.Image(
            z=rgb,
            name=label,
            showlegend=False,
            xaxis='x2',
            yaxis='y2',
            hovertemplate='%{text}<extra></extra>',
            text=[[label] * rgb.shape[1]] * rgb.shape[0],
        )

    # ── frame stride (max 500 frames to keep HTML compact) ───────────────────
    stride       = max(1, n_anim // 500)
    frame_counts = list(range(0, n_anim + 1, stride))
    if frame_counts[-1] != n_anim:
        frame_counts.append(n_anim)

    # ── assemble figure ───────────────────────────────────────────────────────
    if has_images:
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'xy'}]],
            column_widths=[0.58, 0.42],
            subplot_titles=['Latent Space (3-D t-SNE)', 'Range View (depth channel)'],
            horizontal_spacing=0.04,
        )
        for t in bg_traces:
            fig.add_trace(t, row=1, col=1)
        fig.add_trace(_anim_scatter(0), row=1, col=1)
        fig.add_trace(_range_img(0), row=1, col=2)

        # Only the two mutable traces are updated each frame
        plotly_frames = [
            go.Frame(
                data=[_anim_scatter(k), _range_img(k)],
                traces=[n_bg, n_bg + 1],
                name=str(k),
            )
            for k in frame_counts
        ]
        fig_width = 1600
    else:
        fig = go.Figure(data=bg_traces + [_anim_scatter(0)])
        plotly_frames = [
            go.Frame(
                data=bg_traces + [_anim_scatter(k)],
                name=str(k),
            )
            for k in frame_counts
        ]
        fig_width = 1100

    fig.frames = plotly_frames

    # ── layout ────────────────────────────────────────────────────────────────
    sliders = [dict(
        active=0,
        steps=[dict(
            args=[[str(k)], dict(frame=dict(duration=0, redraw=True), mode='immediate')],
            label=str(frame_counts[i]),
            method='animate',
        ) for i, k in enumerate(frame_counts)],
        currentvalue=dict(prefix=f'seq {animate_seq:02d} scans shown: ', font=dict(size=13)),
        len=0.9, x=0.05, y=0.02,
    )]

    layout_kw = dict(
        title=dict(
            text=(
                f'[{arch.upper()}] latent space — animated seq {animate_seq:02d}  '
                f'(ckpt step {step_ckpt})<br>'
                f'<sup>{encoder_desc}</sup>'
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
        margin=dict(l=0, r=0, t=100, b=80),
        width=fig_width, height=850,
        updatemenus=[dict(
            type='buttons', showactive=False, y=0.06, x=0.02, xanchor='left',
            buttons=[
                dict(label='▶ Play',
                     method='animate',
                     args=[None, dict(frame=dict(duration=80, redraw=True),
                                      fromcurrent=True, mode='immediate')]),
                dict(label='⏸ Pause',
                     method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode='immediate')]),
            ],
        )],
        sliders=sliders,
    )
    if has_images:
        # y increases downward for image display; hide tick labels
        layout_kw['yaxis2'] = dict(autorange='reversed', showticklabels=False,
                                   showgrid=False, zeroline=False)
        layout_kw['xaxis2'] = dict(showticklabels=False,
                                   showgrid=False, zeroline=False)

    fig.update_layout(**layout_kw)
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
    p.add_argument('--animate_seq',  type=int, default=-1,
                   help='If >= 0, also save an animated HTML revealing this sequence scan-by-scan')
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
        feats, _ = extract_sequence_features(
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

    if args.animate_seq >= 0:
        if args.animate_seq not in seq_ids_present:
            print(f"Warning: seq {args.animate_seq} not in loaded sequences — skipping animation.")
        elif nc != 3:
            print("Animation is only supported for 3-D t-SNE (--n_components 3) — skipping.")
        else:
            print(f"Loading range-view images for seq {args.animate_seq:02d} …")
            _, images_anim = extract_sequence_features(
                model, cfg, args.animate_seq,
                args.n_frames_per_seq, args.batch_size, device,
                return_images=True,
            )
            print(f"  {len(images_anim)} images loaded  "
                  f"(shape {tuple(images_anim[0].shape)})")

            print(f"Building animated HTML for seq {args.animate_seq:02d} …")
            fig_anim = _make_animated_3d_figure(
                emb, seqs, seq_ids_present,
                animate_seq=args.animate_seq,
                step_ckpt=step_ckpt, arch=args.arch, encoder_desc=encoder_desc,
                images=images_anim,
            )
            anim_path = os.path.join(
                args.output_dir,
                f'{args.arch}_tsne_3d_step{step_ckpt}_seq{args.animate_seq:02d}_animated.html',
            )
            fig_anim.write_html(anim_path, include_plotlyjs='cdn')
            print(f"Saved → {anim_path}")


if __name__ == '__main__':
    main()
