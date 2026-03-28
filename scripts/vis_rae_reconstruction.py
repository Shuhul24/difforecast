#!/usr/bin/env python3
"""
vis_rae_reconstruction.py

Load a Stage 1 RAE checkpoint and visualise GT vs reconstruction for N frames
from a chosen KITTI sequence.

Outputs per frame (saved to --output_dir):
  frame_XXXX_range.png   — 3-panel range channel: GT | Rec | |Error|  (in metres)
  frame_XXXX_all_ch.png  — 5-row × 3-col grid for all 5 channels (normalised)
  metrics.txt            — per-channel MAE / RMSE summary

Usage:
  python scripts/vis_rae_reconstruction.py \\
      --config  configs/rae_config_rangeview.py \\
      --ckpt    outputs/rae-s1/rae_step10000.pkl \\
      --seq     0 \\
      --n_frames 8 \\
      --output_dir outputs/vis_rae
"""

import os, sys, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from utils.config_utils import Config
from utils.bev_utils import render_rangeview_comparison
from models.dino_rae_rangeview import RangeViewRAE
from dataset.dataset_kitti_rangeview import KITTIRangeViewDataset


_CH_NAMES_ALL = ['range', 'x', 'y', 'z', 'intensity']


# ── helpers ──────────────────────────────────────────────────────────────────

def denorm(arr, mean, std, ch_idx=0, log_range=False):
    """Denormalize a numpy array.

    log_range=True  (range channel only, ch_idx=0): 2^(arr*6) - 1  → metres
    log_range=False : arr * std + mean
    """
    arr = arr.astype(np.float32)
    if log_range and ch_idx == 0:
        return np.exp2(arr * 6.0) - 1.0
    return arr * std + mean


def save_all_channels(gt_norm, rec_norm, means, stds, out_path, frame_idx, ch_names, log_range=False):
    """5-row × 3-col grid (GT | Rec | |Error|) for every channel.

    gt_norm / rec_norm : [5, H, W] numpy, normalised.
    """
    n_ch = gt_norm.shape[0]
    fig, axes = plt.subplots(n_ch, 3, figsize=(18, 4 * n_ch))
    for ch in range(n_ch):
        gt  = denorm(gt_norm[ch],  means[ch], stds[ch], ch_idx=ch, log_range=log_range)
        rec = denorm(rec_norm[ch], means[ch], stds[ch], ch_idx=ch, log_range=log_range)
        err = np.abs(rec - gt)

        vmin, vmax = gt.min(), gt.max()
        for col, (img, cmap, title) in enumerate([
            (gt,  'plasma', f'{ch_names[ch]}  GT'),
            (rec, 'plasma', f'{ch_names[ch]}  Rec'),
            (err, 'hot',    f'{ch_names[ch]}  per-pixel L1={err.mean():.4f}'),
        ]):
            ax = axes[ch, col]
            im = ax.imshow(img, cmap=cmap,
                           vmin=(vmin if cmap == 'plasma' else 0),
                           vmax=(vmax if cmap == 'plasma' else err.max() + 1e-6),
                           aspect='auto', interpolation='nearest')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f'Frame {frame_idx} — RAE reconstruction (all channels)', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',     required=True)
    p.add_argument('--ckpt',       required=True,
                   help='Path to rae_stepXXXX.pkl checkpoint')
    p.add_argument('--seq',        type=int, default=0,
                   help='KITTI sequence index (0-9)')
    p.add_argument('--n_frames',   type=int, default=8,
                   help='Number of frames to visualise (evenly spaced)')
    p.add_argument('--output_dir', default='outputs/vis_rae')
    p.add_argument('--device',     default='cuda')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(
        args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    )
    os.makedirs(args.output_dir, exist_ok=True)

    cfg       = Config.fromfile(args.config)
    log_range = bool(getattr(cfg, 'log_range', False))

    # ── Build model and load checkpoint ──────────────────────────────────────
    print(f"Loading RAE from {args.ckpt} …")
    ckpt  = torch.load(args.ckpt, map_location='cpu')
    # Auto-detect channel count from checkpoint so config mismatches don't crash
    ckpt_ch = ckpt['model_state_dict'].get('ch_weights', None)
    if ckpt_ch is not None:
        n_ch_ckpt = ckpt_ch.shape[0]
        if int(getattr(cfg, 'range_channels', 5)) != n_ch_ckpt:
            print(f"  [warn] config range_channels={getattr(cfg, 'range_channels', 5)} "
                  f"but checkpoint has {n_ch_ckpt} channels — overriding.")
            cfg.range_channels = n_ch_ckpt
            cfg.five_channel   = (n_ch_ckpt == 5)
    model = RangeViewRAE(cfg, local_rank=-1)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval().to(device)

    n_ch  = int(getattr(cfg, 'range_channels', 5))
    # Extend means/stds to match the actual channel count (pad with 0/1 if short)
    means_cfg = list(cfg.proj_img_mean)
    stds_cfg  = list(cfg.proj_img_stds)
    means = (means_cfg + [0.0] * n_ch)[:n_ch]
    stds  = (stds_cfg  + [1.0] * n_ch)[:n_ch]
    CH_NAMES = _CH_NAMES_ALL[:n_ch]
    step_ckpt = ckpt.get('step', '?')
    print(f"  Checkpoint step: {step_ckpt}")

    # ── Load dataset ─────────────────────────────────────────────────────────
    ds = KITTIRangeViewDataset(
        sequences_path  = cfg.kitti_sequences_path,
        poses_path      = cfg.kitti_poses_path,
        sequences       = [args.seq],
        condition_frames= 0,
        forward_iter    = 1,
        h=cfg.range_h, w=cfg.range_w,
        fov_up=cfg.fov_up, fov_down=cfg.fov_down,
        fov_left=cfg.fov_left, fov_right=cfg.fov_right,
        proj_img_mean=cfg.proj_img_mean,
        proj_img_stds=cfg.proj_img_stds,
        pc_extension=cfg.pc_extension,
        pc_dtype=getattr(np, cfg.pc_dtype),
        pc_reshape=tuple(cfg.pc_reshape),
        five_channel=getattr(cfg, 'five_channel', False),
        log_range=log_range,
        is_train=False,
    )
    print(f"Seq {args.seq:02d}: {len(ds)} frames total, visualising {args.n_frames}")

    indices = np.linspace(0, len(ds) - 1, min(args.n_frames, len(ds)), dtype=int)

    # ── Per-channel accumulators for summary metrics ──────────────────────────
    all_l1    = np.zeros(n_ch)   # mean per-pixel L1 per channel
    n_frames_done = 0

    with torch.no_grad():
        for frame_idx, ds_idx in enumerate(indices):
            data, _ = ds[int(ds_idx)]           # [T=1, 5, H, W]
            x_norm  = data[0]                   # [5, H, W]  normalised

            x_gpu = x_norm.unsqueeze(0).to(device, dtype=torch.bfloat16)  # [1,5,H,W]
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(x_gpu)

            rec_norm = out['x_rec'][0].float().cpu().numpy()   # [5, H, W]
            gt_norm  = x_norm.float().numpy()                  # [5, H, W]

            # ── Range channel: denormalise + render_rangeview_comparison PNG ──
            gt_depth   = denorm(gt_norm[0],  means[0], stds[0], ch_idx=0, log_range=log_range)
            rec_depth  = denorm(rec_norm[0], means[0], stds[0], ch_idx=0, log_range=log_range)
            range_l1   = float(np.abs(rec_depth - gt_depth).mean())
            render_rangeview_comparison(
                gt_depth=gt_depth,
                pred_depth=rec_depth,
                output_path=os.path.join(
                    args.output_dir, f'frame_{frame_idx:04d}_range.png'),
                frame_idx=int(ds_idx),
                metrics={'range_L1_m': range_l1},
                title_suffix=f' — RAE ckpt step {step_ckpt}  seq {args.seq:02d}',
            )

            # ── All-channel grid PNG ──────────────────────────────────────────
            save_all_channels(
                gt_norm, rec_norm, means, stds,
                out_path=os.path.join(
                    args.output_dir, f'frame_{frame_idx:04d}_all_ch.png'),
                frame_idx=int(ds_idx),
                ch_names=CH_NAMES,
                log_range=log_range,
            )

            # ── Accumulate per-pixel L1 per channel ─────────────────────────
            for ch in range(n_ch):
                gt_ch  = denorm(gt_norm[ch],  means[ch], stds[ch], ch_idx=ch, log_range=log_range)
                rec_ch = denorm(rec_norm[ch], means[ch], stds[ch], ch_idx=ch, log_range=log_range)
                all_l1[ch] += np.abs(rec_ch - gt_ch).mean()
            n_frames_done += 1
            print(f"  frame {frame_idx+1}/{len(indices)}  ds_idx={ds_idx}"
                  f"  range_L1={range_l1:.4f} m")

    # ── Summary ───────────────────────────────────────────────────────────────
    all_l1 /= max(n_frames_done, 1)

    lines = [f"Checkpoint : {args.ckpt}  (step {step_ckpt})",
             f"Sequence   : {args.seq:02d}",
             f"Frames     : {n_frames_done}",
             "",
             f"{'Channel':<12}  {'per-pixel L1':>14}"]
    for ch in range(n_ch):
        unit = ' m' if ch == 0 else ''
        lines.append(f"{CH_NAMES[ch]:<12}  {all_l1[ch]:>12.4f}{unit}")

    summary = '\n'.join(lines)
    print('\n' + summary)
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(summary + '\n')

    print(f"\nSaved {n_frames_done * 2} images + metrics.txt → {args.output_dir}/")
    print(f"Range per-pixel L1 (avg over {n_frames_done} frames): "
          f"{all_l1[0]:.4f} m")


if __name__ == '__main__':
    main()
