"""
Evaluation script for the Stage-1 RangeViewVAE model.

Tests the trained VAE's reconstruction quality by:
  1. Loading a stage-1 checkpoint (``rangeview_vae_<step>.pkl``).
  2. Running encode → decode on KITTI test frames.
  3. Printing and saving per-sample ELBO, Range-view L1, and Chamfer distance.
  4. Saving GT vs Reconstructed range-view images (depth + intensity channels)
     and GT vs Reconstructed BEV point-cloud comparisons.
  5. Writing a metrics CSV and a summary to disk.

Usage
-----
    python scripts/test/test_rangeview_vae.py \\
        --config configs/dit_config_rangeview.py \\
        --resume_path /DATA2/shuhul/exp/ckpt/kitti_rangeview_stage_1/rangeview_vae_<step>.pkl \\
        --exp_name vae_eval \\
        --output_dir eval_results \\
        --num_samples 200

Run with ``--help`` for all options.
"""

import os
import sys
import csv
import time
import argparse

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)

from utils.config_utils import Config
from utils.running import load_parameters
from dataset.dataset_kitti_rangeview import KITTIRangeViewVAEDataset
from dataset.projection import RangeProjection
from models.model_rangeview import RangeViewVAE


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Stage-1 RangeViewVAE reconstruction')
    parser.add_argument('--config',      type=str, required=True)
    parser.add_argument('--resume_path', type=str, required=True,
                        help='Path to rangeview_vae_<step>.pkl checkpoint')
    parser.add_argument('--exp_name',    type=str, required=True)
    parser.add_argument('--output_dir',  type=str, default='eval_results')
    parser.add_argument('--num_samples', type=int, default=200,
                        help='Number of test frames to evaluate (0 = all)')
    parser.add_argument('--save_every',  type=int, default=1,
                        help='Save a visualisation image every N samples')
    parser.add_argument('--sequences',   type=int, nargs='+', default=None,
                        help='Override test sequences (e.g. --sequences 8 9 10)')
    parser.add_argument('--no_cuda',     action='store_true')
    parser.add_argument('--stage',       type=str, default='1', choices=['1'],
                        help='Training stage to evaluate. Only "1" (VAE) is valid here; '
                             'use scripts/test/test_rangeview.py for stage 2.')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.__dict__)
    return cfg


# ---------------------------------------------------------------------------
# Metric helpers (NumPy / CPU — no CUDA kernel dependency)
# ---------------------------------------------------------------------------

def _range_l1(pred_depth: np.ndarray, gt_depth: np.ndarray, min_depth: float = 0.5) -> float:
    valid = gt_depth > min_depth
    if valid.sum() == 0:
        return float('nan')
    return float(np.abs(pred_depth[valid] - gt_depth[valid]).mean())


def _chamfer(pc1: np.ndarray, pc2: np.ndarray, max_pts: int = 4096) -> float:
    """Symmetric Chamfer distance (CPU). Returns nan if either cloud is empty."""
    if pc1.shape[0] < 2 or pc2.shape[0] < 2:
        return float('nan')
    if pc1.shape[0] > max_pts:
        pc1 = pc1[np.random.choice(pc1.shape[0], max_pts, replace=False)]
    if pc2.shape[0] > max_pts:
        pc2 = pc2[np.random.choice(pc2.shape[0], max_pts, replace=False)]
    d2 = ((pc1[:, None] - pc2[None]) ** 2).sum(-1)   # (N, M)
    return float(d2.min(1).mean() + d2.min(0).mean())


def _to_depth(chw: np.ndarray, mean: float, std: float) -> np.ndarray:
    return chw[0] * std + mean


def _backproject(depth_hw: np.ndarray, projector: RangeProjection, min_depth: float = 0.5):
    d = depth_hw.copy()
    d[d < min_depth] = 0.0
    return projector.back_project_range(d)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _save_rv_comparison(gt_chw, recon_chw, out_path, sample_idx, range_mean, range_std,
                        rl1, cd, int_mean=0.0, int_std=1.0):
    """Save a 3-row comparison: depth row, intensity row, error row."""
    gt_depth   = np.clip(_to_depth(gt_chw,    range_mean, range_std), 0, 80)
    pred_depth = np.clip(_to_depth(recon_chw, range_mean, range_std), 0, 80)
    err_depth  = np.abs(pred_depth - gt_depth)

    has_intensity = gt_chw.shape[0] >= 2
    n_rows = 3 if has_intensity else 2     # depth + error [+ intensity]
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 3 * n_rows))

    # Row 0 — depth
    axes[0, 0].imshow(gt_depth,   cmap='plasma', vmin=0, vmax=80, aspect='auto')
    axes[0, 0].set_title('GT depth (m)',   fontsize=9)
    axes[0, 1].imshow(pred_depth, cmap='plasma', vmin=0, vmax=80, aspect='auto')
    axes[0, 1].set_title('Recon depth (m)', fontsize=9)

    # Row 1 — absolute depth error
    im_err = axes[1, 0].imshow(err_depth, cmap='hot', vmin=0, vmax=10, aspect='auto')
    axes[1, 0].set_title('|Error| (m)', fontsize=9)
    plt.colorbar(im_err, ax=axes[1, 0], fraction=0.015, pad=0.02, label='m')
    axes[1, 1].axis('off')

    # Row 2 — intensity (optional)
    if has_intensity:
        gt_int   = np.clip(gt_chw[1]    * int_std + int_mean, 0, 1)
        pred_int = np.clip(recon_chw[1] * int_std + int_mean, 0, 1)
        axes[2, 0].imshow(gt_int,   cmap='gray', aspect='auto', vmin=0, vmax=1)
        axes[2, 0].set_title('GT intensity', fontsize=9)
        axes[2, 1].imshow(pred_int, cmap='gray', aspect='auto', vmin=0, vmax=1)
        axes[2, 1].set_title('Recon intensity', fontsize=9)

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f'Sample {sample_idx} — Range-L1: {rl1:.4f} m  |  Chamfer: {cd:.4f}',
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=90, bbox_inches='tight')
    plt.close(fig)


def _save_bev_comparison(pts_gt, pts_pred, out_path, sample_idx, bev_range, resolution,
                         rl1, cd):
    """Save GT BEV | Pred BEV | Overlay (green=GT, red=Pred)."""
    grid = int(2 * bev_range / resolution)

    def _to_bev(pts):
        img = np.zeros((grid, grid), dtype=np.float32)
        if pts.shape[0] < 2:
            return img
        xi = ((pts[:, 0] + bev_range) / resolution).astype(int)
        yi = ((pts[:, 1] + bev_range) / resolution).astype(int)
        mask = (xi >= 0) & (xi < grid) & (yi >= 0) & (yi < grid)
        img[yi[mask], xi[mask]] = 1.0
        return img

    gt_bev   = _to_bev(pts_gt)
    pred_bev = _to_bev(pts_pred)
    overlay  = np.zeros((grid, grid, 3), dtype=np.float32)
    overlay[..., 1] = np.clip(gt_bev,   0, 1)   # green = GT
    overlay[..., 0] = np.clip(pred_bev, 0, 1)   # red   = Pred

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gt_bev,   cmap='gray', aspect='equal', origin='lower')
    axes[0].set_title('GT BEV',   fontsize=9)
    axes[1].imshow(pred_bev, cmap='gray', aspect='equal', origin='lower')
    axes[1].set_title('Recon BEV', fontsize=9)
    axes[2].imshow(overlay,        aspect='equal', origin='lower')
    axes[2].set_title('Overlay (green=GT, red=Recon)', fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f'Sample {sample_idx} — Range-L1: {rl1:.4f} m  |  Chamfer: {cd:.4f}',
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=90, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(cfg):
    device = torch.device('cpu' if cfg.no_cuda or not torch.cuda.is_available() else 'cuda')
    print(f'Device: {device}')

    out_root = os.path.join(cfg.output_dir, cfg.exp_name)
    rv_dir   = os.path.join(out_root, 'rv_comparisons')
    bev_dir  = os.path.join(out_root, 'bev_comparisons')
    os.makedirs(rv_dir,  exist_ok=True)
    os.makedirs(bev_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Model — load stage-1 RangeViewVAE
    # ------------------------------------------------------------------ #
    print('Building RangeViewVAE …')
    cfg.batch_size = 1    # not used by VAE but kept for compat
    model = RangeViewVAE(cfg, local_rank=0 if device.type == 'cuda' else -1)

    ckpt = torch.load(cfg.resume_path, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)   # handle both formats
    model = load_parameters(model, state)
    model.to(device).eval()
    print(f'Loaded checkpoint: {cfg.resume_path}')

    # ------------------------------------------------------------------ #
    # Dataset — single frames (same as stage-1 training)
    # ------------------------------------------------------------------ #
    test_seqs = cfg.sequences if cfg.sequences else cfg.test_sequences
    print(f'Test sequences: {test_seqs}')

    dataset = KITTIRangeViewVAEDataset(
        sequences_path=cfg.kitti_sequences_path,
        poses_path=cfg.kitti_poses_path,
        sequences=test_seqs,
        h=cfg.range_h,
        w=cfg.range_w,
        fov_up=cfg.fov_up,
        fov_down=cfg.fov_down,
        fov_left=cfg.fov_left,
        fov_right=cfg.fov_right,
        proj_img_mean=cfg.proj_img_mean,
        proj_img_stds=cfg.proj_img_stds,
        pc_extension=cfg.pc_extension,
        pc_dtype=getattr(np, cfg.pc_dtype),
        pc_reshape=tuple(cfg.pc_reshape),
    )

    n_total = len(dataset)
    n_eval  = min(cfg.num_samples, n_total) if cfg.num_samples > 0 else n_total
    from torch.utils.data import Subset, DataLoader
    if n_eval < n_total:
        dataset = Subset(dataset, list(range(n_eval)))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        pin_memory=(device.type == 'cuda'))
    print(f'Evaluating {n_eval} / {n_total} frames …')

    # ------------------------------------------------------------------ #
    # Range projector
    # ------------------------------------------------------------------ #
    projector = RangeProjection(
        fov_up=cfg.fov_up,   fov_down=cfg.fov_down,
        fov_left=cfg.fov_left, fov_right=cfg.fov_right,
        proj_h=cfg.range_h,  proj_w=cfg.range_w,
    )
    range_mean = cfg.proj_img_mean[0]
    range_std  = cfg.proj_img_stds[0]
    int_mean   = cfg.proj_img_mean[1] if len(cfg.proj_img_mean) > 1 else 0.0
    int_std    = cfg.proj_img_stds[1] if len(cfg.proj_img_stds) > 1 else 1.0

    # ------------------------------------------------------------------ #
    # Evaluation loop
    # ------------------------------------------------------------------ #
    records = []
    csv_path = os.path.join(out_root, 'metrics.csv')

    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['sample_idx', 'elbo', 'range_l1', 'chamfer', 'gt_pts', 'recon_pts'])

        t0 = time.time()
        with torch.no_grad():
            for idx, (frame, _) in enumerate(loader):
                # frame: [1, C, H, W]
                frame = frame.to(device)

                # ---------- forward: get ELBO + reconstruction ----------
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=(device.type == 'cuda')):
                    out = model(frame, step=0)

                elbo_val = float(out['loss_elbo'].float().cpu())

                # Reconstruct explicitly for inspection
                # (model.forward already computed x_recon internally; get it cleanly)
                vae_dtype = next(model.vae_tokenizer.vae.parameters()).dtype
                x = frame.to(vae_dtype)
                posterior = model.vae_tokenizer.vae.encode_posterior(x)
                x_recon   = model.vae_tokenizer.vae.decode(posterior.mode())  # deterministic

                # ---------- to numpy ----------
                gt_np    = frame[0].float().cpu().numpy()    # [C, H, W]
                recon_np = x_recon[0].float().cpu().numpy()  # [C, H, W]

                gt_depth   = _to_depth(gt_np,    range_mean, range_std)
                recon_depth = _to_depth(recon_np, range_mean, range_std)

                # ---------- metrics ----------
                rl1 = _range_l1(recon_depth, gt_depth)

                pts_gt    = _backproject(gt_depth,    projector)
                pts_recon = _backproject(recon_depth, projector)
                cd        = _chamfer(pts_recon, pts_gt)

                records.append({'elbo': elbo_val, 'range_l1': rl1, 'chamfer': cd})
                writer.writerow([idx, elbo_val,
                                 rl1  if not np.isnan(rl1) else '',
                                 cd   if not np.isnan(cd)  else '',
                                 pts_gt.shape[0], pts_recon.shape[0]])
                csv_file.flush()

                # ---------- print ----------
                print(
                    f'[{idx+1:>5d}/{n_eval}]  '
                    f'ELBO: {elbo_val:8.4f}  '
                    f'Range-L1: {rl1:.4f} m  '
                    f'Chamfer: {cd:.4f}  '
                    f'({time.time()-t0:.1f}s)'
                )

                # ---------- visualisations ----------
                if idx % cfg.save_every == 0:
                    _save_rv_comparison(
                        gt_chw=gt_np, recon_chw=recon_np,
                        out_path=os.path.join(rv_dir,  f'rv_{idx:05d}.png'),
                        sample_idx=idx,
                        range_mean=range_mean, range_std=range_std,
                        int_mean=int_mean,     int_std=int_std,
                        rl1=rl1, cd=cd,
                    )
                    _save_bev_comparison(
                        pts_gt=pts_gt, pts_pred=pts_recon,
                        out_path=os.path.join(bev_dir, f'bev_{idx:05d}.png'),
                        sample_idx=idx,
                        bev_range=float(getattr(cfg, 'bev_range', 50.0)),
                        resolution=float(getattr(cfg, 'bev_resolution', 0.2)),
                        rl1=rl1, cd=cd,
                    )

    # ------------------------------------------------------------------ #
    # Aggregate summary
    # ------------------------------------------------------------------ #
    valid = [r for r in records
             if not np.isnan(r['elbo']) and not np.isnan(r['range_l1'])]

    if valid:
        mean_elbo = np.mean([r['elbo']     for r in valid])
        mean_rl1  = np.mean([r['range_l1'] for r in valid])
        mean_cd   = np.nanmean([r['chamfer'] for r in valid])

        print('\n' + '=' * 60)
        print(f'Stage-1 VAE evaluation — {len(valid)} frames')
        print(f'  Mean ELBO         : {mean_elbo:.4f}')
        print(f'  Mean Range L1     : {mean_rl1:.4f} m')
        print(f'  Mean Chamfer Dist : {mean_cd:.4f}')
        print(f'  RV images   → {rv_dir}')
        print(f'  BEV images  → {bev_dir}')
        print(f'  Metrics CSV → {csv_path}')
        print('=' * 60)

        # Per-sample plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, key, label in zip(
            axes,
            ['elbo',     'range_l1',    'chamfer'],
            ['ELBO',     'Range L1 (m)', 'Chamfer dist'],
        ):
            vals = [r[key] for r in valid if not np.isnan(r[key])]
            ax.plot(vals, linewidth=0.8)
            ax.axhline(np.mean(vals), color='r', linestyle='--', linewidth=1,
                       label=f'mean={np.mean(vals):.4f}')
            ax.set_title(label, fontsize=10)
            ax.set_xlabel('sample'); ax.legend(fontsize=8)
        fig.suptitle(f'Stage-1 VAE metrics — {cfg.exp_name}', fontsize=11)
        plt.tight_layout()
        plot_path = os.path.join(out_root, 'metrics_plot.png')
        plt.savefig(plot_path, dpi=90, bbox_inches='tight')
        plt.close(fig)
        print(f'  Metrics plot → {plot_path}')

        summary_path = os.path.join(out_root, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f'Experiment  : {cfg.exp_name}\n')
            f.write(f'Checkpoint  : {cfg.resume_path}\n')
            f.write(f'Sequences   : {test_seqs}\n')
            f.write(f'Samples     : {len(valid)}\n')
            f.write(f'Mean ELBO        : {mean_elbo:.6f}\n')
            f.write(f'Mean Range L1    : {mean_rl1:.6f} m\n')
            f.write(f'Mean Chamfer     : {mean_cd:.6f}\n')
        print(f'  Summary txt → {summary_path}')
    else:
        print('No valid frames — check paths and checkpoint.')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    os.chdir(root_path)
    cfg = parse_args()
    evaluate(cfg)
