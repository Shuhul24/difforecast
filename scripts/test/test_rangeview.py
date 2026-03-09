"""
Evaluation script for the Range View DiT model.

Inspired by the ATPPNet evaluation and visualisation workflow
(https://github.com/kaustabpal/ATPPNet), this script:

  1. Loads a trained ``RangeViewDiT`` checkpoint.
  2. Runs inference on the KITTI test sequences to predict the next range-view
     frame given a window of conditioning frames.
  3. Back-projects both the predicted and ground-truth range images to 3-D
     point clouds using ``RangeProjection.back_project_range``.
  4. Saves Bird's Eye View (BEV) comparison images (GT vs. predicted) as PNGs
     — no display / GLFW required, safe for HPC servers.
  5. Computes per-frame metrics (range-view L1, Chamfer distance) and saves a
     summary CSV and a metrics-over-time plot.

Usage
-----
Single GPU (typical on HPC)::

    python scripts/test/test_rangeview.py \\
        --config configs/dit_config_rangeview.py \\
        --resume_path exp/ckpt/<exp_name>/rangeview_dit_<step>.pkl \\
        --exp_name my_eval \\
        --output_dir eval_results \\
        --num_samples 200 \\
        --bev_range 50

Run ``python scripts/test/test_rangeview.py --help`` for all options.
"""

import os
import sys
import csv
import time
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Make root importable from scripts/test/
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)

from utils.config_utils import Config
from utils.preprocess import get_rel_pose
from utils.bev_utils import (
    rangeview_to_pointcloud,
    render_bev_comparison,
    render_rangeview_comparison,
    plot_metrics_summary,
)
from dataset.dataset_kitti_rangeview import KITTIRangeViewTestDataset
from dataset.projection import RangeProjection
from models.model_rangeview import RangeViewDiT


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate RangeViewDiT: predict point clouds and compare BEV'
    )
    parser.add_argument('--config',      type=str, required=True,
                        help='Path to config file (e.g. configs/dit_config_rangeview.py)')
    parser.add_argument('--resume_path', type=str, default=None,
                        help='Path to model checkpoint (.pkl)')
    parser.add_argument('--exp_name',    type=str, required=True,
                        help='Experiment name used for output subdirectory')
    parser.add_argument('--output_dir',  type=str, default='eval_results',
                        help='Root directory for saving evaluation outputs')
    parser.add_argument('--num_samples', type=int, default=200,
                        help='Maximum number of test samples to evaluate (0 = all)')
    parser.add_argument('--bev_range',   type=float, default=50.0,
                        help='BEV half-extent in metres (default: 50 m)')
    parser.add_argument('--bev_resolution', type=float, default=0.2,
                        help='BEV metres per pixel (default: 0.2)')
    parser.add_argument('--save_every',  type=int, default=1,
                        help='Save a BEV image every N samples (1 = all)')
    parser.add_argument('--sequences',   type=int, nargs='+', default=None,
                        help='Override test sequences from config (e.g. --sequences 8 9 10)')
    parser.add_argument('--no_cuda',     action='store_true',
                        help='Run on CPU (slow; mainly for debugging)')

    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.__dict__)
    return cfg


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _chamfer_distance_np(pc1: np.ndarray, pc2: np.ndarray, max_pts: int = 4096) -> float:
    """Symmetric Chamfer distance between two point clouds (NumPy, CPU).

    Returns ``float('nan')`` if either cloud is empty.
    """
    if pc1.shape[0] < 2 or pc2.shape[0] < 2:
        return float('nan')

    if pc1.shape[0] > max_pts:
        idx = np.random.choice(pc1.shape[0], max_pts, replace=False)
        pc1 = pc1[idx]
    if pc2.shape[0] > max_pts:
        idx = np.random.choice(pc2.shape[0], max_pts, replace=False)
        pc2 = pc2[idx]

    diff  = pc1[:, None, :] - pc2[None, :, :]   # (N, M, 3)
    dist2 = (diff ** 2).sum(axis=-1)             # (N, M)
    d1 = dist2.min(axis=1).mean()
    d2 = dist2.min(axis=0).mean()
    return float(d1 + d2)


def _range_l1_np(pred_depth: np.ndarray, gt_depth: np.ndarray, min_depth: float = 0.5) -> float:
    """Mean L1 on the range (depth) channel at GT-valid pixels."""
    valid = gt_depth > min_depth
    if valid.sum() == 0:
        return float('nan')
    return float(np.abs(pred_depth[valid] - gt_depth[valid]).mean())


# ---------------------------------------------------------------------------
# Back-projection helper
# ---------------------------------------------------------------------------

def _feature_to_depth(feature_hw: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Unnormalise a ``(H, W)`` range channel to metres."""
    return feature_hw * std + mean


def _backproject(depth_hw: np.ndarray, projector: RangeProjection, min_depth: float = 0.5):
    """Back-project a depth map to a ``(N, 3)`` point cloud."""
    depth_copy = depth_hw.copy()
    depth_copy[depth_copy < min_depth] = 0.0
    return projector.back_project_range(depth_copy)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(cfg):
    device = torch.device('cpu' if cfg.no_cuda or not torch.cuda.is_available() else 'cuda')
    print(f'Using device: {device}')

    # ------------------------------------------------------------------ #
    # Output directories
    # ------------------------------------------------------------------ #
    output_root = os.path.join(cfg.output_dir, cfg.exp_name)
    bev_dir     = os.path.join(output_root, 'bev_comparisons')
    rv_dir      = os.path.join(output_root, 'rangeview_comparisons')
    os.makedirs(bev_dir, exist_ok=True)
    os.makedirs(rv_dir,  exist_ok=True)

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    print('Building model …')
    # batch_size attr required by RangeViewDiT for img_ids buffer sizing;
    # at inference we run batch_size=1.
    cfg.batch_size = 1

    model = RangeViewDiT(
        cfg,
        local_rank=0,
        load_path=cfg.resume_path,
        condition_frames=cfg.condition_frames // cfg.block_size,
    )
    model.to(device)
    model.eval()
    print(f'Model loaded from: {cfg.resume_path}')

    # ------------------------------------------------------------------ #
    # Dataset
    # ------------------------------------------------------------------ #
    test_seqs = cfg.sequences if cfg.sequences else cfg.test_sequences
    print(f'Test sequences: {test_seqs}')

    test_dataset = KITTIRangeViewTestDataset(
        sequences_path=cfg.kitti_sequences_path,
        poses_path=cfg.kitti_poses_path,
        sequences=test_seqs,
        condition_frames=cfg.condition_frames,
        forward_iter=1,               # we predict only 1 frame at a time
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

    n_total = len(test_dataset)
    if cfg.num_samples > 0:
        n_eval = min(cfg.num_samples, n_total)
        test_dataset = Subset(test_dataset, list(range(n_eval)))
    else:
        n_eval = n_total

    print(f'Evaluating {n_eval} / {n_total} samples …')

    loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                        num_workers=0, pin_memory=(device.type == 'cuda'))

    # ------------------------------------------------------------------ #
    # Range projector (for back-projection)
    # ------------------------------------------------------------------ #
    projector = RangeProjection(
        fov_up=cfg.fov_up,
        fov_down=cfg.fov_down,
        fov_left=cfg.fov_left,
        fov_right=cfg.fov_right,
        proj_h=cfg.range_h,
        proj_w=cfg.range_w,
    )

    range_mean = cfg.proj_img_mean[0]
    range_std  = cfg.proj_img_stds[0]
    cf = cfg.condition_frames // cfg.block_size

    # ------------------------------------------------------------------ #
    # Evaluation loop
    # ------------------------------------------------------------------ #
    metrics_per_frame = []
    csv_path = os.path.join(output_root, 'metrics.csv')

    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['frame_idx', 'range_l1', 'chamfer_dist',
                         'gt_pts', 'pred_pts'])

        t0 = time.time()
        with torch.no_grad():
            for sample_idx, (range_views, poses) in enumerate(loader):
                # range_views: [1, CF+1, 3, H, W]
                # poses:       [1, CF+1, 4, 4]
                range_views = range_views.to(device)
                poses       = poses.to(device)

                # Split conditioning / target
                # conditioning: frames 0 … CF-1
                # GT target:    frame CF
                features_cond = range_views[:, :cf]          # [1, CF, 3, H, W]
                gt_frame      = range_views[:, cf]           # [1, 3, H, W]

                # Relative pose for CF conditioning steps (need CF+1 absolute poses)
                poses_for_rel  = poses[:, :cf + 1]           # [1, CF+1, 4, 4]
                rel_pose, rel_yaw = get_rel_pose(poses_for_rel)
                # rel_pose: [1, CF, 2],  rel_yaw: [1, CF, 1]

                # -------------------------------------------------- #
                # Model inference
                # -------------------------------------------------- #
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=(device.type == 'cuda')):
                    pred_frame = model.step_eval(features_cond, rel_pose, rel_yaw)
                # pred_frame: [1, 3, H, W]  (normalised feature space)

                # -------------------------------------------------- #
                # Move to CPU / NumPy for metric and visualisation
                # -------------------------------------------------- #
                pred_np = pred_frame[0].float().cpu().numpy()    # [3, H, W]
                gt_np   = gt_frame[0].float().cpu().numpy()      # [3, H, W]

                pred_depth = _feature_to_depth(pred_np[0], range_mean, range_std)  # (H, W)
                gt_depth   = _feature_to_depth(gt_np[0],   range_mean, range_std)  # (H, W)

                # -------------------------------------------------- #
                # Back-project to 3-D point clouds
                # -------------------------------------------------- #
                pts_pred = _backproject(pred_depth, projector)   # (N, 3)
                pts_gt   = _backproject(gt_depth,   projector)   # (M, 3)

                # -------------------------------------------------- #
                # Metrics
                # -------------------------------------------------- #
                rl1 = _range_l1_np(pred_depth, gt_depth)
                cd  = _chamfer_distance_np(pts_pred, pts_gt)

                frame_metrics = {
                    'range_l1':     rl1 if not np.isnan(rl1) else 0.0,
                    'chamfer_dist': cd  if not np.isnan(cd)  else 0.0,
                }
                metrics_per_frame.append(frame_metrics)
                writer.writerow([sample_idx, rl1, cd,
                                 pts_gt.shape[0], pts_pred.shape[0]])
                csv_file.flush()

                # -------------------------------------------------- #
                # BEV visualisation
                # -------------------------------------------------- #
                if sample_idx % cfg.save_every == 0:
                    bev_path = os.path.join(bev_dir, f'bev_{sample_idx:05d}.png')
                    render_bev_comparison(
                        points_gt=pts_gt,
                        points_pred=pts_pred,
                        bev_range=cfg.bev_range,
                        resolution=cfg.bev_resolution,
                        output_path=bev_path,
                        frame_idx=sample_idx,
                        metrics=frame_metrics,
                    )

                    rv_path = os.path.join(rv_dir, f'rv_{sample_idx:05d}.png')
                    render_rangeview_comparison(
                        gt_depth=gt_depth,
                        pred_depth=pred_depth,
                        output_path=rv_path,
                        frame_idx=sample_idx,
                        metrics=frame_metrics,
                    )

                # -------------------------------------------------- #
                # Progress logging
                # -------------------------------------------------- #
                if (sample_idx + 1) % 10 == 0 or sample_idx == n_eval - 1:
                    elapsed = time.time() - t0
                    print(
                        f'[{sample_idx + 1:>5d}/{n_eval}] '
                        f'range_l1={rl1:.4f}  chamfer={cd:.4f}  '
                        f'gt_pts={pts_gt.shape[0]:>6d}  pred_pts={pts_pred.shape[0]:>6d}  '
                        f'({elapsed:.1f}s elapsed)'
                    )

    # ------------------------------------------------------------------ #
    # Aggregate summary
    # ------------------------------------------------------------------ #
    valid_frames = [m for m in metrics_per_frame
                    if not np.isnan(m['range_l1']) and not np.isnan(m['chamfer_dist'])]

    if valid_frames:
        mean_rl1 = np.mean([m['range_l1']     for m in valid_frames])
        mean_cd  = np.mean([m['chamfer_dist']  for m in valid_frames])
        print('\n' + '=' * 60)
        print(f'Evaluation complete — {len(valid_frames)} valid frames')
        print(f'  Mean Range L1       : {mean_rl1:.4f}')
        print(f'  Mean Chamfer Dist   : {mean_cd:.4f}')
        print(f'  BEV images saved to : {bev_dir}')
        print(f'  RV images saved to  : {rv_dir}')
        print(f'  Metrics CSV         : {csv_path}')
        print('=' * 60)

        # Per-frame metric plot
        metrics_plot_path = os.path.join(output_root, 'metrics_summary.png')
        plot_metrics_summary(
            metrics_per_frame=valid_frames,
            output_path=metrics_plot_path,
            title=f'Per-frame Metrics — {cfg.exp_name}',
        )
        print(f'  Metrics plot saved  : {metrics_plot_path}')

        # Summary text
        summary_path = os.path.join(output_root, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f'Experiment : {cfg.exp_name}\n')
            f.write(f'Checkpoint : {cfg.resume_path}\n')
            f.write(f'Sequences  : {test_seqs}\n')
            f.write(f'Samples    : {len(valid_frames)}\n')
            f.write(f'Mean Range L1     : {mean_rl1:.6f}\n')
            f.write(f'Mean Chamfer Dist : {mean_cd:.6f}\n')
        print(f'  Summary text        : {summary_path}')
    else:
        print('No valid frames found — check data paths and checkpoint.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    os.chdir(root_path)
    cfg = parse_args()
    evaluate(cfg)
