"""
Evaluation script for the Range View DiT model.

Evaluates the full model (stage 1 VAE + stage 2 DiT) on KITTI test sequences.
For each sample, conditions on ``condition_frames`` GT frames and autoregressively
predicts ``forward_iter`` future frames (teacher-forced: GT frames used as conditioning
at every step).

Outputs
-------
* ``metrics.csv``              — per-step metrics (range L1, Chamfer) for every sample.
* ``rangeview_sequence.png``   — 2×5 grid of GT (top) vs predicted (bottom) range images
                                 annotated with the average range L1 over the 5 steps.
* ``bev_sequence.png``         — 2×5 grid of GT (top) vs predicted (bottom) BEV images
                                 annotated with the average Chamfer distance over the 5 steps.
  Both images are produced for the sample selected by ``--viz_sample_idx`` (default: 0).

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
    render_rangeview_sequence,
    render_bev_sequence,
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
        description='Evaluate RangeViewDiT: 5-step prediction with range-view and BEV outputs'
    )
    parser.add_argument('--config',          type=str, required=True,
                        help='Path to config file (e.g. configs/dit_config_rangeview.py)')
    parser.add_argument('--resume_path',     type=str, default=None,
                        help='Path to model checkpoint (.pkl)')
    parser.add_argument('--exp_name',        type=str, required=True,
                        help='Experiment name used for output subdirectory')
    parser.add_argument('--output_dir',      type=str, default='eval_results',
                        help='Root directory for saving evaluation outputs')
    parser.add_argument('--num_samples',     type=int, default=200,
                        help='Maximum number of test samples to evaluate (0 = all)')
    parser.add_argument('--bev_range',       type=float, default=50.0,
                        help='BEV half-extent in metres (default: 50 m)')
    parser.add_argument('--bev_resolution',  type=float, default=0.2,
                        help='BEV metres per pixel (default: 0.2)')
    parser.add_argument('--sequences',       type=int, nargs='+', default=None,
                        help='Override test sequences from config (e.g. --sequences 8 9 10)')
    parser.add_argument('--viz_sample_idx',  type=int, default=0,
                        help='Which evaluated sample to use for the two summary images (default: 0)')
    parser.add_argument('--no_cuda',         action='store_true',
                        help='Run on CPU (slow; mainly for debugging)')

    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.__dict__)
    return cfg


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _chamfer_distance_np(pc1: np.ndarray, pc2: np.ndarray, max_pts: int = 4096) -> float:
    """Symmetric Chamfer distance between two point clouds (NumPy, CPU)."""
    if pc1.shape[0] < 2 or pc2.shape[0] < 2:
        return float('nan')

    if pc1.shape[0] > max_pts:
        idx = np.random.choice(pc1.shape[0], max_pts, replace=False)
        pc1 = pc1[idx]
    if pc2.shape[0] > max_pts:
        idx = np.random.choice(pc2.shape[0], max_pts, replace=False)
        pc2 = pc2[idx]

    diff  = pc1[:, None, :] - pc2[None, :, :]
    dist2 = (diff ** 2).sum(axis=-1)
    d1 = dist2.min(axis=1).mean()
    d2 = dist2.min(axis=0).mean()
    return float(d1 + d2)


def _range_l1_np(pred_depth: np.ndarray, gt_depth: np.ndarray, min_depth: float = 0.5) -> float:
    """Mean L1 on the range channel at GT-valid pixels."""
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

    forward_iter = getattr(cfg, 'forward_iter', 5)
    cf           = cfg.condition_frames // cfg.block_size

    # ------------------------------------------------------------------ #
    # Output directory
    # ------------------------------------------------------------------ #
    output_root = os.path.join(cfg.output_dir, cfg.exp_name)
    os.makedirs(output_root, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    print('Building model …')
    cfg.batch_size = 1

    model = RangeViewDiT(
        cfg,
        local_rank=0,
        load_path=cfg.resume_path,
        condition_frames=cf,
    )
    model.to(device)
    model.eval()
    print(f'Model loaded from: {cfg.resume_path}')

    # ------------------------------------------------------------------ #
    # Dataset  (need condition_frames + forward_iter frames per sample)
    # ------------------------------------------------------------------ #
    test_seqs = cfg.sequences if cfg.sequences else cfg.test_sequences
    print(f'Test sequences: {test_seqs}')

    test_dataset = KITTIRangeViewTestDataset(
        sequences_path=cfg.kitti_sequences_path,
        poses_path=cfg.kitti_poses_path,
        sequences=test_seqs,
        condition_frames=cfg.condition_frames,
        forward_iter=forward_iter,
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

    print(f'Evaluating {n_eval} / {n_total} samples  ({forward_iter} prediction steps each) …')

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

    # ------------------------------------------------------------------ #
    # Evaluation loop
    # ------------------------------------------------------------------ #
    all_metrics: list[dict] = []
    viz_gt_depths:   list[np.ndarray] = []
    viz_pred_depths: list[np.ndarray] = []
    viz_pts_gt:      list[np.ndarray] = []
    viz_pts_pred:    list[np.ndarray] = []

    csv_path = os.path.join(output_root, 'metrics.csv')

    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['sample_idx', 'step_idx', 'range_l1', 'chamfer_dist',
                         'gt_pts', 'pred_pts'])

        t0 = time.time()
        with torch.no_grad():
            for sample_idx, (range_views, poses) in enumerate(loader):
                # range_views: [1, CF + forward_iter, C, H, W]
                # poses:       [1, CF + forward_iter, 4, 4]
                range_views = range_views.to(device)
                poses       = poses.to(device)

                step_gt_depths:   list[np.ndarray] = []
                step_pred_depths: list[np.ndarray] = []
                step_pts_gt:      list[np.ndarray] = []
                step_pts_pred:    list[np.ndarray] = []
                step_metrics:     list[dict]       = []

                # -------------------------------------------------- #
                # 5-step teacher-forced prediction
                # -------------------------------------------------- #
                for step_i in range(forward_iter):
                    # conditioning: GT frames [step_i .. step_i+cf-1]
                    # GT target:    frame [step_i+cf]
                    features_cond = range_views[:, step_i : step_i + cf]       # [1, CF, C, H, W]
                    gt_frame      = range_views[:, step_i + cf]                 # [1, C, H, W]

                    # Relative poses for the CF conditioning steps
                    poses_for_rel = poses[:, step_i : step_i + cf + 1]         # [1, CF+1, 4, 4]
                    rel_pose, rel_yaw = get_rel_pose(poses_for_rel)

                    # Model inference
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                        enabled=(device.type == 'cuda')):
                        pred_frame = model.step_eval(features_cond, rel_pose, rel_yaw)
                    # pred_frame: [1, C, H, W]

                    # Move to NumPy
                    pred_np = pred_frame[0].float().cpu().numpy()   # [C, H, W]
                    gt_np   = gt_frame[0].float().cpu().numpy()     # [C, H, W]

                    pred_depth = _feature_to_depth(pred_np[0], range_mean, range_std)
                    gt_depth   = _feature_to_depth(gt_np[0],   range_mean, range_std)

                    # Back-project to 3-D
                    pts_pred = _backproject(pred_depth, projector)
                    pts_gt   = _backproject(gt_depth,   projector)

                    # Metrics
                    rl1 = _range_l1_np(pred_depth, gt_depth)
                    cd  = _chamfer_distance_np(pts_pred, pts_gt)

                    writer.writerow([sample_idx, step_i,
                                     rl1, cd,
                                     pts_gt.shape[0], pts_pred.shape[0]])
                    csv_file.flush()

                    step_metrics.append({
                        'range_l1':     rl1 if not np.isnan(rl1) else 0.0,
                        'chamfer_dist': cd  if not np.isnan(cd)  else 0.0,
                    })
                    step_gt_depths.append(gt_depth)
                    step_pred_depths.append(pred_depth)
                    step_pts_gt.append(pts_gt)
                    step_pts_pred.append(pts_pred)

                all_metrics.extend(step_metrics)

                # Cache first viz sample
                if sample_idx == cfg.viz_sample_idx:
                    viz_gt_depths   = step_gt_depths
                    viz_pred_depths = step_pred_depths
                    viz_pts_gt      = step_pts_gt
                    viz_pts_pred    = step_pts_pred

                # Progress
                if (sample_idx + 1) % 10 == 0 or sample_idx == n_eval - 1:
                    avg_rl1 = np.nanmean([m['range_l1']     for m in step_metrics])
                    avg_cd  = np.nanmean([m['chamfer_dist']  for m in step_metrics])
                    elapsed = time.time() - t0
                    print(
                        f'[{sample_idx + 1:>5d}/{n_eval}]  '
                        f'avg_range_l1={avg_rl1:.4f}  avg_chamfer={avg_cd:.4f}  '
                        f'({elapsed:.1f}s elapsed)'
                    )

    # ------------------------------------------------------------------ #
    # Summary images  (for viz_sample_idx)
    # ------------------------------------------------------------------ #
    if viz_gt_depths:
        avg_l1_viz = float(np.nanmean(
            [_range_l1_np(p, g) for p, g in zip(viz_pred_depths, viz_gt_depths)]
        ))
        avg_cd_viz = float(np.nanmean(
            [_chamfer_distance_np(pp, pg)
             for pp, pg in zip(viz_pts_pred, viz_pts_gt)]
        ))

        rv_path  = os.path.join(output_root, 'rangeview_sequence.png')
        bev_path = os.path.join(output_root, 'bev_sequence.png')

        render_rangeview_sequence(
            gt_depths=viz_gt_depths,
            pred_depths=viz_pred_depths,
            output_path=rv_path,
            avg_l1=avg_l1_viz,
        )
        render_bev_sequence(
            pts_gt_list=viz_pts_gt,
            pts_pred_list=viz_pts_pred,
            output_path=bev_path,
            avg_chamfer=avg_cd_viz,
            bev_range=cfg.bev_range,
            resolution=cfg.bev_resolution,
        )

    # ------------------------------------------------------------------ #
    # Aggregate summary
    # ------------------------------------------------------------------ #
    valid = [m for m in all_metrics
             if not np.isnan(m['range_l1']) and not np.isnan(m['chamfer_dist'])]

    if valid:
        mean_rl1 = np.mean([m['range_l1']     for m in valid])
        mean_cd  = np.mean([m['chamfer_dist']  for m in valid])
        print('\n' + '=' * 60)
        print(f'Evaluation complete — {n_eval} samples × {forward_iter} steps')
        print(f'  Mean Range L1       : {mean_rl1:.4f}')
        print(f'  Mean Chamfer Dist   : {mean_cd:.4f}')
        print(f'  Metrics CSV         : {csv_path}')
        if viz_gt_depths:
            print(f'  Range-view image    : {rv_path}')
            print(f'  BEV image           : {bev_path}')
        print('=' * 60)

        summary_path = os.path.join(output_root, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f'Experiment       : {cfg.exp_name}\n')
            f.write(f'Checkpoint       : {cfg.resume_path}\n')
            f.write(f'Sequences        : {test_seqs}\n')
            f.write(f'Samples          : {n_eval}\n')
            f.write(f'Prediction steps : {forward_iter}\n')
            f.write(f'Mean Range L1    : {mean_rl1:.6f}\n')
            f.write(f'Mean Chamfer Dist: {mean_cd:.6f}\n')
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
