"""
Evaluation script for Stage 2 Swin DiT checkpoint.

For each val sample runs autoregressive inference and saves:
  Per-frame compound figure  (range-view GT | Pred | |Error|) + (BEV GT | Pred | Overlay)
  Per-sequence summary PNG   (per-frame metrics curves)
  Overall metrics JSON       (mean chamfer, mean range-L1)

BEV color convention (matches train_rangeview.py / bev_utils.py):
    GT   → red   (220,  50,  50)
    Pred → blue  ( 50, 100, 220)
    Both → purple(180,  50, 220)

Usage:
    python scripts/eval_swin_stage2.py \\
        --config configs/swin_config_rangeview.py \\
        --ckpt   /DATA2/shuhul/exp/swin_ckpt/swin-s2-b64/swin_dit_step4000.pkl \\
        --out    outputs/eval_swin_s2_step4000 \\
        --n_samples 200
"""

import os, sys, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from utils.config_utils import Config
from models.swin_rae_rangeview import RangeViewSwinDiT
from dataset.dataset_kitti_rangeview import KITTIRangeViewValDataset
from dataset.projection import RangeProjection


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',    required=True)
    p.add_argument('--ckpt',      required=True,  help='Path to swin_dit_step*.pkl')
    p.add_argument('--out',       default='outputs/eval_swin_s2')
    p.add_argument('--n_samples', default=200, type=int,
                   help='Max val samples to evaluate (0 = all)')
    p.add_argument('--bev_range', default=None, type=float,
                   help='BEV half-extent in metres (default: from config or 50 m)')
    p.add_argument('--max_depth', default=80.0, type=float)
    return p.parse_args()


# ── Depth unnormalisation ─────────────────────────────────────────────────────

def to_depth(tensor_chw, log_range, mean, std):
    """Normalised channel-0 tensor → metric depth numpy array (H, W)."""
    arr = tensor_chw[0].float().cpu().numpy()
    if log_range:
        return (2.0 ** (arr * 6.0)) - 1.0
    return arr * std + mean


# ── Chamfer distance (CPU, numpy) ─────────────────────────────────────────────

def chamfer_distance_np(pts_gt, pts_pred, max_pts=4096):
    """Symmetric Chamfer distance between two (N,3) and (M,3) point clouds."""
    if pts_gt.shape[0] == 0 or pts_pred.shape[0] == 0:
        return float('nan')

    # subsample for speed
    if pts_gt.shape[0] > max_pts:
        idx = np.random.choice(pts_gt.shape[0], max_pts, replace=False)
        pts_gt = pts_gt[idx]
    if pts_pred.shape[0] > max_pts:
        idx = np.random.choice(pts_pred.shape[0], max_pts, replace=False)
        pts_pred = pts_pred[idx]

    # (N,1,3) - (1,M,3) → (N,M) squared distances
    diff_gp = pts_gt[:, None, :] - pts_pred[None, :, :]          # (N,M,3)
    dist2   = (diff_gp ** 2).sum(-1)                              # (N,M)
    cd = dist2.min(axis=1).mean() + dist2.min(axis=0).mean()
    return float(cd)


# ── Per-frame compound figure ──────────────────────────────────────────────────

def save_frame_figure(
    frame_idx,
    gt_depth, pred_depth,
    pts_gt, pts_pred,
    chamfer, range_l1,
    out_path,
    bev_range=50.0, resolution=0.2, max_depth=80.0,
):
    """3-col range-view row + 3-col BEV row → single PNG."""
    size = int(2 * bev_range / resolution)

    # ---------- range-view panels ----------
    gt_clip   = np.clip(gt_depth,   0.0, max_depth)
    pred_clip = np.clip(pred_depth, 0.0, max_depth)
    abs_err   = np.abs(pred_clip - gt_clip)

    # ---------- BEV canvases ----------
    def make_canvas(pts, color):
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        if pts.shape[0] > 0:
            col = ((pts[:, 1] + bev_range) / resolution).astype(np.int32)
            row = ((bev_range - pts[:, 0]) / resolution).astype(np.int32)
            valid = (col >= 0) & (col < size) & (row >= 0) & (row < size)
            canvas[row[valid], col[valid]] = color
        return canvas

    gt_canvas   = make_canvas(pts_gt,   (220,  50,  50))
    pred_canvas = make_canvas(pts_pred, ( 50, 100, 220))

    gt_mask   = np.any(gt_canvas   > 0, axis=2)
    pred_mask = np.any(pred_canvas > 0, axis=2)
    overlay   = np.zeros((size, size, 3), dtype=np.uint8)
    overlay[gt_mask & ~pred_mask]   = (220,  50,  50)   # GT only   → red
    overlay[pred_mask & ~gt_mask]   = ( 50, 100, 220)   # Pred only → blue
    overlay[gt_mask & pred_mask]    = (180,  50, 220)   # both      → purple

    # ---------- 2-row figure ----------
    fig, axes = plt.subplots(2, 3, figsize=(24, 9))

    # Row 0: range-view
    im0 = axes[0, 0].imshow(gt_clip,   cmap='plasma', vmin=0, vmax=max_depth, aspect='auto')
    im1 = axes[0, 1].imshow(pred_clip, cmap='plasma', vmin=0, vmax=max_depth, aspect='auto')
    im2 = axes[0, 2].imshow(abs_err,   cmap='hot',    vmin=0,
                             vmax=max(abs_err.max(), 1e-6), aspect='auto')
    axes[0, 0].set_title('GT depth (m)',   fontsize=10)
    axes[0, 1].set_title('Pred depth (m)', fontsize=10)
    axes[0, 2].set_title(f'|Error| (m)  — Range L1 (valid): {range_l1:.4f} m', fontsize=10)
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.015, pad=0.02, label='m')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.015, pad=0.02, label='m')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.015, pad=0.02, label='m')
    for ax in axes[0]:
        ax.set_xlabel('azimuth (px)'); ax.set_ylabel('elevation (px)')

    # Row 1: BEV
    for ax, canvas, subtitle in zip(
        axes[1],
        [gt_canvas, pred_canvas, overlay],
        [
            f'GT BEV  ({pts_gt.shape[0]:,} pts)',
            f'Pred BEV  ({pts_pred.shape[0]:,} pts)',
            f'Overlay  — Chamfer: {chamfer:.4f}',
        ],
    ):
        ax.imshow(canvas, origin='upper', aspect='equal')
        ax.set_title(subtitle, fontsize=10)

        # axis ticks in world metres
        ticks_w = np.arange(-bev_range, bev_range + 1e-6, 10.0)
        col_ticks = ((ticks_w + bev_range) / resolution).astype(int)
        row_ticks = ((bev_range - ticks_w) / resolution).astype(int)
        ax.set_xticks(col_ticks); ax.set_xticklabels([f'{v:.0f}' for v in ticks_w], fontsize=6)
        ax.set_yticks(row_ticks); ax.set_yticklabels([f'{v:.0f}' for v in ticks_w], fontsize=6)
        ax.set_xlabel('Y [m]  (left +)', fontsize=8)
        ax.set_ylabel('X [m]  (forward +)', fontsize=8)

    gt_patch   = mpatches.Patch(color=(220/255,  50/255,  50/255), label='GT')
    pred_patch = mpatches.Patch(color=( 50/255, 100/255, 220/255), label='Pred')
    both_patch = mpatches.Patch(color=(180/255,  50/255, 220/255), label='Both')
    axes[1, 2].legend(handles=[gt_patch, pred_patch, both_patch],
                      loc='upper right', fontsize=8)

    fig.suptitle(
        f'Frame {frame_idx:04d}  |  Range L1 (valid): {range_l1:.4f} m  '
        f'|  Chamfer: {chamfer:.4f}',
        fontsize=12,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ── Metrics summary plot ──────────────────────────────────────────────────────

def save_metrics_summary(metrics_list, out_path):
    """Line plots of per-frame chamfer & range_l1, with mean lines."""
    frames   = list(range(len(metrics_list)))
    chamfers = [m['chamfer'] for m in metrics_list]
    l1s      = [m['range_l1'] for m in metrics_list]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    ax1.plot(frames, chamfers, linewidth=1.0, color='steelblue', label='Chamfer dist')
    ax1.axhline(np.nanmean(chamfers), color='red', linestyle='--', linewidth=1.2,
                label=f'mean = {np.nanmean(chamfers):.4f}')
    ax1.set_xlabel('Frame'); ax1.set_ylabel('Chamfer distance')
    ax1.set_title('Per-frame Chamfer Distance'); ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    ax2.plot(frames, l1s, linewidth=1.0, color='darkorange', label='Range L1 (valid)')
    ax2.axhline(np.nanmean(l1s), color='red', linestyle='--', linewidth=1.2,
                label=f'mean = {np.nanmean(l1s):.4f} m')
    ax2.set_xlabel('Frame'); ax2.set_ylabel('Range L1 (m)')
    ax2.set_title('Per-frame Range L1 (valid depth pixels)'); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    fig.suptitle('Stage 2 Swin DiT — Per-frame Evaluation Metrics', fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args_cli = parse_args()
    cfg = Config.fromfile(args_cli.config)
    cfg.batch_size = 1   # model constructor uses batch_size to precompute img_ids

    bev_range  = args_cli.bev_range or float(getattr(cfg, 'bev_range', 50.0))
    max_depth  = args_cli.max_depth
    log_range  = bool(getattr(cfg, 'log_range', True))
    range_mean = cfg.proj_img_mean[0]
    range_std  = cfg.proj_img_stds[0]
    out_dir    = args_cli.out
    frames_dir = os.path.join(out_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    # ── Build RangeProjection for BEV back-projection ─────────────────────────
    projector = RangeProjection(
        fov_up=cfg.fov_up, fov_down=cfg.fov_down,
        proj_w=cfg.range_w, proj_h=cfg.range_h,
        fov_left=cfg.fov_left, fov_right=cfg.fov_right,
    )

    # ── Load model ────────────────────────────────────────────────────────────
    # Suppress swin_ckpt loading in __init__: the Stage 2 checkpoint already
    # contains the frozen encoder+decoder weights, so the Stage 1 swin_ckpt
    # load is redundant and would crash if that path is stale.
    print(f'Loading checkpoint: {args_cli.ckpt}')
    ckpt = torch.load(args_cli.ckpt, map_location='cpu')
    cfg.swin_ckpt = None
    model = RangeViewSwinDiT(cfg, local_rank=0)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model = model.cuda().eval()
    print(f'  Loaded step {ckpt["step"]}')
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'  Total params: {n_params:.1f} M')

    # ── Val dataset ───────────────────────────────────────────────────────────
    val_ds = KITTIRangeViewValDataset(
        sequences=cfg.val_sequences,
        sequences_path=cfg.kitti_sequences_path,
        poses_path=cfg.kitti_poses_path,
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
        condition_frames=cfg.condition_frames,
        forward_iter=cfg.forward_iter,
    )
    n_eval = len(val_ds) if args_cli.n_samples == 0 else min(args_cli.n_samples, len(val_ds))
    print(f'Val dataset: {len(val_ds)} samples  →  evaluating {n_eval}')

    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # ── Eval loop ─────────────────────────────────────────────────────────────
    metrics_list = []

    for sample_idx, (range_views, poses) in enumerate(loader):
        if sample_idx >= n_eval:
            break

        range_views = range_views.cuda().to(torch.bfloat16)   # [1, CF+FW, C, H, W]
        poses       = poses.cuda().float()                     # [1, CF+FW, 4, 4]

        features_cond = range_views[:, :cfg.condition_frames]
        features_gt   = range_views[:, cfg.condition_frames:cfg.condition_frames + 1]
        rot_slice     = poses[:, :cfg.condition_frames + 1]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            predict = model(features_cond, rot_slice)   # [1, C, H, W]  (step_eval)

        # shape: [C, H, W] numpy, float32
        gt_chw   = features_gt[0, 0].detach().float().cpu()
        pred_chw = predict[0].detach().float().cpu()

        gt_depth   = to_depth(gt_chw,   log_range, range_mean, range_std)
        pred_depth = to_depth(pred_chw, log_range, range_mean, range_std)

        # valid-pixel range L1
        valid_mask = gt_depth > 0.5
        range_l1   = float(np.abs(pred_depth - gt_depth)[valid_mask].mean()) \
                     if valid_mask.any() else float('nan')

        # back-project to 3-D
        pts_gt   = projector.back_project_range(np.clip(gt_depth,   0, max_depth))
        pts_pred = projector.back_project_range(np.clip(pred_depth, 0, max_depth))

        chamfer = chamfer_distance_np(pts_gt, pts_pred)

        metrics_list.append({'frame': sample_idx, 'chamfer': chamfer, 'range_l1': range_l1})

        # per-frame compound figure
        frame_path = os.path.join(frames_dir, f'frame_{sample_idx:04d}.png')
        save_frame_figure(
            frame_idx=sample_idx,
            gt_depth=gt_depth, pred_depth=pred_depth,
            pts_gt=pts_gt, pts_pred=pts_pred,
            chamfer=chamfer, range_l1=range_l1,
            out_path=frame_path,
            bev_range=bev_range, resolution=0.2, max_depth=max_depth,
        )

        if sample_idx % 20 == 0:
            print(f'  [{sample_idx:4d}/{n_eval}]  range_l1={range_l1:.4f} m  '
                  f'chamfer={chamfer:.4f}  pts_gt={pts_gt.shape[0]:,}  '
                  f'pts_pred={pts_pred.shape[0]:,}')

    # ── Aggregate & save ──────────────────────────────────────────────────────
    valid_chamfers = [m['chamfer'] for m in metrics_list if not np.isnan(m['chamfer'])]
    valid_l1s      = [m['range_l1'] for m in metrics_list if not np.isnan(m['range_l1'])]

    summary = {
        'checkpoint':      args_cli.ckpt,
        'step':            int(ckpt['step']),
        'n_frames':        len(metrics_list),
        'mean_chamfer':    float(np.mean(valid_chamfers)) if valid_chamfers else None,
        'mean_range_l1_m': float(np.mean(valid_l1s))     if valid_l1s      else None,
    }
    print('\n' + '='*60)
    print(f'  Frames evaluated : {summary["n_frames"]}')
    print(f'  Mean Chamfer     : {summary["mean_chamfer"]:.4f}')
    print(f'  Mean Range L1    : {summary["mean_range_l1_m"]:.4f} m')
    print('='*60)

    json_path = os.path.join(out_dir, 'metrics_summary.json')
    with open(json_path, 'w') as f:
        json.dump({'summary': summary, 'per_frame': metrics_list}, f, indent=2)
    print(f'Metrics saved → {json_path}')

    summary_plot_path = os.path.join(out_dir, 'metrics_curve.png')
    save_metrics_summary(metrics_list, summary_plot_path)
    print(f'Metrics plot  → {summary_plot_path}')
    print(f'Frame PNGs    → {frames_dir}/')


if __name__ == '__main__':
    main()
