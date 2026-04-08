"""
Evaluation script for Stage 2 Swin DiT checkpoint — 5-step autoregressive rollout.

For each val sample (5 condition frames) runs an autoregressive rollout to
produce 5 predicted future frames, then saves:

  range_views/range_XXXX.png
      5-row × 3-col grid  (GT range view | Pred range view | |Error|)
      one row per forecast horizon t+1 … t+5, with per-row Range-L1.

  bev/bev_XXXX.png
      5-row × 3-col grid  (GT BEV | Pred BEV | Overlay)
      one row per forecast horizon t+1 … t+5, with per-row Chamfer.

  metrics_summary.json   overall + per-horizon statistics
  metrics_curve.png      per-horizon Chamfer & Range-L1 curves vs sample index

BEV color convention:
    GT   → red    (220,  50,  50)
    Pred → blue   ( 50, 100, 220)
    Both → purple (180,  50, 220)

Usage:
    python scripts/eval_swin_stage2.py \\
        --config configs/swin_config_rangeview.py \\
        --ckpt   /DATA2/shuhul/exp/swin_ckpt/swin-s2-b64/swin_dit_step4000.pkl \\
        --out    outputs/eval_swin_s2_step4000 \\
        --n_samples 200
"""

import os, sys, json, argparse, time
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
from dataset.dataset_kitti_rangeview import KITTIRangeViewValDataset, KITTIRangeViewTestDataset
from dataset.projection import RangeProjection


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',    required=True)
    p.add_argument('--ckpt',      required=True,  help='Path to swin_dit_step*.pkl')
    p.add_argument('--out',       default='outputs/eval_swin_s2')
    p.add_argument('--n_samples', default=200, type=int,
                   help='Max samples to evaluate (0 = all)')
    p.add_argument('--split',     default='test', choices=['val', 'test'],
                   help='Which split to evaluate (val=[6,7]  test=[8,9,10])')
    p.add_argument('--bev_range', default=None, type=float,
                   help='BEV half-extent in metres (default: from config or 50 m)')
    p.add_argument('--max_depth', default=80.0, type=float)
    p.add_argument('--no_vis', action='store_true',
                   help='Skip saving range-view and BEV figures (metrics only)')
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
    if pts_gt.shape[0] > max_pts:
        pts_gt   = pts_gt[np.random.choice(pts_gt.shape[0],   max_pts, replace=False)]
    if pts_pred.shape[0] > max_pts:
        pts_pred = pts_pred[np.random.choice(pts_pred.shape[0], max_pts, replace=False)]
    diff  = pts_gt[:, None, :] - pts_pred[None, :, :]   # (N,M,3)
    dist2 = (diff ** 2).sum(-1)                          # (N,M)
    return float(dist2.min(axis=1).mean() + dist2.min(axis=0).mean())


# ── BEV canvas helper ─────────────────────────────────────────────────────────

def _make_bev_canvas(pts, color, size, bev_range, resolution):
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    if pts.shape[0] > 0:
        col   = ((pts[:, 1] + bev_range) / resolution).astype(np.int32)
        row   = ((bev_range - pts[:, 0]) / resolution).astype(np.int32)
        valid = (col >= 0) & (col < size) & (row >= 0) & (row < size)
        canvas[row[valid], col[valid]] = color
    return canvas


def _bev_axis_ticks(ax, bev_range, resolution):
    ticks_w   = np.arange(-bev_range, bev_range + 1e-6, 10.0)
    col_ticks = ((ticks_w + bev_range) / resolution).astype(int)
    row_ticks = ((bev_range - ticks_w) / resolution).astype(int)
    ax.set_xticks(col_ticks); ax.set_xticklabels([f'{v:.0f}' for v in ticks_w], fontsize=5)
    ax.set_yticks(row_ticks); ax.set_yticklabels([f'{v:.0f}' for v in ticks_w], fontsize=5)
    ax.set_xlabel('Y [m]  (left +)', fontsize=7)
    ax.set_ylabel('X [m]  (forward +)', fontsize=7)


# ── Range-view figure: 5 horizons × (GT | Pred | |Error|) ────────────────────

def save_range_view_figure(sample_idx, steps_data, out_path, max_depth=80.0):
    """
    steps_data: list of (gt_depth [H,W], pred_depth [H,W], range_l1 float)
                one entry per forecast horizon.
    """
    n = len(steps_data)
    fig, axes = plt.subplots(n, 3, figsize=(24, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]   # ensure 2-D indexing

    for h, (gt_depth, pred_depth, range_l1) in enumerate(steps_data):
        gt_clip   = np.clip(gt_depth,   0.0, max_depth)
        pred_clip = np.clip(pred_depth, 0.0, max_depth)
        abs_err   = np.abs(pred_clip - gt_clip)

        im0 = axes[h, 0].imshow(gt_clip,   cmap='turbo_r', vmin=0, vmax=max_depth, aspect='auto')
        im1 = axes[h, 1].imshow(pred_clip, cmap='turbo_r', vmin=0, vmax=max_depth, aspect='auto')
        im2 = axes[h, 2].imshow(abs_err,   cmap='hot',    vmin=0,
                                 vmax=max(float(abs_err.max()), 1e-6), aspect='auto')

        axes[h, 0].set_title(f't+{h+1}  GT depth (m)',   fontsize=9)
        axes[h, 1].set_title(f't+{h+1}  Pred depth (m)  —  Range L1: {range_l1:.4f} m', fontsize=9)
        axes[h, 2].set_title(f't+{h+1}  |Error| (m)',    fontsize=9)

        plt.colorbar(im0, ax=axes[h, 0], fraction=0.012, pad=0.02, label='m')
        plt.colorbar(im1, ax=axes[h, 1], fraction=0.012, pad=0.02, label='m')
        plt.colorbar(im2, ax=axes[h, 2], fraction=0.012, pad=0.02, label='m')

        for ax in axes[h]:
            ax.set_xlabel('azimuth (px)', fontsize=7)
            ax.set_ylabel('elevation (px)', fontsize=7)

    mean_l1 = np.nanmean([s[2] for s in steps_data])
    fig.suptitle(
        f'Sample {sample_idx:04d} — Range View  (t+1 … t+{n})  |  '
        f'mean Range L1: {mean_l1:.4f} m',
        fontsize=12,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ── BEV figure: 5 horizons × (GT BEV | Pred BEV | Overlay) ──────────────────

def save_bev_figure(sample_idx, steps_data, out_path,
                    bev_range=50.0, resolution=0.2):
    """
    steps_data: list of (pts_gt [N,3], pts_pred [M,3], chamfer float)
                one entry per forecast horizon.
    """
    n    = len(steps_data)
    size = int(2 * bev_range / resolution)

    fig, axes = plt.subplots(n, 3, figsize=(18, 6 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    gt_patch   = mpatches.Patch(color=(220/255,  50/255,  50/255), label='GT')
    pred_patch = mpatches.Patch(color=( 50/255, 100/255, 220/255), label='Pred')
    both_patch = mpatches.Patch(color=(180/255,  50/255, 220/255), label='Both')

    for h, (pts_gt, pts_pred, chamfer) in enumerate(steps_data):
        gt_cv   = _make_bev_canvas(pts_gt,   (220,  50,  50), size, bev_range, resolution)
        pred_cv = _make_bev_canvas(pts_pred, ( 50, 100, 220), size, bev_range, resolution)

        gt_mask   = np.any(gt_cv   > 0, axis=2)
        pred_mask = np.any(pred_cv > 0, axis=2)
        overlay   = np.zeros((size, size, 3), dtype=np.uint8)
        overlay[gt_mask & ~pred_mask] = (220,  50,  50)
        overlay[pred_mask & ~gt_mask] = ( 50, 100, 220)
        overlay[gt_mask & pred_mask]  = (180,  50, 220)

        axes[h, 0].imshow(gt_cv,   origin='upper', aspect='equal')
        axes[h, 1].imshow(pred_cv, origin='upper', aspect='equal')
        axes[h, 2].imshow(overlay, origin='upper', aspect='equal')

        axes[h, 0].set_title(f't+{h+1}  GT BEV  ({pts_gt.shape[0]:,} pts)',   fontsize=9)
        axes[h, 1].set_title(f't+{h+1}  Pred BEV  ({pts_pred.shape[0]:,} pts)', fontsize=9)
        axes[h, 2].set_title(f't+{h+1}  Overlay  —  Chamfer: {chamfer:.4f}',  fontsize=9)

        for ax in axes[h]:
            _bev_axis_ticks(ax, bev_range, resolution)

    axes[-1, 2].legend(handles=[gt_patch, pred_patch, both_patch],
                       loc='upper right', fontsize=8)

    mean_cd = np.nanmean([s[2] for s in steps_data])
    fig.suptitle(
        f'Sample {sample_idx:04d} — BEV  (t+1 … t+{n})  |  '
        f'mean Chamfer: {mean_cd:.4f}',
        fontsize=12,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ── Metrics summary plot ──────────────────────────────────────────────────────

def save_metrics_summary(metrics_list, out_path, n_horizons):
    """One curve per forecast horizon for Chamfer and Range-L1."""
    n_samples = len(metrics_list)
    xs        = list(range(n_samples))
    colors    = plt.cm.viridis(np.linspace(0.15, 0.85, n_horizons))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    for h in range(n_horizons):
        cds = [m['chamfer'][h] for m in metrics_list]
        l1s = [m['range_l1'][h] for m in metrics_list]
        lbl = f't+{h+1}'

        ax1.plot(xs, cds, linewidth=1.0, color=colors[h],
                 label=f'{lbl}  (mean={np.nanmean(cds):.3f})')
        ax2.plot(xs, l1s, linewidth=1.0, color=colors[h],
                 label=f'{lbl}  (mean={np.nanmean(l1s):.4f} m)')

    ax1.axhline(np.nanmean([m['chamfer'][h] for m in metrics_list for h in range(n_horizons)]),
                color='red', linestyle='--', linewidth=1.0, label='overall mean')
    ax2.axhline(np.nanmean([m['range_l1'][h] for m in metrics_list for h in range(n_horizons)]),
                color='red', linestyle='--', linewidth=1.0, label='overall mean')

    ax1.set_xlabel('Sample index'); ax1.set_ylabel('Chamfer distance')
    ax1.set_title('Per-sample Chamfer Distance by forecast horizon')
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2.set_xlabel('Sample index'); ax2.set_ylabel('Range L1 (m)')
    ax2.set_title('Per-sample Range L1 by forecast horizon')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    fig.suptitle('Stage 2 Swin DiT — Multi-horizon Evaluation Metrics', fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args_cli = parse_args()
    cfg      = Config.fromfile(args_cli.config)
    cfg.batch_size = 1

    bev_range  = args_cli.bev_range or float(getattr(cfg, 'bev_range', 50.0))
    max_depth  = args_cli.max_depth
    log_range  = bool(getattr(cfg, 'log_range', True))
    range_mean = cfg.proj_img_mean[0]
    range_std  = cfg.proj_img_stds[0]
    n_horizons = int(cfg.forward_iter)   # 5

    out_dir   = args_cli.out
    rv_dir    = os.path.join(out_dir, 'range_views')
    bev_dir   = os.path.join(out_dir, 'bev')
    os.makedirs(rv_dir,  exist_ok=True)
    os.makedirs(bev_dir, exist_ok=True)

    # ── RangeProjection for BEV back-projection ───────────────────────────────
    projector = RangeProjection(
        fov_up=cfg.fov_up, fov_down=cfg.fov_down,
        proj_w=cfg.range_w, proj_h=cfg.range_h,
        fov_left=cfg.fov_left, fov_right=cfg.fov_right,
    )

    # ── Load model ────────────────────────────────────────────────────────────
    print(f'Loading checkpoint: {args_cli.ckpt}')
    ckpt = torch.load(args_cli.ckpt, map_location='cpu')

    # Auto-detect STT depth from checkpoint so the model is built correctly
    # regardless of what n_layer[0] says in the config.
    ckpt_keys = ckpt['model_state_dict'].keys()
    ckpt_n_layer = max(
        int(k.split('.')[2])
        for k in ckpt_keys
        if k.startswith('model.causal_time_space_blocks.')
    ) + 1
    if ckpt_n_layer != cfg.n_layer[0]:
        print(f'  [info] checkpoint has {ckpt_n_layer} STT blocks '
              f'(config says {cfg.n_layer[0]}) — patching cfg.n_layer[0]')
        cfg.n_layer = [ckpt_n_layer] + list(cfg.n_layer[1:])

    # Auto-detect training batch_size from pose_traj_ids shape [bs*CF, 1, 3]
    # so the model buffer is built with the same size as the checkpoint.
    ckpt_pose_bs = ckpt['model_state_dict']['pose_traj_ids'].shape[0]
    cfg.batch_size = ckpt_pose_bs // cfg.condition_frames
    if cfg.batch_size != 1:
        print(f'  [info] checkpoint trained with batch_size={cfg.batch_size} '
              f'(pose_traj_ids={ckpt_pose_bs}) — patching cfg.batch_size for model init')

    cfg.swin_ckpt = None
    model = RangeViewSwinDiT(cfg, local_rank=0)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model = model.cuda().eval()
    step_ckpt = int(ckpt['step'])
    print(f'  Loaded step {step_ckpt}')
    print(f'  Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M')
    print(f'  Condition frames: {cfg.condition_frames}  |  Forecast horizons: {n_horizons}')

    # ── Dataset (val or test) ─────────────────────────────────────────────────
    ds_kwargs = dict(
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
        depth_only=(int(getattr(cfg, 'range_channels', 2)) == 1),
        condition_frames=cfg.condition_frames,
        forward_iter=cfg.forward_iter,
    )
    if args_cli.split == 'test':
        eval_ds = KITTIRangeViewTestDataset(sequences=cfg.test_sequences, **ds_kwargs)
    else:
        eval_ds = KITTIRangeViewValDataset(sequences=cfg.val_sequences, **ds_kwargs)

    n_eval = len(eval_ds) if args_cli.n_samples == 0 else min(args_cli.n_samples, len(eval_ds))
    print(f'{args_cli.split.capitalize()} dataset: {len(eval_ds)} samples  →  evaluating {n_eval}')

    loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
    )

    # ── Eval loop ─────────────────────────────────────────────────────────────
    metrics_list = []

    # Warm-up: one full AR rollout so CUDA kernels are compiled before timing
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _dummy_rv  = torch.zeros(1, cfg.condition_frames, cfg.range_channels,
                                     cfg.range_h, cfg.range_w, device='cuda')
            _dummy_rot = torch.eye(4, device='cuda', dtype=torch.float32
                                   ).unsqueeze(0).unsqueeze(0).expand(
                                       1, cfg.condition_frames + 1, -1, -1)
            try:
                _ = model(_dummy_rv, _dummy_rot)
            except Exception:
                pass
    torch.cuda.synchronize()

    for sample_idx, (range_views, poses) in enumerate(loader):
        if sample_idx >= n_eval:
            break

        range_views = range_views.cuda().to(torch.bfloat16)  # [1, CF+FW, C, H, W]
        poses       = poses.cuda().float()                    # [1, CF+FW, 4, 4]

        # Sliding condition window, initialised with the GT condition frames
        cond_window = range_views[:, :cfg.condition_frames].clone()  # [1, CF, C, H, W]

        rv_steps    = []   # (gt_depth, pred_depth, range_l1) per horizon
        bev_steps   = []   # (pts_gt, pts_pred, chamfer) per horizon
        chamfers    = []
        range_l1s   = []
        step_ms     = []   # per-step inference time (ms)

        # Time the full AR rollout
        torch.cuda.synchronize()
        t_rollout_start = time.perf_counter()

        for h in range(n_horizons):
            # Pose slice for this step: CF condition poses + 1 target pose
            rot_slice_h = poses[:, h : h + cfg.condition_frames + 1]  # [1, CF+1, 4, 4]

            torch.cuda.synchronize()
            t_step_start = time.perf_counter()

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                pred_h = model(cond_window, rot_slice_h)   # [1, C, H, W]

            torch.cuda.synchronize()
            step_ms.append((time.perf_counter() - t_step_start) * 1000.0)

            gt_chw   = range_views[0, cfg.condition_frames + h].detach().float().cpu()
            pred_chw = pred_h[0].detach().float().cpu()

            gt_depth   = to_depth(gt_chw,   log_range, range_mean, range_std)
            pred_depth = to_depth(pred_chw, log_range, range_mean, range_std)

            valid_mask = gt_depth > 0.5
            range_l1   = (float(np.abs(pred_depth - gt_depth)[valid_mask].mean())
                          if valid_mask.any() else float('nan'))

            pts_gt   = projector.back_project_range(np.clip(gt_depth,   0.0, max_depth))
            pts_pred = projector.back_project_range(np.clip(pred_depth, 0.0, max_depth))
            chamfer  = chamfer_distance_np(pts_gt, pts_pred)

            rv_steps.append((gt_depth, pred_depth, range_l1))
            bev_steps.append((pts_gt, pts_pred, chamfer))
            chamfers.append(chamfer)
            range_l1s.append(range_l1)

            # Slide the window: drop oldest frame, append this prediction
            cond_window = torch.cat([
                cond_window[:, 1:],                              # [1, CF-1, C, H, W]
                pred_h.unsqueeze(1).to(torch.bfloat16),         # [1,    1, C, H, W]
            ], dim=1)

        rollout_ms = (time.perf_counter() - t_rollout_start) * 1000.0

        metrics_list.append({
            'sample':      sample_idx,
            'chamfer':     chamfers,
            'range_l1':    range_l1s,
            'rollout_ms':  round(rollout_ms, 3),
            'step_ms':     [round(t, 3) for t in step_ms],
        })

        # Save per-sample figures (skipped when --no_vis)
        if not args_cli.no_vis:
            save_range_view_figure(
                sample_idx, rv_steps,
                os.path.join(rv_dir,  f'range_{sample_idx:04d}.png'),
                max_depth=max_depth,
            )
            save_bev_figure(
                sample_idx, bev_steps,
                os.path.join(bev_dir, f'bev_{sample_idx:04d}.png'),
                bev_range=bev_range,
            )

        if sample_idx % 20 == 0 or sample_idx == n_eval - 1:
            per_h = '  '.join(
                f't+{h+1}: cd={chamfers[h]:.3f} l1={range_l1s[h]:.4f} ({step_ms[h]:.0f}ms)'
                for h in range(n_horizons)
            )
            print(f'  [{sample_idx:4d}/{n_eval}]  rollout={rollout_ms:.0f}ms  {per_h}')

    # ── Aggregate per-horizon statistics ──────────────────────────────────────
    horizon_stats = []
    for h in range(n_horizons):
        cd_vals = [m['chamfer'][h]  for m in metrics_list if not np.isnan(m['chamfer'][h])]
        l1_vals = [m['range_l1'][h] for m in metrics_list if not np.isnan(m['range_l1'][h])]
        horizon_stats.append({
            'horizon':         h + 1,
            'mean_chamfer':    float(np.mean(cd_vals)) if cd_vals else None,
            'mean_range_l1_m': float(np.mean(l1_vals)) if l1_vals else None,
        })

    all_cd = [v for h in range(n_horizons) for m in metrics_list
              if not np.isnan(v := m['chamfer'][h])]
    all_l1 = [v for h in range(n_horizons) for m in metrics_list
              if not np.isnan(v := m['range_l1'][h])]

    rollout_times = [m['rollout_ms'] for m in metrics_list]
    step_times_by_h = [
        [m['step_ms'][h] for m in metrics_list]
        for h in range(n_horizons)
    ]
    mean_rollout_ms   = float(np.mean(rollout_times))
    median_rollout_ms = float(np.median(rollout_times))
    mean_step_ms      = [float(np.mean(t)) for t in step_times_by_h]

    # Add per-horizon timing into horizon_stats
    for h, hs in enumerate(horizon_stats):
        hs['mean_infer_ms'] = round(mean_step_ms[h], 3)

    summary = {
        'checkpoint':              args_cli.ckpt,
        'step':                    step_ckpt,
        'n_samples':               len(metrics_list),
        'condition_frames':        cfg.condition_frames,
        'forecast_horizons':       n_horizons,
        'overall_mean_chamfer':    float(np.mean(all_cd)) if all_cd else None,
        'overall_mean_range_l1':   float(np.mean(all_l1)) if all_l1 else None,
        'rollout_mean_ms':         round(mean_rollout_ms, 3),
        'rollout_median_ms':       round(median_rollout_ms, 3),
        'rollout_fps':             round(1000.0 / mean_rollout_ms, 2) if mean_rollout_ms > 0 else None,
        'per_step_mean_ms':        [round(t, 3) for t in mean_step_ms],
        'per_horizon':             horizon_stats,
    }

    print('\n' + '=' * 80)
    print(f'  Samples evaluated : {summary["n_samples"]}')
    print(f'  {"Horizon":<10} {"Mean Chamfer":>16} {"Mean L1 (m)":>16} {"Infer (ms)":>12}')
    print('  ' + '-' * 56)
    for hs in horizon_stats:
        cd  = f'{hs["mean_chamfer"]:.4f}'    if hs['mean_chamfer']    is not None else 'N/A'
        l1  = f'{hs["mean_range_l1_m"]:.4f}' if hs['mean_range_l1_m'] is not None else 'N/A'
        ms  = f'{hs["mean_infer_ms"]:.1f}'
        print(f'  t+{hs["horizon"]:<8} {cd:>16} {l1:>16} {ms:>12}')
    print('  ' + '-' * 56)
    print(f'  {"Overall":<10} '
          f'{summary["overall_mean_chamfer"]:.4f if summary["overall_mean_chamfer"] else "N/A":>16} '
          f'{summary["overall_mean_range_l1"]:.4f if summary["overall_mean_range_l1"] else "N/A":>16}')
    print(f'\n  AR rollout  —  mean: {mean_rollout_ms:.1f} ms  '
          f'median: {median_rollout_ms:.1f} ms  '
          f'({summary["rollout_fps"]:.2f} rollouts/s)')
    print('=' * 80)

    json_path = os.path.join(out_dir, 'metrics_summary.json')
    with open(json_path, 'w') as f:
        json.dump({'summary': summary, 'per_sample': metrics_list}, f, indent=2)
    print(f'\nMetrics JSON → {json_path}')

    curve_path = os.path.join(out_dir, 'metrics_curve.png')
    save_metrics_summary(metrics_list, curve_path, n_horizons)
    print(f'Metrics plot → {curve_path}')
    print(f'Range views  → {rv_dir}/')
    print(f'BEV figures  → {bev_dir}/')


if __name__ == '__main__':
    main()
