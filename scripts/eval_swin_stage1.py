"""
Evaluation script for Stage 1 Swin RAE checkpoint — range-view reconstruction.

Runs the trained encoder-decoder on each test/val sample and saves:

  range_views/range_XXXX.png
      3-panel figure: GT depth | Reconstructed depth | |Error| (m)
      with per-sample MAE in the title.

  metrics_summary.json   overall + per-sample statistics
  metrics_curve.png      per-sample MAE curve

Usage:
    python scripts/eval_swin_stage1.py \\
        --config configs/swin_config_rangeview.py \\
        --ckpt   /DATA2/shuhul/exp/swin_ckpt/swin-s1-ch1-b32/swin_rae_step160000.pkl \\
        --out    outputs/eval_swin_s1_step160000 \\
        --split  test \\
        --n_samples 0
"""

import os, sys, json, argparse, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from utils.config_utils import Config
from models.swin_rae_rangeview import RangeViewSwinRAE
from dataset.dataset_kitti_rangeview import KITTIRangeViewVAEDataset


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',    required=True)
    p.add_argument('--ckpt',      required=True,  help='Path to swin_rae_step*.pkl')
    p.add_argument('--out',       default='outputs/eval_swin_s1')
    p.add_argument('--split',     default='test', choices=['val', 'test'],
                   help='Which split to evaluate (val=[6,7]  test=[8,9,10])')
    p.add_argument('--n_samples', default=0, type=int,
                   help='Max samples to evaluate (0 = all)')
    p.add_argument('--max_depth', default=80.0, type=float,
                   help='Depth clip for visualisation')
    p.add_argument('--batch_size', default=8, type=int,
                   help='Inference batch size')
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--no_vis', action='store_true',
                   help='Skip saving per-sample range-view figures (faster, saves disk)')
    return p.parse_args()


# ── Depth unnormalisation ─────────────────────────────────────────────────────

def to_depth(tensor_hw, log_range, mean, std):
    """Normalised single-channel 2-D numpy array → metric depth (H, W)."""
    arr = tensor_hw.float().cpu().numpy() if torch.is_tensor(tensor_hw) else tensor_hw
    if log_range:
        return (2.0 ** (arr * 6.0)) - 1.0
    return arr * std + mean


# ── Per-sample visualisation ──────────────────────────────────────────────────

def save_range_view_figure(sample_idx, gt_depth, pred_depth, mae,
                            out_path, max_depth=80.0):
    gt_clip   = np.clip(gt_depth,   0.0, max_depth)
    pred_clip = np.clip(pred_depth, 0.0, max_depth)
    abs_err   = np.abs(pred_clip - gt_clip)

    fig, axes = plt.subplots(1, 3, figsize=(24, 4))

    im0 = axes[0].imshow(gt_clip,   cmap='plasma', vmin=0, vmax=max_depth, aspect='auto')
    im1 = axes[1].imshow(pred_clip, cmap='plasma', vmin=0, vmax=max_depth, aspect='auto')
    im2 = axes[2].imshow(abs_err,   cmap='hot',    vmin=0,
                          vmax=max(float(abs_err.max()), 1e-6), aspect='auto')

    axes[0].set_title('GT depth (m)',       fontsize=9)
    axes[1].set_title(f'Reconstructed (m)  —  MAE: {mae:.4f} m', fontsize=9)
    axes[2].set_title('|Error| (m)',         fontsize=9)

    for ax, im in zip(axes, [im0, im1, im2]):
        plt.colorbar(im, ax=ax, fraction=0.012, pad=0.02, label='m')
        ax.set_xlabel('azimuth (px)', fontsize=7)
        ax.set_ylabel('elevation (px)', fontsize=7)

    fig.suptitle(f'Sample {sample_idx:04d} — Stage 1 RAE Reconstruction', fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ── Metrics curve plot ────────────────────────────────────────────────────────

def save_metrics_curve(metrics_list, out_path):
    xs   = list(range(len(metrics_list)))
    maes = [m['mae_m'] for m in metrics_list]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(xs, maes, linewidth=0.8, color='steelblue')
    ax.axhline(np.nanmean(maes), color='red', linestyle='--', linewidth=1.0,
               label=f'mean = {np.nanmean(maes):.4f} m')
    ax.set_xlabel('Sample index'); ax.set_ylabel('MAE (m)')
    ax.set_title('Per-sample Depth MAE'); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle('Stage 1 Swin RAE — Reconstruction MAE', fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args_cli = parse_args()
    cfg      = Config.fromfile(args_cli.config)

    log_range  = bool(getattr(cfg, 'log_range', True))
    range_mean = cfg.proj_img_mean[0]
    range_std  = cfg.proj_img_stds[0]
    max_depth  = args_cli.max_depth

    out_dir = args_cli.out
    rv_dir  = os.path.join(out_dir, 'range_views')
    os.makedirs(rv_dir, exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f'Loading checkpoint: {args_cli.ckpt}')
    ckpt = torch.load(args_cli.ckpt, map_location='cpu')
    state = ckpt['model_state_dict']

    # Auto-detect number of input channels from the encoder patch-embed weight
    # Shape: [embed_dim, in_chans, patch_h, patch_w]
    pe_key = next(k for k in state if 'patch_embed' in k and 'proj.weight' in k)
    ckpt_n_ch = state[pe_key].shape[1]
    if ckpt_n_ch != int(getattr(cfg, 'range_channels', 2)):
        print(f'  [info] checkpoint has {ckpt_n_ch} input channel(s) '
              f'(config says {cfg.range_channels}) — patching cfg.range_channels')
        cfg.range_channels = ckpt_n_ch

    # Build model
    cfg.swin_ckpt = None
    model = RangeViewSwinRAE(cfg, local_rank=0)
    model.load_state_dict(state, strict=True)
    model = model.cuda().eval()

    step_ckpt = int(ckpt['step'])
    n_params  = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'  Loaded step {step_ckpt}  |  {n_params:.1f} M params  |  {ckpt_n_ch}-channel input')

    # ── Dataset ───────────────────────────────────────────────────────────────
    sequences = cfg.test_sequences if args_cli.split == 'test' else cfg.val_sequences
    print(f'Split: {args_cli.split}  sequences={sequences}')

    ds = KITTIRangeViewVAEDataset(
        sequences=sequences,
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
        is_train=False,
    )
    n_eval = len(ds) if args_cli.n_samples == 0 else min(args_cli.n_samples, len(ds))
    print(f'Dataset: {len(ds)} frames  →  evaluating {n_eval}')

    loader = torch.utils.data.DataLoader(
        ds, batch_size=args_cli.batch_size, shuffle=False,
        num_workers=args_cli.num_workers, pin_memory=True,
    )

    # ── Eval loop ─────────────────────────────────────────────────────────────
    metrics_list = []
    sample_idx   = 0

    # Warm-up: one forward pass so CUDA kernels are compiled before timing
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _dummy = torch.zeros(1, ckpt_n_ch, cfg.range_h, cfg.range_w, device='cuda')
            _ = model(_dummy)
    torch.cuda.synchronize()

    for frames, _ in loader:
        if sample_idx >= n_eval:
            break

        # Slice to the number of channels the model was trained with
        frames = frames[:, :ckpt_n_ch].cuda().to(torch.bfloat16)   # [B, C, H, W]
        actual_batch = min(frames.shape[0], n_eval - sample_idx)

        torch.cuda.synchronize()
        t_start = time.perf_counter()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                out = model(frames)

        torch.cuda.synchronize()
        batch_ms = (time.perf_counter() - t_start) * 1000.0   # ms for full batch
        per_sample_ms = batch_ms / actual_batch

        x_rec = out['x_rec']   # [B, C, H, W]

        # Process each item in the batch individually
        batch_size = frames.shape[0]
        for b in range(batch_size):
            if sample_idx >= n_eval:
                break

            gt_chw   = frames[b].float().cpu()
            pred_chw = x_rec[b].float().detach().cpu()

            gt_depth   = to_depth(gt_chw[0],   log_range, range_mean, range_std)
            pred_depth = to_depth(pred_chw[0],  log_range, range_mean, range_std)

            valid = gt_depth > 0.5
            if valid.any():
                mae = float(np.abs(pred_depth[valid] - gt_depth[valid]).mean())
            else:
                mae = float('nan')

            metrics_list.append({
                'sample':        sample_idx,
                'mae_m':         mae,
                'infer_ms':      round(per_sample_ms, 3),
            })

            if not args_cli.no_vis:
                save_range_view_figure(
                    sample_idx, gt_depth, pred_depth, mae,
                    os.path.join(rv_dir, f'range_{sample_idx:04d}.png'),
                    max_depth=max_depth,
                )

            if sample_idx % 50 == 0 or sample_idx == n_eval - 1:
                print(f'  [{sample_idx:5d}/{n_eval}]  MAE={mae:.4f} m  '
                      f'infer={per_sample_ms:.1f} ms/sample')

            sample_idx += 1

    # ── Aggregate statistics ──────────────────────────────────────────────────
    maes   = [m['mae_m']    for m in metrics_list if not np.isnan(m['mae_m'])]
    timings = [m['infer_ms'] for m in metrics_list]

    mean_infer_ms   = float(np.mean(timings))
    median_infer_ms = float(np.median(timings))
    fps             = 1000.0 / mean_infer_ms if mean_infer_ms > 0 else None

    summary = {
        'checkpoint':            args_cli.ckpt,
        'step':                  step_ckpt,
        'split':                 args_cli.split,
        'n_samples':             len(metrics_list),
        'n_channels':            ckpt_n_ch,
        'batch_size':            args_cli.batch_size,
        'overall_mean_mae_m':    float(np.mean(maes))    if maes else None,
        'overall_median_mae_m':  float(np.median(maes))  if maes else None,
        'infer_mean_ms':         round(mean_infer_ms, 3),
        'infer_median_ms':       round(median_infer_ms, 3),
        'infer_fps':             round(fps, 2) if fps else None,
    }

    print('\n' + '=' * 60)
    print(f'  Split          : {args_cli.split}')
    print(f'  Samples        : {summary["n_samples"]}')
    print(f'  Mean   MAE     : {summary["overall_mean_mae_m"]:.4f} m')
    print(f'  Median MAE     : {summary["overall_median_mae_m"]:.4f} m')
    print(f'  Infer (mean)   : {mean_infer_ms:.1f} ms/sample  '
          f'({fps:.1f} FPS)')
    print(f'  Infer (median) : {median_infer_ms:.1f} ms/sample')
    print('=' * 60)

    json_path = os.path.join(out_dir, 'metrics_summary.json')
    with open(json_path, 'w') as f:
        json.dump({'summary': summary, 'per_sample': metrics_list}, f, indent=2)
    print(f'\nMetrics JSON → {json_path}')

    curve_path = os.path.join(out_dir, 'metrics_curve.png')
    save_metrics_curve(metrics_list, curve_path)
    print(f'Metrics plot → {curve_path}')
    print(f'Range views  → {rv_dir}/')


if __name__ == '__main__':
    main()
