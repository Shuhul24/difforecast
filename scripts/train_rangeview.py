"""
Training script for Range View DiT Model
Simplified training without trajectory prediction

The script now supports a stage-wise workflow inspired by the
RangeLDM project.  The user may select one of three modes via
``--stage``:

    1   pretrain the RangeLDM VAE component only (ELBO loss);
    2   train the DiT/STT modules with a frozen VAE (requires a
        pretrained ``vae_ckpt`` for best behaviour);
    all full training (default) which behaves as before.

Stage 1 automatically freezes DiT/STT, and stage 2 will freeze the
VAE if a checkpoint path is provided.  The loss function and logging
are adjusted accordingly.
"""

import os
# Reduce CUDA allocator fragmentation — set before any torch import
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys
import math
import time
import torch
import random
import logging
import argparse
import shutil
from einops import rearrange
import numpy as np
from deepspeed.ops.adam import FusedAdam
import torch.distributed as dist
import torch.utils
from torch.utils.data import DataLoader, Subset, DistributedSampler
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import deepspeed
import wandb

# Add Epona root to path BEFORE importing local modules
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Root path: {root_path}")
sys.path.append(root_path)

# Now import local modules
from utils.config_utils import Config
from utils.deepspeed_utils import get_deepspeed_config
from utils.utils import *
from utils.preprocess import get_rel_pose
from models.model_rangeview import RangeViewDiT, RangeViewVAE
from dataset.dataset_kitti_rangeview import (
    KITTIRangeViewTrainDataset,
    KITTIRangeViewValDataset,
    KITTIRangeViewVAEDataset
)
from dataset.projection import RangeProjection
from utils.comm import _init_dist_envi
from utils.running import init_lr_schedule, get_cosine_schedule_with_warmup, save_ckpt, load_parameters, add_weight_decay, save_ckpt_deepspeed, load_from_deepspeed_ckpt
from utils.bev_utils import render_bev_comparison, render_rangeview_comparison
from torch.nn.parallel import DistributedDataParallel as DDP
from models.modules.range_discriminator import (
    NLayerDiscriminatorMetaKernel,
    weights_init as disc_weights_init,
    hinge_d_loss,
    adopt_weight as disc_adopt_weight,
    calculate_adaptive_weight,
)

logger = logging.getLogger('base')


def _compose_predicted_rot(base_rot, pred_xy, pred_yaw_deg):
    """Compose predicted absolute 4×4 pose from previous absolute pose + predicted
    relative (dx, dy, yaw_deg).  Mirrors train_swin_rangeview._compose_predicted_rot.

    base_rot    : [B, 4, 4]  last conditioning frame absolute pose
    pred_xy     : [B, 2]     predicted (dx, dy) in body frame (metres)
    pred_yaw_deg: [B, 1]     predicted relative yaw (degrees)
    Returns     : [B, 4, 4]  predicted absolute pose of the next frame
    """
    B     = base_rot.shape[0]
    yaw   = pred_yaw_deg[:, 0] * (math.pi / 180.)
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    dx    = pred_xy[:, 0]
    dy    = pred_xy[:, 1]
    z     = torch.zeros(B, device=base_rot.device, dtype=base_rot.dtype)
    o     = torch.ones_like(z)
    T_rel = torch.stack([
        cos_y, -sin_y, z, dx,
        sin_y,  cos_y, z, dy,
        z,      z,     o, z,
        z,      z,     z, o,
    ], dim=1).view(B, 4, 4)
    return base_rot @ T_rel


def add_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Range View DiT Model')
    parser.add_argument('--iter', default=60000000, type=int, help='Total training iterations')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size per GPU')
    parser.add_argument('--config', required=True, type=str, help='Path to config file')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=1e-4, help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--resume_path', default=None, type=str, help='Path to resume from')
    parser.add_argument('--resume_step', default=0, type=int, help='Step to resume from')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--launcher', type=str, default='pytorch', help='Job launcher')
    parser.add_argument('--overfit', action='store_true', help='Overfit on small subset')
    parser.add_argument('--eval_steps', type=int, default=2000, help='Checkpoint save interval')
    parser.add_argument('--load_from_deepspeed', default=None, type=str, help='Load from DeepSpeed checkpoint')
    parser.add_argument('--wandb_project', type=str, default='difforecast-rangeview',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='W&B run name (defaults to --exp_name)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--no_log_file', action='store_true',
                        help='Disable saving a log file (useful on HPC where stdout is captured in .out files)')
    parser.add_argument('--vis_steps', type=int, default=500,
                        help='Save training visualizations every N steps (0 = disabled)')
    parser.add_argument('--warmup_steps', type=int, default=2000, help='Warmup steps for LR scheduler')

    parser.add_argument('--vae_ckpt', type=str, default=None, help='Path to pretrained VAE checkpoint for stage 2')
    parser.add_argument('--disc_resume_path', type=str, default=None, help='Path to resume discriminator checkpoint (Stage 1 only)')

    # Stage-wise training support (1 = VAE only, 2 = DiT/STT only, all = default/full)
    parser.add_argument('--stage', type=str, default='all', choices=['1','2','all'],
                        help='Which stage to run: 1 for VAE pretraining, 2 for DiT/STT training, all for both (default)')

    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.__dict__)

    # Enforce stage-specific loss configurations dynamically
    if getattr(cfg, 'stage', 'all') == '2':
        cfg.elbo_weight = 0.0
        cfg.bev_perceptual_weight = 0.0
        # chamfer_loss_weight, repa_weight and range_view_loss_weight are
        # intentionally preserved from the config so Stage 2 can supervise
        # the DiT with geometry-aware and pixel-space losses.
        # Stage 2 needs return_predict=True for Chamfer / RV loss (DiT x_0
        # estimate) and return_hidden=True for REPA (intermediate hidden states).
        cfg.return_predict = True
    elif getattr(cfg, 'stage', 'all') == '1':
        # Config default is 0.0 (for stage 2); force it on for stage 1 VAE training.
        cfg.elbo_weight = 1.0

    return cfg


def init_logs(global_rank, args):
    """Initialize logging directories and loggers"""
    print(f'Initializing logs... (stage={getattr(args, "stage", "all")})')
    log_path = os.path.join(args.logdir, args.exp_name)
    save_model_path = os.path.join(args.outdir, args.exp_name)
    tdir_path = os.path.join(args.tdir, args.exp_name)
    validation_path = os.path.join(args.validation_dir, args.exp_name)

    if global_rank == 0:
        # Create directories
        os.makedirs(save_model_path, exist_ok=True)
        os.makedirs(validation_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(tdir_path, exist_ok=True)

        # Setup logger
        setup_logger('base', log_path, 'train', level=logging.INFO, screen=True, to_file=not getattr(args, 'no_log_file', False))

        # Setup tensorboard
        writer = SummaryWriter(os.path.join(tdir_path, 'train'))
        writer_val = SummaryWriter(os.path.join(tdir_path, 'validate'))

        args.writer = writer
        args.writer_val = writer_val

        # Setup Weights & Biases (rank-0 only)
        if not getattr(args, 'no_wandb', False):
            wandb.init(
                project=getattr(args, 'wandb_project', 'difforecast-rangeview'),
                name=getattr(args, 'wandb_run_name', None) or args.exp_name,
                config={k: v for k, v in vars(args).items()
                        if isinstance(v, (int, float, str, bool, type(None)))},
                resume='allow',
            )
    else:
        args.writer = None
        args.writer_val = None

    args.log_path = log_path
    args.save_model_path = save_model_path
    args.tdir_path = tdir_path
    args.validation_path = validation_path


def init_environment(args):
    """Initialize distributed environment and random seeds"""
    _init_dist_envi(args)

    # Set random seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set backends
    torch.backends.cudnn.benchmark = True


def create_rangeview_dataset(args):
    """Create KITTI range view dataset"""
    print("Creating KITTI range view dataset...")

    stage = getattr(args, 'stage', 'all')
    
    if stage == '1':
        print("Stage 1 selected: Using KITTIRangeViewVAEDataset (single frames, stride=1).")
        train_dataset = KITTIRangeViewVAEDataset(
            sequences_path=args.kitti_sequences_path,
            poses_path=args.kitti_poses_path,
            sequences=args.train_sequences,
            # condition_frames/forward_iter are ignored/overridden by VAE dataset class
            h=args.range_h,
            w=args.range_w,
            fov_up=args.fov_up,
            fov_down=args.fov_down,
            fov_left=args.fov_left,
            fov_right=args.fov_right,
            proj_img_mean=args.proj_img_mean,
            proj_img_stds=args.proj_img_stds,
            augmentation_config=args.augmentation_config if hasattr(args, 'augmentation_config') else None,
            pc_extension=args.pc_extension,
            pc_dtype=getattr(np, args.pc_dtype),
            pc_reshape=tuple(args.pc_reshape),
        )
    else:
        print("Stage 2/All selected: Using KITTIRangeViewTrainDataset (sequences).")
        train_dataset = KITTIRangeViewTrainDataset(
            sequences_path=args.kitti_sequences_path,
            poses_path=args.kitti_poses_path,
            sequences=args.train_sequences,
            condition_frames=args.condition_frames,
            forward_iter=args.forward_iter,
            h=args.range_h,
            w=args.range_w,
            fov_up=args.fov_up,
            fov_down=args.fov_down,
            fov_left=args.fov_left,
            fov_right=args.fov_right,
            proj_img_mean=args.proj_img_mean,
            proj_img_stds=args.proj_img_stds,
            augmentation_config=args.augmentation_config if hasattr(args, 'augmentation_config') else None,
            pc_extension=args.pc_extension,
            pc_dtype=getattr(np, args.pc_dtype),
            pc_reshape=tuple(args.pc_reshape),
        )

    print(f"KITTI training dataset created with {len(train_dataset)} samples")
    return train_dataset


def save_training_visualization(
    step, args, features_cond, features_gt, predict_decoded, projector, vis_dir
):
    """Save range-view and BEV comparison images for one training batch.

    Visualises the first sample in the batch:
      - All conditioning frames (range-view, depth channel coloured by plasma)
      - GT next frame vs predicted next frame (range-view + BEV side-by-side)

    Args:
        step:             Current training step (used for filename).
        args:             Config namespace (provides proj_img_mean/stds, bev_range etc.)
        features_cond:    ``[B, CF, C, H, W]`` conditioning range-view images (normalised).
        features_gt:      ``[B, 1, C, H, W]``  GT next frame (normalised).
        predict_decoded:  ``[(B*CF), C, H, W]`` or ``[B, C, H, W]`` predicted frame
                          (normalised), or None if not available.
        projector:        :class:`RangeProjection` for depth back-projection to 3-D.
        vis_dir:          Directory to save PNG files into.
    """
    if predict_decoded is None:
        return

    is_vae_mode = features_cond is None
    os.makedirs(vis_dir, exist_ok=True)

    range_mean = args.proj_img_mean[0]
    range_std  = args.proj_img_stds[0]

    # Work with the first batch sample only, move to CPU numpy
    if is_vae_mode:
        # VAE Mode: features_gt is [B*T, C, H, W], predict_decoded is [B*T, C, H, W]
        # We just visualize the first sample in the flattened batch
        gt_np = features_gt[0].detach().float().cpu().numpy()      # [C, H, W]
        pred_np = predict_decoded[0].detach().float().cpu().numpy() # [C, H, W]
    else:
        # Forecast Mode: features_cond is [B, CF, C, H, W]
        cond_np  = features_cond[0].detach().float().cpu().numpy()   # [CF, C, H, W]
        # features_gt is [B, 1, C, H, W]
        gt_np    = features_gt[0, 0].detach().float().cpu().numpy()  # [C, H, W]
        # predict_decoded logic below...

    if not is_vae_mode:
        cf = cond_np.shape[0]
        pred_np  = predict_decoded[cf - 1].detach().float().cpu().numpy()  # [C, H, W]

    # --- unnormalise depth channel (ch 0) to metres ---
    def to_depth(feat_chw):
        return feat_chw[0] * range_std + range_mean   # (H, W)

    gt_depth   = to_depth(gt_np)
    pred_depth = to_depth(pred_np)

    # --- 1. Conditioning frames — saved as a single wide range-view strip ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    max_depth = 80.0
    if not is_vae_mode:
        CF = cond_np.shape[0]
        fig, axes = plt.subplots(1, CF, figsize=(6 * CF, 4))
        if CF == 1:
            axes = [axes]
        for t, ax in enumerate(axes):
            depth_t = to_depth(cond_np[t])
            im = ax.imshow(np.clip(depth_t, 0, max_depth),
                           cmap=matplotlib.colormaps['turbo_r'],
                           vmin=0, vmax=max_depth, aspect='auto')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='m')
            # t=0 is oldest frame, t=CF-1 is most recent (t=0 in absolute time)
            time_label = f't-{CF - 1 - t}' if CF - 1 - t > 0 else 't'
            ax.set_title(f'Cond [{time_label}]', fontsize=10)
            ax.set_xlabel('azimuth'); ax.set_ylabel('elevation')
        fig.suptitle(f'Step {step} — Conditioning frames (depth)', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'step{step:07d}_cond.png'), dpi=80, bbox_inches='tight')
        plt.close(fig)

    # --- 1b. Intensity Comparison (If input has >= 2 channels) ---
    # Channel 0 is Range, Channel 1 is Intensity (usually)
    if gt_np.shape[0] >= 2:
        int_mean = args.proj_img_mean[1]
        int_std  = args.proj_img_stds[1]

        def to_intensity(feat_chw):
            # Unnormalize: x * std + mean
            return feat_chw[1] * int_std + int_mean

        gt_int   = to_intensity(gt_np)
        pred_int = to_intensity(pred_np)
        
        # Plot GT vs Recon Intensity
        fig_int, ax_int = plt.subplots(2, 1, figsize=(10, 6))
        
        im_gt = ax_int[0].imshow(gt_int, cmap='gray', aspect='auto', vmin=0, vmax=1)
        ax_int[0].set_title(f'GT Intensity (Step {step})')
        plt.colorbar(im_gt, ax=ax_int[0], fraction=0.046, pad=0.04)
        
        im_pr = ax_int[1].imshow(pred_int, cmap='gray', aspect='auto', vmin=0, vmax=1)
        ax_int[1].set_title(f'Reconstructed Intensity (Step {step})')
        plt.colorbar(im_pr, ax=ax_int[1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'step{step:07d}_{"vae_rec" if is_vae_mode else "forecast"}_intensity.png'), dpi=100)
        plt.close(fig_int)

    # --- 2. GT vs Predicted range-view + abs error ---
    suffix = "vae_rec" if is_vae_mode else "forecast"
    rv_path = os.path.join(vis_dir, f'step{step:07d}_{suffix}_rv.png')
    render_rangeview_comparison(
        gt_depth=gt_depth,
        pred_depth=pred_depth,
        output_path=rv_path,
        frame_idx=step,
        max_depth=max_depth,
        title_suffix=' [t+1]' if not is_vae_mode else ' [VAE recon]',
        metrics={
            'range_l1': float(np.abs(pred_depth - gt_depth)[gt_depth > 0.5].mean())
            if (gt_depth > 0.5).any() else 0.0
        },
    )

    # --- 3. BEV comparison (GT vs predicted point cloud) ---
    try:
        pts_gt   = projector.back_project_range(np.clip(gt_depth,   0, max_depth))
        pts_pred = projector.back_project_range(np.clip(pred_depth, 0, max_depth))
        bev_path = os.path.join(vis_dir, f'step{step:07d}_{suffix}_bev.png')
        bev_range = float(getattr(args, 'bev_range', 50.0))
        render_bev_comparison(
            points_gt=pts_gt, points_pred=pts_pred,
            bev_range=bev_range, resolution=0.2,
            output_path=bev_path, frame_idx=step,
        )
    except Exception:
        pass   # BEV is optional; don't crash training if back-projection fails


def save_multistep_visualization(step, args, pred_frames, gt_frames, projector, vis_dir):
    """Save two images for multi-step forecast quality:

    1. ``_multistep_rv.png``  — range-view grid:
       rows = future frames (+1 … +N), cols = [GT depth | Pred depth | |Error|]
       Row labels include per-step MAE (metres, over valid pixels).

    2. ``_multistep_bev.png`` — BEV grid:
       rows = future frames (+1 … +N), cols = [GT BEV | Pred BEV | Overlay (G=GT, R=Pred)]
       Row labels include per-step symmetric Chamfer distance (metres).

    Args:
        step:        Training step.
        args:        Config namespace.
        pred_frames: List of N tensors, each [B, C, H, W] — predicted frame at chain step j.
        gt_frames:   List of N tensors, each [B, C, H, W] — GT frame at chain step j.
        projector:   RangeProjection for BEV back-projection.
        vis_dir:     Output directory.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(pred_frames)
    if n == 0:
        return

    os.makedirs(vis_dir, exist_ok=True)
    range_mean = args.proj_img_mean[0]
    range_std  = args.proj_img_stds[0]
    max_depth  = 80.0
    bev_range  = float(getattr(args, 'bev_range', 50.0))

    def to_depth(chw):
        return np.clip(chw[0].float().cpu().numpy() * range_std + range_mean, 0, max_depth)

    def make_bev(depth_hw):
        """Return (occupancy_grid [H,W], point_cloud [N,3])."""
        try:
            pts = projector.back_project_range(depth_hw)
            res  = 0.2
            grid = int(2 * bev_range / res)
            img  = np.zeros((grid, grid), dtype=np.float32)
            xi = ((pts[:, 0] + bev_range) / res).astype(int)
            yi = ((pts[:, 1] + bev_range) / res).astype(int)
            mask = (xi >= 0) & (xi < grid) & (yi >= 0) & (yi < grid)
            img[yi[mask], xi[mask]] = 1.0
            return img, pts
        except Exception:
            return np.zeros((100, 100), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    def chamfer_dist_3d(pts_a, pts_b):
        """Symmetric mean Chamfer distance in metres (via scipy cKDTree)."""
        try:
            from scipy.spatial import cKDTree
            if len(pts_a) == 0 or len(pts_b) == 0:
                return float('nan')
            d_ab = cKDTree(pts_b).query(pts_a, k=1, workers=1)[0]
            d_ba = cKDTree(pts_a).query(pts_b, k=1, workers=1)[0]
            return float(np.mean(d_ab) + np.mean(d_ba))
        except Exception:
            return float('nan')

    # Pre-compute depths, BEVs, and per-step metrics for all frames
    gt_depths, pred_depths = [], []
    gt_bevs,   pred_bevs   = [], []
    maes,      chamfers    = [], []

    for fi in range(n):
        gd = to_depth(gt_frames[fi][0])
        pd = to_depth(pred_frames[fi][0])
        gt_depths.append(gd)
        pred_depths.append(pd)

        gb, g_pts = make_bev(gd)
        pb, p_pts = make_bev(pd)
        gt_bevs.append(gb)
        pred_bevs.append(pb)

        # MAE over valid pixels (GT depth > 0.5 m)
        valid = gd > 0.5
        mae = float(np.abs(pd - gd)[valid].mean()) if valid.any() else float('nan')
        maes.append(mae)

        # Symmetric Chamfer distance in 3-D (metres)
        chamfers.append(chamfer_dist_3d(g_pts, p_pts))

    # ------------------------------------------------------------------ #
    # Image 1: Range-view — rows=frames, cols=[GT | Pred | |Error|]
    # ------------------------------------------------------------------ #
    fig_rv, axes_rv = plt.subplots(n, 3, figsize=(18, 2.8 * n))
    if n == 1:
        axes_rv = axes_rv[np.newaxis, :]

    for c, title in enumerate(['GT depth (m)', 'Pred depth (m)', '|Error| (m)']):
        axes_rv[0, c].set_title(title, fontsize=9, fontweight='bold')

    for fi in range(n):
        err = np.abs(pred_depths[fi] - gt_depths[fi])
        _cmap_rv = matplotlib.colormaps['turbo_r']
        axes_rv[fi, 0].imshow(gt_depths[fi],  cmap=_cmap_rv, vmin=0, vmax=max_depth, aspect='auto')
        axes_rv[fi, 1].imshow(pred_depths[fi], cmap=_cmap_rv, vmin=0, vmax=max_depth, aspect='auto')
        im_err = axes_rv[fi, 2].imshow(err,    cmap='hot',    vmin=0, vmax=10.0,      aspect='auto')
        mae_str = f'{maes[fi]:.3f} m' if not np.isnan(maes[fi]) else 'n/a'
        axes_rv[fi, 0].set_ylabel(f't+{fi+1}\nMAE={mae_str}', fontsize=8, fontweight='bold')
        plt.colorbar(im_err, ax=axes_rv[fi, 2], fraction=0.015, pad=0.02, label='m')
        for ax in axes_rv[fi]:
            ax.set_xticks([]); ax.set_yticks([])

    fig_rv.suptitle(f'Step {step} — {n}-Frame Range-View Forecast', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'step{step:07d}_multistep_rv.png'), dpi=90, bbox_inches='tight')
    plt.close(fig_rv)

    # ------------------------------------------------------------------ #
    # Image 2: BEV — rows=frames, cols=[GT BEV | Pred BEV | Overlay]
    # ------------------------------------------------------------------ #
    fig_bev, axes_bev = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes_bev = axes_bev[np.newaxis, :]

    for c, title in enumerate(['GT BEV', 'Pred BEV', 'Overlay (green=GT, red=Pred)']):
        axes_bev[0, c].set_title(title, fontsize=9, fontweight='bold')

    for fi in range(n):
        gb, pb = gt_bevs[fi], pred_bevs[fi]
        h_b, w_b = gb.shape
        overlay = np.zeros((h_b, w_b, 3), dtype=np.float32)
        overlay[..., 1] = np.clip(gb, 0, 1)   # green = GT
        overlay[..., 0] = np.clip(pb, 0, 1)   # red   = Pred

        axes_bev[fi, 0].imshow(gb,      cmap='gray', aspect='equal', origin='lower')
        axes_bev[fi, 1].imshow(pb,      cmap='gray', aspect='equal', origin='lower')
        axes_bev[fi, 2].imshow(overlay,              aspect='equal', origin='lower')
        cd_str = f'{chamfers[fi]:.3f} m' if not np.isnan(chamfers[fi]) else 'n/a'
        axes_bev[fi, 0].set_ylabel(f't+{fi+1}\nCD={cd_str}', fontsize=8, fontweight='bold')
        for ax in axes_bev[fi]:
            ax.set_xticks([]); ax.set_yticks([])

    fig_bev.suptitle(f'Step {step} — {n}-Frame BEV Forecast', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'step{step:07d}_multistep_bev.png'), dpi=90, bbox_inches='tight')
    plt.close(fig_bev)


def main(args):
    """Main training function"""
    init_environment(args)

    if not args.distributed:
        start_training(0, args)
    else:
        if args.launcher == 'pytorch':
            print('Using PyTorch launcher.')
            local_rank = int(os.environ["LOCAL_RANK"])
            start_training(local_rank, args)
        elif args.launcher == 'slurm':
            # Recommended: invoke this script via `torchrun` inside the SLURM job
            # script, in which case LOCAL_RANK is already set by torchrun.
            # Fallback: single-node SLURM without torchrun uses mp.spawn.
            if 'LOCAL_RANK' in os.environ:
                local_rank = int(os.environ['LOCAL_RANK'])
                start_training(local_rank, args)
            else:
                num_gpus_per_nodes = torch.cuda.device_count()
                mp.spawn(start_training, nprocs=num_gpus_per_nodes, args=(args,))
        else:
            raise RuntimeError(f'Launcher {args.launcher} is not supported.')


def start_training(local_rank, args):
    """Initialize training process"""
    torch.cuda.set_device(local_rank)

    if 'RANK' not in os.environ:
        # torchrun not used: derive rank from SLURM env or assume single-node
        if 'SLURM_PROCID' in os.environ:
            global_rank = int(os.environ['SLURM_PROCID'])
        else:
            node_rank = 0
            global_rank = node_rank * torch.cuda.device_count() + local_rank
        os.environ["RANK"] = str(global_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)

    init_logs(int(os.environ["RANK"]), args)

    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])

    print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) starting training...")
    train(local_rank, args)


def train(local_rank, args):
    """Main training loop"""
    print(f"Training configuration:\n{args}")
    writer = args.writer
    rank = int(os.environ['RANK'])
    save_model_path = args.save_model_path
    step = args.resume_step

    # Create model directly in bf16 to avoid the fp32→bf16 conversion peak
    # (which temporarily holds both copies, doubling CPU RAM usage).
    print("Creating Range View DiT model...")
    torch.set_default_dtype(torch.bfloat16)
    
    stage = getattr(args, 'stage', 'all')
    if stage == '1':
        model = RangeViewVAE(args, local_rank=local_rank)
    else:
        model = RangeViewDiT(
            args,
            local_rank=local_rank,
            condition_frames=args.condition_frames // args.block_size
        )
        
    torch.set_default_dtype(torch.float32)

    # stage-specific parameter freezing
    if stage == '1':
        print("Stage 1 selected: Using RangeViewVAE model for ELBO optimization.")
        # ensure a checkpoint is not provided, otherwise VAE will be frozen
        if getattr(args, 'vae_ckpt', None) is not None:
            print("Warning: vae_ckpt is provided but will be ignored for stage 1 pretraining.")
            args.vae_ckpt = None
    elif stage == '2':
        # DiT/STT training: freeze VAE
        print("Stage 2 selected: freezing VAE (if checkpoint exists) and training DiT/STT.")
        # If no checkpoint supplied we still freeze the VAE parameters so that
        # only DiT/STT update; this mirrors the behaviour of RangeLDM stage-2.
        if hasattr(model, 'vae_tokenizer') and hasattr(model.vae_tokenizer, 'vae'):
            for p in model.vae_tokenizer.vae.parameters():
                p.requires_grad = False
        if getattr(args, 'vae_ckpt', None) is None:
            print("Warning: no vae_ckpt provided for stage 2; the model will start with random VAE weights but they will be frozen.")
    else:
        print("Full training (stage=all): no additional freezing applied.")

    # Count parameters
    if stage == '1':
        total_params = count_parameters(model)
        print(f"Total VAE Parameters: {format_number(total_params)}")
    else:
        total_params = count_parameters(model)
        stt_params = count_parameters(model.model)
        dit_params = count_parameters(model.dit)
        print(f"Total Parameters: {format_number(total_params)}")
        print(f"STT Parameters: {format_number(stt_params)}")
        print(f"DiT Parameters: {format_number(dit_params)}")

    # Calculate effective batch size
    # Stage 1 processes single frames; stages 2/all process condition_frames per sample.
    frames_per_sample = 1 if stage == '1' else args.condition_frames // args.block_size
    eff_batch_size = args.batch_size * frames_per_sample * dist.get_world_size()

    # Set learning rate
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print(f"Base LR: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"Actual LR: {args.lr:.2e}")
    print(f"Effective batch size: {eff_batch_size}")

    # Create optimizer
    param_groups = add_weight_decay(model, args.weight_decay)
    optimizer = FusedAdam(param_groups, lr=args.lr, betas=(0.9, 0.95), adam_w_mode=True)
    print(f"Optimizer: {optimizer}")

    # Learning rate schedule
    # lr_schedule = init_lr_schedule(optimizer, milstones=[1000000, 1500000, 2000000], gamma=0.5)
    lr_schedule = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.iter)

    # Load checkpoint if provided
    if args.resume_path is not None:
        checkpoint = torch.load(args.resume_path, map_location="cpu")
        print(f"Loading model from: {args.resume_path}")
        model = load_parameters(model, checkpoint)
        del checkpoint

    # Create dataset
    train_dataset = create_rangeview_dataset(args)

    if args.overfit:
        # For debugging: overfit on small subset
        train_dataset = Subset(train_dataset, list(range(200)))

    # Create distributed sampler
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.seed
    )

    # Create dataloader
    train_data = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 8,
        pin_memory=True,
        pin_memory_device='cuda',
        drop_last=True,
        sampler=sampler,
        persistent_workers=True,
        prefetch_factor=2,
    )

    print(f'Training dataloader length: {len(train_data)}')
    epoch = step // len(train_data) + 1

    # Initialize DeepSpeed
    deepspeed_cfg = get_deepspeed_config(args)
    model, optimizer, _, _ = deepspeed.initialize(
        config_params=deepspeed_cfg,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_schedule,
    )
    load_from_deepspeed_ckpt(args, model)

    # ------------------------------------------------------------------ #
    # Stage 1: Discriminator setup (adversarial VAE training)
    # Following RangeLDM: PatchGAN with MetaKernel geometry-aware blocks,
    # delayed start after disc_start steps, hinge GAN loss, fixed disc_lr.
    # ------------------------------------------------------------------ #
    disc = None
    disc_optimizer = None
    if stage == '1' and getattr(args, 'disc_weight', 0.5) > 0:
        disc = NLayerDiscriminatorMetaKernel(
            input_nc=int(getattr(args, 'range_channels', 2)),
            ndf=int(getattr(args, 'disc_ndf', 64)),
            n_layers=int(getattr(args, 'disc_num_layers', 3)),
            range_mean=float(args.proj_img_mean[0]),
            range_std=float(args.proj_img_stds[0]),
        ).apply(disc_weights_init).cuda(local_rank)
        disc = DDP(disc, device_ids=[local_rank], find_unused_parameters=True)
        disc_lr = float(getattr(args, 'disc_lr', 2e-4))
        disc_optimizer = torch.optim.Adam(
            disc.parameters(), lr=disc_lr, betas=(0.5, 0.9)
        )
        # Resume discriminator checkpoint if provided
        disc_resume = getattr(args, 'disc_resume_path', None)
        if disc_resume and os.path.isfile(disc_resume):
            _sd = torch.load(disc_resume, map_location='cpu')
            disc.module.load_state_dict(_sd.get('disc_state_dict', _sd))
            print(f"Loaded discriminator from {disc_resume}")
        print(
            f"Discriminator created: ndf={getattr(args,'disc_ndf',64)}, "
            f"n_layers={getattr(args,'disc_num_layers',3)}, "
            f"disc_start={getattr(args,'disc_start',50000)}, "
            f"disc_weight={getattr(args,'disc_weight',0.5)}, "
            f"disc_lr={disc_lr}"
        )

    torch.set_float32_matmul_precision('high')

    # Projector for training visualizations (back-projects depth → 3D for BEV)
    vis_projector = RangeProjection(
        fov_up=args.fov_up, fov_down=args.fov_down,
        proj_w=args.range_w, proj_h=args.range_h,
        fov_left=args.fov_left, fov_right=args.fov_right,
    )
    vis_dir = os.path.join(args.validation_path, 'train_vis')

    print('Starting training loop...')
    torch.cuda.synchronize()
    time_stamp = time.time()

    last_ckpt_step = None  # tracks the previous checkpoint step for cleanup

    while step < args.iter:
        sampler.set_epoch(epoch)

        for i, (range_views, rot_matrix) in enumerate(train_data):
            model.train()

            range_views = range_views.cuda()  # Stage 1: [B, C, H, W], Stage 2: [B, T, C, H, W]
            rot_matrix  = rot_matrix.cuda()   # Stage 1: [B, 4, 4],    Stage 2: [B, T, 4, 4]

            data_time_interval = time.time() - time_stamp
            torch.cuda.synchronize()
            time_stamp = time.time()

            stage = getattr(args, 'stage', 'all')
            cf = args.condition_frames // args.block_size

            if stage == '1':
                # --- STAGE 1 (VAE + adversarial discriminator) ---
                # Data is already [B, C, H, W] from KITTIRangeViewVAEDataset
                features_gt = range_views

                B, C, H, W = features_gt.shape

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss_final = model(features_gt, step=step)

                loss_value = loss_final["loss_all"]

                if not math.isfinite(loss_value):
                    print(f"Loss is {loss_value}, stopping training")
                    sys.exit(1)

                # --- Generator adversarial loss (active after disc_start) ---
                disc_start_step  = int(getattr(args, 'disc_start',  50000))
                disc_factor_cfg  = float(getattr(args, 'disc_factor',  1.0))
                disc_weight_cfg  = float(getattr(args, 'disc_weight',  0.5))

                # Pinned VAE save: capture ELBO-only weights just before the
                # first adversarial update.  Named vae_pre_disc_step{N}.pth so
                # the rolling checkpoint cleanup never touches it.
                if disc is not None and step == disc_start_step and rank == 0:
                    try:
                        raw_m = model.module if hasattr(model, 'module') else model
                        vae_state = raw_m.vae_tokenizer.vae.state_dict()
                        pinned_path = os.path.join(
                            save_model_path, f"vae_pre_disc_step{step}.pth"
                        )
                        torch.save(vae_state, pinned_path)
                        print(f"Saved pinned VAE checkpoint (pre-discriminator best ELBO): {pinned_path}")
                    except Exception as e:
                        print(f"Warning: failed to save pinned VAE checkpoint: {e}")

                disc_is_active   = disc is not None and step >= disc_start_step
                g_loss_val = 0.0

                if disc_is_active:
                    disc.train()
                    x_recon = loss_final['x_recon']   # [B, C, H, W], grad retained
                    # Freeze disc during generator backward so DDP allreduce hooks
                    # do not modify disc parameter tensors in-place, which would
                    # corrupt the computation graph for the discriminator update.
                    disc.requires_grad_(False)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits_fake_g = disc(x_recon)
                    g_loss = -torch.mean(logits_fake_g)
                    g_loss_val = float(g_loss.item())

                    # Adaptive weight: balance NLL and GAN gradients at last
                    # decoder layer (falls back to fixed weight on RuntimeError/IndexError).
                    raw_model = model.module if hasattr(model, 'module') else model
                    last_layer = raw_model.vae_tokenizer.vae.decoder.conv_out.weight
                    d_weight = calculate_adaptive_weight(
                        loss_final['nll_loss'], g_loss, last_layer, disc_weight_cfg
                    )
                    disc_factor_val = disc_adopt_weight(
                        disc_factor_cfg, step, threshold=disc_start_step
                    )
                    loss_value = loss_value + d_weight * disc_factor_val * g_loss

                model.backward(loss_value)
                model.step()

                # --- Discriminator update (after generator backward) ---
                d_loss_val = 0.0
                if disc_is_active:
                    # Re-enable disc gradients for discriminator update
                    disc.requires_grad_(True)
                    disc_optimizer.zero_grad()
                    x_recon_d = loss_final['x_recon'].detach()
                    # Single forward pass over real+fake concatenated to avoid
                    # DDP modifying gradient buffers in-place between two
                    # consecutive forwards, which causes the autograd version
                    # mismatch error and prevents d_loss.backward().
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        # Single forward pass over real+fake to avoid DDP inplace
                        # buffer modifications between two consecutive forwards.
                        logits_all = disc(torch.cat([features_gt.detach(), x_recon_d], dim=0))
                    logits_real, logits_fake_d = torch.chunk(logits_all, 2, dim=0)
                    disc_factor_val = disc_adopt_weight(
                        disc_factor_cfg, step, threshold=disc_start_step
                    )
                    d_loss = disc_factor_val * hinge_d_loss(logits_real, logits_fake_d)
                    d_loss_val = float(d_loss.item())
                    d_loss.backward()
                    disc_optimizer.step()
                
            else:
                # --- STAGE 2 / ALL (Diffusion Forecasting) ---
                
                # Data is [B, T, C, H, W]
                B, T, C, H, W = range_views.shape
                
                features_cond = range_views[:, :cf, ...]   # [B, CF, C, H, W]
                rel_pose_cond, rel_yaw_cond = None, None
                # Predicted latents from the previous AR step, used as direct
                # conditioning for the next step (mirrors train_deepspeed.py which
                # keeps latents_cond in latent space throughout the chain).
                # None → model encodes features_cond normally (first step only).
                latents_cond_next = None
                # AR rotation window: tracks predicted absolute poses for the CF
                # conditioning frames so rel_pose_cond is derived from PoseDiT
                # predictions rather than GT after j=0.
                ar_rot_window = rot_matrix[:, :cf].clone()   # [B, CF, 4, 4]

                # Forward iterations
                fw_iter = 1
                if step % args.multifw_perstep == 0:
                    fw_iter = args.forward_iter

                # Collect chain predictions for multi-step visualization (rank 0, vis steps only).
                vis_steps = getattr(args, 'vis_steps', 500)
                collect_chain_vis = (
                    vis_steps > 0 and step % vis_steps == 0 and rank == 0
                )
                chain_pred_frames  = []   # [B, C, H, W] per chain step
                chain_gt_frames    = []   # [B, C, H, W] per chain step
                chain_pred_latents = []   # [B, L, C]    per chain step (normalised)
                # Cumulative losses over fw_iter AR steps (averaged for logging only;
                # backward still fires individually per j, matching train_deepspeed.py).
                cumul_diff = cumul_pose = cumul_cd = cumul_repa = 0.0

                # Number of rotation matrices needed per forward pass:
                # (condition_frames + 1) * block_size
                n_rot = (args.condition_frames + 1) * args.block_size
                for j in range(fw_iter):
                    rot_matrix_cond = rot_matrix[
                        :, j * args.block_size:j * args.block_size + n_rot, ...
                    ]
                    features_gt = range_views[:, j + cf:j + cf + 1, ...]  # [B, 1, C, H, W]

                    # Forward pass with mixed precision.
                    # latents_cond_next is None for j=0 (model encodes features_cond),
                    # and [B, CF, L, latent_C] for j>0 (predicted latents passed
                    # directly, skipping the decode → re-encode round trip).
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss_final = model(
                            features_cond,
                            rot_matrix_cond,
                            features_gt,
                            rel_pose_cond=rel_pose_cond,
                            rel_yaw_cond=rel_yaw_cond,
                            step=step,
                            latents_cond_precomputed=latents_cond_next,
                        )

                    # Rebuild total loss with REPA warmup so the cosine-alignment
                    # term ramps up gradually rather than dominating from step 0.
                    # loss_repa in the dict is the raw unweighted value; repa_weight
                    # (from config) is the target scale at full ramp.
                    _repa_ramp  = min(1.0, float(step) /
                                      max(float(getattr(args, 'repa_warmup_steps', 5000)), 1))
                    _repa_w     = float(getattr(args, 'repa_weight', 0.0))
                    _chamfer_w  = float(getattr(args, 'chamfer_loss_weight', 0.0))
                    _loss_repa  = loss_final.get('loss_repa',
                                                 loss_final['loss_diff'].new_tensor(0.))
                    _loss_cd    = loss_final.get('loss_chamfer',
                                                 loss_final['loss_diff'].new_tensor(0.))
                    loss_value  = (loss_final['loss_diff']
                                   + loss_final['loss_pose']
                                   + _chamfer_w * _loss_cd
                                   + _repa_w * _repa_ramp * _loss_repa)

                    # Range-view pixel-space L1 on decoded DiT prediction.
                    # Controlled by rv_pred_weight in config (default 0 = off).
                    # Provides direct depth/intensity supervision in image space
                    # to complement the latent-space diffusion loss.
                    _rv_pred_w = float(getattr(args, 'rv_pred_weight', 0.0))
                    if _rv_pred_w > 0 and loss_final.get('predict') is not None:
                        _rv_loss = torch.nn.functional.l1_loss(
                            loss_final['predict'].float(),
                            features_gt[:, 0].float(),
                        )
                        loss_value = loss_value + _rv_pred_w * _rv_loss
                        loss_final['loss_rv_pred'] = _rv_loss.detach()

                    # ── Physical-unit pose regression ─────────────────────────
                    # Prevents PoseDiT from collapsing to mean KITTI velocity.
                    # L1 in metres/degrees gives STT a strong pose-discriminative
                    # signal. Controlled by pose_reg_weight in config (default 0).
                    _pose_reg_w = float(getattr(args, 'pose_reg_weight', 0.0))
                    if _pose_reg_w > 0:
                        pred_xy  = loss_final.get('predict_pose_xy')   # [B, 1, 2] metres
                        pred_yaw = loss_final.get('predict_pose_yaw')  # [B, 1, 1] degrees
                        if pred_xy is not None and pred_yaw is not None:
                            with torch.cuda.amp.autocast(enabled=False):
                                _rp, _ry = get_rel_pose(rot_matrix_cond.float())
                            gt_xy  = _rp[:, -1:].float()   # [B, 1, 2]
                            gt_yaw = _ry[:, -1:].float()   # [B, 1, 1]
                            _pose_reg = (
                                torch.nn.functional.l1_loss(pred_xy.float(),  gt_xy)
                                + torch.nn.functional.l1_loss(pred_yaw.float(), gt_yaw)
                            )
                            loss_value = loss_value + _pose_reg_w * _pose_reg
                            loss_final['loss_pose_reg'] = _pose_reg.detach()

                    # Check for NaN
                    if not math.isfinite(loss_value.item()):
                        print(f"Loss is {loss_value.item()}, stopping training")
                        sys.exit(1)

                    # Backward and optimize
                    model.backward(loss_value)
                    model.step()

                    # Accumulate for averaged logging (no effect on gradients).
                    cumul_diff  += loss_final["loss_diff"].item()
                    cumul_pose  += loss_final["loss_pose"].item()
                    cumul_cd    += loss_final.get("loss_chamfer", torch.tensor(0.)).item()
                    # Log effective REPA contribution (weighted + ramped)
                    cumul_repa  += (_repa_w * _repa_ramp * _loss_repa).item()

                    # Collect this step's prediction for multi-step visualization.
                    # predict is [B, C, H, W] — one frame per AR step.
                    # In stage 2 all aux losses are disabled so predict_decoded is None,
                    # but predict_latents (DiT x_0 estimate) is still returned.
                    # Decode locally on vis steps so visualizations are always saved.
                    _vis_pred = loss_final["predict"]
                    if collect_chain_vis and _vis_pred is None and loss_final.get("predict_latents") is not None:
                        with torch.no_grad():
                            _raw_m = model.module if hasattr(model, 'module') else model
                            _pred_lat = loss_final["predict_latents"].detach()
                            _pred_lat = _pred_lat * _raw_m.latent_scale.to(_pred_lat.dtype)
                            _vis_pred = _raw_m.vae_tokenizer.decode_from_z(_pred_lat, _raw_m.h, _raw_m.w)
                    if collect_chain_vis and _vis_pred is not None:
                        chain_pred_frames.append(_vis_pred.detach())  # [B, C, H, W]
                        chain_gt_frames.append(features_gt[:, 0, ...].detach())
                    if collect_chain_vis and loss_final.get("predict_latents") is not None:
                        chain_pred_latents.append(loss_final["predict_latents"].detach())  # [B, L, C]

                    # Prepare for next autoregressive iteration (sliding window).
                    if j < fw_iter - 1:
                        # ── Slide AR rotation window + update pose conditioning ──
                        # Compose predicted absolute rotation for the new frame,
                        # slide ar_rot_window forward, then derive all conditioning
                        # relative poses from accumulated predicted rotations.
                        pred_pose_xy  = loss_final.get('predict_pose_xy')   # [B, 1, 2]
                        pred_pose_yaw = loss_final.get('predict_pose_yaw')  # [B, 1, 1]
                        if pred_pose_xy is not None and pred_pose_yaw is not None:
                            with torch.no_grad():
                                pred_abs_rot = _compose_predicted_rot(
                                    ar_rot_window[:, -1],
                                    pred_pose_xy[:, 0].float(),
                                    pred_pose_yaw[:, 0].float(),
                                )   # [B, 4, 4]
                                ar_rot_window = torch.cat(
                                    [ar_rot_window[:, 1:],
                                     pred_abs_rot.unsqueeze(1)], dim=1
                                )   # [B, CF, 4, 4]
                                with torch.cuda.amp.autocast(enabled=False):
                                    rel_pose_cond, rel_yaw_cond = get_rel_pose(
                                        ar_rot_window.float()
                                    )   # [B, CF, 2], [B, CF, 1]

                        if args.return_predict and loss_final.get("predict_latents") is not None:
                            # -------------------------------------------------- #
                            # Chain-of-forwarding sliding-window update.
                            # predict_latents is [B, L, C] — single frame.
                            # Drop oldest cond frame, append new prediction.
                            #
                            # j=0: slide_from = freshly encoded cond latents
                            # j>0: slide_from = latents_cond_next from prev step
                            # -------------------------------------------------- #
                            pred_lat_single = loss_final["predict_latents"].detach()  # [B, L, C]
                            if latents_cond_next is None:
                                slide_from = loss_final.get("latents_cond_enc")  # [B, CF, L, C]
                            else:
                                slide_from = latents_cond_next                   # [B, CF, L, C]
                            if slide_from is not None:
                                latents_cond_next = torch.cat([
                                    slide_from[:, 1:, :, :],           # [B, CF-1, L, C]
                                    pred_lat_single.unsqueeze(1),      # [B, 1,    L, C]
                                ], dim=1)                               # [B, CF,   L, C]
                            else:
                                latents_cond_next = None
                            # Update pixel-space cond for visualization / fallback.
                            if loss_final["predict"] is not None:
                                features_cond = torch.cat([
                                    features_cond[:, 1:, ...],                    # [B, CF-1, C, H, W]
                                    loss_final["predict"].detach().unsqueeze(1),  # [B, 1,    C, H, W]
                                ], dim=1)                                          # [B, CF,   C, H, W]
                        else:
                            # Full diffusion sampling fallback (slower, no predict_latents).
                            latents_cond_next = None
                            model.eval()
                            with torch.no_grad():
                                predict_features = model(
                                    features_cond,
                                    rot_matrix_cond,
                                    features_gt,
                                    sample_last=False,
                                )
                            model.train()
                            # predict_features: [B, C, H, W] from step_eval — slide by 1
                            features_cond = torch.cat([
                                features_cond[:, 1:, ...],
                                predict_features.unsqueeze(1),
                            ], dim=1)

                # Multi-step visualization after the chain (rank 0, vis steps, chain steps only).
                if collect_chain_vis and chain_pred_frames:
                    vis_pred_frames = chain_pred_frames  # default: per-frame decoded

                    # If the temporal decoder is enabled and we collected chain latents,
                    # re-decode the full AR chain jointly with temporal attention so that
                    # each visualised frame benefits from global AR-chain context.
                    if chain_pred_latents and len(chain_pred_latents) == len(chain_pred_frames):
                        raw_m = model.module if hasattr(model, 'module') else model
                        tokenizer = raw_m.vae_tokenizer
                        if getattr(tokenizer, 'decoder_temporal', None) is not None:
                            with torch.no_grad():
                                lat_seq = torch.stack(chain_pred_latents, dim=1)  # [B, T, L, C]
                                scale   = raw_m.latent_scale.to(lat_seq.dtype)
                                decoded_seq = tokenizer.decode_from_z_temporal(
                                    lat_seq * scale,   # denormalise before VAE decode
                                    raw_m.h, raw_m.w,
                                )  # [B, T, 2, H, W]
                            vis_pred_frames = [
                                decoded_seq[:, j] for j in range(decoded_seq.shape[1])
                            ]

                    with torch.no_grad():
                        save_multistep_visualization(
                            step=step,
                            args=args,
                            pred_frames=vis_pred_frames,
                            gt_frames=chain_gt_frames,
                            projector=vis_projector,
                            vis_dir=vis_dir,
                        )

            step += 1
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            dist.barrier()
            torch.cuda.synchronize()

            # Logging
            if rank == 0:
                # Fetch loss values for logging every step
                current_lr = optimizer.param_groups[0]['lr']
                # log values depending on stage
                stage = getattr(args, 'stage', 'all')

                def get_loss_val(key):
                    val = loss_final.get(key, 0.0)
                    return val.item() if isinstance(val, torch.Tensor) else val

                if stage == '1':
                    loss_all_val  = loss_final["loss_all"].item()
                    loss_elbo_val = get_loss_val("loss_elbo")
                    loss_bev_val  = get_loss_val("loss_bev_percep")
                    loss_diff_val = loss_pose_val = loss_rl1_val = loss_cd_val = loss_repa_val = 0.0
                    # discriminator losses (defined in Stage 1 block above,
                    # default to 0.0 if disc is not yet active)
                    try:
                        _g_loss_val = g_loss_val
                        _d_loss_val = d_loss_val
                    except NameError:
                        _g_loss_val = _d_loss_val = 0.0
                else:
                    loss_diff_val  = cumul_diff  / fw_iter
                    loss_pose_val  = cumul_pose  / fw_iter
                    loss_cd_val    = cumul_cd    / fw_iter
                    # cumul_repa already holds the weighted+ramped effective value
                    loss_repa_val  = cumul_repa  / fw_iter
                    loss_all_val   = loss_diff_val + loss_pose_val + loss_cd_val + loss_repa_val
                    loss_rl1_val   = get_loss_val("loss_range_l1")
                    loss_elbo_val  = get_loss_val("loss_elbo")
                    loss_bev_val   = get_loss_val("loss_bev_percep")
                    loss_pose_reg_val = get_loss_val("loss_pose_reg")

                # Log to console every 50 steps
                epoch = step // len(train_data) + 1
                if step % 50 == 0:
                    if stage == '1':
                        disc_suffix = (
                            f' g_loss:{_g_loss_val:.4e} d_loss:{_d_loss_val:.4e}'
                            if disc is not None else ''
                        )
                        logger.info(
                            f'stage:{stage} epoch:{epoch} step:{step} '
                            f'time:{data_time_interval:.2f}+{train_time_interval:.2f} '
                            f'lr:{current_lr:.4e} '
                            f'loss_avg:{loss_all_val:.4e} '
                            f'elbo:{loss_elbo_val:.4e} '
                            f'bev:{loss_bev_val:.4e}'
                            + disc_suffix
                        )
                    else:
                        logger.info(
                            f'stage:{stage} epoch:{epoch} step:{step} '
                            f'time:{data_time_interval:.2f}+{train_time_interval:.2f} '
                            f'lr:{current_lr:.4e} '
                            f'loss_avg:{loss_all_val:.4e} '
                            f'diff_loss:{loss_diff_val:.4e} '
                            f'pose_loss:{loss_pose_val:.4e} '
                            f'pose_reg:{loss_pose_reg_val:.4e} '
                            f'range_l1:{loss_rl1_val:.4e} '
                            f'chamfer:{loss_cd_val:.4e} '
                            f'repa:{loss_repa_val:.4e} '
                            f'elbo:{loss_elbo_val:.4e} '
                            f'bev:{loss_bev_val:.4e}'
                        )

                # Log to TensorBoard/W&B less frequently
                if step % 100 == 1:
                    # TensorBoard
                    writer.add_scalar('learning_rate/lr',    current_lr,     step)
                    writer.add_scalar('loss/loss_all',       loss_all_val,   step)
                    writer.add_scalar('loss/loss_diff',      loss_diff_val,  step)
                    writer.add_scalar('loss/loss_pose',      loss_pose_val,  step)
                    writer.add_scalar('loss/loss_range_l1',  loss_rl1_val,   step)
                    writer.add_scalar('loss/loss_chamfer',   loss_cd_val,    step)
                    writer.add_scalar('loss/loss_repa',      loss_repa_val,  step)
                    writer.add_scalar('loss/loss_elbo',      loss_elbo_val,  step)
                    writer.add_scalar('loss/loss_bev_percep', loss_bev_val,  step)
                    if stage == '1' and disc is not None:
                        try:
                            writer.add_scalar('loss/g_loss', _g_loss_val, step)
                            writer.add_scalar('loss/d_loss', _d_loss_val, step)
                        except NameError:
                            pass
                    writer.flush()

                    # Weights & Biases
                    if not getattr(args, 'no_wandb', False) and wandb.run is not None:
                        wb_log = {
                            'loss/loss_all':        loss_all_val,
                            'loss/loss_diff':       loss_diff_val,
                            'loss/loss_pose':       loss_pose_val,
                            'loss/loss_range_l1':   loss_rl1_val,
                            'loss/loss_chamfer':    loss_cd_val,
                            'loss/loss_repa':       loss_repa_val,
                            'loss/loss_elbo':       loss_elbo_val,
                            'loss/loss_bev_percep': loss_bev_val,
                            'learning_rate/lr':     current_lr,
                        }
                        if stage == '1' and disc is not None:
                            try:
                                wb_log['loss/g_loss'] = _g_loss_val
                                wb_log['loss/d_loss'] = _d_loss_val
                            except NameError:
                                pass
                        wandb.log(wb_log, step=step)

            # Training visualizations (rank 0 only)
            # Stage 2/all: multi-step visualization is handled inside the AR loop above.
            # Stage 1: single-frame VAE reconstruction visualization saved here.
            vis_steps = getattr(args, 'vis_steps', 500)
            if stage == '1' and vis_steps > 0 and step % vis_steps == 0 and rank == 0:
                predict_vis = loss_final.get("predict")

                if predict_vis is None:
                    with torch.no_grad():
                        raw_model = model.module if hasattr(model, 'module') else model
                        if hasattr(raw_model, 'vae_tokenizer'):
                            features_gt_seq = features_gt.unsqueeze(1)  # [B, 1, C, H, W]
                            z_seq = raw_model.vae_tokenizer.encode_to_z(features_gt_seq)  # [B, 1, L, C]
                            z_flat = rearrange(z_seq, 'b t l c -> (b t) l c')
                            patch_h = int(getattr(args, 'patch_size_h', args.patch_size))
                            patch_w = int(getattr(args, 'patch_size_w', args.patch_size))
                            h_lat = args.range_h // (args.downsample_size * patch_h)
                            w_lat = args.range_w // (args.downsample_size * patch_w)
                            predict_vis = raw_model.vae_tokenizer.decode_from_z(z_flat, h_lat, w_lat)

                with torch.no_grad():
                    save_training_visualization(
                        step=step,
                        args=args,
                        features_cond=None,
                        features_gt=features_gt.detach(),
                        predict_decoded=predict_vis.detach() if predict_vis is not None else None,
                        projector=vis_projector,
                        vis_dir=vis_dir,
                    )

            # Save checkpoint
            if step % args.eval_steps == 0:
                dist.barrier()
                torch.cuda.synchronize()
                save_ckpt_deepspeed(args, save_model_path, model, optimizer, lr_schedule, step)
                dist.barrier()
                if rank == 0:
                    save_ckpt(args, save_model_path, model.module, optimizer, lr_schedule, step)
                    # Stage 1: save standalone VAE checkpoint (for stage 2 --vae_ckpt)
                    if getattr(args, 'stage', 'all') == '1':
                        try:
                            vae_state = model.module.vae_tokenizer.vae.state_dict()
                            vae_path = os.path.join(save_model_path, f"vae_stage1_step{step}.pth")
                            torch.save(vae_state, vae_path)
                            print(f"Saved standalone VAE checkpoint: {vae_path}")
                        except Exception as e:
                            print(f"Warning: failed to save stage1 VAE checkpoint: {e}")
                        # Save discriminator checkpoint
                        if disc is not None:
                            try:
                                disc_path = os.path.join(save_model_path, f"disc_step{step}.pth")
                                torch.save({'disc_state_dict': disc.module.state_dict()}, disc_path)
                                print(f"Saved discriminator checkpoint: {disc_path}")
                            except Exception as e:
                                print(f"Warning: failed to save discriminator checkpoint: {e}")

                    # Delete previous checkpoint files to save disk space
                    if last_ckpt_step is not None:
                        old_ds_dir = os.path.join(save_model_path, str(last_ckpt_step))
                        if os.path.isdir(old_ds_dir):
                            shutil.rmtree(old_ds_dir)
                            print(f"Deleted old DeepSpeed checkpoint: {old_ds_dir}")
                        old_pkl = os.path.join(save_model_path, f"tvar_{last_ckpt_step}.pkl")
                        if os.path.isfile(old_pkl):
                            os.remove(old_pkl)
                            print(f"Deleted old checkpoint: {old_pkl}")
                        old_vae = os.path.join(save_model_path, f"vae_stage1_step{last_ckpt_step}.pth")
                        if os.path.isfile(old_vae):
                            os.remove(old_vae)
                            print(f"Deleted old VAE checkpoint: {old_vae}")
                        old_disc = os.path.join(save_model_path, f"disc_step{last_ckpt_step}.pth")
                        if os.path.isfile(old_disc):
                            os.remove(old_disc)
                            print(f"Deleted old discriminator checkpoint: {old_disc}")

                last_ckpt_step = step
                torch.cuda.synchronize()
                dist.barrier()

        epoch += 1
        dist.barrier()

    # Clean up W&B run on rank 0 when training finishes normally
    if rank == 0 and not getattr(args, 'no_wandb', False) and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    os.chdir(root_path)
    args = add_arguments()
    main(args)
