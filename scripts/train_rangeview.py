"""
Training script for Range View DiT Model
Simplified training without trajectory prediction
"""

import os
import sys
import math
import time
import torch
import random
import logging
import argparse
from einops import rearrange
import numpy as np
from deepspeed.ops.adam import DeepSpeedCPUAdam
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
from models.model_rangeview import RangeViewDiT
from dataset.dataset_kitti_rangeview import KITTIRangeViewTrainDataset, KITTIRangeViewValDataset
from dataset.projection import RangeProjection
from utils.comm import _init_dist_envi
from utils.running import init_lr_schedule, save_ckpt, load_parameters, add_weight_decay, save_ckpt_deepspeed, load_from_deepspeed_ckpt
from utils.bev_utils import render_bev_comparison, render_rangeview_comparison
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger('base')


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
    parser.add_argument('--vis_steps', type=int, default=500,
                        help='Save training visualizations every N steps (0 = disabled)')

    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.__dict__)
    return cfg


def init_logs(global_rank, args):
    """Initialize logging directories and loggers"""
    print('Initializing logs...')
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
        setup_logger('base', log_path, 'train', level=logging.INFO, screen=True, to_file=True)

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

    # Training dataset
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

    os.makedirs(vis_dir, exist_ok=True)

    range_mean = args.proj_img_mean[0]
    range_std  = args.proj_img_stds[0]

    # Work with the first batch sample only, move to CPU numpy
    # features_cond: [B, CF, C, H, W] — take b=0
    cond_np  = features_cond[0].float().cpu().numpy()   # [CF, C, H, W]
    gt_np    = features_gt[0, 0].float().cpu().numpy()  # [C, H, W]

    # predict_decoded may be [(B*CF), C, H, W] (multifw) or [B, C, H, W]
    # We want the prediction corresponding to the first batch item's last forward step.
    # predict_decoded is [(B*CF), C, H, W].  For batch item 0, the prediction
    # matching features_gt (the frame immediately after the CF conditioning frames)
    # is at index CF-1 (the last target for the first batch item).
    cf = cond_np.shape[0]
    pred_np  = predict_decoded[cf - 1].float().cpu().numpy()  # [C, H, W]

    # --- unnormalise depth channel (ch 0) to metres ---
    def to_depth(feat_chw):
        return feat_chw[0] * range_std + range_mean   # (H, W)

    gt_depth   = to_depth(gt_np)
    pred_depth = to_depth(pred_np)

    # --- 1. Conditioning frames — saved as a single wide range-view strip ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    CF = cond_np.shape[0]
    max_depth = 80.0
    fig, axes = plt.subplots(1, CF, figsize=(6 * CF, 4))
    if CF == 1:
        axes = [axes]
    for t, ax in enumerate(axes):
        depth_t = to_depth(cond_np[t])
        im = ax.imshow(np.clip(depth_t, 0, max_depth), cmap='plasma',
                       vmin=0, vmax=max_depth, aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='m')
        ax.set_title(f'Cond frame t-{CF - t}', fontsize=10)
        ax.set_xlabel('azimuth'); ax.set_ylabel('elevation')
    fig.suptitle(f'Step {step} — Conditioning frames (depth)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'step{step:07d}_cond.png'), dpi=80, bbox_inches='tight')
    plt.close(fig)

    # --- 2. GT vs Predicted range-view + abs error ---
    rv_path = os.path.join(vis_dir, f'step{step:07d}_rv.png')
    render_rangeview_comparison(
        gt_depth=gt_depth,
        pred_depth=pred_depth,
        output_path=rv_path,
        frame_idx=step,
        max_depth=max_depth,
        metrics={
            'range_l1': float(np.abs(pred_depth - gt_depth)[gt_depth > 0.5].mean())
            if (gt_depth > 0.5).any() else 0.0
        },
    )

    # --- 3. BEV comparison (GT vs predicted point cloud) ---
    try:
        pts_gt   = projector.back_project_range(np.clip(gt_depth,   0, max_depth))
        pts_pred = projector.back_project_range(np.clip(pred_depth, 0, max_depth))
        bev_path = os.path.join(vis_dir, f'step{step:07d}_bev.png')
        bev_range = float(getattr(args, 'bev_range', 50.0))
        render_bev_comparison(
            points_gt=pts_gt, points_pred=pts_pred,
            bev_range=bev_range, resolution=0.2,
            output_path=bev_path, frame_idx=step,
        )
    except Exception:
        pass   # BEV is optional; don't crash training if back-projection fails


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
    model = RangeViewDiT(
        args,
        local_rank=local_rank,
        condition_frames=args.condition_frames // args.block_size
    )
    torch.set_default_dtype(torch.float32)

    # Count parameters
    total_params = count_parameters(model)
    stt_params = count_parameters(model.model)
    dit_params = count_parameters(model.dit)
    print(f"Total Parameters: {format_number(total_params)}")
    print(f"STT Parameters: {format_number(stt_params)}")
    print(f"DiT Parameters: {format_number(dit_params)}")

    # Calculate effective batch size
    eff_batch_size = args.batch_size * args.condition_frames // args.block_size * dist.get_world_size()

    # Set learning rate
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print(f"Base LR: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"Actual LR: {args.lr:.2e}")
    print(f"Effective batch size: {eff_batch_size}")

    # Create optimizer
    param_groups = add_weight_decay(model, args.weight_decay)
    # DeepSpeedCPUAdam is required for ZeRO Stage 2 + CPU offload:
    # it applies the Adam update directly from bf16 gradients without
    # materialising a full fp32 gradient tensor (~10 GB for 2.53B params),
    # preventing CPU RAM OOM during the optimizer step.
    optimizer = DeepSpeedCPUAdam(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(f"Optimizer: {optimizer}")

    # Learning rate schedule
    lr_schedule = init_lr_schedule(optimizer, milstones=[1000000, 1500000, 2000000], gamma=0.5)

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
        drop_last=True,
        sampler=sampler
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

    while step < args.iter:
        sampler.set_epoch(epoch)

        for i, (range_views, rot_matrix) in enumerate(train_data):
            model.train()

            # Move data to GPU — keep [B, T, C, H, W]; the model's DCAE
            # tokenizer handles the encode/decode internally.
            range_views = range_views.cuda()  # [B, T, C, H, W]
            rot_matrix  = rot_matrix.cuda()   # [B, T, 4, 4]

            B, T, C, H, W = range_views.shape

            data_time_interval = time.time() - time_stamp
            torch.cuda.synchronize()
            time_stamp = time.time()

            cf = args.condition_frames // args.block_size
            features_cond = range_views[:, :cf, ...]   # [B, CF, C, H, W]
            rel_pose_cond, rel_yaw_cond = None, None

            # Forward iterations
            fw_iter = 1
            if step % args.multifw_perstep == 0:
                fw_iter = args.forward_iter

            # Number of rotation matrices needed per forward pass:
            # (condition_frames + 1) * block_size
            n_rot = (args.condition_frames + 1) * args.block_size
            for j in range(fw_iter):
                rot_matrix_cond = rot_matrix[
                    :, j * args.block_size:j * args.block_size + n_rot, ...
                ]
                features_gt = range_views[:, j + cf:j + cf + 1, ...]  # [B, 1, C, H, W]

                # Forward pass with mixed precision
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss_final = model(
                        features_cond,
                        rot_matrix_cond,
                        features_gt,
                        rel_pose_cond=rel_pose_cond,
                        rel_yaw_cond=rel_yaw_cond,
                        step=step,
                    )

                loss_value = loss_final["loss_all"]

                # Check for NaN
                if not math.isfinite(loss_value):
                    print(f"Loss is {loss_value}, stopping training")
                    sys.exit(1)

                # Backward and optimize
                model.backward(loss_value)
                model.step()

                # Prepare for next autoregressive iteration.
                # predict is [(B*CF), C, H, W] decoded range view features.
                if j < fw_iter - 1:
                    if args.return_predict:
                        predict_features = loss_final["predict"].detach()
                    else:
                        model.eval()
                        predict_features = model(
                            features_cond,
                            rot_matrix_cond,
                            features_gt,
                            sample_last=False,
                        )
                        model.train()

                    # Reshape [(B*CF), C, H, W] → [B, CF, C, H, W] for next iter
                    features_cond = rearrange(
                        predict_features, '(b t) c h w -> b t c h w',
                        b=B, t=cf,
                    )

            step += 1
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            dist.barrier()
            torch.cuda.synchronize()

            # Logging
            if step % 100 == 1 and rank == 0:
                current_lr     = optimizer.param_groups[0]['lr']
                loss_all_val   = loss_final["loss_all"].item()
                loss_diff_val  = loss_final["loss_diff"].item()
                loss_rl1_val   = loss_final["loss_range_l1"].item()
                loss_cd_val    = loss_final["loss_chamfer"].item()
                loss_elbo_val  = loss_final["loss_elbo"].item()
                loss_bev_val   = loss_final.get("loss_bev_percep", loss_final["loss_all"].new_zeros(())).item()

                # TensorBoard
                writer.add_scalar('learning_rate/lr',    current_lr,    step)
                writer.add_scalar('loss/loss_all',       loss_all_val,  step)
                writer.add_scalar('loss/loss_diff',      loss_diff_val, step)
                writer.add_scalar('loss/loss_range_l1',  loss_rl1_val,  step)
                writer.add_scalar('loss/loss_chamfer',   loss_cd_val,   step)
                writer.add_scalar('loss/loss_elbo',      loss_elbo_val, step)
                writer.add_scalar('loss/loss_bev_percep', loss_bev_val, step)
                writer.flush()

                # Weights & Biases
                if not getattr(args, 'no_wandb', False) and wandb.run is not None:
                    wandb.log({
                        'loss/loss_all':       loss_all_val,
                        'loss/loss_diff':      loss_diff_val,
                        'loss/loss_range_l1':  loss_rl1_val,
                        'loss/loss_chamfer':   loss_cd_val,
                        'loss/loss_elbo':      loss_elbo_val,
                        'loss/loss_bev_percep': loss_bev_val,
                        'learning_rate/lr':    current_lr,
                    }, step=step)

            if rank == 0:
                logger.info(
                    f'step:{step} time:{data_time_interval:.2f}+{train_time_interval:.2f} '
                    f'lr:{optimizer.param_groups[0]["lr"]:.4e} '
                    f'loss_avg:{loss_all_val:.4e} '
                    f'diff_loss:{loss_diff_val:.4e} '
                    f'range_l1:{loss_rl1_val:.4e} '
                    f'chamfer:{loss_cd_val:.4e} '
                    f'elbo:{loss_elbo_val:.4e} '
                    f'bev:{loss_bev_val:.4e}'
                )

            # Training visualizations (rank 0 only)
            vis_steps = getattr(args, 'vis_steps', 500)
            if vis_steps > 0 and step % vis_steps == 0 and rank == 0:
                predict_vis = loss_final.get("predict")
                with torch.no_grad():
                    save_training_visualization(
                        step=step,
                        args=args,
                        features_cond=features_cond.detach(),
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
