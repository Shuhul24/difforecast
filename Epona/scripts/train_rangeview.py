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
import torch.distributed as dist
import torch.utils
from torch.utils.data import DataLoader, Subset, DistributedSampler
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import deepspeed

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
from utils.comm import _init_dist_envi
from utils.running import init_lr_schedule, save_ckpt, load_parameters, add_weight_decay, save_ckpt_deepspeed, load_from_deepspeed_ckpt
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger('base')


def add_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Range View DiT Model')
    parser.add_argument('--iter', default=60000000, type=int, help='Total training iterations')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size per GPU')
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
            num_gpus_per_nodes = torch.cuda.device_count()
            mp.spawn(start_training, nprocs=num_gpus_per_nodes, args=(args,))
        else:
            raise RuntimeError(f'Launcher {args.launcher} is not supported.')


def start_training(local_rank, args):
    """Initialize training process"""
    torch.cuda.set_device(local_rank)

    if 'RANK' not in os.environ:
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

    # Create model
    print("Creating Range View DiT model...")
    model = RangeViewDiT(
        args,
        local_rank=local_rank,
        condition_frames=args.condition_frames // args.block_size
    )

    # Count parameters
    total_params = count_parameters(model)
    stt_params = count_parameters(model.model)
    dit_params = count_parameters(model.dit)
    print(f"Total Parameters: {format_number(total_params)}")
    print(f"STT Parameters: {format_number(stt_params)}")
    print(f"DiT Parameters: {format_number(dit_params)}")

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Calculate effective batch size
    eff_batch_size = args.batch_size * args.condition_frames // args.block_size * dist.get_world_size()

    # Set learning rate
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print(f"Base LR: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"Actual LR: {args.lr:.2e}")
    print(f"Effective batch size: {eff_batch_size}")

    # Create optimizer
    param_groups = add_weight_decay(model.module, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(f"Optimizer: {optimizer}")

    # Learning rate schedule
    lr_schedule = init_lr_schedule(optimizer, milstones=[1000000, 1500000, 2000000], gamma=0.5)

    # Load checkpoint if provided
    if args.resume_path is not None:
        checkpoint = torch.load(args.resume_path, map_location="cpu")
        print(f"Loading model from: {args.resume_path}")
        model.module = load_parameters(model.module, checkpoint)
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

    print('Starting training loop...')
    torch.cuda.synchronize()
    time_stamp = time.time()

    while step < args.iter:
        sampler.set_epoch(epoch)

        for i, (range_views, rot_matrix) in enumerate(train_data):
            model.train()

            # Move data to GPU
            range_views = range_views.cuda()  # [B, T, C, H, W]
            rot_matrix = rot_matrix.cuda()    # [B, T, 4, 4]

            # Reshape range views to [B, T, L, C] where L = H*W
            B, T, C, H, W = range_views.shape
            range_views = rearrange(range_views, 'b t c h w -> b t (h w) c')

            data_time_interval = time.time() - time_stamp
            torch.cuda.synchronize()
            time_stamp = time.time()

            cf = args.condition_frames // args.block_size
            features_cond = range_views[:, :cf, ...]  # [B, CF, L, C]
            rel_pose_cond, rel_yaw_cond = None, None

            # Forward iterations
            fw_iter = 1
            if step % args.multifw_perstep == 0:
                fw_iter = args.forward_iter

            for j in range(fw_iter):
                rot_matrix_cond = rot_matrix[:, j * args.block_size:j * args.block_size + args.condition_frames, ...]
                features_gt = range_views[:, j + cf:j + cf + 1, ...]

                # Forward pass with mixed precision
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss_final = model(
                        features_cond,
                        rot_matrix_cond,
                        features_gt,
                        rel_pose_cond=rel_pose_cond,
                        rel_yaw_cond=rel_yaw_cond,
                        step=step
                    )

                loss_value = loss_final["loss_all"]

                # Check for NaN
                if not math.isfinite(loss_value):
                    print(f"Loss is {loss_value}, stopping training")
                    sys.exit(1)

                # Backward and optimize
                model.backward(loss_value)
                model.step()

                # Prepare for next iteration (autoregressive)
                if j < fw_iter - 1:
                    if args.return_predict:
                        predict_features = loss_final["predict"].detach()
                    else:
                        model.eval()
                        predict_features = model(
                            features_cond,
                            rot_matrix_cond,
                            features_gt,
                            sample_last=False
                        )
                        model.train()

                    features_cond = rearrange(
                        predict_features, '(b t) l c -> b t l c',
                        b=args.batch_size, t=args.condition_frames // args.block_size
                    )

            step += 1
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            dist.barrier()
            torch.cuda.synchronize()

            # Logging
            if step % 100 == 1 and rank == 0:
                writer.add_scalar('learning_rate/lr', optimizer.param_groups[0]['lr'], step)
                writer.add_scalar('loss/loss_all', loss_final["loss_all"].to(torch.float32), step)
                writer.add_scalar('loss/loss_diff', loss_final["loss_diff"].to(torch.float32), step)
                writer.flush()

            if rank == 0:
                logger.info(
                    f'step:{step} time:{data_time_interval:.2f}+{train_time_interval:.2f} '
                    f'lr:{optimizer.param_groups[0]["lr"]:.4e} '
                    f'loss_avg:{loss_final["loss_all"].to(torch.float32):.4e} '
                    f'diff_loss:{loss_final["loss_diff"].to(torch.float32):.4e}'
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


if __name__ == "__main__":
    os.chdir(root_path)
    args = add_arguments()
    main(args)
