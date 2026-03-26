"""
Training script for DINOv2-RAE Range View Pipeline.

Stage 1  --stage 1
  Train the ViT-XL decoder with frozen DINOv2 encoder (RAE pre-training).
  Loss: per-channel weighted L1 + optional BEV perceptual.
  Save the best checkpoint; set rae_ckpt in the config before Stage 2.

Stage 2  --stage 2
  Train STT + FluxDiT with frozen DINOv2 encoder and ViT-XL decoder.
  Rectified-flow diffusion in DINOv2 latent space [B, T, 256, 384].
  Chain-of-forward autoregressive training (fw_iter AR steps per batch).

Usage:
  # Stage 1
  torchrun --nproc_per_node=1 scripts/train_rae_rangeview.py \\
      --stage 1 --batch_size 4 --exp_name rae-s1 \\
      --config configs/rae_config_rangeview.py

  # Stage 2  (set rae_ckpt in config first)
  torchrun --nproc_per_node=1 scripts/train_rae_rangeview.py \\
      --stage 2 --batch_size 2 --exp_name rae-s2 \\
      --config configs/rae_config_rangeview.py
"""

import os, sys, math, time, random, logging, argparse
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from deepspeed.ops.adam import DeepSpeedCPUAdam
import deepspeed
import wandb

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from utils.config_utils import Config
from utils.deepspeed_utils import get_deepspeed_config
from utils.utils import setup_logger
from utils.comm import _init_dist_envi
from utils.running import (
    init_lr_schedule, get_cosine_schedule_with_warmup,
    save_ckpt_deepspeed, load_from_deepspeed_ckpt,
)
from utils.bev_utils import render_bev_comparison, render_rangeview_comparison
from models.dino_rae_rangeview import RangeViewRAE, RangeViewDINODiT
from dataset.dataset_kitti_rangeview import (
    KITTIRangeViewDataset,
    KITTIRangeViewTrainDataset,
    KITTIRangeViewValDataset,
)
from dataset.projection import RangeProjection

logger = logging.getLogger('base')


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',   required=True)
    p.add_argument('--stage',    default='1', choices=['1','2'])
    p.add_argument('--exp_name', required=True)
    p.add_argument('--iter',     default=60_000_000, type=int)
    p.add_argument('--batch_size', default=4, type=int)
    p.add_argument('--blr',      type=float, default=None)
    p.add_argument('--lr',       type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--resume_path',  default=None)
    p.add_argument('--resume_step',  default=0, type=int)
    p.add_argument('--eval_steps',   default=2000, type=int)
    p.add_argument('--vis_steps',    default=500, type=int)
    p.add_argument('--warmup_steps', default=2000, type=int)
    p.add_argument('--no_wandb',  action='store_true')
    p.add_argument('--no_log_file', action='store_true')
    p.add_argument('--launcher',  default='pytorch')
    p.add_argument('--load_from_deepspeed', default=None)
    p.add_argument('--wandb_project', default='difforecast-rae')
    args = p.parse_args()
    cfg  = Config.fromfile(args.config)
    cfg.merge_from_dict(args.__dict__)
    return cfg


# ── Dataset ──────────────────────────────────────────────────────────────────

def make_dataset(args, train=True):
    """Build KITTIRangeViewDataset with 5-channel projection."""
    common = dict(
        sequences_path=args.kitti_sequences_path,
        poses_path=args.kitti_poses_path,
        h=args.range_h, w=args.range_w,
        fov_up=args.fov_up, fov_down=args.fov_down,
        fov_left=args.fov_left, fov_right=args.fov_right,
        proj_img_mean=args.proj_img_mean,
        proj_img_stds=args.proj_img_stds,
        pc_extension=args.pc_extension,
        pc_dtype=getattr(np, args.pc_dtype),
        pc_reshape=tuple(args.pc_reshape),
        five_channel=True,   # 5-channel: range, x, y, z, intensity
    )
    if args.stage == '1':
        # Single-frame dataset for RAE training
        return KITTIRangeViewDataset(
            sequences=args.train_sequences if train else args.val_sequences,
            condition_frames=0, forward_iter=1, is_train=train,
            augmentation_config=args.augmentation_config if train else None,
            **common,
        )
    else:
        seqs = args.train_sequences if train else args.val_sequences
        return KITTIRangeViewTrainDataset(
            sequences=seqs,
            condition_frames=args.condition_frames,
            forward_iter=args.forward_iter,
            is_train=train,
            augmentation_config=args.augmentation_config if train else None,
            **common,
        ) if train else KITTIRangeViewValDataset(
            sequences=seqs,
            condition_frames=args.condition_frames,
            forward_iter=args.forward_iter,
            is_train=False,
            **common,
        )


# ── Stage 1 training loop ────────────────────────────────────────────────────

def train_stage1(args, model_engine, scheduler, loader, global_rank, step):
    """Train RAE (ViT-XL decoder) using per-channel L1 + BEV perceptual loss."""
    model_engine.train()
    time_stamp = time.time()

    for epoch in range(99999):
        for batch in loader:
            if step >= args.iter:
                return step

            # Stage 1 dataset returns (data, poses) where data is [T, C, H, W]
            # with T=1 (single frame) from condition_frames=0, forward_iter=1
            data, poses = batch
            if data.dim() == 5:   # [B, T, C, H, W] with T=1
                data = data[:, 0]
            x = data.cuda(non_blocking=True).to(torch.bfloat16)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model_engine(x)

            loss = out['loss_all']
            if not math.isfinite(loss.item()):
                print(f"Step {step}: loss={loss.item()}, stopping"); sys.exit(1)

            model_engine.backward(loss)
            model_engine.step()
            if scheduler is not None:
                scheduler.step()

            step += 1
            elapsed = time.time() - time_stamp; time_stamp = time.time()

            if step % 50 == 0 and global_rank == 0:
                lr = model_engine.get_lr()[0]
                msg = (f"[S1] step={step} | loss={loss.item():.4f} | "
                       f"rec={out['loss_rec'].item():.4f} | "
                       f"bev={out['loss_bev'].item():.4f} | "
                       f"lr={lr:.2e} | {elapsed:.2f}s/step")
                logger.info(msg)
                if hasattr(args, 'writer') and args.writer:
                    args.writer.add_scalar('stage1/loss_all', loss.item(), step)
                    args.writer.add_scalar('stage1/loss_rec', out['loss_rec'].item(), step)
                if not getattr(args, 'no_wandb', False) and global_rank == 0:
                    wandb.log({'s1/loss': loss.item(), 's1/rec': out['loss_rec'].item(),
                               's1/bev': out['loss_bev'].item(), 'step': step})

            if step % args.eval_steps == 0 and global_rank == 0:
                raw = model_engine.module if hasattr(model_engine, 'module') else model_engine
                raw.save_model(args.save_model_path, step, rank=global_rank)
                logger.info(f"[S1] Saved checkpoint at step {step}")

    return step


# ── Stage 2 training loop ────────────────────────────────────────────────────

def train_stage2(args, model_engine, scheduler, loader, global_rank, step):
    """Train STT + FluxDiT with chain-of-forward autoregressive training."""
    model_engine.train()
    time_stamp = time.time()
    cf = args.condition_frames
    fw_iter = args.forward_iter

    for epoch in range(99999):
        for batch in loader:
            if step >= args.iter:
                return step

            range_views, poses = batch
            # range_views: [B, T, 5, H, W]  T = cf + fw_iter
            range_views = range_views.cuda(non_blocking=True).to(torch.bfloat16)
            poses       = poses.cuda(non_blocking=True).float()

            rot_matrix    = poses[:, :cf + fw_iter]
            latents_cond_next = None
            rel_pose_cond, rel_yaw_cond = None, None

            # Chain-of-forward: fw_iter AR steps per training batch
            for j in range(fw_iter):
                features_cond = range_views[:, j:j+cf]           # [B, CF, 5, H, W]
                features_gt   = range_views[:, j+cf:j+cf+1]      # [B, 1, 5, H, W]
                rot_slice     = rot_matrix[:, j:(j+cf+1)]         # [B, CF+1, 4, 4]

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    out = model_engine(
                        features_cond, rot_slice, features_gt,
                        rel_pose_cond=rel_pose_cond,
                        rel_yaw_cond=rel_yaw_cond,
                        step=step,
                        latents_cond_precomputed=latents_cond_next,
                    )

                loss = out['loss_all']
                if not math.isfinite(loss.item()):
                    print(f"Step {step}: loss={loss.item()}, stopping"); sys.exit(1)

                model_engine.backward(loss)
                model_engine.step()
                if scheduler is not None:
                    scheduler.step()

                # Sliding window: drop oldest frame, append new prediction
                if j < fw_iter - 1 and out.get('predict_latents') is not None:
                    pred_lat = out['predict_latents'].detach()     # [B,256,384]
                    slide_from = latents_cond_next if latents_cond_next is not None \
                                 else out.get('latents_cond_enc')  # [B,CF,256,384]
                    if slide_from is not None:
                        latents_cond_next = torch.cat([
                            slide_from[:, 1:], pred_lat.unsqueeze(1)
                        ], dim=1)                                   # [B,CF,256,384]
                    if out.get('predict') is not None:
                        features_cond = torch.cat([
                            features_cond[:, 1:],
                            out['predict'].detach().unsqueeze(1),
                        ], dim=1)

            step += 1
            elapsed = time.time() - time_stamp; time_stamp = time.time()

            if step % 50 == 0 and global_rank == 0:
                lr = model_engine.get_lr()[0]
                msg = (f"[S2] step={step} | loss={loss.item():.4f} | "
                       f"diff={out['loss_diff'].item():.4f} | "
                       f"l1={out['loss_range_l1'].item():.4f} | "
                       f"bev={out['loss_bev_percep'].item():.4f} | "
                       f"lr={lr:.2e} | {elapsed:.2f}s/step")
                logger.info(msg)
                if hasattr(args, 'writer') and args.writer:
                    args.writer.add_scalar('stage2/loss_all',  loss.item(), step)
                    args.writer.add_scalar('stage2/loss_diff', out['loss_diff'].item(), step)
                    args.writer.add_scalar('stage2/loss_l1',   out['loss_range_l1'].item(), step)
                if not getattr(args, 'no_wandb', False) and global_rank == 0:
                    wandb.log({'s2/loss': loss.item(), 's2/diff': out['loss_diff'].item(),
                               's2/l1': out['loss_range_l1'].item(),
                               's2/bev': out['loss_bev_percep'].item(), 'step': step})

            # Visualisation
            if (args.vis_steps > 0 and step % args.vis_steps == 0
                    and global_rank == 0 and out.get('predict') is not None):
                _save_vis(step, args, features_cond, features_gt, out['predict'])

            if step % args.eval_steps == 0 and global_rank == 0:
                raw = model_engine.module if hasattr(model_engine, 'module') else model_engine
                raw.save_model(args.save_model_path, step, rank=global_rank)
                logger.info(f"[S2] Saved checkpoint at step {step}")

    return step


def _save_vis(step, args, features_cond, features_gt, predict):
    """Save a quick range-view comparison (range channel only) to disk."""
    try:
        vis_dir = os.path.join(args.validation_path, 'vis')
        os.makedirs(vis_dir, exist_ok=True)
        projector = RangeProjection(
            fov_up=args.fov_up, fov_down=args.fov_down,
            fov_left=args.fov_left, fov_right=args.fov_right,
            proj_h=args.range_h, proj_w=args.range_w,
        )
        render_rangeview_comparison(
            step=step, save_dir=vis_dir,
            cond_frames=features_cond[0, :, 0].cpu().numpy(),   # [CF, H, W] range ch
            gt_frame=features_gt[0, 0, 0].cpu().numpy(),
            pred_frame=predict[0, 0].cpu().numpy(),
        )
    except Exception as e:
        logger.warning(f"Visualisation failed at step {step}: {e}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Distributed setup
    _init_dist_envi(args)
    dist.init_process_group(backend="nccl")
    global_rank = dist.get_rank()
    world_size  = dist.get_world_size()

    # Logging
    log_path        = os.path.join(args.logdir, args.exp_name)
    save_model_path = os.path.join(args.outdir, args.exp_name)
    val_path        = os.path.join(args.validation_dir, args.exp_name)
    tdir_path       = os.path.join(args.tdir, args.exp_name)
    args.save_model_path = save_model_path
    args.validation_path = val_path

    if global_rank == 0:
        for d in [log_path, save_model_path, val_path, tdir_path]:
            os.makedirs(d, exist_ok=True)
        setup_logger('base', log_path, 'train', screen=True,
                     to_file=not getattr(args, 'no_log_file', False))
        args.writer = SummaryWriter(os.path.join(tdir_path, 'train'))
        if not getattr(args, 'no_wandb', False):
            wandb.init(project=args.wandb_project, name=args.exp_name,
                       config={k: v for k, v in vars(args).items()
                                if isinstance(v, (int,float,str,bool,type(None)))},
                       resume='allow')
    else:
        args.writer = None

    # Seeds
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # ── Build model ─────────────────────────────────────────────────────────
    if args.stage == '1':
        model = RangeViewRAE(args, local_rank=global_rank)
    else:
        ckpt = getattr(args, 'resume_path', None)
        model = RangeViewDINODiT(args, local_rank=global_rank, load_path=ckpt)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"[Stage {args.stage}] trainable params: {total_params:.1f} M")

    # ── DeepSpeed ──────────────────────────────────────────────────────────
    eff_batch = args.batch_size * world_size
    blr = getattr(args, 'blr', None) or 1e-4
    lr  = getattr(args, 'lr', None) or (blr * eff_batch / 256)
    ds_cfg = get_deepspeed_config(args)

    optimizer = DeepSpeedCPUAdam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=args.weight_decay, adamw_mode=True,
    )
    # Build scheduler before deepspeed.initialize wraps the optimizer
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.iter,
    )
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer,
        config=ds_cfg, dist_init_required=False,
    )
    if getattr(args, 'load_from_deepspeed', None):
        load_from_deepspeed_ckpt(model_engine, args.load_from_deepspeed)

    # ── Dataset ──────────────────────────────────────────────────────────
    train_ds = make_dataset(args, train=True)
    sampler  = DistributedSampler(train_ds, num_replicas=world_size,
                                   rank=global_rank, shuffle=True)
    loader   = DataLoader(train_ds, batch_size=args.batch_size,
                          sampler=sampler, num_workers=args.num_workers,
                          pin_memory=True, drop_last=True)

    logger.info(f"Training Stage {args.stage}: {len(train_ds)} samples, "
                f"lr={lr:.2e}, eff_batch={eff_batch}")

    # ── Train ──────────────────────────────────────────────────────────────
    step = int(getattr(args, 'resume_step', 0))
    if args.stage == '1':
        train_stage1(args, model_engine, scheduler, loader, global_rank, step)
    else:
        train_stage2(args, model_engine, scheduler, loader, global_rank, step)


if __name__ == '__main__':
    main()
