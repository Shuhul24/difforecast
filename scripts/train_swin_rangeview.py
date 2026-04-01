"""
Training script for TULIP-inspired Swin Transformer Range View Pipeline.

Stage 1  --stage 1
  Train TULIPRangeEncoder + TULIPRangeDecoder (Swin RAE).
  Loss: per-channel Berhu/L1 + optional BEV perceptual.
  Save the best checkpoint; set swin_ckpt in the config before Stage 2.

Stage 2  --stage 2
  Train STT + FluxDiT with frozen Swin encoder and decoder.
  Rectified-flow diffusion in TULIP 4-stage Swin bottleneck latent space [B, T, 64, 768].
  Frozen decoder uses skip features from the last condition frame.
  Chain-of-forward autoregressive training (fw_iter AR steps per batch).

Usage:
  # Stage 1
  torchrun --nproc_per_node=1 scripts/train_swin_rangeview.py \\
      --stage 1 --batch_size 4 --exp_name swin-s1 \\
      --config configs/swin_config_rangeview.py

  # Stage 2  (set swin_ckpt in config first)
  torchrun --nproc_per_node=1 scripts/train_swin_rangeview.py \\
      --stage 2 --batch_size 2 --exp_name swin-s2 \\
      --config configs/swin_config_rangeview.py
"""

import os, sys, math, time, random, logging, argparse, glob
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
from models.swin_rae_rangeview import RangeViewSwinRAE, RangeViewSwinDiT
from dataset.dataset_kitti_rangeview import (
    KITTIRangeViewDataset,
    KITTIRangeViewTrainDataset,
    KITTIRangeViewValDataset,
)

logger = logging.getLogger('base')


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',        required=True)
    p.add_argument('--stage',         default='1', choices=['1', '2'])
    p.add_argument('--exp_name',      required=True)
    p.add_argument('--iter',          default=60_000_000, type=int)
    p.add_argument('--batch_size',    default=4, type=int)
    p.add_argument('--blr',           type=float, default=None)
    p.add_argument('--lr',            type=float, default=None)
    p.add_argument('--weight_decay',  type=float, default=0.01)
    p.add_argument('--resume_path',   default=None)
    p.add_argument('--resume_step',   default=0, type=int)
    p.add_argument('--eval_steps',    default=2000, type=int)
    p.add_argument('--vis_steps',     default=500, type=int)
    p.add_argument('--warmup_steps',  default=2000, type=int)
    p.add_argument('--no_wandb',      action='store_true')
    p.add_argument('--no_log_file',   action='store_true')
    p.add_argument('--launcher',      default='pytorch')
    p.add_argument('--load_from_deepspeed', default=None)
    p.add_argument('--wandb_project', default='difforecast-swin')
    args = p.parse_args()
    cfg  = Config.fromfile(args.config)
    cfg.merge_from_dict(args.__dict__)
    return cfg


# ── Dataset ───────────────────────────────────────────────────────────────────

def make_dataset(args, train=True):
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
        five_channel=getattr(args, 'five_channel', False),
        log_range=getattr(args, 'log_range', True),
    )
    if args.stage == '1':
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


# ── Stage 1 training loop ─────────────────────────────────────────────────────

def train_stage1(args, model_engine, scheduler, loader, global_rank, step):
    """Train Swin RAE: Berhu (range) + L1 (intensity) + optional BEV perceptual."""
    model_engine.train()
    time_stamp = time.time()
    steps_per_epoch = len(loader)

    for epoch in range(99999):
        for batch in loader:
            if step >= args.iter:
                return step

            data, poses = batch
            if data.dim() == 5:
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
            epoch_frac = step / steps_per_epoch

            if step % 50 == 0 and global_rank == 0:
                lr = model_engine.get_lr()[0]
                msg = (f"[S1] epoch={epoch_frac:.2f} | step={step} | loss={loss.item():.4f} | "
                       f"rec={out['loss_rec'].item():.4f} | bev={out['loss_bev'].item():.4f} | "
                       f"lr={lr:.2e} | {elapsed:.2f}s/step")
                logger.info(msg)
                if hasattr(args, 'writer') and args.writer:
                    args.writer.add_scalar('stage1/loss_all', loss.item(), step)
                    args.writer.add_scalar('stage1/loss_rec', out['loss_rec'].item(), step)
                if not getattr(args, 'no_wandb', False) and global_rank == 0:
                    wandb.log({'s1/loss': loss.item(), 's1/rec': out['loss_rec'].item(),
                               's1/bev': out['loss_bev'].item(), 'step': step, 'epoch': epoch_frac})

            if (args.vis_steps > 0 and step % args.vis_steps == 0
                    and global_rank == 0 and out.get('x_rec') is not None):
                _save_vis_stage1(step, args, x, out['x_rec'])

            if step % args.eval_steps == 0 and global_rank == 0:
                raw = model_engine.module if hasattr(model_engine, 'module') else model_engine
                raw.save_model(args.save_model_path, step, rank=global_rank)
                logger.info(f"[S1] Saved checkpoint at step {step}")
                _delete_old_checkpoints(args.save_model_path, step, prefix='swin_rae_step')

    return step


# ── Stage 2 training loop ─────────────────────────────────────────────────────

def train_stage2(args, model_engine, scheduler, loader, global_rank, step):
    """Train STT + FluxDiT with chain-of-forward autoregressive training."""
    model_engine.train()
    time_stamp = time.time()
    fw_iter = args.forward_iter

    for epoch in range(99999):
        for batch in loader:
            if step >= args.iter:
                return step

            range_views, poses = batch
            range_views = range_views.cuda(non_blocking=True).to(torch.bfloat16)
            poses       = poses.cuda(non_blocking=True).float()
            rot_matrix  = poses[:, :args.condition_frames + fw_iter]

            latents_cond_next = None
            rel_pose_cond = rel_yaw_cond = None

            for j in range(fw_iter):
                features_cond = range_views[:, j:j + args.condition_frames]
                features_gt   = range_views[:, j + args.condition_frames:j + args.condition_frames + 1]
                rot_slice     = rot_matrix[:, j:(j + args.condition_frames + 1)]

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

                if j < fw_iter - 1 and out.get('predict_latents') is not None:
                    pred_lat  = out['predict_latents'].detach()
                    slide_from = latents_cond_next if latents_cond_next is not None \
                                 else out.get('latents_cond_enc')
                    if slide_from is not None:
                        latents_cond_next = torch.cat([
                            slide_from[:, 1:], pred_lat.unsqueeze(1)
                        ], dim=1)
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
                       f"bev={out['loss_bev_percep'].item():.4f} | "
                       f"lr={lr:.2e} | {elapsed:.2f}s/step")
                logger.info(msg)
                if hasattr(args, 'writer') and args.writer:
                    args.writer.add_scalar('stage2/loss_all',  loss.item(), step)
                    args.writer.add_scalar('stage2/loss_diff', out['loss_diff'].item(), step)
                if not getattr(args, 'no_wandb', False) and global_rank == 0:
                    wandb.log({'s2/loss': loss.item(), 's2/diff': out['loss_diff'].item(),
                               's2/bev': out['loss_bev_percep'].item(), 'step': step})

            if (args.vis_steps > 0 and step % args.vis_steps == 0
                    and global_rank == 0 and out.get('predict') is not None):
                _save_vis(step, args, features_cond, features_gt, out['predict'])

            if step % args.eval_steps == 0 and global_rank == 0:
                raw = model_engine.module if hasattr(model_engine, 'module') else model_engine
                raw.save_model(args.save_model_path, step, rank=global_rank)
                logger.info(f"[S2] Saved checkpoint at step {step}")
                _delete_old_checkpoints(args.save_model_path, step, prefix='swin_dit_step')

    return step


# ── Utilities ─────────────────────────────────────────────────────────────────

def _delete_old_checkpoints(save_dir, current_step, prefix='swin_rae_step'):
    for f in glob.glob(os.path.join(save_dir, f'{prefix}*.pkl')):
        if f != os.path.join(save_dir, f'{prefix}{current_step}.pkl'):
            os.remove(f)
            logger.info(f"Deleted old checkpoint: {f}")


def _save_vis_stage1(step, args, x, x_rec):
    try:
        vis_dir = os.path.join(args.validation_path, 'vis_s1')
        os.makedirs(vis_dir, exist_ok=True)
        if getattr(args, 'log_range', True):
            gt_depth   = (2.0 ** (x[0, 0].float().detach().cpu().numpy()    * 6.0)) - 1.0
            pred_depth = (2.0 ** (x_rec[0, 0].float().detach().cpu().numpy() * 6.0)) - 1.0
        else:
            m0, s0 = args.proj_img_mean[0], args.proj_img_stds[0]
            gt_depth   = x[0, 0].float().detach().cpu().numpy()    * s0 + m0
            pred_depth = x_rec[0, 0].float().detach().cpu().numpy() * s0 + m0
        mae = float(np.abs(pred_depth - gt_depth).mean())
        render_rangeview_comparison(
            gt_depth=gt_depth, pred_depth=pred_depth,
            output_path=os.path.join(vis_dir, f'step_{step:07d}.png'),
            frame_idx=step, metrics={'MAE_m': mae},
            title_suffix=' — Stage 1 Swin RAE reconstruction',
        )
    except Exception as e:
        logger.warning(f"[S1] Visualisation failed at step {step}: {e}")


def _save_vis(step, args, features_cond, features_gt, predict):
    try:
        vis_dir = os.path.join(args.validation_path, 'vis')
        os.makedirs(vis_dir, exist_ok=True)
        if getattr(args, 'log_range', True):
            gt_depth   = (2.0 ** (features_gt[0, 0, 0].float().cpu().numpy() * 6.0)) - 1.0
            pred_depth = (2.0 ** (predict[0, 0].float().cpu().numpy()         * 6.0)) - 1.0
        else:
            m0, s0 = args.proj_img_mean[0], args.proj_img_stds[0]
            gt_depth   = features_gt[0, 0, 0].float().cpu().numpy() * s0 + m0
            pred_depth = predict[0, 0].float().cpu().numpy()         * s0 + m0
        mae = float(np.abs(pred_depth - gt_depth).mean())
        render_rangeview_comparison(
            gt_depth=gt_depth, pred_depth=pred_depth,
            output_path=os.path.join(vis_dir, f'step_{step:07d}.png'),
            frame_idx=step, metrics={'MAE_m': mae},
            title_suffix=' — Stage 2 Swin prediction',
        )
    except Exception as e:
        logger.warning(f"Visualisation failed at step {step}: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    _init_dist_envi(args)
    dist.init_process_group(backend="nccl")
    global_rank = dist.get_rank()
    world_size  = dist.get_world_size()

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
                                if isinstance(v, (int, float, str, bool, type(None)))},
                       resume='allow')
    else:
        args.writer = None

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # Build model
    if args.stage == '1':
        model = RangeViewSwinRAE(args, local_rank=global_rank)
    else:
        ckpt = getattr(args, 'resume_path', None)
        model = RangeViewSwinDiT(args, local_rank=global_rank, load_path=ckpt)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"[Stage {args.stage}] trainable params: {total_params:.1f} M")

    eff_batch = args.batch_size * world_size
    blr = getattr(args, 'blr', None) or 1e-4
    lr  = getattr(args, 'lr', None) or (blr * eff_batch / 256)
    ds_cfg = get_deepspeed_config(args)

    optimizer = DeepSpeedCPUAdam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=args.weight_decay, adamw_mode=True,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.iter,
    )
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer,
        config=ds_cfg, dist_init_required=False,
    )
    if getattr(args, 'load_from_deepspeed', None):
        load_from_deepspeed_ckpt(model_engine, args.load_from_deepspeed)

    train_ds = make_dataset(args, train=True)
    sampler  = DistributedSampler(train_ds, num_replicas=world_size,
                                   rank=global_rank, shuffle=True)
    loader   = DataLoader(train_ds, batch_size=args.batch_size,
                          sampler=sampler, num_workers=args.num_workers,
                          pin_memory=True, drop_last=True)

    logger.info(f"Training Stage {args.stage}: {len(train_ds)} samples, "
                f"lr={lr:.2e}, eff_batch={eff_batch}")

    step = int(getattr(args, 'resume_step', 0))
    if args.stage == '1':
        train_stage1(args, model_engine, scheduler, loader, global_rank, step)
    else:
        train_stage2(args, model_engine, scheduler, loader, global_rank, step)


if __name__ == '__main__':
    main()
