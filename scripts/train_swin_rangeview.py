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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from deepspeed.ops.adam import FusedAdam
import deepspeed
import wandb
from utils.preprocess import get_rel_pose

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
    
    if cfg.stage == '2' and not getattr(cfg, 'return_predict', False):
        cfg.return_predict = True
        
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
        depth_only=(int(getattr(args, 'range_channels', 2)) == 1),
        mask_channel=getattr(args, 'mask_channel', False),
    )
    if args.stage == '1':
        return KITTIRangeViewDataset(
            sequences=args.train_sequences if train else args.val_sequences,
            condition_frames=0, forward_iter=1, is_train=train,
            augmentation_config=None,   # removed: per-frame aug breaks reconstruction
            **common,
        )
    else:
        seqs = args.train_sequences if train else args.val_sequences
        return KITTIRangeViewTrainDataset(
            sequences=seqs,
            condition_frames=args.condition_frames,
            forward_iter=args.forward_iter,
            is_train=train,
            augmentation_config=None,   # removed: independent per-frame aug contradicts GT poses
            **common,
        ) if train else KITTIRangeViewValDataset(
            sequences=seqs,
            condition_frames=args.condition_frames,
            forward_iter=args.forward_iter,
            is_train=False,
            **common,
        )


# ── Stage 1 training loop ─────────────────────────────────────────────────────

def train_stage1(args, model_engine, scheduler, loader, val_loader, global_rank, step):
    """Train Swin RAE: Berhu (range) + L1 (intensity) + optional BEV perceptual."""
    model_engine.train()
    time_stamp = time.time()
    steps_per_epoch = len(loader)
    best_val_loss = float('inf')

    for epoch in range(99999):
        for batch in loader:
            if step >= args.iter:
                return step

            data, poses = batch
            if data.dim() == 5:
                data = data[:, 0]
            x = data.cuda(non_blocking=True).to(torch.bfloat16)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model_engine(x, step=step)

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
                loss_mask_val = out.get('loss_mask', torch.tensor(0.)).item()
                loss_kl_val   = out.get('loss_kl',   torch.tensor(0.)).item()
                beta = getattr(args, 'kl_weight', 1e-4) * min(
                    1.0, float(step) / max(getattr(args, 'kl_warmup_steps', 10000), 1))
                msg = (f"[S1] epoch={epoch_frac:.2f} | step={step} | loss={loss.item():.4f} | "
                       f"rec={out['loss_rec'].item():.4f} | kl={loss_kl_val:.4f} | "
                       f"beta={beta:.2e} | mask_bce={loss_mask_val:.4f} | "
                       f"bev={out['loss_bev'].item():.4f} | "
                       f"lr={lr:.2e} | {elapsed:.2f}s/step")
                logger.info(msg)
                if hasattr(args, 'writer') and args.writer:
                    args.writer.add_scalar('stage1/loss_all',      loss.item(),           step)
                    args.writer.add_scalar('stage1/loss_rec',      out['loss_rec'].item(), step)
                    args.writer.add_scalar('stage1/loss_kl',       loss_kl_val,           step)
                    args.writer.add_scalar('stage1/kl_beta',       beta,                  step)
                    args.writer.add_scalar('stage1/loss_mask_bce', loss_mask_val,          step)
                if not getattr(args, 'no_wandb', False) and global_rank == 0:
                    wandb.log({'s1/loss': loss.item(), 's1/rec': out['loss_rec'].item(),
                               's1/kl': loss_kl_val, 's1/kl_beta': beta,
                               's1/mask_bce': loss_mask_val,
                               's1/bev': out['loss_bev'].item(), 'step': step, 'epoch': epoch_frac})

            if (args.vis_steps > 0 and step % args.vis_steps == 0
                    and global_rank == 0 and out.get('x_rec') is not None):
                _save_vis_stage1(step, args, x, out['x_rec'])

            if step % args.eval_steps == 0:
                raw = model_engine.module if hasattr(model_engine, 'module') else model_engine
                if global_rank == 0:
                    raw.save_model(args.save_model_path, step, rank=global_rank)
                    logger.info(f"[S1] Saved checkpoint at step {step}")
                    _delete_old_checkpoints(args.save_model_path, step, prefix='swin_rae_step')
                val_loss = _validate_stage1(args, model_engine, val_loader, step, global_rank)
                if global_rank == 0 and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    _save_best_ckpt(raw, args.save_model_path, step, prefix='swin_rae')
                    logger.info(f"[S1] New best val_rec={val_loss:.4f} at step {step}")

    return step


# ── Stage 2 training loop ─────────────────────────────────────────────────────

def train_stage2(args, model_engine, scheduler, loader, val_loader, global_rank, step):
    """Train STT + FluxDiT + PoseDiT with chain-of-forward autoregressive training.

    AR loop (fw_iter steps per batch):
      1. Forward with AR conditioning: predicted frames + predicted poses from
         ar_rot_window (not GT), learning joint p(x, pose | history).
      2. Backward + optimizer step on combined loss (diff + pose + aux).
      3. Slide ALL conditioning windows (pixel, latent, rotation).
         ar_rot_window accumulates predicted absolute rotations so rel_pose_cond
         is derived entirely from model predictions — no GT teacher forcing on
         transitions involving predicted conditioning frames.
      4. Optionally (ar_eval_rollout=True in config): obtain clean AR predictions
         via eval-mode forward pass (no dropout, full diffusion sampling),
         mirroring Epona's clean-sampling approach. Default uses training-pass
         predictions (fast path).
    """
    model_engine.train()
    time_stamp = time.time()
    fw_iter = args.forward_iter
    CF      = args.condition_frames
    ar_eval = getattr(args, 'ar_eval_rollout', False)
    best_val_loss = float('inf')

    for epoch in range(99999):
        for batch in loader:
            if step >= args.iter:
                return step

            range_views, poses = batch
            range_views = range_views.cuda(non_blocking=True).to(torch.bfloat16)
            poses       = poses.cuda(non_blocking=True).float()
            rot_matrix  = poses[:, :CF + fw_iter]   # [B, CF+fw_iter, 4, 4]

            # ── Initialise AR state ───────────────────────────────────────────
            latents_cond_next = None
            rel_pose_cond     = None   # [B, CF, 2] — derived from ar_rot_window after j=0
            rel_yaw_cond      = None   # [B, CF, 1]
            features_cond     = range_views[:, :CF]
            # Tracks predicted absolute rotations for conditioning frames.
            # Starts as GT; after each AR step the oldest GT is replaced by the
            # next predicted absolute rotation (_compose_predicted_rot).
            # Drives rel_pose_cond so the model learns p(x, pose | history)
            # rather than p(x | GT_pose, history) · p(pose | history).
            ar_rot_window     = rot_matrix[:, :CF].clone()   # [B, CF, 4, 4]

            # Cumulative losses over fw_iter steps (for logging)
            cumul_diff = cumul_pose = cumul_rv = cumul_cd = cumul_bev = cumul_repa = 0.0
            last_out        = None   # keep for vis / logging
            all_predictions = []     # collect per-AR-step predictions for vis

            for j in range(fw_iter):
                features_gt = range_views[:, j + CF:j + CF + 1]
                # rot_slice stays all-GT: the conditioning portion is superseded by
                # rel_pose_cond (from ar_rot_window) when j > 0. Only the last
                # (target) entry is used by step_train for the pose loss target,
                # so keeping it GT is correct and does not constitute teacher forcing.
                rot_slice   = rot_matrix[:, j:j + CF + 1]   # [B, CF+1, 4, 4]

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    out = model_engine(
                        features_cond, rot_slice, features_gt,
                        rel_pose_cond=rel_pose_cond,
                        rel_yaw_cond=rel_yaw_cond,
                        step=step,
                        latents_cond_precomputed=latents_cond_next,
                    )

                # Use loss_all as the gradient-carrying tensor.
                # All per-component entries in out (loss_diff, loss_pose, loss_rv,
                # loss_chamfer, loss_repa) are .detach()'d logging copies — they
                # carry NO gradients.  loss_all = diff + pose + chamfer*z_cd +
                # repa_weight*z_repa, with rv loss baked into diff when
                # return_predict=True and range_view_loss_weight > 0.
                loss = out['loss_all']

                # Grab detached scalars for logging (kept consistent with below)
                _repa_w    = float(getattr(args, 'repa_weight', 0.0))
                _chamfer_w = float(getattr(args, 'chamfer_loss_weight', 0.0))
                _rv_w      = float(getattr(args, 'range_view_loss_weight', 0.0))
                _loss_repa = out.get('loss_repa',
                                     out['loss_diff'].new_tensor(0.)).mean()
                _loss_cd   = out.get('loss_chamfer',
                                     out['loss_diff'].new_tensor(0.)).mean()
                _loss_rv   = out.get('loss_rv',
                                     out['loss_diff'].new_tensor(0.)).mean()

                # ── Auxiliary physical-unit pose regression ──────────────────
                # The flow-matching pose loss can be satisfied by predicting the
                # conditional mean (average KITTI velocity).  An L1 on physical
                # units (metres, degrees) penalises mean-seeking directly and
                # provides a much stronger gradient to the STT.
                # Controlled by pose_reg_weight in config (default 0 = off).
                # NOTE: must be computed before backward() so the gradient flows.
                _pose_reg_w = float(getattr(args, 'pose_reg_weight', 0.0))
                if _pose_reg_w > 0:
                    pred_xy  = out.get('predict_pose_xy')   # [B, 1, 2] metres
                    pred_yaw = out.get('predict_pose_yaw')  # [B, 1, 1] degrees
                    if pred_xy is not None and pred_yaw is not None:
                        # rel_pose / rel_yaw: conditioning + target poses
                        # Last entry [-1] is the target the PoseDiT should predict
                        with torch.cuda.amp.autocast(enabled=False):
                            _rp, _ry = get_rel_pose(rot_slice.float())
                        gt_xy  = _rp[:, -1:].float()   # [B, 1, 2]
                        gt_yaw = _ry[:, -1:].float()   # [B, 1, 1]
                        _pose_reg = (
                            torch.nn.functional.l1_loss(pred_xy.float(),  gt_xy)
                            + torch.nn.functional.l1_loss(pred_yaw.float(), gt_yaw)
                        )
                        loss = loss + _pose_reg_w * _pose_reg
                        out['loss_pose_reg'] = _pose_reg.detach()

                if not math.isfinite(loss.item()):
                    print(f"[S2] Step {step} AR-step {j}: loss={loss.item()}, stopping")
                    sys.exit(1)

                # ── Backward + step ───────────────────────────────────────────
                model_engine.backward(loss)
                model_engine.step()
                if scheduler is not None:
                    scheduler.step()

                # ── Accumulate losses for logging ────────────────────────────
                cumul_diff += out['loss_diff'].item()
                cumul_pose += out['loss_pose'].item()
                cumul_rv   += _rv_w * _loss_rv.item()
                cumul_cd   += _chamfer_w * _loss_cd.item()
                cumul_bev  += out.get('loss_bev_percep', torch.tensor(0.)).item()
                # Log effective REPA contribution (weighted + ramped)
                cumul_repa += _repa_w * _repa_ramp * _loss_repa.item()
                last_out    = out
                if global_rank == 0 and out.get('predict') is not None:
                    all_predictions.append(out['predict'].detach())

                if j < fw_iter - 1:
                    # ── Obtain AR predictions for next-step conditioning ──────
                    if ar_eval:
                        # Epona-style clean rollout: eval mode disables dropout and
                        # uses full diffusion sampling for unbiased AR predictions.
                        # Limitation: step_eval re-derives poses from GT rot_slice,
                        # so ar_rot_window-based pose conditioning only applies on
                        # the fast path (ar_eval_rollout=False).
                        inner = model_engine.module if hasattr(model_engine, 'module') \
                                else model_engine
                        inner.eval()
                        with torch.no_grad():
                            with torch.autocast('cuda', dtype=torch.bfloat16):
                                ar_pred_frame = inner(
                                    features_cond, rot_slice, sample_last=True,
                                )  # [B, C, H, W]  — routes to step_eval
                            # Re-encode the clean prediction for latent-space AR
                            ar_pred_lats, _ = inner.encode_sequence(
                                ar_pred_frame.unsqueeze(1).to(torch.bfloat16)
                            )
                            ar_pred_lats = ar_pred_lats[:, 0]   # [B, T, C]
                        inner.train()
                    else:
                        # Fast path: use training-pass predictions (detached)
                        ar_pred_frame = out.get('predict')
                        ar_pred_lats  = out.get('predict_latents')

                    # ── Slide pixel conditioning window ───────────────────────
                    if ar_pred_frame is None:
                        raise RuntimeError(
                            f"[S2] AR step {j}: model returned no 'predict'; "
                            "cannot continue autoregressive rollout. "
                            "Ensure return_predict=True in config."
                        )
                    features_cond = torch.cat(
                        [features_cond[:, 1:],
                         ar_pred_frame.detach().unsqueeze(1)], dim=1
                    )

                    # ── Slide latent conditioning window ──────────────────────
                    slide_from = latents_cond_next if latents_cond_next is not None \
                                 else out.get('latents_cond_enc')
                    if ar_pred_lats is not None and slide_from is not None:
                        latents_cond_next = torch.cat([
                            slide_from[:, 1:],
                            ar_pred_lats.detach().unsqueeze(1),
                        ], dim=1)

                    # ── Advance AR rotation window + update pose conditioning ─
                    # Compose the predicted absolute rotation for the newly added
                    # conditioning frame, slide ar_rot_window forward, then derive
                    # ALL conditioning relative poses from accumulated predictions.
                    # For fw_iter > 2 this correctly uses predicted rotations for
                    # predicted conditioning frames (unlike GT-based rp_prev which
                    # leaked GT rotations for already-predicted frames).
                    pred_pose_xy  = out.get('predict_pose_xy')   # [B, 1, 2]
                    pred_pose_yaw = out.get('predict_pose_yaw')  # [B, 1, 1]
                    if pred_pose_xy is not None and pred_pose_yaw is not None:
                        with torch.no_grad():
                            pred_abs_rot = _compose_predicted_rot(
                                ar_rot_window[:, -1],
                                pred_pose_xy[:, 0].float(),    # [B, 2]  metres
                                pred_pose_yaw[:, 0].float(),   # [B, 1]  degrees
                            )  # [B, 4, 4]
                            ar_rot_window = torch.cat(
                                [ar_rot_window[:, 1:],
                                 pred_abs_rot.unsqueeze(1)], dim=1
                            )  # [B, CF, 4, 4]
                            # All conditioning relative poses from predicted rots —
                            # no GT leakage for transitions involving predicted frames
                            with torch.cuda.amp.autocast(enabled=False):
                                rel_pose_cond, rel_yaw_cond = get_rel_pose(
                                    ar_rot_window.float()
                                )   # [B, CF, 2],  [B, CF, 1]

            step    += 1
            elapsed  = time.time() - time_stamp
            time_stamp = time.time()

            # ── Logging (every 50 steps, rank 0 only) ────────────────────────
            if step % 50 == 0 and global_rank == 0:
                lr       = model_engine.get_lr()[0]
                avg_diff  = cumul_diff / fw_iter
                avg_pose  = cumul_pose / fw_iter
                # cumul_rv/cd/repa already hold weighted effective values
                avg_rv    = cumul_rv   / fw_iter
                avg_cd    = cumul_cd   / fw_iter
                avg_bev   = cumul_bev  / fw_iter
                avg_repa  = cumul_repa / fw_iter
                avg_total = avg_diff + avg_pose + avg_rv + avg_cd + avg_bev + avg_repa

                stt_norm = last_out.get('stt_last_norm', torch.tensor(0.)).item() \
                           if last_out else 0.
                stt_std  = last_out.get('stt_last_std',  torch.tensor(0.)).item() \
                           if last_out else 0.

                pose_reg_val = last_out.get('loss_pose_reg', torch.tensor(0.)).item() \
                              if last_out else 0.
                msg = (
                    f"[S2] step={step} | total={avg_total:.4f} | "
                    f"diff={avg_diff:.4f} | pose={avg_pose:.4f} | "
                    f"pose_reg={pose_reg_val:.4f} | "
                    f"rv={avg_rv:.4f} | cd={avg_cd:.4f} | bev={avg_bev:.4f} | "
                    f"repa={avg_repa:.2e} | "
                    f"stt_norm={stt_norm:.3f} | stt_std={stt_std:.3f} | "
                    f"lr={lr:.2e} | {elapsed:.2f}s/step"
                )
                logger.info(msg)

                if hasattr(args, 'writer') and args.writer:
                    args.writer.add_scalar('stage2/loss_total', avg_total, step)
                    args.writer.add_scalar('stage2/loss_diff',  avg_diff,  step)
                    args.writer.add_scalar('stage2/loss_pose',  avg_pose,  step)
                    args.writer.add_scalar('stage2/loss_rv',    avg_rv,    step)
                    args.writer.add_scalar('stage2/loss_cd',    avg_cd,    step)
                    args.writer.add_scalar('stage2/loss_bev',   avg_bev,   step)
                    args.writer.add_scalar('stage2/loss_repa',  avg_repa,  step)
                    args.writer.add_scalar('debug/stt_last_norm', stt_norm, step)
                    args.writer.add_scalar('debug/stt_last_std',  stt_std,  step)

                if not getattr(args, 'no_wandb', False):
                    wandb.log({
                        # Individual DiT losses — separate W&B tabs/charts
                        's2/FluxDiT/loss_diff':    avg_diff,
                        's2/PoseDiT/loss_pose':    avg_pose,
                        # Auxiliary losses
                        's2/aux/loss_rv':          avg_rv,
                        's2/aux/loss_chamfer':     avg_cd,
                        's2/aux/loss_bev':         avg_bev,
                        's2/aux/loss_repa':        avg_repa,
                        # Combined
                        's2/loss_total':           avg_total,
                        # STT conditioning diagnostics
                        'debug/stt_last_norm':     stt_norm,
                        'debug/stt_last_std':      stt_std,
                        # Training diagnostics
                        'train/lr':                lr,
                        'train/step':              step,
                    })

            if (args.vis_steps > 0 and step % args.vis_steps == 0
                    and global_rank == 0 and all_predictions):
                _save_vis(step, args, range_views, all_predictions, CF, fw_iter)

            if step % args.eval_steps == 0:
                raw = model_engine.module if hasattr(model_engine, 'module') else model_engine
                if global_rank == 0:
                    raw.save_model(args.save_model_path, step, rank=global_rank)
                    logger.info(f"[S2] Saved checkpoint at step {step}")
                    _delete_old_checkpoints(args.save_model_path, step, prefix='swin_dit_step')
                val_loss = _validate_stage2(args, model_engine, val_loader, step, global_rank)
                if global_rank == 0 and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    _save_best_ckpt(raw, args.save_model_path, step, prefix='swin_dit')
                    logger.info(f"[S2] New best val_diff={val_loss:.4f} at step {step}")

    return step


# ── Utilities ─────────────────────────────────────────────────────────────────

def _compose_predicted_rot(base_rot, pred_xy, pred_yaw_deg):
    """Compose a predicted absolute 4×4 pose from the previous frame's absolute
    rotation and a predicted relative (dx, dy, yaw_deg) pose.

    Inverts get_rel_pose: base_rot @ T_rel → predicted absolute transform.

    base_rot    : [B, 4, 4]  last condition frame absolute pose (float32)
    pred_xy     : [B, 2]     predicted (dx, dy) in body frame (metres)
    pred_yaw_deg: [B, 1]     predicted yaw relative rotation (degrees)
    Returns     : [B, 4, 4]  predicted absolute pose of the next frame
    """
    B     = base_rot.shape[0]
    yaw   = pred_yaw_deg[:, 0] * (math.pi / 180.)   # degrees → radians
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


def _save_best_ckpt(raw_model, path, step, prefix):
    """Save model as best checkpoint (fixed name — overwrites previous best)."""
    best_path = os.path.join(path, f'{prefix}_best.pkl')
    torch.save({'model_state_dict': raw_model.state_dict(), 'step': step}, best_path)


def _validate_stage1(args, model_engine, val_loader, step, global_rank):
    """Compute avg reconstruction loss on val set (all ranks, then all_reduce)."""
    raw = model_engine.module if hasattr(model_engine, 'module') else model_engine
    raw.eval()
    total = torch.tensor(0.0, device='cuda')
    n     = torch.tensor(0,   device='cuda')
    with torch.no_grad():
        for batch in val_loader:
            data, _ = batch
            if data.dim() == 5:
                data = data[:, 0]
            x = data.cuda(non_blocking=True).to(torch.bfloat16)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = raw(x)
            total += out['loss_rec'].detach().float()
            n     += 1
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    dist.all_reduce(n,     op=dist.ReduceOp.SUM)
    raw.train()
    val_loss = (total / n.clamp(min=1)).item()
    if global_rank == 0:
        logger.info(f"[S1 Val] step={step} | val_rec={val_loss:.4f}")
        if hasattr(args, 'writer') and args.writer:
            args.writer.add_scalar('stage1/val_rec', val_loss, step)
        if not getattr(args, 'no_wandb', False):
            wandb.log({'s1/val_rec': val_loss, 'step': step})
    return val_loss


def _validate_stage2(args, model_engine, val_loader, step, global_rank):
    """Compute avg diffusion loss on val set — single AR step (j=0) for speed."""
    CF = args.condition_frames
    raw = model_engine.module if hasattr(model_engine, 'module') else model_engine
    raw.train()  # keep in train mode so forward() routes to step_train (loss dict)
    total = torch.tensor(0.0, device='cuda')
    n     = torch.tensor(0,   device='cuda')
    with torch.no_grad():
        for batch in val_loader:
            range_views, poses = batch
            range_views = range_views.cuda(non_blocking=True).to(torch.bfloat16)
            poses       = poses.cuda(non_blocking=True).float()
            features_cond = range_views[:, :CF]
            features_gt   = range_views[:, CF:CF + 1]
            rot_slice     = poses[:, :CF + 1]
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = raw(features_cond, rot_slice, features_gt, step=step)
            total += out['loss_diff'].detach().float().mean()
            n     += 1
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    dist.all_reduce(n,     op=dist.ReduceOp.SUM)
    # model stays in train() mode; train_stage2 manages mode as needed
    val_loss = (total / n.clamp(min=1)).item()
    if global_rank == 0:
        logger.info(f"[S2 Val] step={step} | val_diff={val_loss:.4f}")
        if hasattr(args, 'writer') and args.writer:
            args.writer.add_scalar('stage2/val_diff', val_loss, step)
        if not getattr(args, 'no_wandb', False):
            wandb.log({'s2/val_diff': val_loss, 'step': step})
    return val_loss


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


def _save_vis(step, args, range_views, all_predictions, CF, fw_iter):
    """Save a 3-row grid: condition frames (GT) | GT future | predicted future."""
    try:
        vis_dir = os.path.join(args.validation_path, 'vis')
        os.makedirs(vis_dir, exist_ok=True)
        log_range = getattr(args, 'log_range', True)
        m0, s0 = args.proj_img_mean[0], args.proj_img_stds[0]

        def to_depth(t_bchw):
            arr = t_bchw[0, 0].float().detach().cpu().numpy()
            if log_range:
                return np.clip((2.0 ** (arr * 6.0)) - 1.0, 0.0, 80.0)
            return np.clip(arr * s0 + m0, 0.0, 80.0)

        n_cols = max(CF, fw_iter)
        fig, axes = plt.subplots(3, n_cols, figsize=(n_cols * 8, 6))
        vmax = 80.0

        # Row 0: conditioning frames (GT input)
        for i in range(CF):
            depth = to_depth(range_views[:, i])
            axes[0, i].imshow(depth, cmap='turbo_r', vmin=0, vmax=vmax, aspect='auto')
            axes[0, i].set_title(f't-{CF - 1 - i}', fontsize=8)
            axes[0, i].axis('off')
        for i in range(CF, n_cols):
            axes[0, i].axis('off')

        # Row 1: GT future frames
        for i in range(fw_iter):
            depth = to_depth(range_views[:, CF + i])
            axes[1, i].imshow(depth, cmap='turbo_r', vmin=0, vmax=vmax, aspect='auto')
            axes[1, i].set_title(f't+{i + 1}', fontsize=8)
            axes[1, i].axis('off')
        for i in range(fw_iter, n_cols):
            axes[1, i].axis('off')

        # Row 2: predicted future frames
        for i, pred in enumerate(all_predictions):
            depth = to_depth(pred)
            axes[2, i].imshow(depth, cmap='turbo_r', vmin=0, vmax=vmax, aspect='auto')
            axes[2, i].set_title(f't+{i + 1}', fontsize=8)
            axes[2, i].axis('off')
        for i in range(len(all_predictions), n_cols):
            axes[2, i].axis('off')

        for r, label in enumerate(['Condition (GT)', 'GT Future', 'Predicted']):
            axes[r, 0].set_ylabel(label, fontsize=9)

        fig.suptitle(f'Stage 2 — step {step:,}', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'step_{step:07d}.png'), dpi=80, bbox_inches='tight')
        plt.close(fig)
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
        # Allow warm-starting Stage 1 from a pre-trained checkpoint (e.g. a
        # checkpoint trained without KL).  New keys (mu_proj, logvar_proj) will
        # be missing in old checkpoints and are kept at their default init.
        if getattr(args, 'resume_path', None):
            sd = torch.load(args.resume_path, map_location='cpu').get('model_state_dict', {})
            miss, unex = model.load_state_dict(sd, strict=False)
            logger.info(f"[S1] Loaded checkpoint from {args.resume_path} "
                        f"| missing={len(miss)} (expected: mu/logvar_proj if pre-KL) "
                        f"| unexpected={len(unex)}")
    else:
        ckpt = getattr(args, 'resume_path', None)
        model = RangeViewSwinDiT(args, local_rank=global_rank, load_path=ckpt)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"[Stage {args.stage}] trainable params: {total_params:.1f} M")

    eff_batch = args.batch_size * world_size
    blr = getattr(args, 'blr', None) or 1e-4
    lr  = getattr(args, 'lr', None) or (blr * eff_batch / 256)
    ds_cfg = get_deepspeed_config(args)

    optimizer = FusedAdam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=args.weight_decay, adam_w_mode=True,
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

    val_ds      = make_dataset(args, train=False)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size,
                                     rank=global_rank, shuffle=False)
    val_loader  = DataLoader(val_ds, batch_size=args.batch_size,
                             sampler=val_sampler, num_workers=args.num_workers,
                             pin_memory=True, drop_last=False)

    logger.info(f"Training Stage {args.stage}: {len(train_ds)} train / {len(val_ds)} val samples, "
                f"lr={lr:.2e}, eff_batch={eff_batch}")

    step = int(getattr(args, 'resume_step', 0))
    if args.stage == '1':
        train_stage1(args, model_engine, scheduler, loader, val_loader, global_rank, step)
    else:
        train_stage2(args, model_engine, scheduler, loader, val_loader, global_rank, step)


if __name__ == '__main__':
    main()
