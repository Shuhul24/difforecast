"""
Simplified DiT Model for Range View Image Prediction
This model only predicts the next frame without trajectory planning.
"""

import os
import time
import torch
import torch.nn as nn
import random
from einops import rearrange
from utils.preprocess import get_rel_pose
from models.stt import SpatialTemporalTransformer
from models.flux_dit import FluxParams, FluxDiT
from models.modules.tokenizer import poses_to_indices, yaws_to_indices
from utils.fft_utils import freq_mix, ideal_low_pass_filter
from models.modules.sampling import prepare_ids, get_schedule
from utils.range_losses import (
    make_valid_mask,
    range_view_l1_loss,
    features_to_xyz,
    batch_chamfer_distance,
)


class RangeViewDiT(nn.Module):
    """
    Simplified DiT model for range view image prediction.
    Only predicts next frame, no trajectory planning.
    """

    def __init__(
        self,
        args,
        local_rank=-1,
        load_path=None,
        condition_frames=3,
    ):
        super().__init__()
        self.local_rank = local_rank
        self.args = args
        self.condition_frames = condition_frames

        # For range view, we expect 6 channels: [range, x, y, z, intensity, label]
        self.range_channels = args.range_channels if hasattr(args, 'range_channels') else 6
        self.vae_emb_dim = self.range_channels * self.args.patch_size ** 2

        self.image_size = self.args.image_size
        self.h, self.w = (
            self.image_size[0] // (self.args.downsample_size * self.args.patch_size),
            self.image_size[1] // (self.args.downsample_size * self.args.patch_size)
        )
        self.pkeep = args.pkeep

        self.img_token_size = self.h * self.w
        self.pose_x_vocab_size = self.args.pose_x_vocab_size
        self.pose_y_vocab_size = self.args.pose_y_vocab_size
        self.yaw_vocab_size = self.args.yaw_vocab_size
        self.pose_x_bound = self.args.pose_x_bound
        self.pose_y_bound = self.args.pose_y_bound
        self.yaw_bound = self.args.yaw_bound

        self.pose_token_size = 2 * self.args.block_size
        self.yaw_token_size = 1 * self.args.block_size
        self.total_token_size = self.img_token_size + self.pose_token_size + self.yaw_token_size

        self.token_size_dict = {
            'img_tokens_size': self.img_token_size,
            'pose_tokens_size': self.pose_token_size,
            'yaw_token_size': self.yaw_token_size,
            'total_tokens_size': self.total_token_size
        }

        # Spatial-Temporal Transformer for conditioning
        self.model = SpatialTemporalTransformer(
            block_size=condition_frames * (self.total_token_size),
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            pose_x_vocab_size=self.pose_x_vocab_size,
            pose_y_vocab_size=self.pose_y_vocab_size,
            yaw_vocab_size=self.yaw_vocab_size,
            latent_size=(self.h, self.w),
            local_rank=local_rank,
            condition_frames=self.condition_frames,
            token_size_dict=self.token_size_dict,
            vae_emb_dim=self.vae_emb_dim,
            temporal_block=self.args.block_size
        )
        self.model.cuda()

        # Diffusion Transformer for frame prediction
        self.dit = FluxDiT(FluxParams(
            in_channels=self.vae_emb_dim,
            out_channels=self.vae_emb_dim,
            vec_in_dim=args.n_embd * (self.total_token_size - self.img_token_size),
            context_in_dim=args.n_embd,
            hidden_size=args.n_embd_dit,
            mlp_ratio=4.0,
            num_heads=args.n_head_dit,
            depth=args.n_layer[1],
            depth_single_blocks=args.n_layer[2],
            axes_dim=args.axes_dim_dit,
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ))
        self.dit.cuda()

        # Prepare IDs for positional encoding
        bs = args.batch_size * condition_frames
        self.img_ids, self.cond_ids, _ = prepare_ids(
            bs, self.h, self.w, self.total_token_size, 0
        )

        # ------------------------------------------------------------------ #
        # Auxiliary loss configuration
        # ------------------------------------------------------------------ #
        # Normalisation statistics (needed to recover metric xyz / range)
        self.proj_img_mean = list(
            getattr(args, 'proj_img_mean', [10.839, 0.005, 0.494, -1.13, 0.0, 0.0])
        )
        self.proj_img_stds = list(
            getattr(args, 'proj_img_stds', [9.314, 11.521, 8.262, 0.828, 1.0, 1.0])
        )

        # Loss weights (0.0 disables the respective term)
        self.range_view_loss_weight = float(getattr(args, 'range_view_loss_weight', 0.0))
        self.chamfer_loss_weight    = float(getattr(args, 'chamfer_loss_weight',    0.0))
        self.chamfer_max_pts        = int(getattr(args,   'chamfer_max_pts',        2048))

        if load_path is not None:
            state_dict = torch.load(load_path, map_location='cpu')["model_state_dict"]

            # Load STT model
            model_state_dict = self.model.state_dict()
            for k in model_state_dict.keys():
                if 'module.model.' + k in state_dict:
                    model_state_dict[k] = state_dict['module.model.' + k]
            self.model.load_state_dict(model_state_dict, strict=False)

            # Load DiT model
            dit_state_dict = self.dit.state_dict()
            for k in dit_state_dict.keys():
                if 'module.dit.' + k in state_dict:
                    dit_state_dict[k] = state_dict['module.dit.' + k]
            self.dit.load_state_dict(dit_state_dict, strict=False)

            print(f"Successfully loaded model from {load_path}")

    def model_forward(self, feature_total, rot_matrix, targets, rel_pose_cond=None, rel_yaw_cond=None, step=0):
        """
        Forward pass for training

        Args:
            feature_total: Range view features [B, F, L, C]
            rot_matrix: Rotation matrices [B, T, 4, 4]
            targets: Target range view features [B*F, L, C]
            rel_pose_cond: Optional conditioned relative pose
            rel_yaw_cond: Optional conditioned relative yaw
            step: Training step
        """
        # Get relative pose and yaw from rotation matrices
        if (rel_pose_cond is not None) and (rel_yaw_cond is not None):
            with torch.cuda.amp.autocast(enabled=False):
                rel_pose_gt, rel_yaw_gt = get_rel_pose(
                    rot_matrix[:, (self.condition_frames - 1) * self.args.block_size:(self.condition_frames + 1) * self.args.block_size]
                )
            rel_pose_total = torch.cat([rel_pose_cond, rel_pose_gt[:, -1:]], dim=1)
            rel_yaw_total = torch.cat([rel_yaw_cond, rel_yaw_gt[:, -1:]], dim=1)
        else:
            with torch.cuda.amp.autocast(enabled=False):
                rel_pose_total, rel_yaw_total = get_rel_pose(
                    rot_matrix[:, :(self.condition_frames + 1) * self.args.block_size]
                )

        # Convert pose and yaw to indices
        pose_indices_total = poses_to_indices(
            rel_pose_total, self.pose_x_vocab_size, self.pose_y_vocab_size
        )
        yaw_indices_total = yaws_to_indices(rel_yaw_total, self.yaw_vocab_size)

        # Get features from Spatial-Temporal Transformer
        logits = self.model(
            feature_total, pose_indices_total, yaw_indices_total,
            drop_feature=self.args.drop_feature
        )
        stt_features = logits['logits']
        pose_emb = logits['pose_emb']

        # Predict next frame using DiT
        loss_terms = self.dit.training_losses(
            img=targets,
            img_ids=self.img_ids,
            cond=stt_features,
            cond_ids=self.cond_ids,
            t=torch.rand((targets.shape[0], 1, 1), device=targets.device),
            y=pose_emb,
            return_predict=self.args.return_predict
        )

        diff_loss = loss_terms['loss']
        predict   = loss_terms['predict']

        # ------------------------------------------------------------------ #
        # Auxiliary losses (applied on the denoised *predict* estimate)
        # ------------------------------------------------------------------ #
        # ``predict`` is the clean-frame estimate: x_t + pred * (1 - t).
        # Computing reconstruction losses here gives image-space and 3-D
        # supervision that the flow-matching loss alone cannot provide.
        # Both losses are gated by their weight so they add zero overhead
        # when disabled (weight == 0).

        range_l1_loss   = torch.zeros(1, device=targets.device, dtype=targets.dtype)
        chamfer_loss_val = torch.zeros(1, device=targets.device, dtype=targets.dtype)

        if predict is not None and (self.range_view_loss_weight > 0 or self.chamfer_loss_weight > 0):
            # Valid-pixel mask derived from GT range channel
            gt_valid = make_valid_mask(
                targets,
                range_mean=self.proj_img_mean[0],
                range_std=self.proj_img_stds[0],
            )   # [B*F, L]

            if self.range_view_loss_weight > 0:
                # Per-pixel L1 on all 6 feature channels; masked to GT-valid pixels
                range_l1_loss = range_view_l1_loss(predict, targets, valid_mask=gt_valid)

            if self.chamfer_loss_weight > 0:
                # Extract unnormalised xyz from predicted and GT features
                pred_xyz = features_to_xyz(predict, self.proj_img_mean, self.proj_img_stds)
                gt_xyz   = features_to_xyz(targets, self.proj_img_mean, self.proj_img_stds)

                # Build valid mask for predictions from their own range channel
                pred_valid = make_valid_mask(
                    predict,
                    range_mean=self.proj_img_mean[0],
                    range_std=self.proj_img_stds[0],
                )   # [B*F, L]

                chamfer_loss_val = batch_chamfer_distance(
                    pred_xyz, gt_xyz,
                    pred_valid, gt_valid,
                    max_pts=self.chamfer_max_pts,
                )

        total_loss = (
            diff_loss
            + self.range_view_loss_weight * range_l1_loss
            + self.chamfer_loss_weight    * chamfer_loss_val
        )

        loss = {
            "loss_all":     total_loss,
            "loss_diff":    diff_loss,
            "loss_range_l1": range_l1_loss,
            "loss_chamfer":  chamfer_loss_val,
            "predict": None if not self.args.return_predict else predict,
        }
        return loss

    def step_train(self, features, rot_matrix, features_gt, rel_pose_cond=None, rel_yaw_cond=None, features_aug=None, step=0):
        """
        Training step

        Args:
            features: Input range view features [B, F, L, C]
            rot_matrix: Rotation matrices
            features_gt: Ground truth next frame features
            rel_pose_cond: Optional conditioned relative pose
            rel_yaw_cond: Optional conditioned relative yaw
            features_aug: Optional augmented features
            step: Training step
        """
        self.model.train()

        if features_aug is None:
            features_total = torch.cat([features, features_gt], dim=1)
        else:
            features_total = features_aug

        # Optional: Apply masking/noise augmentation
        pro = random.random()
        if pro < self.args.mask_data:
            mask = torch.bernoulli(random.uniform(0.7, 1) * torch.ones_like(features_total))
            mask = mask.round().to(dtype=torch.int64)
            noise = torch.randn_like(features_total)

            if random.random() < 0.5:
                LPF = ideal_low_pass_filter(
                    features_total.shape, d_s=random.uniform(0.5, 1), dims=(-1,)
                ).cuda()
                features_total = freq_mix(features_total, noise, LPF, dims=(-1,))
            else:
                features_total = mask * features_total + (1 - mask) * noise

        # Prepare targets
        targets = torch.cat([features, features_gt], dim=1)[:, 1:]
        targets = rearrange(targets, 'B F L C -> (B F) L C')

        # Forward pass
        loss = self.model_forward(
            features_total, rot_matrix, targets,
            rel_pose_cond=rel_pose_cond, rel_yaw_cond=rel_yaw_cond, step=step
        )
        return loss

    def forward(self, features, rot_matrix, features_gt, rel_pose_cond=None, rel_yaw_cond=None, features_aug=None, sample_last=True, step=0, **kwargs):
        """
        Forward pass - delegates to train or eval
        """
        if self.training:
            return self.step_train(
                features, rot_matrix, features_gt, rel_pose_cond, rel_yaw_cond, features_aug, step
            )
        else:
            # Compute relative poses from absolute rotation matrices before calling step_eval
            with torch.cuda.amp.autocast(enabled=False):
                rel_pose, rel_yaw = get_rel_pose(rot_matrix)
            return self.step_eval(features, rel_pose, rel_yaw, sample_last=sample_last)

    @torch.no_grad()
    def step_eval(self, features, rel_pose, rel_yaw, sample_last=True):
        """
        Evaluation step - predict next frame

        Args:
            features: Input range view features [B, F, L, C]
            rel_pose: Relative pose [B, T, 2]
            rel_yaw: Relative yaw [B, T, 1]
            sample_last: Whether to sample only the last frame

        Returns:
            predict_features: Predicted next frame features
        """
        self.model.eval()

        # Convert pose and yaw to indices
        pose_total = poses_to_indices(rel_pose, self.pose_x_vocab_size, self.pose_y_vocab_size)
        yaw_total = yaws_to_indices(rel_yaw, self.yaw_vocab_size)

        # Get features from Spatial-Temporal Transformer
        stt_features, pose_emb = self.model.evaluate(
            features, pose_total, yaw_total, sample_last=sample_last
        )

        bsz = stt_features.shape[0]
        img_ids, cond_ids, _ = prepare_ids(bsz, self.h, self.w, self.total_token_size, 0)

        # Predict next frame using DiT
        self.dit.eval()
        noise = torch.randn(bsz, self.img_token_size, self.vae_emb_dim).to(stt_features)
        timesteps = get_schedule(int(self.args.num_sampling_steps), self.img_token_size)
        predict_features = self.dit.sample(noise, img_ids, stt_features, cond_ids, pose_emb, timesteps)
        # Return token format [bsz, L, C] so callers (e.g., training loop) can rearrange as needed
        return predict_features

    def save_model(self, path, epoch, rank=0):
        """Save model checkpoint"""
        if rank == 0:
            torch.save({
                'model_state_dict': self.state_dict(),
                'epoch': epoch
            }, f'{path}/rangeview_dit_{epoch}.pkl')
