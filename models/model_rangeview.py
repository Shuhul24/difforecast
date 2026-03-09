"""
DiT Model for Range View Image Prediction with DCAE Tokenization.

This model mirrors the encoding pattern used by Epona's main TrainTransformersDiT:
  raw range view images → DCAE encoder → latent tokens → STT → DiT → latent tokens
                       → DCAE decoder → predicted range view images

The DCAE tokenizer compresses the range view image (default 64×2048) by 32×
spatially, producing (2×64) = 128 latent tokens per frame with 32 channels.
This makes the token count tractable for the Spatial-Temporal Transformer and
Diffusion Transformer.  Without DCAE the raw feature count (64×2048 = 131 072
tokens) would be prohibitively large.

Key differences from model.py:
  - No TrajDiT (next-frame prediction only, no trajectory planning)
  - RangeViewVAETokenizer used instead of VAETokenizer (6-channel input)
  - Auxiliary losses (L1, Chamfer) computed on DCAE-decoded predictions
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
from models.modules.tokenizer import (
    RangeViewVAETokenizer,
    poses_to_indices,
    yaws_to_indices,
)
from utils.fft_utils import freq_mix, ideal_low_pass_filter
from models.modules.sampling import prepare_ids, get_schedule
from utils.range_losses import (
    make_valid_mask,
    range_view_l1_loss,
    RangeViewProjection,
    batch_chamfer_distance,
)


class RangeViewDiT(nn.Module):
    """DiT model for range view image prediction using DCAE tokenization.

    Input pipeline (training):
        range_views [B, T, C, H, W]
            → RangeViewVAETokenizer.encode_to_z
            → latents [B, T, L, latent_C]
            → SpatialTemporalTransformer
            → FluxDiT (flow-matching)
            → predicted latents [B*F, L, latent_C]
            → RangeViewVAETokenizer.decode_from_z  (for auxiliary losses)
            → predicted range views [B*F, C, H, W]

    The DCAE encoder/decoder weights are frozen; only the STT and DiT
    parameters are trained.
    """

    def __init__(
        self,
        args,
        local_rank: int = -1,
        load_path=None,
        condition_frames: int = 3,
    ):
        super().__init__()
        self.local_rank = local_rank
        self.args = args
        self.condition_frames = condition_frames

        # ------------------------------------------------------------------ #
        # DCAE tokenizer (frozen; 32× spatial compression)
        # ------------------------------------------------------------------ #
        self.vae_tokenizer = RangeViewVAETokenizer(args, local_rank)

        # Latent embedding dimension after patchification
        # vae_embed_dim = DCAE latent channels (32 for f32c32)
        # vae_emb_dim   = latent_channels × patch_size²
        self.vae_emb_dim = self.args.vae_embed_dim * self.args.patch_size ** 2

        # ------------------------------------------------------------------ #
        # Spatial dimensions in latent space
        # downsample_size = 32 (DCAE f32c32 compression factor)
        # ------------------------------------------------------------------ #
        self.image_size = self.args.image_size
        self.h = self.image_size[0] // (self.args.downsample_size * self.args.patch_size)
        self.w = self.image_size[1] // (self.args.downsample_size * self.args.patch_size)
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
            'total_tokens_size': self.total_token_size,
        }

        # ------------------------------------------------------------------ #
        # Spatial-Temporal Transformer (operates on latent tokens)
        # ------------------------------------------------------------------ #
        self.model = SpatialTemporalTransformer(
            block_size=condition_frames * self.total_token_size,
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
            temporal_block=self.args.block_size,
        )
        self.model.cuda()

        # ------------------------------------------------------------------ #
        # Diffusion Transformer (predicts next-frame latents)
        # ------------------------------------------------------------------ #
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

        # Positional encoding IDs (precomputed for efficiency)
        bs = args.batch_size * condition_frames
        self.img_ids, self.cond_ids, _ = prepare_ids(
            bs, self.h, self.w, self.total_token_size, 0
        )

        # ------------------------------------------------------------------ #
        # Auxiliary loss configuration
        # ------------------------------------------------------------------ #
        self.proj_img_mean = list(
            getattr(args, 'proj_img_mean', [10.839, 0.005, 0.494, -1.13, 0.0, 0.0])
        )
        self.proj_img_stds = list(
            getattr(args, 'proj_img_stds', [9.314, 11.521, 8.262, 0.828, 1.0, 1.0])
        )
        self.range_view_loss_weight = float(getattr(args, 'range_view_loss_weight', 0.0))
        self.chamfer_loss_weight    = float(getattr(args, 'chamfer_loss_weight',    0.0))
        self.chamfer_max_pts        = int(getattr(args,   'chamfer_max_pts',        2048))

        # Spherical back-projection for Chamfer distance loss.
        # Precomputes per-pixel (x, y, z) ray-direction factors from the
        # LiDAR FOV so that xyz = depth * factor at training time.
        self.range_projector = RangeViewProjection(
            fov_up=float(getattr(args, 'fov_up',   3.0)),
            fov_down=float(getattr(args, 'fov_down', -25.0)),
            H=int(getattr(args, 'range_h', 64)),
            W=int(getattr(args, 'range_w', 2048)),
        )

        # ------------------------------------------------------------------ #
        # Optional checkpoint loading (STT + DiT only; tokenizer is separate)
        # ------------------------------------------------------------------ #
        if load_path is not None:
            state_dict = torch.load(load_path, map_location='cpu')["model_state_dict"]

            model_state_dict = self.model.state_dict()
            for k in model_state_dict.keys():
                if 'module.model.' + k in state_dict:
                    model_state_dict[k] = state_dict['module.model.' + k]
            self.model.load_state_dict(model_state_dict, strict=False)

            dit_state_dict = self.dit.state_dict()
            for k in dit_state_dict.keys():
                if 'module.dit.' + k in state_dict:
                    dit_state_dict[k] = state_dict['module.dit.' + k]
            self.dit.load_state_dict(dit_state_dict, strict=False)

            print(f"Successfully loaded STT + DiT from {load_path}")

    # ---------------------------------------------------------------------- #
    # Core forward pass (training)
    # ---------------------------------------------------------------------- #

    def model_forward(
        self,
        latents_total,
        rot_matrix,
        latent_targets,
        rel_pose_cond=None,
        rel_yaw_cond=None,
        step=0,
    ):
        """Forward pass for training.

        Args:
            latents_total:   ``[B, F, L, latent_C]`` — condition + target latents
                             (may have augmentation noise applied).
            rot_matrix:      ``[B, T, 4, 4]`` absolute rotation matrices.
            latent_targets:  ``[(B*F), L, latent_C]`` — clean target latents
                             used for flow-matching supervision.
            rel_pose_cond:   Optional pre-computed relative pose  (B, T, 2).
            rel_yaw_cond:    Optional pre-computed relative yaw   (B, T, 1).
            step:            Training step counter.

        Returns:
            dict with keys: loss_all, loss_diff, loss_range_l1, loss_chamfer,
            predict (``[(B*F), C, H, W]`` decoded range view or None).
        """
        # Relative pose / yaw indices
        if (rel_pose_cond is not None) and (rel_yaw_cond is not None):
            with torch.cuda.amp.autocast(enabled=False):
                rel_pose_gt, rel_yaw_gt = get_rel_pose(
                    rot_matrix[
                        :,
                        (self.condition_frames - 1) * self.args.block_size:
                        (self.condition_frames + 1) * self.args.block_size,
                    ]
                )
            rel_pose_total = torch.cat([rel_pose_cond, rel_pose_gt[:, -1:]], dim=1)
            rel_yaw_total  = torch.cat([rel_yaw_cond,  rel_yaw_gt[:, -1:]],  dim=1)
        else:
            with torch.cuda.amp.autocast(enabled=False):
                rel_pose_total, rel_yaw_total = get_rel_pose(
                    rot_matrix[:, :(self.condition_frames + 1) * self.args.block_size]
                )

        pose_indices_total = poses_to_indices(
            rel_pose_total, self.pose_x_vocab_size, self.pose_y_vocab_size
        )
        yaw_indices_total = yaws_to_indices(rel_yaw_total, self.yaw_vocab_size)

        # Spatial-Temporal Transformer
        logits = self.model(
            latents_total, pose_indices_total, yaw_indices_total,
            drop_feature=self.args.drop_feature,
        )
        stt_features = logits['logits']
        pose_emb     = logits['pose_emb']

        # Flow-matching loss (DiT)
        loss_terms = self.dit.training_losses(
            img=latent_targets,
            img_ids=self.img_ids,
            cond=stt_features,
            cond_ids=self.cond_ids,
            t=torch.rand((latent_targets.shape[0], 1, 1), device=latent_targets.device),
            y=pose_emb,
            return_predict=self.args.return_predict,
        )
        diff_loss = loss_terms['loss']
        predict   = loss_terms['predict']   # [(B*F), L, latent_C] or None

        # ------------------------------------------------------------------ #
        # Auxiliary losses — computed on DCAE-decoded predictions
        # ------------------------------------------------------------------ #
        # Decode the DiT's x_0 estimate back to range view feature space so
        # we can supervise with pixel-level L1 and 3-D Chamfer distance.
        # The decode is NOT wrapped in torch.no_grad() so that gradients flow
        # from these losses through the frozen DCAE decoder back to `predict`
        # and ultimately to the DiT parameters.
        # ------------------------------------------------------------------ #
        range_l1_loss    = torch.zeros(1, device=latent_targets.device, dtype=latent_targets.dtype)
        chamfer_loss_val = torch.zeros(1, device=latent_targets.device, dtype=latent_targets.dtype)
        predict_decoded  = None

        if predict is not None:
            # Decode predicted latents → range view features [(B*F), C, H, W]
            predict_decoded = self.vae_tokenizer.decode_from_z(predict, self.h, self.w)

            if self.range_view_loss_weight > 0 or self.chamfer_loss_weight > 0:
                # Decode target latents for supervision (no grad needed for targets)
                with torch.no_grad():
                    targets_decoded = self.vae_tokenizer.decode_from_z(
                        latent_targets.detach(), self.h, self.w
                    )  # [(B*F), C, H, W]

                # Reshape to [..., L, C] for aux-loss helpers
                pred_tok   = rearrange(predict_decoded, 'b c h w -> b (h w) c')
                target_tok = rearrange(targets_decoded, 'b c h w -> b (h w) c')

                gt_valid = make_valid_mask(
                    target_tok,
                    range_mean=self.proj_img_mean[0],
                    range_std=self.proj_img_stds[0],
                )  # [(B*F), L]

                if self.range_view_loss_weight > 0:
                    range_l1_loss = range_view_l1_loss(
                        pred_tok, target_tok, valid_mask=gt_valid
                    )

                if self.chamfer_loss_weight > 0:
                    range_mean = self.proj_img_mean[0]
                    range_std  = self.proj_img_stds[0]

                    # Unnormalize depth channel (ch 0) from feature space to metres.
                    # Works for any number of feature channels — no xyz channels needed.
                    pred_depth = pred_tok[..., 0] * range_std + range_mean    # [B, L]
                    gt_depth   = target_tok[..., 0] * range_std + range_mean  # [B, L]

                    pred_valid = make_valid_mask(
                        pred_tok,
                        range_mean=range_mean,
                        range_std=range_std,
                    )

                    chamfer_loss_val = batch_chamfer_distance(
                        pred_depth, gt_depth,
                        pred_valid, gt_valid,
                        projector=self.range_projector,
                        max_pts=self.chamfer_max_pts,
                    )

        total_loss = (
            diff_loss
            + self.range_view_loss_weight * range_l1_loss
            + self.chamfer_loss_weight    * chamfer_loss_val
        )

        return {
            "loss_all":      total_loss,
            "loss_diff":     diff_loss,
            "loss_range_l1": range_l1_loss,
            "loss_chamfer":  chamfer_loss_val,
            # predict_decoded: [(B*F), C, H, W] or None
            # Callers use this for autoregressive conditioning (next iteration).
            "predict": predict_decoded,
        }

    # ---------------------------------------------------------------------- #
    # Training step
    # ---------------------------------------------------------------------- #

    def step_train(
        self,
        features,
        rot_matrix,
        features_gt,
        rel_pose_cond=None,
        rel_yaw_cond=None,
        features_aug=None,
        step=0,
    ):
        """Training step.

        Args:
            features:     ``[B, CF, C, H, W]`` condition range view images.
            rot_matrix:   ``[B, T, 4, 4]`` absolute rotation matrices.
            features_gt:  ``[B, 1, C, H, W]``  next-frame ground truth.
            rel_pose_cond:  Optional pre-computed relative pose.
            rel_yaw_cond:   Optional pre-computed relative yaw.
            features_aug:   Not used (reserved for future augmentation).
            step:         Training step counter.
        """
        self.model.train()

        # Concatenate all frames and encode to latent space with frozen DCAE
        features_all = torch.cat([features, features_gt], dim=1)  # [B, CF+1, C, H, W]
        latents_all  = self.vae_tokenizer.encode_to_z(features_all)  # [B, CF+1, L, latent_C]

        latents_total = latents_all  # may be replaced by augmented version below

        # Optional: frequency-domain / masking augmentation on latent tokens
        pro = random.random()
        if pro < self.args.mask_data:
            mask  = torch.bernoulli(random.uniform(0.7, 1) * torch.ones_like(latents_total))
            mask  = mask.round().to(dtype=torch.int64)
            noise = torch.randn_like(latents_total)

            if random.random() < 0.5:
                LPF = ideal_low_pass_filter(
                    latents_total.shape, d_s=random.uniform(0.5, 1), dims=(-1,)
                ).cuda()
                latents_total = freq_mix(latents_total, noise, LPF, dims=(-1,))
            else:
                latents_total = mask * latents_total + (1 - mask) * noise

        # Targets: latents of all frames shifted by 1 (each frame predicts the next)
        latent_targets = latents_all[:, 1:]                          # [B, CF, L, latent_C]
        latent_targets = rearrange(latent_targets, 'B F L C -> (B F) L C')

        return self.model_forward(
            latents_total, rot_matrix, latent_targets,
            rel_pose_cond=rel_pose_cond, rel_yaw_cond=rel_yaw_cond, step=step,
        )

    # ---------------------------------------------------------------------- #
    # Unified forward
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        features,
        rot_matrix,
        features_gt,
        rel_pose_cond=None,
        rel_yaw_cond=None,
        features_aug=None,
        sample_last=True,
        step=0,
        **kwargs,
    ):
        """Delegate to training or evaluation step."""
        if self.training:
            return self.step_train(
                features, rot_matrix, features_gt,
                rel_pose_cond, rel_yaw_cond, features_aug, step,
            )
        else:
            with torch.cuda.amp.autocast(enabled=False):
                rel_pose, rel_yaw = get_rel_pose(rot_matrix)
            return self.step_eval(features, rel_pose, rel_yaw, sample_last=sample_last)

    # ---------------------------------------------------------------------- #
    # Evaluation step
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def step_eval(self, features, rel_pose, rel_yaw, sample_last=True):
        """Predict next range view frame from conditioning frames.

        Args:
            features:    ``[B, CF, C, H, W]`` conditioning range view images.
            rel_pose:    ``[B, T, 2]`` relative translations.
            rel_yaw:     ``[B, T, 1]`` relative yaw angles.
            sample_last: If True, only the last condition frame is used to
                         initialise the sampling (matches training behaviour).

        Returns:
            ``[bsz, C, H, W]`` decoded predicted next-frame range view.
        """
        self.model.eval()

        # Encode conditioning frames to latent space
        feature_latents = self.vae_tokenizer.encode_to_z(features)  # [B, CF, L, latent_C]

        # Pose / yaw indices
        pose_total = poses_to_indices(rel_pose, self.pose_x_vocab_size, self.pose_y_vocab_size)
        yaw_total  = yaws_to_indices(rel_yaw,  self.yaw_vocab_size)

        # Spatial-Temporal Transformer
        stt_features, pose_emb = self.model.evaluate(
            feature_latents, pose_total, yaw_total, sample_last=sample_last
        )

        bsz = stt_features.shape[0]
        img_ids, cond_ids, _ = prepare_ids(bsz, self.h, self.w, self.total_token_size, 0)

        # Diffusion sampling in latent space
        self.dit.eval()
        noise     = torch.randn(bsz, self.img_token_size, self.vae_emb_dim).to(stt_features)
        timesteps = get_schedule(int(self.args.num_sampling_steps), self.img_token_size)
        predict_latents = self.dit.sample(noise, img_ids, stt_features, cond_ids, pose_emb, timesteps)
        # predict_latents: [bsz, L, latent_C]

        # Decode latents back to range view feature space
        predict_features = self.vae_tokenizer.decode_from_z(predict_latents, self.h, self.w)
        # predict_features: [bsz, C, H, W]

        return predict_features

    # ---------------------------------------------------------------------- #
    # Checkpoint
    # ---------------------------------------------------------------------- #

    def save_model(self, path, epoch, rank=0):
        """Save model checkpoint."""
        if rank == 0:
            torch.save(
                {'model_state_dict': self.state_dict(), 'epoch': epoch},
                f'{path}/rangeview_dit_{epoch}.pkl',
            )
