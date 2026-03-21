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
import contextlib
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
from utils.bev_perceptual import BEVPerceptualLoss


class RangeViewVAE(nn.Module):
    """
    VAE-only model wrapper for Stage 1 training.
    Trains the encoder/decoder using ELBO loss without the DiT/STT overhead.
    """
    def __init__(self, args, local_rank=-1):
        super().__init__()
        self.args = args
        self.local_rank = local_rank

        self.vae_tokenizer = RangeViewVAETokenizer(args, local_rank)

        if hasattr(self.vae_tokenizer, 'vae'):
            for p in self.vae_tokenizer.vae.parameters():
                p.requires_grad = True

        self.elbo_weight          = float(getattr(args, 'elbo_weight',          1.0))
        self.vae_range_weight     = float(getattr(args, 'vae_range_weight',     40.0))
        self.vae_intensity_weight = float(getattr(args, 'vae_intensity_weight', 10.0))
        self.kl_weight            = float(getattr(args, 'kl_weight',            1e-6))
        self.proj_img_mean        = list(getattr(args, 'proj_img_mean', [10.839, 0.0]))
        self.proj_img_stds        = list(getattr(args, 'proj_img_stds', [9.314,  1.0]))

        logvar_init = float(getattr(args, 'vae_logvar_init', 0.0))
        self.register_buffer('logvar', torch.tensor(logvar_init, dtype=torch.float32))

        self.range_projector = RangeViewProjection(
            fov_up=float(getattr(args, 'fov_up',   3.0)),
            fov_down=float(getattr(args, 'fov_down', -25.0)),
            H=int(getattr(args, 'range_h', 64)),
            W=int(getattr(args, 'range_w', 2048)),
        )

        import math as _math
        self.bev_perceptual_weight = float(getattr(args, 'bev_perceptual_weight', 0.0))
        if self.bev_perceptual_weight > 0:
            self.log_w_bev = nn.Parameter(
                torch.tensor(_math.log(1.0 / self.bev_perceptual_weight), dtype=torch.float32)
            )
            self.bev_perceptual_fn = BEVPerceptualLoss(
                projector=self.range_projector,
                bev_h=int(getattr(args, 'bev_h', 256)),
                bev_w=int(getattr(args, 'bev_w', 256)),
                x_range=float(getattr(args, 'bev_x_range', 25.6)),
                y_range=float(getattr(args, 'bev_y_range', 25.6)),
            )
        else:
            self.log_w_bev         = None
            self.bev_perceptual_fn = None

        print(f"RangeViewVAE initialized. logvar_init={logvar_init}, "
              f"bev_perceptual_weight={self.bev_perceptual_weight}")

    def forward(self, features, step=0):
        """
        Args:
            features: [B, C, H, W] Input range view images
            step: Training step
        """
        elbo_loss, x_recon, nll_loss = self.vae_tokenizer.compute_vae_elbo(
            features,
            logvar=self.logvar,
            range_weight=self.vae_range_weight,
            intensity_weight=self.vae_intensity_weight,
            kl_weight=self.kl_weight,
            return_recon=True,
        )
        total_loss = self.elbo_weight * elbo_loss

        bev_percep_loss = torch.zeros(1, device=features.device, dtype=features.dtype)
        if self.bev_perceptual_weight > 0:
            range_std  = self.proj_img_stds[0]
            range_mean = self.proj_img_mean[0]

            pred_depth = (x_recon[:, 0] * range_std + range_mean).reshape(features.shape[0], -1)
            gt_depth   = (features[:, 0] * range_std + range_mean).reshape(features.shape[0], -1)
            pred_valid = pred_depth > 0.5
            gt_valid   = gt_depth   > 0.5

            bev_percep_loss = self.bev_perceptual_fn(pred_depth, gt_depth, pred_valid, gt_valid)
            lw = self.log_w_bev.clamp(min=0.0)
            total_loss = total_loss + torch.exp(-lw) * bev_percep_loss + lw

        return {
            "loss_all":        total_loss,
            "loss_elbo":       elbo_loss,
            "loss_bev_percep": bev_percep_loss,
            "nll_loss":        nll_loss,
            "x_recon":         x_recon,
            "predict":         None,
        }

    def save_model(self, path, epoch, rank=0):
        if rank == 0:
            torch.save(
                {'model_state_dict': self.state_dict(), 'epoch': epoch},
                f'{path}/rangeview_vae_{epoch}.pkl',
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

        # Non-uniform patch support: patch_size_h / patch_size_w may differ.
        # Falls back to square patch_size when only patch_size is set.
        patch_h = int(getattr(args, 'patch_size_h', args.patch_size))
        patch_w = int(getattr(args, 'patch_size_w', args.patch_size))

        # Latent embedding dimension after patchification:
        # vae_emb_dim = vae_embed_dim × patch_size_h × patch_size_w
        self.vae_emb_dim = self.args.vae_embed_dim * patch_h * patch_w

        # Spatial token grid dimensions in patchified latent space
        self.image_size = self.args.image_size
        self.h = self.image_size[0] // (self.args.downsample_size * patch_h)
        self.w = self.image_size[1] // (self.args.downsample_size * patch_w)
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

        # Positional encoding IDs (precomputed for efficiency).
        # prefix_size = number of non-image tokens prepended by the STT:
        #   [yaw(1), pose_x(1), pose_y(1), img(h*w)] → prefix = 3
        # Passing this ensures spatial RoPE is assigned to the image-token
        # slots in cond_ids, not to the pose/yaw tokens.
        bs = args.batch_size * condition_frames
        _cond_prefix = self.total_token_size - self.img_token_size  # = 3
        self.img_ids, self.cond_ids, _ = prepare_ids(
            bs, self.h, self.w, self.total_token_size, 0, prefix_size=_cond_prefix
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

        # ------------------------------------------------------------------ #
        # Learned uncertainty weights for auxiliary losses (Kendall et al.)
        #
        # Instead of manual weight * loss, we use:
        #   exp(-log_w) * loss + log_w
        # where log_w is a learnable parameter.  This auto-calibrates each
        # loss to a common scale while the +log_w regulariser prevents the
        # weight from collapsing to zero (infinite log_w).
        #
        # Initialised so exp(-log_w_init) == the manual weight, making the
        # learned-weight run identical to the manual-weight run at step 0.
        # ------------------------------------------------------------------ #
        import math as _math
        if self.range_view_loss_weight > 0:
            _log_w_l1_init = _math.log(1.0 / self.range_view_loss_weight)
            self.log_w_l1 = nn.Parameter(
                torch.tensor(_log_w_l1_init, dtype=torch.float32)
            )
        else:
            self.log_w_l1 = None

        if self.chamfer_loss_weight > 0:
            _log_w_cd_init = _math.log(1.0 / self.chamfer_loss_weight)
            self.log_w_chamfer = nn.Parameter(
                torch.tensor(_log_w_cd_init, dtype=torch.float32)
            )
        else:
            self.log_w_chamfer = None

        # ------------------------------------------------------------------ #
        # BEV perceptual loss — VGG16 multi-scale feature distance on BEV
        # occupancy grids (inspired by RangeLDM bev_perceptual branch).
        # ------------------------------------------------------------------ #
        self.bev_perceptual_weight = float(getattr(args, 'bev_perceptual_weight', 0.0))
        if self.bev_perceptual_weight > 0:
            _log_w_bev_init = _math.log(1.0 / self.bev_perceptual_weight)
            self.log_w_bev = nn.Parameter(
                torch.tensor(_log_w_bev_init, dtype=torch.float32)
            )
            # BEVPerceptualLoss is instantiated here; the RangeViewProjection
            # is shared with Chamfer (created below) — we store args and build
            # bev_perceptual_fn after range_projector is constructed.
            self._build_bev_loss = True
        else:
            self.log_w_bev     = None
            self._build_bev_loss = False

        # ------------------------------------------------------------------ #
        # VAE ELBO loss (used when vae_ckpt is None — VAE trained from scratch)
        #
        # When no pre-trained VAE checkpoint is provided the encoder/decoder
        # start from random weights and have no external reconstruction
        # objective.  The ELBO loss gives them a direct training signal:
        #
        #   ELBO = NLL(reconstruction) + kl_weight * KL(posterior || N(0,I))
        #
        # where NLL uses a learnable log-variance (Laplacian NLL formulation
        # matching RangeLDM's RangeImageReconstructionLoss).
        #
        # The ELBO is computed on the raw input frames (before patchification)
        # using a reparameterised sample, so gradients reach both the encoder
        # (via KL) and the decoder (via reconstruction).  The DiT pipeline
        # continues to use the deterministic mode() latents for stability.
        # ------------------------------------------------------------------ #
        vae_ckpt = getattr(args, 'vae_ckpt', None)
        self.vae_is_trainable = (vae_ckpt is None)

        self.elbo_weight          = float(getattr(args, 'elbo_weight',          1.0))
        self.kl_weight            = float(getattr(args, 'kl_weight',            1e-6))
        self.vae_range_weight     = float(getattr(args, 'vae_range_weight',     1.0))
        self.vae_intensity_weight = float(getattr(args, 'vae_intensity_weight', 0.5))

        if self.vae_is_trainable:
            logvar_init = float(getattr(args, 'vae_logvar_init', 0.0))
            self.register_buffer(
                'logvar', torch.ones(size=(), dtype=torch.float32) * logvar_init
            )
            print(
                f"RangeViewDiT: VAE is trainable (no checkpoint). "
                f"ELBO loss enabled — elbo_weight={self.elbo_weight}, "
                f"kl_weight={self.kl_weight}, "
                f"vae_range_weight={self.vae_range_weight}, "
                f"vae_intensity_weight={self.vae_intensity_weight}"
            )
        else:
            self.logvar = None
            print("RangeViewDiT: VAE is frozen (checkpoint loaded). ELBO disabled.")

        # Spherical back-projection for Chamfer distance loss.
        # Precomputes per-pixel (x, y, z) ray-direction factors from the
        # LiDAR FOV so that xyz = depth * factor at training time.
        self.range_projector = RangeViewProjection(
            fov_up=float(getattr(args, 'fov_up',   3.0)),
            fov_down=float(getattr(args, 'fov_down', -25.0)),
            H=int(getattr(args, 'range_h', 64)),
            W=int(getattr(args, 'range_w', 2048)),
        )

        # BEV perceptual loss — built after range_projector so it can be shared.
        if self._build_bev_loss:
            self.bev_perceptual_fn = BEVPerceptualLoss(
                projector=self.range_projector,
                bev_h=int(getattr(args, 'bev_h', 256)),
                bev_w=int(getattr(args, 'bev_w', 256)),
                x_range=float(getattr(args, 'bev_x_range', 25.6)),
                y_range=float(getattr(args, 'bev_y_range', 25.6)),
            )
        else:
            self.bev_perceptual_fn = None

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

            # When vae_ckpt is None the VAE was trained jointly and its weights
            # are saved inside the DiT checkpoint under 'module.vae_tokenizer.*'.
            # Restore them here so eval uses the trained VAE, not random weights.
            if self.vae_is_trainable:
                vae_state_dict = self.vae_tokenizer.state_dict()
                loaded_any = False
                for k in vae_state_dict.keys():
                    ckpt_key = 'module.vae_tokenizer.' + k
                    if ckpt_key in state_dict:
                        vae_state_dict[k] = state_dict[ckpt_key]
                        loaded_any = True
                self.vae_tokenizer.load_state_dict(vae_state_dict, strict=False)
                if loaded_any:
                    print(f"Successfully loaded VAE tokenizer weights from {load_path}")
                else:
                    print(f"Warning: no VAE tokenizer weights found in {load_path} "
                          f"(keys expected under 'module.vae_tokenizer.*')")

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
        gt_images=None,
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
            gt_images:       ``[(B*F), C, H, W]`` original (unnormalised-then-
                             normalised) GT range view images.  When provided,
                             range L1 is computed against these raw pixels rather
                             than re-decoded latents, so the target is never
                             corrupted by a still-learning VAE decoder.

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
            rel_pose_total, self.pose_x_vocab_size, self.pose_y_vocab_size,
            x_range=float(self.pose_x_bound), y_range=float(self.pose_y_bound * 2),
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
        # t_sample is kept as a named variable so auxiliary losses can be
        # weighted by it: d(predict)/d(pred) = (1-t), so weighting auxiliary
        # losses by t inverts that scaling — giving max gradient at t≈1
        # (clean-data regime, reliable predictions) and ~zero at t≈0 (noise).
        t_sample = torch.rand((latent_targets.shape[0], 1, 1), device=latent_targets.device)
        loss_terms = self.dit.training_losses(
            img=latent_targets,
            img_ids=self.img_ids,
            cond=stt_features,
            cond_ids=self.cond_ids,
            t=t_sample,
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
        bev_percep_loss  = torch.zeros(1, device=latent_targets.device, dtype=latent_targets.dtype)
        predict_decoded  = None

        if predict is not None:
            # Decode predicted latents → range view image [(B*F), C, H, W]
            predict_decoded = self.vae_tokenizer.decode_from_z(predict, self.h, self.w)

            if self.range_view_loss_weight > 0 or self.chamfer_loss_weight > 0 or self.bev_perceptual_weight > 0:
                range_mean = self.proj_img_mean[0]
                range_std  = self.proj_img_stds[0]

                # ---------------------------------------------------------- #
                # GT target: use the original range view image (gt_images)
                # instead of re-decoding the GT latents.  This ensures the
                # target is always the clean 64×2048 pixel image, not an
                # approximation produced by a still-learning VAE decoder.
                # ---------------------------------------------------------- #
                if gt_images is not None:
                    gt_range_img = gt_images                 # [(B*F), C, H, W]
                else:
                    # Fallback (e.g. during eval) — decode GT latents as before
                    with torch.no_grad():
                        gt_range_img = self.vae_tokenizer.decode_from_z(
                            latent_targets.detach(), self.h, self.w
                        )

                # Valid mask from the original GT image (range ch > 0.5 m).
                # Invalid pixels (empty LiDAR returns projected to 0) are excluded
                # from both the L1 and Chamfer losses.
                gt_range_unnorm = gt_range_img[:, 0] * range_std + range_mean  # [B*F, H, W]
                gt_valid_spatial = gt_range_unnorm > 0.5                        # [B*F, H, W]

                # ---- Range L1 loss (pred decoded image vs original GT) ---- #
                if self.range_view_loss_weight > 0:
                    pred_range = predict_decoded[:, 0]       # [B*F, H, W]
                    gt_range   = gt_range_img[:, 0]          # [B*F, H, W]
                    l1_range   = torch.abs(pred_range - gt_range)            # [B*F, H, W]
                    # Also supervise the intensity channel (ch 1) — previously
                    # unsupervised, giving the decoder no gradient signal on it.
                    # Weight 0.25 keeps intensity contribution smaller than range
                    # (intensity_weight / range_weight = 10/40 = 0.25 in ELBO).
                    if predict_decoded.shape[1] > 1 and gt_range_img.shape[1] > 1:
                        l1_intensity = torch.abs(predict_decoded[:, 1] - gt_range_img[:, 1])
                        l1_map = l1_range + 0.25 * l1_intensity
                    else:
                        l1_map = l1_range
                    mask_f     = gt_valid_spatial.float()
                    # t_sample [B*F, 1, 1] broadcasts over H, W — per-sample
                    # t-weighting so high-noise steps (t≈0) contribute near-zero
                    # gradient while clean-data steps (t≈1) contribute fully.
                    range_l1_loss = (l1_map * mask_f * t_sample).sum() / (mask_f.sum() + 1e-8)

                # ---- Depth maps shared by Chamfer + BEV perceptual --------- #
                if self.chamfer_loss_weight > 0 or self.bev_perceptual_weight > 0:
                    pred_range_unnorm = (
                        predict_decoded[:, 0] * range_std + range_mean
                    )  # [B*F, H, W]

                    # Flatten spatial dims: [B*F, H*W]
                    pred_depth = pred_range_unnorm.reshape(pred_range_unnorm.shape[0], -1)
                    gt_depth   = gt_range_unnorm.reshape(gt_range_unnorm.shape[0], -1)
                    pred_valid = pred_depth > 0.5
                    gt_valid   = gt_valid_spatial.reshape(gt_valid_spatial.shape[0], -1)

                # ---- Chamfer loss (pred decoded depth vs original GT) ----- #
                if self.chamfer_loss_weight > 0:
                    chamfer_loss_val = batch_chamfer_distance(
                        pred_depth, gt_depth,
                        pred_valid, gt_valid,
                        projector=self.range_projector,
                        max_pts=self.chamfer_max_pts,
                    ) * t_sample.mean()

                # ---- BEV perceptual loss (VGG16 on BEV occupancy grid) ---- #
                if self.bev_perceptual_weight > 0:
                    bev_percep_loss = self.bev_perceptual_fn(
                        pred_depth, gt_depth, pred_valid, gt_valid
                    ) * t_sample.mean()

        # Uncertainty-weighted auxiliary losses: exp(-log_w)*L + log_w
        # log_w is clamped to >= 0 so the effective weight never exceeds exp(0)=1.0,
        # preventing any aux loss from sending gradients larger than the raw loss
        # value itself — this stops a drifting log_w from destabilising the DiT.
        def _uw(log_w, loss):
            lw = log_w.clamp(min=0.0)
            return torch.exp(-lw) * loss + lw

        aux_l1 = (
            _uw(self.log_w_l1, range_l1_loss)
            if self.log_w_l1 is not None else range_l1_loss.new_zeros(())
        )
        aux_cd = (
            _uw(self.log_w_chamfer, chamfer_loss_val)
            if self.log_w_chamfer is not None else chamfer_loss_val.new_zeros(())
        )
        aux_bev = (
            _uw(self.log_w_bev, bev_percep_loss)
            if self.log_w_bev is not None else bev_percep_loss.new_zeros(())
        )
        total_loss = diff_loss + aux_l1 + aux_cd + aux_bev

        return {
            "loss_all":       total_loss,
            "loss_diff":      diff_loss,
            "loss_range_l1":  range_l1_loss,
            "loss_chamfer":   chamfer_loss_val,
            "loss_bev_percep": bev_percep_loss,
            # predict_decoded: [(B*F), C, H, W] or None — decoded pixel images (for vis / L1)
            "predict": predict_decoded,
            # predict_latents: [(B*F), L, latent_C] or None — raw DiT x_0 estimates BEFORE
            # decoding.  Used for autoregressive conditioning so subsequent steps can receive
            # the latents directly, avoiding the lossy decode → re-encode round trip that
            # train_deepspeed.py avoids by operating entirely in latent space.
            "predict_latents": predict,
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
        latents_cond_precomputed=None,
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
            latents_cond_precomputed: ``[B, CF, L, latent_C]`` optional pre-computed
                conditioning latents from the previous autoregressive step.  When
                provided the conditioning frames are NOT re-encoded from ``features``,
                eliminating the lossy decode → re-encode round trip that degrades
                autoregressive quality.  Only the GT frame is encoded.
                Mirrors the approach in train_deepspeed.py which keeps ``latents_cond``
                in latent space throughout the autoregressive chain.
        """
        self.model.train()

        # Concatenate all frames and encode to latent space
        features_all = torch.cat([features, features_gt], dim=1)  # [B, CF+1, C, H, W]
        # When the VAE is frozen (stage 2), wrap encoding in no_grad to avoid
        # building an unnecessary autograd graph through the frozen encoder,
        # which would waste GPU activation memory without contributing gradients.
        _enc_ctx = torch.no_grad() if not self.vae_is_trainable else contextlib.nullcontext()
        with _enc_ctx:
            if latents_cond_precomputed is not None:
                # ---------------------------------------------------------- #
                # Latent-chain autoregressive conditioning: skip re-encoding
                # the conditioning frames and use the DiT's x_0 estimates from
                # the previous step directly.
                #
                # train_deepspeed.py does this naturally because it operates
                # entirely in latent space (latents_cond is [B, CF, L, C]).
                # Here we replicate that by bypassing encode_to_z for the CF
                # conditioning frames when predicted latents are available.
                #
                # Only the GT frame needs to be encoded (it is always real data).
                # ---------------------------------------------------------- #
                latents_gt_enc = self.vae_tokenizer.encode_to_z(features_gt)   # [B, 1, L, latent_C]
                latents_all = torch.cat(
                    [latents_cond_precomputed, latents_gt_enc], dim=1
                )  # [B, CF+1, L, latent_C]
            else:
                latents_all = self.vae_tokenizer.encode_to_z(features_all)  # [B, CF+1, L, latent_C]

        # ------------------------------------------------------------------ #
        # VAE ELBO loss (only when VAE has no pretrained checkpoint)
        #
        # Computed on a reparameterised sample from the encoder posterior so
        # that gradients reach all VAE parameters via:
        #   - KL term  →  encoder (mean + logvar parameters)
        #   - NLL term →  encoder (via z_sample) + decoder
        #
        # The DiT pipeline above uses mode() latents (deterministic) for
        # training stability — the ELBO branch is a separate side-objective.
        # ------------------------------------------------------------------ #
        elbo_loss = torch.zeros(1, device=features.device, dtype=features.dtype)
        if self.vae_is_trainable:
            B_all, T_all, C, H, W = features_all.shape
            if latents_cond_precomputed is not None:
                # Conditioning frames are predictions — compute ELBO only on the
                # real GT frame to avoid drifting from the true data distribution.
                x_flat = features_gt.reshape(B_all * features_gt.shape[1], C, H, W)
            else:
                x_flat = features_all.reshape(B_all * T_all, C, H, W)
            elbo_loss = self.vae_tokenizer.compute_vae_elbo(
                x_flat,
                logvar=self.logvar,
                range_weight=self.vae_range_weight,
                intensity_weight=self.vae_intensity_weight,
                kl_weight=self.kl_weight,
            )

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

        # Original GT range view images for each target frame — used in range L1
        # loss so the target is the raw pixel image, not a re-decoded latent.
        # Shape: [(B*CF), C, H, W]
        gt_images = rearrange(
            features_all[:, 1:].detach(), 'B F C H W -> (B F) C H W'
        )

        result = self.model_forward(
            latents_total, rot_matrix, latent_targets,
            rel_pose_cond=rel_pose_cond, rel_yaw_cond=rel_yaw_cond, step=step,
            gt_images=gt_images,
        )

        # Fold ELBO into total loss (zero when VAE is frozen / ckpt is set).
        if self.vae_is_trainable:
            result["loss_all"]  = result["loss_all"] + self.elbo_weight * elbo_loss
            result["loss_elbo"] = elbo_loss
        else:
            result["loss_elbo"] = torch.zeros(1, device=features.device, dtype=features.dtype)

        return result

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
        latents_cond_precomputed=None,
        **kwargs,
    ):
        """Delegate to training or evaluation step."""
        if self.training:
            return self.step_train(
                features, rot_matrix, features_gt,
                rel_pose_cond, rel_yaw_cond, features_aug, step,
                latents_cond_precomputed=latents_cond_precomputed,
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
        pose_total = poses_to_indices(
            rel_pose, self.pose_x_vocab_size, self.pose_y_vocab_size,
            x_range=float(self.pose_x_bound), y_range=float(self.pose_y_bound * 2),
        )
        yaw_total  = yaws_to_indices(rel_yaw,  self.yaw_vocab_size)

        # Spatial-Temporal Transformer
        stt_features, pose_emb = self.model.evaluate(
            feature_latents, pose_total, yaw_total, sample_last=sample_last
        )

        bsz = stt_features.shape[0]
        _cond_prefix = self.total_token_size - self.img_token_size
        img_ids, cond_ids, _ = prepare_ids(bsz, self.h, self.w, self.total_token_size, 0, prefix_size=_cond_prefix)

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
