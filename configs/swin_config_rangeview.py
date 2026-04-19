"""
Configuration for TULIP-inspired Swin Transformer Range View Pipeline.

Two-stage training:
  Stage 1  train TULIPRangeEncoder + TULIPRangeDecoder (Swin RAE, with skip connections)
  Stage 2  train STT + FluxDiT in TULIP 4-stage Swin bottleneck latent space (forecasting)

Architecture:
  - 4-stage hierarchical Swin encoder/decoder (depths 2-6-2-2, SwinV2 attention)
  - Bottleneck: [B, 64, 768] — grid (2×32), embed_dim×8
  - Skip connections in decoder (U-Net style, as in TULIP)
  - No pretrained backbone; encoder trained from scratch with range-image priors
  - Circular padding on azimuth axis (azimuth wrap-around, from TULIP)
  - Berhu loss on range channel (better for sharp depth edges than plain L1)
  - swin_ckpt used for Stage 2 init (Stage 1 checkpoint)
"""

seed = 43

# ── Dataset ──────────────────────────────────────────────────────────────────
kitti_root           = '/DATA2/shuhul/kitti'
kitti_sequences_path = '/DATA2/shuhul/kitti/dataset/sequences'
kitti_poses_path     = '/DATA2/shuhul/kitti/poses'

train_sequences = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
val_sequences   = [8]
test_sequences  = [8, 9, 10]

pc_extension = '.bin'
pc_dtype     = 'float32'
pc_reshape   = (-1, 4)

# ── Range view projection ─────────────────────────────────────────────────────
fov_up    =  3.0
fov_down  = -25.0
fov_left  = -180.0
fov_right =  180.0
range_h   = 64
range_w   = 2048
image_size = (64, 2048)

# 1-channel [range/depth]; log2 normalisation.
# Validity mask is derived on-the-fly from depth (zero-depth = invalid pixel).
# Mask is used only to restrict the reconstruction loss — not encoded into the latent.
range_channels = 1
five_channel   = False
log_range      = True
mask_channel   = False
proj_img_mean  = [0.0]
proj_img_stds  = [1.0]

# ── Swin Transformer encoder / decoder ───────────────────────────────────────
# Patch size (4, 8) on 64×2048 → initial grid (16, 256) = 4096 patches.
# After 3 PatchMerging ops: (2, 32) = 64 tokens at dim=768 → [B, 64, 768].
# SwinV2 cpb_mlp takes 2-D log-space relative coords; window (2,8) → table [3,15,2].
swin_embed_dim   = 96
swin_depths      = (2, 6, 2, 2)       # TULIP-base depths; stage-1 has 6 blocks
swin_num_heads   = (3, 6, 12, 24)    # attention heads per stage
swin_window_size = (2, 8)             # asymmetric window for 64×2048 range images
swin_mlp_ratio   = 4.0
swin_drop_rate   = 0.0
swin_attn_drop   = 0.0
swin_drop_path   = 0.1                # stochastic depth max rate
swin_v2          = True               # use SwinV2 attention (cosine + cpb_mlp)

# ── Stage 1 RAE loss weights ──────────────────────────────────────────────────
# ch 0 = range only; distance-weighted Berhu, masked to valid (non-zero) pixels.
rae_ch_weights      = [1.]
dist_weighted_loss  = True   # up-weight far pixels (2× at 80m) to improve fidelity

# Stage 1 Swin-RAE checkpoint for Stage 2 init
swin_ckpt = '/DATA2/shuhul/exp/swin_ckpt/swin-s1-ch1-b32/swin_rae_step346000.pkl'

# ── Temporal configuration ────────────────────────────────────────────────────
condition_frames = 5
forward_iter     = 5
block_size       = 1
multifw_perstep  = 1

# ── Augmentation ─────────────────────────────────────────────────────────────
# Point-cloud augmentation is disabled for forecasting:
# each frame in a window was augmented independently with different random
# rotations/translations, making the augmented inter-frame geometry inconsistent
# with the GT relative poses — a direct source of pose/image mismatch in Stage 2.
# (Adapted from DiffLoc which applied augmentation to single-frame pose regression.)
drop_feature = 0
augmentation_config = None

# ── STT (identical to rae_config_rangeview.py) ────────────────────────────────
# n_layer[0]=14 CausalTimeSpaceBlocks (was 10); n_embd=1280 (was 1024)
n_layer = [14, 12, 12]
n_head  = 10
n_embd  = 1280

# ── FluxDiT ───────────────────────────────────────────────────────────────────
# hidden_size 768→1024, 12 double + 12 single blocks (was 8+8), 16 heads (was 12)
# axes_dim must sum to head_dim = n_embd_dit / n_head_dit = 1024/16 = 64
n_embd_dit     = 1280
n_head_dit     = 20             # head_dim = 1280/20 = 64 = sum(axes_dim_dit)
axes_dim_dit   = [16, 16, 32]   # sum=64 = head_dim
mlp_ratio_dit  = 4.0
drop_path_rate = 0.1

# ── PoseDiT (single-step relative-pose prediction: x, y, yaw) ────────────────
# pe_dim = n_embd_dit_traj / n_head_dit_traj = 512/8 = 64 = sum(axes_dim_dit_traj)
n_embd_dit_traj   = 512
n_head_dit_traj   = 8
axes_dim_dit_traj = [16, 16, 32]   # sum = 64 = pe_dim
n_layer_traj      = [6, 6]         # was [4,4]; deeper for better pose representation
lambda_yaw_pose        = 2.0    # weight for pose diffusion loss (raised from 0.1 — gradient ratio was ~1:3350 pose:diff; 20× increase balances contribution)
return_predict_traj    = True   # must be True — predictions used for AR conditioning

# ── Pose encoding ─────────────────────────────────────────────────────────────
pose_x_vocab_size = 128
pose_y_vocab_size = 128
yaw_vocab_size    = 512
pose_x_bound      = 50.
pose_y_bound      = 10.
yaw_bound         = 12.

# ── Diffusion ─────────────────────────────────────────────────────────────────
diffusion_model_type = 'flow'
num_sampling_steps   = 100
return_predict       = True       # run decoder forward during training to compute rv + chamfer losses
traj_len             = 1         # single future step for PoseDiT
latent_scale         = 0.9936    # TODO: replace with std from scripts/compute_latent_stats.py

# ── Temporal skip aggregation (Stage 2) ──────────────────────────────────────
# Replaces static last-frame skip injection with cross-attention pooling over
# all CF conditioning frames, preventing t+2..t+5 copy-frame artefacts.
temporal_skip_agg = True

# ── Auxiliary losses (Stage 2) ────────────────────────────────────────────────
range_view_loss_weight = 0.05   # pixel-space L1 on decoded prediction vs decoded GT; gradients still flow through frozen decoder into DiT latents
# ── VAE KL regularisation (Stage 1) ──────────────────────────────────────────
# β warmup: KL weight ramps from 0 → kl_weight over kl_warmup_steps so that
# reconstruction quality stabilises before KL pressure kicks in.
# Free bits (0.5 nats/dim) prevents posterior collapse on sparse LiDAR latents.
kl_weight       = 1e-4   # final β; raise to 1e-3 if latents remain unnormalised
kl_warmup_steps = 10000  # ramp duration in Stage 1 training steps

chamfer_loss_weight    = 0.1     # λ_chamfer — explicit weight (no uncertainty weighting)
chamfer_max_pts        = 2048
chamfer_start          = 200     # earlier start so TemporalSkipAggregator gets supervision (rv_loss removed)
ar_warmup_steps        = 2000    # steps of single-step AR before enabling full fw_iter rollout

# ── REPA (Representation Alignment) ──────────────────────────────────────────
# Aligns FluxDiT double_blocks[repa_layer_idx] hidden state with the frozen
# Swin encoder's output for the clean GT target frame (lat_target).
# Zero extra encoder forward pass — lat_target already computed in step_train.
# Tune repa_weight ∈ [0.05, 0.5]; start small and increase if loss_diff stalls.
repa_weight        = 0.1           # λ_repa
repa_layer_idx     = 4             # middle of depth=8 double blocks (0-indexed)
repa_start_step    = 0             # enable from step 0; set >0 to delay until flow loss stabilises
repa_warmup_steps  = 500           # ramp REPA from 0→repa_weight over this many steps to prevent early domination

pose_reg_weight        = 0.5     # λ for physical-unit L1 pose regression (metres/degrees); prevents PoseDiT from collapsing to mean velocity
bev_perceptual_weight  = 0.0     # disabled; re-enable (e.g. 0.1) if BEV quality matters
bev_h, bev_w           = 256, 256
bev_x_range            = 25.6
bev_y_range            = 25.6

# ── Training ──────────────────────────────────────────────────────────────────
blr          = 1e-4
warmup_steps = 2000
weight_decay = 0.01
num_workers  = 4
distributed  = True

# ── Output directories ────────────────────────────────────────────────────────
outdir         = '/DATA2/shuhul/exp/swin_ckpt'
logdir         = '/DATA2/shuhul/exp/swin_log'
tdir           = '/DATA2/shuhul/exp/swin_tboard'
validation_dir = '/DATA2/shuhul/exp/swin_validation'

# ── Example usage ─────────────────────────────────────────────────────────────
"""
# Stage 1 — train Swin RAE
torchrun --nproc_per_node=1 scripts/train_swin_rangeview.py \
    --stage 1 --batch_size 4 --exp_name swin-stage1 \
    --config configs/swin_config_rangeview.py

# Stage 2 — update swin_ckpt in config, then:
torchrun --nproc_per_node=1 scripts/train_swin_rangeview.py \
    --stage 2 --batch_size 2 --exp_name swin-stage2 \
    --config configs/swin_config_rangeview.py
"""
