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

train_sequences = [0, 1, 2, 3, 4, 5]
val_sequences   = [6, 7]
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
swin_ckpt = '/DATA2/shuhul/exp/swin_ckpt/swin-s1-ch1-b32/swin_rae_step222000.pkl'

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
n_layer = [10, 8, 8]   # n_layer[0] = CausalTimeSpaceBlock depth (only this value builds blocks)
n_head  = 8
n_embd  = 1024

# ── FluxDiT (identical; latent dim 384 is same as DINOv2) ────────────────────
n_embd_dit     = 768
n_head_dit     = 12
axes_dim_dit   = [16, 16, 32]
mlp_ratio_dit  = 4.0
drop_path_rate = 0.1

# ── PoseDiT (single-step relative-pose prediction: x, y, yaw) ────────────────
# pe_dim = n_embd_dit_traj / n_head_dit_traj = 512/8 = 64 = sum(axes_dim_dit_traj)
n_embd_dit_traj   = 512
n_head_dit_traj   = 8
axes_dim_dit_traj = [16, 16, 32]   # sum = 64 = pe_dim
n_layer_traj      = [4, 4]         # [double_stream_blocks, single_stream_blocks]
lambda_yaw_pose   = 0.1            # weight for pose diffusion loss
return_predict_traj = True         # must be True — predictions used for AR conditioning

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
return_predict       = False      # no aux losses active — skip decoder forward during training
traj_len             = 1         # single future step for PoseDiT
latent_scale         = 0.9936    # TODO: replace with std from scripts/compute_latent_stats.py

# ── Temporal skip aggregation (Stage 2) ──────────────────────────────────────
# Replaces static last-frame skip injection with cross-attention pooling over
# all CF conditioning frames, preventing t+2..t+5 copy-frame artefacts.
temporal_skip_agg = True

# ── Auxiliary losses (Stage 2) ────────────────────────────────────────────────
range_view_loss_weight = 1.0
chamfer_loss_weight    = 0.0     # enable after model stabilises
chamfer_max_pts        = 2048
chamfer_start          = 0
bev_perceptual_weight  = 0.0     # disabled for Stage 1; re-enable (e.g. 0.1) for Stage 2 if desired
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
