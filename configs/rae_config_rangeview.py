"""
Configuration for DINOv2-RAE Range View Pipeline

Two-stage training:
  Stage 1  train ViT-XL decoder with frozen DINOv2 encoder (RAE pre-training)
  Stage 2  train STT + FluxDiT in DINOv2 latent space (forecasting)

Key differences from dit_config_rangeview.py:
  - 5-channel input: [range, x, y, z, intensity]
  - No VAE; DINOv2 encoder + ViT-XL decoder used instead
  - Latent shape: [B, T, 256, 384]  (8×32 tokens × 384-d)
  - Stage 1 output rae_ckpt used as rae_ckpt for Stage 2
"""

seed = 43

# ── Dataset ─────────────────────────────────────────────────────────────────
kitti_root           = '/DATA2/shuhul/kitti'
kitti_sequences_path = '/DATA2/shuhul/kitti/dataset/sequences'
kitti_poses_path     = '/DATA2/shuhul/kitti/poses'

train_sequences = [0]           # expand to [0,1,2,3,4,5] for full training
val_sequences   = [6, 7]
test_sequences  = [8, 9, 10]

pc_extension = '.bin'
pc_dtype     = 'float32'
pc_reshape   = (-1, 4)          # KITTI: (x, y, z, intensity)

# ── Range view projection ────────────────────────────────────────────────────
fov_up    =  3.0
fov_down  = -25.0
fov_left  = -180.0
fov_right =  180.0
range_h   = 64
range_w   = 2048
image_size = (64, 2048)

# 2-channel input/output: range (log-normalised) + intensity (clipped)
# Following LiDARGen (Zyrianov et al.): log2(r+1)/6 for depth, clip [0,1] for intensity.
# xyz dropped — analytically derivable from range + sensor geometry, redundant for CD.
range_channels = 2
five_channel   = False
log_range      = True   # log2(depth+1)/6 normalisation; proj_img_mean/stds unused for range
proj_img_mean  = [0.0, 0.0]   # kept for API compat; not used when log_range=True
proj_img_stds  = [1.0, 1.0]   # kept for API compat; not used when log_range=True

# ── FUTURE (5-channel mean/std normalised) ────────────────────────────────────
# Stats: rough estimates — run scripts/calibrate_stats.py after first epoch
#   range_channels = 5
#   five_channel   = True
#   log_range      = False
#   proj_img_mean  = [10.839, 0.0,  0.0,  0.0, 0.0]   # [range, x, y, z, intensity]
#   proj_img_stds  = [ 9.314, 10.0, 10.0, 2.0, 1.0]
# ─────────────────────────────────────────────────────────────────────────────

# ── DINOv2 encoder ────────────────────────────────────────────────────────────
# Download dinov2_vits14_pretrain.pth from:
#   https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
dino_pretrained_path = '/DATA2/shuhul/dinov2_vits14_pretrain.pth'    # set to local .pth path, or None for torch.hub
dino_embed_dim       = 384     # ViT-S/14 hidden dim
dino_grid_h          = 8       # 64 // patch_stride_h=8
dino_grid_w          = 32      # 2048 // patch_stride_w=64
dino_n_patches       = 256     # 8×32

# ── ViT-XL Decoder (RAE Stage 1) ─────────────────────────────────────────────
# 2-channel weights: range dominant (40), intensity low (1).
rae_ch_weights = [40., 1.]

# Stage 1 RAE checkpoint used to initialise / freeze decoder in Stage 2
rae_ckpt = None    # e.g. 'outputs/rae_stage1/rae_step50000.pkl'

# ── Temporal configuration ───────────────────────────────────────────────────
condition_frames  = 5
forward_iter      = 5
block_size        = 1
multifw_perstep   = 1   # run all fw_iter AR passes every step (chain-of-forward)

# ── Augmentation ─────────────────────────────────────────────────────────────
drop_feature = 0
augmentation_config = {
    'p_transx': 0.5, 'trans_xmin': -0.5, 'trans_xmax': 0.5,
    'p_transy': 0.5, 'trans_ymin': -0.5, 'trans_ymax': 0.5,
    'p_transz': 0.5, 'trans_zmin': -0.1, 'trans_zmax': 0.1,
    'p_rot_roll':  0.5, 'rot_rollmin':  -5., 'rot_rollmax':  5.,
    'p_rot_pitch': 0.5, 'rot_pitchmin': -5., 'rot_pitchmax': 5.,
    'p_rot_yaw':   0.5, 'rot_yawmin':   -5., 'rot_yawmax':   5.,
    'p_scale': 0.5, 'scale_min': 0.95, 'scale_max': 1.05,
}

# ── STT ───────────────────────────────────────────────────────────────────────
# Same as dit_config_rangeview.py; STT projects 384-d tokens → n_embd=1024 internally
n_layer = [6, 8, 8]   # [STT causal, DiT double-stream, DiT single-stream]
n_head  = 8
n_embd  = 1024

# ── FluxDiT ───────────────────────────────────────────────────────────────────
# head_dim = n_embd_dit // n_head_dit = 768 // 12 = 64 = sum(axes_dim_dit)
n_embd_dit     = 768
n_head_dit     = 12
axes_dim_dit   = [16, 16, 32]
mlp_ratio_dit  = 4.0
drop_path_rate = 0.1

# ── Pose encoding ─────────────────────────────────────────────────────────────
pose_x_vocab_size = 128
pose_y_vocab_size = 128
yaw_vocab_size    = 512
pose_x_bound      = 50.
pose_y_bound      = 10.
yaw_bound         = 12.

# ── Diffusion ─────────────────────────────────────────────────────────────────
diffusion_model_type  = 'flow'
num_sampling_steps    = 100
return_predict        = True

# DINOv2 features are near unit-norm by construction, so latent_scale=1.0 is fine.
# Calibrate after Stage 1 if needed.
latent_scale = 1.0

# ── Auxiliary losses (Stage 2) ────────────────────────────────────────────────
range_view_loss_weight = 1.0   # L1 on decoded range + xyz + intensity
chamfer_loss_weight    = 0.0   # disabled; set > 0 after model stabilises
chamfer_max_pts        = 2048
chamfer_start          = 0
bev_perceptual_weight  = 0.1
bev_h, bev_w           = 256, 256
bev_x_range            = 25.6
bev_y_range            = 25.6

# ── Training ──────────────────────────────────────────────────────────────────
blr          = 1e-4
warmup_steps = 2000
weight_decay = 0.01
num_workers  = 8
distributed  = True

# ── Output directories ────────────────────────────────────────────────────────
outdir        = '/DATA2/shuhul/exp/rae_ckpt'
logdir        = '/DATA2/shuhul/exp/rae_log'
tdir          = '/DATA2/shuhul/exp/rae_tboard'
validation_dir = '/DATA2/shuhul/exp/rae_validation'

# ── Example usage ─────────────────────────────────────────────────────────────
"""
# Stage 1 — train RAE (ViT-XL decoder)
torchrun --nproc_per_node=1 scripts/train_rae_rangeview.py \
    --stage 1 --batch_size 4 --exp_name rae-stage1 \
    --config configs/rae_config_rangeview.py

# Stage 2 — train DiT forecaster (update rae_ckpt in config first)
torchrun --nproc_per_node=1 scripts/train_rae_rangeview.py \
    --stage 2 --batch_size 2 --exp_name rae-stage2 \
    --config configs/rae_config_rangeview.py
"""
