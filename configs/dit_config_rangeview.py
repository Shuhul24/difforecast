"""
Configuration file for Range View DiT Training
This config is for training the simplified DiT model on range view images
"""

# Random seed (KITTI experiment seed)
seed = 43

# ===== Dataset Configuration =====
# KITTI Odometry Dataset
kitti_root = '/DATA2/shuhul/kitti'  # Root path to KITTI dataset
kitti_sequences_path = '/DATA2/shuhul/kitti/dataset/sequences'  # Path to sequences
kitti_poses_path = '/DATA2/shuhul/kitti/poses'  # Path to ground truth poses

# KITTI sequence splits (following KITTI Odometry format)
train_sequences = [0, 1, 2, 3, 4, 5]  # Training sequences
val_sequences = [6, 7]  # Validation sequences
test_sequences = [8, 9, 10]  # Test sequences

# Point cloud file settings for KITTI
pc_extension = '.bin'  # Point cloud file extension
pc_dtype = 'float32'  # Data type for loading point cloud
pc_reshape = (-1, 4)  # KITTI: (x, y, z, intensity) - 4 channels only

# ===== Range View Projection Parameters =====
# KITTI Velodyne HDL-64E LiDAR specifications
fov_up = 3.0  # Upper field of view (degrees) for KITTI
fov_down = -25.0  # Lower field of view (degrees) for KITTI
fov_left = -180.0  # Left field of view (degrees)
fov_right = 180.0  # Right field of view (degrees)

# Range image resolution for KITTI
range_h = 64  # Height of range image (64-beam LiDAR)
range_w = 2048  # Width of range image (full 360-degree coverage)
range_channels = 2  # [range, intensity] — matches RangeLDM VAE in_channels=2

# Image size for processing (this is the range view size)
image_size = (64, 2048)  # Fixed for KITTI

# ===== Feature Normalization =====
# KITTI Odometry dataset statistics (computed from training sequences 0-5)
# Format: [range, intensity]
# Matches the 2-channel input/output of the RangeLDM VAE.
proj_img_mean = [10.839, 0.0]  # [range, intensity]
proj_img_stds = [9.314,  1.0]  # [range, intensity]

# ===== Training Parameters =====
downsample_fps = 10  # KITTI is at 10 Hz
condition_frames = 5  # Number of past frames (N_PAST_STEPS from KITTI config)
block_size = 1  # Temporal block size
forward_iter = 5  # Number of future frames to predict (N_FUTURE_STEPS from KITTI config)
multifw_perstep = 10  # Apply multi-forward every N steps

# Augmentation
mask_data = 0  # 1 means apply masking, 0 means no masking
pkeep = 0.7  # Percentage of latent codes to keep
paug = 0  # Augmentation probability
reverse_seq = False

# Data augmentation configuration
augmentation_config = {
    # Translation augmentation
    'p_transx': 0.5,
    'trans_xmin': -0.5,
    'trans_xmax': 0.5,
    'p_transy': 0.5,
    'trans_ymin': -0.5,
    'trans_ymax': 0.5,
    'p_transz': 0.5,
    'trans_zmin': -0.1,
    'trans_zmax': 0.1,

    # Rotation augmentation
    'p_rot_roll': 0.5,
    'rot_rollmin': -5.0,
    'rot_rollmax': 5.0,
    'p_rot_pitch': 0.5,
    'rot_pitchmin': -5.0,
    'rot_pitchmax': 5.0,
    'p_rot_yaw': 0.5,
    'rot_yawmin': -5.0,
    'rot_yawmax': 5.0,

    # Scale augmentation
    'p_scale': 0.5,
    'scale_min': 0.95,
    'scale_max': 1.05,
}

# ===== Model Configuration =====
# Spatial-Temporal Transformer
# Scaled down from [12,6,6]/2048/16 to fit single-GPU training:
# n_embd halved → params scale as n_embd², giving ~7× fewer trainable params (~370M).
# axes_dim_dit must sum to n_embd_dit // n_head_dit = 1024 // 8 = 128 (unchanged).
#
# With patch_size_h=4, patch_size_w=32:
#   latent_C = 4*4*32 = 512  (DiT in_channels; projected internally to n_embd_dit)
#   L        = 4*16   = 64   (img_token_size; feeds STT block_size)
n_layer = [6, 4, 4]  # Number of layers [STT causal, DiT double blocks, DiT single blocks]
n_head = 8  # Number of attention heads for STT (head_dim = n_embd // n_head = 128)
n_embd = 1024  # Embedding dimension for STT

# Diffusion Transformer (DiT)
n_embd_dit = 1024  # Hidden size for DiT
n_head_dit = 8  # Number of attention heads for DiT (head_dim = 1024 // 8 = 128)
axes_dim_dit = [16, 48, 64]  # Axes dimensions for rotary position encoding; must sum to n_embd_dit // n_head_dit = 128

# Pose/Trajectory encoding
pose_x_vocab_size = 128  # Vocabulary size for x-axis pose
pose_y_vocab_size = 128  # Vocabulary size for y-axis pose
yaw_vocab_size = 512  # Vocabulary size for yaw angle
pose_x_bound = 50  # Bound for x-axis pose (meters)
pose_y_bound = 10  # Bound for y-axis pose (meters)
yaw_bound = 12  # Bound for yaw angle (degrees)

# ===== RangeLDM VAE Tokenizer Configuration =====
# Range view images are encoded with the RangeLDM VAE (kitti360 config):
#   Encoder/Decoder: 2-ch range image → 4× spatial compression → z_channels=4
#   Architecture: ch=64, ch_mult=[1,2,4], circular=True, act='silu'
#
# vae_ckpt: path to a RangeLDM training checkpoint (.ckpt) whose state_dict
#   contains 'encoder.*' and 'decoder.*' keys.
#   Set to None to randomly initialise the VAE (not recommended for quality).
#
# vae_embed_dim: VAE latent channels (z_channels=4 for RangeLDM kitti360).
#
# downsample_size: VAE spatial compression factor (4× for RangeLDM).
#   Latent spatial dimensions:
#     h_lat_vae = 64   // 4 = 16
#     w_lat_vae = 2048 // 4 = 512
#
# patch_size_h / patch_size_w: Non-uniform patchification after VAE encoding.
#
#   The VAE latent for a 64×2048 range image is 16×512 (4× compression).
#   A square patch_size=8 gives a 2×64 token grid — the elevation axis is
#   crushed to just 2 tokens, destroying most vertical structure.
#
#   Using asymmetric patches preserves more elevation detail and can reduce
#   the total token count simultaneously:
#
#   patch_size_h  patch_size_w  h_tok  w_tok  L    latent_C  notes
#   -----------  ------------  -----  -----  ---  --------  ------
#       8             8          2     64    128    256     original (square)
#       4            32          4     16     64    512     half tokens, better shape ← default
#       4            16          4     32    128    256     same L, better shape
#       2            16          8     32    256    128     more tokens, best elevation
#
#   Changing patch_size_h / patch_size_w also changes:
#     latent_C  = vae_embed_dim × patch_size_h × patch_size_w
#     L         = h_tok × w_tok  (img_token_size)
#   The DiT in_channels must equal latent_C, so update n_embd_dit accordingly.
vae_ckpt = None  # set to path of pre-trained RangeLDM checkpoint if available
vae_embed_dim = 4        # RangeLDM z_channels
patch_size   = 8         # legacy square fallback (used when patch_size_h/w absent)
patch_size_h = 4         # elevation patch size  → h_tok = 16 // 4 = 4
patch_size_w = 32        # azimuth  patch size   → w_tok = 512 // 32 = 16
# Derived: L = 4×16 = 64,  latent_C = 4×4×32 = 512
add_decoder_temporal = False  # unused for RangeView path (DCAE-only)
temporal_patch_size = 1       # unused for RangeView path (DCAE-only)

# ===== Temporal Latent Encoder (RangeView-specific) =====
# Causal temporal + spatial attention applied in the patchified latent space
# [B, T, L=64, C=512] between the VAE encoder and the STT.
# (L and C are derived from patch_size_h / patch_size_w above.)
#
# When enabled, each conditioning frame's latent tokens attend (causally) to
# all past frames' tokens before being passed to the STT, embedding motion /
# delta information directly into the conditioning representation.
#
# All blocks are zero-initialised → identity residual at training start,
# so enabling this does not disturb a pretrained VAE checkpoint.
#
# n_temporal_blocks: number of interleaved (causal-time, spatial) pairs.
#   4 pairs ≈ 6.5 M extra parameters (dim=256, n_heads=8).
#   Start with 2–4; increase if the model has capacity to spare.
add_encoder_temporal = False   # set True to enable TemporalLatentEncoder
n_temporal_blocks = 4          # interleaved time+space block pairs

# Feature processing
downsample_size = 4   # RangeLDM VAE spatial compression factor (4×)
patch_size = 8        # Square fallback — used by DCAE path and getattr defaults only.
                      # RangeView path uses patch_size_h / patch_size_w (defined above).
drop_feature = 0  # Dropout probability for features

# ===== Diffusion Configuration =====
diffusion_model_type = "flow"  # Type of diffusion model
num_sampling_steps = 100  # Number of sampling steps during inference

# ===== VAE ELBO Loss (active only when vae_ckpt = None) =====
# When the VAE has no pretrained checkpoint, the ELBO gives the encoder and
# decoder a direct reconstruction objective so they learn a meaningful codec
# in parallel with the DiT.
#
# ELBO = NLL(rec_loss) + kl_weight * KL(q(z|x) || N(0,I))
#   rec_loss = vae_range_weight * L1(range) + vae_intensity_weight * L1(intensity)
#   NLL      = rec_loss / exp(logvar) + logvar   (learnable Laplacian NLL)
#   KL       = 0.5 * sum(mu² + sigma² - 1 - log_sigma²)
#
# Tuning guide (all losses operate on *normalised* features):
#   elbo_weight:          overall scale; 1.0 puts ELBO on par with diff loss (~0.05)
#   kl_weight:            1e-6–1e-4  (small → near-deterministic VAE, good for LDM)
#   vae_range_weight:     1.0  (depth channel; dominant signal)
#   vae_intensity_weight: 0.5  (intensity channel; lower weight, noisier signal)
#   vae_logvar_init:      0.0  (start with log σ²=0, i.e. σ=1; adapts during training)
elbo_weight          = 1.0    # weight on total ELBO loss term
                              # NLL is now mean over all pixels → ELBO ≈ 0.5-2 (same scale as diff_loss)
kl_weight            = 1e-6   # β-VAE KL weight (small keeps latents near standard normal)
vae_range_weight     = 1.0    # L1 weight for range/depth channel (normalised space)
vae_intensity_weight = 0.5    # L1 weight for intensity channel   (normalised space)
vae_logvar_init      = 0.0    # initial log-variance for NLL scaling

# ===== Auxiliary Loss Weights =====
# These are applied to the denoised *predict* estimate produced during training,
# giving direct image-space and 3-D geometric supervision on top of the
# flow-matching loss.
#
# range_view_loss_weight: scales the per-pixel L1 loss between the predicted
#   and GT normalised range-view features.  Start small (0.1) and tune up.
#   Set to 0.0 to disable.
#
# chamfer_loss_weight: scales the Chamfer Distance between the 3-D point
#   clouds recovered from the predicted and GT range-view depth maps.  The
#   depth channel (ch 0) is unnormalised to metres and back-projected via
#   RangeViewProjection (spherical geometry), so xyz channels in the feature
#   tensor are NOT required — works with any number of channels including the
#   default 2-ch [range, intensity] format.
#   Requires: git submodule update --init && pip install -e pyTorchChamferDistance
#   Chamfer values are typically in tens of metres²; start with 0.01.
#   Set to 0.0 to disable.
#
# chamfer_max_pts: maximum number of LiDAR points sampled per cloud before
#   computing the O(N*M) distance matrix.  Reduces memory and compute cost.
range_view_loss_weight = 0.1   # L1 between decode(pred_latent) and original GT range image:
                               # both pred and target pass through the same (learning) VAE
                               # decoder, making their difference small even for blank preds.
chamfer_loss_weight    = 0.05  # raised from 0.001 — Chamfer is the primary geometric signal;
                               # blank-prediction penalty in batch_chamfer_distance ensures
                               # empty predictions are no longer silently ignored.
chamfer_max_pts        = 2048  # max points used in Chamfer subsampling

# ===== Training Settings =====
return_predict = True  # Return predictions during training for visualization
diff_only = True  # Train only diffusion model (no trajectory planning)
no_pose = False  # Whether to use pose information

# ===== Output Directories =====
outdir = "/DATA2/shuhul/exp/ckpt"  # Checkpoint directory
logdir = "/DATA2/shuhul/exp/job_log"  # Log directory
tdir = "/DATA2/shuhul/exp/job_tboard"  # TensorBoard directory
validation_dir = "/DATA2/shuhul/exp/validation"  # Validation output directory

# ===== Data Loading =====
num_workers = 8  # Number of data loading workers

# ===== Distributed Training =====
distributed = True  # Enable distributed training

# ===== Example Usage =====
"""
To train with this config on KITTI:

# Recommended hyperparameters from KITTI experiment:
# - batch_size: 4
# - lr: 0.0003 (3e-4)
# - max_epoch: 100
# - num_gpus: 1

# Single GPU training
export NODES_NUM=1
export GPUS_NUM=1

torchrun --nnodes=$NODES_NUM --nproc_per_node=$GPUS_NUM \\
  scripts/train_rangeview.py \\
  --batch_size 4 \\
  --lr 0.0003 \\
  --exp_name "kitti-rangeview-training" \\
  --config configs/dit_config_rangeview.py \\
  --eval_steps 2000

# Multi-GPU training (if available)
export NODES_NUM=1
export GPUS_NUM=4

torchrun --nnodes=$NODES_NUM --nproc_per_node=$GPUS_NUM \\
  scripts/train_rangeview.py \\
  --batch_size 1 \\
  --lr 0.0003 \\
  --exp_name "kitti-rangeview-training-multigpu" \\
  --config configs/dit_config_rangeview.py \\
  --eval_steps 2000

# To resume from a checkpoint:
torchrun --nnodes=$NODES_NUM --nproc_per_node=$GPUS_NUM \\
  scripts/train_rangeview.py \\
  --batch_size 4 \\
  --lr 0.0003 \\
  --exp_name "kitti-rangeview-resume" \\
  --config configs/dit_config_rangeview.py \\
  --resume_path "exp/ckpt/kitti-rangeview-training/rangeview_dit_10000.pkl" \\
  --resume_step 10000 \\
  --eval_steps 2000
"""
