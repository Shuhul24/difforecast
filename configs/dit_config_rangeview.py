"""
Configuration file for Range View DiT Training
This config is for training the simplified DiT model on range view images
"""

# Random seed (KITTI experiment seed)
seed = 43

# ===== Dataset Configuration =====
# KITTI Odometry Dataset
kitti_root = '/scratch/p24cs0005/kitti'  # Root path to KITTI dataset
kitti_sequences_path = '/scratch/p24cs0005/kitti/dataset/sequences'  # Path to sequences
kitti_poses_path = '/scratch/p24cs0005/kitti/poses'  # Path to ground truth poses

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
range_channels = 6  # Number of channels: [range, x, y, z, intensity, label] - label will be zeros for KITTI

# Image size for processing (this is the range view size)
image_size = (64, 2048)  # Fixed for KITTI

# ===== Feature Normalization =====
# KITTI Odometry dataset statistics (computed from training sequences 0-5)
# Format: [range, x, y, z, intensity, label]
# Note: label channel is all zeros for KITTI (no semantic labels)
proj_img_mean = [10.839, 0.005, 0.494, -1.13, 0.0, 0.0]
proj_img_stds = [9.314, 11.521, 8.262, 0.828, 1.0, 1.0]

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
n_layer = [12, 6, 6]  # Number of layers [STT, DiT double blocks, DiT single blocks]
n_head = 16  # Number of attention heads for STT
n_embd = 2048  # Embedding dimension for STT

# Diffusion Transformer (DiT)
n_embd_dit = 2048  # Hidden size for DiT
n_head_dit = 16  # Number of attention heads for DiT
axes_dim_dit = [16, 64, 64]  # Axes dimensions for rotary position encoding (adjusted for KITTI dimensions)

# Pose/Trajectory encoding
pose_x_vocab_size = 128  # Vocabulary size for x-axis pose
pose_y_vocab_size = 128  # Vocabulary size for y-axis pose
yaw_vocab_size = 512  # Vocabulary size for yaw angle
pose_x_bound = 50  # Bound for x-axis pose (meters)
pose_y_bound = 10  # Bound for y-axis pose (meters)
yaw_bound = 12  # Bound for yaw angle (degrees)

# Feature processing
downsample_size = 1  # Downsample factor for range images (usually 1 for range views)
patch_size = 1  # Patch size for tokenization
drop_feature = 0  # Dropout probability for features

# ===== Diffusion Configuration =====
diffusion_model_type = "flow"  # Type of diffusion model
num_sampling_steps = 100  # Number of sampling steps during inference

# ===== Training Settings =====
return_predict = True  # Return predictions during training for visualization
diff_only = True  # Train only diffusion model (no trajectory planning)
no_pose = False  # Whether to use pose information

# ===== Output Directories =====
outdir = "exp/ckpt"  # Checkpoint directory
logdir = "exp/job_log"  # Log directory
tdir = "exp/job_tboard"  # TensorBoard directory
validation_dir = "exp/validation"  # Validation output directory

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
