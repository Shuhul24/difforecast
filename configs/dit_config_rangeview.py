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
# Stage-wise training is supported by the script via the ``--stage`` CLI
# option.  Use ``stage=1`` to pretrain the VAE only (ELBO loss with STT/DiT
# frozen), ``stage=2`` to train the DiT/STT portion with a frozen VAE
# (supply a pretrained ``vae_ckpt`` or resume from a stage‑1 checkpoint),
# and ``stage=all`` (the default) to run the full pipeline as before.
downsample_fps = 10  # KITTI is at 10 Hz
condition_frames = 5  # Number of past frames (N_PAST_STEPS from KITTI config)
block_size = 1  # Temporal block size
forward_iter = 5  # Number of future frames to predict (N_FUTURE_STEPS from KITTI config)
multifw_perstep = 2   # Apply multi-forward every N steps (was 10; more frequent autoregressive training)

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
# With patch_size_h=2, patch_size_w=32, vae_embed_dim=4:
#   latent_C = 4*2*32 = 256  (DiT in_channels; img_in projects 256 → n_embd_dit)
#   L        = 8*16   = 128  (img_token_size; 8 elevation × 16 azimuth tokens)
#   axes_dim_dit must sum to n_embd_dit // n_head_dit = 512 // 8 = 64
# n_layer = [STT causal blocks, DiT double-stream blocks, DiT single-stream blocks]
#
# Scaling guide (48 GB GPU, batch_size=6):
#   [6, 6, 6]  → baseline    (~57 M DiT params)
#   [6, 8, 8]  → +2 per DiT stream, moderate depth gain  (~100 M DiT params) ← current
#
# Increasing n_layer[0] deepens the STT temporal encoder.
# Increasing n_layer[1] / n_layer[2] deepens the DiT diffusion backbone.
n_layer = [6, 8, 8]  # [STT causal=6, DiT double=8, DiT single=8]
n_head = 8  # Number of attention heads for STT (head_dim = n_embd // n_head = 128)
n_embd = 1024  # Embedding dimension for STT

# Diffusion Transformer (DiT)
#
# Scaling rationale (48 GB GPU, batch_size=6):
#   512-hidden / 8-head  → 6+6 blocks  ≈  57 M params  (baseline)
#   768-hidden / 12-head → 8+8 blocks  ≈ 170 M params  (3× capacity, fits 48 GB)
#
# head_dim = n_embd_dit // n_head_dit must equal sum(axes_dim_dit).
#   768 // 12 = 64  →  axes_dim_dit sums to 64  ✓  (unchanged from baseline)
n_embd_dit = 768   # was 512; +50% hidden capacity for richer feature learning
n_head_dit = 12    # was 8; 12 heads × 64 head_dim = 768 ✓
axes_dim_dit = [16, 16, 32]  # Axes for rotary PE; must sum to n_embd_dit // n_head_dit = 64

# DiT MLP expansion ratio (mlp_hidden = hidden_size × mlp_ratio_dit).
# 4.0 is standard; raise to 4.5 for more feed-forward capacity with modest
# memory overhead (≈+12.5% MLP params).
mlp_ratio_dit = 4.0  # default; safe to raise to 4.5 if memory permits

# Stochastic depth (drop path) for DiT double- and single-stream blocks.
# Rates are distributed linearly from 0 → drop_path_rate across all blocks,
# so early blocks retain full gradient flow while later blocks are regularised.
# 0.1 is a good starting value for ~16-block DiTs (DeiT-III, DiT literature).
# Set to 0.0 to disable (equivalent to the original architecture).
drop_path_rate = 0.1  # peak stochastic-depth rate across DiT blocks

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
#       4            32          4     16     64    512     half tokens, better shape
#       4            16          4     32    128    256     same L, better shape      ← default
#       2            16          8     32    256    128     more tokens, best elevation
#
#   Changing patch_size_h / patch_size_w also changes:
#     latent_C  = vae_embed_dim × patch_size_h × patch_size_w
#     L         = h_tok × w_tok  (img_token_size)
#   The DiT in_channels must equal latent_C, so update n_embd_dit accordingly.
vae_ckpt = None  # set to path of pre-trained RangeLDM checkpoint if available
vae_embed_dim = 4        # RangeLDM z_channels
patch_size   = 8         # legacy square fallback (used when patch_size_h/w absent)
patch_size_h = 2         # elevation patch size  → h_tok = 16 // 2 = 8  (was 4, doubled elevation resolution)
patch_size_w = 32        # azimuth  patch size   → w_tok = 512 // 32 = 16
# Derived: L = 8×16 = 128,  latent_C = 4×2×32 = 256  (same token count, same in_channels, better elevation)
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
add_encoder_temporal = True    # enable TemporalLatentEncoder (zero-init, won't disturb VAE checkpoint)
n_temporal_blocks = 6          # was 4; +2 block pairs → richer inter-frame motion encoding
                                # Each pair ≈ 1.6 M extra params (dim=256, n_heads=8). Safe at batch_size=6.

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
#   elbo_weight:          scale ELBO down since range_weight=40 makes NLL ~10-40×larger;
#                         use 0.01-0.05 to keep ELBO contribution on par with diff_loss (~0.05)
#   kl_weight:            1e-6–1e-4  (small → near-deterministic VAE, good for LDM)
#   vae_range_weight:     40.0  matches RangeLDM — depth channel heavily weighted
#   vae_intensity_weight: 10.0  matches RangeLDM — intensity channel
#   vae_logvar_init:      0.0  (start with log σ²=0, i.e. σ=1; adapts during training)
elbo_weight          = 0.02   # Enabled: VAE trains from scratch (vae_ckpt=None), needs reconstruction objective.
                              # Without ELBO the encoder has no direct signal → unstructured latent space →
                              # DiT cannot learn a meaningful velocity field → loss stays at ~0.8–1.0.
                              # 0.02 keeps ELBO contribution on par with diff_loss (~0.05); see tuning guide above.
kl_weight            = 1e-6   # β-VAE KL weight — matches RangeLDM (small keeps latents near standard normal)
vae_range_weight     = 40.0   # L1 weight for range/depth channel — matches RangeLDM (was 1.0)
vae_intensity_weight = 10.0   # L1 weight for intensity channel   — matches RangeLDM (was 0.5)
vae_logvar_init      = 0.0    # initial log-variance for NLL scaling

# ===== Auxiliary Loss Weights =====
# These control the auxiliary losses (Range L1, Chamfer) applied on top of the
# flow-matching diffusion loss.
#
# The model uses *learned uncertainty weighting* (Kendall et al., NeurIPS 2018)
# instead of fixed multipliers.  Each auxiliary loss L_i is combined as:
#
#   exp(-log_w_i) * L_i + log_w_i
#
# where log_w_i is a learnable nn.Parameter.  The +log_w_i regulariser prevents
# the weight from collapsing to zero (i.e. the loss being ignored).
#
# These static values are used ONLY to initialise log_w_i at training start:
#   log_w_i_init = ln(1 / weight)
# so that step-0 behaviour is identical to the manual-weight run, and the model
# then adapts the weights freely.
#
# Set a weight to 0.0 to disable the corresponding loss entirely (no parameter).
#
# chamfer_max_pts: max points per cloud for the O(N*M) distance kernel.
range_view_loss_weight = 1.0  # pixel-space depth L1 supervision through frozen decoder
                                # Enabled from step 0: the t_sample weighting (range_l1 ∝ t)
                                # already provides a natural curriculum — noisy timesteps (t≈0)
                                # contribute ~zero gradient while clean-data steps (t≈1) contribute
                                # fully.  Starting from step 0 gives the DiT essential pixel-level
                                # "what a correct range view looks like" signal to anchor early training.
chamfer_loss_weight    = 0.5   # 3D point-cloud geometry loss — enabled but gated by chamfer_start.
                                # Effective weight adapts via learned uncertainty weighting (Kendall et al.)
chamfer_start          = 100_000  # Step at which Chamfer loss is first activated.
                                  # Rationale: Chamfer requires structurally coherent predictions.
                                  # Before this, range L1 teaches basic depth structure; Chamfer
                                  # gradients on incoherent early predictions are noisy and can
                                  # destabilise training (mirrors disc_start=50_000 in the VAE).
                                  # Set to 0 to enable from step 0.
chamfer_max_pts        = 2048  # max points used in Chamfer subsampling

# ===== BEV Perceptual Loss =====
# Converts depth maps → BEV occupancy grids → VGG16 multi-scale featudre distance.
# Penalises structural/shape errors (missing walls, broken objects) that L1 misses.
# Inspired by RangeLDM's bev_perceptual branch (losses/__init__.py L267-275).
#
# bev_perceptual_weight: init for learned uncertainty weight (exp(-log_w) at step 0).
#   VGG16 feature distances are ~0.1–0.5 per scale; 0.1 keeps BEV loss ~flow loss scale.
#   Set to 0.0 to disable (no VGG16 loaded, no extra memory cost).
#
# bev_h / bev_w: BEV grid resolution in pixels. 256×256 covers ±25.6 m at 0.2 m/cell.
# bev_x_range / bev_y_range: half-extent of the BEV grid in metres.
bev_perceptual_weight = 0.1   # Enabled: VGG16 structural supervision on BEV occupancy grids.
                              # Penalises shape/structural errors (missing walls, broken objects) that
                              # pixel-level L1 on the range channel misses. 0.1 keeps BEV contribution
                              # on par with flow loss scale.
bev_h         = 256           # BEV grid height (forward direction)
bev_w         = 256           # BEV grid width  (lateral direction)
bev_x_range   = 25.6          # ±25.6 m forward coverage
bev_y_range   = 25.6          # ±25.6 m lateral coverage

# ===== Stage 1 Discriminator Configuration =====
# Adversarial training for the VAE encoder-decoder following the RangeLDM
# training recipe (GeneralLPIPSWithDiscriminator, kitti360.yaml).
#
# disc_start:      Step at which the discriminator loss is activated.
#                  The ELBO reconstruction loss runs alone for the first
#                  disc_start steps so the VAE learns a sensible codec
#                  before adversarial pressure is applied.
#                  RangeLDM uses 200 000; 50 000 is a good starting point
#                  for KITTI Odometry (~20 000 steps/epoch at bs=4, 8 GPU).
#
# disc_weight:     Scalar multiplier on the adaptive GAN generator weight.
#                  Adaptive weight = disc_weight × ||∇NLL|| / ||∇g_loss||
#                  (Esser et al., VQGAN).  0.5 matches RangeLDM.
#
# disc_factor:     Hard multiplier on disc loss once disc_start is reached.
#                  Set to 0.0 to disable the discriminator entirely.
#
# disc_ndf:        Base filter count for the discriminator (64 in RangeLDM).
# disc_num_layers: PatchGAN conv layers (3 in RangeLDM).
# disc_lr:         Discriminator Adam learning rate. RangeLDM uses the same base_learning_rate
#                  (4.5e-6) for both the VAE and discriminator. Previously 2e-4.
#
# disc_resume_path: Optional path to a previously saved disc_stepN.pth to
#                   resume discriminator training.
disc_start      = 50000  # Activate discriminator after this many steps
disc_weight     = 0.5    # Adaptive GAN weight scale (matches RangeLDM)
disc_factor     = 1.0    # Hard scale on disc loss after disc_start
disc_ndf        = 64     # Discriminator base filter count
disc_num_layers = 3      # PatchGAN conv layers
disc_lr         = 4.5e-6  # Discriminator Adam learning rate (matches RangeLDM base_learning_rate)
disc_resume_path = None  # Path to resume discriminator from (optional)

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

torchrun --nnodes=$NODES_NUM --nproc_per_node=$GPUS_NUM \
  scripts/train_rangeview.py \
  --batch_size 4 \
  --lr 0.0003 \
  --exp_name "kitti-rangeview-training" \
  --config configs/dit_config_rangeview.py \
  --eval_steps 2000

# Multi-GPU training (if available)
export NODES_NUM=1
export GPUS_NUM=4

torchrun --nnodes=$NODES_NUM --nproc_per_node=$GPUS_NUM \
  scripts/train_rangeview.py \
  --batch_size 1 \
  --lr 0.0003 \
  --exp_name "kitti-rangeview-training-multigpu" \
  --config configs/dit_config_rangeview.py \
  --eval_steps 2000

# To resume from a checkpoint:
torchrun --nnodes=$NODES_NUM --nproc_per_node=$GPUS_NUM \
  scripts/train_rangeview.py \
  --batch_size 4 \
  --lr 0.0003 \
  --exp_name "kitti-rangeview-resume" \
  --config configs/dit_config_rangeview.py \
  --resume_path "exp/ckpt/kitti-rangeview-training/rangeview_dit_10000.pkl" \
  --resume_step 10000 \
  --eval_steps 2000
"""