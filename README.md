# DifForecast — Range View LiDAR Forecasting

Training an autoregressive diffusion model on KITTI range view images, based on the [Epona](https://kevin-thu.github.io/Epona/) world model architecture.

---

## Repository Structure

```
difforecast/
├── configs/                    # Training configuration files
│   └── dit_config_rangeview.py # KITTI range view config
├── data_preparation/           # Data preprocessing scripts
├── dataset/                    # Dataset loaders
├── models/                     # Model definitions (DiT, DCAE, diffusion)
├── scripts/
│   ├── train_rangeview.py      # Range view training entry-point
│   ├── submit_rangeview_slurm.sh  # SLURM job submission script
│   └── test/                   # Inference scripts
├── utils/                      # Helpers (logging, distributed, etc.)
└── requirements.txt
```

---

## Installation

```bash
conda create -n epona python=3.10 -y
conda activate epona

# Install PyTorch matching your CUDA version (example: CUDA 12.1)
pip install torch>=2.1.0 torchvision>=0.16.0 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

---

## Dataset Setup (KITTI Odometry)

Download the [KITTI Odometry dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (velodyne laser data + ground truth poses).

Expected directory layout:

```
/path/to/kitti/
├── dataset/
│   └── sequences/
│       ├── 00/
│       │   └── velodyne/  # *.bin point cloud files
│       ├── 01/
│       └── ...
└── poses/
    ├── 00.txt
    ├── 01.txt
    └── ...
```

Open `configs/dit_config_rangeview.py` and update the three path variables to match your local dataset location:

```python
kitti_root            = '/your/path/to/kitti'
kitti_sequences_path  = '/your/path/to/kitti/dataset/sequences'
kitti_poses_path      = '/your/path/to/kitti/poses'
```

The default sequence splits used are:

| Split | Sequences |
|-------|-----------|
| Train | 0 – 5     |
| Val   | 6 – 7     |
| Test  | 8 – 10    |

---

## Training

The training script now supports a two‑stage workflow similar to RangeLDM.
Using the `--stage` argument you can run:

* `--stage 1` – VAE pre‑training only (optimises ELBO, STT/DiT are frozen).
* `--stage 2` – DiT/STT training with a frozen VAE (provide `--vae_ckpt` or
  resume from a stage‑1 checkpoint).
* `--stage all` (default) – the original full end‑to‑end pipeline.


### Single GPU

```bash
export NODES_NUM=1
export GPUS_NUM=1

torchrun --nnodes=$NODES_NUM --nproc_per_node=$GPUS_NUM \
  scripts/train_rangeview.py \
  --batch_size 4 \
  --lr 3e-4 \
  --exp_name "kitti-rangeview" \
  --config configs/dit_config_rangeview.py \
  --eval_steps 2000
```

### Multi-GPU (single node)

```bash
export NODES_NUM=1
export GPUS_NUM=4

torchrun --nnodes=$NODES_NUM --nproc_per_node=$GPUS_NUM \
  scripts/train_rangeview.py \
  --batch_size 1 \
  --lr 3e-4 \
  --exp_name "kitti-rangeview-multigpu" \
  --config configs/dit_config_rangeview.py \
  --eval_steps 2000
```

### Resume from Checkpoint

```bash
torchrun --nnodes=1 --nproc_per_node=$GPUS_NUM \
  scripts/train_rangeview.py \
  --batch_size 4 \
  --lr 3e-4 \
  --exp_name "kitti-rangeview-resume" \
  --config configs/dit_config_rangeview.py \
  --resume_path "exp/ckpt/kitti-rangeview/rangeview_dit_10000.pkl" \
  --resume_step 10000 \
  --eval_steps 2000
```

### SLURM

Edit the resource directives in `scripts/submit_rangeview_slurm.sh` (partition, GPUs, memory), then submit:

```bash
sbatch scripts/submit_rangeview_slurm.sh
```

---

## Key Configuration Options (`configs/dit_config_rangeview.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `condition_frames` | 5 | Past frames fed as context |
| `forward_iter` | 5 | Future frames to predict |
| `image_size` | `(64, 2048)` | Range image resolution (H×W) |
| `num_sampling_steps` | 100 | Diffusion steps during inference |
| `outdir` | `exp/ckpt` | Checkpoint output directory |
| `logdir` | `exp/job_log` | Log directory |

---

## DCAE Pre-trained Weights

The `RangeViewVAETokenizer` (defined in `models/modules/tokenizer.py`) wraps a `dc_ae_f32c32_rangeview` DCAE model built entirely from local code in `models/modules/dcae.py`. **No weights are fetched from any library or Hugging Face Hub at runtime.** The encoder is kept strictly frozen during training (all parameters have `requires_grad=False`).

### Where weights come from

Epona (the upstream project this work is based on) initialises its RGB DCAE from the **`dc-ae-f32c32-mix-1.0`** checkpoint released by MIT Han Lab:

> **[mit-han-lab/dc-ae-f32c32-mix-1.0](https://huggingface.co/mit-han-lab/dc-ae-f32c32-mix-1.0)**

For the range-view case the same checkpoint is used as a warm-start via **partial loading**: every layer whose weight shape matches the 3-channel RGB checkpoint is initialised from it. The two channel-sensitive layers — `encoder.project_in` (3 → 128 ch) and `decoder.project_out` (128 → 3 ch) — differ from the 6-channel range-view variant and are therefore **randomly initialised**. This is handled automatically by `dc_ae_f32c32_rangeview` when `pretrained_path` is set (`pretrained_source = "dc-ae-partial"`).

### Downloading the weights

```bash
# Install the Hugging Face Hub CLI if not already available
pip install huggingface_hub

# Download the safetensors checkpoint (~400 MB)
huggingface-cli download mit-han-lab/dc-ae-f32c32-mix-1.0 \
    --include "model.safetensors" \
    --local-dir /path/to/weights/dc-ae-f32c32-mix-1.0
```

### Updating the path

Open `configs/dit_config_rangeview.py` and set `vae_ckpt` to the downloaded file:

```python
# configs/dit_config_rangeview.py  (line ~131)
vae_ckpt = '/path/to/weights/dc-ae-f32c32-mix-1.0/model.safetensors'
```

Setting `vae_ckpt = None` (the current default) initialises the entire DCAE randomly, which requires significantly more training iterations for the encoder to converge.

---

## DINOv2 Pretrained Weights (`tsne_latents.py`)

`scripts/tsne_latents.py` uses a **DiffLoc-style** feature extractor
([liw95/DiffLoc](https://github.com/liw95/DiffLoc)) to visualise how
pretrained DINOv2 features cluster across KITTI sequences.  The extractor
consists of a ConvStem (SalsaNext-inspired ResBlocks) that adapts the
2-channel range image to patch tokens, followed by the pretrained
**DINOv2 ViT-S/14** transformer blocks.

DiffLoc initialises its feature learner from the
[DINOv2](https://github.com/facebookresearch/dinov2) foundation model
released by Meta AI Research.

### Downloading the weights

```bash
# ViT-Small, patch-size 14 (~85 MB)
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth \
     -O /path/to/weights/dinov2_vits14_pretrain.pth
```

Alternatively, weights are fetched automatically via `torch.hub` on first run
(requires internet access).

### Running the visualisation

```bash
python scripts/tsne_latents.py \
    --config configs/dit_config_rangeview.py \
    --n_frames_per_seq 50 \
    --output_dir outputs/tsne_latents \
    --pretrained_path /path/to/weights/dinov2_vits14_pretrain.pth
```

Output: `outputs/tsne_latents/dino_tsne_all_sequences.png` — a single t-SNE
scatter plot of DINOv2 CLS-token features across all KITTI sequences 00-09,
coloured by sequence ID.

---

## Acknowledgement

Built on [Epona](https://github.com/Kevin-thu/Epona), [DrivingWorld](https://github.com/YvanYin/DrivingWorld), [Flux](https://github.com/black-forest-labs/flux), [DCAE](https://github.com/mit-han-lab/efficientvit/tree/master/applications/dc_ae), and [DiffLoc](https://github.com/liw95/DiffLoc) (DINOv2 range-view feature extractor).
