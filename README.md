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

## Acknowledgement

Built on [Epona](https://github.com/FoundationVision/Epona), [DrivingWorld](https://github.com/YvanYin/DrivingWorld), [Flux](https://github.com/black-forest-labs/flux), and [DCAE](https://github.com/mit-han-lab/efficientvit/tree/master/applications/dc_ae).
