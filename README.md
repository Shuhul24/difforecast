# DifForecast

A unified repository containing two diffusion-based models for autonomous driving:

| Sub-project | Description | Venue |
|---|---|---|
| [**Epona**](./Epona/) | Autoregressive Diffusion World Model for Autonomous Driving | ICCV 2025 |
| [**DiffLoc**](./DiffLoc/) | Diffusion Model for Outdoor LiDAR Localization | CVPR 2024 |

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Quick Start on HPC (SLURM)](#quick-start-on-hpc-slurm)
   - [Step 1 — Clone the repo on the cluster](#step-1--clone-the-repo-on-the-cluster)
   - [Step 2 — Create and activate the environment](#step-2--create-and-activate-the-environment)
   - [Step 3 — Prepare data & checkpoints](#step-3--prepare-data--checkpoints)
   - [Step 4 — Edit the SLURM script](#step-4--edit-the-slurm-script)
   - [Step 5 — Submit the job](#step-5--submit-the-job)
3. [Manual Training (without SLURM)](#manual-training-without-slurm)
4. [Inference / Testing](#inference--testing)
5. [Configuration Reference](#configuration-reference)
6. [Troubleshooting](#troubleshooting)

---

## Repository Structure

```
difforecast/
├── Epona/                          # World model sub-project
│   ├── configs/                    # Training / inference config files
│   │   ├── dit_config_dcae_nuplan.py
│   │   ├── dit_config_dcae_nuscenes.py
│   │   └── dit_config_rangeview.py
│   ├── data_preparation/           # Data download & preprocessing guides
│   │   ├── README.md
│   │   └── create_nuplan_json.py
│   ├── dataset/                    # Dataset loaders (NuPlan, nuScenes, KITTI)
│   ├── models/                     # Model definitions (DiT, DCAE, diffusion)
│   ├── scripts/
│   │   ├── submit_rangeview_slurm.sh   # <-- SLURM job script
│   │   ├── train_deepspeed.py          # Multi-node DeepSpeed training
│   │   ├── train_rangeview.py          # RangeView training entry-point
│   │   └── test/                       # Inference scripts
│   ├── utils/                      # Helpers (logging, distributed, etc.)
│   └── requirements.txt
└── DiffLoc/                        # LiDAR localisation sub-project
```

---

## Quick Start on HPC (SLURM)

### Step 1 — Clone the repo on the cluster

SSH into a login node, then clone this repository into your scratch / work directory:

```bash
cd $SCRATCH          # or wherever large storage lives on your cluster
git clone https://github.com/Shuhul24/difforecast.git
cd difforecast/Epona
```

### Step 2 — Create and activate the environment

> **Requirement:** Conda must be available (`module load anaconda3` or equivalent on your cluster).

```bash
conda create -n epona python=3.10 -y
conda activate epona
```

Install PyTorch first (pick the version that matches your cluster's CUDA):

```bash
# Example for CUDA 12.1 — adjust the CUDA suffix as needed
pip install torch>=2.1.0 torchvision>=0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

> **Tip:** Comment out the `torch` and `torchvision` lines in `requirements.txt` before running the above so they are not overwritten.

### Step 3 — Prepare data & checkpoints

#### Datasets

| Dataset | Guide |
|---|---|
| NuPlan | [Epona/data_preparation/README.md](./Epona/data_preparation/README.md) |
| nuScenes | [Epona/data_preparation/README.md](./Epona/data_preparation/README.md) |
| KITTI (RangeView) | Provide the dataset path in `configs/dit_config_rangeview.py` |

After downloading and preprocessing, update the `datasets_paths` entries inside the relevant config file:

```python
# Epona/configs/dit_config_dcae_nuplan.py (example)
datasets_paths = dict(
    nuplan_root       = '/path/to/your/nuplan',
    nuplan_json_root  = '/path/to/your/nuplan_json',
    ...
)
```

#### Pre-trained checkpoints

Download the Epona world model weights and the temporal-aware DCAE from Hugging Face:

```bash
# Install the HF CLI if not already available
pip install huggingface_hub

# Download to a local directory
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Kevin-thu/Epona", local_dir="pretrained/")
EOF
```

Then set the checkpoint paths in the config file:

```python
vae_ckpt = 'pretrained/temporal_dcae.pkl'   # DCAE checkpoint
# pass --resume_path pretrained/epona_nuplan.pkl when running scripts
```

### Step 4 — Edit the SLURM script

Open `Epona/scripts/submit_rangeview_slurm.sh` and adjust the resource directives at the top to match your cluster's partition and available hardware:

```bash
#SBATCH --job-name=rangeview-dit
#SBATCH --output=logs/slurm_%j_%x.out
#SBATCH --error=logs/slurm_%j_%x.err
#SBATCH --nodes=1              # number of nodes
#SBATCH --ntasks-per-node=1    # keep at 1; torchrun spawns GPU workers
#SBATCH --gres=gpu:4           # GPUs per node
#SBATCH --cpus-per-task=32     # CPU threads per node
#SBATCH --mem=128G             # RAM per node
#SBATCH --time=48:00:00        # wall-clock limit hh:mm:ss
#SBATCH --partition=gpu        # your cluster's partition name
```

Also uncomment the environment-activation block that applies to your cluster (Conda or venv):

```bash
# Option A — Conda (most common)
module load anaconda3
conda activate epona

# Option B — module + virtualenv
# module load cuda/12.1 cudnn/8.9
# source /path/to/venv/bin/activate
```

### Step 5 — Submit the job

From the repo root (or from `Epona/`):

```bash
# Single-node, 4 GPUs (default in the script)
sbatch Epona/scripts/submit_rangeview_slurm.sh

# Multi-node override (e.g. 2 nodes × 4 GPUs)
sbatch --nodes=2 Epona/scripts/submit_rangeview_slurm.sh
```

Monitor your job:

```bash
squeue -u $USER                      # show running/pending jobs
tail -f logs/slurm_<JOB_ID>_*.out   # stream training logs
```

---

## Manual Training (without SLURM)

If you prefer to run training directly (e.g. on an interactive node), use DeepSpeed:

```bash
cd Epona

export NODES_NUM=1
export GPUS_NUM=4    # adjust to the number of GPUs available

torchrun \
  --nnodes=$NODES_NUM \
  --nproc_per_node=$GPUS_NUM \
  scripts/train_deepspeed.py \
    --batch_size   2 \
    --lr           2e-5 \
    --exp_name     "train-nuplan" \
    --config       configs/dit_config_dcae_nuplan.py \
    --resume_path  pretrained/epona_nuplan.pkl \
    --eval_steps   2000
```

---

## Inference / Testing

All test scripts live in `Epona/scripts/test/`. A single NVIDIA 4090 (or equivalent) is sufficient for inference.

| Script | Dataset | Use case |
|---|---|---|
| `test_nuplan.py` | NuPlan | Fixed-trajectory evaluation |
| `test_free.py` | NuPlan | Long-term video generation (auto-predicted trajectory) |
| `test_ctrl.py` | NuPlan | Trajectory-controlled video generation |
| `test_traj.py` | NuPlan | Trajectory prediction accuracy |
| `test_nuscenes.py` | nuScenes | Fixed-trajectory evaluation |
| `test_demo.py` | Custom input | Run on your own data |

Example — evaluate on NuPlan test set:

```bash
cd Epona
python3 scripts/test/test_nuplan.py \
  --exp_name    "test-nuplan" \
  --start_id    0 \
  --end_id      100 \
  --resume_path pretrained/epona_nuplan.pkl \
  --config      configs/dit_config_dcae_nuplan.py
```

To run inference inside SLURM (single GPU), wrap the command in a minimal batch script:

```bash
#!/bin/bash
#SBATCH --job-name=epona-test
#SBATCH --output=logs/test_%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu

module load anaconda3
conda activate epona

cd $SCRATCH/difforecast/Epona

python3 scripts/test/test_nuplan.py \
  --exp_name    "test-nuplan" \
  --start_id    0 \
  --end_id      100 \
  --resume_path pretrained/epona_nuplan.pkl \
  --config      configs/dit_config_dcae_nuplan.py
```

Save the above as e.g. `run_test.sh` and submit with `sbatch run_test.sh`.

---

## Configuration Reference

All tuneable hyper-parameters live in `Epona/configs/`. Key fields:

| Field | Description |
|---|---|
| `datasets_paths` | Paths to each dataset root and JSON metadata |
| `vae_ckpt` | Path to the DCAE autoencoder checkpoint |
| `image_size` | Input resolution `(H, W)` — default `(512, 1024)` |
| `condition_frames` | Number of context frames fed to the model |
| `n_layer` / `n_head` / `n_embd` | Transformer depth, heads, and embedding dim |
| `outdir` / `logdir` / `tdir` | Output directories for checkpoints and logs |
| `num_sampling_steps` | Diffusion sampling steps during inference |
| `forward_iter` | Number of autoregressive steps per forward pass |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | Reduce `--batch_size` or `image_size` in the config |
| `Address already in use` (torchrun) | The script auto-selects a free port; retry or kill stale processes with `fuser -k <port>/tcp` |
| `ModuleNotFoundError: deepspeed` | Run `pip install deepspeed>=0.12.0` inside the conda env |
| SLURM job stuck in `PD` (pending) | Check partition name and GPU availability with `sinfo -p gpu` |
| Wrong `MASTER_ADDR` on multi-node | Ensure `scontrol` is available; the script resolves it automatically from `$SLURM_NODELIST` |
| `vae_ckpt` path error | Set the correct path in the config file before launching |
