#!/bin/bash
# ===========================================================================
#  SLURM job script — RangeView DiT training on KITTI
#
#  Usage (single-node, 4 GPUs):
#    sbatch scripts/submit_rangeview_slurm.sh
#
#  Usage (multi-node, 2 × 4 GPUs):
#    sbatch --nodes=2 scripts/submit_rangeview_slurm.sh
#
#  The script uses `torchrun` launched via `srun` so that it works for both
#  single-node and multi-node SLURM allocations without any code changes.
# ===========================================================================

# ── Resource requests (adjust to your cluster) ─────────────────────────────
#SBATCH --job-name=rangeview-dit
#SBATCH --output=logs/slurm_%j_%x.out
#SBATCH --error=logs/slurm_%j_%x.err
#SBATCH --nodes=1                  # number of nodes
#SBATCH --ntasks-per-node=1        # ONE srun task per node; torchrun spawns GPU workers
#SBATCH --gres=gpu:4               # GPUs per node — change to match your partition
#SBATCH --cpus-per-task=32         # CPU threads available to all GPU workers on this node
#SBATCH --mem=128G                 # RAM per node
#SBATCH --time=48:00:00            # wall-clock limit hh:mm:ss
#SBATCH --partition=gpu            # partition / queue name on your HPC

# ── Environment setup ───────────────────────────────────────────────────────
# Uncomment whichever applies to your cluster:

# Option A – Conda
# module load anaconda3
# conda activate epona

# Option B – module + venv
# module load cuda/12.1 cudnn/8.9
# source /path/to/venv/bin/activate

# ── Paths ───────────────────────────────────────────────────────────────────
# Resolve the repo root relative to this script's location
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Create output directories if they don't exist
mkdir -p logs exp/ckpt exp/job_log exp/job_tboard exp/validation

# ── Distributed rendezvous ──────────────────────────────────────────────────
# Head node hostname — used as the rendezvous endpoint for torchrun
MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)

# Pick a random free port to avoid collisions between concurrent jobs
MASTER_PORT=$(python -c \
    "import socket; s=socket.socket(); s.bind(('',0)); \
     p=s.getsockname()[1]; s.close(); print(p)")

# Number of GPUs per node (parsed from --gres=gpu:N in this script)
GPUS_PER_NODE=$(echo "$SLURM_JOB_GPUS $SLURM_GPUS_ON_NODE $SLURM_GPUS_PER_NODE" \
    | tr ' ' '\n' | grep -m1 '[0-9]' | grep -oP '[0-9]+$' || echo 4)

export MASTER_ADDR MASTER_PORT

echo "========================================"
echo " Job ID      : $SLURM_JOB_ID"
echo " Nodes       : $SLURM_NODELIST"
echo " MASTER_ADDR : $MASTER_ADDR:$MASTER_PORT"
echo " GPUs/node   : $GPUS_PER_NODE"
echo " Repo root   : $REPO_ROOT"
echo "========================================"

# ── Training command ─────────────────────────────────────────────────────────
# srun launches one process per node; each node runs a `torchrun` that in turn
# spawns GPUS_PER_NODE worker processes (one per GPU).
srun torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    scripts/train_rangeview.py \
    --config       configs/dit_config_rangeview.py \
    --exp_name     "kitti-rangeview-${SLURM_JOB_ID}" \
    --batch_size   1 \
    --blr          1e-4 \
    --weight_decay 0.01 \
    --eval_steps   2000 \
    --iter         600000 \
    --launcher     pytorch
