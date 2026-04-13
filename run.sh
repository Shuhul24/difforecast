#!/bin/bash
# Job name:
#SBATCH --job-name=difforecast
# Partition:
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=phd
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --output=/csehome/p24cs0005/difforecast/run_stage_1.out
#SBATCH --error=/csehome/p24cs0005/difforecast/run_stage_1.out
#SBATCH --nodelist=cn07

nvidia-smi

module load gcc/11.4.0-gcc-12.3.0-73jjveq
module load cuda/11.8.0-gcc-12.3.0-4pg4hmh

# export GPUS_NUM=2
# torchrun --nproc_per_node=$GPUS_NUM scripts/train_rangeview.py --batch_size 4 --lr 3e-4 --exp_name "kitti_rangeview" --config configs/dit_config_rangeview.py --eval_steps 2000

# torchrun --nproc_per_node=1 scripts/train_rangeview.py --batch_size 1 --lr 3e-4 --exp_name "kitti_rangeview" --config configs/dit_config_rangeview.py --eval_steps 1000 

# Stage 1 training:
# torchrun --nproc_per_node=1 scripts/train_rangeview.py --batch_size 13 --lr 3e-4 --exp_name "kitti_rangeview_stage_1" --config configs/dit_config_rangeview.py --eval_steps 1000 --stage 1 --no_log_file

#Stage 1 training from checkpoint:
torchrun --nproc_per_node=1 scripts/train_rangeview.py --batch_size 13 --lr 3e-4 --exp_name "kitti_rangeview_stage_1" --config configs/dit_config_rangeview.py --eval_steps 1000 --stage 1 --load_from_deepspeed "/scratch/p24cs0005/exp/ckpt/kitti_rangeview_stage_1/50000" --resume_step 50000 --no_log_file

#Stage 1 testing:
# python scripts/test/test_rangeview_vae.py --config configs/dit_config_rangeview.py --resume_path .../rangeview_vae_<step>.pkl --exp_name vae_eval --stage 1

#Stage 2 training:
# torchrun --nproc_per_node=1 scripts/train_rangeview.py --batch_size 1 --lr 3e-4 --exp_name "kitti_rangeview_stage_2" --config configs/dit_config_rangeview.py --eval_steps 1000 --stage 2 --vae_ckpt '/scratch/p24cs0005/weights/vae_stage1_step7000.pth'

# ── RAE pipeline (DINOv2 encoder + ViT-XL decoder) ───────────────────────────

# RAE Stage 1 training (ViT-XL decoder; frozen DINOv2 encoder):
# torchrun --nproc_per_node=1 scripts/train_rae_rangeview.py --stage 1 --batch_size 4 --exp_name "rae_stage_1" --config configs/rae_config_rangeview.py --eval_steps 2000 --no_log_file

# RAE Stage 1 training from checkpoint:
# torchrun --nproc_per_node=1 scripts/train_rae_rangeview.py --stage 1 --batch_size 4 --exp_name "rae_stage_1" --config configs/rae_config_rangeview.py --eval_steps 2000 --load_from_deepspeed "/scratch/p24cs0005/exp/ckpt/rae_stage_1/<step>" --resume_step <step> --no_log_file

# RAE Stage 2 training (STT + FluxDiT in DINOv2 latent space; set rae_ckpt in config first):
# torchrun --nproc_per_node=1 scripts/train_rae_rangeview.py --stage 2 --batch_size 2 --exp_name "rae_stage_2" --config configs/rae_config_rangeview.py --eval_steps 2000 --no_log_file

# RAE Stage 2 training from checkpoint:
# torchrun --nproc_per_node=1 scripts/train_rae_rangeview.py --stage 2 --batch_size 2 --exp_name "rae_stage_2" --config configs/rae_config_rangeview.py --eval_steps 2000 --load_from_deepspeed "/scratch/p24cs0005/exp/ckpt/rae_stage_2/<step>" --resume_step <step> --no_log_file

# ── SWIN Transformer pipeline (TULIPRangeEncoder + TULIPRangeDecoder) ────────

# SWIN Stage 1 training (Swin RAE: encoder + decoder; Berhu/L1 recon loss):
# torchrun --nproc_per_node=1 scripts/train_swin_rangeview.py --stage 1 --batch_size 4 --exp_name "swin_stage_1" --config configs/swin_config_rangeview.py --eval_steps 2000 --no_log_file

# SWIN Stage 1 training from checkpoint:
# torchrun --nproc_per_node=1 scripts/train_swin_rangeview.py --stage 1 --batch_size 4 --exp_name "swin_stage_1" --config configs/swin_config_rangeview.py --eval_steps 2000 --load_from_deepspeed "/scratch/p24cs0005/exp/ckpt/swin_stage_1/<step>" --resume_step <step> --no_log_file

# SWIN Stage 1 evaluation (range-view reconstruction metrics + figures):
# python scripts/eval_swin_stage1.py --config configs/swin_config_rangeview.py --ckpt /DATA2/shuhul/exp/swin_ckpt/swin-s1-ch1-b32/swin_rae_step<step>.pkl --out outputs/eval_swin_s1_step<step> --split test --n_samples 0

# SWIN Stage 2 training (STT + FluxDiT in Swin bottleneck latent space; set swin_ckpt in config first):
# torchrun --nproc_per_node=1 scripts/train_swin_rangeview.py --stage 2 --batch_size 2 --exp_name "swin_stage_2" --config configs/swin_config_rangeview.py --eval_steps 2000 --no_log_file

# SWIN Stage 2 training from checkpoint:
# torchrun --nproc_per_node=1 scripts/train_swin_rangeview.py --stage 2 --batch_size 2 --exp_name "swin_stage_2" --config configs/swin_config_rangeview.py --eval_steps 2000 --load_from_deepspeed "/scratch/p24cs0005/exp/ckpt/swin_stage_2/<step>" --resume_step <step> --no_log_file

# SWIN Stage 2 evaluation (autoregressive inference, per-frame metrics + BEV figures):
# python scripts/eval_swin_stage2.py --config configs/swin_config_rangeview.py --ckpt /scratch/p24cs0005/exp/ckpt/swin_stage_2/swin_dit_step<step>.pkl --out outputs/eval_swin_s2_step<step> --n_samples 200
