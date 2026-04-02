"""
Compute mean and std of the Swin Stage-1 bottleneck latents [B, 64, 768]
over the training set.  Run this once before Stage 2 training, then set
`latent_scale` in configs/swin_config_rangeview.py to the printed std value.

Usage:
    python scripts/compute_latent_stats.py \
        --config configs/swin_config_rangeview.py \
        --ckpt   /DATA2/shuhul/exp/swin_ckpt/swin-s1/swin_rae_step46000.pkl \
        --n_batches 200
"""

import os, sys, argparse
import numpy as np
import torch

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from utils.config_utils import Config
from models.swin_rae_rangeview import RangeViewSwinRAE
from dataset.dataset_kitti_rangeview import KITTIRangeViewDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',    required=True)
    p.add_argument('--ckpt',      required=True, help='Stage 1 swin_rae_step*.pkl')
    p.add_argument('--n_batches', default=200, type=int,
                   help='Number of batches to sample (200 × batch 8 ≈ 1600 frames is enough)')
    p.add_argument('--batch_size', default=8, type=int)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config.fromfile(args.config)

    model = RangeViewSwinRAE(cfg, local_rank=0).cuda().eval()
    ckpt  = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'Loaded Stage-1 checkpoint (step {ckpt["step"]})')

    ds = KITTIRangeViewDataset(
        sequences=cfg.train_sequences,
        sequences_path=cfg.kitti_sequences_path,
        poses_path=cfg.kitti_poses_path,
        h=cfg.range_h, w=cfg.range_w,
        fov_up=cfg.fov_up, fov_down=cfg.fov_down,
        fov_left=cfg.fov_left, fov_right=cfg.fov_right,
        proj_img_mean=cfg.proj_img_mean,
        proj_img_stds=cfg.proj_img_stds,
        pc_extension=cfg.pc_extension,
        pc_dtype=getattr(np, cfg.pc_dtype),
        pc_reshape=tuple(cfg.pc_reshape),
        five_channel=getattr(cfg, 'five_channel', False),
        log_range=getattr(cfg, 'log_range', True),
        condition_frames=0, forward_iter=1, is_train=True,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, drop_last=True,
    )

    # Welford online mean/variance (numerically stable, single pass)
    count = 0
    mean  = 0.0
    M2    = 0.0

    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
            if i >= args.n_batches:
                break
            if data.dim() == 5:
                data = data[:, 0]
            x = data.cuda().to(torch.bfloat16)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                z, _ = model.encode(x)   # [B, 64, 768]
            vals = z.float().cpu().numpy().ravel()
            for v in vals:
                count += 1
                delta  = v - mean
                mean  += delta / count
                M2    += delta * (v - mean)
            if (i + 1) % 20 == 0:
                std_so_far = float(np.sqrt(M2 / (count - 1)))
                print(f'  batch {i+1}/{args.n_batches}  '
                      f'n={count:,}  mean={mean:.4f}  std={std_so_far:.4f}')

    std = float(np.sqrt(M2 / (count - 1)))
    print(f'\n{"="*55}')
    print(f'  Latent statistics over {count:,} values')
    print(f'  mean        = {mean:.6f}')
    print(f'  std         = {std:.6f}')
    print(f'\n  → Set in configs/swin_config_rangeview.py:')
    print(f'      latent_scale = {std:.4f}')
    print(f'{"="*55}')


if __name__ == '__main__':
    main()
