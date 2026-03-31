"""
Sanity-check script for Stage 1 Swin RAE pipeline.

Validates without any pre-trained weights:
  1. Forward pass — correct output shapes, finite loss
  2. Backward pass — all trainable parameters receive gradients
  3. Mini-training loop — loss decreases over N steps on real KITTI data
  4. Reconstruction visualisation — saves a PNG at the end

Usage:
    conda run -n difforecast python scripts/test_swin_stage1.py \
        --config configs/swin_config_rangeview.py \
        --steps 50
"""

import os, sys, argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from utils.config_utils import Config
from models.swin_rae_rangeview import RangeViewSwinRAE
from dataset.dataset_kitti_rangeview import KITTIRangeViewDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/swin_config_rangeview.py')
    p.add_argument('--steps',  default=50, type=int,
                   help='Number of gradient steps for the mini-training check')
    p.add_argument('--batch',  default=2, type=int)
    p.add_argument('--out',    default='outputs/swin_stage1_sanity',
                   help='Directory to save visualisation PNG')
    p.add_argument('--pretrained_tulip', default=None, type=str,
                   help='Path to TULIP pre-trained weights to load via model.load_from_tulip()')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def check_gradients(model):
    """Return (has_grad, no_grad) parameter name lists."""
    has_grad, no_grad = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                has_grad.append(name)
            else:
                no_grad.append(name)
    return has_grad, no_grad


def save_reconstruction_vis(gt, pred, log_range, mean, std, out_path):
    """Save a side-by-side range-view reconstruction comparison PNG."""
    def to_depth(t):
        t = t.float().cpu().numpy()
        if log_range:
            return (2.0 ** (t * 6.0)) - 1.0
        return t * std + mean

    gt_d   = to_depth(gt[0, 0])
    pred_d = to_depth(pred[0, 0].detach())
    mae    = float(np.abs(gt_d - pred_d).mean())

    fig, axes = plt.subplots(2, 1, figsize=(20, 4))
    for ax, img, title in zip(axes, [gt_d, pred_d],
                               [f'GT depth', f'Reconstructed (MAE={mae:.3f} m)']):
        im = ax.imshow(img, cmap='turbo', vmin=0, vmax=50)
        ax.set_title(title); ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.01)
    plt.suptitle('Stage 1 Swin RAE — reconstruction sanity check')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    return mae


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args_cli = parse_args()
    args = Config.fromfile(args_cli.config)

    print('\n' + '='*70)
    print('Stage 1 Swin RAE — pipeline sanity check')
    print('='*70)

    # ── 1. Build model ────────────────────────────────────────────────────────
    model = RangeViewSwinRAE(args).cuda()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f'\n[Model]  trainable params: {n_params:.1f} M')

    if args_cli.pretrained_tulip:
        print(f'\n[Model]  Loading pre-trained TULIP weights from {args_cli.pretrained_tulip} ...')
        
        # Snapshot weights before loading to track what gets modified
        before_weights = {k: v.clone() for k, v in model.state_dict().items()}
        
        model.load_from_tulip(args_cli.pretrained_tulip)
        
        # Check what was actually updated
        after_weights = model.state_dict()
        loaded_keys = []
        unchanged_keys = []
        
        for k in before_weights.keys():
            if not torch.equal(before_weights[k], after_weights[k]):
                loaded_keys.append(k)
            else:
                unchanged_keys.append(k)
                
        print(f"         [Weight Check] Successfully updated {len(loaded_keys)} / {len(before_weights)} tensors.")
        if unchanged_keys:
            print(f"         [Weight Check] Showing up to 10 unchanged parameters (expected for custom/new layers):")
            for k in unchanged_keys[:10]:
                print(f"           - {k}")
            if len(unchanged_keys) > 10:
                print(f"           ... and {len(unchanged_keys) - 10} more.")

    # ── 2. Build dataset (single sequence for speed) ─────────────────────────
    print(f'\n[Data]   loading sequence {args.train_sequences[0]} ...')
    ds = KITTIRangeViewDataset(
        sequences=args.train_sequences[:1],
        sequences_path=args.kitti_sequences_path,
        poses_path=args.kitti_poses_path,
        h=args.range_h, w=args.range_w,
        fov_up=args.fov_up, fov_down=args.fov_down,
        fov_left=args.fov_left, fov_right=args.fov_right,
        proj_img_mean=args.proj_img_mean,
        proj_img_stds=args.proj_img_stds,
        pc_extension=args.pc_extension,
        pc_dtype=getattr(np, args.pc_dtype),
        pc_reshape=tuple(args.pc_reshape),
        five_channel=getattr(args, 'five_channel', False),
        log_range=getattr(args, 'log_range', True),
        condition_frames=0, forward_iter=1, is_train=True,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args_cli.batch, shuffle=True,
        num_workers=2, drop_last=True,
    )
    print(f'         {len(ds)} samples')

    # ── 3. Forward pass check ─────────────────────────────────────────────────
    print('\n[Check 1] Forward pass ...')
    model.eval()
    data, _ = next(iter(loader))
    if data.dim() == 5:
        data = data[:, 0]
    x = data.cuda()

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            out = model(x)

    assert out['x_rec'].shape == x.shape, \
        f"Shape mismatch: expected {x.shape}, got {out['x_rec'].shape}"
    assert torch.isfinite(out['loss_all']), \
        f"Non-finite loss: {out['loss_all'].item()}"

    print(f'         loss_all = {out["loss_all"].item():.4f}  '
          f'loss_rec = {out["loss_rec"].item():.4f}  '
          f'x_rec shape = {out["x_rec"].shape}  ✓')

    # ── 4. Backward pass / gradient check ────────────────────────────────────
    print('\n[Check 2] Backward pass + gradient flow ...')
    model.train()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = model(x)
    out['loss_all'].backward()

    has_grad, no_grad = check_gradients(model)
    print(f'         params with gradients : {len(has_grad)}')
    if no_grad:
        print(f'         params WITHOUT grads  : {len(no_grad)}')
        for n in no_grad[:5]:
            print(f'           - {n}')
    else:
        print(f'         all trainable params have gradients  ✓')
    model.zero_grad()

    # ── 5. Mini-training loop (loss should decrease) ──────────────────────────
    print(f'\n[Check 3] Mini-training loop ({args_cli.steps} steps) ...')
    model.train()
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.01,
    )
    scaler = torch.cuda.amp.GradScaler()

    losses = []
    loader_iter = iter(loader)
    for step in range(args_cli.steps):
        try:
            batch_data, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch_data, _ = next(loader_iter)

        if batch_data.dim() == 5:
            batch_data = batch_data[:, 0]
        xb = batch_data.cuda()

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(xb)
        loss = out['loss_all']

        assert torch.isfinite(loss), f"NaN/Inf loss at step {step}"
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        if (step + 1) % 10 == 0:
            avg = sum(losses[-10:]) / 10
            print(f'         step {step+1:3d}/{args_cli.steps}  '
                  f'loss={loss.item():.4f}  avg(last 10)={avg:.4f}')

    first_10  = sum(losses[:10])  / 10
    last_10   = sum(losses[-10:]) / 10
    decreased = last_10 < first_10
    print(f'\n         first-10 avg: {first_10:.4f}  →  last-10 avg: {last_10:.4f}  '
          + ('✓ decreasing' if decreased else '✗ NOT decreasing — check LR/data'))

    # ── 6. Save reconstruction visualisation ─────────────────────────────────
    print(f'\n[Check 4] Saving reconstruction PNG ...')
    model.eval()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            final_out = model(x)

    out_path = os.path.join(args_cli.out, 'reconstruction.png')
    mae = save_reconstruction_vis(
        x, final_out['x_rec'],
        log_range=getattr(args, 'log_range', True),
        mean=args.proj_img_mean[0],
        std=args.proj_img_stds[0],
        out_path=out_path,
    )
    print(f'         MAE = {mae:.4f} m  (random init — will be high)')
    print(f'         saved → {out_path}')

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '='*70)
    print('SUMMARY')
    print('  Forward pass shape + finite loss : ✓')
    print(f'  Gradient flow to all params      : {"✓" if not no_grad else "partial"}')
    print(f'  Loss decreasing over {args_cli.steps} steps    : {"✓" if decreased else "✗"}')
    print(f'  Reconstruction PNG saved         : ✓  ({out_path})')
    print('='*70 + '\n')


if __name__ == '__main__':
    main()
