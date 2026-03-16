import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# Add root to path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from utils.config_utils import Config
from dataset.dataset_kitti_rangeview import KITTIRangeViewDataset

def main():
    parser = argparse.ArgumentParser(description="Generate GIF from KITTI Range View Sequence")
    parser.add_argument('--config', type=str, default='configs/dit_config_rangeview.py', help='Path to config file')
    parser.add_argument('--sequence', type=int, default=0, help='KITTI sequence ID (e.g. 0)')
    parser.add_argument('--output', type=str, default='rangeview_seq00.gif', help='Output GIF path')
    parser.add_argument('--num_frames', type=int, default=100, help='Number of frames to visualize')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for GIF')
    parser.add_argument('--cmap', type=str, default='jet', help='Matplotlib colormap for depth')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        return

    cfg = Config.fromfile(args.config)

    print(f"Loading dataset for sequence {args.sequence}...")
    
    # Initialize dataset with is_train=False to use stride=1 (consecutive frames)
    # We set condition_frames=0 because we only need to visualize one frame at a time
    dataset = KITTIRangeViewDataset(
        sequences_path=cfg.kitti_sequences_path,
        poses_path=cfg.kitti_poses_path,
        sequences=[args.sequence],
        condition_frames=0, 
        forward_iter=1,     # Minimal window size
        h=cfg.range_h,
        w=cfg.range_w,
        fov_up=cfg.fov_up,
        fov_down=cfg.fov_down,
        fov_left=cfg.fov_left,
        fov_right=cfg.fov_right,
        proj_img_mean=cfg.proj_img_mean,
        proj_img_stds=cfg.proj_img_stds,
        pc_extension=cfg.pc_extension,
        pc_dtype=getattr(np, cfg.pc_dtype),
        pc_reshape=tuple(cfg.pc_reshape),
        is_train=False 
    )

    if len(dataset) == 0:
        print(f"No frames found for sequence {args.sequence}. Check paths in config.")
        return

    frames = []
    num_frames = min(len(dataset), args.num_frames)
    print(f"Processing {num_frames} frames...")

    # Colormap for depth
    try:
        cmap = plt.get_cmap(args.cmap)
    except ValueError:
        print(f"Colormap '{args.cmap}' not found, defaulting to 'jet'")
        cmap = plt.get_cmap('jet')

    for i in range(num_frames):
        # Dataset returns (range_views, poses)
        # range_views: [T, C, H, W] -> [1, 6, 64, 2048]
        data, _ = dataset[i]
        
        # Extract range channel (channel 0) -> [H, W]
        range_img = data[0, 0].numpy()
        
        # The data is normalized and masked (0 where invalid).
        # Mask: values exactly 0 are invalid (background)
        mask = np.abs(range_img) > 1e-6
        
        # Prepare image for colormap
        img_vis = np.zeros((range_img.shape[0], range_img.shape[1], 3), dtype=np.uint8)
        
        if np.sum(mask) > 0:
            valid_pixels = range_img[mask]
            
            # Normalize valid pixels to 0-1 for colormap
            v_min = valid_pixels.min()
            v_max = valid_pixels.max()
            norm_pixels = (valid_pixels - v_min) / (v_max - v_min + 1e-6)
            
            # Apply colormap (returns RGBA, we take RGB)
            colored_pixels = cmap(norm_pixels)[:, :3]
            
            # Fill image
            img_vis[mask] = (colored_pixels * 255).astype(np.uint8)
        
        frames.append(Image.fromarray(img_vis))
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_frames}")

    print(f"Saving GIF to {args.output}...")
    frames[0].save(args.output, save_all=True, append_images=frames[1:], optimize=False, duration=1000 // args.fps, loop=0)
    print("Done.")

if __name__ == "__main__":
    main()
