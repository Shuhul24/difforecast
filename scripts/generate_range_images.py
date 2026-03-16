"""
Script to generate range images from point clouds using RangeProjection.
"""

import os
import sys
import argparse
import numpy as np
import glob
from tqdm import tqdm

# Add repository root to path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from dataset.projection import RangeProjection

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Range Images from Point Clouds")
    parser.add_argument("--src_path", type=str, required=True, help="Path to input point cloud file or directory")
    parser.add_argument("--dst_path", type=str, default=None, help="Path to output directory")
    parser.add_argument("--ext", type=str, default=".bin", help="Point cloud extension (.bin or .npy)")
    
    # Projection parameters (defaults for KITTI)
    parser.add_argument("--fov_up", type=float, default=3.0, help="Field of view up (degrees)")
    parser.add_argument("--fov_down", type=float, default=-25.0, help="Field of view down (degrees)")
    parser.add_argument("--width", type=int, default=2048, help="Range image width")
    parser.add_argument("--height", type=int, default=64, help="Range image height")
    parser.add_argument("--fov_left", type=float, default=-180.0, help="Field of view left (degrees)")
    parser.add_argument("--fov_right", type=float, default=180.0, help="Field of view right (degrees)")
    
    return parser.parse_args()

def load_point_cloud(path, ext):
    """Load point cloud from file."""
    if ext == '.bin':
        # Assuming KITTI format: x, y, z, intensity (float32)
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    elif ext == '.npy':
        points = np.load(path)
        if points.shape[1] < 3:
            raise ValueError(f"Point cloud must have at least 3 columns (x, y, z), got {points.shape}")
    else:
        raise ValueError(f"Unsupported extension: {ext}")
    return points

def save_visualization(save_path, range_image, mask):
    """Save a colorized visualization of the range image."""
    import cv2

    # Normalize depth for visualization (0-80m approx)
    depth_vis = range_image.copy()
    
    # Clip and normalize
    max_range = 80.0
    depth_vis = np.clip(depth_vis / max_range, 0, 1) * 255
    depth_vis = depth_vis.astype(np.uint8)
    
    # Apply colormap
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    
    # Set invalid pixels to black
    depth_color[~mask] = 0
    
    if not cv2.imwrite(save_path, depth_color):
        raise IOError(f"Failed to save image to {save_path}")

def main():
    args = parse_args()
    
    if args.dst_path is None:
        if os.path.isfile(args.src_path):
            args.dst_path = os.path.dirname(args.src_path) or "."
        else:
            args.dst_path = args.src_path

    # Check for opencv
    try:
        import cv2
    except ImportError:
        print("Error: opencv-python is not installed. Please install it via `pip install opencv-python`.")
        return

    # Create output directory
    os.makedirs(args.dst_path, exist_ok=True)

    # Initialize projector
    print(f"Initializing RangeProjection with:")
    print(f"  FOV: [{args.fov_down}, {args.fov_up}] vertical, [{args.fov_left}, {args.fov_right}] horizontal")
    print(f"  Size: {args.width}x{args.height}")
    
    projector = RangeProjection(
        fov_up=args.fov_up,
        fov_down=args.fov_down,
        proj_w=args.width,
        proj_h=args.height,
        fov_left=args.fov_left,
        fov_right=args.fov_right
    )

    # Get files
    if os.path.isfile(args.src_path):
        files = [args.src_path]
    else:
        search_pattern = os.path.join(args.src_path, f"*{args.ext}")
        files = sorted(glob.glob(search_pattern))
    
    if not files:
        print(f"No files found matching {args.src_path}")
        return

    print(f"Processing {len(files)} files...")
    
    for file_path in tqdm(files):
        try:
            _, ext = os.path.splitext(file_path)
            # Load points
            points = load_point_cloud(file_path, ext)
            
            # Project
            # Returns: proj_pointcloud, proj_range, proj_idx, proj_mask
            # proj_pointcloud: [H, W, C] (x, y, z, intensity, ...)
            # proj_range: [H, W] depth
            proj_pc, proj_range, proj_idx, proj_mask = projector.doProjection(points)
            
            # Prepare output filename
            basename = os.path.basename(file_path)
            name_no_ext = os.path.splitext(basename)[0]
            
            # Save visualization as png
            out_path = os.path.join(args.dst_path, f"{name_no_ext}.png")
            # Use proj_range > 0 for validity check
            valid_mask = (proj_range > 0)
            save_visualization(out_path, proj_range, valid_mask)
        
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

print("Done.")

if __name__ == "__main__":
    main()
