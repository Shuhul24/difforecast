"""
Test script to verify KITTI dataloader functionality
"""

import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from dataset.dataset_kitti_rangeview import KITTIRangeViewTrainDataset
from utils.config_utils import Config

def test_dataloader():
    """Test the KITTI dataloader"""

    # Load config
    config_path = os.path.join(root_path, 'configs', 'dit_config_rangeview.py')
    args = Config.fromfile(config_path)

    print("="*80)
    print("Testing KITTI Range View Dataloader")
    print("="*80)

    print(f"\nKITTI Sequences Path: {args.kitti_sequences_path}")
    print(f"KITTI Poses Path: {args.kitti_poses_path}")
    print(f"Training Sequences: {args.train_sequences}")

    # Create dataset
    print("\nCreating training dataset...")
    train_dataset = KITTIRangeViewTrainDataset(
        sequences_path=args.kitti_sequences_path,
        poses_path=args.kitti_poses_path,
        sequences=args.train_sequences,
        condition_frames=args.condition_frames,
        h=args.range_h,
        w=args.range_w,
        fov_up=args.fov_up,
        fov_down=args.fov_down,
        fov_left=args.fov_left,
        fov_right=args.fov_right,
        proj_img_mean=args.proj_img_mean,
        proj_img_stds=args.proj_img_stds,
        augmentation_config=None,  # No augmentation for testing
        pc_extension=args.pc_extension,
        pc_dtype=args.pc_dtype,
        pc_reshape=tuple(args.pc_reshape),
    )

    print(f"\nDataset created successfully!")
    print(f"Total samples: {len(train_dataset)}")

    # Test loading a sample
    print("\n" + "="*80)
    print("Testing sample loading...")
    print("="*80)

    if len(train_dataset) > 0:
        range_views, poses = train_dataset[0]

        print(f"\nSample 0:")
        print(f"  Range views shape: {range_views.shape}")  # Expected: [T, C, H, W]
        print(f"  Range views dtype: {range_views.dtype}")
        print(f"  Range views min: {range_views.min():.4f}")
        print(f"  Range views max: {range_views.max():.4f}")
        print(f"  Range views mean: {range_views.mean():.4f}")

        print(f"\n  Poses shape: {poses.shape}")  # Expected: [T, 4, 4]
        print(f"  Poses dtype: {poses.dtype}")

        print("\n  First pose matrix:")
        print(poses[0])

        print("\n" + "="*80)
        print("Dataloader test PASSED!")
        print("="*80)
    else:
        print("\nERROR: Dataset is empty!")
        return False

    return True


if __name__ == '__main__':
    try:
        success = test_dataloader()
        if success:
            print("\n✓ All tests passed!")
            sys.exit(0)
        else:
            print("\n✗ Tests failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
