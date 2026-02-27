"""
KITTI-specific Range View Dataset
Reads directly from KITTI Odometry directory structure without JSON files
"""

import os
import numpy as np
import torch
import random
import sys
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

# Add DiffLoc to path to import range projection utilities
diffloc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../DiffLoc'))
if diffloc_path not in sys.path:
    sys.path.append(diffloc_path)

from datasets.projection import RangeProjection
from datasets.augmentor import Augmentor, AugmentParams

# Add Epona utils to path
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)


def load_kitti_poses(pose_file):
    """
    Load poses from KITTI ground truth file
    Each line contains 12 values: [r11, r12, r13, t1, r21, r22, r23, t2, r31, r32, r33, t3]
    Returns list of 4x4 transformation matrices
    """
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            if len(values) == 12:
                # Reshape to 3x4 matrix
                pose_3x4 = np.array(values).reshape(3, 4)
                # Convert to 4x4 homogeneous transformation matrix
                pose_4x4 = np.eye(4)
                pose_4x4[:3, :] = pose_3x4
                poses.append(pose_4x4)
    return poses


def pose_matrix_to_quaternion_translation(pose_matrix):
    """
    Convert 4x4 pose matrix to quaternion and translation
    Returns: (rotation_quat, translation)
    """
    rotation_matrix = pose_matrix[:3, :3]
    translation = pose_matrix[:3, 3]

    # Convert rotation matrix to quaternion [qw, qx, qy, qz]
    r = R.from_matrix(rotation_matrix)
    quat = r.as_quat()  # Returns [qx, qy, qz, qw]
    quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to [qw, qx, qy, qz]

    return quat, translation


class KITTIRangeViewDataset(Dataset):
    """
    KITTI Odometry dataset for range view projection training
    """

    def __init__(
        self,
        sequences_path,
        poses_path,
        sequences,
        condition_frames=5,
        h=64,
        w=2048,
        # Range projection parameters
        fov_up=3.0,
        fov_down=-25.0,
        fov_left=-180.0,
        fov_right=180.0,
        # Feature normalization parameters
        proj_img_mean=None,
        proj_img_stds=None,
        # Augmentation parameters
        augmentation_config=None,
        # Point cloud file extension
        pc_extension='.bin',
        pc_dtype=np.float32,
        pc_reshape=(-1, 4),
        is_train=True,
    ):
        self.sequences_path = sequences_path
        self.poses_path = poses_path
        self.sequences = sequences
        self.condition_frames = condition_frames
        self.h = h
        self.w = w
        self.pc_extension = pc_extension
        self.pc_dtype = pc_dtype
        self.pc_reshape = pc_reshape
        self.is_train = is_train

        # Build dataset index
        self.data_index = []  # List of (sequence_id, frame_indices, poses)
        self._build_dataset_index()

        print(f"KITTI Range View Dataset - Total samples: {len(self.data_index)}")
        print(f"Sequences: {sequences}")

        # Initialize range projection
        self.projection = RangeProjection(
            fov_up=fov_up,
            fov_down=fov_down,
            fov_left=fov_left,
            fov_right=fov_right,
            proj_h=h,
            proj_w=w
        )

        # Feature normalization
        if proj_img_mean is None:
            # Default mean for [range, x, y, z, intensity, label]
            proj_img_mean = [10.839, 0.005, 0.494, -1.13, 0.0, 0.0]
        if proj_img_stds is None:
            # Default std for [range, x, y, z, intensity, label]
            proj_img_stds = [9.314, 11.521, 8.262, 0.828, 1.0, 1.0]

        self.proj_img_mean = torch.tensor(proj_img_mean, dtype=torch.float)
        self.proj_img_stds = torch.tensor(proj_img_stds, dtype=torch.float)

        # Initialize augmentation
        if augmentation_config is not None and is_train:
            augment_params = AugmentParams()
            augment_params.setTranslationParams(
                p_transx=augmentation_config.get('p_transx', 0.0),
                trans_xmin=augmentation_config.get('trans_xmin', 0.0),
                trans_xmax=augmentation_config.get('trans_xmax', 0.0),
                p_transy=augmentation_config.get('p_transy', 0.0),
                trans_ymin=augmentation_config.get('trans_ymin', 0.0),
                trans_ymax=augmentation_config.get('trans_ymax', 0.0),
                p_transz=augmentation_config.get('p_transz', 0.0),
                trans_zmin=augmentation_config.get('trans_zmin', 0.0),
                trans_zmax=augmentation_config.get('trans_zmax', 0.0)
            )
            augment_params.setRotationParams(
                p_rot_roll=augmentation_config.get('p_rot_roll', 0.0),
                rot_rollmin=augmentation_config.get('rot_rollmin', 0.0),
                rot_rollmax=augmentation_config.get('rot_rollmax', 0.0),
                p_rot_pitch=augmentation_config.get('p_rot_pitch', 0.0),
                rot_pitchmin=augmentation_config.get('rot_pitchmin', 0.0),
                rot_pitchmax=augmentation_config.get('rot_pitchmax', 0.0),
                p_rot_yaw=augmentation_config.get('p_rot_yaw', 0.0),
                rot_yawmin=augmentation_config.get('rot_yawmin', 0.0),
                rot_yawmax=augmentation_config.get('rot_yawmax', 0.0)
            )
            if 'p_scale' in augmentation_config:
                augment_params.sefScaleParams(
                    p_scale=augmentation_config['p_scale'],
                    scale_min=augmentation_config['scale_min'],
                    scale_max=augmentation_config['scale_max']
                )
            self.augmentor = Augmentor(augment_params)
        else:
            self.augmentor = None

    def _build_dataset_index(self):
        """
        Build index of all valid samples
        Each sample needs condition_frames + 1 consecutive frames
        """
        for seq_id in self.sequences:
            seq_str = f"{seq_id:02d}"

            # Path to velodyne point clouds
            velodyne_path = os.path.join(self.sequences_path, seq_str, 'velodyne')

            # Path to pose file
            pose_file = os.path.join(self.poses_path, f"{seq_str}.txt")

            if not os.path.exists(velodyne_path):
                print(f"Warning: Velodyne path does not exist: {velodyne_path}")
                continue

            if not os.path.exists(pose_file):
                print(f"Warning: Pose file does not exist: {pose_file}")
                continue

            # Load poses for this sequence
            poses = load_kitti_poses(pose_file)

            # Get list of point cloud files
            pc_files = sorted([f for f in os.listdir(velodyne_path) if f.endswith(self.pc_extension)])
            num_frames = len(pc_files)

            # Ensure we have matching number of poses and frames
            if len(poses) != num_frames:
                print(f"Warning: Sequence {seq_str} has {num_frames} frames but {len(poses)} poses")
                num_frames = min(len(poses), num_frames)

            # Create samples: each sample contains condition_frames + 1 frames
            num_samples = num_frames - self.condition_frames
            if num_samples <= 0:
                print(f"Warning: Sequence {seq_str} has insufficient frames ({num_frames})")
                continue

            for start_idx in range(num_samples):
                # For training: randomly sample starting points
                # For validation/test: use all possible windows
                if self.is_train:
                    # Sample with stride for training
                    if start_idx % 5 != 0:  # Sample every 5th frame as start
                        continue

                frame_indices = list(range(start_idx, start_idx + self.condition_frames + 1))
                sample_poses = [poses[i] for i in frame_indices]

                self.data_index.append({
                    'sequence': seq_id,
                    'sequence_str': seq_str,
                    'frame_indices': frame_indices,
                    'poses': sample_poses
                })

    def __len__(self):
        return len(self.data_index)

    def load_pointcloud(self, sequence_str, frame_idx):
        """Load point cloud from KITTI bin file"""
        frame_str = f"{frame_idx:06d}"
        pc_path = os.path.join(
            self.sequences_path, sequence_str, 'velodyne', f"{frame_str}{self.pc_extension}"
        )

        try:
            pc = np.fromfile(pc_path, dtype=self.pc_dtype).reshape(self.pc_reshape)
            return pc
        except Exception as e:
            print(f"Error loading point cloud from {pc_path}: {e}")
            return None

    def project_pointcloud(self, pointcloud):
        """Project point cloud to range view image"""
        # Apply range projection
        proj_pointcloud, proj_range, proj_idx, proj_mask = self.projection.doProjection(pointcloud)

        # Create feature tensor: [range, x, y, z, intensity, label]
        proj_range_tensor = torch.from_numpy(proj_range)  # [H, W]
        proj_xyz_tensor = torch.from_numpy(proj_pointcloud[..., :3])  # [H, W, 3]

        # Handle intensity (KITTI has 4 channels: x, y, z, intensity)
        if pointcloud.shape[1] >= 4:
            proj_intensity_tensor = torch.from_numpy(proj_pointcloud[..., 3])  # [H, W]
        else:
            proj_intensity_tensor = torch.zeros_like(proj_range_tensor)

        # KITTI doesn't have semantic labels, so use zeros
        proj_label_tensor = torch.zeros_like(proj_range_tensor)

        proj_mask_tensor = torch.from_numpy(proj_mask)  # [H, W]

        # Stack features: [6, H, W] - [range, x, y, z, intensity, label]
        proj_feature_tensor = torch.cat([
            proj_range_tensor.unsqueeze(0),  # [1, H, W]
            proj_xyz_tensor.permute(2, 0, 1),  # [3, H, W]
            proj_intensity_tensor.unsqueeze(0),  # [1, H, W]
            proj_label_tensor.unsqueeze(0)  # [1, H, W]
        ], dim=0)

        # Normalize
        proj_feature_tensor = (proj_feature_tensor - self.proj_img_mean[:, None, None]) / self.proj_img_stds[:, None, None]
        proj_feature_tensor = proj_feature_tensor * proj_mask_tensor.unsqueeze(0).float()

        return proj_feature_tensor

    def __getitem__(self, index):
        """Get a sample from the dataset"""
        sample = self.data_index[index]
        sequence_str = sample['sequence_str']
        frame_indices = sample['frame_indices']
        poses = sample['poses']

        range_views = []
        poses_matrices = []

        for i, frame_idx in enumerate(frame_indices):
            # Load point cloud
            pc = self.load_pointcloud(sequence_str, frame_idx)

            if pc is None:
                # If loading fails, return a random valid sample
                return self.__getitem__(random.randint(0, len(self) - 1))

            # Apply augmentation during training
            if self.augmentor is not None and self.is_train:
                pc, rotation = self.augmentor.doAugmentation(pc)
                # Note: We could adjust poses based on augmentation rotation, but for now we skip this

            # Project to range view
            range_view = self.project_pointcloud(pc)
            range_views.append(range_view)

            # Get pose matrix
            poses_matrices.append(poses[i])

        # Stack range views: [T, C, H, W]
        range_views_tensor = torch.stack(range_views, dim=0)

        # Stack poses: [T, 4, 4]
        poses_tensor = torch.from_numpy(np.array(poses_matrices)).float()

        return range_views_tensor, poses_tensor


class KITTIRangeViewTrainDataset(KITTIRangeViewDataset):
    """Training dataset"""
    def __init__(self, *args, **kwargs):
        kwargs['is_train'] = True
        super().__init__(*args, **kwargs)


class KITTIRangeViewValDataset(KITTIRangeViewDataset):
    """Validation dataset"""
    def __init__(self, *args, **kwargs):
        kwargs['is_train'] = False
        super().__init__(*args, **kwargs)


class KITTIRangeViewTestDataset(KITTIRangeViewDataset):
    """Test dataset"""
    def __init__(self, *args, **kwargs):
        kwargs['is_train'] = False
        super().__init__(*args, **kwargs)
