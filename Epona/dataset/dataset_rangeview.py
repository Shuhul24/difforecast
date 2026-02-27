"""
Range View Dataset for Epona
This dataset uses range view image projections from point clouds instead of RGB images.
Adapted from Epona's dataset.py and DiffLoc's oxford.py
"""

import json
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

from dataset.datasets_utils import reverse_seq_data, get_meta_data


def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix"""
    q_w, q_x, q_y, q_z = q

    R = np.array([
        [1 - 2*(q_y**2 + q_z**2), 2*(q_x*q_y - q_w*q_z), 2*(q_x*q_z + q_w*q_y)],
        [2*(q_x*q_y + q_w*q_z), 1 - 2*(q_x**2 + q_z**2), 2*(q_y*q_z - q_w*q_x)],
        [2*(q_x*q_z - q_w*q_y), 2*(q_y*q_z + q_w*q_x), 1 - 2*(q_x**2 + q_y**2)]
    ])
    return R


def create_transformation_matrix(rotation, translation):
    """Create 4x4 transformation matrix from rotation and translation"""
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def poses_foraugmentation(rotation, pose):
    """Apply rotation augmentation to pose"""
    # rotation: [3, 3] rotation matrix from augmentation
    # pose: [6] pose vector (x, y, z, roll, pitch, yaw)

    # Extract euler angles from augmentation rotation
    r = R.from_matrix(rotation)
    aug_euler = r.as_euler('xyz', degrees=False)

    # Add augmentation to pose
    pose_new = pose.copy()
    pose_new[3:] += aug_euler

    return pose_new


class TrainRangeViewDataset(Dataset):
    """Training dataset using range view projections"""

    def __init__(
        self,
        data_path,
        json_path,
        condition_frames=9,
        downsample_fps=3,
        h=64,
        w=512,
        # Range projection parameters
        fov_up=10.0,
        fov_down=-30.0,
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
        pc_reshape=(-1, 5),  # (x, y, z, intensity, label)
    ):
        self.pc_path_data = []
        self.pose_data = []
        self.data_path = data_path
        self.condition_frames = condition_frames
        self.h = h
        self.w = w
        self.pc_extension = pc_extension
        self.pc_dtype = pc_dtype
        self.pc_reshape = pc_reshape

        # Load metadata from JSON
        with open(json_path, 'r', encoding='utf-8') as file:
            preprocess_data = json.load(file)

        self.ori_fps = 10
        self.downsample = self.ori_fps // downsample_fps

        # Build dataset
        keys = sorted(list(preprocess_data.keys()))
        for video_key in keys:
            tmp_pc_path = []
            tmp_pose = []
            path_poses = preprocess_data[video_key]

            if len(path_poses) <= self.condition_frames * self.downsample:
                continue

            for path_pose in path_poses:
                # Convert image path to point cloud path
                data_path_str = path_pose['data_path']
                # Replace image extension with point cloud extension
                pc_path = os.path.splitext(data_path_str)[0] + self.pc_extension
                tmp_pc_path.append(os.path.join(data_path, pc_path))
                tmp_pose.append(path_pose['ego_pose'])

            self.pc_path_data.append(tmp_pc_path)
            self.pose_data.append(tmp_pose)

        print(f"Range View Dataset - Total sequences: {len(self.pc_path_data)}")

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
            proj_img_mean = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if proj_img_stds is None:
            # Default std for [range, x, y, z, intensity, label]
            proj_img_stds = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        self.proj_img_mean = torch.tensor(proj_img_mean, dtype=torch.float)
        self.proj_img_stds = torch.tensor(proj_img_stds, dtype=torch.float)

        # Initialize augmentation
        if augmentation_config is not None:
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

    def __len__(self):
        return len(self.pc_path_data)

    def load_pointcloud(self, path):
        """Load point cloud from file"""
        try:
            pc = np.fromfile(path, dtype=self.pc_dtype).reshape(self.pc_reshape)
            return pc
        except Exception as e:
            print(f"Error loading point cloud from {path}: {e}")
            return None

    def project_pointcloud(self, pointcloud):
        """Project point cloud to range view image"""
        # Apply range projection
        proj_pointcloud, proj_range, proj_idx, proj_mask = self.projection.doProjection(pointcloud)

        # Create feature tensor: [range, x, y, z, intensity, label]
        proj_range_tensor = torch.from_numpy(proj_range)  # [H, W]
        proj_xyz_tensor = torch.from_numpy(proj_pointcloud[..., :3])  # [H, W, 3]

        # Handle intensity and label based on pointcloud shape
        if pointcloud.shape[1] >= 4:
            proj_intensity_tensor = torch.from_numpy(proj_pointcloud[..., 3])  # [H, W]
        else:
            proj_intensity_tensor = torch.zeros_like(proj_range_tensor)

        if pointcloud.shape[1] >= 5:
            proj_label_tensor = torch.from_numpy(proj_pointcloud[..., 4])  # [H, W]
        else:
            proj_label_tensor = torch.zeros_like(proj_range_tensor)

        proj_mask_tensor = torch.from_numpy(proj_mask)  # [H, W]

        # Stack features: [6, H, W]
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

    def get_pc_feature(self, index):
        """Load point clouds and poses for a sequence"""
        pc_paths = self.pc_path_data[index]
        poses = self.pose_data[index]
        clip_length = len(pc_paths)

        start = 0  # random.randint(0, clip_length-(self.condition_frames+1)*self.downsample)

        range_views = []
        poses_new = []

        for i in range(self.condition_frames + 1):
            # Load point cloud
            pc = self.load_pointcloud(pc_paths[start + i * self.downsample])

            if pc is None:
                print(f'Warning: Point cloud does not exist at {pc_paths[start + i * self.downsample]}')
                return None, None

            # Apply augmentation during training
            if self.augmentor is not None:
                pc, rotation = self.augmentor.doAugmentation(pc)
            else:
                rotation = np.eye(3)

            # Project to range view
            range_view = self.project_pointcloud(pc)
            range_views.append(range_view)

            # Get pose
            rotation_mat = quaternion_to_rotation_matrix(poses[start + i * self.downsample]["rotation"])
            pose = create_transformation_matrix(rotation_mat, poses[start + i * self.downsample]["translation"])
            poses_new.append(pose)

        poses_array = np.array(poses_new)
        return range_views, poses_array

    def __getitem__(self, index):
        """Get a sample from the dataset"""
        while True:
            range_views, poses = self.get_pc_feature(index)

            if (range_views is not None) and (poses is not None):
                break
            else:
                index = random.randint(0, self.__len__() - 1)

        # Stack range views: [T, C, H, W]
        range_views_tensor = torch.stack(range_views, dim=0)
        poses_tensor = torch.from_numpy(poses).float()

        return range_views_tensor, poses_tensor


class ValRangeViewDataset(Dataset):
    """Validation dataset using range view projections"""

    def __init__(
        self,
        data_path,
        json_path,
        condition_frames=3,
        downsample_fps=3,
        h=64,
        w=512,
        target_frame=-5,
        # Range projection parameters
        fov_up=10.0,
        fov_down=-30.0,
        fov_left=-180.0,
        fov_right=180.0,
        # Feature normalization parameters
        proj_img_mean=None,
        proj_img_stds=None,
        # Point cloud file extension
        pc_extension='.bin',
        pc_dtype=np.float32,
        pc_reshape=(-1, 5),
    ):
        self.pc_path_data = []
        self.pose_data = []
        self.target_frame = target_frame
        assert self.target_frame < 0

        self.data_path = data_path
        self.condition_frames = condition_frames
        self.h = h
        self.w = w
        self.pc_extension = pc_extension
        self.pc_dtype = pc_dtype
        self.pc_reshape = pc_reshape

        # Load metadata from JSON
        with open(json_path, 'r', encoding='utf-8') as file:
            preprocess_data = json.load(file)

        self.ori_fps = 12
        self.downsample = self.ori_fps // downsample_fps

        # Build dataset
        keys = sorted(list(preprocess_data.keys()))
        for video_key in keys:
            tmp_pc_path = []
            tmp_pose = []
            path_poses = preprocess_data[video_key]

            for path_pose in path_poses:
                # Convert image path to point cloud path
                data_path_str = path_pose['data_path']
                pc_path = os.path.splitext(data_path_str)[0] + self.pc_extension
                tmp_pc_path.append(os.path.join(data_path, pc_path))
                tmp_pose.append(path_pose['ego_pose'])

            self.pc_path_data.append(tmp_pc_path)
            self.pose_data.append(tmp_pose)

        print(f"Range View Val Dataset - Total sequences: {len(self.pc_path_data)}")

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
            proj_img_mean = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if proj_img_stds is None:
            proj_img_stds = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        self.proj_img_mean = torch.tensor(proj_img_mean, dtype=torch.float)
        self.proj_img_stds = torch.tensor(proj_img_stds, dtype=torch.float)

    def __len__(self):
        return len(self.pc_path_data)

    def load_pointcloud(self, path):
        """Load point cloud from file"""
        try:
            pc = np.fromfile(path, dtype=self.pc_dtype).reshape(self.pc_reshape)
            return pc
        except Exception as e:
            print(f"Error loading point cloud from {path}: {e}")
            return None

    def project_pointcloud(self, pointcloud):
        """Project point cloud to range view image"""
        proj_pointcloud, proj_range, proj_idx, proj_mask = self.projection.doProjection(pointcloud)

        proj_range_tensor = torch.from_numpy(proj_range)
        proj_xyz_tensor = torch.from_numpy(proj_pointcloud[..., :3])

        if pointcloud.shape[1] >= 4:
            proj_intensity_tensor = torch.from_numpy(proj_pointcloud[..., 3])
        else:
            proj_intensity_tensor = torch.zeros_like(proj_range_tensor)

        if pointcloud.shape[1] >= 5:
            proj_label_tensor = torch.from_numpy(proj_pointcloud[..., 4])
        else:
            proj_label_tensor = torch.zeros_like(proj_range_tensor)

        proj_mask_tensor = torch.from_numpy(proj_mask)

        proj_feature_tensor = torch.cat([
            proj_range_tensor.unsqueeze(0),
            proj_xyz_tensor.permute(2, 0, 1),
            proj_intensity_tensor.unsqueeze(0),
            proj_label_tensor.unsqueeze(0)
        ], dim=0)

        proj_feature_tensor = (proj_feature_tensor - self.proj_img_mean[:, None, None]) / self.proj_img_stds[:, None, None]
        proj_feature_tensor = proj_feature_tensor * proj_mask_tensor.unsqueeze(0).float()

        return proj_feature_tensor

    def get_pc_feature(self, index):
        """Load point clouds and poses for a sequence"""
        pc_paths = self.pc_path_data[index]
        poses = self.pose_data[index]
        clip_length = len(pc_paths)

        target_index = clip_length + self.target_frame
        start_index = target_index - self.downsample * (self.condition_frames + 1)

        range_views = []
        poses_new = []

        for i in range(self.condition_frames + 1):
            pc = self.load_pointcloud(pc_paths[start_index + i * self.downsample])

            if pc is None:
                print(f'Warning: Point cloud does not exist')
                return None, None

            range_view = self.project_pointcloud(pc)
            range_views.append(range_view)

            rotation_mat = quaternion_to_rotation_matrix(poses[start_index + i * self.downsample]["rotation"])
            pose = create_transformation_matrix(rotation_mat, poses[start_index + i * self.downsample]["translation"])
            poses_new.append(pose)

        poses_array = np.array(poses_new)
        return range_views, poses_array

    def __getitem__(self, index):
        """Get a sample from the dataset"""
        range_views, poses = self.get_pc_feature(index)

        if range_views is None or poses is None:
            # Return dummy data if loading fails
            range_views = [torch.zeros(6, self.h, self.w) for _ in range(self.condition_frames + 1)]
            poses = np.zeros((self.condition_frames + 1, 4, 4))

        range_views_tensor = torch.stack(range_views, dim=0)
        poses_tensor = torch.from_numpy(poses).float()

        return range_views_tensor, poses_tensor


class TestRangeViewDataset(Dataset):
    """Test dataset using range view projections"""

    def __init__(
        self,
        data_path,
        json_path,
        condition_frames=3,
        downsample_fps=3,
        h=64,
        w=512,
        # Range projection parameters
        fov_up=10.0,
        fov_down=-30.0,
        fov_left=-180.0,
        fov_right=180.0,
        # Feature normalization parameters
        proj_img_mean=None,
        proj_img_stds=None,
        # Point cloud file extension
        pc_extension='.bin',
        pc_dtype=np.float32,
        pc_reshape=(-1, 5),
    ):
        self.pc_path_data = []
        self.pose_data = []
        self.data_path = data_path
        self.condition_frames = condition_frames
        self.h = h
        self.w = w
        self.pc_extension = pc_extension
        self.pc_dtype = pc_dtype
        self.pc_reshape = pc_reshape

        # Load metadata from JSON
        with open(json_path, 'r', encoding='utf-8') as file:
            preprocess_data = json.load(file)

        self.ori_fps = 12
        self.downsample = self.ori_fps // downsample_fps

        # Build dataset
        keys = sorted(list(preprocess_data.keys()))
        for video_key in keys:
            tmp_pc_path = []
            tmp_pose = []
            path_poses = preprocess_data[video_key]

            for path_pose in path_poses:
                # Convert image path to point cloud path
                data_path_str = path_pose['data_path']
                pc_path = os.path.splitext(data_path_str)[0] + self.pc_extension
                tmp_pc_path.append(os.path.join(data_path, pc_path))
                tmp_pose.append(path_pose['ego_pose'])

            self.pc_path_data.append(tmp_pc_path)
            self.pose_data.append(tmp_pose)

        print(f"Range View Test Dataset - Total sequences: {len(self.pc_path_data)}")

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
            proj_img_mean = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if proj_img_stds is None:
            proj_img_stds = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        self.proj_img_mean = torch.tensor(proj_img_mean, dtype=torch.float)
        self.proj_img_stds = torch.tensor(proj_img_stds, dtype=torch.float)

    def __len__(self):
        return len(self.pc_path_data)

    def load_pointcloud(self, path):
        """Load point cloud from file"""
        try:
            pc = np.fromfile(path, dtype=self.pc_dtype).reshape(self.pc_reshape)
            return pc
        except Exception as e:
            print(f"Error loading point cloud from {path}: {e}")
            return None

    def project_pointcloud(self, pointcloud):
        """Project point cloud to range view image"""
        proj_pointcloud, proj_range, proj_idx, proj_mask = self.projection.doProjection(pointcloud)

        proj_range_tensor = torch.from_numpy(proj_range)
        proj_xyz_tensor = torch.from_numpy(proj_pointcloud[..., :3])

        if pointcloud.shape[1] >= 4:
            proj_intensity_tensor = torch.from_numpy(proj_pointcloud[..., 3])
        else:
            proj_intensity_tensor = torch.zeros_like(proj_range_tensor)

        if pointcloud.shape[1] >= 5:
            proj_label_tensor = torch.from_numpy(proj_pointcloud[..., 4])
        else:
            proj_label_tensor = torch.zeros_like(proj_range_tensor)

        proj_mask_tensor = torch.from_numpy(proj_mask)

        proj_feature_tensor = torch.cat([
            proj_range_tensor.unsqueeze(0),
            proj_xyz_tensor.permute(2, 0, 1),
            proj_intensity_tensor.unsqueeze(0),
            proj_label_tensor.unsqueeze(0)
        ], dim=0)

        proj_feature_tensor = (proj_feature_tensor - self.proj_img_mean[:, None, None]) / self.proj_img_stds[:, None, None]
        proj_feature_tensor = proj_feature_tensor * proj_mask_tensor.unsqueeze(0).float()

        return proj_feature_tensor

    def get_pc_feature(self, index):
        """Load all point clouds and poses for a sequence"""
        pc_paths = self.pc_path_data[index]
        poses = self.pose_data[index]
        clip_length = len(pc_paths) // self.downsample

        range_views = []
        poses_new = []

        for i in range(clip_length):
            pc = self.load_pointcloud(pc_paths[i * self.downsample])

            if pc is None:
                print(f'Warning: Point cloud does not exist')
                return None, None

            range_view = self.project_pointcloud(pc)
            range_views.append(range_view)

            rotation_mat = quaternion_to_rotation_matrix(poses[i * self.downsample]["rotation"])
            pose = create_transformation_matrix(rotation_mat, poses[i * self.downsample]["translation"])
            poses_new.append(pose)

        poses_array = np.array(poses_new)
        return range_views, poses_array

    def __getitem__(self, index):
        """Get a sample from the dataset"""
        range_views, poses = self.get_pc_feature(index)

        if range_views is None or poses is None:
            # Return dummy data if loading fails
            range_views = [torch.zeros(6, self.h, self.w)]
            poses = np.zeros((1, 4, 4))

        range_views_tensor = torch.stack(range_views, dim=0)
        poses_tensor = torch.from_numpy(poses).float()

        return range_views_tensor, poses_tensor
