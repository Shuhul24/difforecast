"""
Range View Dataset for Epona (JSON-based, e.g. nuScenes / nuPlan).

Expects a JSON file whose structure mirrors Epona's standard format:
    {
        "video_key": [
            {"data_path": "rel/path/to/frame.jpg",
             "ego_pose": {"rotation": [qw, qx, qy, qz],
                          "translation": [x, y, z]}},
            ...
        ],
        ...
    }

The ``data_path`` extension is swapped for ``pc_extension`` to locate the
corresponding point-cloud file.

Uses RangeProjection and Augmentor utilities.

Returns per sample:
    range_views : FloatTensor [T, 6, H, W]   (condition_frames + 1 frames)
    poses       : FloatTensor [T, 4, 4]       absolute 4×4 pose matrices
"""

import json
import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset

from .projection import RangeProjection

try:
    from .augmentor import Augmentor, AugmentParams
    _AUGMENTOR_OK = True
except ImportError:
    Augmentor = AugmentParams = None
    _AUGMENTOR_OK = False
    print("Warning: Augmentor unavailable (missing robotcar SDK). "
          "Augmentation will be disabled.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _quat_to_rot(q):
    """Convert [qw, qx, qy, qz] to a 3×3 rotation matrix."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx**2 + qy**2)],
    ])


def _make_pose(ego_pose):
    """Build a 4×4 transformation matrix from an ego_pose dict."""
    T = np.eye(4)
    T[:3, :3] = _quat_to_rot(ego_pose['rotation'])
    T[:3, 3]  = ego_pose['translation']
    return T


def _make_augmentor(cfg):
    """Build a DiffLoc Augmentor from an augmentation config dict."""
    if not _AUGMENTOR_OK or cfg is None:
        return None
    p = AugmentParams()
    p.setTranslationParams(
        p_transx=cfg.get('p_transx', 0.), trans_xmin=cfg.get('trans_xmin', 0.), trans_xmax=cfg.get('trans_xmax', 0.),
        p_transy=cfg.get('p_transy', 0.), trans_ymin=cfg.get('trans_ymin', 0.), trans_ymax=cfg.get('trans_ymax', 0.),
        p_transz=cfg.get('p_transz', 0.), trans_zmin=cfg.get('trans_zmin', 0.), trans_zmax=cfg.get('trans_zmax', 0.),
    )
    p.setRotationParams(
        p_rot_roll=cfg.get('p_rot_roll', 0.),  rot_rollmin=cfg.get('rot_rollmin', 0.),  rot_rollmax=cfg.get('rot_rollmax', 0.),
        p_rot_pitch=cfg.get('p_rot_pitch', 0.), rot_pitchmin=cfg.get('rot_pitchmin', 0.), rot_pitchmax=cfg.get('rot_pitchmax', 0.),
        p_rot_yaw=cfg.get('p_rot_yaw', 0.),   rot_yawmin=cfg.get('rot_yawmin', 0.),   rot_yawmax=cfg.get('rot_yawmax', 0.),
    )
    if 'p_scale' in cfg:
        p.sefScaleParams(p_scale=cfg['p_scale'], scale_min=cfg['scale_min'], scale_max=cfg['scale_max'])
    return Augmentor(p)


# ── Base class ────────────────────────────────────────────────────────────────

class _BaseRangeViewDataset(Dataset):
    """Shared projection / normalisation / loading logic for JSON-based datasets."""

    _DEFAULT_MEAN = [0., 0., 0., 0., 0., 0.]
    _DEFAULT_STD  = [1., 1., 1., 1., 1., 1.]

    def __init__(
        self,
        h, w,
        fov_up, fov_down, fov_left, fov_right,
        proj_img_mean, proj_img_stds,
        pc_extension, pc_dtype, pc_reshape,
    ):
        self.h, self.w      = h, w
        self.pc_extension   = pc_extension
        self.pc_dtype       = pc_dtype
        self.pc_reshape     = pc_reshape

        self.projection = RangeProjection(
            fov_up=fov_up, fov_down=fov_down,
            fov_left=fov_left, fov_right=fov_right,
            proj_h=h, proj_w=w,
        )

        mean = proj_img_mean or self._DEFAULT_MEAN
        std  = proj_img_stds or self._DEFAULT_STD
        self.mean = torch.tensor(mean, dtype=torch.float)
        self.std  = torch.tensor(std,  dtype=torch.float)

    # ── JSON parsing ──────────────────────────────────────────────────────────

    def _parse_json(self, json_path, data_path):
        """Return two parallel lists-of-lists: pc_paths and ego_poses."""
        with open(json_path, encoding='utf-8') as f:
            raw = json.load(f)
        pc_paths_all, poses_all = [], []
        for key in sorted(raw.keys()):
            pc_paths, poses = [], []
            for item in raw[key]:
                stem = os.path.splitext(item['data_path'])[0]
                pc_paths.append(os.path.join(data_path, stem + self.pc_extension))
                poses.append(item['ego_pose'])
            pc_paths_all.append(pc_paths)
            poses_all.append(poses)
        return pc_paths_all, poses_all

    # ── Point-cloud I/O ───────────────────────────────────────────────────────

    def _load_pc(self, path):
        try:
            return np.fromfile(path, dtype=self.pc_dtype).reshape(self.pc_reshape)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    # ── Projection ────────────────────────────────────────────────────────────

    def _project(self, pc):
        """Project a point cloud → normalised [6, H, W] feature tensor.

        Channels: [range, x, y, z, intensity, label].
        Unoccupied pixels are zeroed via the projection mask.
        """
        proj_pc, proj_range, _, proj_mask = self.projection.doProjection(pc)

        depth  = torch.from_numpy(proj_range)                               # [H, W]
        xyz    = torch.from_numpy(proj_pc[..., :3]).permute(2, 0, 1)        # [3, H, W]
        intens = (torch.from_numpy(proj_pc[..., 3])
                  if pc.shape[1] >= 4 else torch.zeros_like(depth))         # [H, W]
        label  = (torch.from_numpy(proj_pc[..., 4])
                  if pc.shape[1] >= 5 else torch.zeros_like(depth))         # [H, W]
        mask   = torch.from_numpy(proj_mask.astype(np.float32))             # [H, W]

        feat = torch.cat([depth.unsqueeze(0), xyz, intens.unsqueeze(0), label.unsqueeze(0)], 0)
        feat = (feat - self.mean[:, None, None]) / self.std[:, None, None]
        return feat * mask.unsqueeze(0)


# ── Training dataset ──────────────────────────────────────────────────────────

class TrainRangeViewDataset(_BaseRangeViewDataset):
    """Training dataset from JSON-annotated point-cloud sequences.

    Each sequence maps to one sample per call to ``__getitem__``; the start
    frame is drawn uniformly at random within the valid range so that
    different parts of long sequences are seen across epochs.
    """

    def __init__(
        self,
        data_path,
        json_path,
        condition_frames=9,
        ori_fps=10,
        downsample_fps=3,
        h=64, w=512,
        fov_up=10., fov_down=-30., fov_left=-180., fov_right=180.,
        proj_img_mean=None,
        proj_img_stds=None,
        augmentation_config=None,
        pc_extension='.bin',
        pc_dtype=np.float32,
        pc_reshape=(-1, 5),
    ):
        super().__init__(h, w, fov_up, fov_down, fov_left, fov_right,
                         proj_img_mean, proj_img_stds, pc_extension, pc_dtype, pc_reshape)
        self.condition_frames = condition_frames
        self.downsample       = ori_fps // downsample_fps
        self.augmentor        = _make_augmentor(augmentation_config)

        all_paths, all_poses = self._parse_json(json_path, data_path)

        # Keep only sequences long enough to yield at least one window
        min_len = condition_frames * self.downsample  # original Epona convention
        self.pc_paths = [p for p, q in zip(all_paths, all_poses) if len(p) > min_len]
        self.poses    = [q for p, q in zip(all_paths, all_poses) if len(p) > min_len]

        print(f"TrainRangeViewDataset: {len(self.pc_paths)} sequences loaded.")

    def __len__(self):
        return len(self.pc_paths)

    def __getitem__(self, idx):
        pc_paths = self.pc_paths[idx]
        poses    = self.poses[idx]

        # Random start so all parts of the sequence are visited
        max_start = len(pc_paths) - (self.condition_frames + 1) * self.downsample
        start = random.randint(0, max(0, max_start))

        views, pose_list = [], []
        for i in range(self.condition_frames + 1):
            pc = self._load_pc(pc_paths[start + i * self.downsample])
            if pc is None:
                return self.__getitem__(random.randint(0, len(self) - 1))
            if self.augmentor is not None:
                pc, _ = self.augmentor.doAugmentation(pc)
            views.append(self._project(pc))
            pose_list.append(_make_pose(poses[start + i * self.downsample]))

        return (
            torch.stack(views),                                 # [T, 6, H, W]
            torch.from_numpy(np.array(pose_list)).float(),      # [T, 4, 4]
        )


# ── Validation dataset ────────────────────────────────────────────────────────

class ValRangeViewDataset(_BaseRangeViewDataset):
    """Validation dataset — fixed window anchored near the end of each sequence.

    ``target_frame`` (negative) picks the target frame index from the end of the
    sequence, and the conditioning window is placed immediately before it.
    No augmentation is applied.
    """

    def __init__(
        self,
        data_path,
        json_path,
        condition_frames=3,
        ori_fps=12,
        downsample_fps=3,
        h=64, w=512,
        target_frame=-5,
        fov_up=10., fov_down=-30., fov_left=-180., fov_right=180.,
        proj_img_mean=None,
        proj_img_stds=None,
        pc_extension='.bin',
        pc_dtype=np.float32,
        pc_reshape=(-1, 5),
    ):
        super().__init__(h, w, fov_up, fov_down, fov_left, fov_right,
                         proj_img_mean, proj_img_stds, pc_extension, pc_dtype, pc_reshape)
        assert target_frame < 0, "target_frame must be negative (offset from end)"
        self.condition_frames = condition_frames
        self.downsample       = ori_fps // downsample_fps
        self.target_frame     = target_frame

        self.pc_paths, self.poses = self._parse_json(json_path, data_path)
        print(f"ValRangeViewDataset: {len(self.pc_paths)} sequences loaded.")

    def __len__(self):
        return len(self.pc_paths)

    def __getitem__(self, idx):
        pc_paths = self.pc_paths[idx]
        poses    = self.poses[idx]
        clip_len = len(pc_paths)

        target_idx = clip_len + self.target_frame
        start_idx  = target_idx - self.downsample * (self.condition_frames + 1)

        if start_idx < 0:
            # Sequence too short — return zeros
            views    = [torch.zeros(6, self.h, self.w) for _ in range(self.condition_frames + 1)]
            pose_arr = np.zeros((self.condition_frames + 1, 4, 4))
            return torch.stack(views), torch.from_numpy(pose_arr).float()

        views, pose_list = [], []
        for i in range(self.condition_frames + 1):
            pc = self._load_pc(pc_paths[start_idx + i * self.downsample])
            if pc is None:
                views    = [torch.zeros(6, self.h, self.w) for _ in range(self.condition_frames + 1)]
                pose_arr = np.zeros((self.condition_frames + 1, 4, 4))
                return torch.stack(views), torch.from_numpy(pose_arr).float()
            views.append(self._project(pc))
            pose_list.append(_make_pose(poses[start_idx + i * self.downsample]))

        return (
            torch.stack(views),                                 # [T, 6, H, W]
            torch.from_numpy(np.array(pose_list)).float(),      # [T, 4, 4]
        )


# ── Test dataset ──────────────────────────────────────────────────────────────

class TestRangeViewDataset(_BaseRangeViewDataset):
    """Test dataset — loads all downsampled frames from each sequence.

    Each call to ``__getitem__`` returns the full (variable-length) sequence,
    so a batch size of 1 is expected.
    """

    def __init__(
        self,
        data_path,
        json_path,
        condition_frames=3,
        ori_fps=12,
        downsample_fps=3,
        h=64, w=512,
        fov_up=10., fov_down=-30., fov_left=-180., fov_right=180.,
        proj_img_mean=None,
        proj_img_stds=None,
        pc_extension='.bin',
        pc_dtype=np.float32,
        pc_reshape=(-1, 5),
    ):
        super().__init__(h, w, fov_up, fov_down, fov_left, fov_right,
                         proj_img_mean, proj_img_stds, pc_extension, pc_dtype, pc_reshape)
        self.condition_frames = condition_frames
        self.downsample       = ori_fps // downsample_fps

        self.pc_paths, self.poses = self._parse_json(json_path, data_path)
        print(f"TestRangeViewDataset: {len(self.pc_paths)} sequences loaded.")

    def __len__(self):
        return len(self.pc_paths)

    def __getitem__(self, idx):
        pc_paths  = self.pc_paths[idx]
        poses     = self.poses[idx]
        clip_len  = len(pc_paths) // self.downsample

        views, pose_list = [], []
        for i in range(clip_len):
            pc = self._load_pc(pc_paths[i * self.downsample])
            if pc is None:
                # Return what has been collected so far (or a single zero frame)
                if not views:
                    views    = [torch.zeros(6, self.h, self.w)]
                    pose_list = [np.zeros((4, 4))]
                break
            views.append(self._project(pc))
            pose_list.append(_make_pose(poses[i * self.downsample]))

        return (
            torch.stack(views),                                 # [T, 6, H, W]
            torch.from_numpy(np.array(pose_list)).float(),      # [T, 4, 4]
        )
