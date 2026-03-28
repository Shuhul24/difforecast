"""
KITTI Range View Dataset for Epona range view forecasting.

Reads directly from KITTI Odometry directory structure (no JSON required).
Uses RangeProjection and Augmentor utilities.

Returns per sample:
    range_views : FloatTensor [T, 2, H, W]   (condition_frames + 1 frames; [range, intensity])
    poses       : FloatTensor [T, 4, 4]       absolute 4×4 pose matrices
"""

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

def load_kitti_poses(pose_file):
    """Return a list of 4×4 float64 pose matrices from a KITTI ground-truth file.

    Each line contains 12 values representing a 3×4 [R | t] matrix.
    """
    poses = []
    with open(pose_file) as f:
        for line in f:
            vals = list(map(float, line.split()))
            if len(vals) == 12:
                T = np.eye(4, dtype=np.float64)
                T[:3] = np.array(vals).reshape(3, 4)
                poses.append(T)
    return poses


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


# ── Dataset ───────────────────────────────────────────────────────────────────

class KITTIRangeViewDataset(Dataset):
    """KITTI Odometry range view dataset for temporal forecasting.

    Builds a sliding-window index over the specified sequences.  During
    training the window stride is ``TRAIN_STRIDE`` (every 5th start frame);
    during validation / test every possible window is used.

    Args:
        sequences_path   : Root of KITTI sequences, e.g. ``…/dataset/sequences/``.
        poses_path       : Directory that contains ``00.txt`` … ``10.txt`` pose files.
        sequences        : List of integer sequence IDs to include.
        condition_frames : Number of *conditioning* frames; total window = condition_frames + 1.
        h / w            : Range image height / width.
        fov_*            : Vertical / horizontal field-of-view limits (degrees).
        proj_img_mean/stds: Per-channel normalisation for [range, intensity].
        augmentation_config: Dict of augmentation parameters (training only); ``None`` disables.
        pc_extension     : Point-cloud file extension (default ``.bin``).
        pc_dtype / pc_reshape: NumPy dtype and reshape tuple for loading binary PCs.
        is_train         : Enables training-mode stride and augmentation.
    """

    # Default normalisation stats for [range, intensity]
    _DEFAULT_MEAN = [10.839, 0.0]
    _DEFAULT_STD  = [ 9.314, 1.0]

    # Default normalisation stats for 5-channel mode [range, x, y, z, intensity]
    # x/y are roughly symmetric around 0 with ~10 m std; z has ~2 m std.
    # Calibrate from data after first use.
    _DEFAULT_MEAN_5CH = [10.839, 0.0,  0.0, 0.0, 0.0]
    _DEFAULT_STD_5CH  = [ 9.314, 10.0, 10.0, 2.0, 1.0]

    # Sliding-window stride when is_train=True
    TRAIN_STRIDE = 5

    def __init__(
        self,
        sequences_path,
        poses_path,
        sequences,
        condition_frames=5,
        forward_iter=1,
        h=64, w=2048,
        fov_up=3.0, fov_down=-25.0,
        fov_left=-180.0, fov_right=180.0,
        proj_img_mean=None,
        proj_img_stds=None,
        augmentation_config=None,
        pc_extension='.bin',
        pc_dtype=np.float32,
        pc_reshape=(-1, 4),
        is_train=True,
        stride=None,
        five_channel=False,
        log_range=False,
    ):
        self.sequences_path  = sequences_path
        self.condition_frames = condition_frames
        self.forward_iter = forward_iter
        self.h, self.w       = h, w
        self.pc_extension    = pc_extension
        self.pc_dtype        = pc_dtype
        self.pc_reshape      = pc_reshape
        self.is_train        = is_train
        self.stride          = stride
        self.five_channel    = five_channel   # return [range,x,y,z,intensity] if True
        self.log_range       = log_range      # use log2(r+1)/6 norm instead of mean/std

        # Range projection
        self.projection = RangeProjection(
            fov_up=fov_up, fov_down=fov_down,
            fov_left=fov_left, fov_right=fov_right,
            proj_h=h, proj_w=w,
        )

        # Feature normalisation tensors
        if proj_img_mean is None:
            mean = self._DEFAULT_MEAN_5CH if five_channel else self._DEFAULT_MEAN
        else:
            mean = proj_img_mean
        if proj_img_stds is None:
            std = self._DEFAULT_STD_5CH if five_channel else self._DEFAULT_STD
        else:
            std = proj_img_stds
        self.mean = torch.tensor(mean, dtype=torch.float)
        self.std  = torch.tensor(std,  dtype=torch.float)

        # Augmentation (training only)
        self.augmentor = _make_augmentor(augmentation_config) if is_train else None

        # Build the flat sample index
        self.index = self._build_index(sequences, poses_path)
        print(f"KITTIRangeViewDataset [{'train' if is_train else 'val/test'}]: "
              f"{len(self.index)} samples — sequences {sequences}")

    # ── Index ──────────────────────────────────────────────────────────────────

    def _build_index(self, sequences, poses_path):
        """Return a list of (seq_str, start_frame, pose_window) tuples."""
        index  = []
        stride = (self.stride if self.stride is not None else self.TRAIN_STRIDE) if self.is_train else 1
        W      = self.condition_frames + self.forward_iter  # window size

        for sid in sequences:
            seq       = f"{sid:02d}"
            velo_dir  = os.path.join(self.sequences_path, seq, 'velodyne')
            pose_file = os.path.join(poses_path, f"{seq}.txt")

            if not (os.path.isdir(velo_dir) and os.path.isfile(pose_file)):
                print(f"  Warning: missing data for sequence {seq}, skipping.")
                continue

            poses = load_kitti_poses(pose_file)
            n_pc  = sum(1 for f in os.listdir(velo_dir) if f.endswith(self.pc_extension))
            n     = min(len(poses), n_pc)

            if n < W:
                print(f"  Warning: sequence {seq} has only {n} frames (need {W}), skipping.")
                continue

            for start in range(0, n - W + 1, stride):
                index.append((seq, start, poses[start:start + W]))

        return index

    # ── I/O and projection ────────────────────────────────────────────────────

    def _load_pc(self, seq, frame_idx):
        """Load a single KITTI velodyne scan."""
        path = os.path.join(
            self.sequences_path, seq, 'velodyne', f"{frame_idx:06d}{self.pc_extension}"
        )
        try:
            return np.fromfile(path, dtype=self.pc_dtype).reshape(self.pc_reshape)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def _project(self, pc):
        """Project a point cloud to a normalised range-view tensor.

        Returns [2, H, W] (range, intensity) by default, or [5, H, W]
        (range, x, y, z, intensity) when ``five_channel=True`` and
        ``log_range=False``.

        Normalisation modes:
          log_range=False  →  (feat - mean) / std  (mean/std z-normalisation)
          log_range=True   →  range: log2(r+1)/6 clipped to [0,1];
                               intensity: clipped to [0,1].
                               Empty pixels (depth=-1) map to 0 after clamping.

        Empty pixels carry a -1 sentinel from doProjection; after z-norm they
        land well below any valid reading so make_valid_mask can recover them.
        """
        proj_pc, proj_range, _, _ = self.projection.doProjection(pc)

        depth  = torch.from_numpy(proj_range)               # [H, W]
        intens = (torch.from_numpy(proj_pc[..., 3])
                  if pc.shape[1] >= 4 else torch.zeros_like(depth))

        if self.log_range:
            # Log-scale depth normalisation from LiDARGen:
            #   empty pixels (depth=-1) → clamp to 0 → log2(1)/6 = 0
            #   1 m  → log2(2)/6  ≈ 0.167
            #   10 m → log2(11)/6 ≈ 0.573
            #   80 m → log2(81)/6 ≈ 1.055 → clipped to 1.0
            depth_norm  = (torch.log2(depth.clamp(min=0.) + 1.) / 6.).clamp(0., 1.)
            intens_norm = intens.clamp(0., 1.)
            return torch.stack([depth_norm, intens_norm], dim=0)   # [2, H, W]

        if self.five_channel:
            xyz  = torch.from_numpy(proj_pc[..., :3]).permute(2, 0, 1)  # [3,H,W]
            feat = torch.cat([depth.unsqueeze(0), xyz, intens.unsqueeze(0)], 0)  # [5,H,W]
        else:
            feat = torch.stack([depth, intens], dim=0)              # [2, H, W]

        return (feat - self.mean[:feat.shape[0], None, None]) / self.std[:feat.shape[0], None, None]

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        seq, start, poses = self.index[idx]

        views = []
        for i in range(self.condition_frames + self.forward_iter):
            pc = self._load_pc(seq, start + i)
            if pc is None:
                return self.__getitem__(random.randint(0, len(self) - 1))
            if self.augmentor is not None:
                pc, _ = self.augmentor.doAugmentation(pc)
            views.append(self._project(pc))

        return (
            torch.stack(views),                            # [T, 2, H, W]
            torch.from_numpy(np.array(poses)).float(),     # [T, 4, 4]
        )


# ── Convenience subclasses ────────────────────────────────────────────────────

class KITTIRangeViewTrainDataset(KITTIRangeViewDataset):
    """Training split — stride-5 window sampling, augmentation enabled."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **dict(kwargs, is_train=True))


class KITTIRangeViewValDataset(KITTIRangeViewDataset):
    """Validation split — all windows, no augmentation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **dict(kwargs, is_train=False))


class KITTIRangeViewTestDataset(KITTIRangeViewDataset):
    """Test split — all windows, no augmentation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **dict(kwargs, is_train=False))


class KITTIRangeViewVAEDataset(KITTIRangeViewDataset):
    """
    VAE Dataset (Stage 1).
    Iterates over individual frames with stride=1 (dense sampling),
    returning single [C, H, W] images instead of sequences.
    Pass is_train=True for training (enables augmentation), is_train=False for eval.
    """
    def __init__(self, *args, **kwargs):
        # Force single-frame window; caller controls is_train / augmentation
        kwargs['condition_frames'] = 0
        kwargs['forward_iter'] = 1
        kwargs['stride'] = 1  # Visit every frame
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        # Parent returns sequence [T=1, C, H, W] and poses [T=1, 4, 4]
        views, poses = super().__getitem__(idx)
        # Return [C, H, W] and [4, 4]
        return views[0], poses[0]
