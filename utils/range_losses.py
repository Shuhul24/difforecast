"""
Auxiliary losses for range-view LiDAR forecasting.

Two complementary signals on top of the flow-matching (diffusion) loss:

  1. Range-View L1 Loss
     Per-pixel L1 between the denoised *prediction* and the GT range-view
     image (both in normalised feature space).  Gives direct image-space
     reconstruction supervision that the diffusion loss alone does not provide.

  2. Chamfer Distance
     3-D geometric loss between the point clouds recovered from the predicted
     and GT range-view images.  The 6-channel features store (range, x, y, z,
     intensity, label); unnormalising channels 1-3 gives the 3-D coordinates
     directly.  Chamfer distance ensures that the back-projected point cloud
     matches the future-frame ground truth – the same quantity visualised at
     inference time.
"""

import math
import os
import sys
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# pyTorchChamferDistance — lazy import (CUDA extension compiled on first use)
# ---------------------------------------------------------------------------
# Inspired by ATPPNet (https://github.com/kaustabpal/ATPPNet) which uses this
# library as a git submodule for efficient Chamfer distance computation.
# The submodule has no setup.py; it is used directly from the repo root via
# sys.path.  Setup: git submodule update --init  (no pip install needed).
# ---------------------------------------------------------------------------

# Repo root = two levels up from utils/range_losses.py
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_chamfer_fn = None
_CHAMFER_AVAILABLE: bool | None = None   # None = not yet attempted


def _get_chamfer_fn():
    """Return a lazily-initialised ChamferDistance instance (singleton)."""
    global _chamfer_fn, _CHAMFER_AVAILABLE
    if _CHAMFER_AVAILABLE is None:
        # Make the submodule importable by ensuring the repo root is on sys.path
        if _REPO_ROOT not in sys.path:
            sys.path.insert(0, _REPO_ROOT)
        try:
            from pyTorchChamferDistance.chamfer_distance import ChamferDistance
            _chamfer_fn        = ChamferDistance()
            _CHAMFER_AVAILABLE = True
        except Exception as e:
            _CHAMFER_AVAILABLE = False
            _get_chamfer_fn._last_err = str(e)
    if not _CHAMFER_AVAILABLE:
        err = getattr(_get_chamfer_fn, '_last_err', '')
        raise RuntimeError(
            "pyTorchChamferDistance failed to load"
            + (f" ({err})" if err else "") + ".\n"
            "Run:  git submodule update --init\n"
            "Then ensure nvcc is available for CUDA JIT compilation."
        )
    return _chamfer_fn


# ---------------------------------------------------------------------------
# Range-view → 3-D back-projection
# ---------------------------------------------------------------------------

class RangeViewProjection:
    """Precomputes per-pixel spherical ray directions for depth → xyz back-projection.

    Inspired by ATPPNet's projection utility:
    https://github.com/kaustabpal/ATPPNet/blob/main/atppnet/utils/projection.py

    Stores (x_fac, y_fac, z_fac) unit-vector factors so that
    ``xyz = depth * (x_fac, y_fac, z_fac)`` gives metric 3-D coordinates
    directly — no repeated trigonometry at training time.
    """

    def __init__(self, fov_up: float, fov_down: float, H: int, W: int):
        """
        Args:
            fov_up:   Upper vertical field of view (degrees, positive above horizon).
            fov_down: Lower vertical field of view (degrees, negative below horizon).
            H, W:     Range image height and width in pixels.
        """
        fov_up_r   = fov_up   / 180.0 * math.pi
        fov_down_r = fov_down / 180.0 * math.pi
        fov        = abs(fov_down_r) + abs(fov_up_r)

        h = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
        w = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)

        yaw   = math.pi * (1.0 - 2.0 * w / W)
        pitch = (1.0 - h / H) * fov - abs(fov_down_r)

        cos_pitch = torch.cos(pitch)
        # Flatten to [H*W] so we can broadcast over the batch dimension.
        self.x_fac = (cos_pitch * torch.cos(yaw)).reshape(-1)   # [L]
        self.y_fac = (cos_pitch * torch.sin(yaw)).reshape(-1)   # [L]
        self.z_fac = torch.sin(pitch).reshape(-1)               # [L]

    def depth_to_xyz(self, depth: torch.Tensor) -> torch.Tensor:
        """Back-project a flat depth map to 3-D Cartesian coordinates.

        Args:
            depth: ``[B, L]`` depth values in metres (0 = invalid).

        Returns:
            ``[B, L, 3]`` xyz coordinates in metres.
        """
        dev = depth.device
        dt  = depth.dtype
        x = depth * self.x_fac.to(device=dev, dtype=dt)
        y = depth * self.y_fac.to(device=dev, dtype=dt)
        z = depth * self.z_fac.to(device=dev, dtype=dt)
        return torch.stack([x, y, z], dim=-1)   # [B, L, 3]


# ---------------------------------------------------------------------------
# Validity mask
# ---------------------------------------------------------------------------

def make_valid_mask(
    features: torch.Tensor,
    range_mean: float = 0.,
    range_std: float = 1.,
    min_range: float = 0.5,
    log_range: bool = False,
) -> torch.Tensor:
    """Return a boolean mask for pixels that contain a real LiDAR return.

    Supports two normalisation modes:

    log_range=False (mean/std):
        During projection empty pixels are filled with range = -1.  After
        normalisation the range channel for an empty pixel is approximately
        ``(-1 - mean) / std``.  We recover validity by checking that the
        *unnormalised* range exceeds ``min_range`` metres.

    log_range=True (log2 normalisation):
        Empty pixels clamp to 0 before log → normalised value = 0.
        Valid pixels satisfy ``2^(feat*6) - 1 > min_range``.

    Args:
        features:   ``[..., L, C]`` normalised range-view features
                    (range channel at index 0).
        range_mean: per-channel mean for the range channel (mean/std mode only).
        range_std:  per-channel std  for the range channel (mean/std mode only).
        min_range:  minimum valid depth (metres); default 0.5 m.
        log_range:  if True, use log2 inverse; otherwise use mean/std inverse.

    Returns:
        ``[..., L]`` bool tensor, True where the pixel is valid.
    """
    if log_range:
        range_unnorm = torch.exp2(features[..., 0] * 6.) - 1.
    else:
        range_unnorm = features[..., 0] * range_std + range_mean
    return range_unnorm > min_range


# ---------------------------------------------------------------------------
# Range-View L1 loss
# ---------------------------------------------------------------------------

def range_view_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-pixel L1 loss on the depth (range) channel only.

    Evaluating only channel 0 (range/depth) is semantically correct for a
    "range view loss" — intensity and z have different physical scales and
    meanings, so including them would dilute the depth supervision signal.

    Args:
        pred:       ``[..., C]`` predicted (normalised) range features.
        target:     ``[..., C]`` GT (normalised) range features.
        valid_mask: ``[...]`` bool mask.  When provided the loss is restricted
                    to GT-valid pixels so empty-pixel hallucinations do not
                    dominate.

    Returns:
        Scalar L1 loss on the range channel.
    """
    loss = torch.abs(pred[..., 0] - target[..., 0])   # [...] — depth channel only
    if valid_mask is not None:
        mask = valid_mask.float()
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)
    return loss.mean()


# ---------------------------------------------------------------------------
# 3-D coordinate extraction from feature tensor
# ---------------------------------------------------------------------------

def features_to_xyz(
    features: torch.Tensor,
    img_mean: list,
    img_std: list,
) -> torch.Tensor:
    """Unnormalise channels 1-3 (x, y, z) from range-view features.

    The projected range image has channels [range, x, y, z, intensity, label].
    Channels 1–3 hold the raw Cartesian coordinates of the LiDAR hit,
    stored during ``RangeProjection.doProjection``.  Unnormalising them gives
    metric 3-D positions directly — no angular back-projection needed.

    Args:
        features: ``[..., L, C]`` normalised range features.
        img_mean: per-channel means  (length C).
        img_std:  per-channel stds   (length C).

    Returns:
        ``[..., L, 3]`` unnormalised xyz coordinates (metres).
    """
    mean = torch.tensor(img_mean, device=features.device, dtype=features.dtype)
    std  = torch.tensor(img_std,  device=features.device, dtype=features.dtype)
    xyz_norm = features[..., 1:4]          # [..., L, 3]
    return xyz_norm * std[1:4] + mean[1:4]


# ---------------------------------------------------------------------------
# Chamfer Distance Loss (CUDA-accelerated via pyTorchChamferDistance)
# ---------------------------------------------------------------------------

def batch_chamfer_distance(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    pred_valid: torch.Tensor,
    gt_valid: torch.Tensor,
    projector: RangeViewProjection,
    max_pts: int = 2048,
    blank_penalty: float = 10.0,
    min_pred_fraction: float = 0.05,
) -> torch.Tensor:
    """Batched Chamfer Distance using the CUDA-accelerated pyTorchChamferDistance kernel.

    Inspired by the ``chamfer_distance`` loss class in ATPPNet:
    https://github.com/kaustabpal/ATPPNet/blob/main/atppnet/models/loss.py

    Depth maps are back-projected to 3-D via ``projector`` before computing
    the pairwise nearest-neighbour distances.  Random subsampling keeps the
    O(N²) distance matrix tractable for full-resolution range images.

    Blank-prediction penalty: when the predicted cloud has fewer than
    ``min_pred_fraction`` of the GT cloud's valid points, the standard
    Chamfer kernel is skipped (it would silently return zero for an empty
    cloud) and a fixed ``blank_penalty`` is added instead, scaled by the
    fraction of GT points that are missing.  This provides a strong gradient
    signal for the case where the model predicts a uniform / near-zero depth
    map that contains no valid LiDAR returns.

    Args:
        pred_depth:          ``[B, L]`` predicted depth in metres (0 = invalid).
        gt_depth:            ``[B, L]`` GT depth in metres (0 = invalid).
        pred_valid:          ``[B, L]`` bool mask — True where predicted pixel is valid.
        gt_valid:            ``[B, L]`` bool mask — True where GT pixel is valid.
        projector:           :class:`RangeViewProjection` for depth → xyz back-projection.
        max_pts:             Maximum points sampled per cloud (reduces O(N²) cost).
        blank_penalty:       Loss magnitude applied when prediction is too sparse.
        min_pred_fraction:   Minimum ratio of pred-valid / gt-valid points below
                             which the blank penalty is applied instead of Chamfer.

    Returns:
        Mean symmetric Chamfer distance (+ blank penalties) over the batch (scalar).
    """
    chamfer_fn = _get_chamfer_fn()

    # Back-project depth maps to 3-D coordinates [B, L, 3]
    pred_xyz = projector.depth_to_xyz(pred_depth)
    gt_xyz   = projector.depth_to_xyz(gt_depth)

    B        = pred_xyz.shape[0]
    # Always accumulate in float32: the CUDA kernel returns float32 tensors
    # regardless of the input dtype (e.g. bfloat16 under autocast).
    total_cd = torch.zeros((), device=pred_depth.device, dtype=torch.float32)

    for b in range(B):
        n_gt   = int(gt_valid[b].sum().item())
        n_pred = int(pred_valid[b].sum().item())

        # ------------------------------------------------------------------ #
        # Blank-prediction penalty
        # When the predicted cloud is nearly empty (< min_pred_fraction of GT
        # points), the Chamfer kernel would silently return 0. Instead we
        # apply a penalty proportional to the missing-point fraction so that
        # blank/uniform predictions always receive a non-zero gradient.
        # ------------------------------------------------------------------ #
        if n_gt < 2:
            # Skip: GT cloud itself is degenerate (edge case).
            continue

        if n_pred < max(2, int(min_pred_fraction * n_gt)):
            missing_fraction = 1.0 - n_pred / n_gt
            total_cd += blank_penalty * missing_fraction
            continue

        pc1 = pred_xyz[b][pred_valid[b]]   # [N, 3]
        pc2 = gt_xyz[b][gt_valid[b]]       # [M, 3]

        if pc1.shape[0] > max_pts:
            idx = torch.randperm(pc1.shape[0], device=pc1.device)[:max_pts]
            pc1 = pc1[idx]
        if pc2.shape[0] > max_pts:
            idx = torch.randperm(pc2.shape[0], device=pc2.device)[:max_pts]
            pc2 = pc2[idx]

        # ChamferDistance expects [1, N, 3]; returns squared distances [1, N] and [1, M]
        dist1, dist2 = chamfer_fn(pc1.unsqueeze(0).float(), pc2.unsqueeze(0).float())
        total_cd += torch.mean(dist1) + torch.mean(dist2)

    return (total_cd / B).to(pred_depth.dtype)
