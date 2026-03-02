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

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Validity mask
# ---------------------------------------------------------------------------

def make_valid_mask(
    features: torch.Tensor,
    range_mean: float,
    range_std: float,
    min_range: float = 0.5,
) -> torch.Tensor:
    """Return a boolean mask for pixels that contain a real LiDAR return.

    During projection empty pixels are filled with range = -1.  After
    normalisation the range channel for an empty pixel is approximately
    ``(-1 - mean) / std``.  We recover validity by checking that the
    *unnormalised* range exceeds ``min_range`` metres.

    Args:
        features:   ``[..., L, C]`` normalised range-view features
                    (range channel at index 0).
        range_mean: per-channel mean for the range channel.
        range_std:  per-channel std  for the range channel.
        min_range:  minimum valid depth (metres); default 0.5 m.

    Returns:
        ``[..., L]`` bool tensor, True where the pixel is valid.
    """
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
    """Per-pixel L1 loss on normalised range-view features.

    Args:
        pred:       ``[..., C]`` predicted (normalised) range features.
        target:     ``[..., C]`` GT (normalised) range features.
        valid_mask: ``[...]`` bool mask.  When provided the loss is restricted
                    to GT-valid pixels so empty-pixel hallucinations do not
                    dominate.

    Returns:
        Scalar L1 loss.
    """
    loss = torch.abs(pred - target)          # [..., C]
    if valid_mask is not None:
        mask = valid_mask.unsqueeze(-1).float()   # [..., 1]
        loss = loss * mask
        return loss.sum() / (mask.sum() * pred.shape[-1] + 1e-8)
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
# Chamfer Distance
# ---------------------------------------------------------------------------

def _chamfer_single(pc1: torch.Tensor, pc2: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer Distance between two small point clouds.

    Args:
        pc1: ``[N, 3]``
        pc2: ``[M, 3]``

    Returns:
        Scalar: mean(min_dist pc1→pc2) + mean(min_dist pc2→pc1).
    """
    diff  = pc1.unsqueeze(1) - pc2.unsqueeze(0)   # [N, M, 3]
    dist2 = (diff ** 2).sum(dim=-1)               # [N, M]
    d1 = dist2.min(dim=1)[0].mean()               # each pc1 pt → nearest pc2
    d2 = dist2.min(dim=0)[0].mean()               # each pc2 pt → nearest pc1
    return d1 + d2


def batch_chamfer_distance(
    pred_xyz: torch.Tensor,
    gt_xyz: torch.Tensor,
    pred_valid: torch.Tensor,
    gt_valid: torch.Tensor,
    max_pts: int = 2048,
) -> torch.Tensor:
    """Batched Chamfer Distance with random subsampling for efficiency.

    A full range image has H*W ≈ 131 072 pixels; computing O(N²) distances
    over all of them is prohibitive.  We randomly subsample up to ``max_pts``
    valid points per cloud before calling the pairwise distance kernel.

    Args:
        pred_xyz:   ``[B, L, 3]`` predicted xyz (unnormalised, metres).
        gt_xyz:     ``[B, L, 3]`` GT xyz (unnormalised, metres).
        pred_valid: ``[B, L]``    bool mask for predicted valid pixels.
        gt_valid:   ``[B, L]``    bool mask for GT valid pixels.
        max_pts:    max points to sample per cloud.

    Returns:
        Mean Chamfer distance over valid batch elements (scalar).
    """
    B = pred_xyz.shape[0]
    total_cd  = pred_xyz.new_zeros(())
    valid_cnt = 0

    for b in range(B):
        pc1 = pred_xyz[b][pred_valid[b]]   # [N, 3]
        pc2 = gt_xyz[b][gt_valid[b]]       # [M, 3]

        if pc1.shape[0] < 2 or pc2.shape[0] < 2:
            continue

        if pc1.shape[0] > max_pts:
            idx = torch.randperm(pc1.shape[0], device=pc1.device)[:max_pts]
            pc1 = pc1[idx]
        if pc2.shape[0] > max_pts:
            idx = torch.randperm(pc2.shape[0], device=pc2.device)[:max_pts]
            pc2 = pc2[idx]

        total_cd  = total_cd + _chamfer_single(pc1, pc2)
        valid_cnt += 1

    if valid_cnt == 0:
        return total_cd          # zero; keeps computation graph intact
    return total_cd / valid_cnt
