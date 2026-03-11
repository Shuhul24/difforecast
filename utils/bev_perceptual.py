"""
BEV Perceptual Loss for LiDAR range-view forecasting.

Pipeline (inspired by RangeLDM ``bev_perceptual`` branch in
vae/sgm/modules/autoencoding/losses/__init__.py L267-275):

    depth map [B, L]
    → back-project to 3-D xyz  (RangeViewProjection)
    → bilinear splat to BEV occupancy grid  [B, 1, bev_h, bev_w]
    → repeat 3× to feed frozen VGG16
    → multi-scale feature-distance (LPIPS-style)

Unlike pixel-space L1, VGG16 feature distances penalise missing
structures (walls, objects) rather than blurry-but-globally-correct
depth values — exactly the signal needed when global context is present
but local geometry is wrong.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------------------------------------------------------------------------
# Bilinear splatting: 3-D points → 2-D BEV occupancy
# ---------------------------------------------------------------------------

def _splat_to_bev(
    xyz: torch.Tensor,      # [B, N, 3]  metric 3-D points
    valid: torch.Tensor,    # [B, N]     bool — True for real LiDAR returns
    bev_h: int,
    bev_w: int,
    x_range: float = 25.6,  # grid covers ±x_range in the forward (X) direction
    y_range: float = 25.6,  # grid covers ±y_range in the lateral (Y) direction
    min_weight: float = 1e-4,
) -> torch.Tensor:
    """Bilinear soft-splatting of point cloud into a BEV occupancy grid.

    Each point votes into its 4 surrounding BEV cells with bilinear weights.
    Inspired by RangeLDM's ``_splat_points_to_volumes`` (trilinear, 3-D);
    simplified to 2-D because the BEV is a top-down projection (Z ignored).

    Returns:
        ``[B, 1, bev_h, bev_w]`` log-normalised occupancy (``log(count + 1)``).
    """
    B, N, _ = xyz.shape
    device   = xyz.device
    in_dtype = xyz.dtype

    # All arithmetic in float32 to avoid bfloat16/float16 dtype mismatches
    # (xi0.float() would silently produce float32 while bev stays bfloat16).
    xyz_f = xyz.float()

    # Pixel coordinates in [0, grid-1]
    xi = ((xyz_f[:, :, 0] + x_range) / (2.0 * x_range)) * (bev_h - 1)  # forward → row
    yi = ((xyz_f[:, :, 1] + y_range) / (2.0 * y_range)) * (bev_w - 1)  # lateral → col

    # Integer lower corners (clamped so +1 stays in bounds)
    xi0 = xi.floor().long().clamp(0, bev_h - 2)
    yi0 = yi.floor().long().clamp(0, bev_w - 2)
    xi1 = xi0 + 1
    yi1 = yi0 + 1

    # Bilinear remainder weights (float32 throughout)
    rx = (xi - xi0.float()).clamp(0.0, 1.0)   # [B, N]
    ry = (yi - yi0.float()).clamp(0.0, 1.0)

    w00 = (1.0 - rx) * (1.0 - ry)
    w01 = (1.0 - rx) * ry
    w10 = rx         * (1.0 - ry)
    w11 = rx         * ry

    # Flat linear indices into [bev_h * bev_w]
    idx00 = (xi0 * bev_w + yi0).clamp(0, bev_h * bev_w - 1)
    idx01 = (xi0 * bev_w + yi1).clamp(0, bev_h * bev_w - 1)
    idx10 = (xi1 * bev_w + yi0).clamp(0, bev_h * bev_w - 1)
    idx11 = (xi1 * bev_w + yi1).clamp(0, bev_h * bev_w - 1)

    mask = valid.float()  # [B, N] — float32, matches w tensors

    bev = torch.zeros(B, bev_h * bev_w, device=device, dtype=torch.float32)
    for idx, w in [(idx00, w00), (idx01, w01), (idx10, w10), (idx11, w11)]:
        bev.scatter_add_(1, idx, w * mask)

    # log(count + 1) normalisation keeps the occupancy in a stable range
    bev = torch.log1p(bev)
    return bev.view(B, 1, bev_h, bev_w).to(in_dtype)


# ---------------------------------------------------------------------------
# Frozen VGG16 multi-scale feature extractor
# ---------------------------------------------------------------------------

class _VGG16Features(nn.Module):
    """Frozen VGG16 feature extractor at 4 intermediate scales.

    Slices match the standard LPIPS VGG variant:
        relu1_2  relu2_2  relu3_3  relu4_3
    All parameters are frozen — this module is purely a perceptual metric.
    """

    _SLICE_ENDS = [4, 9, 16, 23]   # indices into vgg.features

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        feats = vgg.features
        prev = 0
        self.slices = nn.ModuleList()
        for end in self._SLICE_ENDS:
            self.slices.append(feats[prev:end])
            prev = end
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> list:
        outs = []
        h = x
        for s in self.slices:
            h = s(h)
            outs.append(h)
        return outs   # list of 4 feature maps


def _normalize_feats(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalise along the channel dimension (per spatial location)."""
    return t / (t.norm(dim=1, keepdim=True).clamp(min=eps))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class BEVPerceptualLoss(nn.Module):
    """Perceptual loss on Bird's Eye View occupancy grids.

    Converts predicted and GT depth maps to BEV occupancy images,
    then computes a multi-scale VGG16 feature distance (LPIPS-style).

    Usage in model_forward::

        bev_loss = self.bev_perceptual_loss(
            pred_depth, gt_depth, pred_valid, gt_valid
        )   # scalar

    Args:
        projector:  :class:`~utils.range_losses.RangeViewProjection` instance
                    (precomputed ray directions; shared with Chamfer loss).
        bev_h:      BEV grid height in pixels (forward direction).
        bev_w:      BEV grid width  in pixels (lateral direction).
        x_range:    Half-extent of BEV in the forward direction (metres).
        y_range:    Half-extent of BEV in the lateral direction (metres).
    """

    # Minimum spatial size for VGG16 (needs several pooling stages)
    _VGG_MIN_SIZE = 32

    def __init__(
        self,
        projector,
        bev_h: int   = 256,
        bev_w: int   = 256,
        x_range: float = 25.6,
        y_range: float = 25.6,
    ):
        super().__init__()
        self.projector = projector
        self.bev_h     = bev_h
        self.bev_w     = bev_w
        self.x_range   = x_range
        self.y_range   = y_range
        self.vgg       = _VGG16Features()

        # ImageNet normalisation expected by VGG16
        self.register_buffer(
            'img_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'img_std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _depth_to_bev(
        self,
        depth: torch.Tensor,   # [B, L] metres
        valid: torch.Tensor,   # [B, L] bool
    ) -> torch.Tensor:
        """Back-project depth → 3-D xyz → BEV occupancy [B, 3, bev_h, bev_w]."""
        xyz = self.projector.depth_to_xyz(depth)   # [B, L, 3]
        bev = _splat_to_bev(
            xyz, valid, self.bev_h, self.bev_w, self.x_range, self.y_range
        )  # [B, 1, bev_h, bev_w]

        # Match VGG16 weight dtype (may be bfloat16 under DeepSpeed mixed precision)
        vgg_dtype = next(self.vgg.parameters()).dtype
        bev3 = bev.expand(-1, 3, -1, -1).to(vgg_dtype)

        # Ensure minimum spatial size
        if self.bev_h < self._VGG_MIN_SIZE or self.bev_w < self._VGG_MIN_SIZE:
            bev3 = F.interpolate(
                bev3,
                size=(max(self.bev_h, self._VGG_MIN_SIZE),
                      max(self.bev_w, self._VGG_MIN_SIZE)),
                mode='bilinear', align_corners=False,
            )

        return (bev3 - self.img_mean) / self.img_std

    def forward(
        self,
        pred_depth: torch.Tensor,   # [B, L] unnormalized depth in metres
        gt_depth:   torch.Tensor,   # [B, L] unnormalized depth in metres
        pred_valid: torch.Tensor,   # [B, L] bool
        gt_valid:   torch.Tensor,   # [B, L] bool
    ) -> torch.Tensor:
        """Return scalar BEV perceptual loss."""
        pred_bev = self._depth_to_bev(pred_depth, pred_valid)
        gt_bev   = self._depth_to_bev(gt_depth,   gt_valid)

        pred_feats = self.vgg(pred_bev)
        gt_feats   = self.vgg(gt_bev)

        loss = sum(
            (_normalize_feats(pf) - _normalize_feats(gf)).pow(2).mean()
            for pf, gf in zip(pred_feats, gt_feats)
        )
        return (loss / len(pred_feats)).to(pred_depth.dtype)
