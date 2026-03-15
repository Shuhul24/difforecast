"""
Range-image discriminator for Stage 1 VAE adversarial training.

Ported from RangeLDM's NLayerDiscriminatorMetaKernel:
  MetaKernel conditions per-pixel convolution weights on 3-D angular
  coordinates (azimuth, inclination) derived from the local depth
  neighbourhood, giving the discriminator geometric awareness of the
  LiDAR range image.

Used for the generator / discriminator adversarial update in Stage 1
VAE pre-training.  The discriminator is activated only after
``disc_start`` steps so that the ELBO reconstruction loss converges
first (following the RangeLDM training recipe).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Weight init
# ---------------------------------------------------------------------------

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ---------------------------------------------------------------------------
# GAN losses (hinge and vanilla)
# ---------------------------------------------------------------------------

def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Hinge GAN discriminator loss (matches RangeLDM)."""
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Vanilla (sigmoid BCE) GAN discriminator loss."""
    return 0.5 * (
        torch.mean(F.softplus(-logits_real))
        + torch.mean(F.softplus(logits_fake))
    )


# ---------------------------------------------------------------------------
# Scheduling helpers
# ---------------------------------------------------------------------------

def adopt_weight(weight: float, global_step: int, threshold: int = 0,
                 value: float = 0.0) -> float:
    """Return ``value`` before ``threshold``, ``weight`` after."""
    return value if global_step < threshold else weight


def calculate_adaptive_weight(
    nll_loss: torch.Tensor,
    g_loss: torch.Tensor,
    last_layer: torch.Tensor,
    disc_weight: float,
) -> torch.Tensor:
    """Adaptive discriminator weight (Esser et al., VQGAN).

    Balances the NLL reconstruction gradient against the GAN generator
    gradient at the last decoder layer.  Falls back to a fixed
    ``disc_weight`` on ``RuntimeError`` (e.g. during eval / no-grad).
    """
    try:
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads   = torch.autograd.grad(g_loss,   last_layer, retain_graph=True)[0]
        d_weight  = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight  = torch.clamp(d_weight, 0.0, 1e4).detach() * disc_weight
    except RuntimeError:
        d_weight  = torch.tensor(disc_weight, device=nll_loss.device,
                                 dtype=nll_loss.dtype)
    return d_weight


# ---------------------------------------------------------------------------
# MetaKernel geometry-aware convolution
# ---------------------------------------------------------------------------

class MetaKernel(nn.Module):
    """Geometry-aware convolution for LiDAR range images.

    Conditions per-pixel convolution weights on 3-D angular coordinate
    offsets (azimuth Δφ, inclination Δθ) relative to the centre of each
    receptive field, computed from the local depth neighbourhood.

    Ported faithfully from RangeLDM
    (``sgm/modules/autoencoding/lpips/model/model.py``).

    Args:
        in_channels:  Input feature channels.
        out_channels: Output feature channels.
        azi:          Azimuth angular step per pixel (radians).
        inc:          Inclination angular step per pixel (radians).
        kernel_size:  Convolution kernel size (default 4, matching PatchGAN).
        stride:       Convolution stride (default 2).
        padding:      Convolution padding (default 1).
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        azi:          float,
        inc:          float,
        kernel_size:  int = 4,
        stride:       int = 2,
        padding:      int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride      = stride
        self.padding     = padding

        self.mlp_coord = nn.Sequential(
            nn.Linear(3, in_channels),
            nn.LeakyReLU(0.2, True),
            nn.Linear(in_channels, in_channels),
        )
        self.coov = nn.Conv2d(kernel_size * kernel_size * in_channels,
                              out_channels, 1, 1, 0)

        # Pre-compute angular positional-encoding buffers (sin / cos of
        # angular offsets within the receptive field).
        ks = kernel_size
        cos_azi = torch.zeros(ks, ks)
        sin_azi = torch.zeros(ks, ks)
        cos_inc = torch.zeros(ks, ks)
        sin_inc = torch.zeros(ks, ks)
        for sh in range(ks):
            for sw in range(ks):
                dw = float(sw - ks // 2)
                dh = float(sh - ks // 2)
                cos_azi[sh, sw] = torch.cos(torch.tensor(azi * dw))
                sin_azi[sh, sw] = torch.sin(torch.tensor(azi * dw))
                cos_inc[sh, sw] = torch.cos(torch.tensor(inc * dh))
                sin_inc[sh, sw] = torch.sin(torch.tensor(inc * dh))

        shape = (1, 1, 1, 1, ks, ks)
        self.register_buffer('cos_azi', cos_azi.reshape(shape))
        self.register_buffer('sin_azi', sin_azi.reshape(shape))
        self.register_buffer('cos_inc', cos_inc.reshape(shape))
        self.register_buffer('sin_inc', sin_inc.reshape(shape))

    def forward(self, x: torch.Tensor, r: torch.Tensor):
        """
        Args:
            x: Feature map  ``[B, C, H, W]``
            r: Range map    ``[B, 1, H, W]`` (in metres, unnormalised)

        Returns:
            Tuple ``(output [B, out_C, H', W'], r_center [B, 1, H', W'])``
        """
        B, C, H, W = x.shape
        ks, st, pd = self.kernel_size, self.stride, self.padding

        # Pad range map: azimuth (W-dim) with out-of-range value; elevation
        # (H-dim) with wrap-around (same as RangeLDM original implementation).
        r = F.pad(r, (pd, pd, 0,  0),  value=100.)
        r = F.pad(r, (0,   0, pd, pd), mode='circular')
        # Extract patches: [B, 1, H', W', ks, ks]
        r_patches = r.unfold(3, ks, st).unfold(2, ks, st)
        r_center  = r_patches[:, :, :, :, ks // 2, ks // 2]   # [B,1,H',W']

        # Angular positional encoding (3 components)
        pe0 = r_patches * self.cos_azi * self.cos_inc - r_center.unsqueeze(4).unsqueeze(4)
        pe1 = r_patches * self.cos_azi * self.sin_inc
        pe2 = r_patches * self.sin_azi
        pe  = torch.cat([pe0, pe1, pe2], dim=1).permute(0, 2, 3, 4, 5, 1)
        # [B, H', W', ks, ks, 3] → MLP → [B, H', W', ks, ks, C]
        weights = self.mlp_coord(pe.float()).to(x.dtype)
        weights = weights.permute(0, 5, 1, 2, 3, 4)   # [B, C, H', W', ks, ks]

        # Pad feature map the same way as r, then extract patches
        x = F.pad(x, (pd, pd, 0,  0))
        x = F.pad(x, (0,   0, pd, pd), mode='circular')
        x_patches = x.unfold(3, ks, st).unfold(2, ks, st)    # [B,C,H',W',ks,ks]

        # Weight-and-pool: geometry-modulated convolution
        x_patches = weights * x_patches
        H_out, W_out = x_patches.shape[2], x_patches.shape[3]
        x_patches = (x_patches
                     .permute(0, 1, 4, 5, 2, 3)
                     .reshape(B, C * ks * ks, H_out, W_out))
        output = self.coov(x_patches)
        return output, r_center


class MetaKernelSequential(nn.Sequential):
    """``nn.Sequential`` sub-class that threads ``(x, r)`` through MetaKernel layers."""

    def forward(self, x: torch.Tensor, r: torch.Tensor):
        for layer in self:
            if isinstance(layer, MetaKernel):
                x, r = layer(x, r)
            else:
                x = layer(x)
        return x, r


# ---------------------------------------------------------------------------
# PatchGAN discriminator with MetaKernel blocks
# ---------------------------------------------------------------------------

class NLayerDiscriminatorMetaKernel(nn.Module):
    """PatchGAN discriminator using geometry-aware MetaKernel convolutions.

    Operates directly on 2-channel ``[range, intensity]`` range images.
    Azimuth-circular padding is handled inside each MetaKernel block.
    The discriminator is architecture-identical to the one used in
    RangeLDM (kitti360.yaml, ``metakernel: True``).

    Args:
        input_nc:   Input channels (2 for [range, intensity]).
        ndf:        Base filter count (64 in RangeLDM).
        n_layers:   Number of strided conv layers (3 in RangeLDM).
        azi:        Azimuth angular step per pixel (radians).
                    KITTI HDL-64E 360°/2048 ≈ 0.00307 rad; RangeLDM
                    uses 2× that (0.00613592) — kept for compatibility.
        inc:        Inclination angular step per pixel (radians).
                    KITTI HDL-64E 28°/64 ≈ 0.00763 rad (0.0074594 in RangeLDM).
        range_mean: Mean depth (metres) for unnormalising channel 0.
        range_std:  Std-dev depth for unnormalising channel 0.
    """

    def __init__(
        self,
        input_nc:   int   = 2,
        ndf:        int   = 64,
        n_layers:   int   = 3,
        azi:        float = 0.00613592,
        inc:        float = 0.0074594,
        range_mean: float = 10.839,
        range_std:  float = 9.314,
    ):
        super().__init__()
        self.range_mean = range_mean
        self.range_std  = range_std

        norm_layer = nn.BatchNorm2d
        kw, padw = 4, 1

        # First layer: stride-2 MetaKernel, no norm
        sequence = [
            MetaKernel(input_nc, ndf, azi=azi, inc=inc,
                       kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        azi *= 2; inc *= 2

        # Intermediate stride-2 layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                MetaKernel(ndf * nf_mult_prev, ndf * nf_mult, azi=azi, inc=inc,
                           kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
            azi *= 2; inc *= 2

        # Penultimate stride-1 layer
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            MetaKernel(ndf * nf_mult_prev, ndf * nf_mult, azi=azi, inc=inc,
                       kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        # Final stride-1 layer → 1-channel patch map
        sequence += [
            MetaKernel(ndf * nf_mult, 1, azi=azi, inc=inc,
                       kernel_size=kw, stride=1, padding=padw),
        ]
        self.main = MetaKernelSequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[B, 2, H, W]`` normalised [range, intensity] range image.

        Returns:
            ``[B, 1, H', W']`` patch-wise real/fake logits.
        """
        # Unnormalise range channel → metres for angular encoding, then scale to O(1)
        r = (x[:, :1] * self.range_std + self.range_mean) / 10.0
        out, _ = self.main(x, r)
        return out
