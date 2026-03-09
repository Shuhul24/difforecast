"""
Self-contained RangeLDM VAE for range-view LiDAR forecasting.

Copies the minimal Encoder/Decoder building-blocks from
RangeLDM/vae/sgm/modules/diffusionmodules/model.py and
RangeLDM/vae/sgm/modules/distributions/distributions.py,
stripped of edge-conv, xformers, and pytorch-lightning dependencies.

Architecture (matches RangeLDM kitti360.yaml):
    Encoder:  2-ch input  → 2-stage downsampling (4×) → 2*z_channels latent
    Decoder:  z_channels latent → 2-stage upsampling (4×) → 2-ch output
    z_channels = 4,  ch = 64,  ch_mult = [1, 2, 4]
    circular = True  (azimuth dimension wraps around)

For a 64×2048 KITTI Odometry range image this gives a 16×512 latent map
(4 channels).  After patchify with patch_size=8 the token count is
  (16/8) × (512/8) = 2 × 64 = 128 tokens   (same as the old DC-AE path).
"""

import math
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


# ---------------------------------------------------------------------------
# Basic building blocks  (verbatim from RangeLDM model.py, vanilla path only)
# ---------------------------------------------------------------------------

def nonlinearity(x, kind='relu'):
    if kind == 'silu':
        return x * torch.sigmoid(x)
    elif kind == 'relu':
        return F.relu(x)
    raise NotImplementedError(f"nonlinearity '{kind}' not supported")


def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels,
                        eps=1e-6, affine=True)


class Conv2d(nn.Conv2d):
    """Conv2d with optional circular padding along the azimuth (W) dimension."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', device=None, dtype=None,
                 circular=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups,
                         bias=bias, padding_mode=padding_mode,
                         device=device, dtype=dtype)
        self.circular = circular

    def _conv_forward(self, input, weight, bias):
        if self.circular:
            # Circular padding along W (azimuth), zero padding along H (elevation)
            input = F.pad(input, (0, 0, self.padding[0], self.padding[0]),
                          mode="circular")
            input = F.pad(input, (self.padding[1], self.padding[1], 0, 0),
                          mode="constant")
            return F.conv2d(input, weight, bias, self.stride, _pair(0),
                            self.dilation, self.groups)
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice,
                                  mode=self.padding_mode),
                            weight, bias, self.stride, _pair(0),
                            self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride, self.padding,
                        self.dilation, self.groups)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, up_single=False, circular=False, **kwargs):
        super().__init__()
        self.with_conv = with_conv
        self.up_single = up_single
        if self.with_conv:
            self.conv = Conv2d(in_channels, in_channels, kernel_size=3,
                               stride=1, padding=1, circular=circular)

    def forward(self, x):
        scale = (2.0, 1.0) if self.up_single else 2.0
        x = F.interpolate(x, scale_factor=scale, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, down_single=False, circular=False, **kwargs):
        super().__init__()
        self.with_conv = with_conv
        self.down_single = down_single
        self.circular = circular
        if self.with_conv:
            stride = (2, 1) if down_single else 2
            self.conv = Conv2d(in_channels, in_channels, kernel_size=3,
                               stride=stride, padding=0)

    def forward(self, x):
        if self.with_conv:
            if not self.circular:
                pad = (0, 1, 0, 1) if not self.down_single else (1, 1, 0, 1)
                x = F.pad(x, pad, mode="constant", value=0)
            else:
                x = F.pad(x, (0, 0, 0, 1), mode="circular")
                x = F.pad(x, (1 if self.down_single else 0, 1, 0, 0),
                           mode="constant")
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512, act='relu', circular=False, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.nonlinearity = partial(nonlinearity, kind=act)

        self.norm1 = Normalize(in_channels)
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3,
                            stride=1, padding=1, circular=circular)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3,
                            stride=1, padding=1, circular=circular)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = Conv2d(in_channels, out_channels,
                                            kernel_size=3, stride=1, padding=1,
                                            circular=circular)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels,
                                              kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb=None):
        h = self.norm1(x)
        h = self.nonlinearity(h)
        h = self.conv1(h)
        if temb is not None and hasattr(self, 'temb_proj'):
            h = h + self.temb_proj(self.nonlinearity(temb))[:, :, None, None]
        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


def make_attn(in_channels, attn_type="none", attn_kwargs=None):
    """Return an attention block. Only 'none' is supported here."""
    assert attn_type == "none", (
        f"Only attn_type='none' is supported in rangeldm_vae; got '{attn_type}'"
    )
    return nn.Identity()


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 resamp_single_side=False, in_channels, resolution, z_channels,
                 double_z=True, use_linear_attn=False, attn_type="none",
                 act='relu', circular=False, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.nonlinearity = partial(nonlinearity, kind=act)

        # Input projection
        self.conv_in = Conv2d(in_channels, ch, kernel_size=3, stride=1,
                              padding=1, circular=circular)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult

        # Downsampling stages
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(
                    in_channels=block_in, out_channels=block_out,
                    temb_channels=self.temb_ch, dropout=dropout,
                    act=act, circular=circular,
                ))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv,
                                             resamp_single_side, circular=circular)
                curr_res = curr_res // 2
            self.down.append(down)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                        temb_channels=self.temb_ch, dropout=dropout,
                                        act=act, circular=circular)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                        temb_channels=self.temb_ch, dropout=dropout,
                                        act=act, circular=circular)

        # Output projection
        self.norm_out = Normalize(block_in)
        self.conv_out = Conv2d(block_in,
                               2 * z_channels if double_z else z_channels,
                               kernel_size=3, stride=1, padding=1, circular=circular)

    def forward(self, x):
        temb = None
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        h = self.norm_out(h)
        h = self.nonlinearity(h)
        h = self.conv_out(h)
        return h


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 resamp_single_side=False, in_channels, resolution, z_channels,
                 give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="none", act='relu', circular=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.nonlinearity = partial(nonlinearity, kind=act)

        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # Latent → block_in
        self.conv_in = Conv2d(z_channels, block_in, kernel_size=3, stride=1,
                              padding=1, circular=circular)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                        temb_channels=self.temb_ch, dropout=dropout,
                                        act=act, circular=circular)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                        temb_channels=self.temb_ch, dropout=dropout,
                                        act=act, circular=circular)

        # Upsampling stages
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                block.append(ResnetBlock(
                    in_channels=block_in, out_channels=block_out,
                    temb_channels=self.temb_ch, dropout=dropout,
                    act=act, circular=circular,
                ))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv,
                                       resamp_single_side, circular=circular)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        # Output projection
        self.norm_out = Normalize(block_in)
        self.conv_out = Conv2d(block_in, out_ch, kernel_size=3, stride=1,
                               padding=1, circular=circular)

    def get_last_layer(self):
        return self.conv_out.weight

    def forward(self, z):
        temb = None
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = self.nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


# ---------------------------------------------------------------------------
# Diagonal Gaussian posterior  (from distributions.py)
# ---------------------------------------------------------------------------

class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self):
        return self.mean

    def kl(self, other=None):
        if self.deterministic:
            return torch.tensor(0.0)
        if other is None:
            return 0.5 * torch.sum(
                self.mean.pow(2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3]
            )
        return 0.5 * torch.sum(
            (self.mean - other.mean).pow(2) / other.var
            + self.var / other.var - 1.0 - self.logvar + other.logvar,
            dim=[1, 2, 3],
        )


# ---------------------------------------------------------------------------
# RangeLDMVAE — thin wrapper used by the tokenizer
# ---------------------------------------------------------------------------

class RangeLDMVAE(nn.Module):
    """Encoder + Decoder wrapper matching the RangeLDM kitti360 config.

    Architecture (default = kitti360.yaml):
        in_channels = 2   (range, intensity — normalised)
        out_ch      = 2
        ch          = 64
        ch_mult     = [1, 2, 4]   → 4× spatial compression (2 downsamplings)
        z_channels  = 4
        circular    = True
        act         = 'silu'

    For KITTI Odometry 64×2048:
        latent shape: [B, 4, 16, 512]

    Checkpoint loading:
        Expects the RangeLDM training checkpoint (.ckpt) whose ``state_dict``
        contains keys ``encoder.*`` and ``decoder.*``.
        Only the encoder and decoder weights are loaded; loss/discriminator
        weights in the checkpoint are ignored.
    """

    _DDCONFIG = dict(
        ch=64, out_ch=2, ch_mult=(1, 2, 4), num_res_blocks=2,
        attn_resolutions=[], dropout=0.0, resamp_with_conv=True,
        in_channels=2, resolution=256, z_channels=4,
        double_z=True, attn_type="none", act='silu', circular=True,
    )

    def __init__(self, ckpt_path: Optional[str] = None):
        super().__init__()
        cfg = self._DDCONFIG
        self.encoder = Encoder(**cfg)
        self.decoder = Decoder(**cfg)
        self.z_channels = cfg['z_channels']

        if ckpt_path is not None:
            self._load_ckpt(ckpt_path)

    def _load_ckpt(self, path: str):
        sd = torch.load(path, map_location='cpu')
        # Support both raw state_dict and Lightning checkpoints
        sd = sd.get('state_dict', sd)
        enc_sd = {k[len('encoder.'):]: v
                  for k, v in sd.items() if k.startswith('encoder.')}
        dec_sd = {k[len('decoder.'):]: v
                  for k, v in sd.items() if k.startswith('decoder.')}
        missing_e, unexp_e = self.encoder.load_state_dict(enc_sd, strict=False)
        missing_d, unexp_d = self.decoder.load_state_dict(dec_sd, strict=False)
        print(f"RangeLDMVAE loaded from {path}  "
              f"(enc missing={len(missing_e)}, dec missing={len(missing_d)})")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a normalised 2-channel range image to a latent sample.

        Uses the posterior *mode* (mean) for deterministic, lower-variance
        latents during DiT training.  Gradients are not expected here since
        the VAE is frozen.

        Args:
            x: ``[B, 2, H, W]`` normalised [range, intensity] range image.

        Returns:
            ``[B, z_channels, H//4, W//4]`` latent tensor.
        """
        h = self.encoder(x)                        # [B, 2*z_ch, H', W']
        return DiagonalGaussianDistribution(h).mode()  # [B, z_ch, H', W']

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent tensor back to a 2-channel range image.

        Args:
            z: ``[B, z_channels, H', W']`` latent tensor.

        Returns:
            ``[B, 2, H, W]`` reconstructed [range, intensity] range image.
        """
        return self.decoder(z)
