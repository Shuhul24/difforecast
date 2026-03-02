import torch
import torch.nn as nn
from einops import rearrange
from models.modules.dcae import DCAE, dc_ae_f32c32, dc_ae_f32c32_rangeview

def poses_to_indices(poses, x_divisions=128, y_divisions=128):
    x_min = 0
    x_range = 8
    y_range = 1
    y_step = y_range / y_divisions
    y_min = -0.5 + y_step / 2
    x, y = poses[..., 0], poses[..., 1]        
    idx_x = (x * x_divisions / x_range).clip(0, x_divisions).to(torch.float32).unsqueeze(dim=2) ##normalize [0, 1] then multiple x_divisions or y_divisions
    idx_y = ((y-y_min) * y_divisions / y_range).clip(0, y_divisions).to(torch.float32).unsqueeze(dim=2)
    return torch.cat([idx_x, idx_y], dim=2)

def indices_to_pose(idx_x, idx_y, x_division=128, y_divisions=128):
    x_min = 0
    x_range = 8
    y_range = 1
    y_step = y_range / y_divisions
    y_min = -0.5 + y_step / 2
    x_step = x_range / x_division

    x = idx_x * x_step + x_step / 2
    y = idx_y * y_step + y_min + y_step / 2
    return x, y

def yaws_to_indices(yaws, division=512):
    yaw_range = 16
    yaw_step = yaw_range / division
    yaw_min = - yaw_range / 2.0 + yaw_step / 2.0
    idx_yaw = ((yaws - yaw_min) * division / yaw_range).clip(0, division).to(torch.float32)
    return idx_yaw

def indices_to_yaws(idx_yaw, division=512):
    yaw_range = 16
    yaw_step = yaw_range / division
    yaw_min = - yaw_range / 2.0 + yaw_step / 2.0
    yaw = idx_yaw * yaw_step + yaw_min + yaw_step / 2.0
    return yaw

def patchify(x, patch_size):
    bsz, c, h, w = x.shape
    p = patch_size
    h_, w_ = h // p, w // p

    x = x.reshape(bsz, c, h_, p, w_, p)
    x = torch.einsum('nchpwq->nhwcpq', x)
    x = x.reshape(bsz, h_ * w_, c * p ** 2)
    return x  # [b, L, c]

def unpatchify(x, patch_size, vae_embed_dim):
    bsz, h_, w_, _ = x.shape
    p = patch_size
    c = vae_embed_dim

    x = x.reshape(bsz, h_, w_, c, p, p)
    x = torch.einsum('nhwcpq->nchpwq', x)
    x = x.reshape(bsz, c, h_ * p, w_ * p)
    return x

class VAETokenizer(nn.Module):
    def __init__(self, args, local_rank):
        super().__init__()
        self.args = args
        self.vae = DCAE(dc_ae_f32c32("dc-ae-f32c32-mix-1.0", 
                                     pretrained_path=args.vae_ckpt, 
                                     add_encoder_temporal=args.add_encoder_temporal, 
                                     add_decoder_temporal=args.add_decoder_temporal, 
                                     condition_frames=args.temporal_patch_size, 
                                     token_size=(args.image_size[0]*args.image_size[1])//((args.downsample_size*args.patch_size)**2)
                                     )
                        )
        self.vae.cuda(local_rank)
        self.vae.eval()
        print(f"load from {args.vae_ckpt}")

    @torch.no_grad()
    def encode_to_z(self, x):
        b, t, _, _, _ = x.shape
        ts = rearrange(x, 'b t c h w -> (b t) c h w')
        with torch.no_grad():
            latents = self.vae.encode(ts)
        latents = patchify(latents, self.args.patch_size)
        latents = rearrange(latents, '(b t) L c -> b t L c', b=b, t=t)
        return latents

    @torch.no_grad()
    def z_to_image(self, x, is_video=False):
        x = unpatchify(x, self.args.patch_size, self.args.vae_embed_dim)
        with torch.no_grad():
            images = self.vae.decode(x, is_video)
        images = images / 2 + 0.5
        return images.clip(0, 1)    
    
    def poses_to_indices(self, poses):
        # poses: b, F, 2
        x, y = poses[:, :, 0], poses[:, :, 1]
        internal_x, internal_y = torch.floor(x * 16).clip(0, 127), torch.floor((y + 1) * 16).clip(0, 31)
        indices = internal_x * self.latitute_bins + internal_y
        return indices.to(torch.long).unsqueeze(dim=2)


class RangeViewVAETokenizer(nn.Module):
    """DCAE-based tokenizer for multi-channel range view images.

    Mirrors the encoding pattern used by ``VAETokenizer`` for RGB images but
    accepts ``in_channels``-channel (default 6) range view features.  The
    encoder is always run under ``torch.no_grad()`` (frozen); the decoder is
    intentionally *not* wrapped in ``no_grad`` so that auxiliary reconstruction
    losses (L1, Chamfer) can propagate gradients back through the DiT.

    The spatial compression ratio of the underlying ``dc_ae_f32c32_rangeview``
    model is 32×, matching ``downsample_size=32`` in the range view config.

    Args:
        args: Config namespace.  Must expose:
            - ``args.patch_size``     (int)  patchification factor after DCAE
            - ``args.vae_embed_dim``  (int)  DCAE latent channels (32)
            - ``args.range_channels`` (int, optional) input channels (default 6)
            - ``args.vae_ckpt``       (str, optional) pre-trained DCAE path
        local_rank: CUDA device index.
    """

    def __init__(self, args, local_rank: int):
        super().__init__()
        self.patch_size = args.patch_size
        self.vae_embed_dim = args.vae_embed_dim  # latent channels (32 for f32c32)
        in_channels = getattr(args, 'range_channels', 6)
        vae_ckpt = getattr(args, 'vae_ckpt', None)

        dcae_cfg = dc_ae_f32c32_rangeview(
            pretrained_path=vae_ckpt,
            in_channels=in_channels,
        )
        self.vae = DCAE(dcae_cfg)
        self.vae.cuda(local_rank)

        # Freeze all DCAE parameters — only the STT and DiT are trained.
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()

        print(f"RangeViewVAETokenizer: in_channels={in_channels}, "
              f"latent_channels={self.vae_embed_dim}, vae_ckpt={vae_ckpt}")

    @torch.no_grad()
    def encode_to_z(self, x: torch.Tensor) -> torch.Tensor:
        """Encode range view images to latent tokens.

        Args:
            x: ``[B, T, C, H, W]`` normalised range view features.

        Returns:
            ``[B, T, L, latent_C]`` latent tokens where
            ``L = (H / (32*patch_size)) * (W / (32*patch_size))``
            and ``latent_C = vae_embed_dim * patch_size**2``.
        """
        b, t, _, _, _ = x.shape
        ts = rearrange(x, 'b t c h w -> (b t) c h w')
        with torch.no_grad():
            latents = self.vae.encode(ts)          # [(B*T), latent_channels, h_lat, w_lat]
        latents = patchify(latents, self.patch_size)  # [(B*T), L, latent_C]
        latents = rearrange(latents, '(b t) L c -> b t L c', b=b, t=t)
        return latents

    def decode_from_z(
        self,
        z: torch.Tensor,
        h_lat: int,
        w_lat: int,
    ) -> torch.Tensor:
        """Decode latent tokens back to range view features.

        This method does **not** use ``torch.no_grad()`` so that auxiliary
        reconstruction losses computed on the decoded output can propagate
        gradients to the DiT's predicted latents (and hence to the DiT
        weights).  The frozen DCAE weights themselves receive no gradient
        updates.

        Args:
            z:     ``[(B*T), L, latent_C]`` latent tokens.
            h_lat: Latent spatial height  (``H // 32``).
            w_lat: Latent spatial width   (``W // 32``).

        Returns:
            ``[(B*T), C, H, W]`` decoded range view features.
        """
        z = rearrange(z, 'bt (h w) c -> bt h w c', h=h_lat, w=w_lat)
        z = unpatchify(z, self.patch_size, self.vae_embed_dim)  # [(B*T), latent_channels, h_lat, w_lat]
        decoded = self.vae.decode(z)               # [(B*T), in_channels, H, W]
        return decoded