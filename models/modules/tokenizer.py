import torch
import torch.nn as nn
from einops import rearrange
from models.modules.dcae import (
    DCAE, dc_ae_f32c32,
    TransformerBlock, precompute_freqs_cis, zero_initialize,
)
from models.modules.rangeldm_vae import RangeLDMVAE, DiagonalGaussianDistribution

def poses_to_indices(poses, x_divisions=128, y_divisions=128,
                     x_range=50.0, y_range=10.0):
    """Quantise relative translations into integer indices for token embedding.

    Args:
        poses:      ``[B, T, 2]`` relative (x, y) translations in metres.
        x_divisions: Number of bins along x (forward).
        y_divisions: Number of bins along y (lateral).
        x_range:    Max expected forward displacement in metres (e.g. 50 for KITTI).
        y_range:    Full lateral range in metres (symmetric, e.g. 10 → ±5 m).
    """
    y_step = y_range / y_divisions
    y_min = -y_range / 2.0 + y_step / 2.0
    x, y = poses[..., 0], poses[..., 1]
    idx_x = (x * x_divisions / x_range).clip(0, x_divisions).to(torch.float32).unsqueeze(dim=2)
    idx_y = ((y - y_min) * y_divisions / y_range).clip(0, y_divisions).to(torch.float32).unsqueeze(dim=2)
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

def patchify(x, patch_size_h, patch_size_w=None):
    """Flatten spatial patches into token vectors.

    Supports both square (patch_size_h == patch_size_w) and non-uniform
    (patch_size_h != patch_size_w) patches.  Non-uniform patches are
    important for range images whose VAE latent has a very wide aspect ratio
    (e.g. 16×512): a square patch would reduce the elevation dimension to
    just 2 tokens, discarding most vertical structure.

    Args:
        x:            ``[B, C, H, W]`` feature map.
        patch_size_h: Patch height (elevation axis).
        patch_size_w: Patch width  (azimuth axis). Defaults to patch_size_h
                      (square patches, backward-compatible).

    Returns:
        ``[B, L, C * patch_size_h * patch_size_w]`` token tensor where
        ``L = (H // patch_size_h) * (W // patch_size_w)``.
    """
    if patch_size_w is None:
        patch_size_w = patch_size_h
    bsz, c, h, w = x.shape
    ph, pw = patch_size_h, patch_size_w
    h_ = h // ph
    w_ = w // pw

    x = x.reshape(bsz, c, h_, ph, w_, pw)
    x = torch.einsum('nchpwq->nhwcpq', x)
    x = x.reshape(bsz, h_ * w_, c * ph * pw)
    return x  # [B, L, C*ph*pw]


def unpatchify(x, patch_size_h, patch_size_w, vae_embed_dim):
    """Reconstruct a spatial feature map from token vectors.

    Args:
        x:            ``[B, h_, w_, C * patch_size_h * patch_size_w]``
                      token tensor (already reshaped to 2-D grid).
        patch_size_h: Patch height used during patchify.
        patch_size_w: Patch width  used during patchify.
        vae_embed_dim: Original channel count C before patchify.

    Returns:
        ``[B, C, h_ * patch_size_h, w_ * patch_size_w]`` feature map.
    """
    bsz, h_, w_, _ = x.shape
    ph, pw = patch_size_h, patch_size_w
    c = vae_embed_dim

    x = x.reshape(bsz, h_, w_, c, ph, pw)
    x = torch.einsum('nhwcpq->nchpwq', x)
    x = x.reshape(bsz, c, h_ * ph, w_ * pw)
    return x

class TemporalLatentEncoder(nn.Module):
    """Causal temporal + spatial attention applied in patchified latent space.

    Sits between the per-frame VAE encoder and the STT, refining the
    ``[B, T, L, C]`` token grid so that each frame's latents are informed
    by all *past* frames (causal temporal attention) before spatial token
    mixing within each frame (full spatial attention).

    Operating in the patchified space (L=128, C=256 for the default KITTI
    config) rather than the raw VAE latent map (L=8192, C=4) keeps the
    attention cost tractable and the embedding dimension large enough for
    meaningful attention heads.

    All blocks are **zero-initialised** so the module is an exact identity
    residual at training start, fully preserving the pretrained VAE's
    latent distribution.

    Args:
        dim:        Channel dimension of patchified latents
                    (= vae_embed_dim × patch_size², e.g. 4 × 64 = 256).
        n_heads:    Attention heads (head_dim = dim // n_heads = 32).
        n_blocks:   Number of interleaved (causal-time, spatial) block pairs.
        max_frames: Maximum T to precompute RoPE for (condition_frames + 1).
        token_size: Spatial tokens per frame (L = h_lat × w_lat = 128).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_blocks: int,
        max_frames: int,
        token_size: int,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        head_dim = dim // n_heads

        # Interleaved causal-temporal and spatial attention blocks
        self.time_blocks = nn.ModuleList(
            [TransformerBlock(n_heads=n_heads, dim=dim) for _ in range(n_blocks)]
        )
        self.space_blocks = nn.ModuleList(
            [TransformerBlock(n_heads=n_heads, dim=dim) for _ in range(n_blocks)]
        )

        # Zero-init → identity residual; preserves pretrained VAE latents exactly.
        for blk in self.time_blocks:
            zero_initialize(blk)
        for blk in self.space_blocks:
            zero_initialize(blk)

        # RoPE frequencies: precompute_freqs_cis(head_dim, seq_len) → (seq_len, head_dim//2)
        self.register_buffer(
            'freqs_cis_time',
            precompute_freqs_cis(head_dim, max_frames, theta=1000.0),
        )
        self.register_buffer(
            'freqs_cis_space',
            precompute_freqs_cis(head_dim, token_size, theta=1000.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal temporal then full spatial attention.

        Args:
            x: ``[B, T, L, C]`` patchified latents.

        Returns:
            ``[B, T, L, C]`` refined latents (same shape as input).
        """
        B, T, L, C = x.shape

        # Lower-triangular causal mask: frame t attends only to frames 0…t.
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=x.dtype))
        causal_mask = torch.where(
            causal == 0,
            torch.full_like(causal, float('-inf')),
            torch.zeros_like(causal),
        )

        freqs_t = self.freqs_cis_time[:T]   # (T,  head_dim//2)
        freqs_s = self.freqs_cis_space[:L]  # (L,  head_dim//2)

        xx = rearrange(x, 'b t l c -> (b t) l c')

        for i in range(self.n_blocks):
            # Causal temporal attention —————————————————————————————————————
            # Treat each spatial position as an independent sequence over T.
            xx = rearrange(xx, '(b t) l c -> (b l) t c', b=B, t=T)
            xx = self.time_blocks[i](xx, freqs_t, causal_mask)
            xx = rearrange(xx, '(b l) t c -> (b t) l c', b=B, l=L)

            # Spatial attention ——————————————————————————————————————————————
            # Mix all L tokens within each frame independently.
            xx = self.space_blocks[i](xx, freqs_s)

        return rearrange(xx, '(b t) l c -> b t l c', b=B, t=T)


class TemporalLatentDecoder(nn.Module):
    """Full (non-causal) temporal + spatial attention for decoding predicted latents.

    Sits between the DiT's AR-predicted latents and the VAE decoder.  Given
    a sequence ``[B, T, L, C]`` of predicted latents (one per AR step), it
    applies **full temporal attention** (no causal mask) so every predicted
    frame can attend to the entire AR chain, then spatial attention within
    each frame.

    At decode time all T predicted frames are available, so the non-causal
    mask is strictly better than causal: early frames can self-correct using
    information from later predictions, reducing temporal flicker.

    Zero-initialised → exact identity residual at training start, so enabling
    this module has zero impact before any gradient updates.  Trained via
    pixel-space auxiliary losses (range L1, Chamfer) when fw_iter > 1 and
    range_view_loss_weight > 0.

    Args:
        dim:        Channel dimension (= vae_embed_dim × patch_h × patch_w).
        n_heads:    Number of attention heads.
        n_blocks:   Number of interleaved (full-time, spatial) block pairs.
        max_frames: Maximum T to precompute RoPE for (forward_iter).
        token_size: Spatial tokens per frame (L = h_lat × w_lat).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_blocks: int,
        max_frames: int,
        token_size: int,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        head_dim = dim // n_heads

        self.time_blocks = nn.ModuleList(
            [TransformerBlock(n_heads=n_heads, dim=dim) for _ in range(n_blocks)]
        )
        self.space_blocks = nn.ModuleList(
            [TransformerBlock(n_heads=n_heads, dim=dim) for _ in range(n_blocks)]
        )

        for blk in self.time_blocks:
            zero_initialize(blk)
        for blk in self.space_blocks:
            zero_initialize(blk)

        self.register_buffer(
            'freqs_cis_time',
            precompute_freqs_cis(head_dim, max_frames, theta=1000.0),
        )
        self.register_buffer(
            'freqs_cis_space',
            precompute_freqs_cis(head_dim, token_size, theta=1000.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply full (non-causal) temporal then spatial attention.

        Args:
            x: ``[B, T, L, C]`` predicted latent sequence.

        Returns:
            ``[B, T, L, C]`` temporally-refined latents (same shape).
        """
        B, T, L, C = x.shape

        freqs_t = self.freqs_cis_time[:T]
        freqs_s = self.freqs_cis_space[:L]

        xx = rearrange(x, 'b t l c -> (b t) l c')

        for i in range(self.n_blocks):
            # Full (non-causal) temporal attention: each frame attends to all T.
            xx = rearrange(xx, '(b t) l c -> (b l) t c', b=B, t=T)
            xx = self.time_blocks[i](xx, freqs_t, mask=None)
            xx = rearrange(xx, '(b l) t c -> (b t) l c', b=B, l=L)

            # Spatial attention within each frame.
            xx = self.space_blocks[i](xx, freqs_s)

        return rearrange(xx, '(b t) l c -> b t l c', b=B, t=T)


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
        x = unpatchify(x, self.args.patch_size, self.args.patch_size, self.args.vae_embed_dim)
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
    """RangeLDM VAE-based tokenizer for 2-channel range view images.

    Uses the Encoder/Decoder from RangeLDM (kitti360 config) which was
    purpose-built for LiDAR range images with circular (azimuth) padding.

    Architecture summary:
        Input:   2-channel [range, intensity],  shape [B, 2, 64, 2048]
        VAE enc: 4× spatial compression  →  latent [B, 4, 16, 512]
        Patchify: patch_size=8           →  tokens  [B, 128, 256]
                  (L = 2×64 = 128,  latent_C = 4×64 = 256)
        [optional] TemporalLatentEncoder  →  tokens  [B, 128, 256]  (same shape)

    The VAE weights are frozen when a pretrained checkpoint is provided.
    The TemporalLatentEncoder (if enabled) is always trainable and
    zero-initialised so it starts as an identity residual.

    Args:
        args: Config namespace.  Must expose:
            - ``args.patch_size``           (int)   patchification after VAE (default 8)
            - ``args.vae_embed_dim``        (int)   VAE latent channels (4 for RangeLDM)
            - ``args.vae_ckpt``             (str, optional) RangeLDM checkpoint path
            - ``args.add_encoder_temporal`` (bool)  enable TemporalLatentEncoder
            - ``args.n_temporal_blocks``    (int)   number of time+space block pairs (default 4)
            - ``args.condition_frames``     (int)   number of conditioning frames
            - ``args.block_size``           (int)   temporal block size
            - ``args.range_h / range_w``    (int)   range image height / width
            - ``args.downsample_size``      (int)   VAE spatial compression factor
        local_rank: CUDA device index.
    """

    def __init__(self, args, local_rank: int):
        super().__init__()
        # Non-uniform patch support: patch_size_h / patch_size_w can differ.
        # Falls back to the legacy square patch_size when only patch_size is set.
        self.patch_size_h = int(getattr(args, 'patch_size_h', args.patch_size))
        self.patch_size_w = int(getattr(args, 'patch_size_w', args.patch_size))
        self.vae_embed_dim = args.vae_embed_dim  # z_channels = 4
        vae_ckpt = getattr(args, 'vae_ckpt', None)

        self.vae = RangeLDMVAE(ckpt_path=vae_ckpt)
        self.vae.cuda(local_rank)

        # VAE parameters are trainable when no pre-trained checkpoint is provided.
        # With a checkpoint, freeze them to preserve the learned codec.
        if vae_ckpt is None:
            for param in self.vae.parameters():
                param.requires_grad = True
        else:
            for param in self.vae.parameters():
                param.requires_grad = False
            self.vae.eval()

        # ------------------------------------------------------------------ #
        # Optional: causal temporal + spatial attention in patchified space.
        #
        # Operates on [B, T, L, C] tokens (L=128, C=256 for KITTI defaults).
        # Zero-initialised → identity residual at training start; preserves
        # the pretrained VAE latent distribution without any warm-up needed.
        # ------------------------------------------------------------------ #
        self.add_encoder_temporal = getattr(args, 'add_encoder_temporal', False)
        self.encoder_temporal = None

        if self.add_encoder_temporal:
            # Patchified channel dim: vae_embed_dim × patch_size_h × patch_size_w
            latent_dim = args.vae_embed_dim * self.patch_size_h * self.patch_size_w
            n_heads     = 8                  # head_dim = latent_dim // 8
            n_blocks    = getattr(args, 'n_temporal_blocks', 4)

            # Spatial tokens per frame after patchify
            h_lat = args.range_h // (args.downsample_size * self.patch_size_h)
            w_lat = args.range_w // (args.downsample_size * self.patch_size_w)
            token_size = h_lat * w_lat

            # encode_to_z is called with CF frames (eval) or CF+1 (train)
            cf = args.condition_frames // args.block_size
            max_frames = cf + 1

            self.encoder_temporal = TemporalLatentEncoder(
                dim=latent_dim,
                n_heads=n_heads,
                n_blocks=n_blocks,
                max_frames=max_frames,
                token_size=token_size,
            ).cuda(local_rank)

            print(
                f"RangeViewVAETokenizer: TemporalLatentEncoder enabled — "
                f"dim={latent_dim}, n_heads={n_heads}, n_blocks={n_blocks}, "
                f"max_frames={max_frames}, token_size={token_size}"
            )

        # ------------------------------------------------------------------ #
        # Optional: non-causal temporal + spatial attention on the decoder side.
        #
        # Given a sequence [B, T, L, C] of AR-predicted latents, applies full
        # temporal attention (each predicted frame attends to all others) before
        # per-frame VAE decoding.  This reduces temporal flicker because early
        # predictions can self-correct using context from later frames.
        #
        # Zero-initialised → identity residual; has no effect at training start.
        # Trained via pixel-space auxiliary losses when fw_iter > 1 and
        # range_view_loss_weight > 0.
        # ------------------------------------------------------------------ #
        self.add_decoder_temporal = getattr(args, 'add_decoder_temporal', False)
        self.decoder_temporal = None

        if self.add_decoder_temporal:
            latent_dim = args.vae_embed_dim * self.patch_size_h * self.patch_size_w
            n_heads    = 8
            n_blocks   = getattr(args, 'n_decoder_temporal_blocks',
                                  getattr(args, 'n_temporal_blocks', 2))

            h_lat = args.range_h // (args.downsample_size * self.patch_size_h)
            w_lat = args.range_w // (args.downsample_size * self.patch_size_w)
            token_size = h_lat * w_lat

            # Max frames = forward_iter (the decoder sees the full AR chain).
            max_frames = int(getattr(args, 'forward_iter', 5))

            self.decoder_temporal = TemporalLatentDecoder(
                dim=latent_dim,
                n_heads=n_heads,
                n_blocks=n_blocks,
                max_frames=max_frames,
                token_size=token_size,
            ).cuda(local_rank)

            print(
                f"RangeViewVAETokenizer: TemporalLatentDecoder enabled — "
                f"dim={latent_dim}, n_heads={n_heads}, n_blocks={n_blocks}, "
                f"max_frames={max_frames}, token_size={token_size}"
            )

        print(f"RangeViewVAETokenizer (RangeLDM): in_channels=2, "
              f"z_channels={self.vae_embed_dim}, "
              f"patch_size=({self.patch_size_h},{self.patch_size_w}), "
              f"vae_ckpt={vae_ckpt}, "
              f"add_encoder_temporal={self.add_encoder_temporal}, "
              f"add_decoder_temporal={self.add_decoder_temporal}")

    def encode_to_z(self, x: torch.Tensor) -> torch.Tensor:
        """Encode 2-channel range view images to latent tokens.

        Pipeline:
            [B, T, 2, H, W]
            → per-frame VAE encode  → [(B*T), z_ch, H/4, W/4]
            → patchify              → [(B*T), L, latent_C]
            → rearrange             → [B, T, L, latent_C]
            → TemporalLatentEncoder → [B, T, L, latent_C]  (if enabled)

        The TemporalLatentEncoder applies causal temporal attention (frame t
        sees only frames 0…t) followed by spatial token mixing within each
        frame, enriching the conditioning latents with motion information
        before they reach the STT.

        Args:
            x: ``[B, T, 2, H, W]`` normalised [range, intensity] images.

        Returns:
            ``[B, T, L, latent_C]`` latent tokens.
        """
        b, t, _, _, _ = x.shape
        ts = rearrange(x, 'b t c h w -> (b t) c h w')  # [(B*T), 2, H, W]
        vae_dtype = next(self.vae.parameters()).dtype
        latents = self.vae.encode(ts.to(vae_dtype))                          # [(B*T), z_ch, H/4, W/4]
        latents = patchify(latents, self.patch_size_h, self.patch_size_w)    # [(B*T), L, latent_C]
        latents = rearrange(latents, '(b t) L c -> b t L c', b=b, t=t)

        if self.encoder_temporal is not None:
            latents = self.encoder_temporal(latents)    # [B, T, L, latent_C]

        return latents

    def decode_from_z(
        self,
        z: torch.Tensor,
        h_lat: int,
        w_lat: int,
    ) -> torch.Tensor:
        """Decode latent tokens back to 2-channel range view images.

        Not wrapped in ``torch.no_grad()`` so that auxiliary reconstruction
        losses can propagate gradients to the DiT's predicted latents.
        The frozen VAE weights themselves receive no gradient updates.

        Args:
            z:     ``[(B*T), L, latent_C]`` latent tokens.
            h_lat: Latent spatial height after patchify  (``H // (4*patch_size)``).
            w_lat: Latent spatial width  after patchify  (``W // (4*patch_size)``).

        Returns:
            ``[(B*T), 2, H, W]`` decoded [range, intensity] images.
        """
        z = rearrange(z, 'bt (h w) c -> bt h w c', h=h_lat, w=w_lat)
        z = unpatchify(z, self.patch_size_h, self.patch_size_w, self.vae_embed_dim)
        vae_dtype = next(self.vae.parameters()).dtype
        return self.vae.decode(z.to(vae_dtype))  # [(B*T), 2, H, W]

    def decode_from_z_temporal(
        self,
        z_seq: torch.Tensor,
        h_lat: int,
        w_lat: int,
    ) -> torch.Tensor:
        """Decode a sequence of predicted latents with temporal attention.

        If ``decoder_temporal`` is enabled, applies full (non-causal) temporal
        attention across the T predicted frames before per-frame VAE decoding.
        Each predicted frame can thereby attend to the entire AR chain, reducing
        temporal flicker in the decoded sequence.

        Falls back to independent per-frame decoding when the decoder is disabled.

        Args:
            z_seq:  ``[B, T, L, latent_C]`` AR-predicted latent sequence
                    (already in the original VAE scale, not normalised).
            h_lat:  Latent spatial height (``H // (downsample * patch_h)``).
            w_lat:  Latent spatial width  (``W // (downsample * patch_w)``).

        Returns:
            ``[B, T, 2, H, W]`` decoded [range, intensity] image sequence.
        """
        if self.decoder_temporal is not None:
            z_seq = self.decoder_temporal(z_seq)   # [B, T, L, C]

        B, T, L, C = z_seq.shape
        z_flat = rearrange(z_seq, 'b t l c -> (b t) l c')
        decoded = self.decode_from_z(z_flat, h_lat, w_lat)  # [(B*T), 2, H, W]
        return rearrange(decoded, '(b t) c h w -> b t c h w', b=B, t=T)

    def compute_vae_elbo(
        self,
        x_flat: torch.Tensor,
        logvar: torch.nn.Parameter,
        range_weight: float = 1.0,
        intensity_weight: float = 0.5,
        kl_weight: float = 1e-6,
        return_recon: bool = False,
    ):
        """Compute the ELBO loss for VAE training (reconstruction + KL divergence).

        Follows the RangeLDM ``RangeImageReconstructionLoss`` formulation:
          - Reconstruction: ``range_weight * L1(range) + intensity_weight * L1(intensity)``
          - NLL scaling:    ``rec_loss / exp(logvar) + logvar``   (Laplacian NLL with
                            learnable uncertainty; stabilises training when the
                            reconstruction scale is unknown)
          - KL:             ``KL( q(z|x) || N(0,I) )``   summed over latent dims,
                            mean over batch

        This method operates directly on the VAE encoder/decoder (not on
        patchified tokens) so that gradients flow to all VAE parameters.

        Args:
            x_flat:          ``[(B*T), 2, H, W]`` normalised [range, intensity]
                             range images (input frames for reconstruction).
            logvar:          Learnable scalar ``nn.Parameter`` (uncertainty term).
            range_weight:    L1 weight for the range/depth channel.
            intensity_weight: L1 weight for the intensity channel.
            kl_weight:       Weight on the KL divergence term (β in β-VAE).
                             Typical values: 1e-6 – 1e-4.

        Returns:
            Scalar ELBO loss  =  NLL_loss + kl_weight * KL.
        """
        vae_dtype = next(self.vae.parameters()).dtype
        x = x_flat.to(vae_dtype)

        # --- encode → posterior → reparameterised sample ----------------------
        posterior = self.vae.encode_posterior(x)   # DiagonalGaussianDistribution
        z_sample  = posterior.sample()             # [(B*T), z_ch, H/4, W/4]

        # --- decode -----------------------------------------------------------
        x_recon = self.vae.decode(z_sample)        # [(B*T), 2, H, W]

        # --- reconstruction loss (NLL) matching RangeLDM ----------------------
        rec_loss = (
            range_weight     * torch.abs(x[:, 0].contiguous() - x_recon[:, 0].contiguous())
            + intensity_weight * torch.abs(x[:, 1].contiguous() - x_recon[:, 1].contiguous())
        )  # [(B*T), H, W]

        nll_loss = rec_loss / torch.exp(logvar) + logvar          # [(B*T), H, W]
        nll_loss = nll_loss.mean()                                # scalar (mean over B*T, H, W)

        # --- KL divergence ----------------------------------------------------
        # posterior.kl() already sums over dims [1,2,3] (C, H, W) and returns
        # shape [B].  Take the mean over the batch to get a per-sample KL.
        # Do NOT divide by z_elements again — that spatial sum is already done.
        kl_loss = posterior.kl().mean()

        elbo = nll_loss + kl_weight * kl_loss
        if return_recon:
            return elbo, x_recon, nll_loss
        return elbo