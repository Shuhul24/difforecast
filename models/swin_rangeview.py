"""
TULIP-inspired Swin Transformer V2 encoder + decoder for range-view LiDAR images.

Encoder: PatchEmbed(4×8, circular pad on azimuth)
         → grid (16,256)  →  Stage-0 (2 blocks)  →  PatchMerging
         → grid (8,128)   →  Stage-1 (6 blocks)  →  PatchMerging
         → grid (4,64)    →  Stage-2 (2 blocks, bottleneck)
         → flatten  →  [B, 256, 384]   ← same latent shape as DINOv2 pipeline

Decoder: [B, 256, 384]  →  unflatten  →  Stage-0 (2 blocks)  →  PatchExpanding
         → grid (8,128)  →  Stage-1 (6 blocks)  →  PatchExpanding
         → grid (16,256) →  Stage-2 (2 blocks)
         → AsymmetricExpand(4×8) + Conv2d  →  [B, C, 64, 2048]

No skip connections: the bottleneck must encode all information, making it a clean
latent for the DiT diffusion model in Stage 2.

Key adaptations from TULIP (https://github.com/ethz-asl/TULIP):
  - Circular padding on width axis (azimuth wrap-around)
  - Asymmetric patch size (4×8) for the 1:32 aspect ratio of range images
  - SwinV2 cosine attention + continuous relative position bias (MLP-based)
  - Bottleneck latent [B,256,384] directly compatible with FluxDiT (same as DINOv2)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ── Drop path (stochastic depth) ─────────────────────────────────────────────

class _DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device).floor_() + keep
        return x * random_tensor / keep


class _Mlp(nn.Module):
    def __init__(self, in_f, hidden_f=None, out_f=None, drop=0.):
        super().__init__()
        hidden_f = hidden_f or in_f
        out_f = out_f or in_f
        self.fc1 = nn.Linear(in_f, hidden_f)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_f, out_f)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


# ── Window partition helpers ──────────────────────────────────────────────────

def _window_partition(x, wh, ww):
    """[B, H, W, C] → [B*nH*nW, wh*ww, C]"""
    B, H, W, C = x.shape
    x = x.view(B, H // wh, wh, W // ww, ww, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, wh * ww, C)


def _window_reverse(windows, wh, ww, H, W):
    """[B*nH*nW, wh*ww, C] → [B, H, W, C]"""
    B = int(windows.shape[0] / ((H // wh) * (W // ww)))
    x = windows.view(B, H // wh, W // ww, wh, ww, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# ── SwinV2 Window Attention (cosine similarity + continuous RPB) ──────────────

class _WindowAttnV2(nn.Module):
    """Window multi-head self-attention (SwinV2 variant).

    Modifications from SwinV1:
      - Cosine similarity attention with per-head learnable log temperature.
      - Continuous relative position bias via a 2-layer MLP (log-spaced coords).
    """
    def __init__(self, dim, window_size, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        wh, ww = window_size
        N = wh * ww

        # Per-head learnable temperature (log scale, clamped to [0, log(100)])
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones(num_heads, 1, 1)))

        # Continuous relative position bias MLP: 2-d log-spaced coords → bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        # Log-spaced relative coordinate table [(2wh-1)*(2ww-1), 2]
        rh = torch.arange(-(wh - 1), wh, dtype=torch.float32)
        rw = torch.arange(-(ww - 1), ww, dtype=torch.float32)
        rel = torch.stack(torch.meshgrid(rh, rw, indexing='ij'), dim=-1)  # [(2wh-1),(2ww-1),2]
        norm = torch.tensor([max(wh - 1, 1), max(ww - 1, 1)], dtype=torch.float32)
        rel = rel / norm * 8
        rel = torch.sign(rel) * torch.log2(torch.abs(rel) + 1.0) / math.log2(9)
        self.register_buffer('rel_coords', rel.view(-1, 2), persistent=False)

        # Relative position index [N, N]
        ch = torch.arange(wh); cw = torch.arange(ww)
        coords = torch.stack(torch.meshgrid(ch, cw, indexing='ij')).flatten(1)  # [2, N]
        idx = coords[:, :, None] - coords[:, None, :]   # [2, N, N]
        idx = idx.permute(1, 2, 0)
        idx[:, :, 0] += wh - 1
        idx[:, :, 1] += ww - 1
        idx[:, :, 0] *= 2 * ww - 1
        self.register_buffer('rel_idx', idx.sum(-1), persistent=False)  # [N, N]

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        Nh = self.num_heads
        bias = torch.cat([self.q_bias, torch.zeros_like(self.v_bias), self.v_bias])
        qkv = F.linear(x, self.qkv.weight, bias).view(B_, N, 3, Nh, C // Nh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each [B_, Nh, N, head_dim]

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        scale = torch.clamp(self.logit_scale, max=math.log(100.)).exp()
        attn = (q @ k.transpose(-2, -1)) * scale  # [B_, Nh, N, N]

        # Continuous RPB
        cpb = self.cpb_mlp(self.rel_coords)                 # [(2wh-1)*(2ww-1), Nh]
        cpb = 16 * torch.sigmoid(cpb)
        bias_mat = cpb[self.rel_idx.view(-1)].view(N, N, Nh).permute(2, 0, 1)  # [Nh, N, N]
        attn = attn + bias_mat.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, Nh, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, Nh, N, N)

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


# ── SwinV2 Transformer Block ──────────────────────────────────────────────────

class SwinV2Block(nn.Module):
    """Pre-LN SwinV2 block with window attention and SW-MSA option.

    Handles asymmetric windows (wh, ww) and the degenerate case H == wh by
    disabling vertical shifting (only horizontal shift remains useful).
    """
    def __init__(self, dim, input_resolution, num_heads,
                 window_size=(4, 8), shift_size=(0, 0),
                 mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        H, W = input_resolution
        wh = min(window_size[0], H)
        ww = min(window_size[1], W)
        self.window_size = (wh, ww)
        self.input_resolution = input_resolution
        # Disable vertical shift when H == wh (only 1 window row)
        sh = shift_size[0] if H > wh else 0
        sw = shift_size[1] if W > ww else 0
        self.shift_size = (sh, sw)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowAttnV2(dim, self.window_size, num_heads,
                                   attn_drop=attn_drop, proj_drop=drop)
        self.dp = _DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio), drop=drop)

        # Precompute SW-MSA attention mask
        if sh > 0 or sw > 0:
            img_mask = torch.zeros(1, H, W, 1)
            cnt = 0
            for h in (slice(0, -wh), slice(-wh, -sh), slice(-sh, None)):
                for w in (slice(0, -ww), slice(-ww, -sw), slice(-sw, None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_win = _window_partition(img_mask.float(), wh, ww)  # [nW, wh*ww, 1]
            mask_win = mask_win.squeeze(-1)                          # [nW, wh*ww]
            attn_mask = mask_win.unsqueeze(1) - mask_win.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.).masked_fill(attn_mask == 0, 0.)
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask, persistent=False)

    def forward(self, x):
        """x: [B, H*W, C] → [B, H*W, C]"""
        H, W = self.input_resolution
        B, L, C = x.shape
        x2d = x.view(B, H, W, C)

        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            x2d = torch.roll(x2d, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))

        x_win = _window_partition(x2d, *self.window_size)     # [B*nW, wh*ww, C]
        x_win = x_win + self.dp(self.attn(self.norm1(x_win), mask=self.attn_mask))
        x_win = x_win + self.dp(self.mlp(self.norm2(x_win)))
        x2d = _window_reverse(x_win, *self.window_size, H, W) # [B, H, W, C]

        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            x2d = torch.roll(x2d, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

        return x2d.view(B, L, C)


# ── Patch operations ──────────────────────────────────────────────────────────

class SwinPatchEmbed(nn.Module):
    """Patch embedding for range images with circular padding on the azimuth axis.

    Uses a conv with kernel (Ph, Pw+4) after circular-padding 2 pixels each side
    on the width (azimuth) dimension — mirrors TULIP's circular_padding mode.
    """
    def __init__(self, img_size=(64, 2048), patch_size=(4, 8), in_chans=2, embed_dim=96):
        super().__init__()
        Ph, Pw = patch_size
        self.patch_size = patch_size
        self.grid = (img_size[0] // Ph, img_size[1] // Pw)   # (16, 256)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(Ph, Pw + 4), stride=(Ph, Pw))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """[B, C, H, W] → [B, Hp*Wp, embed_dim]"""
        x = F.pad(x, (2, 2, 0, 0), mode='circular')
        x = self.proj(x)                     # [B, embed_dim, Hp, Wp]
        B, C, Hp, Wp = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, Hp * Wp, C)
        return self.norm(x)


class _PatchMerging(nn.Module):
    """2× spatial downsample, 2× channel expansion (SwinV2 style with pre-norm)."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        """[B, H*W, C] → [B, (H//2)*(W//2), 2C]"""
        B, _, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                        x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)   # [B,H/2,W/2,4C]
        x = x.view(B, (H // 2) * (W // 2), 4 * C)
        return self.reduction(self.norm(x))


class _PatchExpanding(nn.Module):
    """2× spatial upsample, halve channels (used in decoder)."""
    def __init__(self, dim):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x, H, W):
        """[B, H*W, C] → [B, (2H)*(2W), C//2]"""
        B = x.shape[0]
        x = self.expand(x)                         # [B, H*W, 2C]
        x = x.view(B, H, W, -1)
        x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=2, P2=2)
        return self.norm(x.view(B, (H * 2) * (W * 2), -1))


class _AsymmetricExpand(nn.Module):
    """Final expand: patch grid → pixel space with asymmetric (Ph, Pw) patch size.

    Goes from [B, Hp*Wp, C] to [B, (Hp*Ph)*(Wp*Pw), C], then conv to output channels.
    """
    def __init__(self, dim, patch_h, patch_w, out_chans):
        super().__init__()
        self.ph, self.pw = patch_h, patch_w
        self.expand = nn.Linear(dim, patch_h * patch_w * dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.pred = nn.Conv2d(dim, out_chans, kernel_size=1, bias=True)

    def forward(self, x, Hp, Wp):
        """[B, Hp*Wp, C] → [B, out_chans, Hp*Ph, Wp*Pw]"""
        B = x.shape[0]
        ph, pw = self.ph, self.pw
        x = self.expand(x)                                    # [B, Hp*Wp, ph*pw*C]
        x = x.view(B, Hp, Wp, -1)
        x = rearrange(x, 'B H W (Ph Pw C) -> B (H Ph) (W Pw) C', Ph=ph, Pw=pw)
        x = self.norm(x.view(B, Hp * ph * Wp * pw, -1))
        x = x.view(B, Hp * ph, Wp * pw, -1).permute(0, 3, 1, 2).contiguous()
        return self.pred(x)


# ── stochastic-depth schedule ─────────────────────────────────────────────────

def _stoch_depth_rates(total_depth, drop_path_rate):
    return torch.linspace(0, drop_path_rate, total_depth).tolist()


# ── Encoder ───────────────────────────────────────────────────────────────────

class TULIPRangeEncoder(nn.Module):
    """Hierarchical Swin encoder for range-view LiDAR images.

    Produces [B, 256, 384] latents — same shape as the DINOv2 encoder in
    dino_rae_rangeview.py — so Stage-2 DiT/STT configs are reused unchanged.

    Spatial flow:
      [B, C, 64, 2048]
        → PatchEmbed(4×8, circular)   [B, 4096, 96]    grid (16,256)
        → Stage 0 (2 SwinV2 blocks)
        → PatchMerging               [B, 1024, 192]   grid (8,128)
        → Stage 1 (6 SwinV2 blocks)
        → PatchMerging               [B, 256,  384]   grid (4,64)
        → Stage 2 (2 SwinV2 blocks, bottleneck)
        → LayerNorm                  [B, 256,  384]
    """
    # Constants exposed so Stage-2 code can query them
    GRID_BOTTLENECK = (4, 64)        # (H, W) at bottleneck
    LATENT_TOKENS   = 256            # 4*64
    LATENT_DIM      = 384            # embed_dim * 4

    def __init__(
        self,
        in_chans       = 2,
        embed_dim      = 96,
        depths         = (2, 6, 2),
        num_heads      = (3, 6, 12),
        window_size    = (4, 8),
        mlp_ratio      = 4.,
        drop_rate      = 0.,
        attn_drop_rate = 0.,
        drop_path_rate = 0.1,
    ):
        super().__init__()
        assert len(depths) == len(num_heads) == 3
        self.embed_dim = embed_dim

        self.patch_embed = SwinPatchEmbed(
            img_size=(64, 2048), patch_size=(4, 8),
            in_chans=in_chans, embed_dim=embed_dim,
        )
        self.pos_drop = nn.Dropout(drop_rate)

        # Stochastic depth schedule shared across all blocks
        dpr = _stoch_depth_rates(sum(depths), drop_path_rate)
        dpr_idx = 0

        self.stages  = nn.ModuleList()
        self.merges  = nn.ModuleList()
        grids = [(16, 256), (8, 128), (4, 64)]
        dims  = [embed_dim, embed_dim * 2, embed_dim * 4]

        for s in range(3):
            H_s, W_s = grids[s]
            d_s = dims[s]
            blocks = nn.ModuleList()
            for i in range(depths[s]):
                sh = window_size[0] // 2 if i % 2 == 1 else 0
                sw = window_size[1] // 2 if i % 2 == 1 else 0
                blocks.append(SwinV2Block(
                    dim=d_s, input_resolution=(H_s, W_s),
                    num_heads=num_heads[s], window_size=window_size,
                    shift_size=(sh, sw), mlp_ratio=mlp_ratio,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[dpr_idx],
                ))
                dpr_idx += 1
            self.stages.append(blocks)
            if s < 2:
                self.merges.append(_PatchMerging(d_s))

        self.norm = nn.LayerNorm(dims[2])

    def forward(self, x):
        """[B, C, 64, 2048] → [B, 256, 384]"""
        x = self.pos_drop(self.patch_embed(x))   # [B, 4096, 96]

        H, W = 16, 256
        for s, blocks in enumerate(self.stages):
            for blk in blocks:
                x = blk(x)
            if s < 2:
                x = self.merges[s](x, H, W)
                H, W = H // 2, W // 2

        return self.norm(x)   # [B, 256, 384]


# ── Decoder ───────────────────────────────────────────────────────────────────

class TULIPRangeDecoder(nn.Module):
    """Hierarchical Swin decoder for range-view LiDAR images.

    Symmetric to TULIPRangeEncoder; no skip connections so it operates entirely
    from the [B, 256, 384] DiT-predicted latent, which already encodes temporal
    context from Stage-2 conditioning.

    Spatial flow:
      [B, 256, 384]
        → Stage 0 (2 SwinV2 blocks)  [B, 256, 384]   grid (4,64)
        → PatchExpanding             [B, 1024, 192]   grid (8,128)
        → Stage 1 (6 SwinV2 blocks)  [B, 1024, 192]
        → PatchExpanding             [B, 4096, 96]    grid (16,256)
        → Stage 2 (2 SwinV2 blocks)  [B, 4096, 96]
        → AsymmetricExpand(4×8)      [B, out_chans, 64, 2048]
    """
    def __init__(
        self,
        out_chans      = 2,
        embed_dim      = 96,
        depths         = (2, 6, 2),
        num_heads      = (3, 6, 12),
        window_size    = (4, 8),
        mlp_ratio      = 4.,
        drop_rate      = 0.,
        attn_drop_rate = 0.,
        drop_path_rate = 0.1,
    ):
        super().__init__()
        assert len(depths) == len(num_heads) == 3
        # Decoder mirrors encoder: bottleneck first, finest last
        dec_depths = depths[::-1]
        dec_heads  = num_heads[::-1]
        grids = [(4, 64), (8, 128), (16, 256)]
        dims  = [embed_dim * 4, embed_dim * 2, embed_dim]

        dpr = _stoch_depth_rates(sum(dec_depths), drop_path_rate)
        dpr_idx = 0

        self.stages  = nn.ModuleList()
        self.expands = nn.ModuleList()

        for s in range(3):
            H_s, W_s = grids[s]
            d_s = dims[s]
            blocks = nn.ModuleList()
            for i in range(dec_depths[s]):
                sh = window_size[0] // 2 if i % 2 == 1 else 0
                sw = window_size[1] // 2 if i % 2 == 1 else 0
                blocks.append(SwinV2Block(
                    dim=d_s, input_resolution=(H_s, W_s),
                    num_heads=dec_heads[s], window_size=window_size,
                    shift_size=(sh, sw), mlp_ratio=mlp_ratio,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[dpr_idx],
                ))
                dpr_idx += 1
            self.stages.append(blocks)
            if s < 2:
                self.expands.append(_PatchExpanding(d_s))

        self.norm = nn.LayerNorm(embed_dim)
        self.final_expand = _AsymmetricExpand(embed_dim, patch_h=4, patch_w=8, out_chans=out_chans)

    def forward(self, z):
        """[B, 256, 384] → [B, out_chans, 64, 2048]"""
        x = z
        H, W = 4, 64
        for s, blocks in enumerate(self.stages):
            for blk in blocks:
                x = blk(x)
            if s < 2:
                x = self.expands[s](x, H, W)
                H, W = H * 2, W * 2

        x = self.norm(x)                      # [B, 4096, 96]
        return self.final_expand(x, H, W)     # [B, out_chans, 64, 2048]


# ── Weight initialisation (from TULIP / Swin paper) ──────────────────────────

def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
