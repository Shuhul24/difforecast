"""
TULIP-compatible Swin Transformer V1 encoder + decoder for range-view LiDAR images.

Uses SwinV1 attention (learnable relative-position-bias table) to match the
TULIP pre-trained KITTI checkpoint exactly, enabling weight transfer for all
Swin blocks, PatchMerging, PatchExpanding, skip projections, and norms.

Encoder: PatchEmbed(4×8, circular pad on azimuth)
         → grid (16,256)  →  Stage-0 (2 blocks)   →  PatchMerging
         → grid (8,128)   →  Stage-1 (2 blocks)   →  PatchMerging
         → grid (4,64)    →  Stage-2 (2 blocks)   →  PatchMerging
         → grid (2,32)    →  Stage-3 (2 blocks, bottleneck)
         → flatten  →  [B, 64, 768]

Decoder: [B, 64, 768]
         → Stage-0 (2 blocks, 2×32) → PatchExpanding → cat(skip-2) → Linear
         → Stage-1 (2 blocks, 4×64) → PatchExpanding → cat(skip-1) → Linear
         → Stage-2 (2 blocks, 8×128) → PatchExpanding → cat(skip-0) → Linear
         → Stage-3 (2 blocks, 16×256)
         → AsymmetricExpand(4×8) + Conv2d  →  [B, C, 64, 2048]

Differences from TULIP kept intentional:
  - 2-channel input (range + intensity) vs TULIP's 1-channel
  - Asymmetric patch (4×8) on 64×2048 vs TULIP's (1×4) on 16×1024
    → both produce the same initial token grid (16×256) and bottleneck (2×32)
  - AsymmetricExpand final head vs TULIP's PixelShuffle (different task:
    same-resolution reconstruction vs 4× upsampling)
  All Swin block weights, merging/expanding, skip projections, and norms
  are directly compatible with TULIP pre-trained weights.
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


# ── SwinV1 Window Attention (learnable relative-position-bias table) ──────────

class _WindowAttnV1(nn.Module):
    """Window multi-head self-attention — SwinV1 variant.

    Uses a learnable relative-position-bias table of shape
    [(2*wh-1)*(2*ww-1), num_heads], identical to the TULIP KITTI checkpoint.
    With window (2, 8): table size = 3*15 = 45 entries  — exact match.
    """
    def __init__(self, dim, window_size, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        wh, ww = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Learnable RPB table  [(2wh-1)*(2ww-1), num_heads]
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * wh - 1) * (2 * ww - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Relative position index  [wh*ww, wh*ww]
        ch = torch.arange(wh); cw = torch.arange(ww)
        coords = torch.stack(torch.meshgrid(ch, cw, indexing='ij')).flatten(1)  # [2, N]
        rel = coords[:, :, None] - coords[:, None, :]   # [2, N, N]
        rel = rel.permute(1, 2, 0)
        rel[:, :, 0] += wh - 1
        rel[:, :, 1] += ww - 1
        rel[:, :, 0] *= 2 * ww - 1
        self.register_buffer('relative_position_index', rel.sum(-1), persistent=False)

        self.qkv      = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        Nh = self.num_heads
        qkv = self.qkv(x).view(B_, N, 3, Nh, C // Nh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Relative position bias
        rpb = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        rpb = rpb.view(N, N, Nh).permute(2, 0, 1)        # [Nh, N, N]
        attn = attn + rpb.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, Nh, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, Nh, N, N)

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


# ── SwinV2 Window Attention (cosine attention + continuous position bias) ──────

class _WindowAttnV2(nn.Module):
    """Window multi-head self-attention — SwinV2 variant.

    Uses cosine attention with a learnable logit_scale, separate q_bias / v_bias
    (k_bias stays zero), and a small MLP (cpb_mlp) that maps 2-D log-space
    relative coordinates to per-head position biases.  Matches the parameter
    layout produced by training with the TULIP V2 / SwinV2 backbone.

    cpb_mlp layout: Linear(2→512, bias=True) → ReLU → Linear(512→num_heads, bias=False)
    """
    def __init__(self, dim, window_size, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        wh, ww = window_size

        # Cosine-attention temperature (one per head)
        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones(num_heads, 1, 1))
        )

        # Continuous position bias MLP
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        # Log-space relative coordinate table  [1, (2wh-1), (2ww-1), 2]
        rh = torch.arange(-(wh - 1), wh, dtype=torch.float32)
        rw = torch.arange(-(ww - 1), ww, dtype=torch.float32)
        table = torch.stack(torch.meshgrid(rh, rw, indexing='ij')).permute(1, 2, 0).unsqueeze(0)
        table[..., 0] /= max(wh - 1, 1)
        table[..., 1] /= max(ww - 1, 1)
        table = table * 8
        table = torch.sign(table) * torch.log2(table.abs() + 1.0) / math.log2(8)
        self.register_buffer('relative_coords_table', table, persistent=False)

        # Relative position index  [wh*ww, wh*ww]
        ch = torch.arange(wh); cw = torch.arange(ww)
        coords = torch.stack(torch.meshgrid(ch, cw, indexing='ij')).flatten(1)
        rel = coords[:, :, None] - coords[:, None, :]
        rel = rel.permute(1, 2, 0)
        rel[:, :, 0] += wh - 1
        rel[:, :, 1] += ww - 1
        rel[:, :, 0] *= 2 * ww - 1
        self.register_buffer('relative_position_index', rel.sum(-1), persistent=False)

        # QKV weight without bias; biases handled separately
        self.qkv      = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias   = nn.Parameter(torch.zeros(dim))
        self.v_bias   = nn.Parameter(torch.zeros(dim))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        Nh = self.num_heads

        qkv_bias = torch.cat([self.q_bias,
                               torch.zeros(C, device=x.device, dtype=x.dtype),
                               self.v_bias])
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.view(B_, N, 3, Nh, C // Nh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Cosine attention
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        scale = torch.clamp(self.logit_scale, max=math.log(100.0)).exp()
        attn  = (q @ k.transpose(-2, -1)) * scale

        # Continuous position bias
        rpb = self.cpb_mlp(self.relative_coords_table)   # [1, 2wh-1, 2ww-1, Nh]
        rpb = 16 * torch.sigmoid(rpb)
        rpb = rpb.view(-1, Nh)                           # [(2wh-1)(2ww-1), Nh]
        rpb = rpb[self.relative_position_index.view(-1)].view(N, N, Nh).permute(2, 0, 1)
        attn = attn + rpb.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, Nh, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, Nh, N, N)

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


# ── Swin Transformer Block ────────────────────────────────────────────────────

class SwinBlock(nn.Module):
    """Pre-LN Swin block with window attention (W-MSA) and SW-MSA option.

    Supports both SwinV1 (_WindowAttnV1) and SwinV2 (_WindowAttnV2) attention
    via the ``use_v2`` flag (default True to match trained checkpoints).

    Handles asymmetric windows (wh, ww) and the degenerate case H == wh
    by disabling vertical shifting.
    """
    def __init__(self, dim, input_resolution, num_heads,
                 window_size=(2, 8), shift_size=(0, 0),
                 mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
                 use_v2=True):
        super().__init__()
        H, W = input_resolution
        wh = min(window_size[0], H)
        ww = min(window_size[1], W)
        self.window_size = (wh, ww)
        self.input_resolution = input_resolution
        sh = shift_size[0] if H > wh else 0
        sw = shift_size[1] if W > ww else 0
        self.shift_size = (sh, sw)

        self.norm1 = nn.LayerNorm(dim)
        attn_cls = _WindowAttnV2 if use_v2 else _WindowAttnV1
        self.attn = attn_cls(dim, self.window_size, num_heads,
                             attn_drop=attn_drop, proj_drop=drop)
        self.dp    = _DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = _Mlp(dim, int(dim * mlp_ratio), drop=drop)

        # Precompute SW-MSA attention mask
        if sh > 0 or sw > 0:
            img_mask = torch.zeros(1, H, W, 1)
            cnt = 0
            for h in (slice(0, -wh), slice(-wh, -sh), slice(-sh, None)):
                for w in (slice(0, -ww), slice(-ww, -sw), slice(-sw, None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_win = _window_partition(img_mask.float(), wh, ww).squeeze(-1)
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

        x_win = _window_partition(x2d, *self.window_size)
        x_win = x_win + self.dp(self.attn(self.norm1(x_win), mask=self.attn_mask))
        x_win = x_win + self.dp(self.mlp(self.norm2(x_win)))
        x2d = _window_reverse(x_win, *self.window_size, H, W)

        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            x2d = torch.roll(x2d, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

        return x2d.view(B, L, C)


# ── Patch operations ──────────────────────────────────────────────────────────

class SwinPatchEmbed(nn.Module):
    """Patch embedding for range images with circular padding on the azimuth axis.

    Uses a conv with kernel (Ph, Pw+4) after circular-padding 2 pixels each side
    on the width (azimuth) dimension — mirrors TULIP's circular_padding mode.

    With patch (4,8) on 64×2048: produces the same (16,256) grid as TULIP's
    patch (1,4) on 16×1024, so all downstream Swin weights are compatible.
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
        x = self.proj(x)
        B, C, Hp, Wp = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, Hp * Wp, C)
        return self.norm(x)


class _PatchMerging(nn.Module):
    """2× spatial downsample, 2× channel expansion (SwinV1 style with pre-norm)."""
    def __init__(self, dim):
        super().__init__()
        self.norm      = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        """[B, H*W, C] → [B, (H//2)*(W//2), 2C]"""
        B, _, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                        x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)
        x = x.view(B, (H // 2) * (W // 2), 4 * C)
        return self.reduction(self.norm(x))


class _PatchExpanding(nn.Module):
    """2× spatial upsample, halve channels (used in decoder).

    Weight-compatible with TULIP's PatchUnmerging Conv2d(dim, 2*dim, 1×1):
    our Linear(dim, 2*dim) weight [2C, C] = Conv2d weight [2C, C, 1, 1].squeeze().
    """
    def __init__(self, dim):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm   = nn.LayerNorm(dim // 2)

    def forward(self, x, H, W):
        """[B, H*W, C] → [B, (2H)*(2W), C//2]"""
        B = x.shape[0]
        x = self.expand(x)
        x = x.view(B, H, W, -1)
        x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=2, P2=2)
        return self.norm(x.view(B, (H * 2) * (W * 2), -1))


class _AsymmetricExpand(nn.Module):
    """Final expand: patch grid → pixel space with asymmetric (Ph, Pw) patch size."""
    def __init__(self, dim, patch_h, patch_w, out_chans):
        super().__init__()
        self.ph, self.pw = patch_h, patch_w
        self.expand = nn.Linear(dim, patch_h * patch_w * dim, bias=False)
        self.norm   = nn.LayerNorm(dim)
        self.pred   = nn.Conv2d(dim, out_chans, kernel_size=1, bias=True)

    def forward(self, x, Hp, Wp):
        """[B, Hp*Wp, C] → [B, out_chans, Hp*Ph, Wp*Pw]"""
        B = x.shape[0]
        ph, pw = self.ph, self.pw
        x = self.expand(x)
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
    """4-stage Swin encoder for range-view LiDAR images.

    Produces [B, 64, 768] latents.  All Swin block weights (qkv, proj, mlp,
    norms, RPB tables) and PatchMerging weights are directly compatible with
    the TULIP KITTI pre-trained checkpoint.

    Spatial flow:
      [B, C, 64, 2048]
        → PatchEmbed(4×8, circular)   [B, 4096, 96]    grid (16,256)
        → Stage 0 (2 blocks)
        → PatchMerging                [B, 1024, 192]   grid (8,128)   ← skip-0
        → Stage 1 (2 blocks)
        → PatchMerging                [B, 256,  384]   grid (4,64)    ← skip-1
        → Stage 2 (2 blocks)
        → PatchMerging                [B, 64,   768]   grid (2,32)    ← skip-2
        → Stage 3 (2 blocks, bottleneck)
        → LayerNorm                   [B, 64,   768]

    Returns (z, skips) where skips = [feat_s0, feat_s1, feat_s2].
    """
    GRID_BOTTLENECK = (2, 32)
    LATENT_TOKENS   = 64
    LATENT_DIM      = 768

    def __init__(
        self,
        in_chans       = 2,
        embed_dim      = 96,
        depths         = (2, 2, 2, 2),
        num_heads      = (3, 6, 12, 24),
        window_size    = (2, 8),
        mlp_ratio      = 4.,
        drop_rate      = 0.,
        attn_drop_rate = 0.,
        drop_path_rate = 0.1,
        use_v2         = True,
    ):
        super().__init__()
        assert len(depths) == len(num_heads) == 4
        self.embed_dim = embed_dim

        self.patch_embed = SwinPatchEmbed(
            img_size=(64, 2048), patch_size=(4, 8),
            in_chans=in_chans, embed_dim=embed_dim,
        )
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = _stoch_depth_rates(sum(depths), drop_path_rate)
        dpr_idx = 0

        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()
        grids = [(16, 256), (8, 128), (4, 64), (2, 32)]
        dims  = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]

        for s in range(4):
            H_s, W_s = grids[s]
            d_s = dims[s]
            blocks = nn.ModuleList()
            for i in range(depths[s]):
                sh = window_size[0] // 2 if i % 2 == 1 else 0
                sw = window_size[1] // 2 if i % 2 == 1 else 0
                blocks.append(SwinBlock(
                    dim=d_s, input_resolution=(H_s, W_s),
                    num_heads=num_heads[s], window_size=window_size,
                    shift_size=(sh, sw), mlp_ratio=mlp_ratio,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[dpr_idx],
                    use_v2=use_v2,
                ))
                dpr_idx += 1
            self.stages.append(blocks)
            if s < 3:
                self.merges.append(_PatchMerging(d_s))

        self.norm = nn.LayerNorm(dims[3])

        # VAE bottleneck: project deterministic features to mean + log-variance.
        # Keeping the same dim (768) preserves Stage 2 STT/DiT compatibility.
        # logvar init to zero → posterior starts as N(0, 1), matching the prior.
        self.mu_proj     = nn.Linear(dims[3], dims[3], bias=True)
        self.logvar_proj = nn.Linear(dims[3], dims[3], bias=True)
        nn.init.zeros_(self.logvar_proj.weight)
        nn.init.zeros_(self.logvar_proj.bias)

    def forward(self, x, sample=True):
        """[B, C, 64, 2048] → (z [B,64,768], mu [B,64,768], logvar [B,64,768], skips)

        sample=True  : reparameterised sample during Stage 1 training.
        sample=False : returns mu only — used by the frozen encoder in Stage 2
                       so that diffusion operates on a deterministic, normalised code.
        """
        x = self.pos_drop(self.patch_embed(x))

        H, W = 16, 256
        skips = []
        for s, blocks in enumerate(self.stages):
            for blk in blocks:
                x = blk(x)
            if s < 3:
                skips.append(x)
                x = self.merges[s](x, H, W)
                H, W = H // 2, W // 2

        h      = self.norm(x)                              # [B, 64, 768]
        mu     = self.mu_proj(h)
        logvar = self.logvar_proj(h).clamp(-30., 20.)      # numerical stability
        if sample and self.training:
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            z = mu                                         # deterministic at eval / Stage 2
        return z, mu, logvar, skips


# ── Decoder ───────────────────────────────────────────────────────────────────

class TULIPRangeDecoder(nn.Module):
    """4-stage Swin decoder with skip connections (TULIP U-Net style).

    All Swin block weights, PatchExpanding weights (Linear ≡ Conv2d 1×1),
    skip projection weights, and the final norm are directly compatible with
    the TULIP KITTI pre-trained checkpoint.

    Spatial flow:
      [B, 64, 768]
        → Stage 0 (2 blocks, 2×32) → PatchExpanding → cat(skip-2) → Linear
        → Stage 1 (2 blocks, 4×64) → PatchExpanding → cat(skip-1) → Linear
        → Stage 2 (2 blocks, 8×128) → PatchExpanding → cat(skip-0) → Linear
        → Stage 3 (2 blocks, 16×256)
        → LayerNorm → AsymmetricExpand(4×8) → [B, out_chans, 64, 2048]
    """
    def __init__(
        self,
        out_chans      = 2,
        embed_dim      = 96,
        depths         = (2, 2, 2, 2),
        num_heads      = (3, 6, 12, 24),
        window_size    = (2, 8),
        mlp_ratio      = 4.,
        drop_rate      = 0.,
        attn_drop_rate = 0.,
        drop_path_rate = 0.1,
        use_v2         = True,
    ):
        super().__init__()
        assert len(depths) == len(num_heads) == 4
        dec_depths = depths[::-1]
        dec_heads  = num_heads[::-1]
        grids = [(2, 32), (4, 64), (8, 128), (16, 256)]
        dims  = [embed_dim * 8, embed_dim * 4, embed_dim * 2, embed_dim]

        dpr = _stoch_depth_rates(sum(dec_depths), drop_path_rate)
        dpr_idx = 0

        self.stages     = nn.ModuleList()
        self.expands    = nn.ModuleList()
        self.skip_projs = nn.ModuleList()

        for s in range(4):
            H_s, W_s = grids[s]
            d_s = dims[s]
            blocks = nn.ModuleList()
            for i in range(dec_depths[s]):
                sh = window_size[0] // 2 if i % 2 == 1 else 0
                sw = window_size[1] // 2 if i % 2 == 1 else 0
                blocks.append(SwinBlock(
                    dim=d_s, input_resolution=(H_s, W_s),
                    num_heads=dec_heads[s], window_size=window_size,
                    shift_size=(sh, sw), mlp_ratio=mlp_ratio,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[dpr_idx],
                    use_v2=use_v2,
                ))
                dpr_idx += 1
            self.stages.append(blocks)
            if s < 3:
                self.expands.append(_PatchExpanding(d_s))
                self.skip_projs.append(nn.Linear(d_s, d_s // 2, bias=False))

        self.norm         = nn.LayerNorm(dims[3])
        self.final_expand = _AsymmetricExpand(dims[3], patch_h=4, patch_w=8, out_chans=out_chans)

    def forward(self, z, skips=None):
        """z: [B,64,768], skips: list of 3 or None → [B, out_chans, 64, 2048]"""
        x = z
        H, W = 2, 32
        for s, blocks in enumerate(self.stages):
            for blk in blocks:
                x = blk(x)
            if s < 3:
                x = self.expands[s](x, H, W)
                H, W = H * 2, W * 2
                if skips is not None:
                    x = self.skip_projs[s](
                        torch.cat([x, skips[2 - s]], dim=-1)
                    )

        x = self.norm(x)
        return self.final_expand(x, H, W)


# ── Weight initialisation ─────────────────────────────────────────────────────

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
