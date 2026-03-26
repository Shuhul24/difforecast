import torch
from torch import Tensor, nn
from dataclasses import dataclass

from models.modules.dit_modules.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)
from models.modules.dit_modules.lora import LinearLora, replace_linear_with_lora

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

@dataclass
class FluxParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool
    drop_path_rate: float = 0.0  # peak stochastic-depth rate (linearly scaled per block)
    # Hidden dimension for the pose/vec MLP (vector_in).
    # When vec_in_dim >> hidden_size a single linear crushing e.g. 3072→512
    # discards most of the pose information.  Setting vec_hidden_dim between
    # vec_in_dim and hidden_size (e.g. vec_in_dim // 3 ≈ 1024 for 3072→1024→512)
    # gives a more gradual compression with negligible extra parameters.
    # 0 means use hidden_size (original single-bottleneck behaviour).
    vec_hidden_dim: int = 0


class FluxDiT(nn.Module):
    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        # Pose/vec projection: use a wide hidden layer when vec_hidden_dim > 0
        # to avoid a single crushing bottleneck (e.g. 3072 → 512 in one step).
        # With vec_hidden_dim = n_embd = 1024: 3072 → 1024 → 512.
        vec_hid = params.vec_hidden_dim if params.vec_hidden_dim > 0 else self.hidden_size
        self.vector_in = MLPEmbedder(params.vec_in_dim, vec_hid, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.cond_in = nn.Linear(params.context_in_dim, self.hidden_size)

        # Linearly increasing drop-path rates: 0 → drop_path_rate across all
        # double + single blocks so shallow blocks get full gradient flow while
        # deeper blocks receive stronger stochastic-depth regularization.
        total_blocks = params.depth + params.depth_single_blocks
        dpr = [
            x.item()
            for x in torch.linspace(0, params.drop_path_rate, total_blocks)
        ]

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    drop_path=dpr[i],
                )
                for i in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    drop_path=dpr[params.depth + i],
                )
                for i in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        cond: Tensor,
        cond_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:
        if img.ndim != 3 or cond.ndim != 3:
            raise ValueError("Input img and cond tensors must have 3 dimensions.")

        # Cast inputs to model dtype to handle float32/bfloat16 mismatches
        # that arise when the VAE encoder runs outside the autocast context.
        dtype = self.img_in.weight.dtype
        img = img.to(dtype)
        cond = cond.to(dtype)
        y = y.to(dtype)
        timesteps = timesteps.to(dtype)

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        cond = self.cond_in(cond)

        ids = torch.cat((cond_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, cond = block(img=img, cond=cond, vec=vec, pe=pe)

        img = torch.cat((cond, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, cond.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img
    
    def training_losses(self, 
                        img: Tensor,     # (B, L, C)
                        img_ids: Tensor,
                        cond: Tensor,
                        cond_ids: Tensor,
                        t: Tensor,
                        y: Tensor,
                        guidance: Tensor | None = None,
                        noise: Tensor | None = None,
                        return_predict=False, 
                    ) -> Tensor:
        if noise is None:
            noise = torch.randn_like(img)
        terms = {}
        
        x_t = t * img + (1. - t) * noise
        target = img - noise
        pred = self(img=x_t, img_ids=img_ids, cond=cond, cond_ids=cond_ids, timesteps=t.reshape(-1), y=y, guidance=guidance)
        assert pred.shape == target.shape == img.shape
        target = target.to(pred.dtype)
        x_t = x_t.to(pred.dtype)
        predict = x_t + pred * (1. - t.to(pred.dtype))
        terms["mse"] = mean_flat((target - pred) ** 2)
        
        terms["loss"] = terms["mse"].mean()
        if return_predict:
            terms["predict"] = predict
        else:
            terms["predict"] = None
        return terms
        
    def sample(self,
                img: Tensor,
                img_ids: Tensor,
                cond: Tensor,
                cond_ids: Tensor,
                vec: Tensor,
                timesteps: list[float],
            ):
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            pred = self(
                img=img,
                img_ids=img_ids,
                cond=cond,
                cond_ids=cond_ids,
                y=vec,
                timesteps=t_vec,
            )
            img = img + (t_prev - t_curr) * pred
        return img


class FluxLoraWrapper(FluxDiT):
    def __init__(
        self,
        lora_rank: int = 128,
        lora_scale: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.lora_rank = lora_rank

        replace_linear_with_lora(
            self,
            max_rank=lora_rank,
            scale=lora_scale,
        )

    def set_lora_scale(self, scale: float) -> None:
        for module in self.modules():
            if isinstance(module, LinearLora):
                module.set_scale(scale=scale)
