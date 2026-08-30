import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import warnings

try:
    from ..SFS_MSDeformAttn.ops.modules import MSDeformAttn_for_sfs as _MSDeformAttnCUDA
    _CUDA_EXTENSION_ERROR = None
except ImportError as error:
    _MSDeformAttnCUDA = None
    _CUDA_EXTENSION_ERROR = error

def generate_structured_grid(n_heads, n_points, n_levels=1, base_radius=1.0, radius_step=1.0):
    """
    Initialization of spiral-aware sampling pattern.

    parameters:
    - n_heads: number of attention heads
    - n_points: number of sampling points of each head
    - n_levels: number of feature levels, default=1
    - base_radius: initial radius of sampling point
    - radius_step: radial step between consecutive points of each head

    return:
    - grid: Tensor, [n_heads, n_levels, n_points, 2]
    """
    offsets = []
    for h in range(n_heads):
        head_offsets = []
        delta_theta = 2 * math.pi * h / n_heads  # initial angle of each head
        for i in range(n_points):
            theta = 2 * math.pi * i / n_points + delta_theta
            r = base_radius + i * radius_step
            dx = r * math.cos(theta)
            dy = r * math.sin(theta)
            head_offsets.append([dx, dy])
        offsets.append(head_offsets)

    grid = torch.tensor(offsets, dtype=torch.float32)
    grid = grid.unsqueeze(1).repeat(1, n_levels, 1, 1)  # [n_heads, n_levels, n_points, 2]
    return grid


class MSDeformAttnTorchFallback(nn.Module):
    """Memory-bounded single-level deformable attention implemented with ``grid_sample``.

    The original fused CUDA extension samples each query independently without
    materializing the query-by-key feature grid.  This fallback follows the
    same sampling rule while processing queries in chunks, so it remains
    usable when the extension cannot be built for the active PyTorch ABI.
    """

    def __init__(
        self,
        d_model=256,
        n_levels=1,
        n_heads=8,
        n_points=4,
        ratio=1.0,
        query_chunk_size=4096,
    ):
        super().__init__()
        if n_levels != 1:
            raise ValueError("MSDeformAttnTorchFallback supports one feature level only")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if int(d_model * ratio) % n_heads != 0:
            raise ValueError("projected value dimension must be divisible by n_heads")
        if query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.ratio = ratio
        self.query_chunk_size = query_chunk_size
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, int(d_model * ratio))
        self.output_proj = nn.Linear(int(d_model * ratio), d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        sampling_offsets,
        input_padding_mask=None,
    ):
        del input_level_start_index
        batch_size, query_length, _ = query.shape
        _, input_length, _ = input_flatten.shape
        height, width = (int(size.item()) for size in input_spatial_shapes[0])
        if input_length != height * width:
            raise ValueError("input_flatten length does not match input_spatial_shapes")
        if reference_points.shape[-1] != 2:
            raise ValueError("only 2D reference points are supported")

        value_dim = int(self.ratio * self.d_model)
        head_dim = value_dim // self.n_heads
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.)
        value = value.view(batch_size, height, width, self.n_heads, head_dim)
        value = value.permute(0, 3, 4, 1, 2).contiguous()

        attention_weights = self.attention_weights(query).view(
            batch_size, query_length, self.n_heads, self.n_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1)
        sampling_offsets = sampling_offsets.reshape(
            batch_size, query_length, self.n_heads, self.n_levels, self.n_points, 2
        )
        offset_normalizer = input_spatial_shapes[:, [1, 0]].to(
            device=query.device, dtype=query.dtype
        )
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )
        sampling_grids = sampling_locations[:, :, :, 0, :, :] * 2.0 - 1.0

        head_outputs = []
        for head_index in range(self.n_heads):
            feature_map = value[:, head_index]
            query_chunks = []
            for start in range(0, query_length, self.query_chunk_size):
                end = min(start + self.query_chunk_size, query_length)
                chunk_length = end - start
                grid = sampling_grids[:, start:end, head_index].reshape(
                    batch_size, chunk_length * self.n_points, 1, 2
                )
                sampled = F.grid_sample(
                    feature_map,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                sampled = sampled.squeeze(-1).transpose(1, 2).reshape(
                    batch_size, chunk_length, self.n_points, head_dim
                )
                weights = attention_weights[:, start:end, head_index].unsqueeze(-1)
                query_chunks.append((sampled * weights).sum(dim=2))
            head_outputs.append(torch.cat(query_chunks, dim=1))

        output = torch.stack(head_outputs, dim=2).reshape(
            batch_size, query_length, value_dim
        )
        return self.output_proj(output)


if _MSDeformAttnCUDA is None:
    warnings.warn(
        "SFS_MSDeformAttn CUDA extension is unavailable; using the memory-bounded "
        f"PyTorch fallback instead ({_CUDA_EXTENSION_ERROR}).",
        RuntimeWarning,
    )
    MSDeformAttn_for_sfs = MSDeformAttnTorchFallback
else:
    MSDeformAttn_for_sfs = _MSDeformAttnCUDA

# SFS Module:
class SpiralAware_CrossDeformAttn2D(nn.Module):
    """
    Spiral-Aware MSDeformAttn.

    Inputs:
        - query_feat: [B, C, H1, W1], larger scale feature maps
        - key_feat:   [B, C, H2, W2], smaller scale feature maps
    Output:
        - out:   [B, C, H1, W1]
    """
    def __init__(self, dim, n_heads=8, n_points=4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_points = n_points

        self.query_Conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.key_Conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        self.shared_offsets_residual = nn.Parameter(torch.zeros(n_heads, n_points, 2))
        # generate uniform spiral-aware sampling pattern, and register it as buffer
        fixed_bias = generate_structured_grid(n_heads, n_points, n_levels=1, base_radius=1.0, radius_step=1.0)
        self.register_buffer("offset_base", fixed_bias.view(1, 1, n_heads, 1, n_points, 2))

        # LayerNorm on flattened features
        self.query_norm = nn.LayerNorm(dim)
        self.key_norm = nn.LayerNorm(dim)
        self.out_norm = nn.LayerNorm(dim)

        self.attn = MSDeformAttn_for_sfs(
            d_model=dim,
            n_levels=1,
            n_heads=n_heads,
            n_points=n_points
        )

    def forward(self, query_feat: Tensor, key_feat: Tensor) -> Tensor:
        B, C, H1, W1 = query_feat.shape
        _, _, H2, W2 = key_feat.shape

        query_feat = self.query_Conv(query_feat)
        key_feat = self.key_Conv(key_feat)

        offsets_residual = self.shared_offsets_residual

        shared_offsets = self.offset_base.view(self.n_heads, 1, self.n_points, 2) + offsets_residual.view(self.n_heads, 1, self.n_points, 2)
        offsets = shared_offsets.view(1, 1, self.n_heads, 1, self.n_points, 2).expand(B, H1 * W1, -1, -1, -1, -1)

        # flatten & transpose to [B, HW, C]
        query = query_feat.flatten(2).transpose(1, 2)       # [B, H1*W1, C]
        kv = key_feat.flatten(2).transpose(1, 2)            # [B, H2*W2, C]
        query = self.query_norm(query)
        kv = self.key_norm(kv)

        spatial_shapes = torch.tensor([[H2, W2]], device=key_feat.device, dtype=torch.long)
        level_start_index = torch.tensor([0], device=key_feat.device, dtype=torch.long)

        # generate normalized reference points for each query position, both shapes: [H1, W1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.5 / H1, 1 - 0.5 / H1, H1, device=query_feat.device),
            torch.linspace(0.5 / W1, 1 - 0.5 / W1, W1, device=query_feat.device),
            indexing='ij'
        )
        reference_points = torch.stack((grid_x, grid_y), -1)  # [H1, W1, 2]
        reference_points = reference_points.view(1, H1 * W1, 1, 2).repeat(B, 1, 1, 1)  # [B, H1*W1, 1, 2]

        # run deformable attention
        attn = self.attn(
            query=query,
            reference_points=reference_points,
            input_flatten=kv,
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start_index,
            sampling_offsets=offsets
        )  # [B, H1*W1, C]

        out = query + query * attn
        out = self.out_norm(out).transpose(1, 2).reshape(B, C, H1, W1)
        return out
