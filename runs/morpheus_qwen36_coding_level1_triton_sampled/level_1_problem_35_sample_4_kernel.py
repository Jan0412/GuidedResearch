import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def group_norm_kernel(
    x_ptr,
    out_ptr,
    gamma_ptr,
   beta_ptr,
    num_elements_per_group,
    num_spatial_elements,
    BLOCK_SIZE: tl.constexpr,
):
    local_offsets = tl.arange(0, BLOCK_SIZE)
    mask = local_offsets < num_elements_per_group

    sum_x = tl.zeros((1,), dtype=tl.float32)
    sum_x2 = tl.zeros((1,), dtype=tl.float32)

    num_tiles = (num_elements_per_group + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Pass 1: Compute mean and variance
    for tile in tl.range(num_tiles):
        offset = tile * BLOCK_SIZE
        x = tl.load(x_ptr + offset + local_offsets, mask=mask, other=0.0)
        sum_x += tl.sum(x)
        sum_x2 += tl.sum(x * x)

    mean = sum_x / num_elements_per_group
    var = sum_x2 / num_elements_per_group - mean * mean
    std = tl.sqrt(var + 1e-5)

    # Store stats in shared memory for Pass 2
    mean_shared = tl.static_shared_memory(1, tl.float32)
    var_shared = tl.static_shared_memory(1, tl.float32)
    mean_shared[0] = mean
    var_shared[0] = var
    tl.barrier()

    mean = mean_shared[0]
    var = var_shared[0]
    std = tl.sqrt(var + 1e-5)

    # Pass 2: Normalize, scale, and shift
    for tile in tl.range(num_tiles):
        offset = tile * BLOCK_SIZE
        x = tl.load(x_ptr + offset + local_offsets, mask=mask, other=0.0)
        x_hat = (x - mean) / std
        channel_idx = (offset + local_offsets) // num_spatial_elements
        gamma = tl.load(gamma_ptr + channel_idx, mask=mask)
        beta = tl.load(beta_ptr + channel_idx, mask=mask)
        out = x_hat * gamma + beta
        tl.store(out_ptr + offset + local_offsets, out, mask=mask)


def triton_group_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, num_groups: int) -> torch.Tensor:
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda
    x = x.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    batch_size, num_features, dim1, dim2 = x.shape
    num_channels_per_group = num_features // num_groups
    num_spatial_elements = dim1 * dim2
    num_elements_per_group = num_channels_per_group * num_spatial_elements

    out = torch.empty_like(x)

    BLOCK_SIZE = 256
    num_groups_per_sample = num_groups
    num_samples = batch_size

    grid = (batch_size * num_groups,)

    for b in range(batch_size):
        for g in range(num_groups):
            group_offset = b * num_features * dim1 * dim2 + g * num_channels_per_group * dim1 * dim2
            gamma_offset = g * num_channels_per_group
            beta_offset = g * num_channels_per_group

            group_norm_kernel[grid](
                x_ptr=x + group_offset,
                out_ptr=out + group_offset,
                gamma_ptr=gamma + gamma_offset,
                beta_ptr=beta + beta_offset,
                num_elements_per_group=num_elements_per_group,
                num_spatial_elements=num_spatial_elements,
                BLOCK_SIZE=BLOCK_SIZE,
            )

    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_group_norm(x, self.gamma, self.beta, self.num_groups)