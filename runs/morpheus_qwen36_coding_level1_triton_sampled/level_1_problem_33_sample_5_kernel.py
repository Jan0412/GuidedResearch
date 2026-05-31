import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def batch_norm_kernel(
    x_ptr, gamma_ptr, beta_ptr, running_mean_ptr, running_var_ptr, out_ptr,
    num_channels, num_elements_per_channel, eps, is_training,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= num_channels:
        return

    # Offsets for the current channel
    x_ptr += pid * num_elements_per_channel
    gamma_ptr += pid
    beta_ptr += pid
    running_mean_ptr += pid
    running_var_ptr += pid

    # Compute batch statistics if training
    if is_training:
        sum_val = 0.0
        sum_sq_val = 0.0
        n = num_elements_per_channel

        for i in range(0, n, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            sum_val += tl.sum(x, axis=0)
            sum_sq_val += tl.sum(x * x, axis=0)

        mean = sum_val / n
        var = sum_sq_val / n - mean * mean
        inv_std = 1.0 / tl.sqrt(var + eps)
    else:
        # Use running statistics for inference
        mean = tl.load(running_mean_ptr)
        var = tl.load(running_var_ptr)
        inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and scale
    n = num_elements_per_channel
    for i in range(0, n, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        out = (x - mean) * inv_std * gamma_ptr + beta_ptr
        tl.store(out_ptr + offsets, out, mask=mask)


def triton_batch_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                      running_mean: torch.Tensor, running_var: torch.Tensor,
                      eps: float, is_training: bool) -> torch.Tensor:
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda
    assert x.is_contiguous() and gamma.is_contiguous() and beta.is_contiguous()

    batch_size, num_channels, dim1, dim2 = x.shape
    num_elements_per_channel = batch_size * dim1 * dim2

    out = torch.empty_like(x)

    BLOCK_SIZE = 1024  # Tunable block size

    grid = (num_channels,)

    batch_norm_kernel[grid](
        x, gamma, beta, running_mean, running_var, out,
        num_channels, num_elements_per_channel, eps, is_training,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)
        self.eps = self.bn.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.bn.weight
        beta = self.bn.bias
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        is_training = self.bn.training

        return triton_batch_norm(x, gamma, beta, running_mean, running_var, self.eps, is_training)