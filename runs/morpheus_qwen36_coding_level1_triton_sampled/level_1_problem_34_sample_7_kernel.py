import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def instance_norm_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    num_features,
    dim1, dim2,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // num_features
    c = pid % num_features
    
    w = tl.load(weight_ptr + c)
    bias = tl.load(bias_ptr + c)
    
    x_ptr_bc = x_ptr + b * num_features * dim1 * dim2 + c * dim1 * dim2
    out_ptr_bc = out_ptr + b * num_features * dim1 * dim2 + c * dim1 * dim2
    
    num_elements = dim1 * dim2
    
    # Pass 1: Compute mean
    sum_val = 0.0
    start = 0
    while start < num_elements:
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        x = tl.load(x_ptr_bc + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(x)
        start += BLOCK_SIZE
    mean = sum_val / num_elements
    
    # Pass 2: Compute variance, normalize, apply affine
    start = 0
    while start < num_elements:
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        x = tl.load(x_ptr_bc + offsets, mask=mask, other=0.0)
        diff = x - mean
        var_sum = tl.sum(diff * diff)
        var = var_sum / num_elements
        std = tl.sqrt(var + eps)
        y = diff / std
        y = y * w + bias
        tl.store(out_ptr_bc + offsets, y, mask=mask)
        start += BLOCK_SIZE


def triton_instance_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    
    batch_size, num_features, dim1, dim2 = x.shape
    
    grid = (batch_size * num_features,)
    
    instance_norm_kernel[grid](
        x, out, weight, bias,
        num_features, dim1, dim2, eps,
        BLOCK_SIZE=1024
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.inorm = nn.InstanceNorm2d(num_features=num_features, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_instance_norm(x, self.inorm.weight, self.inorm.bias, self.inorm.eps)