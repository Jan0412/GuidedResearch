import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr, out_ptr,
    batch_size, features, dim1, dim2,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    slice_size = dim1 * dim2
    
    # Compute indices for (b, d1, d2)
    b = pid // slice_size
    offset_in_slice = pid % slice_size
    d1 = offset_in_slice // dim2
    d2 = offset_in_slice % dim2
    
    # Base offset for the first feature element of this (b, d1, d2)
    base_offset = b * features * slice_size + d1 * dim2 + d2
    
    # Offsets along the features dimension
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < features
    
    # Load input values
    x = tl.load(x_ptr + base_offset + offsets * slice_size, mask=mask, other=0.0)
    
    # Compute sum of squares
    sum_sq = tl.sum(x * x, axis=0)
    
    # Compute RMS
    rms = tl.sqrt(sum_sq / features + eps)
    
    # Normalize
    out = x / rms
    
    # Store result
    tl.store(out_ptr + base_offset + offsets * slice_size, out, mask=mask)


def triton_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    
    batch_size, features, dim1, dim2 = x.shape
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 64  # Matches features dimension
    
    grid = (batch_size * dim1 * dim2,)
    
    rms_norm_kernel[grid](
        x, out,
        batch_size, features, dim1, dim2,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_rms_norm(x, self.eps)