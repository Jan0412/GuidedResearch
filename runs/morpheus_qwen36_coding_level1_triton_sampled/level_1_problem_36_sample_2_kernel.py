import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr, out_ptr,
    num_rows, row_size, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_size
    
    # Load input values for the current row
    x = tl.load(x_ptr + row_idx * row_size + offsets, mask=mask, other=0.0)
    
    # Compute sum of squares along the feature dimension
    x_sq = x * x
    sum_sq = tl.sum(x_sq)
    
    # Compute RMS
    rms = tl.sqrt(sum_sq / row_size + eps)
    
    # Normalize and store output
    out = x / rms
    tl.store(out_ptr + row_idx * row_size + offsets, out, mask=mask)


def triton_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Wrapper function to launch the custom Triton RMS Normalization kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    
    batch_size, features, dim1, dim2 = x.shape
    num_rows = batch_size * dim1 * dim2
    row_size = features
    
    # Block size should be a power of 2 and ideally >= row_size for optimal occupancy.
    # For features=64, 128 is a safe and efficient choice.
    BLOCK_SIZE = 128
    
    grid = (num_rows,)
    rms_norm_kernel[grid](x, out, num_rows, row_size, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_rms_norm(x, self.eps)