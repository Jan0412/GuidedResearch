import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr,
    out_ptr,
    num_rows,
    num_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * num_cols
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < num_cols

    # Load input values for the current row
    x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
    
    # Compute sum of squares
    x_sq = x * x
    sum_sq = tl.sum(x_sq)
    mean_sq = sum_sq / num_cols
    
    # Compute RMS and normalize
    rms = tl.sqrt(mean_sq + eps)
    out = x / rms
    
    # Store result
    tl.store(out_ptr + row_start + cols, out, mask=mask)


def triton_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Wrapper to launch the RMS Normalization Triton kernel.
    """
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)

    # Calculate N (total rows) and D (features per row)
    N = x.shape[0] * x.shape[2] * x.shape[3]
    D = x.shape[1]

    BLOCK_SIZE = 64
    grid = (N,)

    rms_norm_kernel[grid](x, out, N, D, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_rms_norm(x, self.eps)