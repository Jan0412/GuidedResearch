import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def log_softmax_kernel(
    x_ptr,
    out_ptr,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to compute log_softmax along the last dimension.
    Uses online log-sum-exp reduction for numerical stability and efficiency.
    """
    pid = tl.program_id(0)
    row_offset = pid * dim

    # Initialize max and sum for the row
    m = tl.full((1,), -float('inf'), dtype=tl.float32)
    s = tl.full((1,), 0.0, dtype=tl.float32)

    # First pass: compute max and sum of exps
    for start in range(0, dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(x_ptr + row_offset + offsets, mask=mask, other=0.0)
        
        # Local max and sum for the chunk
        l_m = tl.max(x, axis=0)
        # Shift by local max for stability within chunk
        l_s = tl.sum(tl.exp(x - l_m), axis=0)
        
        # Online update
        new_m = tl.maximum(m, l_m)
        s = tl.exp(m - new_m) * s + tl.exp(l_m - new_m) * l_s
        m = new_m

    # Compute log-sum-exp
    log_sum_exp = tl.log(s) + m

    # Second pass: compute output
    for start in range(0, dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(x_ptr + row_offset + offsets, mask=mask, other=0.0)
        out = x - log_sum_exp
        tl.store(out_ptr + row_offset + offsets, out, mask=mask)


def triton_log_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper to launch the Triton log_softmax kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size = x.shape[0]
    dim = x.shape[1]
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 1024  # Tunable parameter
    
    # Grid: one block per row
    grid = (batch_size,)
    
    log_softmax_kernel[grid](x, out, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_log_softmax(x)