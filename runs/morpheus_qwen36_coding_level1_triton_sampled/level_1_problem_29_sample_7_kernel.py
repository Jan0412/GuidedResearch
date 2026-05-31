import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softplus_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate block start and offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Numerically stable Softplus implementation:
    # softplus(x) = max(x, 0) + log(1 + exp(-abs(x)))
    # This formulation avoids overflow for large positive x and underflow issues.
    abs_x = tl.abs(x)
    exp_neg_abs = tl.math.exp(-abs_x)
    log_term = tl.math.log1p(exp_neg_abs)
    out = tl.maximum(x, 0.0) + log_term
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_softplus(x: torch.Tensor) -> torch.Tensor:
    """
    Wraps the Triton softplus kernel for PyTorch integration.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    
    # Tunable block size
    BLOCK_SIZE = 1024
    
    # Calculate grid dimensions
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    softplus_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_softplus(x)

def get_inputs():
    batch_size = 4096
    dim = 393216
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []