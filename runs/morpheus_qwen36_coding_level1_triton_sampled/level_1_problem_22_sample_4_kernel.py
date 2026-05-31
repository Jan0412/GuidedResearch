import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def tanh_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate block start based on program ID
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create offset range for the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask to handle boundary elements
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute tanh using a numerically stable formula
    # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    # To avoid overflow for large |x|, we use:
    # if x > 0: y = exp(-2x), tanh = (1 - y) / (1 + y)
    # if x <= 0: y = exp(2x), tanh = (y - 1) / (y + 1)
    
    pos_mask = x > 0.0
    y = tl.where(pos_mask, tl.exp(-2.0 * x), tl.exp(2.0 * x))
    tanh_val = tl.where(pos_mask, (1.0 - y) / (1.0 + y), (y - 1.0) / (y + 1.0))
    
    # Store result
    tl.store(out_ptr + offsets, tanh_val, mask=mask)


def triton_tanh(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper to launch the custom Triton tanh kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    
    # Tunable block size
    BLOCK_SIZE = 1024
    
    # Grid calculation
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    tanh_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for Tanh activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_tanh(x)


batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []