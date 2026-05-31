import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate offsets for the current block
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create mask to handle elements beyond tensor size
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Precomputed constants for maximum performance
    sqrt_2_pi = 0.7978845608028654
    coef = 0.044715
    
    x_cubed = x * x * x
    inner = x + coef * x_cubed
    tanh_val = tl.math.tanh(sqrt_2_pi * inner)
    out = 0.5 * x * (1.0 + tanh_val)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_gelu(x: torch.Tensor):
    """
    Wrapper function to launch the Triton GELU kernel.
    """
    assert x.is_cuda and x.dtype == torch.float32, "Input must be a contiguous FP32 CUDA tensor."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable block size
    
    # Grid calculation: number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # Replace PyTorch GELU with custom Triton kernel
        return triton_gelu(x)