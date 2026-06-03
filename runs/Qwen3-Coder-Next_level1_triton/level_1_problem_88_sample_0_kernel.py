import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def gelu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the starting offset for this block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for bounds checking
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Constants
    CONST_A = 0.044715
    CONST_B = tl.sqrt(2.0 / math.pi)
    
    # Compute x^3
    x_cubed = x * x * x
    
    # Compute inner expression: x + 0.044715 * x^3
    inner = x + CONST_A * x_cubed
    
    # Compute tanh argument: sqrt(2/pi) * (x + 0.044715 * x^3)
    tanh_arg = CONST_B * inner
    
    # Compute tanh using triton's tanh function
    tanh_val = tl.tanh(tanh_arg)
    
    # Compute final result: 0.5 * x * (1 + tanh_val)
    result = 0.5 * x * (1.0 + tanh_val)
    
    # Store output
    tl.store(output_ptr + offsets, result, mask=mask)


def triton_gelu(x: torch.Tensor):
    """
    Triton-based GELU activation function for FP32 tensors.
    """
    assert x.is_cuda, "Input tensor must be on CUDA device."
    x = x.contiguous()
    
    # Prepare output tensor
    output = torch.empty_like(x)
    
    # Number of elements
    n_elements = x.numel()
    
    # Set block size (tunable parameter)
    BLOCK_SIZE = 512
    
    # Calculate grid size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    gelu_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output


class ModelNew(nn.Module):
    """
    Optimized version of GELU using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x):
        return triton_gelu(x)