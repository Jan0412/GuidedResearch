import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the program's starting position
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask for bounds checking
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sigmoid using the formula: 1 / (1 + exp(-x))
    # To improve numerical stability, we can use different formulations
    # but for FP32, the standard formula works well
    exp_neg_x = tl.exp(-x)
    out = 1.0 / (1.0 + exp_neg_x)
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Triton-based sigmoid implementation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements
    n_elements = x.numel()
    BLOCK_SIZE = 512  # Optimized block size for sigmoid
    
    # Determine grid size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    sigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for sigmoid activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Triton-optimized Sigmoid activation to the input tensor.
        """
        return triton_sigmoid(x)