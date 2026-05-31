import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def selu_kernel(
    x_ptr,  # Pointer to input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask for bounds checking
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # SELU implementation: lambda * (x if x > 0 else alpha * (exp(x) - 1))
    # Constants for SELU
    LAMBDA = 1.0507009873554804934193349852946
    ALPHA = 1.6732632423543772848170429916717
    
    # Compute SELU
    out = tl.where(
        x > 0,
        LAMBDA * x,
        LAMBDA * ALPHA * (tl.exp(x) - 1)
    )
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_selu(x: torch.Tensor):
    """
    Triton-based SELU activation function.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter
    
    # Grid definition
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    selu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for SELU activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies SELU activation using Triton kernel to the input tensor.
        """
        return triton_selu(x)