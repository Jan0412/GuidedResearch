import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the starting offset for this program
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values (FP32)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Apply ReLU: max(0, x)
    out = tl.maximum(x, 0.0)
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_relu(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the custom Triton ReLU kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor (preserves shape and dtype)
    out = torch.empty_like(x)
    
    # Total number of elements
    n_elements = x.numel()
    BLOCK_SIZE = 2048  # Tunable block size for memory-bound operations
    
    # Grid calculation: 1D grid over elements
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using a custom Triton kernel for ReLU activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies custom Triton-based ReLU activation to the input tensor.
        """
        return triton_relu(x)


batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed