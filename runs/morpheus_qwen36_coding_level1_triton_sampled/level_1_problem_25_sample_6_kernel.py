import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def swish_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Create block offsets
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute Swish: x * sigmoid(x)
    # Triton's tl.sigmoid is numerically stable and efficient
    out = x * tl.sigmoid(x)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_swish(x: torch.Tensor) -> torch.Tensor:
    """
    Applies Swish activation using a custom Triton kernel.
    Swish(x) = x * sigmoid(x)
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable block size for optimal occupancy
    
    # Grid calculation
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    swish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a Swish activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Swish activation to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Swish applied, same shape as input.
        """
        return triton_swish(x)


batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim, dtype=torch.float32, device='cuda')
    return [x]

def get_init_inputs():
    return []