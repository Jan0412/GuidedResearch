import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hardtanh_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the starting offset for this program
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle elements that fall outside the tensor bounds
    mask = offsets < n_elements
    # Load input values, using 0.0 as fill value for masked positions
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Apply HardTanh: clamp values between -1.0 and 1.0
    out = tl.maximum(tl.minimum(x, 1.0), -1.0)
    # Store the result back to output tensor
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_hardtanh(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the Triton HardTanh kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Total number of elements
    n_elements = x.numel()
    # Tunable block size for memory coalescing and occupancy
    BLOCK_SIZE = 128
    
    # Grid calculation: number of blocks needed to cover all elements
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    hardtanh_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model using a custom Triton kernel for HardTanh activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_hardtanh(x)