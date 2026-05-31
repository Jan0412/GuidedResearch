import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def hardtanh_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    MIN_VAL: tl.constexpr,
    MAX_VAL: tl.constexpr
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply HardTanh: clamp values between MIN_VAL and MAX_VAL
    clamped = tl.where(x > MAX_VAL, MAX_VAL, x)
    clamped = tl.where(clamped < MIN_VAL, MIN_VAL, clamped)
    
    # Store the result
    tl.store(output_ptr + offsets, clamped, mask=mask)

def triton_hardtanh(x: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0):
    """
    Triton implementation of HardTanh activation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    output = torch.empty_like(x)
    
    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable block size
    
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    hardtanh_kernel[grid](
        x, 
        output, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE,
        MIN_VAL=min_val,
        MAX_VAL=max_val
    )
    return output

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for HardTanh activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies HardTanh activation using Triton kernel for better performance.
        
        Args:
            x (torch.Tensor): Input tensor of any shape.
            
        Returns:
            torch.Tensor: Output tensor with HardTanh applied, same shape as input.
        """
        return triton_hardtanh(x, min_val=-1., max_val=1.)