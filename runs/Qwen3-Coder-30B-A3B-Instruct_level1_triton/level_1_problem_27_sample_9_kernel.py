import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def selu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    alpha: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply SELU: scale * (alpha * exp(x) - alpha) if x <= 0, else scale * x
    condition = x <= 0.0
    exp_x = tl.exp(x)
    selu_val = tl.where(condition, scale * (alpha * exp_x - alpha), scale * x)
    
    # Store the result
    tl.store(output_ptr + offsets, selu_val, mask=mask)

def triton_selu(x: torch.Tensor, alpha: float = 1.6732632423543772848170429916717, scale: float = 1.0507009873554804934193349852946):
    """
    Triton implementation of SELU activation function.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    output = torch.empty_like(x)
    
    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tuned block size for performance
    
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    selu_kernel[grid](x, output, n_elements, alpha, scale, BLOCK_SIZE=BLOCK_SIZE)
    return output

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for SELU activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies SELU activation using Triton kernel for better performance.
        
        Args:
            x (torch.Tensor): Input tensor of any shape.
            
        Returns:
            torch.Tensor: Output tensor with SELU applied, same shape as input.
        """
        return triton_selu(x)