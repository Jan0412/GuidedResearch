import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def leaky_relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    negative_slope,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for LeakyReLU activation.
    
    Args:
        x_ptr: Pointer to input tensor.
        out_ptr: Pointer to output tensor.
        n_elements: Total number of elements.
        negative_slope: The negative slope parameter.
        BLOCK_SIZE: Number of elements per block (constexpr).
    """
    # Calculate block start index
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Create offset vector for the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to handle boundary elements
    mask = offsets < n_elements
    
    # Load input data with masking
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply LeakyReLU: y = x if x > 0 else x * negative_slope
    out = tl.where(x > 0, x, x * negative_slope)
    
    # Store result with masking
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_leaky_relu(x: torch.Tensor, negative_slope: float) -> torch.Tensor:
    """
    Wrapper function to launch the Triton LeakyReLU kernel.
    
    Args:
        x: Input tensor (must be contiguous and on CUDA).
        negative_slope: The negative slope parameter.
        
    Returns:
        Output tensor with LeakyReLU applied.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Total number of elements
    n_elements = x.numel()
    
    # Tunable block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the kernel
    leaky_relu_kernel[grid](
        x, out, n_elements, negative_slope, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for LeakyReLU activation.
    """
    def __init__(self, negative_slope: float = 0.01):
        """
        Initializes the ModelNew module.
        
        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LeakyReLU activation using custom Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor with LeakyReLU applied.
        """
        return triton_leaky_relu(x, self.negative_slope)