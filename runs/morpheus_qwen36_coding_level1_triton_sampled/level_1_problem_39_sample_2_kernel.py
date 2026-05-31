import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l2_norm_kernel(
    x_ptr,
    y_ptr,
    dim,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for L2 normalization along dimension 1.
    Each program handles one row of the input tensor.
    """
    row_idx = tl.program_id(0)
    
    # Pointer to the start of the current row
    x_ptr += row_idx * dim
    y_ptr += row_idx * dim
    
    # Create offsets for the current block
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < dim
    
    # Load the row values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sum of squares
    x_sq = x * x
    sum_sq = tl.sum(x_sq)
    
    # Compute L2 norm with epsilon for numerical stability
    norm = tl.sqrt(sum_sq + 1e-8)
    
    # Normalize the row
    y = x / norm
    
    # Store the result
    tl.store(y_ptr + offsets, y, mask=mask)


def triton_l2_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the L2 normalization Triton kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, dim = x.shape
    
    # Prepare output tensor
    y = torch.empty_like(x)
    
    # Choose block size based on dimension
    # Use power of 2 that covers dim for efficiency
    BLOCK_SIZE = 1
    while BLOCK_SIZE < dim:
        BLOCK_SIZE *= 2
    
    # Grid: one program per batch element
    grid = (batch_size,)
    
    # Launch kernel
    l2_norm_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        dim=dim,
        batch_size=batch_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized model performing L2 normalization using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization to the input tensor using Triton.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).
            
        Returns:
            torch.Tensor: Output tensor with L2 normalization applied.
        """
        return triton_l2_norm(x)