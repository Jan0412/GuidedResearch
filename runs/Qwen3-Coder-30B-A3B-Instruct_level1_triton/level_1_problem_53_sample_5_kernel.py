import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def min_reduction_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    dim_size,
    stride_input_batch,
    stride_input_dim,
    stride_output_batch,
    stride_output_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch index
    batch_idx = tl.program_id(0)
    
    # Calculate the starting position for this batch
    input_base = batch_idx * stride_input_batch
    output_base = batch_idx * stride_output_batch
    
    # Process each element along the reduced dimension
    for i in range(0, dim_size, BLOCK_SIZE):
        # Create offsets
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim_size
        
        # Load input values
        x = tl.load(input_ptr + input_base + offsets * stride_input_dim, mask=mask, other=float('inf'))
        
        # Find minimum value in this block
        min_val = tl.min(x)
        
        # Store the result
        if i == 0:
            # Initialize output with first minimum
            tl.store(output_ptr + output_base + 0, min_val)
        else:
            # Update existing output with smaller values
            current_min = tl.load(output_ptr + output_base + 0)
            updated_min = tl.minimum(current_min, min_val)
            tl.store(output_ptr + output_base + 0, updated_min)

def triton_min_reduction(x: torch.Tensor, dim: int):
    """
    Custom Triton implementation of min reduction over a specific dimension.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Calculate output shape
    output_shape = list(x.shape)
    output_shape.pop(dim)
    
    # Prepare output tensor
    out = torch.empty(output_shape, dtype=torch.float32, device=x.device)
    
    # Get dimensions
    batch_size = x.shape[0]
    dim_size = x.shape[dim]
    
    # Calculate strides
    stride_input_batch = x.stride(0)
    stride_input_dim = x.stride(dim)
    stride_output_batch = out.stride(0) if len(out.shape) > 0 else 1
    stride_output_dim = out.stride(1) if len(out.shape) > 1 else 1
    
    # Number of elements in output
    n_elements = out.numel()
    
    # Block size
    BLOCK_SIZE = 1024
    
    # Grid size
    grid = (batch_size,)
    
    # Launch kernel
    min_reduction_kernel[grid](
        x,
        out,
        n_elements,
        dim_size,
        stride_input_batch,
        stride_input_dim,
        stride_output_batch,
        stride_output_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for min reduction.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies min reduction over the specified dimension using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after min reduction over the specified dimension.
        """
        return triton_min_reduction(x, self.dim)