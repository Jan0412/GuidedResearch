import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def max_reduction_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    dim_size,
    stride_input_batch,
    stride_input_dim,
    stride_output_batch,
    stride_output_dim,
    BLOCK_SIZE: tl.constexpr,
    DIM_SIZE: tl.constexpr
):
    # Get the batch index
    batch_idx = tl.program_id(0)
    
    # Calculate the starting position for this batch
    input_base = batch_idx * stride_input_batch
    output_base = batch_idx * stride_output_batch
    
    # For each element in the reduced dimension
    for i in range(0, DIM_SIZE, BLOCK_SIZE):
        # Calculate offsets
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < DIM_SIZE
        
        # Load data
        input_offsets = input_base + offsets * stride_input_dim
        x = tl.load(input_ptr + input_offsets, mask=mask, other=-float('inf'))
        
        # Compute max along the dimension
        max_val = tl.max(x)
        
        # Store result
        output_offset = output_base + i * stride_output_dim
        tl.store(output_ptr + output_offset, max_val, mask=mask)

def triton_max_reduction(x: torch.Tensor, dim: int):
    """
    Triton implementation of max reduction along a specified dimension.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Get dimensions
    shape = x.shape
    batch_size = 1
    remaining_dims = []
    
    # Calculate batch size (product of all dimensions except the target dim)
    for i, s in enumerate(shape):
        if i != dim:
            batch_size *= s
        else:
            dim_size = s
    
    # Calculate strides
    stride_input = x.stride()
    stride_output = []
    
    # Create output tensor with correct shape
    output_shape = list(shape)
    output_shape.pop(dim)
    output = torch.empty(output_shape, dtype=torch.float32, device=x.device)
    
    # Calculate output strides
    output_stride = output.stride()
    
    # Prepare parameters for kernel
    n_elements = x.numel()
    BLOCK_SIZE = 128
    DIM_SIZE = dim_size
    
    # Grid configuration
    grid = (batch_size,)
    
    # Launch kernel
    max_reduction_kernel[grid](
        x.data_ptr(),
        output.data_ptr(),
        n_elements,
        DIM_SIZE,
        stride_input[0] if len(stride_input) > 0 else 1,
        stride_input[dim] if len(stride_input) > dim else 1,
        output_stride[0] if len(output_stride) > 0 else 1,
        1,
        BLOCK_SIZE=BLOCK_SIZE,
        DIM_SIZE=DIM_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for max reduction.
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
        Applies Max reduction over the specified dimension using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Max reduction over the specified dimension.
        """
        return triton_max_reduction(x, self.dim)