import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def max_reduction_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    stride_input_outer,
    stride_input_inner,
    stride_output,
    BLOCK_SIZE: tl.constexpr,
    DIM_SIZE: tl.constexpr
):
    # Get the block index
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask to avoid out-of-bounds access
    mask = offsets < n_elements
    
    # Calculate the outer dimension index for this block
    outer_idx = offsets // DIM_SIZE
    inner_idx = offsets % DIM_SIZE
    
    # Load input data
    input_data = tl.load(input_ptr + outer_idx * stride_input_outer + inner_idx * stride_input_inner, mask=mask, other=-float('inf'))
    
    # Initialize max value
    max_val = tl.full([BLOCK_SIZE], -float('inf'), dtype=tl.float32)
    
    # Reduce along the inner dimension
    for i in range(DIM_SIZE):
        current_idx = outer_idx * stride_input_outer + i * stride_input_inner
        current_val = tl.load(input_ptr + current_idx, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, current_val)
    
    # Store the result
    tl.store(output_ptr + outer_idx * stride_output, max_val, mask=mask)

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for Max reduction over a specific dimension.
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
        Applies Max reduction over the specified dimension to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Max reduction over the specified dimension.
        """
        if self.dim == 0:
            # Reduce along dimension 0
            output_shape = list(x.shape)
            output_shape.pop(0)
            output = torch.empty(output_shape, dtype=torch.float32, device=x.device)
            
            # Calculate strides
            stride_input_outer = x.stride(0)
            stride_input_inner = x.stride(1) if len(x.shape) > 1 else 1
            stride_output = 1
            
            # Get number of elements in output
            n_elements = output.numel()
            
            # Determine block size and grid
            BLOCK_SIZE = 1024
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            
            # Launch kernel
            max_reduction_kernel[grid](
                x,
                output,
                n_elements,
                stride_input_outer,
                stride_input_inner,
                stride_output,
                BLOCK_SIZE=BLOCK_SIZE,
                DIM_SIZE=x.shape[0]
            )
            
        elif self.dim == 1:
            # Reduce along dimension 1
            output_shape = list(x.shape)
            output_shape.pop(1)
            output = torch.empty(output_shape, dtype=torch.float32, device=x.device)
            
            # Calculate strides
            stride_input_outer = x.stride(1) if len(x.shape) > 1 else 1
            stride_input_inner = x.stride(0)
            stride_output = 1
            
            # Get number of elements in output
            n_elements = output.numel()
            
            # Determine block size and grid
            BLOCK_SIZE = 1024
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            
            # Launch kernel
            max_reduction_kernel[grid](
                x,
                output,
                n_elements,
                stride_input_outer,
                stride_input_inner,
                stride_output,
                BLOCK_SIZE=BLOCK_SIZE,
                DIM_SIZE=x.shape[1]
            )
            
        elif self.dim == 2:
            # Reduce along dimension 2
            output_shape = list(x.shape)
            output_shape.pop(2)
            output = torch.empty(output_shape, dtype=torch.float32, device=x.device)
            
            # Calculate strides
            stride_input_outer = x.stride(2) if len(x.shape) > 2 else 1
            stride_input_inner = x.stride(0) if len(x.shape) > 0 else 1
            stride_output = 1
            
            # Get number of elements in output
            n_elements = output.numel()
            
            # Determine block size and grid
            BLOCK_SIZE = 1024
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            
            # Launch kernel
            max_reduction_kernel[grid](
                x,
                output,
                n_elements,
                stride_input_outer,
                stride_input_inner,
                stride_output,
                BLOCK_SIZE=BLOCK_SIZE,
                DIM_SIZE=x.shape[2]
            )
        else:
            # For other dimensions, fall back to PyTorch implementation
            return torch.max(x, dim=self.dim)[0]
            
        return output