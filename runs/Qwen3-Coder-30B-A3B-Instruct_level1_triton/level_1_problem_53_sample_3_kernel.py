import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def min_reduction_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    stride_input_row,
    stride_input_col,
    stride_output_row,
    stride_output_col,
    reduction_dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index for this program
    row_idx = tl.program_id(0)
    
    # Calculate the starting position for this row
    input_row_start = row_idx * stride_input_row
    
    # Initialize minimum value
    min_val = tl.full([1], float('inf'), dtype=tl.float32)
    
    # Process elements in blocks
    for i in range(0, reduction_dim_size, BLOCK_SIZE):
        # Calculate offsets for current block
        offsets = i + tl.arange(0, BLOCK_SIZE)
        
        # Create mask to avoid going out of bounds
        mask = offsets < reduction_dim_size
        
        # Load elements from input tensor
        input_offsets = input_row_start + offsets * stride_input_col
        x = tl.load(input_ptr + input_offsets, mask=mask, other=float('inf'))
        
        # Compute minimum
        min_val = tl.minimum(min_val, x)
    
    # Store the result
    output_offset = row_idx * stride_output_row
    tl.store(output_ptr + output_offset, min_val)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 0:
            # Reduce along dimension 0
            output_shape = list(x.shape)
            output_shape.pop(0)
            output = torch.empty(output_shape, dtype=torch.float32, device=x.device)
            
            # Calculate strides
            stride_input_row = x.stride(0)
            stride_input_col = x.stride(1) if len(x.shape) > 1 else 1
            stride_output_row = 1
            stride_output_col = 1
            
            # Prepare for kernel launch
            n_elements = x.numel()
            reduction_dim_size = x.shape[0]
            BLOCK_SIZE = 128
            
            # Grid calculation
            grid = lambda meta: (x.shape[1] * x.shape[2] if len(x.shape) > 2 else 1,)
            
            # Launch kernel
            min_reduction_kernel[grid](
                x, output, n_elements, 
                stride_input_row, stride_input_col,
                stride_output_row, stride_output_col,
                reduction_dim_size, BLOCK_SIZE=BLOCK_SIZE
            )
            
        elif self.dim == 1:
            # Reduce along dimension 1
            output_shape = list(x.shape)
            output_shape.pop(1)
            output = torch.empty(output_shape, dtype=torch.float32, device=x.device)
            
            # Calculate strides
            stride_input_row = x.stride(0) if len(x.shape) > 1 else 1
            stride_input_col = x.stride(1) if len(x.shape) > 1 else 1
            stride_output_row = 1
            stride_output_col = 1
            
            # Prepare for kernel launch
            n_elements = x.numel()
            reduction_dim_size = x.shape[1]
            BLOCK_SIZE = 128
            
            # Grid calculation
            grid = lambda meta: (x.shape[0] * x.shape[2] if len(x.shape) > 2 else 1,)
            
            # Launch kernel
            min_reduction_kernel[grid](
                x, output, n_elements, 
                stride_input_row, stride_input_col,
                stride_output_row, stride_output_col,
                reduction_dim_size, BLOCK_SIZE=BLOCK_SIZE
            )
            
        elif self.dim == 2:
            # Reduce along dimension 2
            output_shape = list(x.shape)
            output_shape.pop(2)
            output = torch.empty(output_shape, dtype=torch.float32, device=x.device)
            
            # Calculate strides
            stride_input_row = x.stride(0) if len(x.shape) > 2 else 1
            stride_input_col = x.stride(2) if len(x.shape) > 2 else 1
            stride_output_row = 1
            stride_output_col = 1
            
            # Prepare for kernel launch
            n_elements = x.numel()
            reduction_dim_size = x.shape[2]
            BLOCK_SIZE = 128
            
            # Grid calculation
            grid = lambda meta: (x.shape[0] * x.shape[1],)
            
            # Launch kernel
            min_reduction_kernel[grid](
                x, output, n_elements, 
                stride_input_row, stride_input_col,
                stride_output_row, stride_output_col,
                reduction_dim_size, BLOCK_SIZE=BLOCK_SIZE
            )
            
        else:
            # For other dimensions, use PyTorch's implementation
            return torch.min(x, dim=self.dim)[0]
            
        return output