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
    
    # For each element in the reduced dimension
    for i in range(0, dim_size, BLOCK_SIZE):
        # Create offsets for current block
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim_size
        
        # Load input values
        x = tl.load(input_ptr + input_base + offsets * stride_input_dim, mask=mask, other=float('inf'))
        
        # Find minimum in this block
        local_min = tl.min(x)
        
        # Store the minimum value
        tl.atomic_min(output_ptr + output_base + (offsets // BLOCK_SIZE) * stride_output_dim, local_min)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle the case where we reduce over the last dimension
        if self.dim == len(x.shape) - 1:
            return self._min_reduce_last_dim(x)
        else:
            # For other dimensions, use PyTorch's built-in function
            return torch.min(x, dim=self.dim)[0]

    def _min_reduce_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        # Create output tensor with correct shape
        output_shape = list(x.shape)
        output_shape.pop(self.dim)
        output = torch.empty(output_shape, dtype=torch.float32, device=x.device)
        
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Get dimensions
        batch_size = x.shape[0]
        dim_size = x.shape[-1]
        n_elements = x.numel() // dim_size
        
        # Calculate strides
        stride_input_batch = x.stride(0) if len(x.shape) > 1 else 1
        stride_input_dim = x.stride(-1) if len(x.shape) > 1 else 1
        stride_output_batch = output.stride(0) if len(output.shape) > 0 else 1
        stride_output_dim = output.stride(-1) if len(output.shape) > 1 else 1
        
        # Set up kernel launch parameters
        BLOCK_SIZE = 128
        grid_size = batch_size
        
        # Launch kernel
        min_reduction_kernel[grid_size](
            x,
            output,
            n_elements,
            dim_size,
            stride_input_batch,
            stride_input_dim,
            stride_output_batch,
            stride_output_dim,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output