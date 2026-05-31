import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def argmax_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    dim_size,
    stride_x_batch,
    stride_x_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch index
    batch_idx = tl.program_id(0)
    
    # Calculate the starting position for this batch
    batch_start = batch_idx * stride_x_batch
    
    # Each program handles one batch
    for i in range(0, n_elements, BLOCK_SIZE):
        # Create offsets within the current batch
        offsets = i + tl.arange(0, BLOCK_SIZE)
        
        # Create mask to avoid going out of bounds
        mask = offsets < n_elements
        
        # Load data for this batch
        x_data = tl.load(x_ptr + batch_start + offsets * stride_x_dim, mask=mask, other=-float('inf'))
        
        # Find max value and its index
        max_val = tl.max(x_data)
        max_idx = tl.arg_max(x_data)
        
        # Store the result
        if i == 0:
            tl.store(output_ptr + batch_idx, max_idx)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For argmax along dimension 1, we need to handle it specially
        if self.dim == 1:
            # Flatten the tensor except for the batch dimension
            batch_size = x.shape[0]
            dim1 = x.shape[1]
            dim2 = x.shape[2]
            
            # Reshape for easier processing
            x_reshaped = x.view(batch_size, dim1 * dim2)
            
            # Create output tensor
            output = torch.empty(batch_size, dtype=torch.int64, device=x.device)
            
            # Calculate parameters for kernel
            n_elements = dim1 * dim2
            stride_x_batch = dim1 * dim2
            stride_x_dim = 1
            
            # Launch kernel
            BLOCK_SIZE = 1024
            
            grid = lambda meta: (batch_size,)
            
            argmax_kernel[grid](
                x_reshaped,
                output,
                n_elements,
                dim1,
                stride_x_batch,
                stride_x_dim,
                BLOCK_SIZE=BLOCK_SIZE
            )
            
            return output
        else:
            # For other dimensions, fall back to PyTorch implementation
            return torch.argmax(x, dim=self.dim)