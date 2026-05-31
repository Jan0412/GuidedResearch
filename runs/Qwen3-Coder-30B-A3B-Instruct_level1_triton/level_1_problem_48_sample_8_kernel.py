import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    reduction_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(0)
    
    # Calculate the starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Perform sum reduction
    sum_val = tl.sum(x, axis=0)
    
    # Store the result
    tl.store(output_ptr + pid, sum_val, mask=pid < reduction_size)

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for mean reduction.
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
        Reduces the input tensor along the specified dimension by taking the mean.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with reduced dimension. The shape of the output is the same as the input except for the reduced dimension which is removed.
        """
        # For simplicity, let's assume we're reducing over the last dimension
        if self.dim != len(x.shape) - 1:
            raise NotImplementedError("Currently only supports reducing over the last dimension")
        
        # Flatten the tensor except for the last dimension
        batch_size = x.shape[:-1]
        last_dim_size = x.shape[-1]
        
        # Reshape to 2D for processing
        x_flat = x.view(-1, last_dim_size)
        
        # Prepare output tensor
        output_shape = list(batch_size)
        output = torch.empty(output_shape, dtype=torch.float32, device=x.device)
        
        # Calculate total elements and reduction size
        n_elements = x_flat.numel()
        reduction_size = x_flat.size(0)
        
        # Set up kernel parameters
        BLOCK_SIZE = 1024
        
        # Grid size calculation
        grid_size = (reduction_size + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Launch kernel
        mean_kernel[grid_size](
            x_flat,
            output,
            n_elements,
            reduction_size,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Normalize by the reduction dimension size
        output = output / last_dim_size
        
        return output