import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_reduction_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input
    dim_size,  # Size of the dimension being reduced
    block_size: tl.constexpr,
):
    # Each program handles one output element (i.e., one reduced row)
    batch_idx = tl.program_id(0)
    
    # Calculate offset to the start of the current batch
    batch_offset = batch_idx * dim_size
    
    # Initialize accumulator
    sum_val = tl.zeros([1], dtype=tl.float32)
    
    # Iterate over the reduction dimension in blocks
    for start in range(0, dim_size, block_size):
        offsets = batch_offset + start + tl.arange(0, block_size)
        mask = offsets < (batch_offset + dim_size)
        
        # Load data and accumulate
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
    
    # Store result
    out_offsets = batch_idx
    tl.store(out_ptr + out_offsets, sum_val)


def triton_sum_reduction(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Performs sum reduction over a specified dimension using Triton.
    Only supports keepdim=True behavior for simplicity.
    """
    # Ensure input is contiguous and on GPU
    x = x.contiguous()
    
    # Validate dimension
    if dim < 0:
        dim += x.dim()
    
    # Get input shape and prepare output shape
    shape = list(x.shape)
    out_shape = shape.copy()
    out_shape[dim] = 1
    
    # Calculate total elements and reduction dimension size
    n_elements = x.numel()
    dim_size = shape[dim]
    
    # Calculate number of batches (all dimensions except reduction dim)
    batch_size = n_elements // dim_size
    
    # Prepare output tensor
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Set block size (tunable parameter)
    BLOCK_SIZE = 256
    
    # Grid: one block per batch (one output per batch)
    grid = (batch_size,)
    
    # Launch kernel
    sum_reduction_kernel[grid](
        x, out, n_elements, dim_size, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs sum reduction over a specified dimension using Triton.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies sum reduction over the specified dimension using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        # Normalize negative dimension index
        dim = self.dim if self.dim >= 0 else x.dim() + self.dim
        
        # Handle different dimensions by moving reduction dim to last position for simplicity
        if dim != x.dim() - 1:
            # Permute so reduction dim is last
            dims = list(range(x.dim()))
            # Move reduction dim to the end
            dims.pop(dim)
            dims.append(dim)
            x_permuted = x.permute(dims)
            
            # Perform reduction on last dimension
            result = triton_sum_reduction(x_permuted, dim=-1)
            
            # Restore original dimension order (but with size 1 at reduction position)
            # Result has same shape as permuted input except last dim is 1
            out_shape = list(result.shape)
            # Move the last dimension (reduction result) back to original position
            out_dims = list(range(len(out_shape)))
            out_dims.pop(-1)
            out_dims.insert(dim, len(out_dims))
            
            result = result.permute(out_dims)
        else:
            # Direct reduction on last dimension
            result = triton_sum_reduction(x, dim=-1)
            
        return result