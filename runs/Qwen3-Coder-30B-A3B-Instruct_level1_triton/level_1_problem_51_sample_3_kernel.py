import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def argmax_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element along the non-reduced dimensions
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Initialize max value and index
    max_val = tl.full([BLOCK_SIZE], -float('inf'), dtype=tl.float32)
    max_idx = tl.full([BLOCK_SIZE], 0, dtype=tl.int32)
    
    # Iterate through the reduced dimension
    for i in range(dim_size):
        # Calculate offset for current position
        current_offset = offsets * dim_size + i
        current_val = tl.load(x_ptr + current_offset, mask=mask, other=-float('inf'))
        
        # Update max if current value is greater
        mask_greater = current_val > max_val
        max_val = tl.where(mask_greater, current_val, max_val)
        max_idx = tl.where(mask_greater, i, max_idx)
    
    # Store the result
    tl.store(out_ptr + offsets, max_idx, mask=mask)

class ModelNew(nn.Module):
    """
    Optimized model that performs Argmax over a specified dimension using Triton kernels.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmax.

        Args:
            dim (int): The dimension to perform argmax over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies argmax over the specified dimension to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with argmax applied, with the specified dimension removed.
        """
        # Ensure input is on GPU
        assert x.is_cuda, "Input tensor must be on CUDA."
        
        # Get dimensions
        dims = x.shape
        dim_size = dims[self.dim]
        
        # Calculate total elements in reduced dimension
        n_elements = 1
        for i, d in enumerate(dims):
            if i != self.dim:
                n_elements *= d
                
        # Prepare output tensor
        output_shape = list(dims)
        output_shape.pop(self.dim)
        out = torch.empty(output_shape, dtype=torch.int32, device=x.device)
        
        # Handle special case where dimension size is 0
        if dim_size == 0:
            return out
        
        # Prepare input tensor for contiguous memory access
        x = x.contiguous()
        
        # Calculate grid size
        BLOCK_SIZE = 128
        grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
        
        # Launch the Triton kernel
        argmax_kernel[grid](x, out, n_elements, dim_size, BLOCK_SIZE=BLOCK_SIZE)
        
        return out