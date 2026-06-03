import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmax_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    n_elements,  # Total number of elements in input
    n_cols,  # Size of the dimension we're taking argmax over
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch index
    batch_idx = tl.program_id(0)
    
    # Calculate offset for this batch
    batch_offset = batch_idx * n_cols
    
    # Initialize max value and index
    max_val = tl.full([1], float("-inf"), dtype=tl.float32)
    max_idx = tl.full([1], 0, dtype=tl.int64)
    
    # Iterate through the dimension in blocks
    for start in range(0, n_cols, BLOCK_SIZE):
        # Calculate offsets for current block
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load values
        ptr = x_ptr + batch_offset + offsets
        vals = tl.load(ptr, mask=mask, other=float("-inf")).to(tl.float32)
        
        # Update max and index using online algorithm
        for i in range(BLOCK_SIZE):
            val = vals[i]
            idx = start + i
            # Use tl.where for conditional update
            is_new_max = val > max_val[0]
            max_val = tl.where(is_new_max, val, max_val)
            max_idx = tl.where(is_new_max, idx, max_idx)
    
    # Store result
    out_ptr[batch_idx] = max_idx[0]


def triton_argmax(x: torch.Tensor, dim: int):
    """
    Triton-based argmax implementation.
    
    Args:
        x: Input tensor
        dim: Dimension to perform argmax over
        
    Returns:
        Tensor with argmax results along the specified dimension
    """
    # Ensure tensor is contiguous and on CUDA
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get shape info
    shape = x.shape
    # Normalize negative dimension index
    if dim < 0:
        dim += len(shape)
    
    # Calculate the number of elements in the dimension we're reducing
    n_cols = shape[dim]
    
    # Calculate output shape
    out_shape = list(shape)
    del out_shape[dim]
    
    # Prepare output tensor
    out = torch.empty(out_shape, dtype=torch.int64, device=x.device)
    
    # Determine grid size: one block per batch (all dimensions except the target dimension)
    batch_size = 1
    for i, s in enumerate(shape):
        if i != dim:
            batch_size *= s
    
    # Set block size (tunable parameter)
    BLOCK_SIZE = 256
    
    # Launch kernel
    argmax_kernel[(batch_size,)](
        x, out, x.numel(), n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Argmax over a specified dimension using Triton kernel.
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
        return triton_argmax(x, self.dim)