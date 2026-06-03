import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def min_reduction_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_rows,  # Number of rows (batch dimension)
    n_cols,  # Number of columns (elements to reduce per row)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Calculate base pointer for this row
    row_start = row_idx * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize minimum to a large value (FP32 max)
    min_val = tl.full([BLOCK_SIZE], float("inf"), dtype=tl.float32)
    
    # Process in chunks of BLOCK_SIZE
    for start in range(0, n_cols, BLOCK_SIZE):
        # Compute actual offsets for this chunk
        chunk_offsets = start + offsets
        mask = chunk_offsets < n_cols
        
        # Load data
        x = tl.load(x_ptr + row_start + chunk_offsets, mask=mask, other=float("inf"))
        
        # Update minimum
        min_val = tl.minimum(min_val, x)
    
    # Reduce within the block using tree reduction
    # First, we need to reduce across the block dimension
    for i in range(BLOCK_SIZE // 2):
        # Shift by i+1 positions
        shift = 2 ** i
        shifted = tl.load(tl.view(min_val.data_ptr() + shift * tl.dtype(tl.float32).itemsize, [BLOCK_SIZE]), 
                         mask=tl.arange(0, BLOCK_SIZE) < BLOCK_SIZE - shift)
        min_val = tl.minimum(min_val, shifted)
    
    # After the loop, the minimum is in the first element
    final_min = min_val[0]
    
    # Store the result
    tl.store(out_ptr + row_idx, final_min)


def triton_min(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    This function wraps the Triton kernel call for min reduction.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get the shape information
    shape = x.shape
    # Normalize negative dimension index
    if dim < 0:
        dim += len(shape)
    
    # For simplicity, we assume reduction over the last dimension
    # If reduction is over other dimensions, we'll need to transpose/reshape
    if dim != len(shape) - 1:
        # Permute dimensions so that the reduction dimension is last
        perm = list(range(len(shape)))
        perm.pop(dim)
        perm.append(dim)
        x = x.permute(perm)
        shape = x.shape
    
    # Prepare output tensor
    out_shape = list(shape)
    out_shape[-1] = 1
    out_shape = tuple(out_shape[:-1])  # Remove last dimension as it's reduced
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    n_rows = 1
    for s in shape[:-1]:
        n_rows *= s
    n_cols = shape[-1]
    
    BLOCK_SIZE = 256  # Tunable parameter for block size
    
    # Determine the number of blocks needed (we only need one block per row)
    grid = (n_rows,)
    
    # Launch the Triton kernel
    min_reduction_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs min reduction over a specific dimension using Triton kernel.
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
        return triton_min(x, self.dim)