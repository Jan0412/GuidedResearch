import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def min_reduction_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_rows,  # Number of rows (batch size)
    n_cols,  # Number of columns (elements per row to reduce)
    BLOCK_SIZE: tl.constexpr
):
    # Each program processes one row
    row_idx = tl.program_id(0)
    
    # Calculate starting offset for this row
    row_start = row_idx * n_cols
    
    # Initialize minimum with large value
    min_val = tl.full([BLOCK_SIZE], float('inf'), dtype=tl.float32)
    
    # Iterate over columns in chunks of BLOCK_SIZE
    for start_col in range(0, n_cols, BLOCK_SIZE):
        offsets = start_col + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load data
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=float('inf'))
        
        # Compute minimum with current chunk
        min_val = tl.minimum(min_val, x)
    
    # Now reduce the BLOCK_SIZE elements to a single value
    # First, do parallel reduction within the block
    for i in range(BLOCK_SIZE // 2):
        min_val = tl.minimum(min_val, tl.roll(min_val, 1 << i))
    
    # Store the final minimum for this row
    tl.store(out_ptr + row_idx, min_val[0])


def triton_min(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Custom Triton implementation of min reduction over specified dimension.
    Only supports dim=1 for now (reducing over columns).
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    shape = x.shape
    if dim == 1:
        n_rows = shape[0] * shape[2] if len(shape) > 2 else shape[0]
        n_cols = shape[1]
        # Reshape to 2D for simplicity if needed
        if len(shape) == 3:
            x = x.view(shape[0] * shape[2], shape[1])
            n_rows = shape[0] * shape[2]
            out_shape = (shape[0], shape[2])
        else:
            out_shape = (shape[0],)
    else:
        # Handle other dimensions by permuting to make dim=1
        raise NotImplementedError("Only dim=1 supported in this implementation")
    
    # Prepare output tensor
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # For FP32, use reasonable block size
    BLOCK_SIZE = 128
    
    # Grid: one block per row
    grid = (n_rows,)
    
    # Launch kernel
    min_reduction_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs min reduction over a specific dimension using Triton.
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
        Applies min reduction over the specified dimension to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after min reduction over the specified dimension.
        """
        if self.dim == 1:
            # For 3D input: (batch_size, dim1, dim2)
            # We want to reduce over dim1, so treat each (dim2) vector as a row
            if x.dim() == 3:
                # Reshape to 2D: (batch_size * dim2, dim1)
                batch_size, dim1, dim2 = x.shape
                x_reshaped = x.permute(0, 2, 1).reshape(batch_size * dim2, dim1)
                # Apply min reduction over dim=1 (which is original dim1)
                out = triton_min(x_reshaped, dim=1)
                # Reshape back to (batch_size, dim2)
                out = out.view(batch_size, dim2)
                return out
            elif x.dim() == 2:
                # Direct application
                return triton_min(x, dim=1)
            else:
                raise ValueError("Only 2D or 3D tensors supported")
        else:
            # For other dimensions, use fallback (though current Triton kernel only supports dim=1)
            return torch.min(x, dim=self.dim)[0]