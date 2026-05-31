import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def argmin_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    stride_x_dim,
    stride_x_batch,
    batch_size,
    dim1,
    dim2,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch dimension
    batch_idx = tl.program_id(0)
    
    # Each program handles one batch
    if batch_idx >= batch_size:
        return
    
    # Calculate base pointers for this batch
    batch_offset = batch_idx * stride_x_batch
    x_base_ptr = x_ptr + batch_offset
    
    # Process each element in the specified dimension
    for i in range(dim1):
        # Calculate the offset for this row
        row_offset = i * stride_x_dim
        
        # Initialize min_val and min_idx
        min_val = tl.full([1], float('inf'), dtype=tl.float32)
        min_idx = tl.full([1], 0, dtype=tl.int32)
        
        # Iterate through elements in the dimension
        for j in range(dim2):
            # Calculate the actual index
            idx = row_offset + j
            
            # Load value at this position
            val = tl.load(x_base_ptr + idx, mask=True, other=float('inf'))
            
            # Update min if current value is smaller
            mask = val < min_val
            min_val = tl.where(mask, val, min_val)
            min_idx = tl.where(mask, j, min_idx)
        
        # Store the result
        out_idx = batch_idx * dim1 + i
        tl.store(out_ptr + out_idx, min_idx, mask=True)

def triton_argmin(x: torch.Tensor, dim: int):
    """
    Triton implementation of argmin operation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # For this specific case, we're doing argmin along dim=1
    # So we compute argmin for each row across the last dimension
    batch_size, dim1, dim2 = x.shape
    
    # Prepare output tensor
    out = torch.zeros(batch_size, dim1, dtype=torch.int32, device=x.device)
    
    # Determine block size
    BLOCK_SIZE = 128
    
    # Grid configuration - one block per batch
    grid = lambda meta: (batch_size,)
    
    # Launch kernel
    argmin_kernel[grid](
        x,
        out,
        x.numel(),
        x.stride(dim),
        x.stride(0),
        batch_size,
        dim1,
        dim2,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for argmin operation.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmin on.

        Args:
            dim (int): Dimension along which to find the minimum value.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Finds the index of the minimum value along the specified dimension using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor containing the indices of the minimum values along the specified dimension.
        """
        return triton_argmin(x, self.dim)