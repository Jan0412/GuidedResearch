import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def rev_cumsum_kernel(
    x_ptr, 
    out_ptr, 
    n_elements, 
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (the dimension being summed)
    row_idx = tl.program_id(0)
    
    # Pointer to the start of the current row
    row_start_ptr = x_ptr + row_idx * stride_row
    out_start_ptr = out_ptr + row_idx * stride_row

    # Create offsets for the row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # To perform a reverse cumulative sum (suffix sum), we:
    # 1. Load the elements in reverse order
    # 2. Compute the prefix sum (cumsum) of the reversed sequence
    # 3. Store the result back in reverse order
    
    # Load in reverse: x[n-1], x[n-2], ..., x[0]
    # Use mask to handle n_elements that are not exactly BLOCK_SIZE
    x_rev = tl.load(row_start_ptr + (n_elements - 1) - offsets, mask=mask, other=0.0)
    
    # Compute cumulative sum along the reversed sequence
    res_rev = tl.cumsum(x_rev, axis=0)
    
    # Store back in reverse order to restore original sequence indices
    # res_rev[0] (sum of x[n-1]) goes to out[n-1]
    # res_rev[n-1] (sum of x[n-1...0]) goes to out[0]
    tl.store(out_start_ptr + (n_elements - 1) - offsets, res_rev, mask=mask)

def triton_rev_cumsum(x: torch.Tensor):
    """
    Wrapper for the reverse cumulative sum Triton kernel.
    Assumes x is contiguous and the operation is on the last dimension.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensor is contiguous to simplify pointer arithmetic
    x = x.contiguous()
    n_elements = x.shape[-1]
    stride_row = x.stride(-1) * n_elements if x.ndim > 1 else 1
    # For a contiguous tensor, the stride to the next row is simply the size of the last dim
    # But since we use x.contiguous(), the stride of the last dim is 1.
    # So the offset to the next row is simply n_elements.
    stride_row = n_elements 

    out = torch.empty_like(x)

    # BLOCK_SIZE must be a power of 2 and >= n_elements
    # Given the problem constraints (32768), 32768 is a power of 2.
    BLOCK_SIZE = triton.next_power_of_2(n_elements)
    
    # Grid: one program per row
    grid = (x.shape[0] if x.ndim > 1 else 1,)

    rev_cumsum_kernel[grid](
        x, out, n_elements, stride_row, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a reverse cumulative sum operation 
    along a specified dimension using a custom Triton kernel.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Handle negative dimensions
        ndim = x.ndim
        dim = self.dim if self.dim >= 0 else self.dim + ndim
        
        # The Triton kernel is optimized for the last dimension.
        # We transpose the target dimension to the last position.
        # Transpose is a metadata change in PyTorch (O(1)).
        x_transposed = x.transpose(dim, -1)
        
        # We call the Triton wrapper. The wrapper handles .contiguous() 
        # to ensure the kernel can access memory linearly.
        out_contig = triton_rev_cumsum(x_transposed)
        
        # Reshape and transpose back to original orientation.
        # .view() is O(1) if possible, .transpose() is O(1).
        out = out_contig.view(x_transposed.shape).transpose(dim, -1)
        
        return out