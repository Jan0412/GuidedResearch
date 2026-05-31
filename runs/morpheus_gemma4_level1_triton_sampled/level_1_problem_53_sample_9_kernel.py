import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def min_reduction_kernel(
    x_ptr, 
    out_ptr, 
    n_red, 
    stride_out, 
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to perform min reduction over the last dimension of a tensor.
    Each program (pid) handles one reduction row.
    """
    # Get the program ID (which row we are reducing)
    pid = tl.program_id(0)
    
    # Pointer to the start of the reduction row
    row_ptr = x_ptr + pid * stride_out
    
    # Initialize min_val to infinity
    min_val = float('inf')
    
    # Iterate over the reduction dimension in chunks of BLOCK_SIZE
    for i in range(0, n_red, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_red
        
        # Load a block of elements from the row
        vals = tl.load(row_ptr + offsets, mask=mask, other=float('inf'))
        
        # Reduce the block and update the running minimum
        # tl.min returns a scalar for a 1D tensor when axis=0
        min_val = tl.minimum(min_val, tl.min(vals, axis=0))
    
    # Store the final minimum value in the output tensor
    tl.store(out_ptr + pid, min_val)


def triton_min(x: torch.Tensor, dim: int):
    """
    Wrapper for the Triton min reduction kernel.
    Handles dimension permutation to ensure the reduction is performed on the last dimension.
    """
    # Ensure input is on CUDA and FP32
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.to(torch.float32)
    
    # Move the reduction dimension to the end
    # This allows the kernel to treat the reduction as a contiguous or semi-contiguous operation
    dims = list(range(x.ndim))
    axis = dims.pop(dim)
    dims.append(axis)
    
    # Permute and make contiguous to ensure the reduction dimension is the fastest changing
    # This is necessary for the simple row-based pointer arithmetic in the kernel
    x_permuted = x.permute(*dims).contiguous()
    
    n_red = x_permuted.shape[-1]
    out_shape = x_permuted.shape[:-1]
    
    # Prepare output tensor
    out = torch.empty(out_shape, dtype=torch.float32, device=x.device)
    
    # Flatten the output for the kernel grid
    # The stride to get to the next row in the contiguous x_permuted is exactly n_red
    stride_out = n_red
    n_elements_out = out.numel()
    
    # Tuning parameter for block size
    BLOCK_SIZE = 1024
    
    # Grid is one program per reduction row
    grid = (n_elements_out,)
    
    # Launch kernel
    min_reduction_kernel[grid](
        x_permuted, 
        out, 
        n_red, 
        stride_out, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
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
        Applies min reduction over the specified dimension to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after min reduction over the specified dimension.
        """
        return triton_min(x, self.dim)