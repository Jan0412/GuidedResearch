import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mean_kernel(
    x_ptr,        # Pointer to input tensor
    out_ptr,      # Pointer to output tensor
    n_reduce,     # Size of the dimension being reduced
    stride_reduce, # Stride of the reduction dimension
    S_suffix,     # Product of dimensions after the reduction dimension
    n_out,        # Total number of elements in the output tensor
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element of the output tensor
    pid = tl.program_id(0)
    if pid >= n_out:
        return

    # Map the output index (pid) back to input coordinates
    # pid = p * S_suffix + s
    p = pid // S_suffix
    s = pid % S_suffix

    # Calculate the base pointer for the reduction vector
    # Input offset = p * (n_reduce * S_suffix) + s
    base_offset = p * (n_reduce * S_suffix) + s
    
    # Load the reduction dimension in blocks
    acc = 0.0
    for start in range(0, n_reduce, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_reduce
        # Load values along the reduction dimension
        # Pointer = x_ptr + base_offset + k * stride_reduce
        ptr = x_ptr + base_offset + offsets * stride_reduce
        vals = tl.load(ptr, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)

    # Compute mean and store
    mean_val = acc / n_reduce
    tl.store(out_ptr + pid, mean_val)


def triton_mean(x: torch.Tensor, dim: int):
    """
    Triton wrapper for torch.mean(x, dim=dim)
    """
    # Ensure input is contiguous for the simplified indexing logic
    x = x.contiguous()
    shape = x.shape
    n_dims = len(shape)
    
    # Handle negative dimensions
    if dim < 0:
        dim += n_dims

    n_reduce = shape[dim]
    
    # Calculate S_prefix and S_suffix
    S_prefix = 1
    for i in range(dim):
        S_prefix *= shape[i]
        
    S_suffix = 1
    for i in range(dim + 1, n_dims):
        S_suffix *= shape[i]
        
    n_out = S_prefix * S_suffix
    
    # Prepare output tensor
    out_shape = shape[:dim] + shape[dim+1:]
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    
    # The stride of the reduction dimension in a contiguous tensor
    stride_reduce = S_suffix
    
    # Grid: one program per output element
    grid = (n_out,)
    
    # Block size for the reduction loop
    # 1024 is a safe and efficient balance for most GPUs
    BLOCK_SIZE = 1024
    
    mean_kernel[grid](
        x, out, 
        n_reduce, 
        stride_reduce, 
        S_suffix, 
        n_out, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs mean reduction over a specific dimension using Triton.
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
            torch.Tensor: Output tensor with reduced dimension.
        """
        # Use the Triton-optimized mean implementation
        return triton_mean(x, self.dim)