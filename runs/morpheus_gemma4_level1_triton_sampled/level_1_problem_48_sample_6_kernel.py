import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mean_kernel(
    x_ptr,
    out_ptr,
    S_reduction,
    S_after,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one reduction vector
    pid = tl.program_id(0)
    
    # S_reduced = S_before * S_after
    # We map the flat pid back to the indices of dimensions before and after the reduction dim
    idx_before = pid // S_after
    idx_after = pid % S_after
    
    # Calculate the starting offset for the reduction vector
    # Offset = (index_before * size_of_reduction_dim * size_of_after_dims) + index_after
    start_offset = idx_before * (S_reduction * S_after) + idx_after
    
    acc = 0.0
    # Loop over the reduction dimension in blocks
    for i in range(0, S_reduction, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < S_reduction
        # Elements in the reduction dimension are spaced by S_after in a contiguous tensor
        vals = tl.load(x_ptr + start_offset + offsets * S_after, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
    
    # Store the mean (sum / count)
    tl.store(out_ptr + pid, acc / S_reduction)

def triton_mean(x: torch.Tensor, dim: int):
    # Normalize dimension
    ndim = x.ndim
    if dim < 0:
        dim += ndim
    
    # The kernel logic assumes a contiguous tensor for offset calculations
    x = x.contiguous()
    
    shape = x.shape
    S_reduction = shape[dim]
    
    # Calculate product of dimensions after the reduction dimension
    S_after = 1
    for i in range(dim + 1, ndim):
        S_after *= shape[i]
        
    S_reduced = x.numel() // S_reduction
    
    # Prepare output tensor (flattened)
    out = torch.empty(S_reduced, device=x.device, dtype=x.dtype)
    
    # Tuning parameter for block size
    BLOCK_SIZE = 1024
    grid = (S_reduced,)
    
    mean_kernel[grid](
        x, 
        out, 
        S_reduction, 
        S_after, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape output to remove the reduced dimension
    out_shape = list(shape)
    del out_shape[dim]
    return out.view(*out_shape)

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
        Reduces the input tensor along the specified dimension by taking the mean using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with reduced dimension.
        """
        return triton_mean(x, self.dim)