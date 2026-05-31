import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mean_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_cols,  # Number of elements along the reduction dimension
    stride,  # Stride between consecutive elements in the reduction dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element (i.e., one mean calculation)
    # Program ID corresponds to the index in the output tensor
    out_idx = tl.program_id(0)
    
    # Compute the starting offset in the input tensor for this output element
    # We assume the reduction is along the last dimension for simplicity
    # For general dimension, we need to compute the offset based on the output index
    # Here we assume the reduction is along the last dimension (dim=-1)
    offset = out_idx * n_cols
    
    # Accumulate sum
    sum_ = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over the reduction dimension in blocks
    for start in range(0, n_cols, BLOCK_SIZE):
        # Compute offsets for current block
        block_offsets = start + tl.arange(0, BLOCK_SIZE)
        # Create mask for valid elements
        mask = block_offsets < n_cols
        # Load values
        x = tl.load(x_ptr + offset + block_offsets * stride, mask=mask)
        # Accumulate sum
        sum_ += tl.sum(x.to(tl.float32))
    
    # Compute mean
    mean_val = sum_ / n_cols
    
    # Store result
    tl.store(out_ptr + out_idx, mean_val)


def triton_mean(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute mean along specified dimension using Triton kernel.
    
    Args:
        x: Input tensor
        dim: Dimension to reduce over
        
    Returns:
        Tensor with mean computed along specified dimension
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Handle negative dimension indexing
    if dim < 0:
        dim = x.dim() + dim
        
    # Get shapes
    shape = list(x.shape)
    n_reduced = shape[dim]
    
    # Calculate output shape
    out_shape = shape[:dim] + shape[dim+1:] if dim < len(shape) - 1 else shape[:dim]
    
    # Create output tensor
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Calculate total number of elements to process in output
    n_out = out.numel()
    
    # For simplicity, we'll assume reduction over the last dimension
    # If reduction is over another dimension, we need to transpose/reshape
    if dim != x.dim() - 1:
        # Move reduction dimension to the end
        perm = list(range(x.dim()))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        x_perm = x.permute(perm)
        shape_perm = list(x_perm.shape)
        
        # Create output tensor with permuted shape
        out_perm_shape = shape_perm[:-1]
        out_perm = torch.empty(out_perm_shape, dtype=x.dtype, device=x.device)
        
        # Calculate parameters for kernel
        n_cols = shape_perm[-1]
        stride = 1
        for i in range(len(shape_perm)-1, 0, -1):
            stride *= shape_perm[i]
        stride = 1  # Since we moved dim to last position, consecutive elements in reduction dim are contiguous
        
        BLOCK_SIZE = 128
        grid = lambda meta: (n_out,)
        
        # Launch kernel
        mean_kernel[grid](x_perm, out_perm, n_cols, stride, BLOCK_SIZE=BLOCK_SIZE)
        
        # Reshape output to match expected output shape
        return out_perm.reshape(out_shape)
    
    # For reduction over last dimension
    n_cols = shape[-1]
    stride = 1
    
    BLOCK_SIZE = 128
    grid = lambda meta: (n_out,)
    
    # Launch kernel
    mean_kernel[grid](x, out, n_cols, stride, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs mean reduction over a specific dimension using Triton kernel.
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
        Reduces the input tensor along the specified dimension by taking the mean using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with reduced dimension.
        """
        return triton_mean(x, dim=self.dim)