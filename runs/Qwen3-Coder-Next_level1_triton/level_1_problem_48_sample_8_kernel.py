import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mean_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_cols,  # Number of columns in the reduction dimension
    stride_x,  # Stride in the reduction dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element (i.e., one position in non-reduced dimensions)
    # Calculate the starting index for this program's output
    out_idx = tl.program_id(0)
    
    # Accumulator for the mean
    sum_val = tl.zeros([1], dtype=tl.float32)
    
    # Iterate over the reduction dimension in blocks
    for start_col in range(0, n_cols, BLOCK_SIZE):
        offsets = start_col + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load data from input tensor
        # We need to compute the actual pointer offset for this position
        # Since x_ptr is a flattened representation, we compute offset based on out_idx and current column
        # For a 3D tensor [batch, dim1, dim2], when reducing dim=2 (last dim), out_idx = batch*dim1 + dim1_idx
        # The pointer arithmetic for 3D tensor with shape [B, D1, D2] when reducing dim=2:
        #   offset = out_idx * D2 + col
        # But here we use stride_x which represents the stride in the reduction dimension
        # The stride_x is set to be the stride of the reduced dimension in the original tensor
        ptr = x_ptr + out_idx * stride_x + offsets
        x = tl.load(ptr, mask=mask, other=0.0)
        
        # Accumulate sum
        sum_val += tl.sum(x.to(tl.float32), axis=0)
    
    # Compute mean: average over n_cols elements
    mean_val = sum_val / n_cols
    
    # Store result
    tl.store(out_ptr + out_idx, mean_val)


def triton_mean(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute mean along a specified dimension using Triton kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Get the shape and compute output shape
    shape = list(x.shape)
    if dim < 0:
        dim += len(shape)
    
    # Create output tensor with reduced dimension
    out_shape = shape[:dim] + shape[dim+1:]
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Calculate parameters for the kernel
    # We need to flatten all dimensions except the reduction dimension
    # and determine the stride in the reduction dimension
    
    # Calculate total number of elements in the reduction dimension
    n_cols = shape[dim]
    
    # Calculate number of elements in the output (all other dimensions product)
    n_out = 1
    for i, s in enumerate(shape):
        if i != dim:
            n_out *= s
    
    # Calculate stride in the reduction dimension
    # For a tensor with shape [d0, d1, ..., dn], the stride for dimension i is
    # stride[i] = product of sizes of dimensions j where j > i
    # But since we'll use contiguous layout, stride[dim] gives us what we need
    stride_x = x.stride(dim)
    
    BLOCK_SIZE = 128  # Tunable parameter for block size
    
    # Grid: one program per output element
    grid = (n_out,)
    
    # Launch kernel
    mean_kernel[grid](x, out, n_cols, stride_x, BLOCK_SIZE=BLOCK_SIZE)
    
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
        return triton_mean(x, self.dim)