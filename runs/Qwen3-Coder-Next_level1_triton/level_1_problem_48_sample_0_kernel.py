import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mean_reduce_kernel(
    x_ptr,
    out_ptr,
    n_cols,  # Number of elements in the reduction dimension
    stride_batch,  # Stride between batches
    stride_dim1,   # Stride between rows in dim1
    stride_dim2,   # Stride between elements in dim2
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID corresponds to the batch index (assuming 1D grid)
    batch_idx = tl.program_id(0)
    
    # Compute base pointer for this batch
    base_ptr = x_ptr + batch_idx * stride_batch
    
    # Accumulator for the sum
    sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Iterate over the reduction dimension (dim1 * dim2 elements)
    # Since we're reducing over a specific dimension, we need to traverse the memory layout accordingly
    # For simplicity, let's assume we're reducing over the last dimension (dim2)
    # and the tensor is laid out as [batch, dim1, dim2]
    
    # Calculate offsets for the reduction
    for i in range(n_cols):
        # Compute the offset based on the current batch and position in reduction dim
        # We assume we're reducing over dim2 (last dimension), so we skip by stride_dim2 each time
        offset = i * stride_dim2
        # Load the value
        val = tl.load(base_ptr + offset)
        # Accumulate in float32
        sum = sum + val.to(tl.float32)
    
    # Compute mean
    mean = sum / n_cols
    
    # Store the result (assuming output is [batch, dim1] after reducing dim2)
    # But according to the spec, it seems we want to reduce over the specified dimension
    # Let's handle general dimension reduction by calculating the appropriate output shape
    
    # For this implementation, we'll handle reducing over the last dimension (dim2)
    # and output shape would be [batch, dim1]
    # The actual implementation should be more general, but for simplicity and performance,
    # let's focus on the common case of reducing the last dimension
    
    # Store the mean for each row in the batch
    # Since we're only computing one mean per batch, we need to adjust this
    # Actually, if reducing over dim2, we should have one mean per [batch, dim1] position
    # So we need to loop over dim1 as well
    
    # Let's restructure this kernel to handle reducing over any dimension
    # For now, let's assume we're reducing over dim2, and the output is [batch, dim1]
    row_idx = tl.program_id(1) if tl.program_id(1) < tl.num_programs(1) else 0
    
    # Compute mean for each [batch, row] position
    sum_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for i in range(n_cols):
        offset = row_idx * stride_dim1 + i * stride_dim2
        val = tl.load(x_ptr + batch_idx * stride_batch + offset)
        sum_row = sum_row + val.to(tl.float32)
    
    mean_row = sum_row / n_cols
    
    # Store the result
    tl.store(out_ptr + batch_idx * (stride_batch // n_cols) + row_idx, mean_row)


def triton_mean(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Computes mean reduction along the specified dimension using Triton kernel.
    Optimized for FP32 precision.
    """
    # Ensure tensor is contiguous
    x = x.contiguous()
    
    # Get tensor shape
    shape = x.shape
    n_dims = len(shape)
    
    # Normalize dimension
    if dim < 0:
        dim += n_dims
    
    # Compute output shape
    output_shape = list(shape)
    output_shape.pop(dim)
    output_shape = tuple(output_shape)
    
    # Create output tensor
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Calculate strides
    strides = x.stride()
    dim_stride = strides[dim]
    
    # For simplicity and performance, let's handle the common case of reducing the last dimension
    # If we need to reduce over a different dimension, we can transpose the tensor
    if dim != len(shape) - 1:
        # Permute dimensions to move the target dimension to the last position
        perm = list(range(n_dims))
        perm.pop(dim)
        perm.append(dim)
        x_perm = x.permute(perm)
        
        # Compute mean over last dimension
        out_perm = triton_mean(x_perm, len(shape) - 1)
        
        # Permute back to original shape
        inv_perm = [0] * n_dims
        for i, p in enumerate(perm):
            inv_perm[p] = i
        return out_perm.permute(inv_perm)
    
    # For reducing the last dimension, we can use a simple kernel
    batch_size = shape[0]
    dim1 = shape[1] if len(shape) > 1 else 1
    dim2 = shape[-1]  # The dimension we're reducing
    
    # For simplicity, let's assume we're working with 3D tensors [batch, dim1, dim2]
    # and reducing over dim2
    
    # Adjust for 2D case
    if len(shape) == 2:
        batch_size = shape[0]
        dim1 = 1
        dim2 = shape[1]
    elif len(shape) > 3:
        # Flatten all dimensions except the last one into batch
        batch_size = 1
        for s in shape[:-1]:
            batch_size *= s
        dim1 = 1
        dim2 = shape[-1]
    
    # Define kernel parameters
    BLOCK_SIZE = 256
    
    # Grid configuration
    grid = (batch_size, dim1)
    
    # Launch kernel
    mean_reduce_kernel[grid](
        x_ptr=x.data_ptr(),
        out_ptr=out.data_ptr(),
        n_cols=dim2,
        stride_batch=strides[0] if len(strides) > 0 else 1,
        stride_dim1=strides[1] if len(strides) > 1 else 1,
        stride_dim2=strides[-1],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
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