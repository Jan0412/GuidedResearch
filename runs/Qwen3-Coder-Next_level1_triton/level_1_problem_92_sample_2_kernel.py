import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def exclusive_cumsum_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    n_elements,  # Total number of elements
    dim_size,  # Size of the dimension along which we compute cumsum
    stride_dim,  # Stride along the dimension
    stride_other,  # Stride for other dimensions
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch/outer dimension index
    batch_idx = tl.program_id(0)
    
    # Calculate the starting offset for this batch
    base_offset = batch_idx * stride_other
    
    # We'll process the dimension in blocks
    for start in range(0, dim_size, BLOCK_SIZE):
        # Calculate actual offsets for this block
        offsets = base_offset + tl.arange(0, BLOCK_SIZE) * stride_dim
        offsets = tl.multiple_of(offsets, 1)  # Tell Triton these are aligned
        
        # Load input values (with masking for boundary)
        mask = (start + tl.arange(0, BLOCK_SIZE)) < dim_size
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Compute exclusive cumsum
        # For exclusive cumsum, we need the running sum from previous elements
        # We'll use a simple sequential approach within the kernel since cumsum is inherently sequential
        cumsum = 0.0
        result = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        for i in range(BLOCK_SIZE):
            if start + i < dim_size:
                result = result.at[i].set(cumsum)
                cumsum = cumsum + x_vals[i]
        
        # Store results
        tl.store(out_ptr + offsets, result.to(x_ptr.dtype.element_ty), mask=mask)


def triton_exclusive_cumsum(x: torch.Tensor, dim: int):
    """
    Compute exclusive cumulative sum along specified dimension using Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get shape information
    shape = x.shape
    dim_size = shape[dim]
    n_elements = x.numel()
    
    # Calculate strides
    stride = x.stride()
    stride_dim = stride[dim]
    
    # Calculate stride for other dimensions (for batch processing)
    if dim == 0:
        stride_other = 1
    else:
        stride_other = 1
        for i in range(dim):
            stride_other *= shape[i]
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # For 1D case (batch_size=1), we can treat it differently
    if len(shape) == 1:
        # Use a simpler kernel for 1D case
        BLOCK_SIZE = 256
        grid = (1,)  # Single batch
        
        # For 1D exclusive cumsum, we can use a more efficient approach
        exclusive_cumsum_1d_kernel[grid](
            x, out, dim_size, BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        # For multi-dimensional case, process each batch separately
        batch_size = shape[0] if dim != 0 else 1
        for b in range(batch_size):
            # Extract the slice for this batch
            if dim == 0:
                x_batch = x[b]
            else:
                # For dim=1 case in our problem
                x_batch = x[b] if len(shape) == 2 else x.view(-1, dim_size)[b]
            
            # Create a view with the dimension we want to process as last dimension
            if dim != len(shape) - 1:
                # Permute to move dim to the end
                dims = list(range(len(shape)))
                dims.pop(dim)
                dims.append(dim)
                x_batch = x_batch.permute(dims)
            
            # Reshape to 2D for easier processing
            batch_dim = x_batch.shape[0] if len(x_batch.shape) > 1 else 1
            inner_dim = x_batch.numel() // batch_dim if batch_dim > 0 else dim_size
            
            # Process each sub-batch
            for i in range(batch_dim):
                x_view = x_batch[i].contiguous() if len(x_batch.shape) > 1 else x_batch.contiguous()
                out_view = torch.empty_like(x_view)
                
                # Launch kernel for this 1D slice
                BLOCK_SIZE = 128
                grid_size = (dim_size + BLOCK_SIZE - 1) // BLOCK_SIZE
                
                exclusive_cumsum_1d_kernel[grid_size](
                    x_view, out_view, dim_size, BLOCK_SIZE=BLOCK_SIZE
                )
                
                # Store result back
                if len(shape) == 2:
                    out[b] = out_view
                else:
                    # Reshape and store back to original position
                    out_view = out_view.view(x_batch.shape[1:] if len(x_batch.shape) > 1 else x_batch.shape)
                    if dim != len(shape) - 1:
                        # Permute back
                        back_dims = list(range(len(shape)-1))
                        back_dims.insert(dim, len(shape)-1)
                        out_view = out_view.permute(back_dims)
                    out[b] = out_view
    
    return out


@triton.jit
def exclusive_cumsum_1d_kernel(
    x_ptr,
    out_ptr,
    dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for 1D exclusive cumulative sum.
    """
    # For exclusive cumsum, we need sequential processing
    # Use a simple approach: compute cumsum first, then shift
    # But for better performance, we'll do it in a single pass
    
    # This kernel processes one dimension at a time
    # We'll use a simple sequential algorithm since cumsum is inherently sequential
    
    # For better performance, we'll use a two-pass approach:
    # Pass 1: Compute regular cumsum
    # Pass 2: Shift and zero first element
    
    # However, for simplicity and correctness, let's use a direct approach
    cumsum = 0.0
    for i in range(dim_size):
        offset = i
        # Load current value
        val = tl.load(x_ptr + offset)
        # Store previous cumsum (exclusive)
        tl.store(out_ptr + offset, cumsum)
        # Update cumsum
        cumsum = cumsum + val


class ModelNew(nn.Module):
    """
    Optimized model that performs an exclusive cumulative sum using Triton kernel.

    Parameters:
        dim (int): The dimension along which to perform the exclusive cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Use our optimized Triton kernel for exclusive cumsum
        # For the specific case in the problem (dim=1), we can optimize
        if self.dim == 1 and len(x.shape) == 2:
            return triton_exclusive_cumsum_2d(x, 1)
        else:
            return triton_exclusive_cumsum(x, self.dim)


@triton.jit
def exclusive_cumsum_2d_kernel(
    x_ptr,  # Input tensor (2D)
    out_ptr,  # Output tensor (2D)
    batch_size,  # Number of rows
    dim_size,  # Number of columns
    stride_row,  # Stride for rows
    stride_col,  # Stride for columns
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for 2D exclusive cumulative sum along dimension 1.
    """
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Base offset for this row
    base_offset = row_idx * stride_row
    
    # Compute exclusive cumsum for this row
    cumsum = 0.0
    for col in range(dim_size):
        offset = base_offset + col * stride_col
        # Load current value
        val = tl.load(x_ptr + offset)
        # Store previous cumsum (exclusive)
        tl.store(out_ptr + offset, cumsum)
        # Update cumsum
        cumsum = cumsum + val


def triton_exclusive_cumsum_2d(x: torch.Tensor, dim: int):
    """
    Optimized exclusive cumsum for 2D tensors along specified dimension.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    shape = x.shape
    assert len(shape) == 2, "This kernel is optimized for 2D tensors"
    
    # Ensure dimension is 1 for our optimized kernel
    if dim == 1:
        out = torch.empty_like(x)
        batch_size, dim_size = shape
        
        # Calculate strides
        stride_row, stride_col = x.stride()
        
        # Launch kernel with one program per row
        grid = (batch_size,)
        BLOCK_SIZE = 128
        
        exclusive_cumsum_2d_kernel[grid](
            x, out, batch_size, dim_size, stride_row, stride_col, BLOCK_SIZE=BLOCK_SIZE
        )
        return out
    else:
        # For other dimensions, use the general approach
        return triton_exclusive_cumsum(x, dim)


# Register the custom autograd function for proper gradient computation
class ExclusiveCumsumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        ctx.dim = dim
        ctx.input_shape = x.shape
        if dim == 1 and len(x.shape) == 2:
            return triton_exclusive_cumsum_2d(x, dim)
        else:
            return triton_exclusive_cumsum(x, dim)
    
    @staticmethod
    def backward(ctx, grad_output):
        # The gradient of exclusive cumsum is the cumulative sum of gradients
        # Since exclusive_cumsum(x)[i] = sum(x[:i]), d/dx[j] exclusive_cumsum[i] = 1 if j < i else 0
        # So gradient flows back as cumulative sum from right to left (reverse cumsum)
        dim = ctx.dim
        grad_input = torch.empty_like(grad_output)
        
        if dim == 1 and len(ctx.input_shape) == 2:
            # For 2D case, we need reverse cumsum along dim=1
            grad_output_flipped = torch.flip(grad_output, dims=[1])
            # Compute cumsum along dim=1
            cumsum_flipped = torch.cumsum(grad_output_flipped, dim=1)
            # Flip back
            grad_input = torch.flip(cumsum_flipped, dims=[1])
        else:
            # General case: compute reverse cumsum
            grad_input = torch.flip(torch.cumsum(torch.flip(grad_output, dims=[dim]), dim=dim), dims=[dim])
        
        return grad_input, None


def triton_exclusive_cumsum_with_grad(x: torch.Tensor, dim: int):
    return ExclusiveCumsumFunction.apply(x, dim)


class ModelNew(nn.Module):
    """
    Optimized model that performs an exclusive cumulative sum using Triton kernel.

    Parameters:
        dim (int): The dimension along which to perform the exclusive cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Use our optimized Triton kernel with autograd support
        return triton_exclusive_cumsum_with_grad(x, self.dim)