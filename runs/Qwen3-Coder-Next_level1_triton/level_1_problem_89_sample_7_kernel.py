import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumsum_kernel(
    x_ptr,  # Input tensor pointer
    y_ptr,  # Output tensor pointer
    batch_size,  # Number of sequences
    seq_len,  # Length of each sequence
    stride_b,  # Stride between batches
    stride_s,  # Stride between elements in sequence
    BLOCK_SIZE: tl.constexpr,
):
    """
    Cumulative sum kernel along the last dimension (sequence dimension).
    Each batch is processed independently in parallel.
    """
    # Get batch index
    batch_id = tl.program_id(0)
    
    # Compute starting pointer for this batch
    x_offset = batch_id * stride_b
    y_offset = batch_id * stride_b
    
    # Initialize running sum
    cumsum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Process the sequence in blocks
    for start in range(0, seq_len, BLOCK_SIZE):
        # Compute current block's end
        end = tl.minimum(start + BLOCK_SIZE, seq_len)
        block_size_actual = end - start
        
        # Create offsets for this block
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len
        
        # Load input values
        x_offsets = x_offset + offsets * stride_s
        x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        
        # Convert to float32 for accumulation
        x_vals_f32 = x_vals.to(tl.float32)
        
        # Compute running sum within block
        cumsum = cumsum + x_vals_f32
        
        # Store result
        y_offsets = y_offset + offsets * stride_s
        tl.store(y_ptr + y_offsets, cumsum.to(x_ptr.dtype.element_ty), mask=mask)


@triton.jit
def cumsum_kernel_2d(
    x_ptr,
    y_ptr,
    n_rows,
    n_cols,
    stride_row,
    stride_col,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized cumsum kernel for 2D tensors along the last dimension.
    Uses a two-pass algorithm: first pass computes block sums, second pass adds cumulative block sums.
    """
    # Each program handles one row
    row_id = tl.program_id(0)
    
    # Compute starting pointers for this row
    x_row_start = x_ptr + row_id * stride_row
    y_row_start = y_ptr + row_id * stride_row
    
    # Phase 1: Compute block sums and store in temporary buffer (simulated with local memory)
    # Since we don't have external temporary buffer, we'll do a simple single-pass for now
    # For larger sequences, a two-pass algorithm would be more efficient
    
    cumsum = 0.0
    for col_id in range(0, n_cols, BLOCK_SIZE):
        # Process a block
        block_end = tl.minimum(col_id + BLOCK_SIZE, n_cols)
        block_size_actual = block_end - col_id
        
        # Offsets for this block
        offsets = col_id + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load input
        x_offsets = x_row_start + offsets * stride_col
        x_block = tl.load(x_offsets, mask=mask, other=0.0)
        x_block_f32 = x_block.to(tl.float32)
        
        # Compute cumulative sum within block
        cumsum_block = tl.cumsum(x_block_f32)
        
        # Add global cumulative sum
        cumsum_block = cumsum_block + cumsum
        
        # Update global cumulative sum
        cumsum = cumsum + tl.sum(x_block_f32)
        
        # Store result
        y_offsets = y_row_start + offsets * stride_col
        tl.store(y_offsets, cumsum_block.to(x_ptr.dtype.element_ty), mask=mask)


class TritonCumsumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Create output tensor
        out = torch.empty_like(x)
        
        # Handle dimensions
        if dim < 0:
            dim = x.dim() + dim
            
        # For 2D case (batch, seq) where dim=1, use optimized kernel
        if x.dim() == 2 and dim == 1:
            batch_size, seq_len = x.shape
            stride_b = x.stride(0)
            stride_s = x.stride(1)
            
            # Use block size that matches hardware efficiency
            BLOCK_SIZE = 128
            
            # Launch kernel: one program per batch
            grid = (batch_size,)
            
            cumsum_kernel[grid](
                x, out,
                batch_size, seq_len,
                stride_b, stride_s,
                BLOCK_SIZE=BLOCK_SIZE
            )
        else:
            # For higher dimensions or different dim, flatten and use 2D kernel
            # Reshape to 2D temporarily
            shape = x.shape
            if dim != x.dim() - 1:
                # Move dim to last position
                permute_dims = list(range(x.dim()))
                permute_dims.append(permute_dims.pop(dim))
                x_permuted = x.permute(permute_dims)
            else:
                x_permuted = x
                
            # Flatten all dimensions except last
            flat_shape = (-1, shape[dim]) if dim >= 0 else (-1, shape[dim + x.dim()])
            x_flat = x_permuted.reshape(flat_shape)
            
            # Process as 2D
            n_rows, n_cols = x_flat.shape
            stride_row = x_flat.stride(0)
            stride_col = x_flat.stride(1)
            
            BLOCK_SIZE = 128
            grid = (n_rows,)
            
            cumsum_kernel_2d[grid](
                x_flat, x_flat,
                n_rows, n_cols,
                stride_row, stride_col,
                BLOCK_SIZE=BLOCK_SIZE
            )
            
            # Reshape back
            out = x_flat.reshape(x_permuted.shape)
            
            # If we permuted, reverse the permutation
            if dim != x.dim() - 1:
                reverse_permute = [0] * len(permute_dims)
                for i, d in enumerate(permute_dims):
                    reverse_permute[d] = i
                out = out.permute(reverse_permute)
                
        ctx.save_for_backward(x)
        ctx.dim = dim
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # For cumsum, backward is cumsum in reverse direction
        # grad_input[i] = sum(grad_output[i:])
        # This is equivalent to reverse cumsum
        
        grad_output = grad_output.contiguous()
        
        if ctx.dim < 0:
            dim = ctx.dim + grad_output.dim()
        else:
            dim = ctx.dim
            
        # Simple implementation using torch for backward
        # Reverse along the dimension, cumsum, then reverse back
        grad_input = torch.flip(grad_output, [dim])
        grad_input = torch.cumsum(grad_input, dim=dim)
        grad_input = torch.flip(grad_input, [dim])
        
        return grad_input, None


def triton_cumsum(x, dim):
    return TritonCumsumFunction.apply(x, dim)


class ModelNew(nn.Module):
    """
    Optimized Scan model using custom Triton kernels for cumulative sum.
    """
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_cumsum(x, self.dim)