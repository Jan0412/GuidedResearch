import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumprod_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    batch_size,  # Number of batches
    seq_len,  # Sequence length along the cumprod dimension
    dim,  # Dimension along which to compute cumprod
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for cumulative product along a specified dimension.
    Handles 2D case where dim=1 (most common for sequences).
    For higher dimensions, uses a flattened approach assuming contiguous layout.
    """
    # For simplicity, we handle the common case where we're cumprod along the last dimension
    # For general dim, we would need to compute strides, but for now assume dim=-1 or dim=1 for 2D
    # Since the example is 2D (batch_size, seq_len) and dim=1, we optimize for that
    
    # Batch index
    batch_idx = tl.program_id(0)
    
    if batch_idx >= batch_size:
        return

    # Compute starting offset for this batch
    # For dim=1, strides are: [seq_len, 1]
    batch_offset = batch_idx * seq_len

    # Create sequence offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len

    # Load first element
    x0 = tl.load(x_ptr + batch_offset + offsets, mask=mask, other=1.0)
    
    # Initialize cumulative product
    cumprod = tl.zeros_like(x0)
    
    # Compute cumulative product
    for i in range(seq_len):
        val = tl.load(x_ptr + batch_offset + i, mask=(i < seq_len), other=1.0)
        if i == 0:
            cumprod = val
        else:
            cumprod = cumprod * val
        tl.store(out_ptr + batch_offset + i, cumprod, mask=(i < seq_len))


# Optimized kernel for 2D cumprod along dimension 1 (most common case)
@triton.jit
def cumprod_kernel_2d_dim1(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    batch_size,  # Number of batches
    seq_len,  # Sequence length
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized Triton kernel for cumulative product along dimension 1 for 2D tensors.
    """
    # Batch index
    batch_idx = tl.program_id(0)
    
    if batch_idx >= batch_size:
        return

    # Compute starting offset for this batch
    batch_offset = batch_idx * seq_len

    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32) + 1.0
    
    # Process in chunks for better memory access pattern
    for start in range(0, seq_len, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len
        
        # Load current values
        x_val = tl.load(x_ptr + batch_offset + offsets, mask=mask, other=1.0).to(tl.float32)
        
        # Update cumulative product
        # For positions beyond the first chunk, multiply with previous cumulative result
        if start > 0:
            # Load previous cumulative product value from output
            prev_offset = batch_offset + start - 1
            prev_cumprod = tl.load(out_ptr + prev_offset)
            acc = acc * prev_cumprod
        
        # Compute cumulative product within this chunk
        chunk_cumprod = tl.zeros_like(x_val)
        for i in range(BLOCK_SIZE):
            if i == 0:
                chunk_cumprod = x_val
            else:
                chunk_cumprod = chunk_cumprod * x_val
        
        # Store results
        tl.store(out_ptr + batch_offset + offsets, chunk_cumprod, mask=mask)


# Even more optimized kernel using scan-like operations
@triton.jit
def cumprod_kernel_fused(
    x_ptr,
    out_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized cumulative product kernel using a segmented scan approach.
    Processes one batch per program, handles arbitrary sequence lengths.
    """
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return
    
    batch_offset = batch_idx * seq_len
    
    # Initialize accumulator
    acc = 1.0
    
    # Process element by element
    for i in range(seq_len):
        offset = batch_offset + i
        val = tl.load(x_ptr + offset)
        acc = acc * val
        tl.store(out_ptr + offset, acc)


# Best approach: use a simple loop per batch since cumprod is inherently sequential
@triton.jit
def cumprod_kernel_sequential(
    x_ptr,
    out_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Sequential cumulative product kernel - each batch processed by one program.
    This is optimal since cumprod has sequential dependencies.
    """
    batch_idx = tl.program_id(0)
    
    if batch_idx >= batch_size:
        return
    
    # Calculate base offset for this batch
    base = batch_idx * seq_len
    
    # Sequential accumulation
    acc = 1.0
    for i in range(seq_len):
        val = tl.load(x_ptr + base + i)
        acc = acc * val
        tl.store(out_ptr + base + i, acc)


def triton_cumprod(x: torch.Tensor, dim: int = 1):
    """
    Compute cumulative product along specified dimension using Triton kernel.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute cumulative product
        
    Returns:
        Tensor with cumulative product computed along dim
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    x = x.contiguous()
    
    # Ensure we're working with FP32
    if x.dtype != torch.float32:
        x = x.float()
    
    out = torch.empty_like(x)
    
    # Get dimensions
    if dim == 1 and x.dim() == 2:
        # Special case: 2D tensor, cumprod along dim=1
        batch_size, seq_len = x.shape
        BLOCK_SIZE = 32  # Small block size since we're doing sequential accumulation
        
        # Use the sequential kernel - most efficient for cumprod
        grid = (batch_size,)
        cumprod_kernel_sequential[grid](x, out, batch_size, seq_len, BLOCK_SIZE=BLOCK_SIZE)
    else:
        # General case: flatten all dimensions except the cumprod dimension
        # Reshape to 2D for easier processing
        original_shape = x.shape
        x_flat = x.transpose(dim, -1).reshape(-1, x.shape[dim])
        out_flat = torch.empty_like(x_flat)
        
        batch_size = x_flat.shape[0]
        seq_len = x_flat.shape[1]
        BLOCK_SIZE = 32
        
        grid = (batch_size,)
        cumprod_kernel_sequential[grid](x_flat, out_flat, batch_size, seq_len, BLOCK_SIZE=BLOCK_SIZE)
        
        # Reshape back to original shape
        out = out_flat.reshape(x.shape[:-1] + (x.shape[dim],)).transpose(-1, dim)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs cumulative product using Triton kernel.
    """
    
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return triton_cumprod(x, dim=self.dim)