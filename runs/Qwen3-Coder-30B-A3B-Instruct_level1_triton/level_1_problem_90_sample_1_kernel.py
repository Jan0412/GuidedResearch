import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def cumulative_product_kernel(
    input_ptr,
    output_ptr,
    N,
    dim_size,
    stride,
    BLOCK_SIZE: tl.constexpr
):
    # Get the block ID
    block_id = tl.program_id(0)
    
    # Calculate the starting position for this block
    start_pos = block_id * BLOCK_SIZE
    
    # Each thread processes one element
    offsets = start_pos + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to avoid going out of bounds
    mask = offsets < N
    
    # Load input values
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=1.0)
    
    # Initialize cumulative product
    cumprod = 1.0
    
    # Process elements sequentially to compute cumulative product
    # This approach assumes we're processing along a single dimension
    # For simplicity, we'll process along the last dimension
    
    # For each element, compute cumulative product
    for i in range(dim_size):
        # Calculate actual index considering the stride
        idx = i * stride + offsets
        # Check bounds
        valid_mask = (idx >= 0) & (idx < N) & mask
        
        # Load current value
        val = tl.load(input_ptr + idx, mask=valid_mask, other=1.0)
        
        # Update cumulative product
        cumprod = cumprod * val
        
        # Store result
        tl.store(output_ptr + idx, cumprod, mask=valid_mask)

@triton.jit
def cumulative_product_kernel_fused(
    input_ptr,
    output_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr
):
    # Get the block ID
    block_id = tl.program_id(0)
    
    # Each block handles one sequence
    seq_id = block_id
    
    # Ensure we don't exceed batch size
    if seq_id >= batch_size:
        return
    
    # Each sequence is processed by one block
    # Compute cumulative product for this sequence
    for i in range(seq_len):
        # Calculate offset for this position
        offset = seq_id * seq_len + i
        
        # Load previous cumulative product (or initialize to 1)
        if i == 0:
            cumprod = 1.0
        else:
            prev_offset = seq_id * seq_len + (i - 1)
            cumprod = tl.load(output_ptr + prev_offset)
        
        # Load current input value
        input_val = tl.load(input_ptr + offset)
        
        # Compute new cumulative product
        cumprod = cumprod * input_val
        
        # Store result
        tl.store(output_ptr + offset, cumprod)

def triton_cumprod(x: torch.Tensor, dim: int):
    """
    Triton implementation of cumulative product along a specific dimension.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Handle different dimensions
    if dim == -1 or dim == len(x.shape) - 1:
        # Most common case: last dimension
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        seq_len = x.shape[-1]
        
        # Prepare output tensor
        out = torch.empty_like(x)
        
        # Number of elements
        N = x.numel()
        
        # Block size
        BLOCK_SIZE = 1024
        
        # Grid size
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
        
        # Launch kernel
        cumulative_product_kernel_fused[grid](x, out, batch_size, seq_len, BLOCK_SIZE=BLOCK_SIZE)
        
        return out
    else:
        # For other dimensions, use PyTorch default (fallback)
        return torch.cumprod(x, dim=dim)

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for cumulative product operations.
    """
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass using Triton kernel for cumulative product.
        """
        return triton_cumprod(x, self.dim)