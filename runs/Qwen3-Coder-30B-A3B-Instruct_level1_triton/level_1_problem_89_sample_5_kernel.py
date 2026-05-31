import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def scan_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    dim_size,
    stride_x,
    stride_out,
    BLOCK_SIZE: tl.constexpr,
    IS_SCAN_FORWARD: tl.constexpr
):
    # Get the block index
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Perform inclusive scan operation
    if IS_SCAN_FORWARD:
        # Forward scan (cumulative sum from left to right)
        accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for i in range(dim_size):
            # Calculate the offset for current position
            pos_offset = i * stride_x
            # Load current element
            current_val = tl.load(x_ptr + pos_offset + offsets, mask=mask, other=0.0)
            # Accumulate
            accumulator = accumulator + current_val
            # Store result
            tl.store(out_ptr + pos_offset + offsets, accumulator, mask=mask)
    else:
        # Backward scan (cumulative sum from right to left)
        accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for i in range(dim_size - 1, -1, -1):
            # Calculate the offset for current position
            pos_offset = i * stride_x
            # Load current element
            current_val = tl.load(x_ptr + pos_offset + offsets, mask=mask, other=0.0)
            # Accumulate
            accumulator = accumulator + current_val
            # Store result
            tl.store(out_ptr + pos_offset + offsets, accumulator, mask=mask)

def triton_scan(x: torch.Tensor, dim: int):
    """
    Triton implementation of cumulative sum (scan) operation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Get dimensions
    shape = x.shape
    dim_size = shape[dim]
    batch_size = 1
    for i in range(len(shape)):
        if i != dim:
            batch_size *= shape[i]
    
    # Calculate strides
    stride_x = 1
    stride_out = 1
    for i in range(dim + 1, len(shape)):
        stride_x *= shape[i]
        stride_out *= shape[i]
    
    # Flatten to 1D for easier processing
    flat_size = x.numel()
    BLOCK_SIZE = 128
    
    # Grid configuration
    grid = lambda meta: (triton.cdiv(flat_size, meta["BLOCK_SIZE"]),)
    
    # For simplicity, we'll use a simpler approach for now
    # This is a basic implementation - a more optimized version would use shared memory
    
    # Direct approach using a simpler kernel
    @triton.jit
    def simple_scan_kernel(
        x_ptr,
        out_ptr,
        size: tl.constexpr,
        dim_size: tl.constexpr,
        stride_x: tl.constexpr,
        stride_out: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        block_start = tl.program_id(0) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < size
        
        # Process along the specified dimension
        for i in range(dim_size):
            # Calculate the actual position
            pos = i * stride_x
            # Load current element
            val = tl.load(x_ptr + pos + offsets, mask=mask, other=0.0)
            # If this is the first element, store it directly
            if i == 0:
                tl.store(out_ptr + pos + offsets, val, mask=mask)
            else:
                # Otherwise, accumulate from previous result
                prev_pos = (i - 1) * stride_x
                prev_val = tl.load(out_ptr + prev_pos + offsets, mask=mask, other=0.0)
                result = prev_val + val
                tl.store(out_ptr + pos + offsets, result, mask=mask)
    
    # Run the kernel
    simple_scan_kernel[grid](
        x, out, flat_size, dim_size, stride_x, stride_out, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_scan(x, self.dim)