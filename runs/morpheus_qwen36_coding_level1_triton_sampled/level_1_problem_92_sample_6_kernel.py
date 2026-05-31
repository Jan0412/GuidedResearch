import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def exclusive_cumsum_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    dim_size,
    stride_dim,
    stride_other,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID corresponds to a unique index in the batch dimensions
    pid = tl.program_id(0)
    
    # Base pointer for this program's slice
    base_ptr = x_ptr + pid * stride_other
    
    # Initialize cumulative sum
    cum_sum = 0.0
    
    # Iterate over the dimension with BLOCK_SIZE chunks
    for block_start in range(0, dim_size, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, dim_size)
        block_size = block_end - block_start
        
        # Create offsets for the current block
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim_size
        
        # Load input values
        x = tl.load(base_ptr + offsets * stride_dim, mask=mask, other=0.0)
        
        # Compute exclusive cumsum for this block
        # We need to accumulate the sum from previous blocks
        for i in range(block_size):
            if block_start + i < dim_size:
                out_val = cum_sum
                cum_sum += x[i]
                tl.store(out_ptr + (pid * stride_other) + (block_start + i) * stride_dim, out_val, mask=(block_start + i < n_elements))


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for exclusive cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
            
        # Ensure x is contiguous for efficient memory access
        x = x.contiguous()
        
        # Get shape and strides
        shape = x.shape
        n_elements = x.numel()
        dim_size = shape[self.dim]
        stride_dim = x.stride(self.dim)
        
        # Compute stride_other: product of strides of all dimensions except dim
        stride_other = 1
        for i in range(len(shape)):
            if i != self.dim:
                stride_other *= shape[i]
        
        # Prepare output tensor
        out = torch.empty_like(x)
        
        # Grid: one program per batch element
        grid = (n_elements // dim_size,)
        
        # Choose block size based on dim_size
        BLOCK_SIZE = triton.next_power_of_2(dim_size)
        if BLOCK_SIZE > 1024:
            BLOCK_SIZE = 1024
            
        # Launch Triton kernel
        exclusive_cumsum_kernel[grid](
            x_ptr=x.data_ptr(),
            out_ptr=out.data_ptr(),
            n_elements=n_elements,
            dim_size=dim_size,
            stride_dim=stride_dim,
            stride_other=stride_other,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return out


def get_inputs():
    return [torch.rand(32768, 32768)]

def get_init_inputs():
    return [1]