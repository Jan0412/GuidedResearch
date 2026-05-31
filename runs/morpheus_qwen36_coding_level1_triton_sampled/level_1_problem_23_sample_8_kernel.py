import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel1(
    x_ptr,
    out_ptr,
    l_ptr,
    batch_size,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * dim
    
    # Pointers for current row
    x_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    out_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    
    # Initialize max and sum
    M = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)
    L = tl.full((BLOCK_SIZE,), 0.0, dtype=tl.float32)
    
    # Iterate over blocks
    num_blocks = (dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    for b in range(num_blocks):
        block_start = b * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        # Load data
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Compute local max and sum
        local_m = tl.max(x, axis=0)
        local_l = tl.sum(tl.exp(x - local_m), axis=0)
        
        # Update global max and sum
        new_m = tl.maximum(M, local_m)
        L = L * tl.exp(M - new_m) + local_l * tl.exp(local_m - new_m)
        M = new_m
        
        # Store exp(x - M) for normalization later
        out = tl.exp(x - M)
        tl.store(out_ptr + offsets, out, mask=mask)
    
    # Store the final sum L for this row
    tl.store(l_ptr + pid, L)


@triton.jit
def softmax_kernel2(
    out_ptr,
    l_ptr,
    batch_size,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * dim
    
    # Load the normalization constant for this row
    L = tl.load(l_ptr + pid)
    
    # Iterate over blocks to normalize
    num_blocks = (dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    for b in range(num_blocks):
        block_start = b * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        # Load pre-computed exp values
        out = tl.load(out_ptr + offsets, mask=mask, other=0.0)
        
        # Normalize
        out = out / L
        
        # Store result
        tl.store(out_ptr + offsets, out, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    
    batch_size, dim = x.shape
    out = torch.empty_like(x)
    l_ptr = torch.empty(batch_size, dtype=torch.float32, device='cuda')
    
    BLOCK_SIZE = 4096  # Tunable block size
    
    # Grid configuration
    grid = (batch_size,)
    
    # First pass: compute exp(x - M) and store L
    softmax_kernel1[grid](x, out, l_ptr, batch_size, dim, BLOCK_SIZE=BLOCK_SIZE)
    
    # Second pass: normalize by L
    softmax_kernel2[grid](out, l_ptr, batch_size, dim, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_softmax(x)


def get_inputs():
    batch_size = 4096
    dim = 393216
    x = torch.rand(batch_size, dim, dtype=torch.float32, device='cuda')
    return [x]


def get_init_inputs():
    return []