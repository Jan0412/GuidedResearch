import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def tril_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate starting positions for this block
    row_start = pid_m * BLOCK_SIZE
    col_start = pid_n * BLOCK_SIZE
    
    # Shared memory for tiles
    a_tile = tl.shared_ptr(tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32), 
                          shape=(BLOCK_SIZE, BLOCK_SIZE))
    b_tile = tl.shared_ptr(tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32), 
                          shape=(BLOCK_SIZE, BLOCK_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Loop over tiles
    for k in range(0, N, BLOCK_SIZE):
        # Load tile of A
        a_k = k + tl.arange(0, BLOCK_SIZE)
        a_m = row_start + tl.arange(0, BLOCK_SIZE)
        a_mask = (a_m[:, None] < N) & (a_k[None, :] < N)
        
        a_load = tl.load(a_ptr + a_m[:, None] * N + a_k[None, :], mask=a_mask, other=0.0)
        
        # Load tile of B
        b_k = k + tl.arange(0, BLOCK_SIZE)
        b_n = col_start + tl.arange(0, BLOCK_SIZE)
        b_mask = (b_k[:, None] < N) & (b_n[None, :] < N)
        
        b_load = tl.load(b_ptr + b_k[:, None] * N + b_n[None, :], mask=b_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(a_load, b_load)
    
    # Apply lower triangular mask
    row_indices = row_start + tl.arange(0, BLOCK_SIZE)[:, None]
    col_indices = col_start + tl.arange(0, BLOCK_SIZE)[None, :]
    mask = row_indices >= col_indices
    
    # Store result
    c_m = row_start + tl.arange(0, BLOCK_SIZE)
    c_n = col_start + tl.arange(0, BLOCK_SIZE)
    c_mask = (c_m[:, None] < N) & (c_n[None, :] < N) & mask
    
    out = tl.where(c_mask, acc, 0.0)
    tl.store(c_ptr + c_m[:, None] * N + c_n[None, :], out, mask=c_mask)

def triton_tril_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Triton implementation of lower triangular matrix multiplication
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    assert A.shape == B.shape, "Matrices must have same shape"
    assert len(A.shape) == 2, "Input tensors must be 2D"
    
    N = A.shape[0]
    BLOCK_SIZE = 128
    
    # Ensure tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Allocate output tensor
    C = torch.zeros(N, N, dtype=torch.float32, device='cuda')
    
    # Grid dimensions
    grid = (triton.cdiv(N, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))
    
    # Launch kernel
    tril_matmul_kernel[grid](A, B, C, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B.
        """
        # Use our Triton kernel for the computation
        result = triton_tril_matmul(A, B)
        return result