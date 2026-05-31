import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def tril_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Block start positions
    block_start_m = pid_m * BLOCK_SIZE
    block_start_n = pid_n * BLOCK_SIZE
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Loop over tiles
    for k in range(0, N, BLOCK_SIZE):
        # Load A tile
        a_tile = tl.load(
            A_ptr + (block_start_m + tl.arange(0, BLOCK_SIZE)[:, None]) * N +
            (k + tl.arange(0, BLOCK_SIZE)[None, :]),
            mask=(block_start_m + tl.arange(0, BLOCK_SIZE)[:, None] < N) &
                  (k + tl.arange(0, BLOCK_SIZE)[None, :] < N),
            other=0.0
        )
        
        # Load B tile
        b_tile = tl.load(
            B_ptr + (k + tl.arange(0, BLOCK_SIZE)[:, None]) * N +
            (block_start_n + tl.arange(0, BLOCK_SIZE)[None, :]),
            mask=(k + tl.arange(0, BLOCK_SIZE)[:, None] < N) &
                  (block_start_n + tl.arange(0, BLOCK_SIZE)[None, :] < N),
            other=0.0
        )
        
        # Accumulate
        acc += tl.dot(a_tile, b_tile)
    
    # Apply lower triangular mask
    row_indices = block_start_m + tl.arange(0, BLOCK_SIZE)[:, None]
    col_indices = block_start_n + tl.arange(0, BLOCK_SIZE)[None, :]
    
    # Create triangular mask
    triangular_mask = row_indices >= col_indices
    
    # Apply mask and store result
    c_tile = acc * triangular_mask
    
    # Store result
    tl.store(
        C_ptr + (block_start_m + tl.arange(0, BLOCK_SIZE)[:, None]) * N +
        (block_start_n + tl.arange(0, BLOCK_SIZE)[None, :]),
        c_tile,
        mask=(block_start_m + tl.arange(0, BLOCK_SIZE)[:, None] < N) &
              (block_start_n + tl.arange(0, BLOCK_SIZE)[None, :] < N)
    )

def triton_tril_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Triton implementation of matrix multiplication with lower triangular masking.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    assert A.shape == B.shape, "Matrices must have the same shape"
    assert len(A.shape) == 2, "Input tensors must be 2D"
    
    N = A.shape[0]
    BLOCK_SIZE = 64  # Tuned block size
    
    # Ensure tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Allocate output tensor
    C = torch.empty(N, N, dtype=torch.float32, device=A.device)
    
    # Calculate grid
    grid = (triton.cdiv(N, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))
    
    # Launch kernel
    tril_matmul_kernel[grid](A, B, C, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return C

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for matrix multiplication with lower triangular masking.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B.
        Uses Triton kernel for optimized performance.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        return triton_tril_matmul(A, B)

M = 4096

def get_inputs():
    A = torch.rand(M, M)
    B = torch.rand(M, M)
    A = torch.tril(A)
    B = torch.tril(B)
    return [A.cuda(), B.cuda()]

def get_init_inputs():
    return []