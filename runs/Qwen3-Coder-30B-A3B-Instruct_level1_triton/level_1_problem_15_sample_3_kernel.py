import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def tril_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Tile coordinates
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load tiles
        a_tile = tl.load(
            A_ptr + m_start * stride_am + k * stride_ak,
            mask=(m_start + tl.arange(0, BLOCK_SIZE_M)[:, None] < M) &
                  (k + tl.arange(0, BLOCK_SIZE_K)[None, :] < K),
            other=0.0
        )
        
        b_tile = tl.load(
            B_ptr + k * stride_bk + n_start * stride_bn,
            mask=(k + tl.arange(0, BLOCK_SIZE_K)[:, None] < K) &
                  (n_start + tl.arange(0, BLOCK_SIZE_N)[None, :] < N),
            other=0.0
        )
        
        # Accumulate
        acc += tl.dot(a_tile, b_tile)
    
    # Apply lower triangular mask
    m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)[:, None]
    n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)[None, :]
    mask = m_offsets >= n_offsets
    
    # Store result
    tl.store(
        C_ptr + m_start * stride_cm + n_start * stride_cn,
        acc * mask,
        mask=(m_offsets < M) & (n_offsets < N)
    )

def triton_tril_matmul(A, B):
    """
    Custom Triton implementation of lower triangular matrix multiplication
    """
    assert A.is_cuda and B.is_cuda, "Both tensors must be on CUDA"
    assert A.shape[1] == B.shape[0], "Matrix dimensions incompatible for multiplication"
    assert A.shape[0] == A.shape[1] and B.shape[0] == B.shape[1], "Matrices must be square"
    
    M, K = A.shape
    _, N = B.shape
    
    # Prepare output tensor
    C = torch.zeros(M, N, device=A.device, dtype=torch.float32)
    
    # Configure block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # Grid size
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N)
    )
    
    # Launch kernel
    tril_matmul_kernel[grid](
        A, B, C,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return C

class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernels for lower triangular matrix multiplication
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B.
        Uses custom Triton kernel for better performance.

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