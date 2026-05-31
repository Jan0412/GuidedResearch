import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Map program ID to the block of C it should compute.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Create pointers for the first blocks of A and B.
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Initialize accumulator with zero.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over k dimension.
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A and B tiles.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        
        # Compute dot product.
        accumulator += tl.dot(a, b)
        
        # Advance pointers.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Apply activation if needed.
    if ACTIVATION == "relu":
        accumulator = tl.where(accumulator > 0, accumulator, 0.0)
    
    # Create pointer to C.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    
    # Store the result.
    tl.store(c_ptrs, accumulator, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

@triton.jit
def triu_kernel(
    x_ptr, y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the global thread index
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M * N
    
    # Calculate row and column indices
    row = offsets // N
    col = offsets % N
    
    # Load input value
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Set to zero if below diagonal
    condition = row <= col
    y = tl.where(condition, x, 0.0)
    
    # Store result
    tl.store(y_ptr + offsets, y, mask=mask)

def triton_matmul_triangular(A, B):
    """Custom Triton implementation of triangular matrix multiplication"""
    assert A.is_cuda and B.is_cuda, "Both tensors must be on CUDA"
    assert A.shape[1] == B.shape[0], "Matrix dimensions must be compatible for multiplication"
    assert A.shape[0] == A.shape[1] and B.shape[0] == B.shape[1], "Matrices must be square"
    
    M, K = A.shape
    K, N = B.shape
    
    # Create output tensor
    C = torch.empty(M, N, device=A.device, dtype=torch.float32)
    
    # Configure block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Grid configuration
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        ACTIVATION=None
    )
    
    return C

def triton_triu(X):
    """Custom Triton implementation of upper triangular masking"""
    assert X.is_cuda, "Tensor must be on CUDA"
    
    M, N = X.shape
    Y = torch.empty_like(X)
    
    BLOCK_SIZE = 1024
    grid = lambda meta: ((M * N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    triu_kernel[grid](
        X, Y,
        M, N,
        X.stride(0), X.stride(1),
        Y.stride(0), Y.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return Y

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for matrix multiplication and upper triangular operations.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices using Triton optimizations.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        # Use Triton kernel for matrix multiplication
        C = triton_matmul_triangular(A, B)
        # Use Triton kernel for upper triangular masking
        return triton_triu(C)