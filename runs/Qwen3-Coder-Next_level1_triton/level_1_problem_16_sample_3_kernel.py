import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_transpose_kernel(
    A_ptr,  # Pointer to A (K x M), but we treat it as (M x K) due to transpose
    B_ptr,  # Pointer to B (K x N)
    C_ptr,  # Pointer to output C (M x N)
    M, N, K,  # Dimensions
    stride_am, stride_ak,  # Strides for A (after transpose: A^T is MxK)
    stride_bk, stride_bn,  # Strides for B (KxN)
    stride_cm, stride_cn,  # Strides for C (MxN)
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program IDs
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create block offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to A^T (we access A as if it's MxK, but stored as KxM)
    # A^T[offs_m, :] = A[:, offs_m], so A_ptr[offs_k, offs_m] with stride_ka and stride_am
    a_ptrs = A_ptr + (offs_k[:, None] * stride_ak + offs_m[None, :] * stride_am)
    
    # Pointers to B: B[offs_k, offs_n]
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    # Accumulator for C
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Matrix multiply loop
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load tiles
        a = tl.load(a_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # Matrix multiplication (a is (BLOCK_SIZE_K x BLOCK_SIZE_M), b is (BLOCK_SIZE_K x BLOCK_SIZE_N))
        # But since A is actually stored as KxM and we want A^T @ B, 
        # we are accessing A correctly as MxK (A^T) with the right indexing
        acc += tl.dot(a, b, out_dtype=tl.float32)
        
        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store result
    c = acc.to(tl.float32)  # Keep as FP32 as required
    
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul_transpose(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes C = A^T @ B using a Triton kernel.
    
    Args:
        A: Input tensor of shape (K, M)
        B: Input tensor of shape (K, N)
    
    Returns:
        Output tensor of shape (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only FP32 supported"
    
    A = A.contiguous()
    B = B.contiguous()
    
    K, M = A.shape
    _, N = B.shape
    
    # Output tensor
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    # Strides
    stride_am = A.stride(1)
    stride_ak = A.stride(0)
    stride_bk = B.stride(0)
    stride_bn = B.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)
    
    # Tunable block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Grid
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
    
    # Launch kernel
    matmul_transpose_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for matrix multiplication with transpose.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication: C = A^T @ B using a custom Triton kernel.
        
        Args:
            A: Input tensor of shape (M, K) - note: in the original code, A is (K, M)
            B: Input tensor of shape (K, N) - note: in the original code, B is (K, N)
            
        Returns:
            Output tensor of shape (M, N)
        """
        return triton_matmul_transpose(A, B)