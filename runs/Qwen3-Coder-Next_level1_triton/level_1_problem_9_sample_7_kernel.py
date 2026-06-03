import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Matrix multiplication kernel for C = A @ B
    # A: (M, K), B: (K, N), C: (M, N)
    
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute starting positions for blocks
    # Use swizzling to improve cache locality (similar to CUTLASS)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid_m // GROUP_M
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid_m % group_size_m)
    pid_n = (pid_m % GROUP_M) * num_pid_n + (pid_n // GROUP_M)
    
    # Create block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Create masks
    amask = offs_m[:, None] < M
    bmask = offs_n[None, :] < N
    kmask = offs_k < K
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_offset = k + offs_k
        
        # Load A block: (BLOCK_M, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + k_offset[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=amask & (k_offset[None, :] < K), other=0.0)
        
        # Load B block: (BLOCK_K, BLOCK_N)
        b_ptrs = B_ptr + (k_offset[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=kmask[:, None] & bmask, other=0.0)
        
        # Accumulate matrix multiplication
        acc = tl.dot(a, b, acc)
    
    # Convert accumulator to float16 if needed, but keep float32 for precision
    acc = acc.to(C_ptr.dtype.element_ty)
    
    # Store result
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Optimized matrix multiplication using Triton kernel.
    Supports both A @ B and A.T @ B.T etc. by ensuring proper memory layout.
    """
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Check dimensions
    assert A.dim() == 2 and B.dim() == 2, "Only 2D tensors supported"
    assert A.shape[1] == B.shape[0], f"Dimension mismatch: {A.shape} @ {B.shape}"
    
    M, K = A.shape
    _, N = B.shape
    
    # Allocate output
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Set block sizes for optimization (tuned for the problem sizes given: M=32768, N=32)
    # With M >> N, we want more blocks in M dimension and fewer in N dimension
    BLOCK_M = 128
    BLOCK_N = 32
    BLOCK_K = 32
    GROUP_M = 8
    
    # Grid dimensions
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication using optimized Triton kernel.
        """
        return triton_matmul(A, B)