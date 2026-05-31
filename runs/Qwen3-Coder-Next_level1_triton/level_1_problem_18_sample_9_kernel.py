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
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
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

    # Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Pointers
    A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    # Accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Matrix multiplication loop
    for k in range(0, K, BLOCK_SIZE_K):
        # Load tiles
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(B_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        
        # Accumulate
        accumulator = tl.dot(a, b, accumulator)
        
        # Update pointers
        A_ptrs += BLOCK_SIZE_K * stride_ak
        B_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Convert to FP16 if needed (but keeping FP32 per instructions)
    C_block = accumulator.to(tl.float32)
    
    # Store result
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, C_block, mask=c_mask)


def triton_matmul_transposed(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes C = A^T @ B^T using custom Triton kernel.
    
    Note: torch.matmul(A.T, B.T) == (B @ A)^T, but we implement directly for clarity.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    # Input shapes: A is (K, M), B is (N, K)
    # So A.T is (M, K), B.T is (K, N)
    # Result should be (M, N)
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Incompatible shapes: A.shape={A.shape}, B.shape={B.shape}"
    
    # Output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Grid configuration
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
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
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication equivalent to torch.matmul(A.T, B.T)
        using a custom Triton kernel.
        """
        return triton_matmul_transposed(A, B)