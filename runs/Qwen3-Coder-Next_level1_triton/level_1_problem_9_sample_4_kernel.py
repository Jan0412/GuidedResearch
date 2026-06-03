import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr = 3,
    GROUP_M: tl.constexpr = 8,
):
    # Matrix multiplication kernel optimized for M >> N case
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Create block offsets
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Create pointers for A and B
    A_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_mask = k + offs_k < K
        a = tl.load(A_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(B_ptrs, mask=k_mask[:, None], other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk
    
    # Store result
    C_ptrs = C + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(C_ptrs, accumulator, mask=mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Optimized Triton kernel for matrix multiplication.
    Optimized for cases where M >> N or N >> M.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Matrix dimensions must match for multiplication"
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Configure block sizes optimized for M >> N case
    BLOCK_M = 128
    BLOCK_N = 32
    BLOCK_K = 32
    
    # Calculate grid dimensions
    num_blocks_m = (M + BLOCK_M - 1) // BLOCK_M
    num_blocks_n = (N + BLOCK_N - 1) // BLOCK_N
    grid = (num_blocks_m * num_blocks_n,)
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NUM_STAGES=3,
        GROUP_M=8,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for matrix multiplication.
    Specifically optimized for cases where one dimension is much larger than the other.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the optimized matrix multiplication using Triton kernel.
        
        Args:
            A (torch.Tensor): Input matrix of shape (M, K) or (K, M)
            B (torch.Tensor): Input matrix of shape (K, N) or (N, K)
            
        Returns:
            torch.Tensor: Output matrix of shape (M, N) or (N, M)
        """
        return triton_matmul(A, B)