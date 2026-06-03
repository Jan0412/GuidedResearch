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
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    # Program ID represents the block of rows (BLOCK_SIZE_M) and columns (BLOCK_SIZE_N)
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create offsets for rows and columns
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for bounds checking
    am_mask = offsets_am[:, None] < M
    bn_mask = offsets_bn[None, :] < N
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute K offsets for this iteration
        k_offset = k * BLOCK_SIZE_K
        k_mask = k_offset + offsets_k < K
        
        # Load A block
        a_offset = (offsets_am[:, None] * stride_am + (k_offset + offsets_k)[None, :] * stride_ak)
        a = tl.load(A + a_offset, mask=am_mask & (k_offset + offsets_k)[None, :] < K, other=0.0)
        
        # Load B block
        b_offset = ((k_offset + offsets_k)[:, None] * stride_bk + offsets_bn[None, :] * stride_bn)
        b = tl.load(B + b_offset, mask=k_mask[:, None] & bn_mask, other=0.0)
        
        # Accumulate matrix multiplication
        accumulator += tl.dot(a, b)
    
    # Convert accumulator to float16 if needed, but for FP32 we keep it as is
    c = accumulator.to(tl.float32)
    
    # Store result
    c_offset = (offsets_am[:, None] * stride_cm + offsets_bn[None, :] * stride_cn)
    cm_mask = offsets_am[:, None] < M
    cn_mask = offsets_bn[None, :] < N
    tl.store(C + c_offset, c, mask=cm_mask & cn_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication using Triton kernel.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Incompatible dimensions"
    
    # Allocate output tensor
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    # Define block sizes for optimization
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Compute grid dimensions
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
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
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using Triton kernel.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)