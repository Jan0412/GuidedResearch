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
    # Matrix multiplication kernel optimized for FP32
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
    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Determine masks for boundary conditions
    am_mask = offset_m[:, None] < M
    bn_mask = offset_n[None, :] < N
    bk_mask = offset_k[None, :] < K
    
    # Load tile from A: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
    a_ptr = A_ptr + stride_am * offset_m[:, None] + stride_ak * offset_k[None, :]
    A = tl.load(a_ptr, mask=am_mask & bk_mask, other=0.0)
    
    # Load tile from B: shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
    # Note: B.T is needed in original code, so we access B as if it's already transposed
    # Since B is (N, K), we want B.T which is (K, N), so we use stride_bk and stride_bn directly
    b_ptr = B_ptr + stride_bk * offset_k[:, None] + stride_bn * offset_n[None, :]
    B = tl.load(b_ptr, mask=bk_mask[:, None] & bn_mask, other=0.0)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Matrix multiplication loop
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load more data as needed (with bounds checking)
        k_start = k * BLOCK_SIZE_K
        k_offset = k_start + offset_k
        
        # Re-apply masks for boundary conditions
        a_ptr = A_ptr + stride_am * offset_m[:, None] + stride_ak * k_offset[None, :]
        a_tile = tl.load(a_ptr, mask=am_mask & (k_offset[None, :] < K), other=0.0)
        
        b_ptr = B_ptr + stride_bk * k_offset[:, None] + stride_bn * offset_n[None, :]
        b_tile = tl.load(b_ptr, mask=(k_offset[:, None] < K) & bn_mask, other=0.0)
        
        # Accumulate multiplication
        accumulator += tl.dot(a_tile, b_tile, out_dtype=tl.float32)
    
    # Store the result
    c_ptr = C_ptr + stride_cm * offset_m[:, None] + stride_cn * offset_n[None, :]
    C = accumulator.to(tl.float32)  # Ensure output is float32
    c_mask = am_mask & bn_mask
    tl.store(c_ptr, C, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication C = A @ B.T using Triton kernel.
    Optimized for FP32 precision.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    # Validate dimensions
    M, K = A.shape
    N, K_b = B.shape
    assert K == K_b, f"Incompatible dimensions: A is {A.shape}, B is {B.shape}"
    
    # Output tensor
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    # Configure kernel parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Grid configuration
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
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
    Optimized version of Model using Triton kernel for matrix multiplication.
    Performs C = A @ B.T using an optimized Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using Triton kernel.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (N, K).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)