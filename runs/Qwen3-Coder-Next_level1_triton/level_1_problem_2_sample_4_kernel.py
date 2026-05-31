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
    GROUP_SIZE_M: tl.constexpr
):
    # Matrix multiplication kernel
    # Each program handles a block of the output matrix C of size BLOCK_SIZE_M x BLOCK_SIZE_N
    
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create ranges for m and n dimensions
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create pointer arrays for A and B
    # A has shape (M, K), so we need offsets for m and k dimensions
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    A_ptrs = A_ptr + (offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn)
    
    # Initialize accumulator for the dot product
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over the K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks from A and B
        a = tl.load(A_ptrs, mask=offsets_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(B_ptrs, mask=offsets_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # Accumulate the matrix multiplication
        accumulator = tl.dot(a, b, accumulator)
        
        # Update pointers for next iteration
        A_ptrs += BLOCK_SIZE_K * stride_ak
        B_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Convert accumulator to float16 if needed (but we're using FP32 per requirements)
    C_block = accumulator.to(tl.float32)
    
    # Store the result
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    C_ptrs = C_ptr + (offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn)
    c_mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    tl.store(C_ptrs, C_block, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication using Triton kernel.
    
    Args:
        A: Input tensor of shape (M, K)
        B: Input tensor of shape (K, N)
    
    Returns:
        Output tensor of shape (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape[1] == B.shape[0], "Incompatible matrix dimensions"
    
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, "Inner dimensions must match"
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Set block sizes (tunable parameters)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    
    # Grid size
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
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
        GROUP_SIZE_M=8,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using Triton kernel
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using custom Triton kernel.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)