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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Custom Triton kernel for matrix multiplication C = A @ B.
    Optimized for cases where K is small (e.g., K=32) and M, N are large.
    Uses tl.dot for efficient tensor core utilization on blocks.
    """
    # Program ID determines the block of C this thread block computes
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create offsets for the current block in M and N dimensions
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)

    # Create masks to handle boundaries if M or N are not multiples of block sizes
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    mask_k = offsets_k < K

    # Base pointers for A and B blocks
    # A is (M, K), C is (M, N), B is (K, N)
    # A block pointer: starts at row pid_m*BLOCK_M, col 0
    # Strides: stride_am, stride_ak
    a_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0)  # A is row-major, but we load blocks; order=(1,0) means inner dim is K
    )

    # B block pointer: starts at row 0, col pid_n*BLOCK_N
    # B is (K, N), strides: stride_bk, stride_bn
    b_block_ptr = tl.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1)  # B is row-major, order=(0,1) matches
    )

    # Load A and B blocks into registers
    # Since K is small and matches BLOCK_K, we load the entire K dimension at once
    a_block = tl.load(a_block_ptr, boundary_check=[0, 1], padding_option="zero")
    b_block = tl.load(b_block_ptr, boundary_check=[0, 1], padding_option="zero")

    # Perform matrix multiplication on the blocks using tl.dot
    # This leverages tensor cores for high throughput
    c_block = tl.dot(a_block, b_block)

    # Create pointer for the output C block
    c_block_ptr = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )

    # Store the result
    tl.store(c_block_ptr, c_block, boundary_check=[0, 1])


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the custom Triton matmul kernel.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "FP32 precision required."
    
    # Ensure inputs are contiguous for optimal memory access
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    K_b, N = B.shape
    
    assert K == K_b, f"Inner dimensions mismatch: {K} vs {K_b}"

    # Allocate output tensor
    C = torch.empty((M, N), dtype=torch.float32, device='cuda')

    # Define block sizes
    # Optimized for K=32: BLOCK_K covers the entire K dimension
    BLOCK_M = 256
    BLOCK_N = 256
    BLOCK_K = K  # K is small (32), so we load all K at once

    # Grid configuration
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), 1)

    # Launch kernel
    matmul_kernel[grid](
        A.data_ptr(), B.data_ptr(), C.data_ptr(),
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K
    )

    return C


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A, B):
        # Use custom Triton kernel for matrix multiplication
        return triton_matmul(A, B)