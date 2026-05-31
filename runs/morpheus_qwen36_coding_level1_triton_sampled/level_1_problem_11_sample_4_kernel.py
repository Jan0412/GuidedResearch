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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Map program IDs to block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create offsets for the current block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Create masks for bounds checking
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A and B tiles with masking
        a_offsets = offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        b_offsets = (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
        
        a_tile = tl.load(A_ptr + a_offsets, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b_tile = tl.load(B_ptr + b_offsets, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Perform matrix multiplication tile
        acc += tl.dot(a_tile, b_tile)

    # Store result
    c_offsets = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptr + c_offsets, acc, mask=mask_c)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication using a custom Triton kernel.
    A: (M, K)
    B: (K, N)
    Returns: (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, "Inner dimensions must match."

    # Output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)

    # Block sizes (tunable)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid calculation
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
        1
    )

    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N
    )

    return C


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for 4D tensor-matrix multiplication.
    
    Performs: C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
    
    Strategy:
    1. Reshape A from (b, i, j, l) to (b*i*j, l).
    2. Perform matrix multiplication using Triton kernel.
    3. Reshape result back to (b, i, j, k).
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        # Get shapes
        b, i, j, l = A.shape
        l_b, k = B.shape
        
        assert l == l_b, "Dimensions of A and B must match on the multiplication axis."
        
        # Reshape A to 2D: (b*i*j, l)
        # This creates a view, so it's efficient
        A_flat = A.reshape(b * i * j, l)
        
        # Perform matmul using Triton kernel
        # Output shape will be (b*i*j, k)
        C_flat = triton_matmul(A_flat, B)
        
        # Reshape back to 4D: (b, i, j, k)
        C = C_flat.reshape(b, i, j, k)
        
        return C


def get_inputs():
    b = 8
    i = 256
    j = 512
    l = 256
    k = 768
    
    A = torch.rand(b, i, j, l)
    B = torch.rand(l, k)
    return [A, B]


def get_init_inputs():
    return []