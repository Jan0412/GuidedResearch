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
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create row and column offsets for the output tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Masks for output tile boundaries
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Accumulator for the output tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over the K dimension
    for k in range(0, K, BLOCK_K):
        # Current K offsets
        offs_k_curr = k + offs_k
        
        # Load tile from A
        # A has shape (K, M). We need A[k_block+dk, i_block+di]
        # Row indices: k_block + dk
        # Col indices: i_block + di
        # Offset = row * M + col
        row_offsets_a = offs_k_curr[:, None]
        col_offsets_a = (pid_m * BLOCK_M + offs_m)[None, :]
        offsets_a = row_offsets_a * stride_am + col_offsets_a * stride_ak
        mask_a = col_offsets_a < M
        A_tile = tl.load(A_ptr + offsets_a, mask=mask_a, other=0.0)
        
        # Load tile from B
        # B has shape (N, K). We need B[k_block+dk, j_block+dj]
        # Row indices: k_block + dk
        # Col indices: j_block + dj
        # Offset = row * N + col
        row_offsets_b = offs_k_curr[:, None]
        col_offsets_b = (pid_n * BLOCK_N + offs_n)[None, :]
        offsets_b = row_offsets_b * stride_bk + col_offsets_b * stride_bn
        mask_b = col_offsets_b < N
        B_tile = tl.load(B_ptr + offsets_b, mask=mask_b, other=0.0)
        
        # Compute dot product: A_tile.T @ B_tile
        # A_tile is (BLOCK_K, BLOCK_M), B_tile is (BLOCK_K, BLOCK_N)
        # A_tile.T is (BLOCK_M, BLOCK_K)
        # Result is (BLOCK_M, BLOCK_N)
        acc += tl.dot(A_tile.T, B_tile)
    
    # Store result to C
    # C has shape (M, N)
    row_offsets_c = (pid_m * BLOCK_M + offs_m)[:, None]
    col_offsets_c = (pid_n * BLOCK_N + offs_n)[None, :]
    offsets_c = row_offsets_c * stride_cm + col_offsets_c * stride_cn
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptr + offsets_c, acc, mask=mask_c)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication C = A.T @ B.T using a custom Triton kernel.
    A has shape (K, M), B has shape (N, K).
    Result C has shape (M, N).
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M = A.shape[1]
    K = A.shape[0]
    N = B.shape[0]
    
    # Prepare output tensor
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    # Block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    
    # Grid calculation
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # Strides
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)
    stride_bn = B.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication C = A.T @ B.T using Triton kernel.
        """
        return triton_matmul(A, B)