import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bmm_kernel(
    A_ptr,  # Pointer to input A of shape (batch_size, m, k)
    B_ptr,  # Pointer to input B of shape (batch_size, k, n)
    C_ptr,  # Pointer to output C of shape (batch_size, m, n)
    batch_size,  # Number of matrices in batch
    m, n, k,  # Matrix dimensions
    stride_ab, stride_am, stride_ak,  # Strides for A
    stride_bb, stride_bk, stride_bn,  # Strides for B
    stride_cb, stride_cm, stride_cn,  # Strides for C
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Triton kernel for batched matrix multiplication C = A @ B
    """
    # Get batch index
    batch_id = tl.program_id(2)
    
    # Compute matrix row/col indices for this batch
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Adjust batch pointers
    A_ptr = A_ptr + batch_id * stride_ab
    B_ptr = B_ptr + batch_id * stride_bb
    C_ptr = C_ptr + batch_id * stride_cb
    
    # Create tile offsets for M dimension
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # Create tile offsets for N dimension
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for bounds checking
    rm = tl.maximum(0, tl.minimum(rm, m - 1))
    rn = tl.maximum(0, tl.minimum(rn, n - 1))
    mask_m = rm < m
    mask_n = rn < n
    
    # Initialize accumulator for this tile
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over K dimension in blocks
    for k_start in range(0, k, BLOCK_SIZE_K):
        k_end = tl.minimum(k_start + BLOCK_SIZE_K, k)
        
        # Create K offsets
        rk = k_start + tl.arange(0, BLOCK_SIZE_K)
        rk = tl.maximum(0, tl.minimum(rk, k - 1))
        mask_k = rk < k
        
        # Load tile from A: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a = tl.load(
            A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        )
        
        # Load tile from B: shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
        b = tl.load(
            B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0
        )
        
        # Accumulate matrix multiplication
        acc += tl.dot(a, b)
    
    # Store result to C
    c = acc.to(tl.float32)
    tl.store(
        C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        c,
        mask=mask_m[:, None] & mask_n[None, :]
    )


def triton_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Batched matrix multiplication using Triton kernel.
    
    Args:
        A: Input tensor of shape (batch_size, m, k)
        B: Input tensor of shape (batch_size, k, n)
    
    Returns:
        C: Output tensor of shape (batch_size, m, n)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape[0] == B.shape[0], "Batch dimensions must match"
    
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    batch_size, m, k = A.shape
    _, _, n = B.shape
    
    # Create output tensor
    C = torch.empty((batch_size, m, n), dtype=A.dtype, device=A.device)
    
    # Set block sizes (tunable parameters for optimization)
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 16
    
    # Calculate grid dimensions
    grid_m = (m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m, grid_n, batch_size)
    
    # Launch kernel
    bmm_kernel[grid](
        A, B, C,
        batch_size, m, n, k,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for batched matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication using optimized Triton kernel.
        """
        return triton_bmm(A, B)