import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    Triton kernel for matrix multiplication.
    C = A @ B
    A: (M, K), B: (K, N), C: (M, N)
    """
    # Map program IDs to the block of C it computes
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create offsets for the block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the first block of A and B
    # a_ptr + (rm * stride_am + rk * stride_ak)
    # b_ptr + (rk * stride_bk + rn * stride_bn)
    a_ptr += (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptr += (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load tiles from A and B
        # Use masking to handle cases where M, N, or K are not multiples of block sizes
        a = tl.load(a_ptr, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(b_ptr, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)

        # Matrix multiplication of tiles
        accumulator += tl.dot(a, b)

        # Advance pointers to the next block along the K dimension
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * stride_bk

    # Store the result back to C
    c_ptr += (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptr, accumulator, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Performs 3D tensor-matrix multiplication using a Triton kernel.
    A: (N, M, K)
    B: (K, L)
    Output: (N, M, L)
    """
    # Reshape A to (N*M, K) to treat it as a single large matrix multiplication
    N_batch, M_dim, K_dim = A.shape
    L_dim = B.shape[1]
    
    # Ensure tensors are contiguous and on GPU
    A = A.contiguous()
    B = B.contiguous()
    
    # Flatten A to 2D: (M_total, K) where M_total = N_batch * M_dim
    A_flat = A.view(-1, K_dim)
    M_total = A_flat.shape[0]
    
    # Prepare output tensor
    C_flat = torch.empty((M_total, L_dim), device=A.device, dtype=A.dtype)

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid dimensions
    grid = (
        triton.cdiv(M_total, BLOCK_SIZE_M),
        triton.cdiv(L_dim, BLOCK_SIZE_N),
    )

    # Launch kernel
    matmul_kernel[grid](
        A_flat, B, C_flat,
        M_total, L_dim, K_dim,
        A_flat.stride(0), A_flat.stride(1),
        B.stride(0), B.stride(1),
        C_flat.stride(0), C_flat.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    # Reshape output back to 3D: (N, M, L)
    return C_flat.view(N_batch, M_dim, L_dim)


class ModelNew(nn.Module):
    """
    Optimized version of Model using a custom Triton kernel for tensor-matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication.

        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L).
        """
        return triton_matmul(A, B)