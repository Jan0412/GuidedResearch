import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    # Compute pointers for the current batch
    A_batch_ptr = A_ptr + pid_b * stride_ab
    B_batch_ptr = B_ptr + pid_b * stride_bb
    C_batch_ptr = C_ptr + pid_b * stride_cb

    # Range of offsets for the block
    offs_am = (pid_m * BLOCK_SIZE_M) + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = (pid_n * BLOCK_SIZE_N) + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the blocks of A and B
    # A: (M, K), B: (K, N)
    a_ptr = A_batch_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptr = B_batch_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks of A and B
        # Masking for boundary conditions
        mask_a = (offs_am[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_bn[None, :] < N)
        
        a = tl.load(a_ptr, mask=mask_a, other=0.0)
        b = tl.load(b_ptr, mask=mask_b, other=0.0)
        
        # Matrix multiplication of blocks
        accumulator += tl.dot(a, b)
        
        # Advance pointers to the next block in K
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * stride_bk

    # Pointer to the block of C
    c_ptr = C_batch_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    mask_c = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    
    # Store the result
    tl.store(c_ptr, accumulator, mask=mask_c)


def triton_bmm(A: torch.Tensor, B: torch.Tensor):
    # Ensure inputs are contiguous on GPU
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    batch_size, M, K = A.shape
    _, K_B, N = B.shape
    assert K == K_B, "Inner dimensions must match."

    C = torch.empty((batch_size, M, N), device=A.device, dtype=A.dtype)

    # Strides
    stride_ab, stride_am, stride_ak = A.stride()
    stride_bb, stride_bk, stride_bn = B.stride()
    stride_cb, stride_cm, stride_cn = C.stride()

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid: (M_blocks, N_blocks, Batch_blocks)
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
        batch_size,
    )

    bmm_kernel[grid](
        A, B, C,
        M, N, K,
        stride_ab, stride_am, stride_ak,
        stride_bb, stride_bk, stride_bn,
        stride_cb, stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M, 
        BLOCK_SIZE_N=BLOCK_SIZE_N, 
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return C


class ModelNew(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return triton_bmm(A, B)