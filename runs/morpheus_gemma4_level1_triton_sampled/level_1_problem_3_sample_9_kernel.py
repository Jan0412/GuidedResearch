import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_batch_a, stride_batch_b, stride_batch_c,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Pointers to the start of the matrices for the current batch
    A_batch_ptr = A_ptr + pid_batch * stride_batch_a
    B_batch_ptr = B_ptr + pid_batch * stride_batch_b
    C_batch_ptr = C_ptr + pid_batch * stride_batch_c

    # Range of indices for the current block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers for the current block of A and B
    # A is (M, K), B is (K, N)
    # A_ptr[rm, rk]
    a_ptr = A_batch_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    # B_ptr[rk, rn]
    b_ptr = B_batch_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Accumulator for the dot product
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load tiles
        a = tl.load(a_ptr, mask=(rm[:, None] < M) & (rk[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        # Shift B pointer for the next block of K
        b_ptr_current = b_ptr + k * BLOCK_SIZE_K * stride_bk
        b = tl.load(b_ptr_current, mask=(rk[:, None] < K - k * BLOCK_SIZE_K) & (rn[None, :] < N), other=0.0)
        
        # Matrix multiplication
        accumulator += tl.dot(a, b)
        
        # Update A pointer for the next block of K
        a_ptr += BLOCK_SIZE_K * stride_ak

    # Store the result in C
    c_ptr = C_batch_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptr, accumulator, mask=(rm[:, None] < M) & (rn[None, :] < N))

def triton_bmm(A: torch.Tensor, B: torch.Tensor):
    # A: (batch, m, k), B: (batch, k, n)
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    
    # Ensure tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    batch, m, k = A.shape
    _, k_b, n = B.shape
    assert k == k_b, "Inner dimensions must match"
    
    C = torch.empty((batch, m, n), device=A.device, dtype=A.dtype)
    
    # Strides
    stride_am, stride_ak = A.stride(1), A.stride(2)
    stride_bk, stride_bn = B.stride(1), B.stride(2)
    stride_cm, stride_cn = C.stride(1), C.stride(2)
    stride_batch_a, stride_batch_b, stride_batch_c = A.stride(0), B.stride(0), C.stride(0)
    
    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    grid = (batch, triton.cdiv(m, BLOCK_SIZE_M), triton.cdiv(n, BLOCK_SIZE_N))
    
    bmm_kernel[grid](
        A, B, C,
        m, n, k,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_batch_a, stride_batch_b, stride_batch_c,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
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