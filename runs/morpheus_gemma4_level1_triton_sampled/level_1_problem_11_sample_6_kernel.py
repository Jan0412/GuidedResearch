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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Range of indices for this block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the start of the blocks
    # a_ptr is (M, K), b_ptr is (K, N)
    # we use [:, None] and [None, :] to broadcast into 2D blocks
    a_ptr_block = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptr_block = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Accumulator for the dot product
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks from A and B
        # Shift pointers by k * BLOCK_SIZE_K
        a = tl.load(
            a_ptr + (rm[:, None] * stride_am + (rk[None, :] + k * BLOCK_SIZE_K) * stride_ak),
            mask=(rm[:, None] < M) & ((rk[None, :] + k * BLOCK_SIZE_K) < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + ((rk[:, None] + k * BLOCK_SIZE_K) * stride_bk + rn[None, :] * stride_bn),
            mask=((rk[:, None] + k * BLOCK_SIZE_K) < K) & (rn[None, :] < N),
            other=0.0,
        )
        # Perform matrix multiplication block
        accumulator += tl.dot(a, b)

    # Store the result in C
    c_ptr_block = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(
        c_ptr_block, 
        accumulator, 
        mask=(rm[:, None] < M) & (rn[None, :] < N)
    )

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    # a shape: (M, K), b shape: (K, N)
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Inner dimensions must match"

    # Ensure tensors are contiguous
    a = a.contiguous()
    b = b.contiguous()
    
    # Output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Strides
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return c

class ModelNew(nn.Module):
    """
    Optimized 4D tensor-matrix multiplication using a custom Triton kernel.
    C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        """
        Args:
            A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
            B (torch.Tensor): Input matrix of shape (l, k)

        Returns:
            torch.Tensor: Output 4D tensor of shape (b, i, j, k)
        """
        b, i, j, l = A.shape
        k = B.shape[1]
        
        # Flatten A to (b*i*j, l) to treat it as a standard matrix multiplication
        A_flat = A.view(-1, l)
        
        # Perform the matmul using the Triton kernel
        C_flat = triton_matmul(A_flat, B)
        
        # Reshape the result back to the 4D output shape (b, i, j, k)
        return C_flat.view(b, i, j, k)