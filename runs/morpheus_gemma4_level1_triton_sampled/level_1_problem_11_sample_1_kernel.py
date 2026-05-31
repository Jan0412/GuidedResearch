import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Map program IDs to the block of C it computes
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets for the current block
    rm = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the first element of the blocks in A and B
    # a_ptr + rm * stride_am + rk * stride_ak
    # b_ptr + rk * stride_bk + rn * stride_bn
    a_ptr += (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptr += (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks from A and B
        a = tl.load(a_ptr, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(b_ptr, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)

        # Matrix multiplication
        accumulator += tl.dot(a, b)

        # Advance pointers to the next block in K
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * stride_bk

    # Store the result in C
    c_ptr += (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptr, accumulator, mask=c_mask)


def triton_matmul_4d(A: torch.Tensor, B: torch.Tensor):
    # A: (b, i, j, l), B: (l, k) -> C: (b, i, j, k)
    b, i, j, l = A.shape
    l_b, k = B.shape
    assert l == l_b, "Dimension mismatch between A and B"

    # Flatten A to (M, K) where M = b*i*j, K = l
    # Flatten C to (M, N) where M = b*i*j, N = k
    M = b * i * j
    N = k
    K = l

    # Ensure tensors are contiguous and on GPU
    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Strides for the flattened interpretations
    stride_am = l
    stride_ak = 1
    stride_bk = k
    stride_bn = 1
    stride_cm = k
    stride_cn = 1

    # Heuristics for block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    # Reshape C back to (b, i, j, k)
    return C.view(b, i, j, k)


class ModelNew(nn.Module):
    """
    Optimized version of 4D tensor-matrix multiplication using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        """
        Performs the 4D tensor-matrix multiplication using a custom Triton kernel.

        Args:
            A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
            B (torch.Tensor): Input matrix of shape (l, k)

        Returns:
            torch.Tensor: Output 4D tensor of shape (b, i, j, k)
        """
        return triton_matmul_4d(A, B)