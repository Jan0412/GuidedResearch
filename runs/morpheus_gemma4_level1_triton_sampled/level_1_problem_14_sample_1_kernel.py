import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def upper_tri_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # If the block is entirely below the diagonal, we can skip it.
    if pid_m > pid_n:
        return

    # Define the range of indices for the current block
    rm = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    rn = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Accumulator for the result block
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # The product C[i, j] = sum_{k=i}^j A[i, k] * B[k, j].
    # For a block (pid_m, pid_n), i is in [pid_m*BS, (pid_m+1)*BS) 
    # and j is in [pid_n*BS, (pid_n+1)*BS).
    # The union of all k ranges [i, j] for the block is [pid_m*BS, (pid_n+1)*BS).
    k_start = pid_m * BLOCK_SIZE
    k_end = (pid_n + 1) * BLOCK_SIZE

    for k in range(k_start, k_end, BLOCK_SIZE):
        rk = k + tl.arange(0, BLOCK_SIZE)
        
        # Load blocks from A and B
        # A is (N, N), B is (N, N)
        # Masking is necessary for boundaries, though N=4096 is a multiple of 64.
        a = tl.load(A_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak), 
                    mask=(rm[:, None] < N) & (rk[None, :] < N), other=0.0)
        b = tl.load(B_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn), 
                    mask=(rk[:, None] < N) & (rn[None, :] < N), other=0.0)
        
        # Compute dot product
        acc += tl.dot(a, b)

    # Store the result. Only store the upper triangular part (i <= j).
    # Since we only run for pid_m <= pid_n, we only need to handle the diagonal block 
    # where pid_m == pid_n.
    mask = (rm[:, None] < N) & (rn[None, :] < N) & (rm[:, None] <= rn[None, :])
    tl.store(C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn), acc, mask=mask)


def triton_upper_tri_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Wrapper for the Triton upper triangular matrix multiplication kernel.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    
    N = A.shape[0]
    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty((N, N), device=A.device, dtype=torch.float32)

    BLOCK_SIZE = 64
    # Grid is (N/BS, N/BS)
    grid = (triton.cdiv(N, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))

    upper_tri_matmul_kernel[grid](
        A, B, C,
        N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication for upper triangular matrices
    using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        # Use the custom Triton implementation to avoid computing the lower triangle
        return triton_upper_tri_matmul(A, B)