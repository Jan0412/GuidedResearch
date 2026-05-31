import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triu_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr,
):
    # Program IDs for the output block (i, j)
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    # Optimization: Since A and B are upper triangular, the product C is also upper triangular.
    # If the block index pid_i > pid_j, then for all i in this block and j in that block, i > j.
    # Thus, all elements in this block of C are 0.
    if pid_i > pid_j:
        return

    # Create offsets for the block
    i_offsets = pid_i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    j_offsets = pid_j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Accumulator for the dot product
    accumulator = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # The inner loop over k. 
    # For upper triangular matrices, C[i, j] = sum_{k=i}^j A[i, k] * B[k, j].
    # For a block (pid_i, pid_j), k ranges from pid_i * BLOCK_SIZE to (pid_j + 1) * BLOCK_SIZE.
    for pid_k in range(pid_i, pid_j + 1):
        k_offsets = pid_k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        # Load A block: A[i, k]
        # Mask: A is upper triangular, so A[i, k] = 0 if k < i
        a_mask = (k_offsets[None, :] >= 0) & (k_offsets[None, :] < N) & (k_offsets[None, :] >= i_offsets[:, None])
        a = tl.load(A_ptr + i_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak, mask=a_mask, other=0.0)

        # Load B block: B[k, j]
        # Mask: B is upper triangular, so B[k, j] = 0 if k > j
        b_mask = (k_offsets[:, None] >= 0) & (k_offsets[:, None] < N) & (k_offsets[:, None] <= j_offsets[None, :])
        b = tl.load(B_ptr + k_offsets[:, None] * stride_bk + j_offsets[None, :] * stride_bn, mask=b_mask, other=0.0)

        # Compute dot product
        accumulator += tl.dot(a, b)

    # Store the result in C
    # Mask: C is upper triangular, so C[i, j] = 0 if i > j
    c_mask = (i_offsets[:, None] >= 0) & (i_offsets[:, None] < N) & (j_offsets[None, :] >= 0) & (j_offsets[None, :] < N) & (i_offsets[:, None] <= j_offsets[None, :])
    tl.store(C_ptr + i_offsets[:, None] * stride_cm + j_offsets[None, :] * stride_cn, accumulator, mask=c_mask)


def triton_triu_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Wrapper for the Triton upper triangular matrix multiplication kernel.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Matrices must be square and of the same size."
    
    N = A.shape[0]
    A = A.contiguous()
    B = B.contiguous()
    
    out = torch.empty((N, N), device=A.device, dtype=torch.float32)
    
    BLOCK_SIZE = 64
    # Grid is (N // BLOCK_SIZE, N // BLOCK_SIZE)
    grid = (triton.cdiv(N, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))
    
    triu_matmul_kernel[grid](
        A, B, out,
        N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication (C = A * B) for upper triangular matrices
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
        # Ensure inputs are FP32 as requested
        A = A.to(torch.float32)
        B = B.to(torch.float32)
        return triton_triu_matmul(A, B)