import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def diag_matmul_kernel(
    a_ptr,  # Pointer to diagonal vector A
    b_ptr,  # Pointer to matrix B
    out_ptr, # Pointer to output matrix C
    N, M,
    stride_bn, stride_bm,
    stride_cn, stride_cm,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    # Program IDs
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Ranges for the current block
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # Masks to handle boundaries
    mask_n = rn < N
    mask_m = rm < M

    # Load diagonal elements A[rn]
    # a shape: (BLOCK_SIZE_N,)
    a = tl.load(a_ptr + rn, mask=mask_n)

    # Load block of matrix B
    # B index: rn * stride_bn + rm * stride_bm
    # We use broadcasting to create a 2D grid of offsets
    b_offsets = rn[:, None] * stride_bn + rm[None, :] * stride_bm
    b = tl.load(b_ptr + b_offsets, mask=mask_n[:, None] & mask_m[None, :])

    # Perform element-wise multiplication: C[i, j] = A[i] * B[i, j]
    # a[:, None] broadcasts (BLOCK_SIZE_N,) to (BLOCK_SIZE_N, 1)
    out = a[:, None] * b

    # Store the result in matrix C
    out_offsets = rn[:, None] * stride_cn + rm[None, :] * stride_cm
    tl.store(out_ptr + out_offsets, out, mask=mask_n[:, None] & mask_m[None, :])


def triton_diag_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Efficiently computes C = diag(A) @ B using a Triton kernel.
    This is equivalent to A.unsqueeze(1) * B.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous for predictable striding
    A = A.contiguous()
    B = B.contiguous()
    
    N = A.shape[0]
    M = B.shape[1]
    
    # Prepare output tensor
    out = torch.empty((N, M), device=B.device, dtype=B.dtype)
    
    # Strides
    stride_bn, stride_bm = B.stride()
    stride_cn, stride_cm = out.stride()

    # Tuning parameters
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_M = 32

    # Grid: one block for every (BLOCK_SIZE_N, BLOCK_SIZE_M) chunk of the output
    grid = lambda meta: (
        (N + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"],
        (M + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"]
    )

    diag_matmul_kernel[grid](
        A, B, out,
        N, M,
        stride_bn, stride_bm,
        stride_cn, stride_cm,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication using a custom Triton kernel.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        return triton_diag_matmul(A, B)