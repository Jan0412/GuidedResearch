import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def lower_triangular_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row and column indices for this block
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Only compute elements in the lower triangular part
    # We only process blocks where row >= col (pid_m >= pid_n)
    if pid_m < pid_n:
        return
    
    # Create offset arrays for rows and columns
    rm = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    rn = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for valid indices (within N)
    rm_mask = rm < N
    rn_mask = rn < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Loop over the K dimension
    for k in range(0, N, BLOCK_SIZE):
        # Create k offsets
        rk = k + tl.arange(0, BLOCK_SIZE)
        rk_mask = rk < N
        
        # Create masks for lower triangular access
        # We only load elements where the row index >= column index in each matrix
        a_mask = rm_mask[:, None] & rk_mask[None, :] & (rm[:, None] >= rk[None, :])
        b_mask = rk_mask[:, None] & rn_mask[None, :] & (rk[:, None] >= rn[None, :])
        
        # Load A and B with masks for lower triangular structure
        a = tl.load(A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak, 
                   mask=a_mask, other=0.0)
        b = tl.load(B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
                   mask=b_mask, other=0.0)
        
        # Accumulate the product
        acc += tl.dot(a, b)
    
    # Store only the lower triangular part
    c_mask = rm_mask[:, None] & rn_mask[None, :] & (rm[:, None] >= rn[None, :])
    tl.store(C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
            acc, mask=c_mask)


def triton_lower_triangular_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Performs matrix multiplication of lower triangular matrices A and B,
    returning only the lower triangular part of the result.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    N = A.size(0)
    assert A.shape == (N, N) and B.shape == (N, N), "Inputs must be square matrices"
    
    # Create output tensor
    C = torch.empty_like(A)
    
    # Set block size
    BLOCK_SIZE = 128
    
    # Calculate grid dimensions
    grid = (triton.cdiv(N, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))
    
    # Launch kernel
    lower_triangular_matmul_kernel[grid](
        A, B, C,
        N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication of lower triangular matrices
    using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B,
        returning only the lower triangular part of the result.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        return triton_lower_triangular_matmul(A, B)