import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def lower_triangular_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # Get program IDs
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N, BLOCK_SIZE)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE)
    num_pid_in_group = GROUP_SIZE * num_pid_m
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Create block offsets
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Loop over k dimension
    for k in range(0, N, BLOCK_SIZE):
        # Calculate k offset
        offs_k = k + tl.arange(0, BLOCK_SIZE)
        
        # Load A block: A[i,k] is non-zero only if i >= k
        # We need to mask based on whether row index >= k column index
        a_mask = (offs_m[:, None] >= offs_k[None, :])
        a_block = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_an,
            mask=a_mask,
            other=0.0
        )
        
        # Load B block: B[k,j] is non-zero only if k >= j
        # We need to mask based on whether k row index >= column index j
        b_mask = (offs_k[:, None] >= offs_n[None, :])
        b_block = tl.load(
            B_ptr + offs_k[:, None] * stride_bm + offs_n[None, :] * stride_bn,
            mask=b_mask,
            other=0.0
        )
        
        # Accumulate: only compute where i >= j in the output
        accumulator = tl.dot(a_block, b_block, accumulator)
    
    # Store result with lower triangular mask
    c_mask = (offs_m[:, None] >= offs_n[None, :])
    
    # Convert to output dtype
    c_block = accumulator.to(C_ptr.dtype.element_ty)
    
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c_block,
        mask=c_mask
    )


def triton_lower_triangular_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Matrix multiplication for lower triangular matrices.
    
    Args:
        A: Lower triangular matrix of shape (N, N)
        B: Lower triangular matrix of shape (N, N)
    
    Returns:
        Lower triangular matrix C of shape (N, N) where C = A @ B
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Input matrices must have the same shape."
    assert len(A.shape) == 2, "Input matrices must be 2D."
    assert A.shape[0] == A.shape[1], "Input matrices must be square."
    
    N = A.shape[0]
    
    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Prepare output tensor
    C = torch.zeros((N, N), dtype=A.dtype, device=A.device)
    
    # Block size for Triton kernel
    BLOCK_SIZE = 128
    GROUP_SIZE = 4
    
    # Grid dimensions
    grid = lambda meta: (
        triton.cdiv(N, meta["BLOCK_SIZE"]) * triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    
    # Launch kernel
    lower_triangular_matmul_kernel[grid](
        A, B, C,
        N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=GROUP_SIZE,
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
        Performs matrix multiplication of lower triangular matrices A and B.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        return triton_lower_triangular_matmul(A, B)