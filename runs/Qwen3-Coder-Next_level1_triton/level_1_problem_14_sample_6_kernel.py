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
    # Get program indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create offsets for rows and columns
    offsets_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for valid indices
    mask_m = offsets_m < N
    mask_n = offsets_n < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Loop over k dimension
    for k in range(0, N, BLOCK_SIZE):
        # For upper triangular matrices, only load valid elements
        # A is upper triangular: only load when row <= col
        # B is upper triangular: only load when row <= col
        
        # Create column offsets for A
        offsets_k_a = k + tl.arange(0, BLOCK_SIZE)
        # Mask for A: only load if row <= col (upper triangular)
        mask_a_col = offsets_k_a[None, :] >= offsets_m[:, None]
        mask_a = mask_m[:, None] & mask_a_col & (offsets_k_a[None, :] < N)
        
        # Load A with mask
        a = tl.load(A_ptr + offsets_m[:, None] * stride_am + offsets_k_a[None, :] * stride_ak, 
                   mask=mask_a, other=0.0)
        
        # Create row offsets for B
        offsets_k_b = k + tl.arange(0, BLOCK_SIZE)
        # Mask for B: only load if row <= col (upper triangular)
        mask_b_row = offsets_k_b[:, None] <= offsets_n[None, :]
        mask_b = mask_b_row & (offsets_k_b[:, None] < N) & mask_n[None, :]
        
        # Load B with mask
        b = tl.load(B_ptr + offsets_k_b[:, None] * stride_bk + offsets_n[None, :] * stride_bn,
                   mask=mask_b, other=0.0)
        
        # Accumulate multiplication
        acc += tl.dot(a, b)
    
    # Convert to upper triangular mask for output
    # Only keep elements where row <= col
    mask_c = (offsets_m[:, None] <= offsets_n[None, :]) & mask_m[:, None] & mask_n[None, :]
    
    # Store result
    tl.store(C_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn,
             acc.to(tl.float32), mask=mask_c)


def triton_triu_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Performs matrix multiplication for upper triangular matrices using Triton kernel.
    Only computes the upper triangular part of the result.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Matrices must have the same shape."
    assert A.shape[0] == A.shape[1], "Matrices must be square."
    
    N = A.shape[0]
    
    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Prepare output tensor
    C = torch.empty_like(A)
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Calculate grid dimensions
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE, 
            (N + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch kernel
    triu_matmul_kernel[grid](
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
    Optimized model that performs matrix multiplication for upper triangular matrices
    using a custom Triton kernel that only computes the upper triangular part.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs optimized matrix multiplication for upper triangular matrices.
        
        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).
            
        Returns:
            torch.Tensor: The product of A and B, upper triangular part computed via Triton.
        """
        return triton_triu_matmul(A, B)