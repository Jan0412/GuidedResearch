import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_4d_2d_kernel(
    A_ptr,  # 4D tensor pointer (b, i, j, l)
    B_ptr,  # 2D matrix pointer (l, k)
    C_ptr,  # 4D output tensor pointer (b, i, j, k)
    B_stride_l, B_stride_k,  # Strides for B matrix
    C_stride_b, C_stride_i, C_stride_j, C_stride_k,  # Strides for output C
    B_stride_0, B_stride_1,  # B strides (l, k)
    B_size_l, B_size_k,  # Dimensions of B (l, k)
    B_size_b, B_size_i, B_size_j,  # Dimensions of A (b, i, j)
    BLOCK_SIZE_M: tl.constexpr,  # Block size for k dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for l dimension
    BLOCK_SIZE_K: tl.constexpr,  # Block size for l dimension (same as B_size_l)
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_j = tl.program_id(2)
    pid_k = tl.program_id(3)  # For k dimension partitioning
    
    # Calculate offsets
    # We'll process one (b, i, j) block at a time, and partition k dimension
    
    # Offsets for the output matrix k dimension
    k_offset = pid_k * BLOCK_SIZE_M
    k_offsets = k_offset + tl.arange(0, BLOCK_SIZE_M)
    k_mask = k_offsets < B_size_k
    
    # For each l dimension
    l_offsets = tl.arange(0, BLOCK_SIZE_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Pointer to the start of A[b, i, j, :]
    A_ptr_block = A_ptr + pid_b * (B_size_i * B_size_j * B_size_l) + \
                         pid_i * (B_size_j * B_size_l) + \
                         pid_j * B_size_l
    
    # Pointer to the start of B[:, k_offset:k_offset+BLOCK_SIZE_M]
    B_ptr_block = B_ptr + k_offset * B_stride_k
    
    # Matrix multiplication over l dimension
    for l_start in range(0, B_size_l, BLOCK_SIZE_N):
        l_offsets = l_start + tl.arange(0, BLOCK_SIZE_N)
        l_mask = l_offsets < B_size_l
        
        # Load A[b,i,j,l_start:l_start+BLOCK_SIZE_N]
        a = tl.load(A_ptr_block + l_offsets, mask=l_mask, other=0.0)
        
        # Load B[l_start:l_start+BLOCK_SIZE_N, k_offset:k_offset+BLOCK_SIZE_M]
        # Reshape to (BLOCK_SIZE_N, BLOCK_SIZE_M)
        b = tl.load(B_ptr_block + l_offsets[:, None] * B_stride_l + k_offsets[None, :], 
                    mask=l_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Accumulate: a is (BLOCK_SIZE_N,), b is (BLOCK_SIZE_N, BLOCK_SIZE_M)
        # Need to broadcast a to match b's shape
        acc += tl.sum(a[:, None] * b, axis=0)
    
    # Store result
    c = acc.to(tl.float32)
    C_ptr_block = C_ptr + pid_b * C_stride_b + pid_i * C_stride_i + pid_j * C_stride_j + k_offsets * C_stride_k
    tl.store(C_ptr_block, c, mask=k_mask)


def triton_matmul_4d_2d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs 4D tensor-matrix multiplication:
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
    
    Args:
        A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
        B (torch.Tensor): Input matrix of shape (l, k)
    
    Returns:
        torch.Tensor: Output 4D tensor of shape (b, i, j, k)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only FP32 supported"
    
    A = A.contiguous()
    B = B.contiguous()
    
    b, i, j, l = A.shape
    l2, k = B.shape
    assert l == l2, "Dimension mismatch"
    
    # Create output tensor
    C = torch.empty((b, i, j, k), dtype=torch.float32, device=A.device)
    
    # Set block sizes for optimization
    BLOCK_SIZE_M = 64  # Block size for k dimension
    BLOCK_SIZE_N = 64  # Block size for l dimension
    BLOCK_SIZE_K = 32  # Not used for l dimension in this kernel
    
    # Grid dimensions: (b, i, j, k_blocks)
    grid_b = b
    grid_i = i
    grid_j = j
    grid_k = (k + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    # Launch kernel
    matmul_4d_2d_kernel[(
        grid_b, grid_i, grid_j, grid_k
    )](
        A, B, C,
        B.stride(0), B.stride(1),  # B strides
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),  # C strides
        B.stride(0), B.stride(1),  # B strides
        l, k,  # B dimensions
        b, i, j,  # A dimensions
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_warps=4,
        num_stages=3,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized version of Model that uses custom Triton kernel for 4D tensor-matrix multiplication.
    
    Performs 4D tensor-matrix multiplication: 
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        """
        Performs the 4D tensor-matrix multiplication using optimized Triton kernel.

        Args:
            A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
            B (torch.Tensor): Input matrix of shape (l, k)

        Returns:
            torch.Tensor: Output 4D tensor of shape (b, i, j, k)
        """
        return triton_matmul_4d_2d(A, B)