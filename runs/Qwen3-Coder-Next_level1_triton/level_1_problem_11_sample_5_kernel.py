import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_4d_kernel(
    A_ptr,  # Pointer to 4D tensor A of shape (b, i, j, l)
    B_ptr,  # Pointer to matrix B of shape (l, k)
    C_ptr,  # Pointer to output tensor C of shape (b, i, j, k)
    B, I, J, L, K,  # Dimensions
    stride_a_b, stride_a_i, stride_a_j, stride_a_l,
    stride_b_l, stride_b_k,
    stride_c_b, stride_c_i, stride_c_j, stride_c_k,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output rows (i*j dimension)
    BLOCK_SIZE_N: tl.constexpr,  # Block size for output columns (k dimension)
    BLOCK_SIZE_K: tl.constexpr,  # Block size for reduction dimension (l)
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)  # Covers combined i*j dimension
    pid_n = tl.program_id(2)
    
    # Calculate which (i, j) position this program handles
    i_idx = pid_m // J
    j_idx = pid_m % J
    
    # Check bounds
    if pid_b >= B or i_idx >= I or j_idx >= J:
        return
    
    # Initialize offsets for output matrix C[b, i, j, k]
    off_k = tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + pid_b * stride_c_b + i_idx * stride_c_i + j_idx * stride_c_j + off_k * stride_c_k
    c_mask = off_k < K
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    # Loop over K dimension (l) in blocks
    for k_start in range(0, L, BLOCK_SIZE_K):
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < L
        
        # Load block of A: [b, i, j, k_start:k_start+BLOCK_SIZE_K]
        a_offsets = pid_b * stride_a_b + i_idx * stride_a_i + j_idx * stride_a_j + k_offsets * stride_a_l
        a_ptrs = A_ptr + a_offsets
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)
        
        # Load block of B: [k_start:k_start+BLOCK_SIZE_K, k_start_n:k_start_n+BLOCK_SIZE_N]
        off_k_n = tl.arange(0, BLOCK_SIZE_N)
        b_offsets = k_offsets[:, None] * stride_b_l + off_k_n[None, :] * stride_b_k
        b_ptrs = B_ptr + b_offsets
        b = tl.load(b_ptrs, mask=(k_mask[:, None] & (off_k_n < K)[None, :]), other=0.0)
        
        # Accumulate product
        acc += tl.sum(a[:, None] * b, 0)
    
    # Store result
    tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=c_mask)


def triton_matmul_4d(A: torch.Tensor, B: torch.Tensor):
    """
    Performs 4D tensor-matrix multiplication: C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
    
    Args:
        A: Input 4D tensor of shape (b, i, j, l)
        B: Input matrix of shape (l, k)
    
    Returns:
        Output 4D tensor of shape (b, i, j, k)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    B_dim, I_dim, J_dim, L_dim = A.shape
    _, K_dim = B.shape
    
    # Ensure tensor dimensions are correct
    assert A.shape[3] == B.shape[0], f"Dimension mismatch: A.shape[3]={A.shape[3]}, B.shape[0]={B.shape[0]}"
    
    # Prepare output tensor
    C = torch.empty((B_dim, I_dim, J_dim, K_dim), dtype=A.dtype, device=A.device)
    
    # Calculate strides
    stride_a_b, stride_a_i, stride_a_j, stride_a_l = A.stride()
    stride_b_l, stride_b_k = B.stride()
    stride_c_b, stride_c_i, stride_c_j, stride_c_k = C.stride()
    
    # Define block sizes (tuned for FP32)
    BLOCK_SIZE_M = 32  # Process 32 positions in (i,j) per block
    BLOCK_SIZE_N = 64  # Process 64 columns of K per block
    BLOCK_SIZE_K = 32  # Process 32 elements of L per block
    
    # Grid dimensions: (batch, combined i*j, k_blocks)
    grid = (
        B_dim, 
        (I_dim * J_dim + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M, 
        (K_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    )
    
    # Launch kernel
    matmul_4d_kernel[grid](
        A, B, C,
        B_dim, I_dim, J_dim, L_dim, K_dim,
        stride_a_b, stride_a_i, stride_a_j, stride_a_l,
        stride_b_l, stride_b_k,
        stride_c_b, stride_c_i, stride_c_j, stride_c_k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized version of the 4D tensor-matrix multiplication using Triton kernel.
    Performs: C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
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
        return triton_matmul_4d(A, B)