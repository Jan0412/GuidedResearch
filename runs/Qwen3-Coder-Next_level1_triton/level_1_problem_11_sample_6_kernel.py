import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_4d_2d_kernel(
    A_ptr,  # Pointer to 4D tensor A of shape (b, i, j, l)
    B_ptr,  # Pointer to 2D matrix B of shape (l, k)
    C_ptr,  # Pointer to output 4D tensor C of shape (b, i, j, k)
    B_k_stride,  # Stride of B along dimension k (should be 1 for contiguous)
    B_l_stride,  # Stride of B along dimension l (should be k for contiguous)
    A_b_stride,  # Stride of A along batch dimension b
    A_i_stride,  # Stride of A along dimension i
    A_j_stride,  # Stride of A along dimension j
    A_l_stride,  # Stride of A along dimension l (should be 1 for contiguous)
    C_b_stride,  # Stride of C along batch dimension b
    C_i_stride,  # Stride of C along dimension i
    C_j_stride,  # Stride of C along dimension j
    C_k_stride,  # Stride of C along dimension k (should be 1 for contiguous)
    B,  # batch size (b)
    I,  # dimension i
    J,  # dimension j
    L,  # dimension l (must match B's first dimension)
    K,  # dimension k
    BLOCK_SIZE_M: tl.constexpr,  # Block size for k dimension (output rows)
    BLOCK_SIZE_N: tl.constexpr,  # Block size for l dimension (reduction)
    BLOCK_SIZE_K: tl.constexpr,  # Block size for i*j combined (output cols)
):
    # Batch index
    batch = tl.program_id(2)
    # Block index for the i*j combined dimension
    ij_block = tl.program_id(0)
    # Block index for the k dimension
    k_block = tl.program_id(1)
    
    # Convert ij_block to i and j indices
    ij_size = I * J
    i_idx = ij_block // J
    j_idx = ij_block % J
    
    # Calculate base offsets for A and C
    a_base = batch * A_b_stride + i_idx * A_i_stride + j_idx * A_j_stride
    c_base = batch * C_b_stride + i_idx * C_i_stride + j_idx * C_j_stride
    
    # Create offsets for the k dimension (output columns)
    k_offsets = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    k_mask = k_offsets < K
    
    # Initialize accumulator for C values
    accumulator = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32)
    
    # Loop over the reduction dimension L in blocks
    for l_block in range(0, L, BLOCK_SIZE_N):
        l_offsets = l_block + tl.arange(0, BLOCK_SIZE_N)
        l_mask = l_offsets < L
        
        # Load A block: shape (BLOCK_SIZE_N,)
        a_offsets = a_base + l_offsets * A_l_stride
        a_block = tl.load(A_ptr + a_offsets, mask=l_mask, other=0.0)
        
        # Load B block: shape (BLOCK_SIZE_N, BLOCK_SIZE_K)
        b_l_offsets = l_offsets[:, None] * B_l_stride
        b_k_offsets = k_offsets[None, :] * B_k_stride
        b_block = tl.load(B_ptr + b_l_offsets + b_k_offsets, mask=l_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Compute partial product and accumulate
        accumulator += tl.sum(a_block[:, None] * b_block, axis=0)
    
    # Store result to C
    c_offsets = c_base + k_offsets * C_k_stride
    c_values = accumulator.to(tl.float32)
    tl.store(C_ptr + c_offsets, c_values, mask=k_mask)


def triton_matmul_4d_2d(A: torch.Tensor, B: torch.Tensor):
    """
    Performs 4D tensor-matrix multiplication: 
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
    
    Args:
        A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
        B (torch.Tensor): Input matrix of shape (l, k)
    
    Returns:
        torch.Tensor: Output 4D tensor of shape (b, i, j, k)
    """
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    b, i, j, l = A.shape
    l2, k = B.shape
    assert l == l2, f"Dimension mismatch: A has l={l}, B has l={l2}"
    
    # Create output tensor
    C = torch.empty((b, i, j, k), dtype=A.dtype, device=A.device)
    
    # Compute strides
    A_strides = A.stride()
    B_strides = B.stride()
    C_strides = C.stride()
    
    # Define block sizes (tunable parameters for performance)
    BLOCK_SIZE_K = 32  # Block size for k dimension (output columns)
    BLOCK_SIZE_N = 32  # Block size for reduction dimension l
    BLOCK_SIZE_M = 64  # Block size for i*j combined dimension
    
    # Compute grid dimensions
    grid = (
        (i * j + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,  # Number of blocks for i*j dimension
        (k + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K,     # Number of blocks for k dimension
        b  # Batch dimension
    )
    
    # Launch kernel
    matmul_4d_2d_kernel[grid](
        A, B, C,
        B_strides[1], B_strides[0],
        A_strides[0], A_strides[1], A_strides[2], A_strides[3],
        C_strides[0], C_strides[1], C_strides[2], C_strides[3],
        b, i, j, l, k,
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
        Performs the 4D tensor-matrix multiplication using Triton kernel.
        
        Args:
            A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
            B (torch.Tensor): Input matrix of shape (l, k)
        
        Returns:
            torch.Tensor: Output 4D tensor of shape (b, i, j, k)
        """
        return triton_matmul_4d_2d(A, B)