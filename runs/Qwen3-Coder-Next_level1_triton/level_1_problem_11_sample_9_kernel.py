import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr,  # Pointer to input tensor A (b, i, j, l)
    B_ptr,  # Pointer to input matrix B (l, k)
    C_ptr,  # Pointer to output tensor C (b, i, j, k)
    B_dim, I_dim, J_dim, L_dim, K_dim,  # Dimensions
    stride_b_a, stride_i_a, stride_j_a, stride_l_a,  # Strides for A
    stride_l_b, stride_k_b,  # Strides for B
    stride_b_c, stride_i_c, stride_j_c, stride_k_c,  # Strides for C
    BLOCK_SIZE_M: tl.constexpr,  # Block size for M dimension (b*i*j)
    BLOCK_SIZE_N: tl.constexpr,  # Block size for N dimension (k)
    BLOCK_SIZE_K: tl.constexpr,  # Block size for K dimension (l)
):
    # Get program IDs
    pid_m = tl.program_id(0)  # For M dimension (b*i*j)
    pid_n = tl.program_id(1)  # For N dimension (k)
    
    # Compute the starting offsets for M and N
    n_bij = B_dim * I_dim * J_dim
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for bounds checking
    mask_m = offsets_m < n_bij
    mask_n = offsets_n < K_dim
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension (l)
    for k_start in range(0, L_dim, BLOCK_SIZE_K):
        offsets_k = k_start + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offsets_k < L_dim
        
        # Load block of A: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # Need to convert 1D index to 3D indices (b, i, j)
        # For index idx in [0, n_bij), we have:
        # b = idx // (I_dim * J_dim)
        # i = (idx % (I_dim * J_dim)) // J_dim
        # j = (idx % (I_dim * J_dim)) % J_dim
        # Then the offset in A is b*stride_b_a + i*stride_i_a + j*stride_j_a + k*stride_l_a
        
        # Compute b, i, j from idx
        b_idx = offsets_m // (I_dim * J_dim)
        rest = offsets_m % (I_dim * J_dim)
        i_idx = rest // J_dim
        j_idx = rest % J_dim
        
        # Compute A pointer offsets
        a_offsets = (
            b_idx[:, None] * stride_b_a +
            i_idx[:, None] * stride_i_a +
            j_idx[:, None] * stride_j_a +
            offsets_k[None, :] * stride_l_a
        )
        
        # Load A block
        A_block = tl.load(A_ptr + a_offsets, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Load B block: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        b_offsets = (
            offsets_k[:, None] * stride_l_b +
            offsets_n[None, :] * stride_k_b
        )
        B_block = tl.load(B_ptr + b_offsets, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        # Accumulate: A_block @ B_block
        accumulator += tl.dot(A_block, B_block)
    
    # Store result
    # Convert 1D index to 3D indices (b, i, j) for C
    b_idx_c = offsets_m // (I_dim * J_dim)
    rest_c = offsets_m % (I_dim * J_dim)
    i_idx_c = rest_c // J_dim
    j_idx_c = rest_c % J_dim
    
    # Compute C pointer offsets
    c_offsets = (
        b_idx_c[:, None] * stride_b_c +
        i_idx_c[:, None] * stride_i_c +
        j_idx_c[:, None] * stride_j_c +
        offsets_n[None, :] * stride_k_c
    )
    
    # Store result
    tl.store(C_ptr + c_offsets, accumulator, mask=mask_m[:, None] & mask_n[None, :])


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
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
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    B_dim, I_dim, J_dim, L_dim = A.shape
    _, K_dim = B.shape
    
    # Prepare output tensor
    C = torch.empty((B_dim, I_dim, J_dim, K_dim), dtype=A.dtype, device=A.device)
    
    # Compute strides
    stride_b_a, stride_i_a, stride_j_a, stride_l_a = A.stride()
    stride_l_b, stride_k_b = B.stride()
    stride_b_c, stride_i_c, stride_j_c, stride_k_c = C.stride()
    
    # Set block sizes (tunable parameters)
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    # Compute grid dimensions
    n_bij = B_dim * I_dim * J_dim
    grid_m = (n_bij + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (K_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    matmul_kernel[grid_m, grid_n](
        A, B, C,
        B_dim, I_dim, J_dim, L_dim, K_dim,
        stride_b_a, stride_i_a, stride_j_a, stride_l_a,
        stride_l_b, stride_k_b,
        stride_b_c, stride_i_c, stride_j_c, stride_k_c,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized version of Model that uses a custom Triton kernel for the einsum operation.
    Performs 4D tensor-matrix multiplication: 
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
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
        return triton_matmul(A, B)