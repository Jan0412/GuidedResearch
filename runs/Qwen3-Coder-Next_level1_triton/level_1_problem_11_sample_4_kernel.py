import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def einsum_kernel(
    A_ptr,  # Pointer to 4D tensor A: (b, i, j, l)
    B_ptr,  # Pointer to matrix B: (l, k)
    C_ptr,  # Pointer to output tensor C: (b, i, j, k)
    B_dim0,  # l (dimension of B)
    B_dim1,  # k (dimension of B)
    A_strides_b,  # stride for batch dimension in A
    A_strides_i,  # stride for i dimension in A
    A_strides_j,  # stride for j dimension in A
    A_strides_l,  # stride for l dimension in A
    C_strides_b,  # stride for batch dimension in C
    C_strides_i,  # stride for i dimension in C
    C_strides_j,  # stride for j dimension in C
    C_strides_k,  # stride for k dimension in C
    B_strides_l,  # stride for l dimension in B
    B_strides_k,  # stride for k dimension in B
    n_batches,  # b
    n_i,  # i
    n_j,  # j
    n_l,  # l
    n_k,  # k
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Determine batch, i, j indices
    batch_idx = tl.program_id(0)
    i_idx = tl.program_id(1)
    j_idx = tl.program_id(2)
    
    # Compute offset for A and C for this batch, i, j
    a_offset = batch_idx * A_strides_b + i_idx * A_strides_i + j_idx * A_strides_j
    c_offset = batch_idx * C_strides_b + i_idx * C_strides_i + j_idx * C_strides_j
    
    # Initialize accumulator for k dimension
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    c_sum = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32)
    
    # Iterate over l dimension in blocks
    for l_start in range(0, n_l, BLOCK_SIZE_L):
        l_offsets = l_start + tl.arange(0, BLOCK_SIZE_L)
        l_mask = l_offsets < n_l
        
        # Load A[b,i,j,l] values
        a_ptrs = A_ptr + a_offset + l_offsets * A_strides_l
        a_vals = tl.load(a_ptrs, mask=l_mask, other=0.0)
        
        # Load B[l,k] values
        b_ptrs = B_ptr + l_offsets[:, None] * B_strides_l + k_offsets[None, :] * B_strides_k
        b_vals = tl.load(b_ptrs, mask=(l_mask[:, None] & (k_offsets < n_k)[None, :]), other=0.0)
        
        # Compute partial sum for this block: (BLOCK_SIZE_L,) @ (BLOCK_SIZE_L, BLOCK_SIZE_K) -> (BLOCK_SIZE_K,)
        c_sum += tl.sum(a_vals[:, None] * b_vals, axis=0)
    
    # Store result
    c_ptrs = C_ptr + c_offset + k_offsets * C_strides_k
    tl.store(c_ptrs, c_sum.to(tl.float32), mask=k_offsets < n_k)


def triton_einsum(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs 4D tensor-matrix multiplication using Triton kernel:
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
    
    b, i, j, l = A.shape
    l2, k = B.shape
    assert l == l2, f"Dimension mismatch: A has l={l}, B has l={l2}"
    
    # Prepare output tensor
    C = torch.empty((b, i, j, k), dtype=A.dtype, device=A.device)
    
    # Get strides for efficient memory access
    A_strides = A.stride()
    B_strides = B.stride()
    C_strides = C.stride()
    
    # Define block sizes for tuning
    BLOCK_SIZE_L = 64
    BLOCK_SIZE_K = 128
    
    # Grid dimensions: (batch, i, j)
    grid = (b, i, j)
    
    # Launch kernel
    einsum_kernel[grid](
        A, B, C,
        B_dim0=l, B_dim1=k,
        A_strides_b=A_strides[0], A_strides_i=A_strides[1], A_strides_j=A_strides[2], A_strides_l=A_strides[3],
        C_strides_b=C_strides[0], C_strides_i=C_strides[1], C_strides_j=C_strides[2], C_strides_k=C_strides[3],
        B_strides_l=B_strides[0], B_strides_k=B_strides[1],
        n_batches=b, n_i=i, n_j=j, n_l=l, n_k=k,
        BLOCK_SIZE_L=BLOCK_SIZE_L, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for 4D tensor-matrix multiplication.
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
        return triton_einsum(A, B)