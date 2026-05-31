import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def upper_tri_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    stride_a, stride_b, stride_c,
    BLOCK_SIZE_J: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid
    
    # Pointers to the current row of A and the start of the upper triangle in C
    A_row_ptr = A_ptr + i * stride_a
    C_row_ptr = C_ptr + i * stride_c + i
    
    # Iterate over j in blocks for the upper triangle
    for j_start in range(i, N, BLOCK_SIZE_J):
        j_end = min(j_start + BLOCK_SIZE_J, N)
        j_offsets = j_start + tl.arange(0, BLOCK_SIZE_J)
        mask_j = j_offsets < N
        
        # Initialize accumulators for this j-block
        acc = tl.zeros((BLOCK_SIZE_J,), dtype=tl.float32)
        
        # Iterate over k in blocks
        for k_start in range(i, N, BLOCK_SIZE_K):
            k_end = min(k_start + BLOCK_SIZE_K, N)
            k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
            mask_k = k_offsets < N
            
            # Load A[i, k]
            A_vals = tl.load(A_row_ptr + k_offsets, mask=mask_k, other=0.0)
            
            # Load B[k, j] with mask to respect upper triangular structure of B
            # B[k, j] is valid only if k <= j
            mask_B = (k_offsets[:, None] <= j_offsets[None, :]) & mask_k[:, None] & mask_j[None, :]
            B_vals = tl.load(B_ptr + k_offsets[:, None] * stride_b + j_offsets[None, :], 
                             mask=mask_B, other=0.0)
            
            # Accumulate product
            acc += tl.sum(A_vals[:, None] * B_vals, axis=0)
        
        # Store result
        tl.store(C_row_ptr + j_offsets, acc, mask=mask_j)


def triton_upper_tri_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    N = A.shape[0]
    
    # Initialize output with zeros to ensure lower triangle is zero
    C = torch.zeros_like(A)
    
    # Grid: one program per row
    grid = (N,)
    
    # Tunable block sizes
    BLOCK_SIZE_J = 128
    BLOCK_SIZE_K = 128
    
    upper_tri_matmul_kernel[grid](
        A, B, C,
        N,
        A.stride(0), B.stride(0), C.stride(0),
        BLOCK_SIZE_J, BLOCK_SIZE_K
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_upper_tri_matmul(A, B)