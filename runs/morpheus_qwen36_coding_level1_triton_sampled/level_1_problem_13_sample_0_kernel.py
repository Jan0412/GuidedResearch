import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sym_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N, K,
    stride_a_row, stride_a_col,
    stride_b_row, stride_b_col,
    stride_c_row, stride_c_col,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets for rows of A and B
    row_idx_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_idx_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_idx_k = tl.arange(0, BLOCK_K)
    
    # Create masks for bounds checking
    mask_m = row_idx_m < N
    mask_n = row_idx_n < N
    mask_k = col_idx_k < K
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load tile from A: shape (BLOCK_M, BLOCK_K)
        # A is accessed row-wise for contiguous memory access
        offs_a = (row_idx_m[:, None] * stride_a_row + 
                  (k + col_idx_k)[None, :] * stride_a_col)
        A_tile = tl.load(A_ptr + offs_a, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Load tile from B: shape (BLOCK_N, BLOCK_K)
        # B is symmetric, so B_kj = B_jk. We load row j of B to get B_jk values.
        # This allows contiguous row-wise access for B as well.
        offs_b = (row_idx_n[:, None] * stride_b_row + 
                  (k + col_idx_k)[None, :] * stride_b_col)
        B_tile = tl.load(B_ptr + offs_b, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        # Dot product accumulation
        acc += tl.dot(A_tile, B_tile)
        
    # Store result
    offs_c = (row_idx_m[:, None] * stride_c_row + 
              (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] * stride_c_col)
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptr + offs_c, acc, mask=mask_c)


def triton_sym_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    assert A.shape == B.shape, "Symmetric matrices must have same shape."
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "FP32 precision required."
    
    N = A.shape[0]
    K = A.shape[1]
    
    C = torch.empty((N, N), dtype=torch.float32, device=A.device)
    
    # Tunable block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    
    # Grid configuration
    grid = lambda META: (
        triton.cdiv(N, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )
    
    # Launch kernel
    sym_matmul_kernel[grid](
        A_ptr=A.data_ptr(),
        B_ptr=B.data_ptr(),
        C_ptr=C.data_ptr(),
        N=N, K=K,
        stride_a_row=A.stride(0), stride_a_col=A.stride(1),
        stride_b_row=B.stride(0), stride_b_col=B.stride(1),
        stride_c_row=C.stride(0), stride_c_col=C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of two symmetric matrices using a 
        custom Triton kernel that exploits symmetry for contiguous memory access.
        """
        return triton_sym_matmul(A, B)