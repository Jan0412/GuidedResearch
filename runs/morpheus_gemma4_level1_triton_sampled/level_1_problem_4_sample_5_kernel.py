import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matvec_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K,
    stride_am, stride_ak,
    stride_bk,
    stride_cm,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Triton kernel for Matrix-Vector multiplication (C = A * B).
    A: (M, K), B: (K, 1), C: (M, 1)
    """
    # Each program handles a block of rows in M
    pid = tl.program_id(0)
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # Accumulator for the dot product results for the block of rows
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Iterate over the K dimension in chunks of BLOCK_K
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        mask_k = rk < K
        
        # Load a chunk of the vector B: shape (BLOCK_K,)
        # B is (K, 1), so we index it as a 1D array
        b = tl.load(B_ptr + rk * stride_bk, mask=mask_k, other=0.0)
        
        # Load a chunk of the matrix A: shape (BLOCK_M, BLOCK_K)
        # Offset logic: row_idx * stride_am + col_idx * stride_ak
        a = tl.load(
            A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak, 
            mask=(rm[:, None] < M) & mask_k[None, :], 
            other=0.0
        )
        
        # Perform element-wise multiplication and sum across the K dimension
        # a: (BLOCK_M, BLOCK_K), b[None, :]: (1, BLOCK_K)
        # Result of sum: (BLOCK_M,)
        acc += tl.sum(a * b[None, :], axis=1)
        
    # Store the final accumulated results into the output vector C
    # C is (M, 1), so we index it as C_ptr + row_idx * stride_cm
    tl.store(C_ptr + rm * stride_cm, acc, mask=rm < M)


def triton_matvec(A: torch.Tensor, B: torch.Tensor):
    """
    Wrapper function to launch the Triton matvec kernel.
    """
    # Ensure inputs are on GPU and contiguous for efficient memory access
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K_B, One = B.shape
    assert K == K_B, "Matrix A and Vector B dimension mismatch."
    
    # Prepare the output tensor C of shape (M, 1)
    C = torch.empty((M, 1), device=A.device, dtype=A.dtype)
    
    # Extract strides for pointer arithmetic in the kernel
    stride_am, stride_ak = A.stride()
    stride_bk, _ = B.stride()
    stride_cm, _ = C.stride()
    
    # Tunable parameters
    BLOCK_M = 32
    BLOCK_K = 1024
    
    # Grid: parallelize over the rows of A
    grid = (triton.cdiv(M, BLOCK_M),)
    
    # Launch kernel
    matvec_kernel[grid](
        A, B, C,
        M, K,
        stride_am, stride_ak,
        stride_bk,
        stride_cm,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-vector multiplication using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication A * B.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return triton_matvec(A, B)