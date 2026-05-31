import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create offsets for M and N dimensions
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        
        # Load A block with masking
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        A_block = tl.load(A_ptr + offs_m[:, None] * K + offs_k[None, :], mask=a_mask, other=0.0)
        
        # Load B block with masking and transpose for efficient dot product
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        B_block = tl.load(B_ptr + offs_k[:, None] * N + offs_n[None, :], mask=b_mask, other=0.0)
        B_block = tl.trans(B_block)
        
        # Perform dot product and accumulate
        acc += tl.dot(A_block, B_block)
    
    # Store result with masking
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "FP32 precision required."
    
    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, "Incompatible dimensions for matrix multiplication."
    
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
        1
    )
    
    matmul_kernel[grid](
        A, B, C,
        M, K, N,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)


M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]

def get_init_inputs():
    return []