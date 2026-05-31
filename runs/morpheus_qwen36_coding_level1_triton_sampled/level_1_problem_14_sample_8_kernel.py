import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triu_matmul_kernel(
    A_ptr, B_T_ptr, C_ptr,
    N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    m_block_id = tl.program_id(0)
    n_block_id = tl.program_id(1)

    m_offsets = m_block_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)

    m_mask = m_offsets < N
    n_mask = n_offsets < N
    k_mask = k_offsets < N

    # Load A block: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
    A_block = tl.load(
        A_ptr + m_offsets[:, None] * N + k_offsets[None, :],
        mask=m_mask[:, None] & k_mask[None, :],
        other=0.0
    )
    
    # Load B_T block: shape (BLOCK_SIZE_N, BLOCK_SIZE_K)
    B_T_block = tl.load(
        B_T_ptr + n_offsets[:, None] * N + k_offsets[None, :],
        mask=n_mask[:, None] & k_mask[None, :],
        other=0.0
    )
    
    # Compute dot product: (M, K) @ (K, N) -> (M, N)
    C_block = tl.dot(A_block, B_T_block.T)
    
    # Create mask for upper triangular elements within the block
    upper_mask = (m_mask[:, None] & n_mask[None, :]) & (m_offsets[:, None] <= n_offsets[None, :])
    
    # Store result only for upper triangular elements
    tl.store(
        C_ptr + m_offsets[:, None] * N + n_offsets[None, :],
        C_block,
        mask=upper_mask
    )


def triu_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    N = A.shape[0]
    
    # Transpose B to enable contiguous memory access for both operands in the kernel
    B_T = B.t().contiguous()
    
    C = torch.empty_like(A)
    
    # Grid configuration: one block per tile in the output matrix
    grid = (
        (N + 127) // 128,
        (N + 127) // 128,
        1
    )
    
    # Launch kernel
    triu_matmul_kernel[grid](
        A, B_T, C, N,
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triu_matmul(A, B)


N = 4096

def get_inputs():
    A = torch.triu(torch.rand(N, N))
    B = torch.triu(torch.rand(N, N))
    return [A, B]

def get_init_inputs():
    return []