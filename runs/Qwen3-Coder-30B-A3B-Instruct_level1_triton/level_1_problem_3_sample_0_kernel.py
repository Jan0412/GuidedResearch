import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    batch_size, m, n, k,
    stride_a_batch, stride_a_m, stride_a_k,
    stride_b_batch, stride_b_k, stride_b_n,
    stride_c_batch, stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    # Get the batch index for this program
    batch_idx = tl.program_id(0)
    # Get the group index for this program
    group_idx = tl.program_id(1)
    
    # Calculate the starting positions for this group
    m_start = group_idx * GROUP_M * BLOCK_M
    m_end = min((group_idx + 1) * GROUP_M * BLOCK_M, m)
    
    if batch_idx >= batch_size:
        return
    
    # Initialize pointers to matrices A and B
    a_ptr = A_ptr + batch_idx * stride_a_batch
    b_ptr = B_ptr + batch_idx * stride_b_batch
    
    # Initialize pointers to output matrix C
    c_ptr = C_ptr + batch_idx * stride_c_batch
    
    # Loop over the m dimension in chunks
    for m_block in range(m_start, m_end, BLOCK_M):
        # Loop over the n dimension in chunks
        for n_block in range(0, n, BLOCK_N):
            # Initialize accumulator
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            
            # Loop over the k dimension in chunks
            for k_block in range(0, k, BLOCK_K):
                # Load tiles from memory
                a_tile = tl.load(
                    a_ptr + 
                    (m_block + tl.arange(0, BLOCK_M)[:, None]) * stride_a_m +
                    (k_block + tl.arange(0, BLOCK_K)[None, :]) * stride_a_k,
                    mask=(
                        (m_block + tl.arange(0, BLOCK_M)[:, None]) < m) &
                        ((k_block + tl.arange(0, BLOCK_K)[None, :]) < k),
                    other=0.0
                )
                
                b_tile = tl.load(
                    b_ptr + 
                    (k_block + tl.arange(0, BLOCK_K)[:, None]) * stride_b_k +
                    (n_block + tl.arange(0, BLOCK_N)[None, :]) * stride_b_n,
                    mask=(
                        (k_block + tl.arange(0, BLOCK_K)[:, None]) < k) &
                        ((n_block + tl.arange(0, BLOCK_N)[None, :]) < n),
                    other=0.0
                )
                
                # Compute dot product
                acc += tl.dot(a_tile, b_tile)
            
            # Store result
            c_tile = acc.to(tl.float32)
            tl.store(
                c_ptr + 
                (m_block + tl.arange(0, BLOCK_M)[:, None]) * stride_c_m +
                (n_block + tl.arange(0, BLOCK_N)[None, :]) * stride_c_n,
                c_tile,
                mask=(
                    (m_block + tl.arange(0, BLOCK_M)[:, None]) < m) &
                    ((n_block + tl.arange(0, BLOCK_N)[None, :]) < n)
            )

def triton_bmm(A: torch.Tensor, B: torch.Tensor):
    """
    Performs batched matrix multiplication using Triton kernel.
    """
    assert A.is_cuda and B.is_cuda, "Both tensors must be on CUDA."
    assert A.dim() == 3 and B.dim() == 3, "Both tensors must be 3D."
    assert A.shape[0] == B.shape[0], "Batch dimensions must match."
    assert A.shape[2] == B.shape[1], "Inner dimensions must match."
    
    batch_size, m, k = A.shape
    _, _, n = B.shape
    
    # Prepare output tensor
    C = torch.empty(batch_size, m, n, dtype=torch.float32, device=A.device)
    
    # Define block sizes and group size
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    GROUP_M = 8
    
    # Calculate grid dimensions
    grid = (
        batch_size,
        (m + BLOCK_M - 1) // (GROUP_M * BLOCK_M)
    )
    
    # Launch kernel
    bmm_kernel[grid](
        A, B, C,
        batch_size, m, n, k,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_M, BLOCK_N, BLOCK_K,
        GROUP_M
    )
    
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_bmm(A, B)