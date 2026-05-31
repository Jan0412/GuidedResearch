import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    b, i, j, l, k,
    BLOCK_K: tl.constexpr,
):
    idx = tl.program_id(0)
    
    # Base pointers for the current batch element
    row_ptr = A_ptr + idx * l
    out_ptr = C_ptr + idx * k
    
    # Load A block: shape (l,)
    a_block = tl.load(row_ptr + tl.arange(0, l))
    
    # Accumulator: shape (1, BLOCK_K)
    acc = tl.zeros((1, BLOCK_K), dtype=tl.float32)
    
    # Loop over k blocks
    for start_k in range(0, k, BLOCK_K):
        col_block = tl.arange(0, BLOCK_K)
        
        # B block offsets: shape (l, BLOCK_K)
        # B[r, c] offset = r * k + c
        b_offsets = tl.arange(0, l)[:, None] * k + (start_k + col_block)[None, :]
        b_block = tl.load(B_ptr + b_offsets)
        
        # Dot product: (1, l) @ (l, BLOCK_K) -> (1, BLOCK_K)
        a_block_2d = a_block.reshape(1, l)
        acc += tl.dot(a_block_2d, b_block)
    
    # Store result: flatten acc to match output stride
    tl.store(out_ptr + tl.arange(0, BLOCK_K), acc.reshape(BLOCK_K,))


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    b, i, j, l = A.shape
    l_B, k = B.shape
    assert l == l_B, "Inner dimensions must match."
    
    out = torch.empty((b, i, j, k), dtype=A.dtype, device=A.device)
    
    num_batches = b * i * j
    BLOCK_K = 256  # Tunable block size for k dimension
    
    grid = (num_batches,)
    
    matmul_kernel[grid](
        A.data_ptr(), B.data_ptr(), out.data_ptr(),
        b, i, j, l, k,
        BLOCK_K=BLOCK_K,
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_matmul(A, B)