import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = pid
    acc = tl.zeros((), dtype=tl.float32)

    # Iterate over K in chunks
    for start_k in range(0, K, BLOCK_SIZE_K):
        cols = start_k + tl.arange(0, BLOCK_SIZE_K)
        mask = cols < K

        # Load row chunk of A and corresponding chunk of B
        a = tl.load(A_ptr + row_idx * K + cols, mask=mask, other=0.0)
        b = tl.load(B_ptr + cols, mask=mask, other=0.0)

        # Element-wise multiply and accumulate
        acc += tl.sum(a * b)

    # Store the final dot product result
    tl.store(C_ptr + row_idx, acc, mask=row_idx < M)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wraps the Triton kernel for matrix-vector multiplication.
    Optimized for FP32 precision with tiled memory access.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    out = torch.empty(M, 1, dtype=torch.float32, device=A.device)
    
    BLOCK_SIZE_K = 128
    grid = (M,)
    
    matmul_kernel[grid](A, B, out, M, K, BLOCK_SIZE_K)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using a custom Triton kernel for matrix-vector multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)


M = 256 * 8  # 2048
K = 131072 * 8  # 1048576

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, 1)
    return [A, B]

def get_init_inputs():
    return []