import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N, M, K, L,
    stride_a_n, stride_a_m, stride_a_k,
    stride_b_k, stride_b_l,
    stride_c_n, stride_c_m, stride_c_l,
    BLOCK_K: tl.constexpr = 128,
    BLOCK_L: tl.constexpr = 256,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_l = tl.program_id(2)
    
    k_offsets = tl.arange(0, BLOCK_K)
    l_offsets = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    
    # Base pointers for A and B
    A_ptr_base = A_ptr + pid_n * stride_a_n + pid_m * stride_a_m
    A_ptr_k = A_ptr_base + k_offsets * stride_a_k
    
    B_ptr_base = B_ptr + k_offsets[:, None] * stride_b_k + l_offsets[None, :] * stride_b_l
    
    acc = tl.zeros((BLOCK_L,), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        # Load A chunk
        mask_k = k_offsets < K - k
        A_chunk = tl.load(A_ptr_k, mask=mask_k, other=0.0)
        
        # Load B tile
        mask_k_tile = k_offsets[:, None] < K - k
        mask_l_tile = l_offsets[None, :] < L - pid_l * BLOCK_L
        mask_tile = mask_k_tile & mask_l_tile
        B_chunk = tl.load(B_ptr_base, mask=mask_tile, other=0.0)
        
        # Compute dot product
        acc += tl.dot(A_chunk.reshape(1, BLOCK_K), B_chunk).reshape(BLOCK_L,)
        
        # Advance pointers
        A_ptr_k += BLOCK_K * stride_a_k
        B_ptr_base += BLOCK_K * stride_b_k

    # Store result
    C_ptr_offset = C_ptr + pid_n * stride_c_n + pid_m * stride_c_m + l_offsets * stride_c_l
    mask_store = l_offsets < L - pid_l * BLOCK_L
    tl.store(C_ptr_offset, acc, mask=mask_store)


def triton_matmul(A, B):
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    
    N, M, K = A.shape
    K2, L = B.shape
    assert K == K2
    
    C = torch.empty((N, M, L), dtype=A.dtype, device=A.device)
    
    stride_a_n = A.stride(0)
    stride_a_m = A.stride(1)
    stride_a_k = A.stride(2)
    stride_b_k = B.stride(0)
    stride_b_l = B.stride(1)
    stride_c_n = C.stride(0)
    stride_c_m = C.stride(1)
    stride_c_l = C.stride(2)
    
    BLOCK_K = 128
    BLOCK_L = 256
    
    num_l_tiles = (L + BLOCK_L - 1) // BLOCK_L
    
    grid = (N, M, num_l_tiles)
    
    matmul_kernel[grid](
        A, B, C,
        N, M, K, L,
        stride_a_n, stride_a_m, stride_a_k,
        stride_b_k, stride_b_l,
        stride_c_n, stride_c_m, stride_c_l,
        BLOCK_K=BLOCK_K,
        BLOCK_L=BLOCK_L
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_matmul(A, B)


def get_inputs():
    N = 16
    M = 1024
    K = 2048
    L = 768
    A = torch.rand(N, M, K)
    B = torch.rand(K, L)
    return [A, B]

def get_init_inputs():
    return []