import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def einsum_bijl_lk_bijk_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    b, i, j, k, l,
    stride_ab, stride_ai, stride_aj, stride_al,
    stride_bl, stride_bk,
    stride_cb, stride_ci, stride_cj, stride_ck,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Compute block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Compute starting indices for this block
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Create pointers for A and B
    A_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(b, i, j, l),
        strides=(stride_ab, stride_ai, stride_aj, stride_al),
        offsets=(pid_b, m_start, n_start, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K),
        order=(3, 2, 1)
    )
    
    B_block_ptr = tl.make_block_ptr(
        base=B_ptr,
        shape=(l, k),
        strides=(stride_bl, stride_bk),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0)
    )
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for _ in range(0, l, BLOCK_SIZE_K):
        # Load A and B tiles
        A_tile = tl.load(A_block_ptr, boundary_check=(0, 1, 2))
        B_tile = tl.load(B_block_ptr, boundary_check=(0, 1))
        
        # Perform matrix multiplication
        acc += tl.dot(A_tile, B_tile)
        
        # Advance pointers
        A_block_ptr = tl.advance(A_block_ptr, (0, 0, 0, BLOCK_SIZE_K))
        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_SIZE_K, 0))
    
    # Compute output pointer
    C_block_ptr = tl.make_block_ptr(
        base=C_ptr,
        shape=(b, i, j, k),
        strides=(stride_cb, stride_ci, stride_cj, stride_ck),
        offsets=(pid_b, m_start, n_start, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K),
        order=(3, 2, 1)
    )
    
    # Store result
    tl.store(C_block_ptr, acc, boundary_check=(0, 1, 2))

def triton_einsum_bijl_lk_bijk(A, B):
    """
    Custom Triton implementation of torch.einsum("bijl,lk->bijk", A, B)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only FP32 supported"
    
    b, i, j, l = A.shape
    _, k = B.shape
    
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Create output tensor
    C = torch.empty(b, i, j, k, dtype=torch.float32, device=A.device)
    
    # Define block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimensions
    grid_m = (i + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (j + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_b = b
    
    # Launch kernel
    grid = (grid_m, grid_n, grid_b)
    
    einsum_bijl_lk_bijk_kernel[grid](
        A_ptr=A.data_ptr(),
        B_ptr=B.data_ptr(),
        C_ptr=C.data_ptr(),
        b=b, i=i, j=j, k=k, l=l,
        stride_ab=A.stride(0), stride_ai=A.stride(1), stride_aj=A.stride(2), stride_al=A.stride(3),
        stride_bl=B.stride(0), stride_bk=B.stride(1),
        stride_cb=C.stride(0), stride_ci=C.stride(1), stride_cj=C.stride(2), stride_ck=C.stride(3),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_einsum_bijl_lk_bijk(A, B)