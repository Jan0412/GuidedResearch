import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_scalar_kernel(
    A_ptr,
    out_ptr,
    s,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the starting offset for this program
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to handle elements beyond the tensor size
    mask = offsets < (M * N)
    
    # Load data from A
    x = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    
    # Perform scalar multiplication
    out = x * s
    
    # Store result to out
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_matmul_scalar(A: torch.Tensor, s: float) -> torch.Tensor:
    """
    Wrapper function to launch the Triton kernel for matrix-scalar multiplication.
    """
    assert A.is_cuda, "Input tensor must be on CUDA."
    A = A.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(A)
    
    M, N = A.shape
    total_elements = M * N
    
    # Tunable block size
    BLOCK_SIZE = 4096
    
    # Calculate grid size
    grid = lambda meta: (triton.cdiv(total_elements, meta["BLOCK_SIZE"]),)
    
    # Launch the kernel
    matmul_scalar_kernel[grid](
        A, out, s, M, N, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        """
        Performs matrix-scalar multiplication using a custom Triton kernel.
        """
        return triton_matmul_scalar(A, s)


M = 16384 * 4
N = 4096 * 4

def get_inputs():
    A = torch.rand(M, N)
    s = 3.14
    return [A, s]

def get_init_inputs():
    return []