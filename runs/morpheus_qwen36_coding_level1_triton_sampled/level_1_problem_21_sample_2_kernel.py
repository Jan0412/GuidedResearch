import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate starting offset for this program
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a vector of offsets for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to prevent out-of-bounds memory access
    mask = offsets < n_elements
    
    # Load input data in FP32
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute Sigmoid: 1 / (1 + exp(-x))
    # Triton's math functions operate in FP32 by default, matching the precision requirement
    out = 1.0 / (1.0 + tl.exp(-x))
    
    # Store result back to global memory
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the custom Triton Sigmoid kernel.
    Optimized for large FP32 tensors with contiguous memory layout.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    
    n_elements = x.numel()
    # BLOCK_SIZE tuned for high memory throughput on large tensors
    BLOCK_SIZE = 1024
    
    # Dynamically compute grid size based on tensor length and block size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the kernel
    sigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_sigmoid(x)


batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []