import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Constants for the GELU approximation
    c1 = 0.7978845608028654  # sqrt(2 / pi)
    c2 = 0.044715
    
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute GELU: 0.5 * x * (1.0 + tanh(c1 * (x + c2 * x^3)))
    x3 = x * x * x
    inner = c1 * (x + c2 * x3)
    tanh_val = tl.math.tanh(inner)
    out = 0.5 * x * (1.0 + tanh_val)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_gelu(x: torch.Tensor):
    """
    Wrapper function to launch the custom GELU Triton kernel.
    """
    assert x.is_cuda and x.dtype == torch.float32, "Input must be a contiguous FP32 CUDA tensor."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    
    # Tuned block size for large tensors (8192x8192)
    BLOCK_SIZE = 4096
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x):
        return triton_gelu(x)