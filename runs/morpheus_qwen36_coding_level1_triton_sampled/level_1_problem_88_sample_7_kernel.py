import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Constants for GELU approximation
    alpha = 0.044715
    c = 0.7978845608028654  # sqrt(2 / pi)

    # Calculate block start and offsets
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute GELU: 0.5 * x * (1 + tanh(c * (x + alpha * x^3)))
    x3 = x * x * x
    t = c * (x + alpha * x3)
    out = 0.5 * x * (1.0 + tl.math.tanh(t))

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the custom GELU Triton kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    
    # Tunable block size
    BLOCK_SIZE = 128
    
    # Grid calculation
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x):
        # Replace the PyTorch GELU approximation with the custom Triton kernel
        return triton_gelu(x)


def get_inputs():
    batch_size = 8192
    dim = 8192
    return [torch.rand(batch_size, dim).cuda()]


def get_init_inputs():
    return []