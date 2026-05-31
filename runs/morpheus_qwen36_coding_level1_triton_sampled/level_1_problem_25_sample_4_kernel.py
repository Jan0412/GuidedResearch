import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def swish_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sigmoid(x) = 1 / (1 + exp(-x))
    sig_x = 1.0 / (1.0 + tl.exp(-x))
    
    # Compute Swish: x * sigmoid(x)
    out = x * sig_x
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_swish(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the custom Triton Swish kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable block size for FP32 workloads

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    swish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_swish(x)