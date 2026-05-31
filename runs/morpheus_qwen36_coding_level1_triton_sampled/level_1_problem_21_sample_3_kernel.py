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
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Numerically stable sigmoid implementation
    # For x >= 0: sigmoid(x) = 1 / (1 + exp(-x))
    # For x < 0:  sigmoid(x) = exp(x) / (1 + exp(x))
    # This avoids overflow in exp() for large magnitude inputs.
    exp_x = tl.exp(x)
    exp_neg_x = tl.exp(-x)
    
    out = tl.where(
        x >= 0,
        1.0 / (1.0 + exp_neg_x),
        exp_x / (1.0 + exp_x)
    )
    
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Wraps the Triton sigmoid kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable block size
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    sigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for Sigmoid activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_sigmoid(x)