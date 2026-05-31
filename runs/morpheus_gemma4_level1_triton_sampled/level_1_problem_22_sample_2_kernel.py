import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def tanh_kernel(
    x_ptr, 
    out_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask)

    # Compute tanh(x) = sign(x) * (1 - 2 / (exp(2 * |x|) + 1))
    # This form is numerically stable for large values of x
    abs_x = tl.abs(x)
    exp_2abs_x = tl.exp(2.0 * abs_x)
    res = 1.0 - 2.0 / (exp_2abs_x + 1.0)
    
    # Apply the sign of x to the result
    out = tl.where(x < 0, -res, res)

    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_tanh(x: torch.Tensor):
    """
    Triton wrapper for the tanh operation.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    # Ensure input is contiguous for pointer arithmetic
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    # Use a block size that is a power of 2
    BLOCK_SIZE = 1024

    # Grid calculation
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    tanh_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a Tanh activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Tanh activation to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Tanh applied, same shape as input.
        """
        return triton_tanh(x)