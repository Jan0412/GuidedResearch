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
    # Program ID determines which block of data this instance processes
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute tanh using Triton's math intrinsic for efficiency
    # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    # Using tl.math.tanh is optimized and numerically stable
    out = tl.math.tanh(x)

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_tanh(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the custom Triton tanh kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Total number of elements
    n_elements = x.numel()

    # Choose block size based on tensor size and hardware characteristics
    # 1024 is a good default for large tensors
    BLOCK_SIZE = 1024

    # Calculate grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch kernel
    tanh_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for Tanh activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies custom Triton-based Tanh activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Tanh applied, same shape as input.
        """
        return triton_tanh(x)


batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []