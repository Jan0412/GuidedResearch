import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hardsigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Create a range of offsets for the current block
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask to handle cases where n_elements is not a multiple of BLOCK_SIZE
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Apply HardSigmoid: max(0, min(1, (x + 3) / 6))
    # This can be computed as:
    # y = (x + 3) / 6
    # y = tl.maximum(tl.minimum(y, 1.0), 0.0)
    y = (x + 3.0) / 6.0
    y = tl.maximum(tl.minimum(y, 1.0), 0.0)
    # Store the result
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_hardsigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Applies HardSigmoid activation using a custom Triton kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable block size

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    hardsigmoid_kernel[grid](
        x_ptr=x.data_ptr(),
        out_ptr=out.data_ptr(),
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a HardSigmoid activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies HardSigmoid activation to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with HardSigmoid applied, same shape as input.
        """
        return triton_hardsigmoid(x)