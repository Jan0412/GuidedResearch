import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def leaky_relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    negative_slope: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Create offset blocks
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Apply LeakyReLU: max(x, 0) + negative_slope * min(x, 0)
    out = tl.where(x > 0, x, x * negative_slope)

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_leaky_relu(x: torch.Tensor, negative_slope: float) -> torch.Tensor:
    """
    Applies LeakyReLU activation using a custom Triton kernel.
    
    Args:
        x (torch.Tensor): Input tensor.
        negative_slope (float): The negative slope of the activation function.
        
    Returns:
        torch.Tensor: Output tensor with LeakyReLU applied.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable block size for performance
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    leaky_relu_kernel[grid](
        x_ptr=x.data_ptr(),
        out_ptr=out.data_ptr(),
        n_elements=n_elements,
        negative_slope=negative_slope,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a LeakyReLU activation using a custom Triton kernel.
    """
    def __init__(self, negative_slope: float = 0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LeakyReLU activation to the input tensor using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with LeakyReLU applied, same shape as input.
        """
        return triton_leaky_relu(x, self.negative_slope)