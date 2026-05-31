import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def selu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Constants for SELU
    # alpha = 1.6732632423543772848170429916717
    # lambda = 1.0507009873554804934193349852946
    ALPHA = 1.6732632423543773
    LAMBDA = 1.0507009873554805
    LAMBDA_ALPHA = 1.75809934084633

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)

    # SELU: lambda * x if x > 0 else lambda * alpha * (exp(x) - 1)
    out = tl.where(x > 0, LAMBDA * x, LAMBDA_ALPHA * (tl.exp(x) - 1.0))

    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_selu(x: torch.Tensor):
    """
    Triton implementation of the SELU activation function.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    
    # Ensure input is contiguous for efficient memory access
    x = x.contiguous()
    n_elements = x.numel()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Block size for the kernel
    BLOCK_SIZE = 1024
    
    # Grid calculation
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    selu_kernel[grid](
        x, 
        out, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a SELU activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies SELU activation to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with SELU applied, same shape as input.
        """
        return triton_selu(x)