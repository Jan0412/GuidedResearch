import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softplus_kernel(
    x_ptr,      # Pointer to input
    out_ptr,    # Pointer to output
    n_elements, # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask)

    # Softplus implementation: log(1 + exp(x))
    # For numerical stability, we use the identity:
    # softplus(x) = x if x > threshold else log(1 + exp(x))
    # A common threshold is 20.0, beyond which exp(x) is very large
    # and log(1 + exp(x)) is effectively x.
    res = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))

    # Store the result
    tl.store(out_ptr + offsets, res, mask=mask)

def triton_softplus(x: torch.Tensor):
    """
    Wrapper for the Triton softplus kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    # Ensure the input is contiguous for the kernel
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)

    # Total number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable block size for element-wise operations

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    softplus_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a Softplus activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softplus activation to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Softplus applied, same shape as input.
        """
        return triton_softplus(x)