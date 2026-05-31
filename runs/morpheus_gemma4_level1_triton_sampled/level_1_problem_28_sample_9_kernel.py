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
    # Calculate the offset for the current program
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load the input data
    x = tl.load(x_ptr + offsets, mask=mask)

    # HardSigmoid calculation: clamp((x + 3) / 6, 0, 1)
    # Using tl.maximum and tl.minimum for clamping
    out = (x + 3.0) / 6.0
    out = tl.maximum(0.0, tl.minimum(1.0, out))

    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_hardsigmoid(x: torch.Tensor):
    """
    Triton wrapper for the HardSigmoid operation.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    n_elements = x.numel()
    out = torch.empty_like(x)
    
    # Block size can be tuned; 1024 is generally a good starting point for element-wise ops
    BLOCK_SIZE = 1024
    grid = ( (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    hardsigmoid_kernel[grid](
        x, 
        out, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
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