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
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask)

    # HardSigmoid formula: clamp(0.16666666666666666 * (x + 3), 0, 1)
    # which is equivalent to clamp((x + 3) / 6, 0, 1)
    res = (x + 3.0) * (1.0 / 6.0)
    res = tl.maximum(0.0, tl.minimum(1.0, res))

    # Store the result
    tl.store(out_ptr + offsets, res, mask=mask)

def triton_hardsigmoid(x: torch.Tensor):
    """
    Triton wrapper for the HardSigmoid activation.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    out = torch.empty_like(x)
    n_elements = x.numel()
    
    # BLOCK_SIZE can be tuned; 1024 is generally a good default for element-wise kernels
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    hardsigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
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

def get_inputs():
    batch_size = 4096
    dim = 393216
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []