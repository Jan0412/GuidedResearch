import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def frobenius_norm_kernel(
    x_ptr,
    norm_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_squared = x * x
    tl.atomic_add(norm_ptr, tl.sum(x_squared), sem="acq_rel")

@triton.jit
def normalize_kernel(
    x_ptr,
    norm_val,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x / norm_val
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_frobenius_norm(x: torch.Tensor):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Initialize norm tensor
    norm_tensor = torch.zeros(1, dtype=torch.float32, device=x.device)
    
    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel for computing squared sum
    frobenius_norm_kernel[grid](x, norm_tensor, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Get the actual norm value
    norm_val = torch.sqrt(norm_tensor.item())
    
    return norm_val

def triton_normalize(x: torch.Tensor, norm_val: float):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel for normalization
    normalize_kernel[grid](x, norm_val, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs Frobenius norm normalization using Triton kernels.
    """
    def __init__(self):
        """
        Initializes the Frobenius norm normalization layer.
        """
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Frobenius norm normalization to the input tensor using Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        # Compute the Frobenius norm using Triton
        norm = triton_frobenius_norm(x)
        
        # Normalize the tensor using Triton
        return triton_normalize(x, norm)