import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sum_sq_kernel(
    x_ptr, 
    out_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sum of squares for this block
    sq = x * x
    res = tl.sum(sq, axis=0)
    
    # Store the partial sum
    tl.store(out_ptr + pid, res)

@triton.jit
def scale_kernel(
    x_ptr, 
    out_ptr, 
    norm, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Divide by the Frobenius norm
    out = x / norm
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_frobenius_norm(x: torch.Tensor):
    """
    Custom Triton implementation of Frobenius norm normalization.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous().view(-1) # Flatten to treat as 1D
    n_elements = x.numel()
    
    # 1. Compute partial sums of squares
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    partial_sums = torch.empty(grid_size, device=x.device, dtype=torch.float32)
    
    grid = (grid_size,)
    sum_sq_kernel[grid](x, partial_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # 2. Final reduction and square root (using PyTorch for the small reduced tensor)
    total_sum = torch.sum(partial_sums)
    norm = torch.sqrt(total_sum).item()
    
    # 3. Scale the original tensor
    out = torch.empty_like(x)
    scale_kernel[grid](x, out, norm, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out.view_as(x.view(n_elements)) # This will be reshaped back in ModelNew

class ModelNew(nn.Module):
    """
    Optimized model that performs Frobenius norm normalization using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Frobenius norm normalization to the input tensor using custom Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        original_shape = x.shape
        # Call the Triton-based normalization
        # We flatten the input and reshape the output to match the original shape
        out_flat = triton_frobenius_norm(x)
        return out_flat.view(original_shape)