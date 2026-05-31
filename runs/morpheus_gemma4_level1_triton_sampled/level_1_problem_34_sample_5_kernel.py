import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def instance_norm_kernel(
    x_ptr, 
    out_ptr, 
    n_elements, 
    eps, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, channel) instance
    # n_elements = H * W
    
    # Pass 1: Compute Mean and Variance
    sum_val = 0.0
    sum_sq_val = 0.0
    
    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x_block = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Use mask to ensure we only sum valid elements
        sum_val += tl.sum(x_block * mask)
        sum_sq_val += tl.sum((x_block * x_block) * mask)
    
    mean = sum_val / n_elements
    var = (sum_sq_val / n_elements) - (mean * mean)
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Pass 2: Normalize and Store
    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x_block = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        out_block = (x_block - mean) * inv_std
        tl.store(out_ptr + offsets, out_block, mask=mask)

def triton_instance_norm(x: torch.Tensor, eps: float = 1e-5):
    """
    Triton wrapper for Instance Normalization.
    Input x shape: (N, C, H, W)
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    N, C, H, W = x.shape
    x = x.contiguous()
    
    out = torch.empty_like(x)
    n_elements = H * W
    
    # Grid is (N * C,), one program per instance
    grid = (N * C,)
    BLOCK_SIZE = 1024
    
    # We pass the pointers to the start of each instance inside the kernel
    # or calculate the offset using the program ID.
    # To simplify, we'll pass the base pointers and let the kernel handle offsets.
    
    # However, to make the kernel cleaner, we can launch it with a custom 
    # pointer logic or use a wrapper that calculates the offset.
    # Let's modify the kernel to take the base pointers and use program_id.
    
    # Redefining kernel slightly to handle (N, C) offset
    return _launch_instance_norm(x, out, N, C, n_elements, eps, BLOCK_SIZE, grid)

@triton.jit
def instance_norm_kernel_with_offset(
    x_ptr, 
    out_ptr, 
    N, C, 
    n_elements, 
    eps, 
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Offset to the start of the current (n, c) instance
    instance_offset = pid * n_elements
    
    # Local pointers for this instance
    x_ptr = x_ptr + instance_offset
    out_ptr = out_ptr + instance_offset
    
    sum_val = 0.0
    sum_sq_val = 0.0
    
    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x_block = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(x_block * mask)
        sum_sq_val += tl.sum((x_block * x_block) * mask)
    
    mean = sum_val / n_elements
    var = (sum_sq_val / n_elements) - (mean * mean)
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x_block = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        out_block = (x_block - mean) * inv_std
        tl.store(out_ptr + offsets, out_block, mask=mask)

def _launch_instance_norm(x, out, N, C, n_elements, eps, BLOCK_SIZE, grid):
    instance_norm_kernel_with_offset[grid](
        x, out, N, C, n_elements, eps, BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs Instance Normalization using Triton.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        # PyTorch InstanceNorm2d default eps is 1e-5
        return triton_instance_norm(x, eps=1e-5)