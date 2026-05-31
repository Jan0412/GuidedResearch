import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr,
    out_ptr,
    rms_ptr,
    n_features,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate RMS for this block
    # We need to compute mean of squares across feature dimension
    # For this kernel, we'll compute RMS per element and store it
    # But since we're doing this per element, we'll do it in two passes
    # First pass: compute sum of squares
    square = x * x
    tl.store(out_ptr + offsets, square, mask=mask)
    
    # Second pass: compute RMS and normalize
    # This is a simplified approach - in practice you'd need proper reduction
    # For now, let's compute RMS per batch element and broadcast
    # This kernel needs to be restructured for proper RMS computation
    
    # Actually, let's implement a better version that computes RMS properly
    # We'll compute the RMS for each sample in the batch
    # Let's rewrite this properly
    pass

@triton.jit
def rms_norm_mean_kernel(
    x_ptr,
    mean_ptr,
    n_elements,
    n_features,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    square = x * x
    tl.store(mean_ptr + offsets, square, mask=mask)

@triton.jit
def rms_norm_finalize_kernel(
    x_ptr,
    out_ptr,
    rms_ptr,
    n_elements,
    n_features,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    rms = tl.load(rms_ptr + (offsets // n_features), mask=mask, other=0.0)
    out = x / (rms + eps)
    tl.store(out_ptr + offsets, out, mask=mask)

# Better implementation using proper reduction for RMS
@triton.jit
def rms_norm_kernel_v2(
    x_ptr,
    out_ptr,
    rms_ptr,
    batch_size,
    features,
    dim1,
    dim2,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute for each batch element
    batch_idx = tl.program_id(0)
    
    if batch_idx >= batch_size:
        return
        
    # Calculate base offset for this batch element
    base_offset = batch_idx * features * dim1 * dim2
    
    # Compute mean of squares for this batch element
    mean_sq = 0.0
    for i in range(features * dim1 * dim2):
        offset = base_offset + i
        x_val = tl.load(x_ptr + offset)
        mean_sq += x_val * x_val
    
    # Normalize by total elements
    mean_sq /= (features * dim1 * dim2)
    
    # Add epsilon and take sqrt to get RMS
    rms = tl.sqrt(mean_sq + eps)
    
    # Store RMS for this batch element
    tl.store(rms_ptr + batch_idx, rms)
    
    # Normalize the entire batch element
    for i in range(features * dim1 * dim2):
        offset = base_offset + i
        x_val = tl.load(x_ptr + offset)
        out_val = x_val / rms
        tl.store(out_ptr + offset, out_val)

def triton_rms_norm(x: torch.Tensor, eps: float = 1e-5):
    """
    Triton-based RMS Normalization implementation
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, features, dim1, dim2 = x.shape
    out = torch.empty_like(x)
    
    # Create RMS storage
    rms = torch.empty(batch_size, dtype=torch.float32, device='cuda')
    
    # Grid configuration
    BLOCK_SIZE = 1024
    grid = lambda meta: (batch_size,)
    
    # Launch kernel
    rms_norm_kernel_v2[grid](
        x, out, rms, 
        batch_size, features, dim1, dim2, 
        eps, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized RMS Normalization using Triton kernels.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        return triton_rms_norm(x, self.eps)