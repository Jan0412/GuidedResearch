import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr,
    out_ptr,
    rms_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(0)
    
    # Calculate the starting offset for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to ensure we don't go out of bounds
    mask = offsets < N
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared values
    x_sq = x * x
    
    # Store squared values for reduction
    tl.store(out_ptr + offsets, x_sq, mask=mask)
    
    # Synchronize threads before reduction
    tl.sync()

@triton.jit
def rms_reduce_kernel(
    x_ptr,
    rms_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(0)
    
    # Calculate the starting offset for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to ensure we don't go out of bounds
    mask = offsets < N
    
    # Load squared values
    x_sq = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Sum up squared values
    sum_sq = tl.sum(x_sq, axis=0)
    
    # Store the sum
    tl.store(rms_ptr + pid, sum_sq, mask=pid < (N + BLOCK_SIZE - 1) // BLOCK_SIZE)

@triton.jit
def rms_norm_final_kernel(
    x_ptr,
    out_ptr,
    rms_val,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(0)
    
    # Calculate the starting offset for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to ensure we don't go out of bounds
    mask = offsets < N
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Normalize using RMS value
    normalized = x / (rms_val + 1e-5)  # Add epsilon to prevent division by zero
    
    # Store the result
    tl.store(out_ptr + offsets, normalized, mask=mask)

def triton_rms_norm(x: torch.Tensor, eps: float = 1e-5):
    """
    Triton implementation of RMS normalization.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    
    # Flatten the tensor to 2D for processing
    batch_size, features, *dims = x.shape
    flat_x = x.view(batch_size, features, -1)
    flat_x = flat_x.transpose(1, 2).contiguous().view(-1, features)
    
    # Get the total number of elements
    N = flat_x.numel()
    
    # Allocate intermediate tensors
    squared = torch.empty_like(flat_x)
    sum_squared = torch.empty((N + 127) // 128, dtype=torch.float32, device=x.device)
    
    # Calculate grid size
    grid_size = (N + 127) // 128
    
    # First kernel: compute squared values
    rms_norm_kernel[(grid_size,),](flat_x, squared, sum_squared, N, BLOCK_SIZE=128)
    
    # Second kernel: reduce sum of squares
    rms_reduce_kernel[(grid_size,),](squared, sum_squared, N, BLOCK_SIZE=128)
    
    # Compute RMS value from reduced sum
    sum_val = sum_squared.sum().item()
    rms_value = torch.sqrt(sum_val / features + eps)
    
    # Third kernel: normalize the values
    out = torch.empty_like(flat_x)
    rms_norm_final_kernel[(grid_size,),](flat_x, out, rms_value, N, BLOCK_SIZE=128)
    
    # Reshape back to original shape
    out = out.view(batch_size, -1, features).transpose(1, 2)
    out = out.view(batch_size, features, *dims)
    
    return out

class ModelNew(nn.Module):
    """
    Optimized Model with custom Triton kernels for RMS Normalization.
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
        # Use the custom Triton implementation
        return triton_rms_norm(x, self.eps)