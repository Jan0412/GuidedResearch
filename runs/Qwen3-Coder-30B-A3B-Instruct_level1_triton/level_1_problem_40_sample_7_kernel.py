import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(0)
    
    # Each program processes one row
    row_start = pid * N
    
    # Create offsets for this row
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (pid + 1) * N
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean
    sum_x = tl.sum(x, axis=0)
    mean = sum_x / N
    
    # Store mean for later use
    tl.store(mean_ptr + pid, mean)
    
    # Compute variance
    x_centered = x - mean
    x_squared = x_centered * x_centered
    sum_sq = tl.sum(x_squared, axis=0)
    var = sum_sq / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Store reciprocal standard deviation
    tl.store(rstd_ptr + pid, rstd)
    
    # Normalize and apply scale and bias
    x_norm = x_centered * rstd
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    out = x_norm * weight + bias
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """
    Triton implementation of Layer Normalization
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA"
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Get dimensions
    batch_size = x.shape[0]
    features = x.shape[1:]
    num_features = 1
    for f in features:
        num_features *= f
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Prepare mean and rstd tensors
    mean = torch.empty(batch_size, dtype=torch.float32, device=x.device)
    rstd = torch.empty(batch_size, dtype=torch.float32, device=x.device)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 1024
    if num_features < BLOCK_SIZE:
        BLOCK_SIZE = num_features
    
    # Determine grid size
    grid = (batch_size,)
    
    # Launch kernel
    layer_norm_kernel[grid](
        x,
        weight,
        bias,
        out,
        mean,
        rstd,
        num_features,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model with custom Triton kernels for Layer Normalization.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer with custom Triton implementation.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization using custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return triton_layer_norm(x, self.weight, self.bias)