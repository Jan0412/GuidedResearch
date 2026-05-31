import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    X_ptr,  # Input tensor pointer
    W_ptr,  # Weight tensor pointer
    B_ptr,  # Bias tensor pointer
    Y_ptr,  # Output tensor pointer
    M, N,   # M = number of batches, N = normalized shape size
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one batch (one normalization instance)
    batch_idx = tl.program_id(0)
    
    # Offset to the start of this batch's data
    x_offset = batch_idx * N
    
    # Compute mean
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, N, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(X_ptr + x_offset + offsets, mask=mask, other=0.0)
        sum_val += x.to(tl.float32)
    
    mean = tl.sum(sum_val) / N
    
    # Compute variance
    var_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, N, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(X_ptr + x_offset + offsets, mask=mask, other=0.0)
        x_shifted = x - mean
        var_sum += x_shifted * x_shifted
    
    var = tl.sum(var_sum) / N
    std = tl.sqrt(var + eps)
    
    # Normalize and apply weight/bias
    for i in range(0, N, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        x = tl.load(X_ptr + x_offset + offsets, mask=mask, other=0.0)
        w = tl.load(W_ptr + offsets, mask=mask, other=1.0)
        b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
        
        # Normalize: (x - mean) / std
        x_norm = (x - mean) / std
        # Scale and shift: w * x_norm + b
        y = w * x_norm + b
        
        tl.store(Y_ptr + x_offset + offsets, y.to(X_ptr.dtype.element_ty), mask=mask)


def triton_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """
    Triton-based Layer Normalization implementation.
    
    Args:
        x: Input tensor of shape (*, normalized_shape)
        weight: Scale parameter of shape (normalized_shape,)
        bias: Bias parameter of shape (normalized_shape,)
        eps: Small constant for numerical stability
    
    Returns:
        Output tensor with same shape as x
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Get original shape and reshape to 2D: [batch_size, normalized_shape]
    original_shape = x.shape
    normalized_shape = weight.shape
    
    # Calculate total elements in the batch dimension
    batch_size = 1
    for dim in original_shape[:-len(normalized_shape)]:
        batch_size *= dim
    
    # Reshape input to 2D: [batch_size, normalized_shape_size]
    x_2d = x.view(batch_size, -1)
    
    # Prepare output tensor
    y = torch.empty_like(x_2d)
    
    # Get normalized shape size
    normalized_size = x_2d.shape[1]
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 256  # Tunable parameter
    grid = (batch_size,)
    
    # Launch kernel
    layer_norm_kernel[grid](
        x_2d, weight, bias, y,
        batch_size, normalized_size,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape output back to original shape
    return y.view(*original_shape)


class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization using custom Triton kernel.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer with Triton kernel implementation.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = 1e-5
        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization using Triton kernel to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return triton_layer_norm(x, self.weight, self.bias, self.eps)