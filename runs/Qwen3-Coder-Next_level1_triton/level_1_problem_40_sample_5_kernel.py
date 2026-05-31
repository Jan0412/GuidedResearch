import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    X,  # Pointer to input tensor
    W,  # Pointer to weight tensor
    B,  # Pointer to bias tensor
    Y,  # Pointer to output tensor
    M,  # Number of rows (all dimensions except last)
    N,  # Number of columns (last dimension)
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (one normalization computation)
    row_idx = tl.program_id(0)
    
    # Compute row start pointer
    row_start = row_idx * N
    
    # Compute mean
    sum_x = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(0, N, BLOCK_SIZE):
        offsets = row_start + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_start + N
        x = tl.load(X + offsets, mask=mask, other=0.0)
        sum_x += x.to(tl.float32)
    
    # Use tree reduction for sum
    for i in range(BLOCK_SIZE // 2, 0, BLOCK_SIZE // 2):
        sum_x = sum_x[:i] + sum_x[i:i+i]
    
    mean = sum_x[0] / N
    
    # Compute variance
    sum_sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(0, N, BLOCK_SIZE):
        offsets = row_start + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_start + N
        x = tl.load(X + offsets, mask=mask, other=0.0)
        x = x.to(tl.float32)
        diff = x - mean
        sum_sq += diff * diff
    
    # Tree reduction for variance sum
    for i in range(BLOCK_SIZE // 2, 0, BLOCK_SIZE // 2):
        sum_sq = sum_sq[:i] + sum_sq[i:i+i]
    
    var = sum_sq[0] / N
    rstd = tl.rsqrt(var + eps)
    
    # Normalize and apply weight/bias
    for i in range(0, N, BLOCK_SIZE):
        offsets = row_start + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_start + N
        x = tl.load(X + offsets, mask=mask, other=0.0)
        x = x.to(tl.float32)
        
        # Normalize
        normalized = (x - mean) * rstd
        
        # Load weight and bias for this position
        w = tl.load(W + (offsets - row_start) % N, mask=mask)
        b = tl.load(B + (offsets - row_start) % N, mask=mask)
        
        # Apply weight and bias
        out = normalized * w.to(tl.float32) + b.to(tl.float32)
        
        # Store result
        tl.store(Y + offsets, out.to(X.dtype.element_ty), mask=mask)


def triton_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """
    Triton implementation of Layer Normalization.
    
    Args:
        x: Input tensor of shape (*, normalized_shape)
        weight: Weight tensor of shape (normalized_shape,)
        bias: Bias tensor of shape (normalized_shape,)
        eps: Small value for numerical stability
        
    Returns:
        Output tensor with same shape as input
    """
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Get input shape
    shape = x.shape
    normalized_shape = weight.shape
    
    # Reshape to 2D: (M, N) where M is product of all dims except last, N is last dim
    x_flat = x.view(-1, shape[-1])
    M, N = x_flat.shape
    
    # Create output tensor
    y = torch.empty_like(x)
    y_flat = y.view(-1, shape[-1])
    
    # Determine block size (tuned for FP32)
    BLOCK_SIZE = 256
    
    # Grid: one block per row
    grid = (M,)
    
    # Launch kernel
    layer_norm_kernel[grid](
        x_flat, weight, bias, y_flat,
        M, N, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized model with LayerNorm implemented using Triton kernel.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer with Triton implementation.
        
        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization using Triton kernel to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).
            
        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return triton_layer_norm(x, self.weight, self.bias)