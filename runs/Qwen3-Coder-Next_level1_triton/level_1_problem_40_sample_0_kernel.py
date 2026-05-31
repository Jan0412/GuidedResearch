import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weight (gamma)
    B,  # pointer to the bias (beta)
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_xm,  # stride in the normalization dimension
    stride_xn,  # stride in the feature dimension
    M,  # number of rows (batch dimensions)
    N,  # number of columns (normalized dimensions)
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input
    row_start = tl.program_id(0)
    
    # Compute mean and variance for this row
    mean = 0.0
    var = 0.0
    
    # First pass: compute mean
    for col_start in range(0, N, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        
        x = tl.load(X + row_start * stride_xm + cols * stride_xn, mask=mask, other=0.0)
        mean += x
        
    mean = tl.sum(mean) / N
    
    # Second pass: compute variance
    for col_start in range(0, N, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        
        x = tl.load(X + row_start * stride_xm + cols * stride_xn, mask=mask, other=0.0)
        diff = x - mean
        var += diff * diff
        
    var = tl.sum(var) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Store mean and 1/std if requested
    if Mean is not None:
        tl.store(Mean + row_start, mean)
    if Rstd is not None:
        tl.store(Rstd + row_start, rstd)
    
    # Third pass: normalize and apply weight/bias
    for col_start in range(0, N, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        
        x = tl.load(X + row_start * stride_xm + cols * stride_xn, mask=mask, other=0.0)
        x_hat = (x - mean) * rstd
        
        # Load weight and bias
        w = tl.load(W + cols, mask=mask, other=0.0)
        b = tl.load(B + cols, mask=mask, other=0.0)
        
        y = x_hat * w + b
        
        tl.store(Y + row_start * stride_xm + cols * stride_xn, y, mask=mask)


class TritonLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        # Ensure input is contiguous
        x = x.contiguous()
        weight = weight.contiguous() if weight is not None else None
        bias = bias.contiguous() if bias is not None else None
        
        # Get dimensions
        *batch_dims, normalized_dim = x.shape
        N = normalized_dim
        
        # Compute total number of rows to process
        M = 1
        for d in batch_dims:
            M *= d
            
        # Reshape to 2D for easier processing
        x_2d = x.view(M, N)
        
        # Prepare output tensor
        y = torch.empty_like(x_2d)
        
        # Set up block size
        BLOCK_SIZE = min(1024, triton.next_power_of_2(N))
        
        # Calculate strides
        stride_xm = x_2d.stride(0)
        stride_xn = x_2d.stride(1)
        
        # Allocate mean and std buffers if needed (not strictly necessary for inference but useful for gradient computation)
        mean = torch.empty(M, dtype=x.dtype, device=x.device)
        rstd = torch.empty(M, dtype=x.dtype, device=x.device)
        
        # Launch kernel
        grid = (M,)
        layer_norm_kernel[grid](
            x_2d, y, weight, bias, mean, rstd,
            stride_xm, stride_xn,
            M, N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Save for backward pass
        ctx.save_for_backward(x_2d, weight, mean, rstd)
        ctx.normalized_shape = normalized_shape
        ctx.M = M
        ctx.N = N
        
        return y.view(*batch_dims, normalized_dim)
    
    @staticmethod
    def backward(ctx, grad_output):
        # This is a simplified backward that doesn't implement gradients
        # For a complete implementation, we would need to implement gradient computation
        # Since the problem only asks for the forward pass optimization, 
        # we'll return None for gradients to avoid complexity
        return None, None, None, None, None


def triton_layer_norm(x, normalized_shape, weight, bias, eps=1e-5):
    return TritonLayerNorm.apply(x, normalized_shape, weight, bias, eps)


class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization using Triton kernel.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer with Triton kernel implementation.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        
        # Initialize learnable parameters
        # For LayerNorm, weight and bias are of size normalized_shape[-1] or full normalized_shape
        # PyTorch LayerNorm supports per-element affine transformation
        # We'll create parameters that match the expected behavior
        
        # Calculate total normalized dimensions
        total_norm_dim = 1
        for dim in normalized_shape:
            total_norm_dim *= dim
        
        self.weight = nn.Parameter(torch.ones(total_norm_dim))
        self.bias = nn.Parameter(torch.zeros(total_norm_dim))
        self.eps = 1e-5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        # Handle the case where normalized_shape might be a tuple
        # We need to ensure our weight and bias match the normalized dimensions
        
        # For our implementation, we flatten all dimensions except the last one for normalization
        *batch_dims, last_dim = x.shape[:-1], x.shape[-1]
        
        # Reshape to 2D for processing
        M = 1
        for d in batch_dims:
            M *= d
        N = last_dim
        
        # Reshape input to 2D [M, N]
        x_2d = x.view(M, N)
        
        # Use our Triton kernel for layer normalization
        y_2d = triton_layer_norm(x_2d, (N,), self.weight, self.bias, self.eps)
        
        # Reshape back to original shape
        return y_2d.view(*batch_dims, N)