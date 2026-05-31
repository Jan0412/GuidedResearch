import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def instance_norm_forward_kernel(
    X_ptr,  # Input tensor pointer (B, C, H, W)
    Y_ptr,  # Output tensor pointer
    mean_ptr,  # Mean tensor pointer (B, C)
    var_ptr,  # Variance tensor pointer (B, C)
    weight_ptr,  # Gamma (scale) tensor pointer (C)
    bias_ptr,  # Beta (shift) tensor pointer (C)
    B, C, H, W,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one (batch, channel) pair
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Compute mean and variance for this (b, c) pair
    # We'll process H*W elements in chunks
    n_elements = H * W
    # Initialize accumulators for mean and variance
    sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sum_sq_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over the elements in blocks
    for start in range(0, n_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Calculate linear index: ((b * C + c) * H * W) + offset
        idx = ((b * C + c) * H * W) + offsets
        x = tl.load(X_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        
        sum_val = tl.where(mask, sum_val + x, sum_val)
        sum_sq_val = tl.where(mask, sum_sq_val + x * x, sum_sq_val)
    
    # Reduce to get final sum and sum of squares
    sum_val = tl.sum(sum_val, axis=0)
    sum_sq_val = tl.sum(sum_sq_val, axis=0)
    
    # Compute mean and variance
    mean = sum_val / n_elements
    var = sum_sq_val / n_elements - mean * mean
    
    # Store mean and variance if needed (for backward pass or verification)
    if mean_ptr is not None:
        tl.store(mean_ptr + b * C + c, mean)
    if var_ptr is not None:
        tl.store(var_ptr + b * C + c, var)
    
    # Compute standard deviation
    std = tl.sqrt(var + eps)
    
    # Load gamma and beta
    gamma = 1.0
    beta = 0.0
    if weight_ptr is not None:
        gamma = tl.load(weight_ptr + c).to(tl.float32)
    if bias_ptr is not None:
        beta = tl.load(bias_ptr + c).to(tl.float32)
    
    # Normalize and scale
    for start in range(0, n_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        idx = ((b * C + c) * H * W) + offsets
        x = tl.load(X_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        
        # Normalize
        x_norm = (x - mean) / std
        # Scale and shift
        y = gamma * x_norm + beta
        
        tl.store(Y_ptr + idx, y.to(X_ptr.dtype.element_ty), mask=mask)


class TritonInstanceNorm2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        B, C, H, W = x.shape
        y = torch.empty_like(x)
        
        # Precompute mean and variance (we'll store them for backward pass)
        mean = torch.empty(B, C, device=x.device, dtype=torch.float32)
        var = torch.empty(B, C, device=x.device, dtype=torch.float32)
        
        # Configure grid
        grid = (B, C)
        BLOCK_SIZE = 256
        
        # Launch kernel
        instance_norm_forward_kernel[grid](
            x, y, mean, var, weight, bias, B, C, H, W, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, bias, mean, var)
        ctx.eps = eps
        ctx.B, ctx.C, ctx.H, ctx.W = B, C, H, W
        
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, mean, var = ctx.saved_tensors
        eps = ctx.eps
        B, C, H, W = ctx.B, ctx.C, ctx.H, ctx.W
        
        # Compute standard deviation
        std = torch.sqrt(var + eps)
        
        # Initialize gradients
        grad_x = torch.zeros_like(x)
        grad_weight = torch.zeros_like(weight) if weight is not None else None
        grad_bias = torch.zeros_like(bias) if bias is not None else None
        
        # Compute gradients
        # For each (b, c), compute gradient w.r.t. x, weight, bias
        for b in range(B):
            for c in range(C):
                # Extract slice for this (b, c)
                x_slice = x[b, c].view(-1)  # H*W
                grad_out_slice = grad_output[b, c].view(-1)  # H*W
                
                # Get mean and std for this (b, c)
                m = mean[b, c]
                s = std[b, c]
                
                # Gamma and beta values
                g = weight[c] if weight is not None else 1.0
                # beta is not used in gradient computation for x
                
                # Compute gradients
                # dy/dx = gamma / std
                # Also need to account for the fact that mean depends on x
                n = H * W
                dx_norm = grad_out_slice * g
                
                # Gradient w.r.t. x
                # dL/dx = (dL/dy * gamma - mean(dL/dy * gamma) - mean(dL/dy * gamma * x_norm) * x_norm) / std
                dx = (dx_norm - torch.mean(dx_norm) - torch.mean(dx_norm * x_slice) * x_slice) / s
                
                grad_x[b, c] = dx.view(H, W)
                
                # Gradient w.r.t. gamma and beta (only if they exist)
                if weight is not None:
                    x_norm = (x_slice - m) / s
                    grad_weight[c] += torch.sum(grad_out_slice * x_norm)
                if bias is not None:
                    grad_bias[c] += torch.sum(grad_out_slice)
        
        return grad_x, grad_weight, grad_bias, None


class ModelNew(nn.Module):
    """
    Optimized model that performs Instance Normalization using Triton kernels.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        # Initialize weight and bias as in PyTorch's InstanceNorm2d
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5  # Default epsilon for numerical stability
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization to the input tensor using custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return TritonInstanceNorm2d.apply(x, self.weight, self.bias, self.eps)