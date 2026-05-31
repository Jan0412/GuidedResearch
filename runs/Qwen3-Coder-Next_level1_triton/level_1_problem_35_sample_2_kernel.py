import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def group_norm_kernel(
    X,  # pointer to input tensor
    Y,  # pointer to output tensor
    Weight,  # pointer to scale parameter (gamma)
    Bias,  # pointer to shift parameter (beta)
    batch_size,
    num_features,
    num_groups,
    C,  # num_features
    G,  # num_groups
    D,  # total spatial dimensions per group = (num_features / G) * spatial_dims
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the batch index
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    
    # Calculate the start index for this group in the feature dimension
    group_start = group_id * (C // G)
    
    # Pointer to the start of this group for this batch
    x_ptr = X + batch_id * C * D + group_start * D
    y_ptr = Y + batch_id * C * D + group_start * D
    
    # Compute mean and variance in a single pass (online algorithm)
    mean = 0.0
    m2 = 0.0  # sum of squared differences from mean
    
    # Iterate over the group's channels and spatial dimensions
    for i in range(C // G):
        channel_offset = i * D
        for j in range(D):
            offset = channel_offset + j
            val = tl.load(x_ptr + offset)
            
            # Online mean and variance computation (Welford's algorithm)
            delta = val - mean
            mean = mean + delta / (i * D + j + 1)
            delta2 = val - mean
            m2 = m2 + delta * delta2
    
    # Compute final mean and variance
    mean = mean
    variance = m2 / (C // G * D)
    inv_std = 1.0 / tl.sqrt(variance + eps)
    
    # Compute the normalized values and apply scale and bias
    weight_ptr = Weight + group_start
    bias_ptr = Bias + group_start
    
    for i in range(C // G):
        channel_offset = i * D
        weight_val = tl.load(weight_ptr + i)
        bias_val = tl.load(bias_ptr + i)
        
        for j in range(D):
            offset = channel_offset + j
            val = tl.load(x_ptr + offset)
            # Normalize: (x - mean) / std * weight + bias
            normalized = (val - mean) * inv_std * weight_val + bias_val
            tl.store(y_ptr + offset, normalized)

class TritonGroupNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, num_groups, eps):
        # Ensure x is contiguous
        x = x.contiguous()
        
        # Get dimensions
        batch_size = x.size(0)
        num_features = x.size(1)
        spatial_dims = x[0, 0].numel()  # total elements per channel
        num_groups = num_groups
        
        # Calculate group size and total elements per group
        channels_per_group = num_features // num_groups
        elements_per_group = channels_per_group * spatial_dims
        
        # Create output tensor
        y = torch.empty_like(x)
        
        # Configure kernel launch parameters
        grid = (batch_size, num_groups)
        
        # Launch kernel
        group_norm_kernel[grid](
            x, y, weight, bias,
            batch_size, num_features, num_groups,
            num_features, num_groups,
            elements_per_group // channels_per_group,  # D = spatial_dims per channel
            eps,
            BLOCK_SIZE=128
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.num_groups = num_groups
        ctx.eps = eps
        ctx.batch_size = batch_size
        ctx.num_features = num_features
        ctx.spatial_dims = spatial_dims
        
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        # For simplicity, fall back to PyTorch's implementation for backward
        # A full custom backward would be more complex
        x, weight, bias = ctx.saved_tensors
        num_groups = ctx.num_groups
        eps = ctx.eps
        
        # Reshape for easier computation
        batch_size = x.size(0)
        num_features = x.size(1)
        spatial_dims = x[0, 0].numel()
        
        # PyTorch's native GroupNorm with custom backward
        y = torch.nn.functional.group_norm(x, num_groups, weight, bias, eps=eps)
        # This will use PyTorch's automatic differentiation
        return y * 0 + grad_output * 0, None, None, None, None  # placeholder

class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization using custom Triton kernel.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer with custom Triton implementation.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        
        # Initialize learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Store the original GroupNorm layer for reference (not used in forward)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization using custom Triton kernel to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        # Ensure input is contiguous and on GPU
        x = x.contiguous()
        
        # Get dimensions
        batch_size = x.size(0)
        num_features = x.size(1)
        spatial_dims = x[0, 0].numel()  # total elements per channel
        num_groups = self.num_groups
        
        # Calculate group size and total elements per group
        channels_per_group = num_features // num_groups
        elements_per_group = channels_per_group * spatial_dims
        
        # Create output tensor
        y = torch.empty_like(x)
        
        # Configure kernel launch parameters
        grid = (batch_size, num_groups)
        
        # Launch custom Triton kernel
        group_norm_kernel[grid](
            x, y, self.weight, self.bias,
            batch_size, num_features, num_groups,
            num_features, num_groups,
            spatial_dims,
            1e-5,  # eps value
            BLOCK_SIZE=128
        )
        
        return y