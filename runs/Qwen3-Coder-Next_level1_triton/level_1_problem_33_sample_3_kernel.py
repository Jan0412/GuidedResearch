import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def compute_stats_kernel(
    x_ptr,
    n_elements,
    num_features,
    feat_size,
    mean_ptr,
    var_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one feature channel
    feat_id = tl.program_id(0)
    
    # Compute offsets for this channel
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize accumulators
    sum_x = 0.0
    sum_x2 = 0.0
    
    # Iterate through all elements for this feature
    for i in range(0, n_elements, BLOCK_SIZE):
        actual_offsets = i + offsets
        mask = actual_offsets < n_elements
        
        # Load data for current feature channel
        x = tl.load(x_ptr + feat_id * feat_size + actual_offsets, mask=mask, other=0.0)
        
        sum_x += tl.sum(x * mask)
        sum_x2 += tl.sum(x * x * mask)
    
    # Compute mean and variance
    count = tl.load(n_elements // feat_size) * 1.0
    mean = sum_x / count
    var = (sum_x2 / count) - (mean * mean)
    
    # Store results
    tl.store(mean_ptr + feat_id, mean)
    tl.store(var_ptr + feat_id, var)


@triton.jit
def batchnorm_forward_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    eps,
    n_elements,
    num_features,
    feat_size,
    out_ptr,
    is_training: tl.constexpr,
    running_mean_ptr,
    running_var_ptr,
    momentum,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one feature channel
    feat_id = tl.program_id(0)
    
    # Load statistics for this feature
    mean = tl.load(mean_ptr + feat_id)
    var = tl.load(var_ptr + feat_id)
    
    # Load weight and bias if they exist
    w = 1.0
    b = 0.0
    if weight_ptr is not None:
        w = tl.load(weight_ptr + feat_id)
    if bias_ptr is not None:
        b = tl.load(bias_ptr + feat_id)
    
    # Compute standard deviation with epsilon for numerical stability
    std = tl.sqrt(var + eps)
    
    # Process elements for this feature
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Update running statistics if in training mode
    if is_training:
        # We'll handle running stats outside this kernel for simplicity
        pass
    
    # Iterate through all elements for this feature
    for i in range(0, n_elements, BLOCK_SIZE):
        actual_offsets = i + offsets
        mask = actual_offsets < n_elements
        
        # Load input
        x = tl.load(x_ptr + feat_id * feat_size + actual_offsets, mask=mask, other=0.0)
        
        # Apply batch normalization
        y = w * (x - mean) / std + b
        
        # Store result
        tl.store(out_ptr + feat_id * feat_size + actual_offsets, y, mask=mask)


def triton_batchnorm_forward(x, weight, bias, running_mean, running_var, training, momentum, eps):
    """
    Triton implementation of BatchNorm2d forward pass.
    
    Args:
        x: Input tensor of shape (batch_size, num_features, height, width)
        weight: Scale parameter of shape (num_features,)
        bias: Shift parameter of shape (num_features,)
        running_mean: Running mean buffer of shape (num_features,)
        running_var: Running variance buffer of shape (num_features,)
        training: Whether in training mode
        momentum: Momentum for running statistics update
        eps: Small value for numerical stability
        
    Returns:
        Output tensor and updated running statistics
    """
    batch_size, num_features, height, width = x.shape
    feat_size = height * width
    n_elements = batch_size * feat_size
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Compute mean and variance for each feature channel
    mean = torch.zeros(num_features, device=x.device)
    var = torch.zeros(num_features, device=x.device)
    
    # Launch kernel to compute statistics
    BLOCK_SIZE = 256
    grid = (num_features,)
    
    # Compute stats using Triton kernel
    compute_stats_kernel[grid](
        x, n_elements, num_features, feat_size, mean, var,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Update running statistics if in training mode
    if training:
        with torch.no_grad():
            # Update running mean and variance
            running_mean.mul_(1 - momentum).add_(mean * momentum)
            running_var.mul_(1 - momentum).add_(var * momentum)
    
    # Launch batch normalization kernel
    batchnorm_forward_kernel[grid](
        x, mean, var, weight, bias, eps, n_elements, num_features, feat_size, out,
        training,
        running_mean, running_var, momentum,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Batch Normalization using custom Triton kernels.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer with custom Triton implementation.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = 1e-5
        self.momentum = 0.1
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.training = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization using custom Triton kernel to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).
            
        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        return triton_batchnorm_forward(
            x, self.weight, self.bias, self.running_mean, self.running_var,
            self.training, self.momentum, self.eps
        )