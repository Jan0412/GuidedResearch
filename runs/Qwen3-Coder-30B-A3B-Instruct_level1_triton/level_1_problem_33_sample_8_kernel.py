import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def batch_norm_forward_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    var_ptr,
    output_ptr,
    N,
    C,
    HxW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get the program ID for the channel dimension
    channel_id = tl.program_id(0)
    
    if channel_id >= C:
        return
    
    # Load weight and bias for this channel
    weight = tl.load(weight_ptr + channel_id)
    bias = tl.load(bias_ptr + channel_id)
    
    # Load mean and variance for this channel
    mean = tl.load(mean_ptr + channel_id)
    var = tl.load(var_ptr + channel_id)
    
    # Calculate the standard deviation with epsilon for numerical stability
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Process all elements for this channel
    for i in range(0, HxW, BLOCK_SIZE):
        # Create offsets for this block
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < HxW
        
        # Load input data for this channel
        x_offsets = channel_id + offsets * C
        x_data = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        
        # Normalize and apply scale and shift
        normalized = (x_data - mean) * inv_std
        output = normalized * weight + bias
        
        # Store the result
        output_offsets = channel_id + offsets * C
        tl.store(output_ptr + output_offsets, output, mask=mask)

@triton.jit
def batch_norm_backward_kernel(
    x_ptr,
    grad_output_ptr,
    weight_ptr,
    mean_ptr,
    var_ptr,
    grad_x_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    N,
    C,
    HxW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get the program ID for the channel dimension
    channel_id = tl.program_id(0)
    
    if channel_id >= C:
        return
    
    # Load parameters for this channel
    weight = tl.load(weight_ptr + channel_id)
    mean = tl.load(mean_ptr + channel_id)
    var = tl.load(var_ptr + channel_id)
    
    # Calculate the standard deviation with epsilon
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Initialize gradients for weight and bias
    grad_weight = 0.0
    grad_bias = 0.0
    
    # Process all elements for this channel
    for i in range(0, HxW, BLOCK_SIZE):
        # Create offsets for this block
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < HxW
        
        # Load input and gradient data
        x_offsets = channel_id + offsets * C
        x_data = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        grad_output = tl.load(grad_output_ptr + x_offsets, mask=mask, other=0.0)
        
        # Compute intermediate values
        x_centered = x_data - mean
        normalized = x_centered * inv_std
        
        # Accumulate gradients for weight and bias
        grad_weight += tl.sum(grad_output * normalized)
        grad_bias += tl.sum(grad_output)
        
        # Compute gradient w.r.t. input
        grad_input = grad_output * weight * inv_std
        grad_input -= grad_output * tl.sum(grad_output) / HxW
        grad_input -= grad_output * normalized * tl.sum(grad_output * normalized) / HxW
        
        # Store gradient w.r.t. input
        tl.store(grad_x_ptr + x_offsets, grad_input, mask=mask)
    
    # Store accumulated gradients
    tl.atomic_add(grad_weight_ptr + channel_id, grad_weight)
    tl.atomic_add(grad_bias_ptr + channel_id, grad_bias)

def triton_batch_norm_forward(x, weight, bias, mean, var, eps=1e-5):
    """
    Triton implementation of BatchNorm forward pass
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    assert weight.is_cuda and bias.is_cuda and mean.is_cuda and var.is_cuda, "All parameters must be on CUDA"
    
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Prepare output tensor
    output = torch.empty_like(x)
    
    # Get dimensions
    N, C, H, W = x.shape
    HxW = H * W
    
    # Configure block size
    BLOCK_SIZE = 128
    
    # Grid configuration
    grid = lambda meta: (C,)
    
    # Launch kernel
    batch_norm_forward_kernel[grid](
        x, weight, bias, mean, var, output,
        N, C, HxW, eps, BLOCK_SIZE
    )
    
    return output

def triton_batch_norm_backward(x, grad_output, weight, mean, var, eps=1e-5):
    """
    Triton implementation of BatchNorm backward pass
    """
    assert x.is_cuda and grad_output.is_cuda, "Input tensors must be on CUDA"
    assert weight.is_cuda and mean.is_cuda and var.is_cuda, "Parameters must be on CUDA"
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    grad_output = grad_output.contiguous()
    
    # Prepare output gradients
    grad_x = torch.empty_like(x)
    grad_weight = torch.zeros_like(weight)
    grad_bias = torch.zeros_like(bias)
    
    # Get dimensions
    N, C, H, W = x.shape
    HxW = H * W
    
    # Configure block size
    BLOCK_SIZE = 128
    
    # Grid configuration
    grid = lambda meta: (C,)
    
    # Launch kernel
    batch_norm_backward_kernel[grid](
        x, grad_output, weight, mean, var, 
        grad_x, grad_weight, grad_bias,
        N, C, HxW, eps, BLOCK_SIZE
    )
    
    return grad_x, grad_weight, grad_bias

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for Batch Normalization operations.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.eps = 1e-5
        self.momentum = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization to the input tensor using Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        # For inference mode, use running statistics
        if not self.training:
            # Use the stored running mean and variance
            mean = self.running_mean
            var = self.running_var
            
            # Apply normalization
            x_normalized = (x - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + self.eps)
            
            # Apply scale and shift
            return x_normalized * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        else:
            # Training mode: compute batch statistics
            # For simplicity, we'll compute batch stats manually and then use our Triton kernel
            batch_mean = x.mean(dim=(0, 2, 3))
            batch_var = x.var(dim=(0, 2, 3), unbiased=False)
            
            # Update running statistics
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
                self.num_batches_tracked += 1
            
            # Use Triton kernel for forward pass
            return triton_batch_norm_forward(
                x, self.weight, self.bias, batch_mean, batch_var, self.eps
            )

# Note: The above implementation uses a simplified approach where we compute batch statistics
# in PyTorch and then delegate the actual normalization computation to Triton kernels.
# A full end-to-end Triton implementation would require more complex handling of the 
# batch statistics computation within Triton itself, which is more involved.