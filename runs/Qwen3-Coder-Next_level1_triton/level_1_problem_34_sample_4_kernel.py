import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def instance_norm_kernel(
    x_ptr,  # Input tensor: (B, C, H, W)
    weight_ptr,  # Gamma (scale) parameter: (C,)
    bias_ptr,  # Beta (shift) parameter: (C,)
    y_ptr,  # Output tensor: (B, C, H, W)
    n_elements,  # Total number of elements
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, channel) pair
    bc_id = tl.program_id(0)
    batch_id = bc_id // C
    channel_id = bc_id % C
    
    # Compute the offset for this batch-channel pair
    # Each BC pair processes H*W elements
    hw_start = (batch_id * C * H * W) + (channel_id * H * W)
    
    # Compute mean and variance in one pass
    mean = 0.0
    var_sum = 0.0
    count = 0
    
    for hw_offset in range(0, H * W, BLOCK_SIZE):
        hw_idx = hw_start + hw_offset + tl.arange(0, BLOCK_SIZE)
        mask = hw_idx < (batch_id + 1) * C * H * W + channel_id * H * W
        
        # Load values
        x_vals = tl.load(x_ptr + hw_idx, mask=mask, other=0.0)
        
        # Update mean and variance using Welford's online algorithm
        for i in range(BLOCK_SIZE):
            if hw_offset + i < H * W:
                x_val = x_vals[i]
                count += 1
                delta = x_val - mean
                mean += delta / count
                var_sum += delta * (x_val - mean)
    
    # Compute final mean and variance
    mean = mean if count > 0 else 0.0
    var = var_sum / count if count > 0 else 0.0
    
    # Compute standard deviation with epsilon for numerical stability
    std = tl.sqrt(var + eps)
    
    # Load weight and bias for this channel
    w = tl.load(weight_ptr + channel_id) if weight_ptr is not None else 1.0
    b = tl.load(bias_ptr + channel_id) if bias_ptr is not None else 0.0
    
    # Normalize and apply affine transform
    for hw_offset in range(0, H * W, BLOCK_SIZE):
        hw_idx = hw_start + hw_offset + tl.arange(0, BLOCK_SIZE)
        mask = hw_idx < (batch_id + 1) * C * H * W + channel_id * H * W
        
        x_vals = tl.load(x_ptr + hw_idx, mask=mask, other=0.0)
        
        # Normalize: (x - mean) / std
        y_vals = (x_vals - mean) / std * w + b
        
        tl.store(y_ptr + hw_idx, y_vals, mask=mask)


def triton_instance_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """
    Triton implementation of InstanceNorm2d.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        weight: Gamma parameter of shape (C,)
        bias: Beta parameter of shape (C,)
        eps: Small value for numerical stability
        
    Returns:
        Normalized output tensor of same shape as x
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    B, C, H, W = x.shape
    n_elements = x.numel()
    
    # Prepare output tensor
    y = torch.empty_like(x)
    
    # Grid: one block for each (batch, channel) pair
    grid = (B * C,)
    
    # Launch kernel
    instance_norm_kernel[grid](
        x, weight, bias, y, n_elements,
        B=B, C=C, H=H, W=W,
        eps=eps,
        BLOCK_SIZE=256,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized model that performs Instance Normalization using Triton kernels.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer with Triton kernel implementation.
        
        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        # Initialize gamma and beta parameters (same as nn.InstanceNorm2d)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.eps = 1e-5
        self.momentum = 0.1
        self.track_running_stats = False  # InstanceNorm doesn't use running stats by default

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization to the input tensor using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).
            
        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return triton_instance_norm(x, self.weight, self.bias, self.eps)