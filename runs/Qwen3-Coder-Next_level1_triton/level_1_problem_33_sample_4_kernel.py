import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel for batch normalization inference
@triton.jit
def batchnorm_forward_kernel(
    x_ptr, y_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    n_elements, C, HxW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one feature channel
    c = tl.program_id(0)
    
    # Load statistics for this channel
    mean = tl.load(running_mean_ptr + c)
    var = tl.load(running_var_ptr + c)
    
    # Load weight and bias if available, else use default values
    w = tl.load(weight_ptr + c) if weight_ptr is not None else 1.0
    b = tl.load(bias_ptr + c) if bias_ptr is not None else 0.0
    
    # Compute normalization factor
    std = tl.sqrt(var + eps)
    scale = w / std
    shift = b - mean * scale
    
    # Process elements in this channel
    # Total elements in one channel = H * W
    # We'll process in blocks across spatial dimensions
    block_size_spatial = BLOCK_SIZE // (HxW) + 1  # Ensure we cover all spatial elements
    
    # Calculate starting offset for this channel
    channel_stride = C * HxW
    channel_start = c * HxW
    
    # Process in spatial blocks
    for i in range(0, n_elements, channel_stride):
        # Process elements in this batch
        batch_offset = i
        for j in range(0, HxW, BLOCK_SIZE):
            spatial_idx = j
            offsets = batch_offset + channel_start + spatial_idx + tl.arange(0, BLOCK_SIZE)
            mask = offsets < (i + channel_stride + HxW)  # Ensure we don't go beyond current batch
            
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y = x * scale + shift
            tl.store(y_ptr + offsets, y, mask=mask)

# Triton kernel for batch normalization training (with update of running stats)
@triton.jit
def batchnorm_train_kernel(
    x_ptr, y_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    save_mean_ptr, save_invstd_ptr,
    n_elements, C, HxW, total_elements,
    momentum: tl.constexpr, eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # First pass: compute mean and variance per channel
    c = tl.program_id(0)
    
    # Compute sum and sum of squares for this channel
    sum_val = 0.0
    sum_sq_val = 0.0
    count = 0
    
    # Iterate through all elements in this channel
    for i in range(0, total_elements, C * HxW):
        for j in range(0, HxW):
            offset = i + c * HxW + j
            if offset < total_elements:
                x = tl.load(x_ptr + offset)
                sum_val += x
                sum_sq_val += x * x
                count += 1
    
    # Compute mean and variance
    mean = sum_val / count
    var = sum_sq_val / count - mean * mean
    
    # Update running statistics
    running_mean_new = (1 - momentum) * tl.load(running_mean_ptr + c) + momentum * mean
    running_var_new = (1 - momentum) * tl.load(running_var_ptr + c) + momentum * var * count / (count - 1) if count > 1 else var
    
    # Save for backward pass (using simplified version)
    invstd = 1.0 / tl.sqrt(var + eps)
    
    tl.store(save_mean_ptr + c, mean)
    tl.store(save_invstd_ptr + c, invstd)
    
    # Store updated running statistics
    tl.store(running_mean_ptr + c, running_mean_new)
    tl.store(running_var_ptr + c, running_var_new)
    
    # Second pass: apply normalization
    w = tl.load(weight_ptr + c) if weight_ptr is not None else 1.0
    b = tl.load(bias_ptr + c) if bias_ptr is not None else 0.0
    
    scale = w * invstd
    shift = b - mean * scale
    
    for i in range(0, total_elements, C * HxW):
        batch_offset = i
        for j in range(0, HxW, BLOCK_SIZE):
            offsets = batch_offset + c * HxW + j + tl.arange(0, BLOCK_SIZE)
            mask = offsets < total_elements
            
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y = x * scale + shift
            tl.store(y_ptr + offsets, y, mask=mask)


def triton_batchnorm(x, running_mean, running_var, weight, bias, training, momentum, eps):
    """
    Triton implementation of BatchNorm2d forward pass.
    
    Args:
        x: Input tensor of shape (batch_size, num_features, H, W)
        running_mean: Running mean buffer
        running_var: Running variance buffer
        weight: Scale parameter
        bias: Shift parameter
        training: Whether in training mode
        momentum: Momentum for running stats update
        eps: Small value for numerical stability
    
    Returns:
        Output tensor and updated running statistics if training
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, C, H, W = x.shape
    HxW = H * W
    total_elements = x.numel()
    
    # Prepare output tensor
    y = torch.empty_like(x)
    
    # Allocate buffers for saved statistics if training
    if training:
        save_mean = torch.empty(C, device=x.device, dtype=x.dtype)
        save_invstd = torch.empty(C, device=x.device, dtype=x.dtype)
    else:
        save_mean = None
        save_invstd = None
    
    BLOCK_SIZE = 256
    
    if training:
        # Use training kernel with stats update
        grid = (C,)
        batchnorm_train_kernel[grid](
            x, y, running_mean, running_var, weight, bias,
            save_mean, save_invstd,
            total_elements, C, HxW, total_elements,
            momentum=momentum, eps=eps, BLOCK_SIZE=BLOCK_SIZE
        )
        return y, save_mean, save_invstd
    else:
        # Use inference kernel
        grid = (C,)
        batchnorm_forward_kernel[grid](
            x, y, running_mean, running_var, weight, bias,
            total_elements, C, HxW,
            eps=eps, BLOCK_SIZE=BLOCK_SIZE
        )
        return y, None, None


class ModelNew(nn.Module):
    """
    Optimized BatchNorm2d using Triton kernels.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer with Triton implementation.
        
        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('weight', torch.ones(num_features))
        self.register_buffer('bias', torch.zeros(num_features))
        self.training = True  # Default to training mode
        self.momentum = 0.1
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization using Triton kernels.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, H, W).
        
        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied.
        """
        # Ensure input is contiguous and on GPU
        x = x.contiguous()
        
        # Call our Triton batchnorm implementation
        output, _, _ = triton_batchnorm(
            x, self.running_mean, self.running_var,
            self.weight, self.bias, self.training, self.momentum, self.eps
        )
        
        return output

    def train(self, mode=True):
        """Set the module in training mode."""
        super().train(mode)
        self.training = mode
        return self

    def eval(self):
        """Set the module in evaluation mode."""
        super().eval()
        self.training = False
        return self