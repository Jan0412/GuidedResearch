import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def batchnorm2d_kernel(
    x_ptr,          # Input tensor pointer [B, C, H, W]
    weight_ptr,     # Weight tensor pointer [C]
    bias_ptr,       # Bias tensor pointer [C]
    mean_ptr,       # Running mean pointer [C]
    var_ptr,        # Running var pointer [C]
    y_ptr,          # Output tensor pointer [B, C, H, W]
    n_elements,     # Total number of elements = B * C * H * W
    C: tl.constexpr,  # Number of channels
    H: tl.constexpr,  # Height
    W: tl.constexpr,  # Width
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID corresponds to the channel index
    c = tl.program_id(0)
    
    # Compute offsets for this channel
    # We'll process all B*H*W elements for this channel
    # Each block processes BLOCK_SIZE elements
    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Calculate the total number of elements per channel
    hw = H * W
    total_per_channel = n_elements // C
    
    # Compute batch statistics (mean and var) for this channel
    # We'll compute online mean and variance using Welford's algorithm
    
    # Initialize for Welford's algorithm
    mean = 0.0
    M = 0.0  # Count of elements processed
    M2 = 0.0  # Sum of squared differences from mean
    
    # Process all elements in this channel
    for i in range(0, total_per_channel, BLOCK_SIZE):
        # Calculate actual offsets for the full tensor
        # For channel c, we need to access elements at positions where (index // (H*W)) % C == c
        # More precisely: index = b * (C*H*W) + c * (H*W) + hw_idx
        hw_idx = tl.arange(0, BLOCK_SIZE) if i + BLOCK_SIZE <= total_per_channel else tl.arange(i, total_per_channel)
        if i + BLOCK_SIZE > total_per_channel:
            # Adjust for the last block
            actual_block_size = total_per_channel - i
            mask = tl.arange(0, BLOCK_SIZE) < actual_block_size
        else:
            actual_block_size = BLOCK_SIZE
            mask = tl.ones([BLOCK_SIZE], dtype=tl.int1)
        
        # Compute the actual tensor index for each element
        bhw_idx = i + tl.arange(0, BLOCK_SIZE)
        b = bhw_idx // (H * W)
        hw_offset = bhw_idx % (H * W)
        indices = b * (C * H * W) + c * (H * W) + hw_offset
        
        # Load the data
        x_val = tl.load(x_ptr + indices, mask=mask, other=0.0)
        
        # Update Welford's algorithm
        for j in range(actual_block_size):
            if mask[j]:
                x_j = x_val[j]
                M += 1.0
                delta = x_j - mean
                mean += delta / M
                delta2 = x_j - mean
                M2 += delta * delta2
    
    # Compute final variance
    if M > 1:
        var = M2 / M
    else:
        var = 0.0
    
    # Store computed statistics (this is simplified - in practice you'd want to use running stats)
    # For inference, we'd use provided running stats; for training, we'd update them
    # Here we'll assume we're using the computed batch statistics for simplicity
    tl.store(mean_ptr + c, mean)
    tl.store(var_ptr + c, var)
    
    # Now apply normalization
    inv_std = 1.0 / tl.sqrt(var + eps)
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + c)
    else:
        weight = 1.0
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + c)
    else:
        bias = 0.0
    
    # Normalize and apply weight/bias
    scale = weight * inv_std
    shift = bias - scale * mean
    
    # Apply transformation to all elements
    for i in range(0, total_per_channel, BLOCK_SIZE):
        hw_idx = tl.arange(0, BLOCK_SIZE) if i + BLOCK_SIZE <= total_per_channel else tl.arange(i, total_per_channel)
        if i + BLOCK_SIZE > total_per_channel:
            actual_block_size = total_per_channel - i
            mask = tl.arange(0, BLOCK_SIZE) < actual_block_size
        else:
            actual_block_size = BLOCK_SIZE
            mask = tl.ones([BLOCK_SIZE], dtype=tl.int1)
        
        bhw_idx = i + tl.arange(0, BLOCK_SIZE)
        b = bhw_idx // (H * W)
        hw_offset = bhw_idx % (H * W)
        indices = b * (C * H * W) + c * (H * W) + hw_offset
        
        x_val = tl.load(x_ptr + indices, mask=mask, other=0.0)
        y_val = x_val * scale + shift
        tl.store(y_ptr + indices, y_val, mask=mask)


class TritonBatchNorm2d(nn.Module):
    """
    Custom BatchNorm2d implementation using Triton kernels.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(TritonBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
    def forward(self, x):
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Get dimensions
        B, C, H, W = x.shape
        assert C == self.num_features, f"Expected {self.num_features} channels, got {C}"
        
        # Determine if we're in training mode or not
        if self.training:
            # Compute batch statistics using our kernel
            # For simplicity, we'll use the kernel to compute batch stats
            
            # Set up output tensor
            y = torch.empty_like(x)
            
            # Set up temporary storage for statistics
            mean = torch.empty(C, device=x.device)
            var = torch.empty(C, device=x.device)
            
            # Launch kernel
            # Grid: [num_channels, ceil(total_per_channel / BLOCK_SIZE)]
            total_per_channel = B * H * W
            BLOCK_SIZE = 256
            grid = (C, (total_per_channel + BLOCK_SIZE - 1) // BLOCK_SIZE)
            
            batchnorm2d_kernel[grid](
                x, self.weight, self.bias,
                mean, var, y,
                B * C * H * W,
                C, H, W,
                self.eps,
                BLOCK_SIZE=BLOCK_SIZE
            )
            
            # Update running statistics
            with torch.no_grad():
                # For batch norm, we typically use unbiased variance
                unbiased_var = var * total_per_channel / (total_per_channel - 1) if total_per_channel > 1 else var
                self.running_mean.mul_(1 - self.momentum).add_(mean * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(unbiased_var * self.momentum)
                self.num_batches_tracked.add_(1)
            
            return y
        else:
            # Inference mode: use running statistics
            y = torch.empty_like(x)
            
            # Use running statistics directly for normalization
            inv_std = 1.0 / torch.sqrt(self.running_var + self.eps)
            scale = self.weight * inv_std
            shift = self.bias - scale * self.running_mean
            
            # Reshape for broadcasting: [1, C, 1, 1]
            scale = scale.view(1, C, 1, 1)
            shift = shift.view(1, C, 1, 1)
            
            return x * scale + shift


class ModelNew(nn.Module):
    """
    Optimized model using Triton BatchNorm2d.
    """
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.bn = TritonBatchNorm2d(num_features=num_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)