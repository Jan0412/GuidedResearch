import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def group_norm_kernel(
    X,  # pointer to input tensor
    W,  # pointer to weight tensor
    B,  # pointer to bias tensor
    Y,  # pointer to output tensor
    N,  # number of channels
    G,  # number of groups
    C,  # channels per group
    H,  # spatial height
    W_spatial,  # spatial width
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one group in one batch element
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    
    # Calculate starting indices
    start_c = group_idx * C
    start_offset = batch_idx * N * H * W_spatial + start_c * H * W_spatial
    
    # Initialize accumulators for mean and variance
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    sum_sq_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Compute sum and sum of squares for mean/variance
    for i in range(0, C * H * W_spatial, BLOCK_SIZE):
        offsets = start_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < start_offset + C * H * W_spatial
        x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_val += x * mask
        sum_sq_val += x * x * mask
    
    # Reduce to get single values
    sum_val = tl.sum(sum_val)
    sum_sq_val = tl.sum(sum_sq_val)
    
    # Compute mean and variance
    count = C * H * W_spatial
    mean = sum_val / count
    var = sum_sq_val / count - mean * mean
    
    # Compute standard deviation with epsilon for numerical stability
    std = tl.sqrt(var + eps)
    
    # Apply normalization and affine transformation
    for i in range(0, C * H * W_spatial, BLOCK_SIZE):
        offsets = start_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < start_offset + C * H * W_spatial
        x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Normalize
        x_norm = (x - mean) / std
        
        # Apply weight and bias (broadcast weight/bias per channel)
        channel_idx = (offsets - start_offset) // (H * W_spatial)
        w_val = tl.load(W + start_c + channel_idx, mask=mask, other=0.0).to(tl.float32)
        b_val = tl.load(B + start_c + channel_idx, mask=mask, other=0.0).to(tl.float32)
        
        # Apply affine transformation
        out = x_norm * w_val + b_val
        
        tl.store(Y + offsets, out.to(X.dtype.element_ty), mask=mask)

class GroupNormTriton(nn.Module):
    """
    Triton implementation of Group Normalization.
    """
    def __init__(self, num_features: int, num_groups: int, eps: float = 1e-5, affine: bool = True):
        super(GroupNormTriton, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.eps = eps
        
        assert num_features % num_groups == 0, "num_features must be divisible by num_groups"
        
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Get dimensions
        batch_size, num_features, height, width = x.shape
        
        # Calculate channels per group
        channels_per_group = num_features // self.num_groups
        
        # Prepare output tensor
        output = torch.empty_like(x)
        
        # Set block size based on channels_per_group * height * width
        spatial_size = channels_per_group * height * width
        BLOCK_SIZE = min(128, max(32, spatial_size // 4))
        
        # Define grid: one block per batch element per group
        grid = (batch_size, self.num_groups)
        
        # Launch kernel
        group_norm_kernel[grid](
            x, self.weight, self.bias, output,
            num_features, self.num_groups, channels_per_group,
            height, width,
            eps=self.eps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output

class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization using custom Triton kernel.
    """
    def __init__(self, num_features: int, num_groups: int):
        super(ModelNew, self).__init__()
        self.gn = GroupNormTriton(num_groups=num_groups, num_channels=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gn(x)