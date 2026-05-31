import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def group_norm_stats_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    B, C, H, W, G,
    C_per_group,
    S_spatial,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, group) pair
    bid = tl.program_id(0)
    b = bid // G
    g = bid % G
    
    sum_val = 0.0
    sum_sq_val = 0.0
    
    # Loop over channels within the group
    for c_rel in range(C_per_group):
        c = g * C_per_group + c_rel
        # Loop over spatial dimensions in blocks
        for hw_start in range(0, S_spatial, BLOCK_SIZE):
            offsets = hw_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < S_spatial
            # Index: b * (C * H * W) + c * (H * W) + hw
            ptr = x_ptr + b * (C * S_spatial) + c * S_spatial + offsets
            val = tl.load(ptr, mask=mask, other=0.0)
            sum_val += tl.sum(val, axis=0)
            sum_sq_val += tl.sum(val * val, axis=0)
            
    # Total elements in the group: C_per_group * H * W
    S_group = C_per_group * S_spatial
    mean = sum_val / S_group
    var = (sum_sq_val / S_group) - (mean * mean)
    
    # Store mean and var for this (batch, group)
    tl.store(mean_ptr + bid, mean)
    tl.store(var_ptr + bid, var)

@triton.jit
def group_norm_apply_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    var_ptr,
    gamma_ptr,
    beta_ptr,
    B, C, H, W, G,
    C_per_group,
    S_spatial,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid: (B * G, C_per_group, (S_spatial + BLOCK_SIZE - 1) // BLOCK_SIZE)
    bid = tl.program_id(0) # batch * G + group
    c_rel = tl.program_id(1)
    pid = tl.program_id(2)
    
    b = bid // G
    g = bid % G
    c = g * C_per_group + c_rel
    
    # Load group statistics
    mean = tl.load(mean_ptr + bid)
    var = tl.load(var_ptr + bid)
    
    # Load affine parameters for the specific channel
    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)
    
    # Load and normalize spatial block
    hw_start = pid * BLOCK_SIZE
    offsets = hw_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < S_spatial
    
    x_offset = b * (C * S_spatial) + c * S_spatial + offsets
    val = tl.load(x_ptr + x_offset, mask=mask)
    
    norm_val = (val - mean) / tl.sqrt(var + eps)
    out = norm_val * gamma + beta
    
    tl.store(out_ptr + x_offset, out, mask=mask)

def triton_group_norm(x, weight, bias, num_groups, eps=1e-5):
    B, C, H, W = x.shape
    G = num_groups
    C_per_group = C // G
    S_spatial = H * W
    
    x = x.contiguous()
    out = torch.empty_like(x)
    mean = torch.empty((B, G), device=x.device, dtype=x.dtype)
    var = torch.empty((B, G), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE = 1024
    
    # Step 1: Compute Mean and Variance
    stats_grid = (B * G,)
    group_norm_stats_kernel[stats_grid](
        x, mean, var, B, C, H, W, G, C_per_group, S_spatial, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Step 2: Apply Normalization and Affine Transform
    apply_grid = (B * G, C_per_group, (S_spatial + BLOCK_SIZE - 1) // BLOCK_SIZE)
    group_norm_apply_kernel[apply_grid](
        x, out, mean, var, weight, bias, B, C, H, W, G, C_per_group, S_spatial, 
        eps, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization using custom Triton kernels.
    """
    def __init__(self, num_features: int, num_groups: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.eps = 1e-5
        
        # Learnable parameters matching nn.GroupNorm
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor using Triton kernels.
        """
        return triton_group_norm(x, self.weight, self.bias, self.num_groups, self.eps)