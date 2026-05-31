import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def calc_stats_kernel(
    x_ptr, 
    stats_ptr, 
    N, C, H, W, G, CpG, group_size, 
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one group of one sample
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G
    
    # Offset to the start of the group in the (N, C, H, W) tensor
    # Elements in a group are contiguous across channels in the group and spatial dimensions
    offset = n * C * H * W + g * CpG * H * W
    
    sum_val = 0.0
    sum_sq_val = 0.0
    
    i = 0
    while i < group_size:
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < group_size
        vals = tl.load(x_ptr + offset + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(vals)
        sum_sq_val += tl.sum(vals * vals)
        i += BLOCK_SIZE
    
    mean = sum_val / group_size
    var = (sum_sq_val / group_size) - (mean * mean)
    
    # Store mean and variance for this group
    tl.store(stats_ptr + pid * 2, mean)
    tl.store(stats_ptr + pid * 2 + 1, var)

@triton.jit
def apply_norm_kernel(
    x_ptr, 
    out_ptr, 
    stats_ptr, 
    gamma_ptr, 
    beta_ptr, 
    N, C, H, W, G, CpG, 
    eps, 
    BLOCK_SIZE: tl.constexpr
):
    # Grid: (N, C, (H*W + BLOCK_SIZE - 1) // BLOCK_SIZE)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    g = pid_c // CpG
    
    # Spatial offsets for this block
    offsets = pid_hw * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (H * W)
    
    # Pointers for this (n, c)
    x_base = x_ptr + pid_n * (C * H * W) + pid_c * (H * W)
    out_base = out_ptr + pid_n * (C * H * W) + pid_c * (H * W)
    
    x = tl.load(x_base + offsets, mask=mask)
    
    # Load group stats and channel weights
    mean = tl.load(stats_ptr + (pid_n * G + g) * 2)
    var = tl.load(stats_ptr + (pid_n * G + g) * 2 + 1)
    gamma = tl.load(gamma_ptr + pid_c)
    beta = tl.load(beta_ptr + pid_c)
    
    # Normalize and scale/shift
    out = (x - mean) * tl.math.rsqrt(var + eps) * gamma + beta
    tl.store(out_base + offsets, out, mask=mask)

def triton_group_norm(x, weight, bias, num_groups, eps=1e-5):
    assert x.is_cuda, "Tensors must be on CUDA"
    x = x.contiguous().float()
    weight = weight.contiguous().float()
    bias = bias.contiguous().float()
    
    N, C, H, W = x.shape
    CpG = C // num_groups
    group_size = CpG * H * W
    
    # Buffer for mean and variance: (N * G, 2)
    stats = torch.empty((N * num_groups, 2), device=x.device, dtype=torch.float32)
    
    # Step 1: Calculate stats
    calc_grid = (N * num_groups,)
    calc_stats_kernel[calc_grid](
        x, stats, N, C, H, W, num_groups, CpG, group_size, 
        BLOCK_SIZE=1024
    )
    
    # Step 2: Apply normalization
    out = torch.empty_like(x)
    apply_grid = (N, C, (H * W + 1023) // 1024)
    apply_norm_kernel[apply_grid](
        x, out, stats, weight, bias, N, C, H, W, num_groups, CpG, eps, 
        BLOCK_SIZE=1024
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization using custom Triton kernels.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        
        # Learnable parameters for GroupNorm
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        return triton_group_norm(x, self.weight, self.bias, self.num_groups)