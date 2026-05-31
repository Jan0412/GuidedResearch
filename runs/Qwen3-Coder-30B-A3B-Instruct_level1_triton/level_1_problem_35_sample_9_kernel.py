import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def group_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    batch_size,
    num_features,
    height,
    width,
    num_groups,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate global thread index
    pid = tl.program_id(0)
    group_size = num_features // num_groups
    group_id = pid // (batch_size * height * width)
    batch_id = (pid % (batch_size * height * width)) // (height * width)
    hw_id = (pid % (batch_size * height * width)) % (height * width)
    
    # Ensure valid thread index
    if group_id >= num_groups or batch_id >= batch_size or hw_id >= height * width:
        return
    
    # Calculate base pointers for this thread's data
    x_base = x_ptr + batch_id * (num_features * height * width) + group_id * (group_size * height * width) + hw_id
    out_base = out_ptr + batch_id * (num_features * height * width) + group_id * (group_size * height * width) + hw_id
    weight_base = weight_ptr + group_id * group_size
    bias_base = bias_ptr + group_id * group_size
    
    # Shared memory for reduction
    mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Load data into registers
    x_vals = tl.load(x_base + tl.arange(0, BLOCK_SIZE) * (height * width), mask=tl.arange(0, BLOCK_SIZE) < group_size, other=0.0)
    
    # Compute mean
    mean_val = tl.sum(x_vals) / group_size
    
    # Compute variance
    diff = x_vals - mean_val
    var_val = tl.sum(diff * diff) / group_size
    
    # Store mean and rstd
    if hw_id == 0 and batch_id == 0:
        mean_ptr[group_id] = mean_val
        rstd_ptr[group_id] = 1.0 / tl.sqrt(var_val + eps)
    
    # Synchronize to ensure mean/rstd are computed
    tl.sync()
    
    # Read mean and rstd
    mean_read = mean_ptr[group_id]
    rstd_read = rstd_ptr[group_id]
    
    # Normalize and apply scale/shift
    normalized = (x_vals - mean_read) * rstd_read
    weights = tl.load(weight_base + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < group_size, other=0.0)
    biases = tl.load(bias_base + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < group_size, other=0.0)
    
    output_vals = normalized * weights + biases
    
    # Store output
    tl.store(out_base + tl.arange(0, BLOCK_SIZE) * (height * width), output_vals, mask=tl.arange(0, BLOCK_SIZE) < group_size)

# More efficient version using proper reduction and better memory access patterns
@triton.jit
def group_norm_kernel_v2(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    batch_size,
    num_features,
    height,
    width,
    num_groups,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate thread index within a group
    group_id = tl.program_id(0)
    batch_id = tl.program_id(1)
    hw_id = tl.program_id(2)
    
    # Early exit if indices are invalid
    if group_id >= num_groups or batch_id >= batch_size or hw_id >= height * width:
        return
        
    # Calculate group size
    group_size = num_features // num_groups
    
    # Calculate base pointers
    x_base = x_ptr + batch_id * (num_features * height * width) + group_id * (group_size * height * width) + hw_id
    out_base = out_ptr + batch_id * (num_features * height * width) + group_id * (group_size * height * width) + hw_id
    weight_base = weight_ptr + group_id * group_size
    bias_base = bias_ptr + group_id * group_size
    
    # Shared memory for reduction
    mean_val = tl.zeros([1], dtype=tl.float32)
    var_val = tl.zeros([1], dtype=tl.float32)
    
    # Load data for this thread
    x_vals = tl.load(x_base + tl.arange(0, BLOCK_SIZE) * (height * width), mask=tl.arange(0, BLOCK_SIZE) < group_size, other=0.0)
    
    # Compute sum and squared sum
    sum_x = tl.sum(x_vals)
    sum_x2 = tl.sum(x_vals * x_vals)
    
    # Reduce across threads in the same block
    sum_x = tl.sum(sum_x)
    sum_x2 = tl.sum(sum_x2)
    
    # Compute mean and variance
    mean_local = sum_x / group_size
    var_local = (sum_x2 / group_size) - (mean_local * mean_local)
    
    # Store mean and rstd for this group
    if hw_id == 0:
        mean_ptr[group_id] = mean_local
        rstd_ptr[group_id] = 1.0 / tl.sqrt(var_local + eps)
    
    # Synchronize
    tl.sync()
    
    # Read mean and rstd
    mean_read = mean_ptr[group_id]
    rstd_read = rstd_ptr[group_id]
    
    # Apply normalization and scale/shift
    normalized = (x_vals - mean_read) * rstd_read
    weights = tl.load(weight_base + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < group_size, other=0.0)
    biases = tl.load(bias_base + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < group_size, other=0.0)
    
    output_vals = normalized * weights + biases
    
    # Store output
    tl.store(out_base + tl.arange(0, BLOCK_SIZE) * (height * width), output_vals, mask=tl.arange(0, BLOCK_SIZE) < group_size)

# Even more optimized version that computes all elements in one pass per group
@triton.jit
def group_norm_kernel_optimized(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    batch_size,
    num_features,
    height,
    width,
    num_groups,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Thread and block indexing
    group_id = tl.program_id(0)
    batch_id = tl.program_id(1)
    hw_id = tl.program_id(2)
    
    # Validate indices
    if group_id >= num_groups or batch_id >= batch_size or hw_id >= height * width:
        return
    
    # Calculate group size
    group_size = num_features // num_groups
    
    # Base pointers for this thread's data
    x_base = x_ptr + batch_id * (num_features * height * width) + group_id * (group_size * height * width) + hw_id
    out_base = out_ptr + batch_id * (num_features * height * width) + group_id * (group_size * height * width) + hw_id
    weight_base = weight_ptr + group_id * group_size
    bias_base = bias_ptr + group_id * group_size
    
    # Initialize accumulators
    sum_x = tl.zeros([1], dtype=tl.float32)
    sum_x2 = tl.zeros([1], dtype=tl.float32)
    
    # Load all elements in this group for this spatial location
    x_vals = tl.load(x_base + tl.arange(0, BLOCK_SIZE) * (height * width), mask=tl.arange(0, BLOCK_SIZE) < group_size, other=0.0)
    
    # Accumulate sum and sum of squares
    sum_x = tl.sum(x_vals)
    sum_x2 = tl.sum(x_vals * x_vals)
    
    # Reduce across all threads in block (assuming single block per group for now)
    sum_x = tl.sum(sum_x)
    sum_x2 = tl.sum(sum_x2)
    
    # Compute mean and variance
    mean_local = sum_x / group_size
    var_local = (sum_x2 / group_size) - (mean_local * mean_local)
    
    # Store statistics for this group
    if hw_id == 0 and batch_id == 0:
        mean_ptr[group_id] = mean_local
        rstd_ptr[group_id] = 1.0 / tl.sqrt(var_local + eps)
    
    # Wait for all threads to finish computation
    tl.sync()
    
    # Read mean and std
    mean_read = mean_ptr[group_id]
    rstd_read = rstd_ptr[group_id]
    
    # Normalize and apply scale/shift
    normalized = (x_vals - mean_read) * rstd_read
    weights = tl.load(weight_base + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < group_size, other=0.0)
    biases = tl.load(bias_base + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < group_size, other=0.0)
    
    output_vals = normalized * weights + biases
    
    # Store results
    tl.store(out_base + tl.arange(0, BLOCK_SIZE) * (height * width), output_vals, mask=tl.arange(0, BLOCK_SIZE) < group_size)

def triton_group_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, 
                     num_groups: int, eps: float = 1e-5) -> torch.Tensor:
    """
    Triton implementation of Group Normalization
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    assert weight.is_cuda and bias.is_cuda, "Weight and bias must be on CUDA"
    
    batch_size, num_features, height, width = x.shape
    
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Allocate memory for intermediate statistics
    group_size = num_features // num_groups
    mean = torch.empty(num_groups, dtype=torch.float32, device=x.device)
    rstd = torch.empty(num_groups, dtype=torch.float32, device=x.device)
    
    # Determine grid dimensions
    grid = (
        num_groups,  # groups
        batch_size,  # batches
        height * width  # spatial locations
    )
    
    # Choose block size based on group size
    BLOCK_SIZE = min(1024, group_size)
    if BLOCK_SIZE > 32:
        BLOCK_SIZE = 32
    
    # Launch kernel
    group_norm_kernel_optimized[grid](
        x, weight, bias, out, mean, rstd,
        batch_size, num_features, height, width, num_groups, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized Model using Triton kernels for Group Normalization.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer with optimized Triton kernel.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.eps = 1e-5
        
        # Initialize learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        return triton_group_norm(x, self.weight, self.bias, self.num_groups, self.eps)