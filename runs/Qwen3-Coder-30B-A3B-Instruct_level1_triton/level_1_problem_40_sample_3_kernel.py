import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    N,
    M,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPS: tl.constexpr
):
    # Compute mean and variance for each row
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE)
    num_groups = num_pid_m // GROUP_SIZE_M
    
    if num_groups == 0:
        num_groups = 1
    
    group_id = pid // GROUP_SIZE_M
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(GROUP_SIZE_M, num_pid_m - first_pid_m)
    
    if group_id < num_groups - 1:
        end_m = first_pid_m + group_size_m
    else:
        end_m = num_pid_m
    
    # Find the range of rows this thread block processes
    start_m = first_pid_m + (pid % GROUP_SIZE_M)
    
    if start_m >= num_pid_m:
        return
    
    # Process each row
    for m in range(start_m, end_m, GROUP_SIZE_M):
        # Calculate offset for this row
        row_offset = m * N
        
        # Compute mean
        mean = 0.0
        for i in range(0, N, BLOCK_SIZE):
            off = row_offset + i + tl.arange(0, BLOCK_SIZE)
            mask = off < row_offset + N
            x_vals = tl.load(x_ptr + off, mask=mask, other=0.0)
            mean += tl.sum(x_vals)
        
        mean = mean / N
        
        # Store mean
        tl.store(mean_ptr + m, mean)
        
        # Compute variance
        var = 0.0
        for i in range(0, N, BLOCK_SIZE):
            off = row_offset + i + tl.arange(0, BLOCK_SIZE)
            mask = off < row_offset + N
            x_vals = tl.load(x_ptr + off, mask=mask, other=0.0)
            diff = x_vals - mean
            var += tl.sum(diff * diff)
        
        var = var / N
        rstd = 1.0 / tl.sqrt(var + EPS)
        
        # Store reciprocal standard deviation
        tl.store(rstd_ptr + m, rstd)
        
        # Normalize and apply scale/shift
        for i in range(0, N, BLOCK_SIZE):
            off = row_offset + i + tl.arange(0, BLOCK_SIZE)
            mask = off < row_offset + N
            
            x_vals = tl.load(x_ptr + off, mask=mask, other=0.0)
            norm_vals = (x_vals - mean) * rstd
            
            weight_vals = tl.load(weight_ptr + (off % N), mask=mask, other=0.0)
            bias_vals = tl.load(bias_ptr + (off % N), mask=mask, other=0.0)
            
            out_vals = norm_vals * weight_vals + bias_vals
            
            tl.store(out_ptr + off, out_vals, mask=mask)

def triton_layer_norm(x, weight, bias, eps=1e-5):
    """
    Triton implementation of LayerNorm
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA"
    assert x.dtype == torch.float32, "Only FP32 supported"
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Get dimensions
    M, N = x.shape
    out = torch.empty_like(x)
    mean = torch.empty(M, dtype=torch.float32, device=x.device)
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 1024
    GROUP_SIZE_M = 8
    
    # Determine grid
    grid = lambda meta: (tl.cdiv(M, GROUP_SIZE_M),)
    
    # Launch kernel
    layer_norm_kernel[grid](
        x, weight, bias, out, mean, rstd, 
        N, M, 
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE_M=GROUP_SIZE_M,
        EPS=eps
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for Layer Normalization.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer with Triton optimization.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return triton_layer_norm(x, self.weight, self.bias, self.eps)