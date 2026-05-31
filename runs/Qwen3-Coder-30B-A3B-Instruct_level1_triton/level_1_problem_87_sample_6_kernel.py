import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv1d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    USE_BIAS: tl.constexpr
):
    # Get the block ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(height, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(out_channels, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Compute offsets
    offset_m = pid_m * BLOCK_SIZE_M
    offset_n = pid_n * BLOCK_SIZE_N
    offset_k = 0
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, in_channels, BLOCK_SIZE_K):
        # Load input and weight tiles
        input_tile = tl.load(input_ptr + 
                            offset_m * width * in_channels + 
                            k + 
                            tl.arange(0, BLOCK_SIZE_M)[:, None] * width * in_channels +
                            tl.arange(0, BLOCK_SIZE_K)[None, :] * in_channels,
                            mask=(offset_m + tl.arange(0, BLOCK_SIZE_M)[:, None] < height) &
                                 (k + tl.arange(0, BLOCK_SIZE_K)[None, :] < in_channels),
                            other=0.0)
        
        weight_tile = tl.load(weight_ptr + 
                             offset_n * in_channels + 
                             k + 
                             tl.arange(0, BLOCK_SIZE_N)[:, None] * in_channels +
                             tl.arange(0, BLOCK_SIZE_K)[None, :],
                             mask=(offset_n + tl.arange(0, BLOCK_SIZE_N)[:, None] < out_channels) &
                                  (k + tl.arange(0, BLOCK_SIZE_K)[None, :] < in_channels),
                             other=0.0)
        
        # Matrix multiplication
        acc += tl.dot(input_tile, weight_tile)
    
    # Apply bias if needed
    if USE_BIAS:
        bias_tile = tl.load(bias_ptr + offset_n + tl.arange(0, BLOCK_SIZE_N),
                           mask=offset_n + tl.arange(0, BLOCK_SIZE_N) < out_channels,
                           other=0.0)
        acc += bias_tile[None, :]
    
    # Write output
    output_tile = acc.to(tl.float32)
    tl.store(output_ptr + 
             offset_m * width * out_channels + 
             offset_n + 
             tl.arange(0, BLOCK_SIZE_M)[:, None] * width * out_channels +
             tl.arange(0, BLOCK_SIZE_N)[None, :],
             output_tile,
             mask=(offset_m + tl.arange(0, BLOCK_SIZE_M)[:, None] < height) &
                  (offset_n + tl.arange(0, BLOCK_SIZE_N)[None, :] < out_channels))

def triton_conv1d(input_tensor, weight, bias=None):
    """
    Triton implementation of 1x1 convolution using matrix multiplication
    """
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight.shape[0]
    
    # Reshape input to (batch_size * height, width * in_channels)
    input_reshaped = input_tensor.permute(0, 2, 3, 1).contiguous().view(batch_size * height, width * in_channels)
    
    # Reshape weight to (out_channels, in_channels)
    weight_reshaped = weight.view(out_channels, in_channels)
    
    # Allocate output
    output = torch.empty(batch_size, out_channels, height, width, device=input_tensor.device, dtype=torch.float32)
    
    # Launch kernel
    grid = lambda meta: (
        triton.cdiv(height, meta["BLOCK_SIZE_M"]) * triton.cdiv(out_channels, meta["BLOCK_SIZE_N"]),
    )
    
    # Use fixed parameters for now - these could be tuned based on input sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    USE_BIAS = bias is not None
    
    conv1d_kernel[grid](
        input_reshaped,
        weight_reshaped,
        output.view(batch_size * height, width * out_channels),
        bias,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        1,  # stride_h
        1,  # stride_w
        0,  # padding_h
        0,  # padding_w
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
        USE_BIAS
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use Triton kernel for convolution
        return triton_conv1d(x, self.weight, self.bias)