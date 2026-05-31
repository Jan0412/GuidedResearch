import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    input_height,
    input_width,
    output_height,
    output_width,
    in_channels,
    out_channels,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    batch_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(output_height, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(output_width, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Initialize pointers for output
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize output accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over the kernel dimensions
    for c in range(0, in_channels):
        # Compute input offsets
        input_offset = c * input_height * input_width
        
        # Compute weight offsets
        weight_offset = c * out_channels * kernel_h * kernel_w
        
        # Load input tile
        input_tile = tl.load(input_ptr + input_offset + 
                            (offs_am[:, None] * input_width + offs_bn[None, :]) * stride_h * stride_w + 
                            (tl.arange(0, kernel_h)[:, None] * input_width + tl.arange(0, kernel_w)[None, :]))
        
        # Load weight tile
        weight_tile = tl.load(weight_ptr + weight_offset + 
                             (offs_am[:, None] * out_channels + offs_bn[None, :]) * kernel_h * kernel_w)
        
        # Perform matrix multiplication
        acc += tl.dot(input_tile, weight_tile)
    
    # Apply bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_am)
        acc += bias[:, None]
    
    # Store output
    output_offset = (offs_am[:, None] * output_width + offs_bn[None, :])
    tl.store(output_ptr + output_offset, acc)

class TritonConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(TritonConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
    def forward(self, x):
        # Convert to contiguous tensors
        x = x.contiguous()
        weight = self.weight.contiguous()
        bias = self.bias.contiguous()
        
        # Get dimensions
        batch_size, _, input_height, input_width = x.shape
        _, _, kernel_h, kernel_w = weight.shape
        stride_h, stride_w = self.stride, self.stride
        padding_h, padding_w = self.padding, self.padding
        
        # Calculate output dimensions
        output_height = (input_height + 2 * padding_h - kernel_h) // stride_h + 1
        output_width = (input_width + 2 * padding_w - kernel_w) // stride_w + 1
        
        # Create output tensor
        output = torch.empty(batch_size, self.out_channels, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Define kernel parameters
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16
        BLOCK_SIZE_K = 32
        GROUP_SIZE_M = 8
        
        # Determine grid size
        grid = lambda META: (
            triton.cdiv(output_height, META['BLOCK_SIZE_M']) * triton.cdiv(output_width, META['BLOCK_SIZE_N']),
        )
        
        # Launch kernel
        conv2d_kernel[grid](
            x, weight, output, bias,
            input_height, input_width, output_height, output_width,
            self.in_channels, self.out_channels, kernel_h, kernel_w,
            stride_h, stride_w, padding_h, padding_w, batch_size,
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M
        )
        
        return output

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = TritonConv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
    
    def forward(self, x):
        x = self.conv1(x)
        return x