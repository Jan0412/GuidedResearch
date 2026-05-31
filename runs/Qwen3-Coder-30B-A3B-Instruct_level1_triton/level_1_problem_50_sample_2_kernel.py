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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get the block ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(output_height, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(output_width, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Compute the output location
    output_row_start = pid_m * BLOCK_SIZE_M
    output_col_start = pid_n * BLOCK_SIZE_N
    
    # Loop over the kernel
    for k in range(0, tl.cdiv(in_channels * kernel_h * kernel_w, BLOCK_SIZE_K)):
        # Compute input indices
        input_row_start = output_row_start * stride_h - padding_h
        input_col_start = output_col_start * stride_w - padding_w
        
        # Load weights
        weight_offsets = tl.arange(0, BLOCK_SIZE_K)
        weight_mask = weight_offsets < (in_channels * kernel_h * kernel_w)
        
        # Load input tiles
        input_offsets = (
            tl.arange(0, BLOCK_SIZE_M)[:, None] * input_width +
            tl.arange(0, BLOCK_SIZE_N)[None, :]
        )
        input_mask = (
            (input_offsets >= 0) &
            (input_offsets < input_height * input_width)
        )
        
        # Load input data
        input_data = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
        
        # Load weights
        weight_data = tl.load(weight_ptr + k * BLOCK_SIZE_K + weight_offsets, mask=weight_mask, other=0.0)
        
        # Compute dot product
        acc += tl.dot(input_data, weight_data)
    
    # Add bias
    bias_offsets = tl.arange(0, BLOCK_SIZE_M) * output_width + tl.arange(0, BLOCK_SIZE_N)[None, :]
    bias_mask = (bias_offsets < output_height * output_width)
    bias_data = tl.load(bias_ptr + bias_offsets, mask=bias_mask, other=0.0)
    
    # Store result
    output_offsets = output_row_start * output_width + output_col_start
    tl.store(output_ptr + output_offsets, acc, mask=bias_mask)

def triton_conv2d(input_tensor, weight, bias, stride=(1, 1), padding=(0, 0)):
    """
    Triton implementation of 2D convolution
    """
    # Ensure tensors are on GPU
    if not input_tensor.is_cuda:
        input_tensor = input_tensor.cuda()
    if not weight.is_cuda:
        weight = weight.cuda()
    if bias is not None and not bias.is_cuda:
        bias = bias.cuda()
    
    # Get dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - kernel_h) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - kernel_w) // stride[1] + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, dtype=torch.float32, device='cuda')
    
    # Configure kernel parameters
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid
    grid = lambda meta: (
        triton.cdiv(output_height, meta["BLOCK_SIZE_M"]) *
        triton.cdiv(output_width, meta["BLOCK_SIZE_N"]),
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        input_height,
        input_width,
        output_height,
        output_width,
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        
    def forward(self, x):
        # Replace standard conv2d with our Triton implementation
        weight = self.conv1.weight.data
        bias = self.conv1.bias.data if self.conv1.bias is not None else None
        
        # Convert to float32 if needed
        if x.dtype != torch.float32:
            x = x.float()
            
        # Apply Triton-based convolution
        x = triton_conv2d(x, weight, bias, 
                         stride=self.conv1.stride, 
                         padding=self.conv1.padding)
        return x