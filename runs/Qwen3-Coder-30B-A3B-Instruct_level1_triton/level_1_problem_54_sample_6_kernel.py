import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv3d_kernel(
    input_ptr,   # Input tensor pointer
    weight_ptr,  # Weight tensor pointer
    output_ptr,  # Output tensor pointer
    batch_size,
    in_channels,
    out_channels,
    input_depth,
    input_width,
    input_height,
    output_depth,
    output_width,
    output_height,
    kernel_depth,
    kernel_width,
    kernel_height,
    stride_d,
    stride_w,
    stride_h,
    padding_d,
    padding_w,
    padding_h,
    dilation_d,
    dilation_w,
    dilation_h,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get the thread index
    pid = tl.program_id(axis=0)
    
    # Number of output blocks
    num_blocks_m = tl.cdiv(output_depth * output_width * output_height, BLOCK_SIZE_M)
    num_blocks_n = tl.cdiv(out_channels, BLOCK_SIZE_N)
    
    # Grid size
    grid_size = num_blocks_m * num_blocks_n
    
    # Which block this thread is processing
    block_id_m = pid % num_blocks_m
    block_id_n = pid // num_blocks_m
    
    # Compute output indices
    output_idx_m = block_id_m * BLOCK_SIZE_M
    output_idx_n = block_id_n * BLOCK_SIZE_N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension (input channels and kernel elements)
    for k in range(0, tl.cdiv(in_channels * kernel_depth * kernel_width * kernel_height, BLOCK_SIZE_K)):
        # Compute offsets for input tensor
        input_offset = (
            (output_idx_m // (output_width * output_height)) * stride_d * input_width * input_height +
            ((output_idx_m % (output_width * output_height)) // output_width) * stride_w * input_height +
            ((output_idx_m % (output_width * output_height)) % output_width) * stride_h +
            k * BLOCK_SIZE_K
        )
        
        # Compute offsets for weight tensor
        weight_offset = k * BLOCK_SIZE_K
        
        # Compute offsets for output tensor
        output_offset = output_idx_m * out_channels + output_idx_n
        
        # Load input data
        input_data = tl.load(input_ptr + input_offset, mask=(k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < in_channels * kernel_depth * kernel_width * kernel_height))
        
        # Load weight data
        weight_data = tl.load(weight_ptr + weight_offset, mask=(k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < in_channels * kernel_depth * kernel_width * kernel_height))
        
        # Perform dot product
        acc += tl.sum(input_data[:, None] * weight_data[None, :], axis=0)
    
    # Store result
    tl.store(output_ptr + output_offset, acc, mask=(output_idx_n + tl.arange(0, BLOCK_SIZE_N) < out_channels))

def triton_conv3d(input_tensor, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1)):
    """
    Custom Triton implementation of 3D convolution
    """
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    
    # Get dimensions
    batch_size, in_channels, input_depth, input_width, input_height = input_tensor.shape
    out_channels, _, kernel_depth, kernel_width, kernel_height = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth + 2 * padding[0] - (dilation[0] * (kernel_depth - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    output_height = (input_height + 2 * padding[2] - (dilation[2] * (kernel_height - 1) + 1)) // stride[2] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_width, output_height, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid size
    grid = lambda meta: (
        triton.cdiv(output_depth * output_width * output_height, meta["BLOCK_SIZE_M"]) *
        triton.cdiv(out_channels, meta["BLOCK_SIZE_N"]),
    )
    
    # Launch kernel
    conv3d_kernel[grid](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_width,
        input_height,
        output_depth,
        output_width,
        output_height,
        kernel_depth,
        kernel_width,
        kernel_height,
        stride[0],
        stride[1],
        stride[2],
        padding[0],
        padding[1],
        padding[2],
        dilation[0],
        dilation[1],
        dilation[2],
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using Triton kernel.
        """
        return triton_conv3d(
            x, 
            self.weight, 
            self.bias,
            stride=(self.stride, self.stride, self.stride),
            padding=(self.padding, self.padding, self.padding),
            dilation=(self.dilation, self.dilation, self.dilation)
        )