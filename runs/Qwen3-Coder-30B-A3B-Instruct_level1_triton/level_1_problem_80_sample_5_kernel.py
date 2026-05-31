import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get the block index
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(output_height, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(output_width, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Compute output coordinates
    out_y = pid_m * BLOCK_SIZE_M
    out_x = pid_n * BLOCK_SIZE_N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over the kernel
    for k in range(0, tl.cdiv(in_channels * kernel_height * kernel_width, BLOCK_SIZE_K)):
        # Compute indices for input and weight
        c = k * BLOCK_SIZE_K // (kernel_height * kernel_width)
        kh = (k * BLOCK_SIZE_K % (kernel_height * kernel_width)) // kernel_width
        kw = (k * BLOCK_SIZE_K % (kernel_height * kernel_width)) % kernel_width
        
        # Check bounds for channel
        if c >= in_channels:
            break
            
        # Check bounds for kernel
        if kh >= kernel_height or kw >= kernel_width:
            break
            
        # Compute input indices
        input_y_start = out_y * stride_h - pad_h + kh * dilation_h
        input_x_start = out_x * stride_w - pad_w + kw * dilation_w
        
        # Load input data
        input_data = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for i in range(BLOCK_SIZE_M):
            for j in range(BLOCK_SIZE_N):
                y = input_y_start + i
                x = input_x_start + j
                
                if 0 <= y < input_height and 0 <= x < input_width:
                    input_data[i, j] = tl.load(input_ptr + 
                        (0 * input_height * input_width + c * input_height * input_width + 
                         y * input_width + x))
        
        # Load weight data
        weight_data = tl.load(weight_ptr + 
            (0 * in_channels * kernel_height * kernel_width + 
             c * kernel_height * kernel_width + 
             kh * kernel_width + kw))
        
        # Accumulate
        acc += input_data * weight_data
    
    # Write output
    for i in range(BLOCK_SIZE_M):
        for j in range(BLOCK_SIZE_N):
            if out_y + i < output_height and out_x + j < output_width:
                output_idx = (0 * output_height * output_width + 
                             0 * output_height * output_width + 
                             (out_y + i) * output_width + (out_x + j))
                tl.store(output_ptr + output_idx, acc[i, j])

def triton_conv2d(input_tensor, weight, bias, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """
    Custom Triton implementation of 2D convolution
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    
    # Allocate output tensor
    output = torch.zeros(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Define kernel parameters
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Grid size calculation
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
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with square input and asymmetric kernel, with dilation and padding.
    Optimized using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Use our custom Triton convolution implementation
        return triton_conv2d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )

# Test code
batch_size = 8
in_channels = 32
out_channels = 64
kernel_size = (5, 9)
width = 512
height = 512
stride = 1
padding = (2, 4)
dilation = (2, 3)

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]