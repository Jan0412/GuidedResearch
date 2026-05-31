import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv3d_kernel(
    input_ptr,     # Input tensor pointer
    weight_ptr,    # Weight tensor pointer  
    output_ptr,    # Output tensor pointer
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
):
    # Get thread indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Compute output coordinates
    output_d = pid_m * BLOCK_SIZE_M
    output_w = pid_n * BLOCK_SIZE_N
    output_h = pid_k * BLOCK_SIZE_K
    
    # Loop over input channels and kernel dimensions
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
    
    for c in range(0, in_channels):
        for kd in range(0, kernel_depth):
            for kw in range(0, kernel_width):
                for kh in range(0, kernel_height):
                    # Calculate input coordinates with padding and dilation
                    input_d = output_d * stride_d - padding_d + kd * dilation_d
                    input_w = output_w * stride_w - padding_w + kw * dilation_w
                    input_h = output_h * stride_h - padding_h + kh * dilation_h
                    
                    # Check bounds
                    if (input_d >= 0 and input_d < input_depth and 
                        input_w >= 0 and input_w < input_width and 
                        input_h >= 0 and input_h < input_height):
                        
                        # Load input value
                        input_val = tl.load(input_ptr + 
                                          (c * input_depth * input_width * input_height +
                                           input_d * input_width * input_height +
                                           input_w * input_height +
                                           input_h))
                        
                        # Load weight value
                        weight_val = tl.load(weight_ptr + 
                                           (c * out_channels * kernel_depth * kernel_width * kernel_height +
                                            output_d * out_channels * kernel_depth * kernel_width * kernel_height +
                                            output_w * out_channels * kernel_depth * kernel_width * kernel_height +
                                            output_h * out_channels * kernel_depth * kernel_width * kernel_height +
                                            kd * out_channels * kernel_width * kernel_height +
                                            kw * out_channels * kernel_height +
                                            kh * out_channels))
                        
                        acc += input_val * weight_val
    
    # Write output
    tl.store(output_ptr + 
             (output_d * out_channels * output_width * output_height +
              output_w * out_channels * output_height +
              output_h * out_channels),
             acc)

def triton_conv3d(input_tensor, weight, bias=None, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1)):
    """
    Triton implementation of 3D convolution
    """
    # Ensure inputs are contiguous and on GPU
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    batch_size, in_channels, input_depth, input_width, input_height = input_tensor.shape
    out_channels, _, kernel_depth, kernel_width, kernel_height = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth + 2 * padding[0] - (dilation[0] * (kernel_depth - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    output_height = (input_height + 2 * padding[2] - (dilation[2] * (kernel_height - 1) + 1)) // stride[2] + 1
    
    # Create output tensor
    output = torch.zeros((batch_size, out_channels, output_depth, output_width, output_height), 
                         device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE_M = 8
    BLOCK_SIZE_N = 8
    BLOCK_SIZE_K = 8
    
    # Grid configuration
    grid = (
        triton.cdiv(output_depth, BLOCK_SIZE_M),
        triton.cdiv(output_width, BLOCK_SIZE_N),
        triton.cdiv(output_height, BLOCK_SIZE_K)
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
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
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
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, width, height).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, width_out, height_out).
        """
        return triton_conv3d(
            x, 
            self.weight, 
            self.bias,
            stride=(self.stride, self.stride, self.stride),
            padding=(self.padding, self.padding, self.padding),
            dilation=(self.dilation, self.dilation, self.dilation)
        )

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
depth = 64
width = 64
height = 64

def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, width, height)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization