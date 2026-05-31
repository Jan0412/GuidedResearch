import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
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
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Grid shape calculation
    grid_size = (output_depth * output_width * output_height + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Each program processes one output element
    if pid >= grid_size:
        return
        
    # Calculate output position
    output_idx = pid
    output_d = output_idx // (output_width * output_height)
    remaining = output_idx % (output_width * output_height)
    output_w = remaining // output_height
    output_h = remaining % output_height
    
    # Calculate corresponding input positions
    input_d = output_d - padding_d
    input_w = output_w - padding_w
    input_h = output_h - padding_h
    
    # Check bounds
    if input_d < 0 or input_d >= input_depth or input_w < 0 or input_w >= input_width or input_h < 0 or input_h >= input_height:
        return
    
    # Loop over output channels and input channels
    for oc in range(out_channels):
        acc = 0.0
        for ic in range(in_channels):
            for kd in range(kernel_depth):
                for kw in range(kernel_width):
                    for kh in range(kernel_height):
                        # Calculate input indices
                        id = input_d + kd * stride_d
                        iw = input_w + kw * stride_w
                        ih = input_h + kh * stride_h
                        
                        # Check bounds
                        if id >= 0 and id < input_depth and iw >= 0 and iw < input_width and ih >= 0 and ih < input_height:
                            # Get input value
                            input_val = tl.load(input_ptr + 
                                              (0 * input_depth * input_width * input_height +
                                               ic * input_depth * input_width * input_height +
                                               id * input_width * input_height +
                                               iw * input_height +
                                               ih))
                            
                            # Get weight value
                            weight_val = tl.load(weight_ptr + 
                                               (oc * in_channels * kernel_depth * kernel_width * kernel_height +
                                                ic * kernel_depth * kernel_width * kernel_height +
                                                kd * kernel_width * kernel_height +
                                                kw * kernel_height +
                                                kh))
                            
                            acc += input_val * weight_val
        
        # Store result
        tl.store(output_ptr + 
                (0 * output_depth * output_width * output_height * out_channels +
                 oc * output_depth * output_width * output_height +
                 output_d * output_width * output_height +
                 output_w * output_height +
                 output_h), acc)

def triton_conv_transpose3d(input_tensor, weight, bias=None, stride=(1,1,1), padding=(0,0,0)):
    """
    Triton implementation of 3D transposed convolution
    """
    # Ensure inputs are on GPU
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    batch_size, in_channels, input_depth, input_width, input_height = input_tensor.shape
    out_channels, _, kernel_depth, kernel_width, kernel_height = weight.shape
    
    # Calculate output dimensions
    stride_d, stride_w, stride_h = stride
    pad_d, pad_w, pad_h = padding
    
    output_depth = (input_depth - 1) * stride_d - 2 * pad_d + kernel_depth
    output_width = (input_width - 1) * stride_w - 2 * pad_w + kernel_width
    output_height = (input_height - 1) * stride_h - 2 * pad_h + kernel_height
    
    # Initialize output tensor
    output = torch.zeros((batch_size, out_channels, output_depth, output_width, output_height), 
                         dtype=torch.float32, device=input_tensor.device)
    
    # Set up kernel launch parameters
    BLOCK_SIZE = 1024
    grid_size = (output_depth * output_width * output_height + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    conv_transpose3d_kernel[grid_size](
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
        stride_d,
        stride_w,
        stride_h,
        pad_d,
        pad_w,
        pad_h,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE_M=8
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution with a square input and an asymmetric kernel.
    Optimized using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Ensure proper initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, width, height).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, width_out, height_out).
        """
        # Call our Triton-based convolution
        output = triton_conv_transpose3d(x, self.weight, self.bias, self.stride, self.padding)
        return output