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
    input_stride_n,
    input_stride_c,
    input_stride_h,
    input_stride_w,
    weight_stride_o,
    weight_stride_i,
    weight_stride_h,
    weight_stride_w,
    output_stride_n,
    output_stride_c,
    output_stride_h,
    output_stride_w,
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_height,
    kernel_width,
    padding_h,
    padding_w,
    stride_h,
    stride_w,
    dilation_h,
    dilation_w,
    has_bias,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program IDs
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate output position
    output_h = pid_h
    output_w = pid_w
    
    # Early exit if out of bounds
    if output_h >= output_height or output_w >= output_width:
        return
        
    # Loop over output channels
    for oc in range(pid_c, out_channels, BLOCK_SIZE):
        # Initialize accumulator
        acc = tl.zeros((1,), dtype=tl.float32)
        
        # Loop over input channels
        for ic in range(in_channels):
            # Loop over kernel height
            for kh in range(kernel_height):
                # Loop over kernel width
                for kw in range(kernel_width):
                    # Calculate input position
                    ih = output_h * stride_h - padding_h + kh * dilation_h
                    iw = output_w * stride_w - padding_w + kw * dilation_w
                    
                    # Check bounds
                    if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                        # Load input value
                        input_val = tl.load(input_ptr + 
                                          pid_n * input_stride_n +
                                          ic * input_stride_c +
                                          ih * input_stride_h +
                                          iw * input_stride_w,
                                          mask=(ih < input_height) & (iw < input_width))
                        
                        # Load weight value
                        weight_val = tl.load(weight_ptr + 
                                           oc * weight_stride_o +
                                           ic * weight_stride_i +
                                           kh * weight_stride_h +
                                           kw * weight_stride_w)
                        
                        # Accumulate
                        acc += input_val * weight_val
        
        # Apply bias if present
        if has_bias:
            bias_val = tl.load(bias_ptr + oc)
            acc += bias_val
            
        # Store output
        tl.store(output_ptr + 
                pid_n * output_stride_n +
                oc * output_stride_c +
                output_h * output_stride_h +
                output_w * output_stride_w,
                acc)

def triton_conv2d(input_tensor, weight, bias, stride, padding, dilation, groups):
    """
    Custom Triton implementation of 2D convolution
    """
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block size
    BLOCK_SIZE = 32
    
    # Grid configuration
    grid = (
        batch_size,
        min(BLOCK_SIZE, out_channels),
        output_height,
        output_width
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        input_tensor.stride(0),
        input_tensor.stride(1),
        input_tensor.stride(2),
        input_tensor.stride(3),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        weight.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        padding[0],
        padding[1],
        stride[0],
        stride[1],
        dilation[0],
        dilation[1],
        bias is not None,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with a square input and square kernel.

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
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)