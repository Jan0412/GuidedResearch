import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    height,
    width,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    output_height,
    output_width,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program IDs
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    output_h_id = tl.program_id(2)
    
    # Calculate output dimensions
    output_w_id = tl.program_id(3)
    
    # Shared memory for input tile
    tile_size = 32
    input_tile = tl.shared_tensor(tl.arange(0, tile_size * tile_size), dtype=tl.float32)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel elements
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            # Calculate input position
            ih = output_h_id * stride_h + kh * dilation_h - padding_h
            iw = output_w_id * stride_w + kw * dilation_w - padding_w
            
            # Check bounds
            if ih >= 0 and ih < height and iw >= 0 and iw < width:
                # Load input value
                input_val = tl.load(input_ptr + 
                                  batch_id * in_channels * height * width +
                                  channel_id * height * width +
                                  ih * width + iw)
                
                # Load weight value
                weight_val = tl.load(weight_ptr + 
                                   channel_id * kernel_height * kernel_width +
                                   kh * kernel_width + kw)
                
                # Accumulate
                acc += input_val * weight_val
    
    # Store result
    if output_h_id < output_height and output_w_id < output_width:
        tl.store(output_ptr + 
                batch_id * in_channels * output_height * output_width +
                channel_id * output_height * output_width +
                output_h_id * output_width + output_w_id,
                acc)

def triton_depthwise_conv2d(input_tensor, weight, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """
    Triton implementation of depthwise 2D convolution
    """
    batch_size, in_channels, height, width = input_tensor.shape
    kernel_height, kernel_width = weight.shape[2], weight.shape[3]
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    
    # Calculate output dimensions
    output_height = (height + 2 * padding_h - (dilation_h * (kernel_height - 1) + 1)) // stride_h + 1
    output_width = (width + 2 * padding_w - (dilation_w * (kernel_width - 1) + 1)) // stride_w + 1
    
    # Create output tensor
    output = torch.empty(batch_size, in_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define grid
    grid = (
        batch_size,
        in_channels,
        output_height,
        output_width
    )
    
    # Launch kernel
    BLOCK_SIZE = 1024
    depthwise_conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        height,
        width,
        kernel_height,
        kernel_width,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        output_height,
        output_width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weight
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, 1))
        
        # Initialize bias if needed
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter('bias_param', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        # Use Triton kernel for depthwise convolution
        output = triton_depthwise_conv2d(
            x, 
            self.weight, 
            stride=(self.stride, 1),
            padding=(self.padding, 0),
            dilation=(self.dilation, 1)
        )
        
        # Add bias if present
        if self.bias_param is not None:
            output = output + self.bias_param.view(1, -1, 1, 1)
            
        return output