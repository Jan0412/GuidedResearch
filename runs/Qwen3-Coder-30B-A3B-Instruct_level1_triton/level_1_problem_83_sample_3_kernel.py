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
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_h_idx = tl.program_id(2)
    
    # Calculate output dimensions
    output_w_idx = tl.program_id(3)
    
    # Shared memory for input tile
    input_tile = tl.shared_tile(input_ptr, (BLOCK_SIZE, BLOCK_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Calculate output position
    out_h = output_h_idx * stride_h
    out_w = output_w_idx * stride_w
    
    # Iterate over kernel
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            # Calculate input positions with dilation and padding
            ih = out_h - padding_h + kh * dilation_h
            iw = out_w - padding_w + kw * dilation_w
            
            # Check bounds
            if ih >= 0 and ih < height and iw >= 0 and iw < width:
                # Load input value
                input_val = tl.load(input_ptr + 
                                  batch_idx * (in_channels * height * width) +
                                  channel_idx * (height * width) +
                                  ih * width + iw)
                
                # Load weight value
                weight_val = tl.load(weight_ptr + 
                                   channel_idx * (kernel_height * kernel_width) +
                                   kh * kernel_width + kw)
                
                # Accumulate
                acc += input_val * weight_val
    
    # Store result
    if output_h_idx < output_height and output_w_idx < output_width:
        tl.store(output_ptr + 
                batch_idx * (in_channels * output_height * output_width) +
                channel_idx * (output_height * output_width) +
                output_h_idx * output_width + output_w_idx,
                acc[0])

def triton_depthwise_conv2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """
    Triton implementation of depthwise convolution
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
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Define grid
    grid = (
        batch_size,
        in_channels,
        (output_height + stride_h - 1) // stride_h,
        (output_width + stride_w - 1) // stride_w
    )
    
    # Launch kernel
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
        BLOCK_SIZE=32
    )
    
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with a square input and an asymmetric kernel.
    Optimized using Triton kernels.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using Triton kernel.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        return triton_depthwise_conv2d(
            x, 
            self.weight, 
            self.bias, 
            stride=(self.stride, 1),
            padding=(self.padding, 0),
            dilation=(self.dilation, 1)
        )

# Note: The actual Triton kernel implementation above has some limitations in terms of 
# memory access patterns and shared memory usage compared to a fully optimized version.
# For production use, a more sophisticated kernel would be needed with better tiling strategies
# and memory coalescing patterns.