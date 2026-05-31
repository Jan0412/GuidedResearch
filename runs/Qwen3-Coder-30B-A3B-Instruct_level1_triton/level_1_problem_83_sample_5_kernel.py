import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

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
    CHANNELS_PER_BLOCK: tl.constexpr,
    HEIGHT_PER_BLOCK: tl.constexpr,
    WIDTH_PER_BLOCK: tl.constexpr
):
    # Get block indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    out_w_idx = tl.program_id(3)
    
    # Calculate output dimensions
    output_h = output_height
    output_w = output_width
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(HEIGHT_PER_BLOCK + 2 * padding_h, WIDTH_PER_BLOCK + 2 * padding_w))
    
    # Calculate global indices
    global_out_h = out_h_idx * HEIGHT_PER_BLOCK
    global_out_w = out_w_idx * WIDTH_PER_BLOCK
    
    # Initialize accumulator
    acc = tl.zeros((HEIGHT_PER_BLOCK, WIDTH_PER_BLOCK), dtype=tl.float32)
    
    # Loop over kernel height and width
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            # Calculate input positions
            input_h_start = global_out_h * stride_h + kh * dilation_h - padding_h
            input_w_start = global_out_w * stride_w + kw * dilation_w - padding_w
            
            # Load input tile
            for ih in range(HEIGHT_PER_BLOCK):
                for iw in range(WIDTH_PER_BLOCK):
                    h = input_h_start + ih
                    w = input_w_start + iw
                    
                    if h >= 0 and h < height and w >= 0 and w < width:
                        val = tl.load(input_ptr + 
                                    batch_idx * (in_channels * height * width) +
                                    channel_idx * (height * width) +
                                    h * width + w)
                    else:
                        val = 0.0
                        
                    shared_input[ih + padding_h, iw + padding_w] = val
            
            # Load weight
            weight_val = tl.load(weight_ptr + channel_idx * kernel_height * kernel_width + kh * kernel_width + kw)
            
            # Compute convolution
            for ih in range(HEIGHT_PER_BLOCK):
                for iw in range(WIDTH_PER_BLOCK):
                    acc[ih, iw] += shared_input[ih + padding_h, iw + padding_w] * weight_val
    
    # Write output
    for ih in range(HEIGHT_PER_BLOCK):
        for iw in range(WIDTH_PER_BLOCK):
            out_h = global_out_h + ih
            out_w = global_out_w + iw
            
            if out_h < output_h and out_w < output_w:
                tl.store(output_ptr + 
                        batch_idx * (in_channels * output_h * output_w) +
                        channel_idx * (output_h * output_w) +
                        out_h * output_w + out_w,
                        acc[ih, iw])

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
    
    # Define block sizes
    BLOCK_SIZE = 256
    CHANNELS_PER_BLOCK = 1
    HEIGHT_PER_BLOCK = 8
    WIDTH_PER_BLOCK = 8
    
    # Grid configuration
    grid = (
        batch_size,
        in_channels,
        (output_height + HEIGHT_PER_BLOCK - 1) // HEIGHT_PER_BLOCK,
        (output_width + WIDTH_PER_BLOCK - 1) // WIDTH_PER_BLOCK
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
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        HEIGHT_PER_BLOCK=HEIGHT_PER_BLOCK,
        WIDTH_PER_BLOCK=WIDTH_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
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
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter('bias_param', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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