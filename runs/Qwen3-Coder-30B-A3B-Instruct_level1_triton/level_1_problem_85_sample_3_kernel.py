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
    height_in,
    width_in,
    height_out,
    width_out,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    HEIGHT_PER_BLOCK: tl.constexpr,
    WIDTH_PER_BLOCK: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    height_id = tl.program_id(2)
    width_id = tl.program_id(3)
    
    # Calculate global indices
    out_h_start = height_id * HEIGHT_PER_BLOCK
    out_w_start = width_id * WIDTH_PER_BLOCK
    
    # Shared memory for input tile
    input_tile = tl.shared_memory(dtype=tl.float32, shape=(HEIGHT_PER_BLOCK + 2 * padding_h, WIDTH_PER_BLOCK + 2 * padding_w))
    
    # Load weights for this channel
    weight = tl.load(weight_ptr + channel_id * kernel_h * kernel_w)
    
    # Process multiple output positions per thread
    for out_h in range(HEIGHT_PER_BLOCK):
        for out_w in range(WIDTH_PER_BLOCK):
            if out_h_start + out_h < height_out and out_w_start + out_w < width_out:
                # Calculate input position
                in_h_start = out_h_start + out_h
                in_w_start = out_w_start + out_w
                
                # Initialize accumulator
                acc = 0.0
                
                # Perform convolution
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        # Calculate input coordinates with dilation and padding
                        ih = in_h_start * stride_h + kh * dilation_h - padding_h
                        iw = in_w_start * stride_w + kw * dilation_w - padding_w
                        
                        # Check bounds
                        if 0 <= ih < height_in and 0 <= iw < width_in:
                            # Load input value
                            input_val = tl.load(input_ptr + 
                                              batch_id * (in_channels * height_in * width_in) +
                                              channel_id * (height_in * width_in) +
                                              ih * width_in + iw)
                            # Load weight
                            weight_val = tl.load(weight_ptr + 
                                               channel_id * (kernel_h * kernel_w) +
                                               kh * kernel_w + kw)
                            acc += input_val * weight_val
                        else:
                            # Padding is zero
                            acc += 0.0
                
                # Store output
                tl.store(output_ptr + 
                        batch_id * (in_channels * height_out * width_out) +
                        channel_id * (height_out * width_out) +
                        out_h_start * width_out + out_w_start, 
                        acc)

def triton_depthwise_conv2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """
    Triton implementation of depthwise convolution 2D
    """
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    
    batch_size, in_channels, height_in, width_in = input_tensor.shape
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]
    height_out = (height_in + 2 * padding[0] - (dilation[0] * (kernel_h - 1) + 1)) // stride[0] + 1
    width_out = (width_in + 2 * padding[1] - (dilation[1] * (kernel_w - 1) + 1)) // stride[1] + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, in_channels, height_out, width_out, dtype=torch.float32, device=input_tensor.device)
    
    # Define block sizes
    BLOCK_SIZE = 1024
    CHANNELS_PER_BLOCK = 1
    HEIGHT_PER_BLOCK = 8
    WIDTH_PER_BLOCK = 8
    
    # Calculate grid dimensions
    grid = (
        batch_size,           # batch dimension
        in_channels,          # channel dimension
        (height_out + HEIGHT_PER_BLOCK - 1) // HEIGHT_PER_BLOCK,   # height blocks
        (width_out + WIDTH_PER_BLOCK - 1) // WIDTH_PER_BLOCK      # width blocks
    )
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        height_in,
        width_in,
        height_out,
        width_out,
        kernel_h,
        kernel_w,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        HEIGHT_PER_BLOCK=HEIGHT_PER_BLOCK,
        WIDTH_PER_BLOCK=WIDTH_PER_BLOCK
    )
    
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and asymmetric kernel.
    Optimized using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups
        self.bias = bias
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size_h, kernel_size_w))
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
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_depthwise_conv2d(
            x, 
            self.weight, 
            self.bias,
            stride=(self.stride_h, self.stride_w),
            padding=(self.padding_h, self.padding_w),
            dilation=(self.dilation_h, self.dilation_w)
        )