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
    kernel_h,
    kernel_w,
    out_height,
    out_width,
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
    out_h_id = tl.program_id(2)
    out_w_id = tl.program_id(3)
    
    # Calculate output position
    out_h_start = out_h_id * HEIGHT_PER_BLOCK
    out_w_start = out_w_id * WIDTH_PER_BLOCK
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(HEIGHT_PER_BLOCK + 2 * padding_h, WIDTH_PER_BLOCK + 2 * padding_w))
    
    # Initialize accumulator
    acc = tl.zeros((HEIGHT_PER_BLOCK, WIDTH_PER_BLOCK), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Calculate input positions
            ih_start = out_h_start * stride_h + kh * dilation_h - padding_h
            iw_start = out_w_start * stride_w + kw * dilation_w - padding_w
            
            # Load input tile with padding
            for i in range(HEIGHT_PER_BLOCK):
                for j in range(WIDTH_PER_BLOCK):
                    ih = ih_start + i
                    iw = iw_start + j
                    
                    # Check bounds
                    if ih >= 0 and ih < height and iw >= 0 and iw < width:
                        input_val = tl.load(input_ptr + 
                                          batch_id * (in_channels * height * width) +
                                          channel_id * (height * width) +
                                          ih * width + iw)
                    else:
                        input_val = 0.0
                    
                    # Store in shared memory
                    shared_input[i + padding_h, j + padding_w] = input_val
            
            # Load kernel weight
            weight_val = tl.load(weight_ptr + channel_id * kernel_h * kernel_w + kh * kernel_w + kw)
            
            # Compute convolution
            for i in range(HEIGHT_PER_BLOCK):
                for j in range(WIDTH_PER_BLOCK):
                    if (out_h_start + i < out_height and out_w_start + j < out_width):
                        acc[i, j] += shared_input[i + padding_h, j + padding_w] * weight_val
    
    # Write output
    for i in range(HEIGHT_PER_BLOCK):
        for j in range(WIDTH_PER_BLOCK):
            if (out_h_start + i < out_height and out_w_start + j < out_width):
                output_idx = batch_id * (in_channels * out_height * out_width) + \
                           channel_id * (out_height * out_width) + \
                           (out_h_start + i) * out_width + (out_w_start + j)
                tl.store(output_ptr + output_idx, acc[i, j])

def triton_depthwise_conv2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """
    Triton implementation of depthwise 2D convolution
    """
    batch_size, in_channels, height, width = input_tensor.shape
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]
    out_height = (height + 2 * padding[0] - (dilation[0] * (kernel_h - 1) + 1)) // stride[0] + 1
    out_width = (width + 2 * padding[1] - (dilation[1] * (kernel_w - 1) + 1)) // stride[1] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, in_channels, out_height, out_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 1
    HEIGHT_PER_BLOCK = 8
    WIDTH_PER_BLOCK = 8
    
    # Grid configuration
    grid = (
        batch_size,
        in_channels,
        (out_height + HEIGHT_PER_BLOCK - 1) // HEIGHT_PER_BLOCK,
        (out_width + WIDTH_PER_BLOCK - 1) // WIDTH_PER_BLOCK
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
        kernel_h,
        kernel_w,
        out_height,
        out_width,
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
    Optimized using custom Triton kernels.
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
        
        # Initialize weights and biases
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

# Note: The actual implementation above has limitations due to Triton's shared memory constraints
# For a production version, more sophisticated tiling and memory management would be required
# This serves as a conceptual demonstration of how such an optimization might be structured