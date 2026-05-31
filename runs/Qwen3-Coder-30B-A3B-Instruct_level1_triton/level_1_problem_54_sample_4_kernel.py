import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv3d_kernel(
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
    dilation_d,
    dilation_w,
    dilation_h,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    output_idx = tl.program_id(2)
    
    # Calculate output coordinates
    output_d = output_idx // (output_width * output_height)
    remaining = output_idx % (output_width * output_height)
    output_w = remaining // output_height
    output_h = remaining % output_height
    
    # Shared memory for input tile
    input_tile = tl.shared.tensor([KERNEL_DEPTH, KERNEL_WIDTH, KERNEL_HEIGHT], tl.float32)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels
    for c in range(0, in_channels, CHANNELS_PER_BLOCK):
        # Load weights for this channel
        weight = tl.load(weight_ptr + 
                        batch_id * out_channels * in_channels * kernel_depth * kernel_width * kernel_height +
                        channel_id * in_channels * kernel_depth * kernel_width * kernel_height +
                        c * kernel_depth * kernel_width * kernel_height +
                        tl.arange(0, kernel_depth)[:, None, None] * kernel_width * kernel_height +
                        tl.arange(0, kernel_width)[None, :, None] * kernel_height +
                        tl.arange(0, kernel_height)[None, None, :])
        
        # Load input tile
        input_d_start = output_d * stride_d - padding_d
        input_w_start = output_w * stride_w - padding_w
        input_h_start = output_h * stride_h - padding_h
        
        for kd in range(kernel_depth):
            for kw in range(kernel_width):
                for kh in range(kernel_height):
                    input_d = input_d_start + kd * dilation_d
                    input_w = input_w_start + kw * dilation_w
                    input_h = input_h_start + kh * dilation_h
                    
                    if (input_d >= 0 and input_d < input_depth and
                        input_w >= 0 and input_w < input_width and
                        input_h >= 0 and input_h < input_height):
                        input_val = tl.load(input_ptr + 
                                          batch_id * in_channels * input_depth * input_width * input_height +
                                          c * input_depth * input_width * input_height +
                                          input_d * input_width * input_height +
                                          input_w * input_height +
                                          input_h)
                        acc += input_val * weight[kd, kw, kh]
    
    # Store result
    if output_idx < output_depth * output_width * output_height:
        tl.store(output_ptr + 
                batch_id * out_channels * output_depth * output_width * output_height +
                channel_id * output_depth * output_width * output_height +
                output_d * output_width * output_height +
                output_w * output_height +
                output_h, acc)

def triton_conv3d(input_tensor, weight, bias=None, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1)):
    """
    Custom Triton implementation of 3D convolution.
    """
    batch_size, in_channels, input_depth, input_width, input_height = input_tensor.shape
    out_channels, _, kernel_depth, kernel_width, kernel_height = weight.shape
    
    stride_d, stride_w, stride_h = stride
    pad_d, pad_w, pad_h = padding
    dil_d, dil_w, dil_h = dilation
    
    # Calculate output dimensions
    output_depth = (input_depth + 2 * pad_d - (dil_d * (kernel_depth - 1) + 1)) // stride_d + 1
    output_width = (input_width + 2 * pad_w - (dil_w * (kernel_width - 1) + 1)) // stride_w + 1
    output_height = (input_height + 2 * pad_h - (dil_h * (kernel_height - 1) + 1)) // stride_h + 1
    
    # Create output tensor
    output = torch.zeros(batch_size, out_channels, output_depth, output_width, output_height, device=input_tensor.device, dtype=torch.float32)
    
    # Launch kernel
    grid = (
        batch_size,
        out_channels,
        output_depth * output_width * output_height
    )
    
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 4
    OUTPUT_ELEMENTS_PER_BLOCK = 1
    
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
        stride_d,
        stride_w,
        stride_h,
        pad_d,
        pad_w,
        pad_h,
        dil_d,
        dil_w,
        dil_h,
        BLOCK_SIZE,
        CHANNELS_PER_BLOCK,
        OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with square input and square kernel.
    Optimized with custom Triton kernels.
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
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Handle grouped convolutions
        if groups != 1:
            raise NotImplementedError("Grouped convolutions not supported in this implementation")
        
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

# Helper function for testing
def get_inputs():
    batch_size = 16
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    depth = 64
    width = 64
    height = 64
    x = torch.rand(batch_size, in_channels, depth, width, height, device='cuda', dtype=torch.float32)
    return [x]

def get_init_inputs():
    return [3, 64, 3]  # in_channels, out_channels, kernel_size