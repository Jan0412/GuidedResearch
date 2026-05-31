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
    input_height,
    input_width,
    input_depth,
    output_height,
    output_width,
    output_depth,
    kernel_height,
    kernel_width,
    kernel_depth,
    stride_h,
    stride_w,
    stride_d,
    padding_h,
    padding_w,
    padding_d,
    dilation_h,
    dilation_w,
    dilation_d,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_CHANNELS_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    out_w_idx = tl.program_id(3)
    out_d_idx = tl.program_id(4)
    
    # Shared memory for input tile
    input_tile = tl.shared_tensor(tl.float32, (BLOCK_SIZE_H + 2*padding_h, BLOCK_SIZE_W + 2*padding_w, BLOCK_SIZE_D + 2*padding_d))
    
    # Initialize accumulator
    acc = tl.zeros((OUTPUT_CHANNELS_PER_BLOCK,), dtype=tl.float32)
    
    # Loop over input channels
    for c in range(0, in_channels, CHANNELS_PER_BLOCK):
        # Load weight tile (output_channels, input_channels, kernel_h, kernel_w, kernel_d)
        weight_tile = tl.load(weight_ptr + 
                             out_c_idx * in_channels * kernel_height * kernel_width * kernel_depth +
                             c * kernel_height * kernel_width * kernel_depth +
                             tl.arange(0, OUTPUT_CHANNELS_PER_BLOCK)[:, None, None, None, None] *
                             in_channels * kernel_height * kernel_width * kernel_depth +
                             tl.arange(0, kernel_height)[None, :, None, None, None] *
                             in_channels * kernel_width * kernel_depth +
                             tl.arange(0, kernel_width)[None, None, :, None, None] *
                             in_channels * kernel_depth +
                             tl.arange(0, kernel_depth)[None, None, None, :, None] *
                             in_channels)
        
        # Load input tile with padding
        input_tile = tl.zeros((BLOCK_SIZE_H + 2*padding_h, BLOCK_SIZE_W + 2*padding_w, BLOCK_SIZE_D + 2*padding_d), dtype=tl.float32)
        
        # Compute input positions
        input_h_start = out_h_idx * stride_h - padding_h
        input_w_start = out_w_idx * stride_w - padding_w
        input_d_start = out_d_idx * stride_d - padding_d
        
        # Load input data with boundary checking
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                for kd in range(kernel_depth):
                    h = input_h_start + kh * dilation_h
                    w = input_w_start + kw * dilation_w
                    d = input_d_start + kd * dilation_d
                    
                    # Check if position is valid
                    if h >= 0 and h < input_height and w >= 0 and w < input_width and d >= 0 and d < input_depth:
                        for c_idx in range(CHANNELS_PER_BLOCK):
                            if c + c_idx < in_channels:
                                input_tile[h + padding_h, w + padding_w, d + padding_d] = tl.load(
                                    input_ptr + 
                                    batch_idx * in_channels * input_height * input_width * input_depth +
                                    (c + c_idx) * input_height * input_width * input_depth +
                                    h * input_width * input_depth +
                                    w * input_depth +
                                    d
                                )
        
        # Perform convolution computation
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                for kd in range(kernel_depth):
                    for c_idx in range(CHANNELS_PER_BLOCK):
                        if c + c_idx < in_channels:
                            acc += weight_tile[:, c_idx, kh, kw, kd] * input_tile[kh * dilation_h, kw * dilation_w, kd * dilation_d]
    
    # Store result
    output_offset = batch_idx * out_channels * output_height * output_width * output_depth + \
                   out_c_idx * output_height * output_width * output_depth + \
                   out_h_idx * output_width * output_depth + \
                   out_w_idx * output_depth + \
                   out_d_idx
    
    tl.store(output_ptr + output_offset, acc[0])

def triton_conv3d(input_tensor, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1)):
    """
    Triton implementation of 3D convolution
    """
    batch_size, in_channels, input_height, input_width, input_depth = input_tensor.shape
    out_channels, _, kernel_height, kernel_width, kernel_depth = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    output_depth = (input_depth + 2 * padding[2] - (dilation[2] * (kernel_depth - 1) + 1)) // stride[2] + 1
    
    # Create output tensor
    output = torch.zeros(batch_size, out_channels, output_height, output_width, output_depth, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Define grid dimensions
    grid = (
        batch_size,
        (out_channels + 31) // 32,  # Blocks per output channel
        (output_height + 7) // 8,   # Blocks per height
        (output_width + 7) // 8,    # Blocks per width
        (output_depth + 7) // 8     # Blocks per depth
    )
    
    # Launch kernel
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    BLOCK_SIZE_D = 8
    CHANNELS_PER_BLOCK = 32
    OUTPUT_CHANNELS_PER_BLOCK = 32
    
    conv3d_kernel[grid](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        input_depth,
        output_height,
        output_width,
        output_depth,
        kernel_height,
        kernel_width,
        kernel_depth,
        stride[0],
        stride[1],
        stride[2],
        padding[0],
        padding[1],
        padding[2],
        dilation[0],
        dilation[1],
        dilation[2],
        BLOCK_SIZE_H,
        BLOCK_SIZE_W,
        BLOCK_SIZE_D,
        CHANNELS_PER_BLOCK,
        OUTPUT_CHANNELS_PER_BLOCK
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with an asymmetric input and a square kernel.
    Optimized using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Set up grouping if needed
        if groups != 1:
            raise NotImplementedError("Groups > 1 not supported in this implementation")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width, depth).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out, depth_out).
        """
        return triton_conv3d(
            x, 
            self.weight, 
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )