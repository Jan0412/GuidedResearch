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
    # Get program IDs
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    out_d = tl.program_id(2)
    out_w = tl.program_id(3)
    out_h = tl.program_id(4)
    
    # Calculate output indices
    out_idx = batch_id * out_channels * output_depth * output_width * output_height + \
              out_channel_id * output_depth * output_width * output_height + \
              out_d * output_width * output_height + \
              out_w * output_height + \
              out_h
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels and kernel dimensions
    for ic in range(in_channels):
        for kd in range(kernel_depth):
            for kw in range(kernel_width):
                for kh in range(kernel_height):
                    # Calculate input positions with stride and padding
                    input_d = out_d * stride_d - padding_d + kd * dilation_d
                    input_w = out_w * stride_w - padding_w + kw * dilation_w
                    input_h = out_h * stride_h - padding_h + kh * dilation_h
                    
                    # Check bounds
                    if (input_d >= 0 and input_d < input_depth and 
                        input_w >= 0 and input_w < input_width and 
                        input_h >= 0 and input_h < input_height):
                        
                        # Calculate input index
                        input_idx = batch_id * in_channels * input_depth * input_width * input_height + \
                                   ic * input_depth * input_width * input_height + \
                                   input_d * input_width * input_height + \
                                   input_w * input_height + \
                                   input_h
                        
                        # Calculate weight index
                        weight_idx = out_channel_id * in_channels * kernel_depth * kernel_width * kernel_height + \
                                    ic * kernel_depth * kernel_width * kernel_height + \
                                    kd * kernel_width * kernel_height + \
                                    kw * kernel_height + \
                                    kh
                        
                        # Accumulate
                        input_val = tl.load(input_ptr + input_idx, mask=True)
                        weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                        acc += input_val * weight_val
    
    # Store result
    tl.store(output_ptr + out_idx, acc, mask=True)

def triton_conv3d(input_tensor, weight, bias=None, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1)):
    """
    Triton implementation of 3D convolution
    """
    # Ensure tensors are contiguous and on CUDA
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
    output = torch.empty(batch_size, out_channels, output_depth, output_width, output_height, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE_M = 1
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_K = 32
    
    # Grid configuration
    grid = (
        batch_size,
        out_channels,
        output_depth,
        output_width,
        output_height
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
    
    # Add bias if present
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
        self.stride = (stride, stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        
        # Handle groups
        if groups > 1:
            raise NotImplementedError("Grouped convolution not yet implemented")
    
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
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )

# Test code (not included in final version)
# batch_size = 16
# in_channels = 3
# out_channels = 64
# kernel_size = 3
# depth = 64
# width = 64
# height = 64

# def get_inputs():
#     x = torch.rand(batch_size, in_channels, depth, width, height)
#     return [x]

# def get_init_inputs():
#     return [in_channels, out_channels, kernel_size]