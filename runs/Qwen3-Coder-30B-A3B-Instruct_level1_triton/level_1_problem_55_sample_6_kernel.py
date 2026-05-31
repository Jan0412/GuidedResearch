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
    input_stride_0,
    input_stride_1,
    input_stride_2,
    input_stride_3,
    weight_stride_0,
    weight_stride_1,
    weight_stride_2,
    weight_stride_3,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    output_stride_3,
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
    TILE_H: tl.constexpr,
    TILE_W: tl.constexpr,
    TILE_C: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_y = tl.program_id(2)
    out_x = tl.program_id(3)
    
    # Shared memory for tile
    tile_input = tl.shared_pointer(tl.float32, TILE_H * TILE_W * TILE_C)
    tile_weight = tl.shared_pointer(tl.float32, TILE_H * TILE_W * TILE_C)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    if has_bias:
        acc = tl.load(bias_ptr + out_ch_idx)
    
    # Loop over input channels
    for c in range(0, in_channels, TILE_C):
        # Load input tile
        input_offset = batch_idx * input_stride_0 + c * input_stride_1 + out_y * stride_h * input_stride_2 + out_x * stride_w * input_stride_3
        input_tile = tl.zeros((TILE_H, TILE_W), dtype=tl.float32)
        
        # Load weights tile
        weight_offset = out_ch_idx * weight_stride_0 + c * weight_stride_1
        weight_tile = tl.zeros((kernel_height, kernel_width), dtype=tl.float32)
        
        # Process kernel elements
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input position
                ih = out_y * stride_h + kh * dilation_h - padding_h
                iw = out_x * stride_w + kw * dilation_w - padding_w
                
                # Check bounds
                if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                    input_val = tl.load(input_ptr + input_offset + ih * input_stride_2 + iw * input_stride_3)
                    weight_val = tl.load(weight_ptr + weight_offset + kh * weight_stride_2 + kw * weight_stride_3)
                    acc += input_val * weight_val
                else:
                    acc += 0.0
    
    # Store output
    output_offset = batch_idx * output_stride_0 + out_ch_idx * output_stride_1 + out_y * output_stride_2 + out_x * output_stride_3
    tl.store(output_ptr + output_offset, acc)

def triton_conv2d(input_tensor, weight, bias, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """
    Triton implementation of 2D convolution using shared memory tiling
    """
    # Ensure input is contiguous
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
    
    # Set up kernel parameters
    BLOCK_SIZE = 256
    TILE_H = 8
    TILE_W = 8
    TILE_C = 16
    
    # Grid configuration
    grid = (
        batch_size,
        out_channels,
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
        BLOCK_SIZE=BLOCK_SIZE,
        TILE_H=TILE_H,
        TILE_W=TILE_W,
        TILE_C=TILE_C
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights and bias
        if groups == 1:
            self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        else:
            raise NotImplementedError("Grouped convolution not implemented")
            
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel
        """
        return triton_conv2d(
            x, 
            self.weight, 
            self.bias, 
            stride=(self.stride, self.stride), 
            padding=(self.padding, self.padding), 
            dilation=(self.dilation, self.dilation)
        )

# Placeholder functions for compatibility
def get_inputs():
    batch_size = 8
    height = 512
    width = 1024
    in_channels = 64
    out_channels = 128
    kernel_size = 3
    
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [64, 128, 3]