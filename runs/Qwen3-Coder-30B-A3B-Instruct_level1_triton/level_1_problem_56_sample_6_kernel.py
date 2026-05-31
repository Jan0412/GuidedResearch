import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    input_height,
    input_width,
    output_height,
    output_width,
    in_channels,
    out_channels,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    batch_size,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get the block indices
    pid_batch = tl.program_id(0)
    pid_out_h = tl.program_id(1)
    pid_out_w = tl.program_id(2)
    pid_out_c = tl.program_id(3)
    
    # Shared memory for input tile
    TILE_H = BLOCK_SIZE_H
    TILE_W = BLOCK_SIZE_W
    
    # Calculate the starting positions for this thread block
    batch_offset = pid_batch * input_height * input_width * in_channels
    output_offset = pid_batch * output_height * output_width * out_channels
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over the kernel dimensions
    for kh in range(0, kernel_h):
        for kw in range(0, kernel_w):
            # Calculate input position
            ih = pid_out_h * stride_h + kh * dilation_h - pad_h
            iw = pid_out_w * stride_w + kw * dilation_w - pad_w
            
            # Check bounds
            if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                # Load input tile
                input_tile = tl.load(input_ptr + batch_offset + ih * input_width * in_channels + iw * in_channels + tl.arange(0, BLOCK_SIZE_C))
                
                # Load weight
                weight_val = tl.load(weight_ptr + pid_out_c * kernel_h * kernel_w * in_channels + kh * kernel_w * in_channels + kw * in_channels + tl.arange(0, BLOCK_SIZE_C))
                
                # Accumulate
                acc += tl.sum(input_tile[:, None] * weight_val[None, :], axis=0)
    
    # Apply bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + pid_out_c)
        acc += bias_val
    
    # Store result
    tl.store(output_ptr + output_offset + pid_out_h * output_width * out_channels + pid_out_w * out_channels + pid_out_c, acc)

def triton_conv2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """
    Custom Triton implementation of 2D convolution
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_h - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_w - 1) + 1)) // stride[1] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Define block sizes
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 32
    
    # Grid dimensions
    grid = (
        batch_size,
        (output_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (output_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W,
        out_channels
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        input_height,
        input_width,
        output_height,
        output_width,
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        batch_size,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        GROUP_SIZE_M=8
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with asymmetric input and kernel sizes.
    Uses a custom Triton kernel for optimization.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using the Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv2d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation
        )

# Test code
batch_size = 8
in_channels = 64
out_channels = 128
kernel_size = (5, 7)
height = 512
width = 256

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization