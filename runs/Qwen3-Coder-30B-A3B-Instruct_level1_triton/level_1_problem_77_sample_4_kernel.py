import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_depth,
    input_height,
    input_width,
    output_depth,
    output_height,
    output_width,
    kernel_depth,
    kernel_height,
    kernel_width,
    stride_d,
    stride_h,
    stride_w,
    padding_d,
    padding_h,
    padding_w,
    dilation_d,
    dilation_h,
    dilation_w,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get block indices
    pid = tl.program_id(0)
    pid_batch = tl.program_id(1)
    
    # Each block processes one output element
    # Calculate global output position
    output_idx = pid
    
    # Convert linear index to 3D coordinates
    out_w = output_idx % output_width
    out_h = (output_idx // output_width) % output_height
    out_d = (output_idx // (output_width * output_height)) % output_depth
    
    if out_d >= output_depth or out_h >= output_height or out_w >= output_width:
        return
    
    # Loop over input channels and output channels
    for oc in range(out_channels):
        # Initialize accumulator
        acc = 0.0
        
        # Loop over kernel dimensions
        for kd in range(kernel_depth):
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    # Calculate input position
                    input_d = out_d * stride_d - padding_d + kd * dilation_d
                    input_h = out_h * stride_h - padding_h + kh * dilation_h
                    input_w = out_w * stride_w - padding_w + kw * dilation_w
                    
                    # Check bounds
                    if (input_d >= 0 and input_d < input_depth and
                        input_h >= 0 and input_h < input_height and
                        input_w >= 0 and input_w < input_width):
                        
                        # Load input value
                        input_val = tl.load(input_ptr + 
                                          pid_batch * in_channels * input_depth * input_height * input_width +
                                          input_d * input_height * input_width +
                                          input_h * input_width +
                                          input_w)
                        
                        # Load weight value
                        weight_val = tl.load(weight_ptr + 
                                           oc * in_channels * kernel_depth * kernel_height * kernel_width +
                                           input_idx * kernel_depth * kernel_height * kernel_width +
                                           kd * kernel_height * kernel_width +
                                           kh * kernel_width +
                                           kw)
                        
                        acc += input_val * weight_val
        
        # Store result
        tl.store(output_ptr + 
                pid_batch * out_channels * output_depth * output_height * output_width +
                oc * output_depth * output_height * output_width +
                out_d * output_height * output_width +
                out_h * output_width +
                out_w, acc)

def triton_conv_transpose3d(input_tensor, weight, bias=None, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1)):
    """
    Triton implementation of 3D transposed convolution
    """
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    dil_d, dil_h, dil_w = dilation
    
    output_depth = (input_depth - 1) * stride_d - 2 * pad_d + dil_d * (kernel_depth - 1) + 1
    output_height = (input_height - 1) * stride_h - 2 * pad_h + dil_h * (kernel_height - 1) + 1
    output_width = (input_width - 1) * stride_w - 2 * pad_w + dil_w * (kernel_width - 1) + 1
    
    # Create output tensor
    output = torch.zeros(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Grid configuration
    total_elements = batch_size * out_channels * output_depth * output_height * output_width
    BLOCK_SIZE = 1024
    grid = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, batch_size
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_depth,
        kernel_height,
        kernel_width,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        dil_d,
        dil_h,
        dil_w,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE_M=8
    )
    
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation, dilation) if isinstance(dilation, int) else dilation
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights using Kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x, 
            self.weight, 
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )