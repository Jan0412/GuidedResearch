import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose2d_kernel(
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
    batch_size,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get block indices
    pid = tl.program_id(0)
    batch_pid = tl.program_id(1)
    
    # Each block processes one output pixel per batch
    output_h = pid // output_width
    output_w = pid % output_width
    
    if output_h >= output_height or output_w >= output_width:
        return
        
    # Initialize accumulator
    acc = tl.zeros((out_channels,), dtype=tl.float32)
    
    # Loop over input channels
    for c in range(in_channels):
        # Loop over kernel
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate input position
                ih = output_h * stride_h - pad_h + kh
                iw = output_w * stride_w - pad_w + kw
                
                # Check bounds
                if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                    # Load input value
                    input_val = tl.load(input_ptr + 
                                       batch_pid * input_height * input_width * in_channels +
                                       ih * input_width * in_channels +
                                       iw * in_channels +
                                       c)
                    
                    # Load weight
                    weight_val = tl.load(weight_ptr +
                                        c * out_channels * kernel_h * kernel_w +
                                        (kernel_h - 1 - kh) * out_channels * kernel_w +
                                        (kernel_w - 1 - kw) * out_channels +
                                        0)  # Assuming out_channels dimension is last
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_vals = tl.load(bias_ptr + tl.arange(0, out_channels))
        acc += bias_vals
    
    # Store output
    output_offset = batch_pid * output_height * output_width * out_channels + \
                   output_h * output_width * out_channels + \
                   output_w * out_channels
    tl.store(output_ptr + output_offset + tl.arange(0, out_channels), acc)

def triton_conv_transpose2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0)):
    """
    Triton implementation of ConvTranspose2d
    """
    # Ensure inputs are contiguous and on GPU
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_h
    output_width = (input_width - 1) * stride[1] - 2 * padding[1] + kernel_w
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Launch kernel
    grid = (output_height * output_width, batch_size)
    
    # Use a reasonable block size
    BLOCK_SIZE = 256
    GROUP_SIZE = 8
    
    # Call kernel
    conv_transpose2d_kernel[grid](
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
        batch_size,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=GROUP_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding
        )

# Helper function to initialize parameters (if needed)
def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)