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
    bias_ptr,
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
    OUTPUT_BLOCK_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1)
    out_d_idx = tl.program_id(2)
    out_h_idx = tl.program_id(3)
    out_w_idx = tl.program_id(4)
    
    # Calculate output position
    out_d = out_d_idx * OUTPUT_BLOCK_SIZE + tl.arange(0, OUTPUT_BLOCK_SIZE)
    out_h = out_h_idx * OUTPUT_BLOCK_SIZE + tl.arange(0, OUTPUT_BLOCK_SIZE)
    out_w = out_w_idx * OUTPUT_BLOCK_SIZE + tl.arange(0, OUTPUT_BLOCK_SIZE)
    
    # Mask for valid output positions
    valid_d = (out_d < output_depth)
    valid_h = (out_h < output_height)
    valid_w = (out_w < output_width)
    
    # Initialize accumulator
    acc = tl.zeros((OUTPUT_BLOCK_SIZE, OUTPUT_BLOCK_SIZE, OUTPUT_BLOCK_SIZE), dtype=tl.float32)
    
    # Iterate over input channels and kernel positions
    for ic in range(in_channels):
        # For each kernel position
        for kd in range(kernel_depth):
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    # Calculate input position
                    input_d = out_d * stride_d - padding_d + kd * dilation_d
                    input_h = out_h * stride_h - padding_h + kh * dilation_h
                    input_w = out_w * stride_w - padding_w + kw * dilation_w
                    
                    # Check if input position is valid
                    valid_input = (input_d >= 0) & (input_d < input_depth) & \
                                  (input_h >= 0) & (input_h < input_height) & \
                                  (input_w >= 0) & (input_w < input_width)
                    
                    # Load input value
                    input_val = tl.load(input_ptr + 
                                       batch_idx * in_channels * input_depth * input_height * input_width +
                                       ic * input_depth * input_height * input_width +
                                       input_d * input_height * input_width +
                                       input_h * input_width +
                                       input_w, mask=valid_input & valid_d & valid_h & valid_w, other=0.0)
                    
                    # Load weight
                    weight_val = tl.load(weight_ptr + 
                                        out_c_idx * in_channels * kernel_depth * kernel_height * kernel_width +
                                        ic * kernel_depth * kernel_height * kernel_width +
                                        kd * kernel_height * kernel_width +
                                        kh * kernel_width +
                                        kw)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Apply bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_c_idx)
        acc += bias_val
    
    # Store result
    for i in range(OUTPUT_BLOCK_SIZE):
        for j in range(OUTPUT_BLOCK_SIZE):
            for k in range(OUTPUT_BLOCK_SIZE):
                if (i < len(out_d) and j < len(out_h) and k < len(out_w) and 
                    out_d[i] < output_depth and out_h[j] < output_height and out_w[k] < output_width):
                    tl.store(output_ptr + 
                            batch_idx * out_channels * output_depth * output_height * output_width +
                            out_c_idx * output_depth * output_height * output_width +
                            out_d[i] * output_height * output_width +
                            out_h[j] * output_width +
                            out_w[k],
                            acc[i, j, k])

def triton_conv_transpose3d(input_tensor, weight, bias, stride, padding, dilation, output_padding):
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_depth - 1) + 1
    output_height = (input_height - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_height - 1) + 1
    output_width = (input_width - 1) * stride[2] - 2 * padding[2] + dilation[2] * (kernel_width - 1) + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Set up kernel launch parameters
    BLOCK_SIZE = 16
    OUTPUT_BLOCK_SIZE = 4
    
    # Grid configuration
    grid = (
        batch_size,
        out_channels,
        (output_depth + OUTPUT_BLOCK_SIZE - 1) // OUTPUT_BLOCK_SIZE,
        (output_height + OUTPUT_BLOCK_SIZE - 1) // OUTPUT_BLOCK_SIZE,
        (output_width + OUTPUT_BLOCK_SIZE - 1) // OUTPUT_BLOCK_SIZE
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
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
        stride[0],
        stride[1],
        stride[2],
        padding[0],
        padding[1],
        padding[2],
        dilation[0],
        dilation[1],
        dilation[2],
        BLOCK_SIZE=BLOCK_SIZE,
        OUTPUT_BLOCK_SIZE=OUTPUT_BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding, padding) if isinstance(padding, int) else tuple(padding)
        self.dilation = (dilation, dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        
        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract dimensions
        batch_size, _, input_depth, input_height, input_width = x.shape
        
        # Calculate output dimensions
        output_depth = (input_depth - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] * (self.kernel_size - 1) + 1
        output_height = (input_height - 1) * self.stride[1] - 2 * self.padding[1] + self.dilation[1] * (self.kernel_size - 1) + 1
        output_width = (input_width - 1) * self.stride[2] - 2 * self.padding[2] + self.dilation[2] * (self.kernel_size - 1) + 1
        
        # Call Triton kernel
        if self.bias is not None:
            return triton_conv_transpose3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, (0, 0, 0))
        else:
            return triton_conv_transpose3d(x, self.weight, None, self.stride, self.padding, self.dilation, (0, 0, 0))

# Helper function to compute output shape
def compute_output_shape(input_shape, kernel_size, stride, padding, dilation):
    batch_size, in_channels, input_depth, input_height, input_width = input_shape
    output_depth = (input_depth - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size - 1) + 1
    output_height = (input_height - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size - 1) + 1
    output_width = (input_width - 1) * stride[2] - 2 * padding[2] + dilation[2] * (kernel_size - 1) + 1
    return (batch_size, in_channels, output_depth, output_height, output_width)