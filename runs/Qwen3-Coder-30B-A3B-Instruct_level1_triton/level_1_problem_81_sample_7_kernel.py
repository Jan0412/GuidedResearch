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
    batch_size,
    in_channels,
    out_channels,
    height_in,
    width_in,
    height_out,
    width_out,
    kernel_size,
    stride,
    padding,
    dilation,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_idx = tl.program_id(2)
    
    # Calculate output dimensions per block
    output_elements_per_block = OUTPUT_ELEMENTS_PER_BLOCK
    num_output_blocks = (height_out * width_out + output_elements_per_block - 1) // output_elements_per_block
    
    # Each block processes multiple output elements
    output_block_start = output_idx * output_elements_per_block
    output_offsets = output_block_start + tl.arange(0, output_elements_per_block)
    
    # Calculate which output position this block covers
    out_h = output_offsets // width_out
    out_w = output_offsets % width_out
    
    # Only process valid outputs
    valid_mask = (output_offsets < height_out * width_out) & (out_h < height_out) & (out_w < width_out)
    
    # Shared memory for input window and weights
    shared_input = tl.shared_ptr(input_ptr + batch_idx * in_channels * height_in * width_in, 
                                in_channels * height_in * width_in, 0)
    shared_weight = tl.shared_ptr(weight_ptr, out_channels * in_channels * kernel_size * kernel_size, 0)
    
    # Initialize accumulator
    acc = tl.zeros((output_elements_per_block,), dtype=tl.float32)
    
    # Process each kernel element
    for k_h in range(kernel_size):
        for k_w in range(kernel_size):
            # Calculate input coordinates
            # For transposed conv: input is at (out_h * stride - padding + k_h * dilation, out_w * stride - padding + k_w * dilation)
            input_h = out_h * stride - padding + k_h * dilation
            input_w = out_w * stride - padding + k_w * dilation
            
            # Check if input coordinate is valid
            input_valid = (input_h >= 0) & (input_w >= 0) & (input_h < height_in) & (input_w < width_in)
            
            # Load input values
            input_vals = tl.zeros((output_elements_per_block,), dtype=tl.float32)
            input_mask = input_valid & valid_mask
            
            if input_mask.any():
                input_offset = input_h * width_in + input_w
                # Load from input tensor
                input_vals = tl.load(input_ptr + batch_idx * in_channels * height_in * width_in +
                                   channel_idx * height_in * width_in + input_offset,
                                   mask=input_mask, other=0.0)
            
            # Load weight values
            weight_val = tl.load(weight_ptr + channel_idx * out_channels * kernel_size * kernel_size +
                               (k_h * kernel_size + k_w) * out_channels + (output_idx % out_channels),
                               mask=True, other=0.0)
            
            # Accumulate
            acc += input_vals * weight_val
    
    # Add bias if available
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + (output_idx % out_channels), mask=True, other=0.0)
        acc += bias_val
    
    # Store results
    output_offset = batch_idx * out_channels * height_out * width_out + \
                   (output_idx // out_channels) * height_out * width_out + \
                   out_h * width_out + out_w
    tl.store(output_ptr + output_offset, acc, mask=valid_mask)

def triton_conv_transpose2d(input_tensor, weight, bias, stride, padding, dilation):
    """
    Custom Triton implementation of ConvTranspose2d
    """
    batch_size, in_channels, height_in, width_in = input_tensor.shape
    out_channels, _, kernel_size, _ = weight.shape
    
    # Calculate output dimensions
    height_out = (height_in - 1) * stride - 2 * padding + (kernel_size - 1) * dilation + 1
    width_out = (width_in - 1) * stride - 2 * padding + (kernel_size - 1) * dilation + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, height_out, width_out, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Grid configuration
    grid = (
        batch_size,
        in_channels,
        (height_out * width_out + 127) // 128
    )
    
    # Launch kernel
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 1
    OUTPUT_ELEMENTS_PER_BLOCK = 128
    
    # Simple version using basic approach - more complex optimizations would involve
    # shared memory tiling and better memory coalescing strategies
    conv_transpose2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        batch_size,
        in_channels,
        out_channels,
        height_in,
        width_in,
        height_out,
        width_out,
        kernel_size,
        stride,
        padding,
        dilation,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.weight)
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using custom Triton kernel.
        """
        return triton_conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)