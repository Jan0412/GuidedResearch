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
    
    # Calculate global output position
    output_row = output_idx // width_out
    output_col = output_idx % width_out
    
    # Shared memory for input tile
    shared_input = tl.shared_pointer(input_ptr + batch_idx * in_channels * height_in * width_in, 
                                    (in_channels, height_in, width_in))
    
    # Shared memory for weight tile
    shared_weight = tl.shared_pointer(weight_ptr + channel_idx * in_channels * kernel_size * kernel_size,
                                     (in_channels, kernel_size, kernel_size))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Compute convolution for this output position
    for k in range(kernel_size):
        for l in range(kernel_size):
            # Calculate input coordinates
            input_row = output_row * stride - padding + k * dilation
            input_col = output_col * stride - padding + l * dilation
            
            # Check bounds
            if input_row >= 0 and input_row < height_in and input_col >= 0 and input_col < width_in:
                # Load input value
                input_val = tl.load(input_ptr + batch_idx * in_channels * height_in * width_in + 
                                  channel_idx * height_in * width_in + 
                                  input_row * width_in + input_col)
                
                # Load weight value
                weight_val = tl.load(weight_ptr + channel_idx * in_channels * kernel_size * kernel_size + 
                                   k * in_channels * kernel_size + l * in_channels + channel_idx)
                
                acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr != 0:
        bias_val = tl.load(bias_ptr + channel_idx)
        acc += bias_val
    
    # Store result
    if output_idx < height_out * width_out:
        tl.store(output_ptr + batch_idx * out_channels * height_out * width_out + 
                channel_idx * height_out * width_out + 
                output_row * width_out + output_col, acc)

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
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias_param = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias_param', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get dimensions
        batch_size, _, height_in, width_in = x.shape
        
        # Calculate output dimensions
        height_out = (height_in - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1
        width_out = (width_in - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1
        
        # Prepare output tensor
        output = torch.empty(batch_size, self.out_channels, height_out, width_out, device=x.device, dtype=torch.float32)
        
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        if self.bias_param is not None:
            bias = self.bias_param.contiguous()
        else:
            bias = None
        
        # Launch kernel
        BLOCK_SIZE = 128
        CHANNELS_PER_BLOCK = 8
        OUTPUT_ELEMENTS_PER_BLOCK = 64
        
        # Grid configuration
        grid_batch = batch_size
        grid_channels = (self.out_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK
        grid_output = (height_out * width_out + OUTPUT_ELEMENTS_PER_BLOCK - 1) // OUTPUT_ELEMENTS_PER_BLOCK
        
        # Launch kernel
        conv_transpose2d_kernel[(grid_batch, grid_channels, grid_output)](
            x,
            weight,
            output,
            bias,
            batch_size,
            self.in_channels,
            self.out_channels,
            height_in,
            width_in,
            height_out,
            width_out,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            BLOCK_SIZE,
            CHANNELS_PER_BLOCK,
            OUTPUT_ELEMENTS_PER_BLOCK
        )
        
        return output