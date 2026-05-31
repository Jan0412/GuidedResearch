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
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    output_row = tl.program_id(2)
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((OUTPUT_ELEMENTS_PER_BLOCK,), dtype=tl.float32)
    
    # Process input channels in chunks
    for channel_chunk in range(0, in_channels, CHANNELS_PER_BLOCK):
        # Load weights for this channel chunk
        weight_chunk = tl.load(weight_ptr + 
                              out_channel_id * in_channels * kernel_height * kernel_width +
                              channel_chunk * kernel_height * kernel_width +
                              tl.arange(0, CHANNELS_PER_BLOCK)[:, None, None] * kernel_height * kernel_width +
                              tl.arange(0, kernel_height)[None, :, None] * kernel_width +
                              tl.arange(0, kernel_width)[None, None, :])
        
        # Process output elements in this block
        for output_elem in range(OUTPUT_ELEMENTS_PER_BLOCK):
            if output_elem >= output_height * output_width:
                break
                
            # Calculate output position
            out_y = output_elem // output_width
            out_x = output_elem % output_width
            
            # Skip if out of bounds
            if out_y >= output_height or out_x >= output_width:
                continue
                
            # Calculate input positions
            input_y_start = out_y * stride_h - pad_h
            input_x_start = out_x * stride_w - pad_w
            
            # Convolution computation
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    input_y = input_y_start + kh * dilation_h
                    input_x = input_x_start + kw * dilation_w
                    
                    # Check bounds
                    if input_y >= 0 and input_y < input_height and input_x >= 0 and input_x < input_width:
                        # Load input value
                        input_val = tl.load(input_ptr + 
                                          batch_id * in_channels * input_height * input_width +
                                          channel_chunk * input_height * input_width +
                                          input_y * input_width + 
                                          input_x)
                        
                        # Accumulate
                        acc[output_elem] += input_val * weight_chunk[kh, kw]
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_channel_id)
        acc += bias_val
    
    # Store results
    for i in range(OUTPUT_ELEMENTS_PER_BLOCK):
        if i < output_height * output_width:
            out_idx = batch_id * out_channels * output_height * output_width + \
                     out_channel_id * output_height * output_width + \
                     i
            tl.store(output_ptr + out_idx, acc[i])

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_height, self.kernel_width = kernel_size
        self.stride_h, self.stride_w = stride if isinstance(stride, tuple) else (stride, stride)
        self.pad_h, self.pad_w = padding
        self.dilation_h, self.dilation_w = dilation
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get dimensions
        batch_size, _, input_height, input_width = x.shape
        
        # Compute output dimensions
        output_height = (input_height + 2 * self.pad_h - (self.dilation_h * (self.kernel_height - 1) + 1)) // self.stride_h + 1
        output_width = (input_width + 2 * self.pad_w - (self.dilation_w * (self.kernel_width - 1) + 1)) // self.stride_w + 1
        
        # Ensure we're working with contiguous tensors
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_channels, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Set up parameters for kernel launch
        BLOCK_SIZE = 16
        CHANNELS_PER_BLOCK = 16
        OUTPUT_ELEMENTS_PER_BLOCK = 64
        
        # Grid configuration
        grid = (
            batch_size,
            self.out_channels,
            (output_height * output_width + OUTPUT_ELEMENTS_PER_BLOCK - 1) // OUTPUT_ELEMENTS_PER_BLOCK
        )
        
        # Launch kernel
        conv2d_kernel[grid](
            x,
            weight,
            output,
            self.bias,
            batch_size,
            self.in_channels,
            self.out_channels,
            input_height,
            input_width,
            output_height,
            output_width,
            self.kernel_height,
            self.kernel_width,
            self.stride_h,
            self.stride_w,
            self.pad_h,
            self.pad_w,
            self.dilation_h,
            self.dilation_w,
            BLOCK_SIZE=BLOCK_SIZE,
            CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
            OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
        )
        
        return output

# Test code
batch_size = 8
in_channels = 32
out_channels = 64
kernel_size = (5, 9)
width = 512
height = 512
stride = 1
padding = (2, 4)
dilation = (2, 3)

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]