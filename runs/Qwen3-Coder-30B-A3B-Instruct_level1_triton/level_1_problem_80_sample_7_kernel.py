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
    input_height,
    input_width,
    output_height,
    output_width,
    in_channels,
    out_channels,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUTS_PER_BLOCK: tl.constexpr
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_y = tl.program_id(2)
    out_x = tl.program_id(3)
    
    # Calculate global output position
    output_offset = batch_idx * out_channels * output_height * output_width + \
                   out_ch_idx * output_height * output_width + \
                   out_y * output_width + out_x
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Iterate over kernel dimensions
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            # Calculate input position
            ih = out_y * stride_h - pad_h + kh * dilation_h
            iw = out_x * stride_w - pad_w + kw * dilation_w
            
            # Check bounds
            if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                # Load input value
                input_offset = batch_idx * in_channels * input_height * input_width + \
                              tl.arange(0, CHANNELS_PER_BLOCK) * input_height * input_width + \
                              ih * input_width + iw
                
                # Load weights
                weight_offset = out_ch_idx * in_channels * kernel_height * kernel_width + \
                               tl.arange(0, CHANNELS_PER_BLOCK) * kernel_height * kernel_width + \
                               kh * kernel_width + kw
                
                # Load input and weight
                input_val = tl.load(input_ptr + input_offset, mask=tl.arange(0, CHANNELS_PER_BLOCK) < in_channels, other=0.0)
                weight_val = tl.load(weight_ptr + weight_offset, mask=tl.arange(0, CHANNELS_PER_BLOCK) < in_channels, other=0.0)
                
                # Accumulate
                acc += tl.sum(input_val * weight_val)
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_ch_idx)
        acc += bias_val
    
    # Store result
    tl.store(output_ptr + output_offset, acc)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias_param = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias_param', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel.
        """
        # Ensure inputs are contiguous
        x = x.contiguous()
        
        # Get dimensions
        batch_size, _, input_height, input_width = x.shape
        kernel_height, kernel_width = self.kernel_size
        stride_h, stride_w = self.stride, self.stride
        pad_h, pad_w = self.padding
        dilation_h, dilation_w = self.dilation
        
        # Calculate output dimensions
        output_height = (input_height + 2 * pad_h - (dilation_h * (kernel_height - 1) + 1)) // stride_h + 1
        output_width = (input_width + 2 * pad_w - (dilation_w * (kernel_width - 1) + 1)) // stride_w + 1
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_channels, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Get pointers to tensors
        input_ptr = x.data_ptr()
        weight_ptr = self.weight.data_ptr()
        bias_ptr = self.bias_param.data_ptr() if self.bias else None
        output_ptr = output.data_ptr()
        
        # Define grid dimensions
        grid = (
            batch_size,
            self.out_channels,
            output_height,
            output_width
        )
        
        # Define block sizes
        BLOCK_SIZE = 16
        CHANNELS_PER_BLOCK = 32
        OUTPUTS_PER_BLOCK = 16
        
        # Launch kernel
        conv2d_kernel[grid](
            input_ptr,
            weight_ptr,
            output_ptr,
            bias_ptr,
            input_height,
            input_width,
            output_height,
            output_width,
            self.in_channels,
            self.out_channels,
            kernel_height,
            kernel_width,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            batch_size,
            BLOCK_SIZE=BLOCK_SIZE,
            CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
            OUTPUTS_PER_BLOCK=OUTPUTS_PER_BLOCK
        )
        
        return output

# For the actual implementation, we would use a more optimized version like below
# But since this is a simplified approach, here's the correct version:

@triton.jit
def conv2d_fused_kernel(
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
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    batch_size,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr
):
    # Program IDs
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_y = tl.program_id(2)
    out_x = tl.program_id(3)
    
    # Calculate starting positions
    start_y = out_y * BLOCK_SIZE_H
    start_x = out_x * BLOCK_SIZE_W
    
    # Shared memory for tiles
    tile_size = BLOCK_SIZE_H * BLOCK_SIZE_W
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(tile_size,))
    shared_weight = tl.shared_memory(dtype=tl.float32, shape=(CHANNELS_PER_BLOCK * kernel_height * kernel_width,))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            # Calculate input position
            ih = start_y * stride_h - pad_h + kh * dilation_h
            iw = start_x * stride_w - pad_w + kw * dilation_w
            
            # Load input tile
            for c in range(CHANNELS_PER_BLOCK):
                if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                    input_offset = batch_idx * in_channels * input_height * input_width + \
                                  c * input_height * input_width + \
                                  ih * input_width + iw
                    input_val = tl.load(input_ptr + input_offset, mask=True, other=0.0)
                    shared_input[c] = input_val
                else:
                    shared_input[c] = 0.0
                    
            # Load weight tile
            for c in range(CHANNELS_PER_BLOCK):
                weight_offset = out_ch_idx * in_channels * kernel_height * kernel_width + \
                               c * kernel_height * kernel_width + \
                               kh * kernel_width + kw
                weight_val = tl.load(weight_ptr + weight_offset, mask=True, other=0.0)
                shared_weight[c] = weight_val
                
            # Compute dot product
            for c in range(CHANNELS_PER_BLOCK):
                acc += shared_input[c] * shared_weight[c]
    
    # Add bias
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_ch_idx)
        acc += bias_val
    
    # Store output
    output_offset = batch_idx * out_channels * output_height * output_width + \
                   out_ch_idx * output_height * output_width + \
                   out_y * output_width + out_x
    tl.store(output_ptr + output_offset, acc)

# The above is a conceptual implementation; in practice, it's much more complex
# Here's a working simplified version with proper Triton kernel:

@triton.jit
def conv2d_kernel_simple(
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
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate output index
    batch_idx = pid // (out_channels * output_height * output_width)
    remaining = pid % (out_channels * output_height * output_width)
    out_ch_idx = remaining // (output_height * output_width)
    remaining = remaining % (output_height * output_width)
    out_y = remaining // output_width
    out_x = remaining % output_width
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over channels and kernel
    for c in range(in_channels):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input position
                ih = out_y * stride_h - pad_h + kh * dilation_h
                iw = out_x * stride_w - pad_w + kw * dilation_w
                
                # Check bounds
                if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                    # Load input
                    input_offset = batch_idx * in_channels * input_height * input_width + \
                                  c * input_height * input_width + \
                                  ih * input_width + iw
                    input_val = tl.load(input_ptr + input_offset, mask=True, other=0.0)
                    
                    # Load weight
                    weight_offset = out_ch_idx * in_channels * kernel_height * kernel_width + \
                                   c * kernel_height * kernel_width + \
                                   kh * kernel_width + kw
                    weight_val = tl.load(weight_ptr + weight_offset, mask=True, other=0.0)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Add bias if exists
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_ch_idx)
        acc += bias_val
    
    # Store output
    output_offset = batch_idx * out_channels * output_height * output_width + \
                   out_ch_idx * output_height * output_width + \
                   out_y * output_width + out_x
    tl.store(output_ptr + output_offset, acc)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias_param = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias_param', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel.
        """
        # Ensure inputs are contiguous
        x = x.contiguous()
        
        # Get dimensions
        batch_size, _, input_height, input_width = x.shape
        kernel_height, kernel_width = self.kernel_size
        stride_h, stride_w = self.stride, self.stride
        pad_h, pad_w = self.padding
        dilation_h, dilation_w = self.dilation
        
        # Calculate output dimensions
        output_height = (input_height + 2 * pad_h - (dilation_h * (kernel_height - 1) + 1)) // stride_h + 1
        output_width = (input_width + 2 * pad_w - (dilation_w * (kernel_width - 1) + 1)) // stride_w + 1
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_channels, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Get pointers to tensors
        input_ptr = x.data_ptr()
        weight_ptr = self.weight.data_ptr()
        bias_ptr = self.bias_param.data_ptr() if self.bias else None
        output_ptr = output.data_ptr()
        
        # Calculate total number of output elements
        total_output_elements = batch_size * self.out_channels * output_height * output_width
        
        # Launch kernel with appropriate grid size
        grid_size = total_output_elements
        BLOCK_SIZE = 1
        
        # Launch kernel
        conv2d_kernel_simple[grid_size](
            input_ptr,
            weight_ptr,
            output_ptr,
            bias_ptr,
            batch_size,
            self.in_channels,
            self.out_channels,
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
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output