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
    input_shape,
    weight_shape,
    output_shape,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    weight_h,
    weight_w,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get thread indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Calculate block start positions
    block_start_h = pid_m * BLOCK_SIZE_H
    block_start_w = pid_n * BLOCK_SIZE_W
    
    # Shared memory for weight caching
    shared_weight = tl.shared_tensor(tl.make_block_ptr(weight_ptr, shape=(out_channels, in_channels, weight_h, weight_w), 
                                                       strides=(weight_h * weight_w * in_channels, weight_h * weight_w, weight_w, 1),
                                                       offsets=(0, 0, 0, 0), block_shape=(BLOCK_SIZE_C, 1, weight_h, weight_w)),
                                     (BLOCK_SIZE_C, 1, weight_h, weight_w))
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over input channels
    for c in range(0, in_channels, 1):
        # Calculate input position
        input_offset = pid_b * (in_channels * input_height * input_width) + c * (input_height * input_width)
        
        # Loop over kernel height and width
        for kh in range(weight_h):
            for kw in range(weight_w):
                # Calculate output position
                out_h_start = block_start_h + kh
                out_w_start = block_start_w + kw
                
                # Check bounds
                if out_h_start < output_height and out_w_start < output_width:
                    # Calculate input position from output position
                    input_h = (out_h_start - pad_h) // stride_h
                    input_w = (out_w_start - pad_w) // stride_w
                    
                    # Check if input position is valid
                    if input_h >= 0 and input_w >= 0 and input_h < input_height and input_w < input_width:
                        # Calculate actual input position
                        input_pos = input_offset + input_h * input_width + input_w
                        
                        # Load input value
                        input_val = tl.load(input_ptr + input_pos, mask=True)
                        
                        # Load weight value
                        weight_val = tl.load(weight_ptr + (c * weight_h * weight_w + kh * weight_w + kw) * out_channels + 
                                           (out_h_start * output_width + out_w_start), mask=True)
                        
                        # Accumulate
                        acc += input_val * weight_val
    
    # Store result
    output_offset = pid_b * (out_channels * output_height * output_width) + \
                   (block_start_h * output_width + block_start_w)
    
    # Write output
    tl.store(output_ptr + output_offset, acc, mask=(block_start_h < output_height) & (block_start_w < output_width))

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on GPU and contiguous
        x = x.contiguous()
        if self.bias is not None:
            self.bias = self.bias.contiguous()
        
        # Get dimensions
        batch_size, _, input_height, input_width = x.shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        
        # Calculate output dimensions
        output_height = (input_height - 1) * stride_h - 2 * pad_h + kernel_h
        output_width = (input_width - 1) * stride_w - 2 * pad_w + kernel_w
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_channels, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Define block sizes
        BLOCK_SIZE_H = 16
        BLOCK_SIZE_W = 16
        BLOCK_SIZE_C = 8
        
        # Grid configuration
        grid_h = (output_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
        grid_w = (output_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
        grid = (grid_h, grid_w, batch_size)
        
        # Launch kernel
        conv_transpose2d_kernel[grid](
            x,
            self.weight,
            output,
            self.bias,
            x.shape,
            self.weight.shape,
            output.shape,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            batch_size,
            self.in_channels,
            self.out_channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_h,
            kernel_w,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_W=BLOCK_SIZE_W,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
            GROUP_SIZE_M=8
        )
        
        return output