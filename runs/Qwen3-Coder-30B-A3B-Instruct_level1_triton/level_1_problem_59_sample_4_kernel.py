import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    input_depth,
    output_height,
    output_width,
    output_depth,
    kernel_height,
    kernel_width,
    kernel_depth,
    stride_h,
    stride_w,
    stride_d,
    padding_h,
    padding_w,
    padding_d,
    dilation_h,
    dilation_w,
    dilation_d,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(axis=0)
    
    # Grid dimensions
    grid_m = (output_height + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (output_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Which block of (output_height, output_width) to compute
    m_block = pid // grid_n
    n_block = pid % grid_n
    
    # Compute the starting position of this block
    start_m = m_block * BLOCK_SIZE_M
    start_n = n_block * BLOCK_SIZE_N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over the kernel depth dimension
    for k in range(0, kernel_depth):
        # Compute output coordinates
        out_h = start_m + tl.arange(0, BLOCK_SIZE_M)
        out_w = start_n + tl.arange(0, BLOCK_SIZE_N)
        
        # Check bounds for output coordinates
        out_h_mask = out_h < output_height
        out_w_mask = out_w < output_width
        
        # Loop over input channels
        for c in range(0, in_channels, BLOCK_SIZE_K):
            # Compute input coordinates
            input_h_start = out_h * stride_h - padding_h
            input_w_start = out_w * stride_w - padding_w
            
            # Create masks for valid input positions
            input_h = input_h_start[:, None] + tl.arange(0, kernel_height)[:, None] * dilation_h
            input_w = input_w_start[None, :] + tl.arange(0, kernel_width) * dilation_w
            
            # Apply padding and dilation constraints
            input_h_valid = (input_h >= 0) & (input_h < input_height)
            input_w_valid = (input_w >= 0) & (input_w < input_width)
            
            # Create combined mask
            h_mask = input_h_valid & out_h_mask[:, None]
            w_mask = input_w_valid & out_w_mask[None, :]
            valid_mask = h_mask & w_mask
            
            # Load weights
            weight = tl.load(weight_ptr + 
                           (c * out_channels + tl.arange(0, BLOCK_SIZE_K)[None, :]) * 
                           (kernel_height * kernel_width * kernel_depth) +
                           (k * kernel_height * kernel_width))
            
            # Load input data
            input_data = tl.load(input_ptr + 
                               (c + tl.arange(0, BLOCK_SIZE_K)[None, :]) * 
                               (input_height * input_width * input_depth) +
                               (input_h[:, None, :, None] * input_width * input_depth + 
                                input_w[None, :, None, :] * input_depth + 
                                k * input_depth))
            
            # Accumulate
            acc += tl.dot(input_data, weight)
    
    # Write back the result
    output_row = start_m + tl.arange(0, BLOCK_SIZE_M)
    output_col = start_n + tl.arange(0, BLOCK_SIZE_N)
    
    # Bounds checking for output indices
    row_mask = output_row < output_height
    col_mask = output_col < output_width
    
    # Write output
    tl.store(output_ptr + 
             (output_row[:, None] * output_width + output_col[None, :]),
             acc, mask=(row_mask[:, None] & col_mask[None, :]))

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
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width, depth = x.shape
        
        # Set up parameters
        kernel_height = self.kernel_size
        kernel_width = self.kernel_size
        kernel_depth = 1
        stride_h = self.stride
        stride_w = self.stride
        stride_d = 1
        padding_h = self.padding
        padding_w = self.padding
        padding_d = 0
        dilation_h = self.dilation
        dilation_w = self.dilation
        dilation_d = 1
        
        # Calculate output dimensions
        output_height = (height + 2 * padding_h - (dilation_h * (kernel_height - 1) + 1)) // stride_h + 1
        output_width = (width + 2 * padding_w - (dilation_w * (kernel_width - 1) + 1)) // stride_w + 1
        output_depth = (depth + 2 * padding_d - (dilation_d * (kernel_depth - 1) + 1)) // stride_d + 1
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_channels, output_height, output_width, output_depth, device=x.device, dtype=torch.float32)
        
        # Convert to contiguous tensors
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Configure kernel launch parameters
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16
        BLOCK_SIZE_K = 32
        GROUP_SIZE_M = 8
        
        # Calculate grid size
        grid = lambda meta: (
            (output_height + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"] *
            (output_width + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"]
        )
        
        # Launch kernel
        conv3d_kernel[grid](
            x, weight, output,
            batch_size, self.in_channels, self.out_channels,
            height, width, depth,
            output_height, output_width, output_depth,
            kernel_height, kernel_width, kernel_depth,
            stride_h, stride_w, stride_d,
            padding_h, padding_w, padding_d,
            dilation_h, dilation_w, dilation_d,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M
        )
        
        # Add bias if present
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
            
        return output