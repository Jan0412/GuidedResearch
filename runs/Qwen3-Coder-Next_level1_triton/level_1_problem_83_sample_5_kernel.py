import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv1d_kernel(
    x_ptr,  # Input tensor pointer
    w_ptr,  # Weight tensor pointer
    b_ptr,  # Bias pointer (can be None)
    out_ptr,  # Output tensor pointer
    batch_size,  # Batch size
    height,  # Height of input
    width,  # Width of input
    kernel_size,  # Kernel size (1D)
    stride,  # Stride
    padding,  # Padding
    dilation,  # Dilation
    n_cols,  # Number of output columns (width_out)
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID represents the batch index * height + row index
    pid = tl.program_id(0)
    batch_idx = pid // height
    row_idx = pid % height
    
    # Calculate starting position in the input
    start_col = tl.program_id(1) * BLOCK_SIZE
    
    # Compute output column indices for this block
    offsets = start_col + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    # Compute the starting position in the input for this output position
    # For each output column, we need to gather kernel_size input elements
    out_col = offsets
    
    # Compute input column positions considering stride, padding, and dilation
    input_col_start = out_col * stride - padding
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over kernel positions
    for k in range(kernel_size):
        # Compute input column index for this kernel position
        in_col = input_col_start + k * dilation
        
        # Check if this input column is within bounds
        valid = (in_col >= 0) & (in_col < width)
        
        # Compute input pointer offset for this batch, row, and column
        input_idx = batch_idx * (height * width) + row_idx * width + in_col
        input_idx = tl.where(valid, input_idx, 0)  # Avoid out-of-bounds access
        
        # Load input value (0 if out of bounds)
        x_val = tl.load(x_ptr + input_idx, mask=valid, other=0.0).to(tl.float32)
        
        # Load corresponding weight
        weight_idx = k  # Weight shape is (in_channels, kernel_size) for depthwise
        w_val = tl.load(w_ptr + weight_idx).to(tl.float32)
        
        # Accumulate
        acc += x_val * w_val
    
    # Add bias if provided
    if b_ptr is not None:
        bias_idx = batch_idx * height * width + row_idx * n_cols + offsets
        bias = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += bias
    
    # Store result
    out_idx = batch_idx * (height * n_cols) + row_idx * n_cols + offsets
    tl.store(out_ptr + out_idx, acc.to(x_ptr.dtype.element_ty), mask=mask)


def triton_depthwise_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """
    Performs depthwise 2D convolution with asymmetric kernel (kernel_size, 1)
    """
    batch_size, in_channels, height, width = x.shape
    kernel_size = weight.shape[1]  # weight shape: (in_channels, kernel_size)
    
    # Calculate output dimensions
    out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Ensure input is contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Prepare output tensor
    out = torch.empty(batch_size, in_channels, out_height, out_width, dtype=x.dtype, device=x.device)
    
    # Bias handling
    bias_ptr = bias.contiguous() if bias is not None else None
    
    # Grid configuration: one block per (batch, height) pair for rows, and columns per block
    BLOCK_SIZE = 128
    grid = lambda meta: (
        batch_size * height,
        triton.cdiv(out_width, meta["BLOCK_SIZE"])
    )
    
    # Launch kernel
    depthwise_conv1d_kernel[grid](
        x, weight, bias_ptr, out,
        batch_size, height, width, kernel_size,
        stride, padding, dilation, out_width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the depthwise 2D convolution using Triton kernel.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the weight and bias parameters
        self.weight = nn.Parameter(torch.empty(in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
        else:
            self.register_parameter('bias', None)
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in = kernel_size  # Only kernel_size dimension matters for depthwise
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_depthwise_conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation
        )


import math