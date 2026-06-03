import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def im2col_kernel(
    x_ptr,  # Input tensor pointer (N, C, H, W)
    output_ptr,  # Output im2col tensor pointer (N * H_out * W_out, C * K_h * K_w)
    batch_size,  # N
    in_channels,  # C
    height,  # H
    width,  # W
    kernel_height,  # K_h
    kernel_width,  # K_w
    stride_h,  # stride_h
    stride_w,  # stride_w
    pad_h,  # padding_h
    pad_w,  # padding_w
    dil_h,  # dilation_h
    dil_w,  # dilation_w
    out_h,  # output height
    out_w,  # output width
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output rows
    BLOCK_SIZE_N: tl.constexpr,  # Block size for output columns
):
    # Calculate the global row and column indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Total number of columns in im2col output
    total_cols = in_channels * kernel_height * kernel_width
    
    # Compute the start row and column for this block
    row_start = pid_m * BLOCK_SIZE_M
    col_start = pid_n * BLOCK_SIZE_N
    
    # Create row offsets
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    # Create column offsets
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Mask to ensure we don't go out of bounds
    mask_rows = row_offsets < batch_size * out_h * out_w
    mask_cols = col_offsets < total_cols
    
    # Create a meshgrid for the row and column offsets
    row_idx, col_idx = tl.meshgrid(row_offsets, col_offsets, indexing='ij')
    mask = mask_rows[:, None] & mask_cols[None, :]
    
    # Compute the corresponding input indices
    # For each (row, col) in the im2col output, find the corresponding input location
    batch_idx = row_idx // (out_h * out_w)
    temp = row_idx % (out_h * out_w)
    out_h_idx = temp // out_w
    out_w_idx = temp % out_w
    
    # Calculate kernel offsets
    kernel_h_idx = col_idx // (kernel_width * in_channels)
    temp2 = col_idx % (kernel_width * in_channels)
    kernel_w_idx = temp2 // in_channels
    channel_idx = temp2 % in_channels
    
    # Calculate input coordinates
    in_h = out_h_idx * stride_h + kernel_h_idx * dil_h - pad_h
    in_w = out_w_idx * stride_w + kernel_w_idx * dil_w - pad_w
    
    # Calculate input pointer offset
    input_idx = batch_idx * (in_channels * height * width) + \
                channel_idx * (height * width) + \
                in_h * width + in_w
    
    # Create mask for valid input coordinates
    valid_input = (in_h >= 0) & (in_h < height) & (in_w >= 0) & (in_w < width)
    
    # Load data from input tensor
    data = tl.load(x_ptr + input_idx, mask=mask & valid_input, other=0.0)
    
    # Store to output tensor
    tl.store(output_ptr + row_idx * total_cols + col_idx, data, mask=mask)


@triton.jit
def matmul_kernel(
    a_ptr,  # im2col input (N * H_out * W_out, C * K_h * K_w)
    b_ptr,  # weights (out_channels, C * K_h * K_w)
    bias_ptr,  # bias (out_channels,)
    output_ptr,  # output (N, out_channels, H_out, W_out)
    batch_size,  # N
    out_h,  # output height
    out_w,  # output width
    out_channels,  # output channels
    k_size,  # C * K_h * K_w
    has_bias: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for rows of A
    BLOCK_SIZE_N: tl.constexpr,  # Block size for columns of B
    BLOCK_SIZE_K: tl.constexpr,  # Block size for columns of A / rows of B
):
    # Calculate the global tile indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the start row and column for this block
    row_start = pid_m * BLOCK_SIZE_M
    col_start = pid_n * BLOCK_SIZE_N
    
    # Create row and column offsets
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks
    mask_rows = row_offsets < batch_size * out_h * out_w
    mask_cols = col_offsets < out_channels
    
    # Create a meshgrid for the row and column offsets
    row_idx, col_idx = tl.meshgrid(row_offsets, col_offsets, indexing='ij')
    mask = mask_rows[:, None] & mask_cols[None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, k_size, BLOCK_SIZE_K):
        # Calculate K offsets
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offsets < k_size
        
        # Load A block
        a_offset = row_idx * k_size + k_offsets[None, :]
        a_mask = mask[:, :, None] & mask_k[None, None, :]
        a_block = tl.load(a_ptr + a_offset, mask=a_mask, other=0.0)
        
        # Load B block
        b_offset = col_idx[None, :] * k_size + k_offsets[:, None]
        b_mask = mask[None, :, :] & mask_k[:, None]
        b_block = tl.load(b_ptr + b_offset, mask=b_mask, other=0.0)
        
        # Matrix multiplication
        acc += tl.dot(a_block, b_block)
    
    # Convert accumulator to float32 if needed
    acc = acc.to(tl.float32)
    
    # Add bias if present
    if has_bias:
        bias_offsets = col_idx
        bias_mask = mask_cols
        bias_val = tl.load(bias_ptr + bias_offsets, mask=bias_mask, other=0.0)
        acc += bias_val[None, :]
    
    # Store result
    tl.store(output_ptr + row_idx * out_channels + col_idx, acc, mask=mask)


@triton.jit
def finalize_output_kernel(
    output_ptr,  # Output from matmul (N * H_out * W_out, out_channels)
    final_output_ptr,  # Final output tensor (N, out_channels, H_out, W_out)
    batch_size,
    out_channels,
    out_h,
    out_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Calculate the global tile indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the start row and column for this block
    row_start = pid_m * BLOCK_SIZE_M
    col_start = pid_n * BLOCK_SIZE_N
    
    # Create row and column offsets
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks
    mask_rows = row_offsets < batch_size * out_h * out_w
    mask_cols = col_offsets < out_channels
    
    # Create a meshgrid for the row and column offsets
    row_idx, col_idx = tl.meshgrid(row_offsets, col_idx, indexing='ij')
    mask = mask_rows[:, None] & mask_cols[None, :]
    
    # Load data
    data = tl.load(output_ptr + row_idx * out_channels + col_idx, mask=mask, other=0.0)
    
    # Reshape and store to final output tensor
    # Convert row_idx to (batch, h, w) and col_idx to channel
    batch_idx = row_idx // (out_h * out_w)
    temp = row_idx % (out_h * out_w)
    h_idx = temp // out_w
    w_idx = temp % out_w
    
    # Calculate final output offset
    final_offset = batch_idx * (out_channels * out_h * out_w) + \
                   col_idx * (out_h * out_w) + \
                   h_idx * out_w + w_idx
    
    tl.store(final_output_ptr + final_offset, data, mask=mask)


def triton_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> torch.Tensor:
    """
    Performs 2D convolution using Triton kernels.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, height, width)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_height, kernel_width)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride of the convolution
        padding: Padding applied to the input
        dilation: Spacing between kernel elements
        groups: Number of blocked connections
    
    Returns:
        Output tensor of shape (batch_size, out_channels, height_out, width_out)
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    out_h = (height + 2 * padding - dilation * (kernel_height - 1) - 1) // stride + 1
    out_w = (width + 2 * padding - dilation * (kernel_width - 1) - 1) // stride + 1
    
    # Prepare output tensor
    output = torch.empty((batch_size, out_channels, out_h, out_w), dtype=x.dtype, device=x.device)
    
    # Reshape weight for matrix multiplication: (out_channels, in_channels * kernel_height * kernel_width)
    weight_reshaped = weight.view(out_channels, in_channels * kernel_height * kernel_width)
    
    # Calculate im2col dimensions
    im2col_size = batch_size * out_h * out_w
    k_size = in_channels * kernel_height * kernel_width
    
    # Allocate im2col buffer
    im2col_buffer = torch.empty((im2col_size, k_size), dtype=x.dtype, device=x.device)
    
    # Set up kernel parameters
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimensions for im2col
    grid_im2col = (
        triton.cdiv(im2col_size, BLOCK_SIZE_M),
        triton.cdiv(k_size, BLOCK_SIZE_N)
    )
    
    # Launch im2col kernel
    im2col_kernel[grid_im2col](
        x, im2col_buffer,
        batch_size, in_channels, height, width,
        kernel_height, kernel_width,
        stride, stride, padding, padding,
        dilation, dilation,
        out_h, out_w,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    # Set up matmul grid
    grid_matmul = (
        triton.cdiv(im2col_size, BLOCK_SIZE_M),
        triton.cdiv(out_channels, BLOCK_SIZE_N)
    )
    
    # Launch matmul kernel
    matmul_kernel[grid_matmul](
        im2col_buffer, weight_reshaped,
        bias if bias is not None else None,
        output.view(-1, out_channels),  # Flatten to (im2col_size, out_channels)
        batch_size, out_h, out_w,
        out_channels, k_size,
        has_bias=bias is not None,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernels for 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Register the parameters as buffers since we'll use our own convolution implementation
        self.register_buffer('weight', torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.register_buffer('bias', torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernels.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv2d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups
        )


import math  # Add math import for initialization