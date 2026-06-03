import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def im2col_kernel(
    x_ptr,  # Input tensor pointer (N, C, H, W)
    col_ptr,  # Output column tensor pointer (N, H_out, W_out, C, K_h, K_w)
    n, c, h, w,  # Input dimensions
    h_out, w_out,  # Output dimensions
    k_h, k_w,  # Kernel dimensions
    stride_h, stride_w,  # Strides
    pad_h, pad_w,  # Padding
    dil_h, dil_w,  # Dilation
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Compute batch index
    n_idx = tl.program_id(0)
    # Compute output spatial indices
    h_out_idx = tl.program_id(1)
    w_out_idx = tl.program_id(2)
    
    # Compute the starting position in the input for this output position
    h_start = h_out_idx * stride_h - pad_h
    w_start = w_out_idx * stride_w - pad_w
    
    # Iterate over channels and kernel dimensions in blocks
    for c_off in range(0, c, BLOCK_SIZE_C):
        c_range = c_off + tl.arange(0, BLOCK_SIZE_C)
        c_mask = c_range < c
        
        for kh_off in range(0, k_h, BLOCK_SIZE_KH):
            kh_range = kh_off + tl.arange(0, BLOCK_SIZE_KH)
            kh_mask = kh_range < k_h
            
            for kw_off in range(0, k_w, BLOCK_SIZE_KW):
                kw_range = kw_off + tl.arange(0, BLOCK_SIZE_KW)
                kw_mask = kw_range < k_w
                
                # Compute actual input positions
                h_pos = h_start + kh_range[None, None] * dil_h
                w_pos = w_start + kw_range[None, None] * dil_w
                
                # Check bounds for input positions
                h_valid = (h_pos >= 0) & (h_pos < h)
                w_valid = (w_pos >= 0) & (w_pos < w)
                mask = c_mask[None, None] & h_valid & w_valid
                
                # Load from input tensor
                x_offset = n_idx * (c * h * w) + c_range[:, None, None] * (h * w) + h_pos[:, :, None] * w + w_pos[:, :, None]
                x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
                
                # Store to column tensor
                col_offset = (n_idx * h_out * w_out + h_out_idx * w_out + w_out_idx) * (c * k_h * k_w)
                col_offset += (c_range[:, None, None] * k_h * k_w + kh_range[None, :, None] * k_w + kw_range[None, None, :])
                tl.store(col_ptr + col_offset, x_val, mask=mask)


@triton.jit
def col2im_gemm_kernel(
    col_ptr,  # Column tensor pointer (N, H_out, W_out, C, K_h, K_w)
    w_ptr,  # Weight tensor pointer (out_c, in_c, k_h, k_w)
    b_ptr,  # Bias tensor pointer (out_c)
    out_ptr,  # Output tensor pointer (N, out_c, H_out, W_out)
    n, out_c, in_c, h_out, w_out, k_h, k_w,
    stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_OUT_C: tl.constexpr,
    BLOCK_SIZE_IN_C: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Compute batch index
    n_idx = tl.program_id(0)
    # Compute output channel index
    oc_idx = tl.program_id(1)
    
    # Load weight for this output channel
    w_offset = oc_idx * (in_c * k_h * k_w) + tl.arange(0, BLOCK_SIZE_IN_C)[:, None, None] * (k_h * k_w) + tl.arange(0, BLOCK_SIZE_KH)[None, :, None] * k_w + tl.arange(0, BLOCK_SIZE_KW)[None, None, :]
    w_mask = (tl.arange(0, BLOCK_SIZE_IN_C)[:, None, None] < in_c) & (tl.arange(0, BLOCK_SIZE_KH)[None, :, None] < k_h) & (tl.arange(0, BLOCK_SIZE_KW)[None, None, :] < k_w)
    w_val = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)
    
    # Accumulator for output
    acc = tl.zeros((BLOCK_SIZE_IN_C,), dtype=tl.float32)
    
    # Iterate over input channels and kernel positions
    for ic in range(in_c):
        for kh in range(k_h):
            for kw in range(k_w):
                # Compute input position
                h_start = tl.arange(0, BLOCK_SIZE_N)[:, None] * (h_out * w_out) + tl.arange(0, BLOCK_SIZE_N)[:, None] * 0  # Simplified
                # This is a simplified version; actual implementation would need more complex indexing
    
    # For simplicity, we'll use a different approach - we'll compute output directly
    # This kernel is a placeholder; the actual implementation would use im2col + GEMM


@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor pointer (N, C, H, W)
    w_ptr,  # Weight tensor pointer (out_c, in_c, k_h, k_w)
    b_ptr,  # Bias tensor pointer (out_c)
    out_ptr,  # Output tensor pointer (N, out_c, H_out, W_out)
    n, c, h, w, out_c, k_h, k_w, h_out, w_out,
    stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_OUT_C: tl.constexpr,
    BLOCK_SIZE_H_OUT: tl.constexpr,
    BLOCK_SIZE_W_OUT: tl.constexpr,
    BLOCK_SIZE_IN_C: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Get program indices
    n_idx = tl.program_id(0)
    oc_idx = tl.program_id(1)
    h_out_idx = tl.program_id(2)
    w_out_idx = tl.program_id(3)
    
    # Compute bias
    bias_val = 0.0
    if b_ptr is not tl.core.NULL_PTR:
        bias_val = tl.load(b_ptr + oc_idx)
    
    # Accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Iterate over input channels and kernel positions
    for ic in range(c):
        for kh in range(k_h):
            for kw in range(k_w):
                # Compute input position
                h_in = h_out_idx * stride_h + kh * dil_h - pad_h
                w_in = w_out_idx * stride_w + kw * dil_w - pad_w
                
                # Check bounds
                if (h_in >= 0) and (h_in < h) and (w_in >= 0) and (w_in < w):
                    # Compute offsets
                    x_offset = n_idx * (c * h * w) + ic * (h * w) + h_in * w + w_in
                    w_offset = oc_idx * (c * k_h * k_w) + ic * (k_h * k_w) + kh * k_w + kw
                    
                    # Load values
                    x_val = tl.load(x_ptr + x_offset)
                    w_val = tl.load(w_ptr + w_offset)
                    
                    # Accumulate
                    acc += x_val * w_val
    
    # Apply bias and store result
    acc += bias_val
    out_offset = n_idx * (out_c * h_out * w_out) + oc_idx * (h_out * w_out) + h_out_idx * w_out + w_out_idx
    tl.store(out_ptr + out_offset, acc)


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
    Performs 2D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, height, width)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_height, kernel_width)
        bias: Optional bias tensor of shape (out_channels,)
        stride, padding, dilation, groups: Convolution parameters
        
    Returns:
        Output tensor of shape (batch_size, out_channels, output_height, output_width)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    stride_h = stride_w = stride if isinstance(stride, int) else stride[1]
    pad_h = pad_w = padding if isinstance(padding, int) else padding[1]
    dil_h = dil_w = dilation if isinstance(dilation, int) else dilation[1]
    
    height_out = (height + 2 * pad_h - dil_h * (kernel_height - 1) - 1) // stride_h + 1
    width_out = (width + 2 * pad_w - dil_w * (kernel_width - 1) - 1) // stride_w + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, height_out, width_out, dtype=x.dtype, device=x.device)
    
    # Configure kernel launch parameters
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_OUT_C = 16
    BLOCK_SIZE_H_OUT = 8
    BLOCK_SIZE_W_OUT = 8
    
    grid = (
        batch_size,
        out_channels,
        height_out,
        width_out
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, height, width, out_channels, kernel_height, kernel_width,
        height_out, width_out,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_OUT_C=BLOCK_SIZE_OUT_C,
        BLOCK_SIZE_H_OUT=BLOCK_SIZE_H_OUT,
        BLOCK_SIZE_W_OUT=BLOCK_SIZE_W_OUT,
        BLOCK_SIZE_IN_C=1,  # Process one input channel at a time for simplicity
        BLOCK_SIZE_KH=1,    # Process one kernel height at a time for simplicity
        BLOCK_SIZE_KW=1     # Process one kernel width at a time for simplicity
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize weights and bias directly instead of using nn.Conv2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
        
        # Store convolution parameters
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights using Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using the optimized Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv2d(
            x, self.weight, self.bias,
            stride=self.stride, 
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )


import math