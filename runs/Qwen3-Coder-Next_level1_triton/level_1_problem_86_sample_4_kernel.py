import torch
import torch.nn as nn
import triton
import triton.language as tl

# Depthwise convolution kernel - optimized for per-channel processing
@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,              # Input tensor (B, C, H, W)
    w_ptr,              # Depthwise weights (C, 1, kH, kW)
    out_ptr,            # Output tensor (B, C, H_out, W_out)
    b_ptr,              # Bias tensor (C,) - can be None
    batch_size,         # B
    channels,           # C
    in_h,               # Input height
    in_w,               # Input width
    out_h,              # Output height
    out_w,              # Output width
    k_h,                # Kernel height
    k_w,                # Kernel width
    stride_h,           # Stride height
    stride_w,           # Stride width
    pad_h,              # Padding height
    pad_w,              # Padding width
    dil_h,              # Dilation height
    dil_w,              # Dilation width
    has_bias: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_KH: tl.constexpr,
    BLOCK_KW: tl.constexpr,
):
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate output position
    h_out = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w_out = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create mask for valid output positions
    mask_h = h_out < out_h
    mask_w = w_out < out_w
    mask = mask_h[:, None] & mask_w[None, :]
    
    # Compute input positions for this output
    h_in = h_out * stride_h - pad_h + tl.arange(0, BLOCK_SIZE_H)[:, None] * stride_h
    w_in = w_out * stride_w - pad_w + tl.arange(0, BLOCK_SIZE_W)[None, :] * stride_w
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Iterate over kernel dimensions
    for kh in range(0, k_h, BLOCK_KH):
        for kw in range(0, k_w, BLOCK_KW):
            # Compute input coordinates with dilation
            h_in_k = h_in + kh * dil_h
            w_in_k = w_in + kw * dil_w
            
            # Check if input coordinates are within bounds
            mask_in_h = (h_in_k >= 0) & (h_in_k < in_h)
            mask_in_w = (w_in_k >= 0) & (w_in_k < in_w)
            mask_in = mask_in_h & mask_in_w
            
            # Calculate input pointer offset
            offset_in = pid_batch * (channels * in_h * in_w) + \
                       pid_c * (in_h * in_w) + \
                       h_in_k * in_w + w_in_k
            
            # Load input values with bounds checking
            x_vals = tl.load(x_ptr + offset_in, 
                           mask=mask_in & mask, 
                           other=0.0)
            
            # Calculate weight pointer offset
            offset_w = pid_c * (k_h * k_w) + kh * k_w + kw
            
            # Load weight value (same for all positions in this kernel)
            w_val = tl.load(w_ptr + offset_w)
            
            # Accumulate
            acc += x_vals * w_val
    
    # Add bias if present
    if has_bias:
        bias_val = tl.load(b_ptr + pid_c)
        acc += bias_val
    
    # Store result
    offset_out = pid_batch * (channels * out_h * out_w) + \
                pid_c * (out_h * out_w) + \
                h_out[:, None] * out_w + w_out[None, :]
    
    tl.store(out_ptr + offset_out, acc.to(x_ptr.dtype.element_ty), mask=mask)

# Pointwise convolution kernel - optimized as a matrix multiplication
@triton.jit
def pointwise_conv2d_kernel(
    x_ptr,              # Input tensor (B, C, H, W)
    w_ptr,              # Pointwise weights (C_out, C, 1, 1)
    out_ptr,            # Output tensor (B, C_out, H, W)
    b_ptr,              # Bias tensor (C_out,) - can be None
    batch_size,         # B
    in_channels,        # C
    out_channels,       # C_out
    h,                  # Height (same as input)
    w,                  # Width (same as input)
    has_bias: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    
    # Compute output position
    c_out_start = pid_c_out * BLOCK_C_OUT
    c_out_range = tl.arange(0, BLOCK_C_OUT) + c_out_start
    mask_c_out = c_out_range < out_channels
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_C_OUT, h, w), dtype=tl.float32)
    
    # Iterate over input channels
    for c_in_start in range(0, in_channels, BLOCK_C_IN):
        c_in_range = tl.arange(0, BLOCK_C_IN) + c_in_start
        mask_c_in = c_in_range < in_channels
        
        # Calculate input pointer offset for current batch and channel
        offset_in = pid_b * (in_channels * h * w) + c_in_range[:, None, None] * (h * w)
        
        # Load input: shape (BLOCK_C_IN, h, w)
        x_vals = tl.load(x_ptr + offset_in, mask=mask_c_in[:, None, None], other=0.0)
        
        # Calculate weight pointer offset
        # w_ptr shape: (C_out, C, 1, 1) -> flatten to (C_out, C)
        offset_w = c_out_range[:, None, None] * in_channels + c_in_range[None, :, None, None]
        
        # Load weights: shape (BLOCK_C_OUT, BLOCK_C_IN)
        w_vals = tl.load(w_ptr + offset_w, mask=mask_c_out[:, None, None, None] & mask_c_in[None, :, None, None], other=0.0)
        
        # Accumulate: expand dimensions for broadcasting
        # x_vals: (BLOCK_C_IN, h, w)
        # w_vals: (BLOCK_C_OUT, BLOCK_C_IN, 1, 1)
        # result: (BLOCK_C_OUT, h, w)
        acc += tl.sum(w_vals * x_vals[None, :, :, :], axis=1)
    
    # Add bias if present
    if has_bias:
        bias_vals = tl.load(b_ptr + c_out_range, mask=mask_c_out)
        acc += bias_vals[:, None, None]
    
    # Store result
    offset_out = pid_b * (out_channels * h * w) + \
                c_out_range[:, None, None] * (h * w)
    
    tl.store(out_ptr + offset_out, acc.to(x_ptr.dtype.element_ty), mask=mask_c_out[:, None, None])

# Fused depthwise + pointwise convolution kernel (optional optimization)
@triton.jit
def fused_depthwise_pointwise_kernel(
    x_ptr,              # Input tensor (B, C_in, H, W)
    dw_ptr,             # Depthwise weights (C_in, 1, kH, kW)
    pw_ptr,             # Pointwise weights (C_out, C_in, 1, 1)
    b_ptr,              # Bias tensor (C_out,) - can be None
    out_ptr,            # Output tensor (B, C_out, H_out, W_out)
    batch_size,         # B
    in_channels,        # C_in
    out_channels,       # C_out
    in_h,               # Input height
    in_w,               # Input width
    out_h,              # Output height
    out_w,              # Output width
    k_h,                # Kernel height
    k_w,                # Kernel width
    stride_h,           # Stride height
    stride_w,           # Stride width
    pad_h,              # Padding height
    pad_w,              # Padding width
    dil_h,              # Dilation height
    dil_w,              # Dilation width
    has_bias: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Compute output position
    h_out = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    
    mask_h = h_out < out_h
    mask_w = w_out < out_w
    mask_hw = mask_h[:, None] & mask_w[None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Process each input channel
    for c_in_start in range(0, in_channels, 1):  # Can optimize with tiling over channels
        c_in = c_in_start
        
        # Depthwise convolution part
        depthwise_sum = 0.0
        for kh in range(k_h):
            for kw in range(k_w):
                h_in = h_out * stride_h - pad_h + kh * dil_h
                w_in = w_out * stride_w - pad_w + kw * dil_w
                
                mask_in_h = (h_in >= 0) & (h_in < in_h)
                mask_in_w = (w_in >= 0) & (w_in < in_w)
                mask_in = mask_in_h & mask_in_w
                
                # Calculate input offset
                offset_in = pid_b * (in_channels * in_h * in_w) + \
                           c_in * (in_h * in_w) + \
                           h_in * in_w + w_in
                
                x_val = tl.load(x_ptr + offset_in, mask=mask_in & mask_hw, other=0.0)
                
                # Calculate weight offset
                offset_dw = c_in * (k_h * k_w) + kh * k_w + kw
                w_dw = tl.load(dw_ptr + offset_dw)
                
                depthwise_sum += tl.sum(x_val * w_dw)
        
        # Pointwise convolution part
        offset_pw = pid_c_out * in_channels + c_in
        w_pw = tl.load(pw_ptr + offset_pw)
        acc += depthwise_sum * w_pw
    
    # Add bias if present
    if has_bias:
        bias_val = tl.load(b_ptr + pid_c_out)
        acc += bias_val
    
    # Store result
    offset_out = pid_b * (out_channels * out_h * out_w) + \
                pid_c_out * (out_h * out_w) + \
                h_out[:, None] * out_w + w_out[None, :]
    
    tl.store(out_ptr + offset_out, acc.to(x_ptr.dtype.element_ty), mask=mask_hw)

def triton_depthwise_pointwise(x, dw_weight, pw_weight, bias, stride, padding, dilation):
    """
    Performs depthwise-separable convolution using Triton kernels.
    
    Args:
        x: Input tensor (B, C_in, H, W)
        dw_weight: Depthwise weights (C_in, 1, kH, kW)
        pw_weight: Pointwise weights (C_out, C_in, 1, 1)
        bias: Bias tensor (C_out,) or None
        stride: Stride tuple (stride_h, stride_w)
        padding: Padding tuple (pad_h, pad_w)
        dilation: Dilation tuple (dil_h, dil_w)
    
    Returns:
        Output tensor (B, C_out, H_out, W_out)
    """
    # Extract parameters
    batch_size, in_channels, in_h, in_w = x.shape
    out_channels = pw_weight.shape[0]
    k_h, k_w = dw_weight.shape[2], dw_weight.shape[3]
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    
    # Calculate output dimensions
    out_h = (in_h + 2 * pad_h - dil_h * (k_h - 1) - 1) // stride_h + 1
    out_w = (in_w + 2 * pad_w - dil_w * (k_w - 1) - 1) // stride_w + 1
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    dw_weight = dw_weight.contiguous()
    pw_weight = pw_weight.contiguous()
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Determine if bias is present
    has_bias = bias is not None
    if has_bias:
        bias = bias.contiguous()
    
    # Choose the appropriate kernel based on size
    # For larger models, use separate kernels; for smaller ones, fused kernel might be better
    if in_channels <= 32 and out_channels <= 64 and k_h <= 5 and k_w <= 5:
        # Use fused kernel for smaller cases
        BLOCK_B = 1
        BLOCK_C_OUT = 16
        BLOCK_H = 8
        BLOCK_W = 8
        
        grid = (batch_size, out_channels, (out_h + BLOCK_H - 1) // BLOCK_H, (out_w + BLOCK_W - 1) // BLOCK_W)
        
        fused_depthwise_pointwise_kernel[grid](
            x, dw_weight, pw_weight, bias, out,
            batch_size, in_channels, out_channels,
            in_h, in_w, out_h, out_w,
            k_h, k_w, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            has_bias=has_bias,
            BLOCK_B=BLOCK_B,
            BLOCK_C_OUT=BLOCK_C_OUT,
            BLOCK_H=BLOCK_H,
            BLOCK_W=BLOCK_W,
        )
    else:
        # Use separate kernels for larger cases
        # Depthwise convolution
        BLOCK_H = 8
        BLOCK_W = 8
        BLOCK_KH = 2
        BLOCK_KW = 2
        
        grid_dw = (batch_size, in_channels, (out_h + BLOCK_H - 1) // BLOCK_H, (out_w + BLOCK_W - 1) // BLOCK_W)
        
        depthwise_conv2d_kernel[grid_dw](
            x, dw_weight, out, bias if has_bias else None,
            batch_size, in_channels, in_h, in_w, out_h, out_w,
            k_h, k_w, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            has_bias=False,  # Will add bias in pointwise
            BLOCK_SIZE_H=BLOCK_H,
            BLOCK_SIZE_W=BLOCK_W,
            BLOCK_KH=BLOCK_KH,
            BLOCK_KW=BLOCK_KW,
        )
        
        # Pointwise convolution (note: we need to modify out in-place to add bias)
        # For simplicity, use separate pointwise kernel without bias, then add bias separately
        BLOCK_C_OUT = 32
        
        grid_pw = (batch_size, (out_channels + BLOCK_C_OUT - 1) // BLOCK_C_OUT)
        
        pointwise_conv2d_kernel[grid_pw](
            out, pw_weight, out, bias if has_bias else None,
            batch_size, in_channels, out_channels, out_h, out_w,
            has_bias=has_bias,
            BLOCK_B=1,
            BLOCK_C_OUT=BLOCK_C_OUT,
            BLOCK_C_IN=8,
        )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized depthwise-separable 2D convolution using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.use_bias = bias
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to FP32 for Triton kernel if needed
        original_dtype = x.dtype
        if x.dtype != torch.float32:
            x = x.float()
        
        # Call Triton implementation
        out = triton_depthwise_pointwise(
            x, 
            self.depthwise.weight, 
            self.pointwise.weight, 
            self.bias,
            stride=(self.depthwise.stride[0], self.depthwise.stride[1]),
            padding=(self.depthwise.padding[0], self.depthwise.padding[1]),
            dilation=(self.depthwise.dilation[0], self.depthwise.dilation[1])
        )
        
        # Convert back to original dtype
        if out.dtype != original_dtype:
            out = out.to(original_dtype)
        
        return out