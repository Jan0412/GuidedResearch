import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def im2col_kernel(
    x_ptr,  # Input tensor pointer (B, C, H, W)
    col_ptr,  # Output column tensor pointer (B, C, kH, kW, H_out, W_out)
    B, C, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    H_out, W_out,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Calculate batch, channel, and spatial indices
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)
    h_idx = tl.program_id(2)
    w_idx = tl.program_id(3)
    
    # Calculate kernel positions
    kh = tl.program_id(4)  # kernel height
    kw = tl.program_id(5)  # kernel width
    
    # Calculate input coordinates
    in_h = h_idx * stride_h - pad_h + kh * dilation_h
    in_w = w_idx * stride_w - pad_w + kw * dilation_w
    
    # Check bounds
    mask = (
        (b_idx < B) & 
        (c_idx < C) & 
        (h_idx < H_out) & 
        (w_idx < W_out) &
        (kh < kH) &
        (kw < kW)
    )
    
    if mask:
        # Calculate input pointer offset
        if (in_h >= 0) and (in_h < H) and (in_w >= 0) and (in_w < W):
            offset = b_idx * C * H * W + c_idx * H * W + in_h * W + in_w
            x_val = tl.load(x_ptr + offset)
        else:
            x_val = 0.0  # Padding value
            
        # Calculate output pointer offset
        offset_out = (
            b_idx * C * kH * kW * H_out * W_out +
            c_idx * kH * kW * H_out * W_out +
            kh * kW * H_out * W_out +
            kw * H_out * W_out +
            h_idx * W_out +
            w_idx
        )
        tl.store(col_ptr + offset_out, x_val)


@triton.jit
def conv2d_matmul_kernel(
    col_ptr,  # Im2col output (B, C*kH*kW, H_out*W_out)
    weight_ptr,  # Weight tensor (out_channels, in_channels*kH*kW)
    bias_ptr,  # Bias tensor (out_channels,)
    out_ptr,  # Output tensor (B, out_channels, H_out, W_out)
    B, C, kH, kW, H_out, W_out, out_channels,
    USE_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Define block sizes for matrix multiplication
    # M: output channels, N: spatial positions, K: input channels * kernel size
    
    # Calculate output channel and spatial position
    m = tl.program_id(0)  # output channel index
    n = tl.program_id(1)  # spatial position index
    
    # Accumulator for the dot product
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over K dimension (C * kH * kW)
    for k in range(0, C * kH * kW, BLOCK_SIZE_K):
        # Load weight block: (out_channels, K)
        k_offset = k + tl.arange(0, BLOCK_SIZE_K)
        weight_mask = (tl.arange(0, BLOCK_SIZE_M) < out_channels)[:, None] & (k_offset < C * kH * kW)[None, :]
        weight_block = tl.load(
            weight_ptr + m * (C * kH * kW) + k_offset,
            mask=(tl.arange(0, BLOCK_SIZE_M) < out_channels),
            other=0.0
        )
        
        # Load column block: (K, N) - but we need to handle this differently
        # Actually, col_ptr is organized as (B, C*kH*kW, H_out*W_out), so we need to handle it per batch
        # This kernel will be launched per batch element, so adjust indexing
        
    # For simplicity, let's implement a more straightforward version
    # This kernel will be called per batch element, so:
    # m: output channel, n: spatial position (n from 0 to H_out*W_out-1)
    
    # We'll use a different approach - compute dot product for one output position
    if m < out_channels and n < H_out * W_out:
        acc = 0.0
        for c in range(C):
            for kh in range(kH):
                for kw in range(kW):
                    k_idx = c * kH * kW + kh * kW + kw
                    # Load weight for this output channel and input combination
                    w_val = tl.load(weight_ptr + m * (C * kH * kW) + k_idx)
                    # Load column value
                    col_val = tl.load(col_ptr + m * (C * kH * kW) + k_idx * (H_out * W_out) + n)
                    acc += w_val * col_val
        
        # Add bias if enabled
        if USE_BIAS:
            b_val = tl.load(bias_ptr + m)
            acc += b_val
        
        # Store result
        # Convert n from flat index to (h, w) for output tensor
        out_h = n // W_out
        out_w = n % W_out
        out_offset = m * (H_out * W_out) + out_h * W_out + out_w
        tl.store(out_ptr + out_offset, acc)


@triton.jit
def conv2d_fused_kernel(
    x_ptr,  # Input tensor (B, C, H, W)
    weight_ptr,  # Weight tensor (out_channels, C, kH, kW)
    bias_ptr,  # Bias tensor (out_channels,)
    out_ptr,  # Output tensor (B, out_channels, H_out, W_out)
    B, C, H, W, out_channels, kH, kW, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    H_out, W_out,
    USE_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_OUT: tl.constexpr,
):
    # Batch and output channel indices
    b_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1)
    
    # Spatial position within output feature map
    h_idx = tl.program_id(2)
    w_idx = tl.program_id(3)
    
    # Compute output value
    if b_idx < B and out_c_idx < out_channels and h_idx < H_out and w_idx < W_out:
        sum_val = 0.0
        
        # Compute convolution sum
        for c in range(C):
            for kh in range(kH):
                for kw in range(kW):
                    # Calculate input position
                    in_h = h_idx * stride_h + kh * dilation_h - pad_h
                    in_w = w_idx * stride_w + kw * dilation_w - pad_w
                    
                    # Check bounds
                    if (in_h >= 0 and in_h < H and in_w >= 0 and in_w < W):
                        # Input offset: (b, c, in_h, in_w)
                        x_offset = b_idx * C * H * W + c * H * W + in_h * W + in_w
                        x_val = tl.load(x_ptr + x_offset)
                        
                        # Weight offset: (out_c, c, kh, kw)
                        w_offset = out_c_idx * C * kH * kW + c * kH * kW + kh * kW + kw
                        w_val = tl.load(weight_ptr + w_offset)
                        
                        sum_val += x_val * w_val
        
        # Add bias if enabled
        if USE_BIAS:
            bias_val = tl.load(bias_ptr + out_c_idx)
            sum_val += bias_val
        
        # Store output
        out_offset = b_idx * out_channels * H_out * W_out + out_c_idx * H_out * W_out + h_idx * W_out + w_idx
        tl.store(out_ptr + out_offset, sum_val)


def triton_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Triton implementation of 2D convolution.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Extract dimensions
    B, C, H, W = x.shape
    out_channels, _, kH, kW = weight.shape
    
    # Calculate output dimensions
    H_out = (H + 2 * padding - dilation * (kH - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (kW - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(B, out_channels, H_out, W_out, device=x.device, dtype=x.dtype)
    
    # Check if bias is provided
    use_bias = bias is not None
    if use_bias:
        bias = bias.contiguous()
    
    # Set block sizes (tunable parameters)
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    BLOCK_SIZE_C = 16
    BLOCK_SIZE_OUT = 16
    
    # Grid configuration
    grid = lambda meta: (
        B,
        triton.cdiv(out_channels, meta['BLOCK_SIZE_OUT']),
        triton.cdiv(H_out, meta['BLOCK_SIZE_H']),
        triton.cdiv(W_out, meta['BLOCK_SIZE_W'])
    )
    
    # Launch the fused convolution kernel
    conv2d_fused_kernel[grid](
        x, weight, bias, out,
        B, C, H, W, out_channels, kH, kW,
        stride, stride, padding, padding, dilation, dilation,
        H_out, W_out,
        USE_BIAS=use_bias,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_OUT=BLOCK_SIZE_OUT
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized Model using Triton kernel for 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters similar to the original Conv2d
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.bias_flag = bias
        
        # Initialize weights and bias similar to nn.Conv2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        # Kaiming initialization similar to nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 2D convolution using Triton kernel.
        """
        return triton_conv2d(x, self.weight, self.bias, 
                            stride=self.stride, 
                            padding=self.padding, 
                            dilation=self.dilation, 
                            groups=self.groups)