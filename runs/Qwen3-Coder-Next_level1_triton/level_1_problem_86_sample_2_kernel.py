import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Input tensor: (batch, channels, height, width)
    weight_ptr,  # Weight tensor: (channels, 1, kh, kw)
    bias_ptr,  # Bias tensor: (channels,) or None
    output_ptr,  # Output tensor: (batch, channels, out_h, out_w)
    batch_size, in_channels, out_h, out_w,
    kh, kw, stride, padding, dilation,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_KH: tl.constexpr, BLOCK_KW: tl.constexpr,
):
    # Batch and channel indices
    b = tl.program_id(0)
    c = tl.program_id(1)
    
    # Output position
    oh = tl.program_id(2)
    ow = tl.program_id(3)
    
    # Compute input position
    ih = oh * stride - padding
    iw = ow * stride - padding
    
    # Accumulator
    acc = 0.0
    
    # Load bias if present
    if bias_ptr is not None:
        acc = tl.load(bias_ptr + c)
    
    # Iterate over kernel elements
    for kh_idx in range(0, kh, BLOCK_KH):
        for kw_idx in range(0, kw, BLOCK_KW):
            # Compute input position with kernel offset
            h_idx = ih + kh_idx * dilation
            w_idx = iw + kw_idx * dilation
            
            # Check bounds
            if (0 <= h_idx < out_h * stride + padding * 2) and (0 <= w_idx < out_w * stride + padding * 2):
                # Input index
                in_idx = b * (in_channels * out_h * stride * out_w * stride) + \
                         c * (out_h * stride * out_w * stride) + \
                         h_idx * (out_w * stride) + w_idx
                
                # Weight index
                w_idx_kernel = c * (1 * kh * kw) + 0 * (kh * kw) + kh_idx * kw + kw_idx
                
                # Load values
                x_val = tl.load(x_ptr + in_idx, mask=(h_idx >= 0) & (h_idx < out_h * stride + padding * 2) & 
                               (w_idx >= 0) & (w_idx < out_w * stride + padding * 2), other=0.0)
                w_val = tl.load(weight_ptr + w_idx_kernel)
                
                # Accumulate
                acc += x_val * w_val
    
    # Store result
    out_idx = b * (in_channels * out_h * out_w) + c * (out_h * out_w) + oh * out_w + ow
    tl.store(output_ptr + out_idx, acc)


@triton.jit
def depthwise_conv2d_fused_kernel(
    x_ptr,  # Input tensor: (batch, channels, height, width)
    weight_ptr,  # Weight tensor: (channels, 1, kh, kw)
    bias_ptr,  # Bias tensor: (channels,) or None
    output_ptr,  # Output tensor: (batch, channels, out_h, out_w)
    batch_size, in_channels, out_h, out_w,
    kh, kw, stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute global indices
    total_threads = batch_size * in_channels * out_h * out_w
    pid = tl.program_id(0)
    
    if pid >= total_threads:
        return
    
    # Decode indices
    tmp = pid
    ow = tmp % out_w
    tmp //= out_w
    oh = tmp % out_h
    tmp //= out_h
    c = tmp % in_channels
    b = tmp // in_channels
    
    # Compute input position
    ih = oh * stride - padding
    iw = ow * stride - padding
    
    # Accumulator
    acc = 0.0
    
    # Load bias if present
    if bias_ptr is not None:
        acc = tl.load(bias_ptr + c)
    
    # Iterate over kernel elements
    for kh_idx in range(kh):
        for kw_idx in range(kw):
            # Compute input position with kernel offset
            h_idx = ih + kh_idx * dilation
            w_idx = iw + kw_idx * dilation
            
            # Check bounds
            valid = (h_idx >= 0) & (h_idx < x_ptr.shape[2]) & (w_idx >= 0) & (w_idx < x_ptr.shape[3])
            
            # Input index
            in_idx = b * (in_channels * x_ptr.shape[2] * x_ptr.shape[3]) + \
                     c * (x_ptr.shape[2] * x_ptr.shape[3]) + \
                     h_idx * x_ptr.shape[3] + w_idx
            
            # Weight index
            w_idx_kernel = c * (1 * kh * kw) + 0 * (kh * kw) + kh_idx * kw + kw_idx
            
            # Load values with masking
            x_val = tl.load(x_ptr + in_idx, mask=valid, other=0.0)
            w_val = tl.load(weight_ptr + w_idx_kernel)
            
            # Accumulate
            acc += x_val * w_val
    
    # Store result
    out_idx = b * (in_channels * out_h * out_w) + c * (out_h * out_w) + oh * out_w + ow
    tl.store(output_ptr + out_idx, acc)


@triton.jit
def pointwise_conv2d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, height, width)
    weight_ptr,  # Weight tensor: (out_channels, in_channels, 1, 1)
    bias_ptr,  # Bias tensor: (out_channels,) or None
    output_ptr,  # Output tensor: (batch, out_channels, height, width)
    batch_size, in_channels, out_channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    # Batch, output channel, height, width indices
    b = tl.program_id(0)
    oc = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)
    
    # Accumulator
    acc = 0.0
    
    # Load bias if present
    if bias_ptr is not None:
        acc = tl.load(bias_ptr + oc)
    
    # Iterate over input channels
    for ic in range(in_channels):
        # Input index
        in_idx = b * (in_channels * height * width) + \
                 ic * (height * width) + \
                 h * width + w
        
        # Weight index for pointwise conv (1x1)
        w_idx_kernel = oc * (in_channels * 1 * 1) + ic * (1 * 1) + 0 * 1 + 0
        
        # Load values
        x_val = tl.load(x_ptr + in_idx)
        w_val = tl.load(weight_ptr + w_idx_kernel)
        
        # Accumulate
        acc += x_val * w_val
    
    # Store result
    out_idx = b * (out_channels * height * width) + \
              oc * (height * width) + \
              h * width + w
    tl.store(output_ptr + out_idx, acc)


def triton_depthwise_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """Depthwise convolution with Triton kernel"""
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, height, width = x.shape
    _, _, kh, kw = weight.shape
    
    # Compute output dimensions
    out_h = (height + 2 * padding - dilation * (kh - 1) - 1) // stride + 1
    out_w = (width + 2 * padding - dilation * (kw - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, in_channels, out_h, out_w, device=x.device, dtype=x.dtype)
    
    # Grid configuration: (batch, channels, out_h, out_w)
    total_threads = batch_size * in_channels * out_h * out_w
    BLOCK_SIZE = 128
    grid = ((total_threads + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Launch kernel
    depthwise_conv2d_fused_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_h, out_w,
        kh, kw, stride, padding, dilation,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def triton_pointwise_conv2d(x, weight, bias=None):
    """Pointwise convolution (1x1) with Triton kernel"""
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, height, width = x.shape
    out_channels = weight.shape[0]
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, height, width, device=x.device, dtype=x.dtype)
    
    # Grid configuration: (batch, out_channels, height, width)
    grid = (batch_size, out_channels, height, width)
    
    # Launch kernel
    pointwise_conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, height, width,
        BLOCK_SIZE=128,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized depthwise-separable 2D convolution using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Depthwise layer: groups=in_channels means each input channel is convolved separately
        self.depthwise_weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        if bias:
            self.depthwise_bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('depthwise_bias', None)
        
        # Pointwise layer: 1x1 convolution to combine channels
        self.pointwise_weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
        if bias:
            self.pointwise_bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('pointwise_bias', None)
        
        # Store hyperparameters
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.depthwise_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pointwise_weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.depthwise_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.depthwise_bias, -bound, bound)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.pointwise_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.pointwise_bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise-separable 2D convolution using Triton kernels.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Depthwise convolution
        x = triton_depthwise_conv2d(x, self.depthwise_weight, self.depthwise_bias, 
                                   self.stride, self.padding, self.dilation)
        
        # Pointwise convolution
        x = triton_pointwise_conv2d(x, self.pointwise_weight, self.pointwise_bias)
        
        return x