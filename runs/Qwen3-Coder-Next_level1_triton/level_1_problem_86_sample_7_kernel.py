import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def depthwise_pointwise_conv2d_kernel(
    x_ptr,  # Input tensor (B, C_in, H, W)
    depthwise_weight_ptr,  # Depthwise weights (C_in, 1, K_h, K_w)
    pointwise_weight_ptr,  # Pointwise weights (C_out, C_in, 1, 1)
    bias_ptr,  # Bias (C_out,) or None
    out_ptr,  # Output tensor (B, C_out, H_out, W_out)
    B, C_in, C_out, 
    H_in, W_in,
    H_out, W_out,
    K_h: tl.constexpr, K_w: tl.constexpr,
    stride: tl.constexpr, padding: tl.constexpr, dilation: tl.constexpr,
    BLOCK_SIZE_C_OUT: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate output position
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Compute input position corresponding to this output position
    in_h = out_h * stride - padding + dilation * tl.arange(0, BLOCK_SIZE_H)[:, None]
    in_w = out_w * stride - padding + dilation * tl.arange(0, BLOCK_SIZE_W)[None, :]
    
    # Create masks for valid input positions
    h_mask = (in_h >= 0) & (in_h < H_in)
    w_mask = (in_w >= 0) & (in_w < W_in)
    
    # Initialize accumulator for each output channel
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Process depthwise convolution for each input channel
    for c_in in range(C_in):
        # Get depthwise kernel weights for this channel
        kernel_offset = c_in * (K_h * K_w)
        # We'll compute the depthwise result for this channel first
        depthwise_result = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
        
        # Iterate over kernel positions
        for kh in range(K_h):
            for kw in range(K_w):
                # Calculate input position for this kernel element
                pos_h = in_h + kh * dilation
                pos_w = in_w + kw * dilation
                
                # Create masks for valid positions
                kh_mask = (pos_h >= 0) & (pos_h < H_in)
                kw_mask = (pos_w >= 0) & (pos_w < W_in)
                mask = kh_mask & kw_mask
                
                # Load input values
                x_offset = pid_b * (C_in * H_in * W_in) + c_in * (H_in * W_in) + pos_h * W_in + pos_w
                # For simplicity, we'll use a scatter-gather approach
                # In practice, we'd want more sophisticated memory access patterns
                
                # For now, use a simpler approach: process one element at a time
                # This is not optimal but shows the concept
                
                # Get the weight for this kernel position
                weight_offset = kernel_offset + kh * K_w + kw
                kernel_val = tl.load(depthwise_weight_ptr + weight_offset)
                
                # Load input value (we need to handle the 2D indexing properly)
                # This is simplified - in practice, we'd need more complex indexing
                
        # Add depthwise result to accumulator
        acc += depthwise_result * tl.load(pointwise_weight_ptr + pid_c_out * C_in + c_in)
    
    # Add bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + pid_c_out * BLOCK_SIZE_C_OUT + tl.arange(0, BLOCK_SIZE_H))
        acc += bias[:, None]
    
    # Store output
    out_offset = pid_b * (C_out * H_out * W_out) + pid_c_out * (H_out * W_out) + out_h * W_out + out_w
    tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty), mask=(out_h < H_out) & (out_w < W_out))


# Simpler approach: implement depthwise and pointwise convolutions separately with optimized kernels


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Input tensor (B, C, H, W)
    weight_ptr,  # Depthwise weights (C, 1, K_h, K_w)
    bias_ptr,  # Bias (C,) or None
    out_ptr,  # Output tensor (B, C, H_out, W_out)
    B, C, 
    H_in, W_in,
    H_out, W_out,
    K_h: tl.constexpr, K_w: tl.constexpr,
    stride: tl.constexpr, padding: tl.constexpr, dilation: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate output position
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Compute input position corresponding to this output position
    in_h_start = out_h * stride - padding
    in_w_start = out_w * stride - padding
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Iterate over kernel positions
    for kh in range(K_h):
        for kw in range(K_w):
            # Calculate input position for this kernel element
            pos_h = in_h_start + kh * dilation
            pos_w = in_w_start + kw * dilation
            
            # Create masks for valid positions
            h_mask = (pos_h >= 0) & (pos_h < H_in)
            w_mask = (pos_w >= 0) & (pos_w < W_in)
            mask = h_mask[:, None] & w_mask[None, :]
            
            # Load input values
            x_offset = pid_b * (C * H_in * W_in) + pid_c * (H_in * W_in) + pos_h[:, None] * W_in + pos_w[None, :]
            x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
            
            # Get the weight for this kernel position
            weight_offset = pid_c * (K_h * K_w) + kh * K_w + kw
            kernel_val = tl.load(weight_ptr + weight_offset)
            
            # Accumulate
            acc += x_val * kernel_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + pid_c)
        acc += bias
    
    # Store output
    out_offset = pid_b * (C * H_out * W_out) + pid_c * (H_out * W_out) + out_h[:, None] * W_out + out_w[None, :]
    tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty), mask=(out_h[:, None] < H_out) & (out_w[None, :] < W_out))


@triton.jit
def pointwise_conv2d_kernel(
    x_ptr,  # Input tensor (B, C_in, H, W)
    weight_ptr,  # Pointwise weights (C_out, C_in, 1, 1)
    bias_ptr,  # Bias (C_out,) or None
    out_ptr,  # Output tensor (B, C_out, H, W)
    B, C_in, C_out,
    H, W,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr, BLOCK_SIZE_C: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate output position
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Process input channels in blocks
    for c_start in range(0, C_in, BLOCK_SIZE_C):
        c_offsets = c_start + tl.arange(0, BLOCK_SIZE_C)
        
        # Create masks for valid channels
        c_mask = c_offsets < C_in
        
        # Load input values for this block of channels
        x_offset = pid_b * (C_in * H * W) + c_offsets[:, None, None] * (H * W) + out_h[None, :, None] * W + out_w[None, None, :]
        x_val = tl.load(x_ptr + x_offset, mask=c_mask[:, None, None], other=0.0)
        
        # Load weight values for this block of channels
        weight_offset = pid_c_out * C_in + c_offsets[:, None]
        w_val = tl.load(weight_ptr + weight_offset, mask=c_mask[:, None], other=0.0)
        
        # Accumulate
        acc += tl.sum(x_val * w_val[:, None, None], axis=0)
    
    # Add bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + pid_c_out)
        acc += bias
    
    # Store output
    out_offset = pid_b * (C_out * H * W) + pid_c_out * (H * W) + out_h[:, None] * W + out_w[None, :]
    tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty), mask=(out_h[:, None] < H) & (out_w[None, :] < W))


def triton_depthwise_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """
    Triton implementation of depthwise 2D convolution.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        weight: Depthwise weight tensor of shape (C, 1, K_h, K_w)
        bias: Optional bias tensor of shape (C,)
        stride, padding, dilation: Convolution parameters
    
    Returns:
        Output tensor of shape (B, C, H_out, W_out)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    B, C, H_in, W_in = x.shape
    _, _, K_h, K_w = weight.shape
    
    # Calculate output dimensions
    H_out = (H_in + 2 * padding - dilation * (K_h - 1) - 1) // stride + 1
    W_out = (W_in + 2 * padding - dilation * (K_w - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(B, C, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Set up grid
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    
    grid = lambda meta: (
        B,
        C,
        triton.cdiv(H_out, BLOCK_SIZE_H),
        triton.cdiv(W_out, BLOCK_SIZE_W)
    )
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        x, weight, bias, out,
        B, C,
        H_in, W_in,
        H_out, W_out,
        K_h, K_w,
        stride, padding, dilation,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W
    )
    
    return out


def triton_pointwise_conv2d(x, weight, bias=None):
    """
    Triton implementation of pointwise 2D convolution (1x1 convolution).
    
    Args:
        x: Input tensor of shape (B, C_in, H, W)
        weight: Pointwise weight tensor of shape (C_out, C_in, 1, 1)
        bias: Optional bias tensor of shape (C_out,)
    
    Returns:
        Output tensor of shape (B, C_out, H, W)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    B, C_in, H, W = x.shape
    C_out, _, _, _ = weight.shape
    
    # Prepare output tensor
    out = torch.empty(B, C_out, H, W, dtype=x.dtype, device=x.device)
    
    # Set up grid
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    BLOCK_SIZE_C = 16  # Tune this for performance
    
    grid = lambda meta: (
        B,
        C_out,
        triton.cdiv(H, BLOCK_SIZE_H),
        triton.cdiv(W, BLOCK_SIZE_W)
    )
    
    # Launch kernel
    pointwise_conv2d_kernel[grid](
        x, weight, bias, out,
        B, C_in, C_out,
        H, W,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a depthwise-separable 2D convolution operation using Triton kernels.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Use the same parameters as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias_flag = bias
        
        # Create parameters that match the original model's structure
        # But we'll override the forward method to use Triton kernels
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
        # Replace the parameters with our own to maintain compatibility
        # but we'll ignore the original implementation in forward()
        with torch.no_grad():
            # Copy original weights to maintain initialization
            depthwise_weight = self.depthwise.weight.clone()
            pointwise_weight = self.pointwise.weight.clone()
            depthwise_bias = self.depthwise.bias.clone() if self.depthwise.bias is not None else None
            pointwise_bias = self.pointwise.bias.clone() if self.pointwise.bias is not None else None
            
            # Create new parameters that will be used in the Triton kernels
            self.depthwise_weight = nn.Parameter(depthwise_weight)
            self.pointwise_weight = nn.Parameter(pointwise_weight)
            if bias:
                self.depthwise_bias = nn.Parameter(depthwise_bias) if depthwise_bias is not None else None
                self.pointwise_bias = nn.Parameter(pointwise_bias) if pointwise_bias is not None else None
            else:
                self.depthwise_bias = None
                self.pointwise_bias = None
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise-separable 2D convolution using Triton kernels.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Use the depthwise convolution with Triton kernel
        x = triton_depthwise_conv2d(x, self.depthwise_weight, self.depthwise_bias, 
                                   self.stride, self.padding, self.dilation)
        
        # Use the pointwise convolution with Triton kernel
        x = triton_pointwise_conv2d(x, self.pointwise_weight, self.pointwise_bias)
        
        return x