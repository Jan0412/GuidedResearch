import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Input tensor: (B, C, H, W)
    w_ptr,  # Weight tensor: (C, 1, K_h, K_w)
    b_ptr,  # Bias tensor: (C,) or None
    y_ptr,  # Output tensor: (B, C, H_out, W_out)
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    K_h: tl.constexpr,
    K_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr = 8,
    BLOCK_SIZE_W: tl.constexpr = 8,
    BLOCK_SIZE_C: tl.constexpr = 1,
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate output position
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for valid output positions
    mask_h = out_h < H_out
    mask_w = out_w < W_out
    mask = mask_h[:, None] & mask_w[None, :]
    
    # Compute input starting position for this output
    in_h_start = out_h * stride_h - padding_h
    in_w_start = out_w * stride_w - padding_w
    
    # Accumulator for the convolution result
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over kernel height
    for kh in range(K_h):
        # Compute input height positions with dilation
        in_h = in_h_start + kh * dilation_h
        
        # Check if within input bounds
        mask_h_k = (in_h >= 0) & (in_h < H)
        
        # Loop over kernel width
        for kw in range(K_w):
            # Compute input width positions with dilation
            in_w = in_w_start + kw * dilation_w
            
            # Check if within input bounds
            mask_w_k = (in_w >= 0) & (in_w < W)
            
            # Combined mask for this kernel position
            mask_k = mask_h_k[:, None] & mask_w_k[None, :] & mask
            
            # Calculate input pointer offset
            # Input layout: (B, C, H, W) - contiguous in W dimension
            input_offset = (pid_batch * C * H * W + 
                          pid_c * H * W + 
                          in_h[:, None] * W + 
                          in_w[None, :])
            
            # Load input values (with masking)
            x_val = tl.load(x_ptr + input_offset, mask=mask_k, other=0.0)
            
            # Load weight value
            # Weight layout: (C, 1, K_h, K_w) - contiguous in W dimension
            weight_offset = (pid_c * K_h * K_w + 
                           kh * K_w + 
                           kw)
            w_val = tl.load(w_ptr + weight_offset)
            
            # Accumulate: x * w
            acc += x_val * w_val
    
    # Convert accumulator to output dtype if needed
    acc = acc.to(y_ptr.dtype.element_ty)
    
    # Add bias if present
    if has_bias:
        bias_val = tl.load(b_ptr + pid_c)
        acc += bias_val
    
    # Store output
    output_offset = (pid_batch * C * H_out * W_out + 
                   pid_c * H_out * W_out + 
                   out_h[:, None] * W_out + 
                   out_w[None, :])
    tl.store(y_ptr + output_offset, acc, mask=mask)


def depthwise_conv2d_triton(x, weight, bias=None, 
                           stride=(1, 1), padding=(0, 0), 
                           dilation=(1, 1)):
    """
    Custom depthwise 2D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        weight: Weight tensor of shape (C, 1, K_h, K_w)
        bias: Optional bias tensor of shape (C,)
        stride: Tuple (stride_h, stride_w)
        padding: Tuple (padding_h, padding_w)
        dilation: Tuple (dilation_h, dilation_w)
    
    Returns:
        Output tensor of shape (B, C, H_out, W_out)
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C, H, W = x.shape
    C_out, _, K_h, K_w = weight.shape  # For depthwise, C_out should equal C
    
    # Calculate output dimensions
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    
    H_out = (H + 2 * padding_h - dilation_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W + 2 * padding_w - dilation_w * (K_w - 1) - 1) // stride_w + 1
    
    # Prepare output tensor
    y = torch.empty((B, C, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Determine grid dimensions
    # We'll use a 4D grid: [batch, channel, H_blocks, W_blocks]
    BLOCK_SIZE_H = 4
    BLOCK_SIZE_W = 4
    
    grid_h = (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    
    grid = (B, C, grid_h, grid_w)
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        x, weight, bias, y,
        B, C, H, W, H_out, W_out,
        K_h, K_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        bias is not None,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution using Triton kernel.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size_h (int): Height of the convolution kernel.
        kernel_size_w (int): Width of the convolution kernel.
        stride_h (int, optional): Stride of the convolution in height dimension. Defaults to 1.
        stride_w (int, optional): Stride of the convolution in width dimension. Defaults to 1.
        padding_h (int, optional): Padding applied to the input in height dimension. Defaults to 0.
        padding_w (int, optional): Padding applied to the input in width dimension. Defaults to 0.
        dilation_h (int, optional): Spacing between kernel elements in height dimension. Defaults to 1.
        dilation_w (int, optional): Spacing between kernel elements in width dimension. Defaults to 1.
        groups (int, optional): Number of blocked connections. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, 
                 stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, 
                 dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups
        
        # Create weight tensor with proper shape for depthwise convolution
        # Note: For depthwise, we need weight shape (C, 1, K_h, K_w)
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size_h, kernel_size_w))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized depthwise 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return depthwise_conv2d_triton(
            x, self.weight, self.bias,
            stride=(self.stride_h, self.stride_w),
            padding=(self.padding_h, self.padding_w),
            dilation=(self.dilation_h, self.dilation_w)
        )


import math