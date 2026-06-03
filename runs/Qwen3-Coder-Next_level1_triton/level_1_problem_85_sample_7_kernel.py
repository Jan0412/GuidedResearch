import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,                # Input tensor pointer (N, C, H, W)
    w_ptr,                # Weight tensor pointer (C, 1, kH, kW)
    b_ptr,                # Bias tensor pointer (C,) - optional
    out_ptr,              # Output tensor pointer (N, C, H_out, W_out)
    N, C, H, W,           # Input dimensions
    kH, kW,               # Kernel dimensions
    stride_h, stride_w,   # Stride
    padding_h, padding_w, # Padding
    dilation_h, dilation_w, # Dilation
    out_H, out_W,         # Output dimensions
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # Program ID represents the (batch, channel) pair
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate starting positions for the output
    out_h_start = tl.program_id(2) * BLOCK_SIZE_H
    out_w_start = tl.program_id(3) * BLOCK_SIZE_W
    
    # Offset for this specific batch and channel in input
    x_offset = batch_idx * C * H * W + channel_idx * H * W
    # Offset for this channel in weights
    w_offset = channel_idx * kH * kW
    # Offset for bias if present
    b_offset = channel_idx
    
    # Compute output position
    for out_h_block in range(BLOCK_SIZE_H):
        out_h = out_h_start + out_h_block
        if out_h >= out_H:
            break
            
        # Calculate input starting position for this output position
        in_h = out_h * stride_h - padding_h
        
        for out_w_block in range(BLOCK_SIZE_W):
            out_w = out_w_start + out_w_block
            if out_w >= out_W:
                break
                
            # Calculate input starting position for this output position
            in_w = out_w * stride_w - padding_w
            
            # Accumulator for convolution
            acc = 0.0
            
            # Iterate over kernel
            for kh in range(kH):
                in_h_k = in_h + kh * dilation_h
                # Check if within bounds
                if in_h_k >= 0 and in_h_k < H:
                    for kw in range(kW):
                        in_w_k = in_w + kw * dilation_w
                        if in_w_k >= 0 and in_w_k < W:
                            # Calculate input index
                            x_idx = x_offset + in_h_k * W + in_w_k
                            # Calculate weight index
                            w_idx = w_offset + kh * kW + kw
                            
                            # Load and multiply
                            x_val = tl.load(x_ptr + x_idx)
                            w_val = tl.load(w_ptr + w_idx)
                            acc += x_val * w_val
            
            # Add bias if present
            if HAS_BIAS:
                b_val = tl.load(b_ptr + b_offset)
                acc += b_val
            
            # Store output
            out_idx = batch_idx * C * out_H * out_W + channel_idx * out_H * out_W + out_h * out_W + out_w
            tl.store(out_ptr + out_idx, acc)


def depthwise_conv2d_triton(x, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """
    Triton implementation of depthwise 2D convolution
    """
    N, C, H, W = x.shape
    kH, kW = weight.shape[2], weight.shape[3]
    
    # Calculate output dimensions
    out_H = (H + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0] + 1
    out_W = (W + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1] + 1
    
    # Prepare output tensor
    out = torch.empty((N, C, out_H, out_W), dtype=x.dtype, device=x.device)
    
    # Set up kernel parameters
    BLOCK_SIZE_H = 4
    BLOCK_SIZE_W = 8
    
    # Grid: (batch_size, channels, out_h_blocks, out_w_blocks)
    grid = (
        N,
        C,
        (out_H + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (out_W + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    )
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        x, weight, bias,
        out,
        N, C, H, W,
        kH, kW,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        out_H, out_W,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        HAS_BIAS=bias is not None,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and asymmetric kernel.
    
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
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
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
        
        # Initialize weights - note: for depthwise conv, out_channels should equal in_channels
        # but we keep the parameter for compatibility
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size_h, kernel_size_w))
        
        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Use our custom Triton implementation
        return depthwise_conv2d_triton(
            x, self.weight, self.bias,
            stride=(self.stride_h, self.stride_w),
            padding=(self.padding_h, self.padding_w),
            dilation=(self.dilation_h, self.dilation_w)
        )