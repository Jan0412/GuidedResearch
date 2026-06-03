import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Input tensor pointer
    w_ptr,  # Weight tensor pointer
    b_ptr,  # Bias tensor pointer (optional, can be None)
    out_ptr,  # Output tensor pointer
    batch_size,  # Batch size
    in_channels,  # Number of input channels
    height,  # Input height
    width,  # Input width
    kernel_h,  # Kernel height
    kernel_w,  # Kernel width
    stride_h,  # Stride height
    stride_w,  # Stride width
    padding_h,  # Padding height
    padding_w,  # Padding width
    dilation_h,  # Dilation height
    dilation_w,  # Dilation width
    out_h,  # Output height
    out_w,  # Output width
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Get program IDs
    bc = tl.program_id(0)  # Batch index
    c = tl.program_id(1)   # Channel index
    oh = tl.program_id(2)  # Output height index
    ow = tl.program_id(3)  # Output width index

    # Compute input starting position for this output position
    ih = oh * stride_h - padding_h
    iw = ow * stride_w - padding_w

    # Initialize accumulator
    acc = 0.0

    # Loop over kernel dimensions
    for kh in range(BLOCK_SIZE_KH):
        if kh < kernel_h:
            h_pos = ih + kh * dilation_h
            # Check if within input bounds
            if h_pos >= 0 and h_pos < height:
                for kw in range(BLOCK_SIZE_KW):
                    if kw < kernel_w:
                        w_pos = iw + kw * dilation_w
                        # Check if within input bounds
                        if w_pos >= 0 and w_pos < width:
                            # Compute indices
                            x_idx = bc * (in_channels * height * width) + \
                                    c * (height * width) + \
                                    h_pos * width + w_pos
                            w_idx = c * (kernel_h * kernel_w) + \
                                    kh * kernel_w + kw
                            acc += tl.load(x_ptr + x_idx) * tl.load(w_ptr + w_idx)

    # Add bias if provided
    if b_ptr is not None:
        b = tl.load(b_ptr + c)
        acc += b

    # Store result
    out_idx = bc * (in_channels * out_h * out_w) + \
              c * (out_h * out_w) + \
              oh * out_w + ow
    tl.store(out_ptr + out_idx, acc)


def triton_depthwise_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride_h: int = 1,
    stride_w: int = 1,
    padding_h: int = 0,
    padding_w: int = 0,
    dilation_h: int = 1,
    dilation_w: int = 1,
):
    """
    Triton-based depthwise convolution implementation.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, height, width)
        weight: Weight tensor of shape (in_channels, kernel_h, kernel_w)
        bias: Optional bias tensor of shape (in_channels,)
        stride_h, stride_w: Stride in height and width dimensions
        padding_h, padding_w: Padding in height and width dimensions
        dilation_h, dilation_w: Dilation in height and width dimensions
    
    Returns:
        Output tensor of shape (batch_size, in_channels, out_h, out_w)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, height, width = x.shape
    kernel_h, kernel_w = weight.shape[1], weight.shape[2]
    
    # Calculate output dimensions
    out_h = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    
    # Prepare output tensor
    out = torch.empty((batch_size, in_channels, out_h, out_w), dtype=x.dtype, device=x.device)
    
    # Configure grid and block sizes
    BLOCK_SIZE_C = 1  # Process one channel per program for depthwise conv
    BLOCK_SIZE_KH = 32  # Can be tuned based on kernel size
    BLOCK_SIZE_KW = 32  # Can be tuned based on kernel size
    
    grid = (
        batch_size,
        in_channels,
        out_h,
        out_w,
    )
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, height, width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        out_h, out_w,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_KH=BLOCK_SIZE_KH,
        BLOCK_SIZE_KW=BLOCK_SIZE_KW,
    )
    
    return out


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
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, 
                 stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, 
                 dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
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
        
        # Create weight and bias parameters (note: for depthwise conv, out_channels should equal in_channels)
        self.weight = nn.Parameter(torch.randn(in_channels, kernel_size_h, kernel_size_w))
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if bias:
            fan_in = kernel_size_h * kernel_size_w
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        # Ensure weight and bias are on same device as input
        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)
        if self.bias is not None and self.bias.device != x.device:
            self.bias = self.bias.to(x.device)
            
        return triton_depthwise_conv2d(
            x, 
            self.weight, 
            self.bias,
            self.stride_h, 
            self.stride_w, 
            self.padding_h, 
            self.padding_w, 
            self.dilation_h, 
            self.dilation_w
        )