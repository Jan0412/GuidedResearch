import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose2d_kernel(
    X,  # Input tensor (B, C_in, H_in, W_in)
    W,  # Weight tensor (C_in, C_out, K_h, K_w)
    B,  # Bias tensor (C_out,)
    Y,  # Output tensor (B, C_out, H_out, W_out)
    batch_size, in_channels, out_channels,
    in_h, in_w,
    out_h, out_w,
    kernel_h, kernel_w,
    stride_h, stride_w,
    pad_h, pad_w,
    out_pad_h, out_pad_w,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch size
    BLOCK_SIZE_K: tl.constexpr,  # Block size for input channels
    BLOCK_SIZE_H: tl.constexpr,  # Block size for output height
    BLOCK_SIZE_W: tl.constexpr,  # Block size for output width
):
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_out_c = tl.program_id(1)
    pid_out_h = tl.program_id(2)
    pid_out_w = tl.program_id(3)
    
    # Compute output position
    out_c = pid_out_c * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    out_h = pid_out_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_out_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create meshgrid for output positions
    out_h_grid, out_w_grid = tl.meshgrid(out_h, out_w)
    out_h_grid = out_h_grid.T
    out_w_grid = out_w_grid.T
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M,), tl.float32)
    
    # Compute corresponding input positions
    in_h_start = out_h_grid * stride_h - pad_h
    in_w_start = out_w_grid * stride_w - pad_w
    
    # Iterate over input channels and kernel dimensions
    for in_c in range(in_channels):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Compute input position
                in_h_pos = in_h_start + kh
                in_w_pos = in_w_start + kw
                
                # Check if input position is valid
                valid_mask = (in_h_pos >= 0) & (in_h_pos < in_h) & (in_w_pos >= 0) & (in_w_pos < in_w)
                
                # Load input values
                x_ptr = X + pid_batch * in_channels * in_h * in_w + \
                        in_c * in_h * in_w + \
                        in_h_pos * in_w + in_w_pos
                x_val = tl.load(x_ptr, mask=valid_mask, other=0.0)
                
                # Load weight values
                w_ptr = W + in_c * out_channels * kernel_h * kernel_w + \
                        pid_out_c * kernel_h * kernel_w + \
                        kh * kernel_w + kw
                w_val = tl.load(w_ptr)
                
                # Accumulate
                acc += x_val * w_val
                
    # Add bias if present
    if B is not None:
        b_ptr = B + pid_out_c
        b_val = tl.load(b_ptr)
        acc += b_val
    
    # Store result
    y_ptr = Y + pid_batch * out_channels * out_h * out_w + \
            pid_out_c * out_h * out_w + \
            out_h_grid * out_w + out_w_grid
    tl.store(y_ptr, acc, mask=(out_c < out_channels))


def triton_conv_transpose2d(x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
    """
    Triton implementation of ConvTranspose2d.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, height, width)
        weight: Weight tensor of shape (in_channels, out_channels, kernel_h, kernel_w)
        bias: Optional bias tensor of shape (out_channels,)
        stride, padding, output_padding, groups: Convolution parameters
    
    Returns:
        Output tensor of shape (batch_size, out_channels, out_h, out_w)
    """
    # Get dimensions
    batch_size, in_channels, in_h, in_w = x.shape
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]
    out_channels = weight.shape[1]
    
    # Calculate output dimensions
    out_h = (in_h - 1) * stride - 2 * padding + output_padding + kernel_h
    out_w = (in_w - 1) * stride - 2 * padding + output_padding + kernel_w
    
    # Create output tensor
    y = torch.empty((batch_size, out_channels, out_h, out_w), dtype=x.dtype, device=x.device)
    
    # Kernel block sizes (tunable parameters for performance)
    BLOCK_SIZE_M = 8
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_K = 8
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    
    # Grid dimensions
    grid = (
        batch_size,  # batch dimension
        triton.cdiv(out_channels, BLOCK_SIZE_M),  # output channels
        triton.cdiv(out_h, BLOCK_SIZE_H),  # output height
        triton.cdiv(out_w, BLOCK_SIZE_W)   # output width
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias, y,
        batch_size, in_channels, out_channels,
        in_h, in_w, out_h, out_w,
        kernel_h, kernel_w,
        stride, stride,
        padding, padding,
        output_padding, output_padding,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of Model with Triton implementation of transposed convolution.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        output_padding (int or tuple, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Initialize weights and bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.groups = groups
        self.bias_flag = bias
        
        # Create weight parameter
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, self.kernel_size[0], self.kernel_size[1]))
        
        # Create bias parameter if needed
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using custom Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding[0],  # Assuming symmetric padding for simplicity
            output_padding=self.output_padding,
            groups=self.groups
        )