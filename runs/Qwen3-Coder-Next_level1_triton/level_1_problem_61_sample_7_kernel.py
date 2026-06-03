import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # Input tensor (batch, in_channels, D, H, W)
    w_ptr,  # Weight tensor (in_channels, out_channels, kD, kH, kW)
    b_ptr,  # Bias tensor (out_channels,)
    out_ptr,  # Output tensor (batch, out_channels, D_out, H_out, W_out)
    n_batch, n_in_channels, n_out_channels,
    D, H, W,  # Input dimensions
    kD, kH, kW,  # Kernel dimensions
    stride, padding, output_padding,
    D_out, H_out, W_out,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get batch, out_d, out_h, out_w indices
    batch_idx = tl.program_id(0)
    out_d = tl.program_id(1)
    out_h = tl.program_id(2)
    out_w = tl.program_id(3)
    
    # Compute start indices for input and kernel
    in_d_start = out_d * stride - padding
    in_h_start = out_h * stride - padding
    in_w_start = out_w * stride - padding
    
    # Iterate over output channels in blocks
    for out_c_block in range(0, n_out_channels, BLOCK_SIZE_K):
        out_c = out_c_block + tl.arange(0, BLOCK_SIZE_K)
        out_c_mask = out_c < n_out_channels
        out_c = tl.where(out_c_mask, out_c, 0)
        
        # Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32)
        
        # Iterate over input channels
        for in_c in range(n_in_channels):
            # Iterate over kernel elements
            for kd in range(kD):
                in_d = in_d_start + kd
                if 0 <= in_d < D:
                    for kh in range(kH):
                        in_h = in_h_start + kh
                        if 0 <= in_h < H:
                            for kw in range(kW):
                                in_w = in_w_start + kw
                                if 0 <= in_w < W:
                                    # Compute offsets
                                    x_offset = ((batch_idx * n_in_channels + in_c) * D * H * W +
                                               in_d * H * W + in_h * W + in_w)
                                    w_offset = ((in_c * n_out_channels + out_c) * kD * kH * kW +
                                               kd * kH * kW + kh * kW + kw)
                                    
                                    # Load values
                                    x_val = tl.load(x_ptr + x_offset)
                                    w_val = tl.load(w_ptr + w_offset)
                                    
                                    # Accumulate
                                    acc += x_val * w_val
        
        # Store output
        out_offset = ((batch_idx * n_out_channels + out_c) * D_out * H_out * W_out +
                     out_d * H_out * W_out + out_h * W_out + out_w)
        
        # Add bias if provided
        if b_ptr is not None:
            bias_val = tl.load(b_ptr + out_c)
            acc += bias_val
        
        tl.store(out_ptr + out_offset, acc.to(tl.float32), mask=out_c_mask)


def triton_conv_transpose3d(x, weight, bias=None, stride=1, padding=0, output_padding=0):
    """
    Performs transposed 3D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch, in_channels, D, H, W)
        weight: Weight tensor of shape (in_channels, out_channels, kD, kH, kW)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride of the convolution
        padding: Padding applied to the input
        output_padding: Additional size added to one side of the output shape
    """
    batch, in_channels, D, H, W = x.shape
    _, out_channels, kD, kH, kW = weight.shape
    
    # Calculate output dimensions
    D_out = (D - 1) * stride - 2 * padding + (kD - 1) + 1 + output_padding
    H_out = (H - 1) * stride - 2 * padding + (kH - 1) + 1 + output_padding
    W_out = (W - 1) * stride - 2 * padding + (kW - 1) + 1 + output_padding
    
    # Prepare output tensor
    out = torch.empty((batch, out_channels, D_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Configure kernel parameters
    BLOCK_SIZE_K = min(8, out_channels)  # Output channel block size
    
    # Grid: [batch, D_out, H_out, W_out]
    grid = (batch, D_out, H_out, W_out)
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        batch, in_channels, out_channels,
        D, H, W,
        kD, kH, kW,
        stride, padding, output_padding,
        D_out, H_out, W_out,
        BLOCK_SIZE_D=1,
        BLOCK_SIZE_H=1,
        BLOCK_SIZE_W=1,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution with square input and square kernel.
    Uses optimized Triton kernel for the convolution operation.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.has_bias = bias
        
        # Initialize weights with same shape as nn.ConvTranspose3d
        # Weight shape: (in_channels, out_channels, kD, kH, kW)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        # Ensure weights are contiguous
        weight = self.weight.contiguous()
        
        # Call Triton kernel
        return triton_conv_transpose3d(
            x, weight, self.bias,
            stride=self.stride, padding=self.padding, output_padding=self.output_padding
        )