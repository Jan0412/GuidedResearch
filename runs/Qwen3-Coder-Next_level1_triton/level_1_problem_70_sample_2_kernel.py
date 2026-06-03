import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def conv_transpose3d_kernel(
    # Pointers to tensors
    X_ptr,  # Input: (B, C_in, D, H, W)
    W_ptr,  # Weight: (C_in, C_out, K_d, K_h, K_w)
    B_ptr,  # Bias: (C_out,) or None
    Y_ptr,  # Output: (B, C_out, D_out, H_out, W_out)
    # Tensor dimensions
    B, C_in, C_out,
    D, H, W,
    K_d, K_h, K_w,
    D_out, H_out, W_out,
    # Strides
    stride_x_batch, stride_x_ch, stride_x_d, stride_x_h, stride_x_w,
    stride_w_in, stride_w_out, stride_w_kd, stride_w_kh, stride_w_kw,
    stride_y_batch, stride_y_ch, stride_y_d, stride_y_h, stride_y_w,
    # Convolution parameters
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    output_pad_d, output_pad_h, output_pad_w,
    dil_d, dil_h, dil_w,
    # Block sizes
    BLOCK_C_in: tl.constexpr,
    BLOCK_C_out: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Compute output positions
    out_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    out_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    
    # Check bounds
    mask_d = out_d < D_out
    mask_h = out_h < H_out
    mask_w = out_w < W_out
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Iterate over input channels
    for c_in in range(0, C_in, BLOCK_C_in):
        # Compute input positions for this block
        in_d = (out_d - output_pad_d + pad_d) // stride_d
        in_h = (out_h - output_pad_h + pad_h) // stride_h
        in_w = (out_w - output_pad_w + pad_w) // stride_w
        
        # Calculate kernel positions
        k_d = (out_d - output_pad_d + pad_d) % stride_d + in_d * dil_d - pad_d
        k_h = (out_h - output_pad_h + pad_h) % stride_h + in_h * dil_h - pad_h
        k_w = (out_w - output_pad_w + pad_w) % stride_w + in_w * dil_w - pad_w
        
        # Create masks for valid kernel positions
        mask_kd = (k_d >= 0) & (k_d < K_d)
        mask_kh = (k_h >= 0) & (k_h < K_h)
        mask_kw = (k_w >= 0) & (k_w < K_w)
        
        # Create combined mask for valid positions
        mask_valid = (in_d >= 0) & (in_d < D) & \
                     (in_h >= 0) & (in_h < H) & \
                     (in_w >= 0) & (in_w < W) & \
                     mask_kd & mask_kh & mask_kw
        
        # Broadcast masks to 3D
        mask_dh = mask_d[:, None, None] & mask_h[None, :, None]
        mask_dhw = mask_dh[:, :, None] & mask_w[None, None, :]
        mask_final = mask_dhw & mask_valid[None, :, :, :]
        
        # Load input data
        x_offsets = (pid_b * stride_x_batch + 
                    tl.arange(0, BLOCK_C_in)[None, None, None, :] * stride_x_ch +
                    in_d[:, None, None, None] * stride_x_d +
                    in_h[None, :, None, None] * stride_x_h +
                    in_w[None, None, :, None] * stride_x_w +
                    tl.arange(0, BLOCK_C_in)[None, None, None, :])
        
        # Load weights
        w_offsets = (tl.arange(0, BLOCK_C_in)[None, None, None, :] * stride_w_in +
                    pid_c_out * stride_w_out +
                    k_d[:, None, None, None] * stride_w_kd +
                    k_h[None, :, None, None] * stride_w_kh +
                    k_w[None, None, :, None] * stride_w_kw)
        
        # Load and multiply
        x_val = tl.load(X_ptr + x_offsets, mask=mask_final & (tl.arange(0, BLOCK_C_in)[None, None, None, :] < C_in), other=0.0)
        w_val = tl.load(W_ptr + w_offsets, mask=mask_final & (tl.arange(0, BLOCK_C_in)[None, None, None, :] < C_in), other=0.0)
        
        # Accumulate
        acc += tl.sum(x_val * w_val, axis=3)
    
    # Add bias if present
    if B_ptr is not None:
        bias = tl.load(B_ptr + pid_c_out * BLOCK_C_out)
        acc += bias
    
    # Store output
    y_offsets = (pid_b * stride_y_batch +
                pid_c_out * stride_y_ch +
                out_d[:, None, None] * stride_y_d +
                out_h[None, :, None] * stride_y_h +
                out_w[None, None, :] * stride_y_w)
    
    tl.store(Y_ptr + y_offsets, acc.to(Y_ptr.dtype.element_ty), mask=mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :])


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: Tuple[int, int, int] = (1, 1, 1),
    padding: Tuple[int, int, int] = (0, 0, 0),
    output_padding: Tuple[int, int, int] = (0, 0, 0),
    dilation: Tuple[int, int, int] = (1, 1, 1),
    groups: int = 1
) -> torch.Tensor:
    """
    Performs 3D transposed convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        weight: Weight tensor of shape (in_channels, out_channels, k_d, k_h, k_w)
        bias: Optional bias tensor of shape (out_channels,)
        stride, padding, output_padding, dilation, groups: Convolution parameters
    
    Returns:
        Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out)
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C_in, D, H, W = x.shape
    _, C_out, K_d, K_h, K_w = weight.shape
    
    # Calculate output dimensions manually to match PyTorch behavior
    D_out = (D - 1) * stride[0] - 2 * padding[0] + dilation[0] * (K_d - 1) + output_padding[0] + 1
    H_out = (H - 1) * stride[1] - 2 * padding[1] + dilation[1] * (K_h - 1) + output_padding[1] + 1
    W_out = (W - 1) * stride[2] - 2 * padding[2] + dilation[2] * (K_w - 1) + output_padding[2] + 1
    
    # Create output tensor
    y = torch.empty(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Calculate strides
    stride_x = x.stride()
    stride_w = weight.stride()
    stride_y = y.stride()
    
    # Define block sizes for optimization
    BLOCK_C_in = 16
    BLOCK_C_out = 16
    BLOCK_K = 4
    BLOCK_D = 4
    BLOCK_H = 4
    BLOCK_W = 4
    
    # Grid dimensions
    grid = (
        B,  # batch
        triton.cdiv(C_out, BLOCK_C_out),  # output channels
        triton.cdiv(D_out, BLOCK_D),  # depth
        triton.cdiv(H_out, BLOCK_H),  # height
        triton.cdiv(W_out, BLOCK_W),  # width
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, y,
        B, C_in, C_out,
        D, H, W,
        K_d, K_h, K_w,
        D_out, H_out, W_out,
        stride_x[0], stride_x[1], stride_x[2], stride_x[3], stride_x[4],
        stride_w[0], stride_w[1], stride_w[2], stride_w[3], stride_w[4],
        stride_y[0], stride_y[1], stride_y[2], stride_y[3], stride_y[4],
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        dilation[0], dilation[1], dilation[2],
        BLOCK_C_in=BLOCK_C_in,
        BLOCK_C_out=BLOCK_C_out,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernels for ConvTranspose3d.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters for reconstruction
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize the same weight and bias as original
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize parameters similar to PyTorch's default initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized transposed 3D convolution.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        # Convert scalar parameters to tuples if needed
        stride = (self.stride,) * 3 if isinstance(self.stride, int) else self.stride
        padding = (self.padding,) * 3 if isinstance(self.padding, int) else self.padding
        output_padding = (self.output_padding,) * 3 if isinstance(self.output_padding, int) else self.output_padding
        dilation = (self.dilation,) * 3 if isinstance(self.dilation, int) else self.dilation
        
        return triton_conv_transpose3d(
            x, self.weight, self.bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=self.groups
        )


import math

# Patch the get_inputs and get_init_inputs functions as needed
def get_inputs():
    batch_size = 8
    in_channels = 48
    out_channels = 24
    kernel_size = 3
    depth = 96
    height = 96
    width = 96
    x = torch.rand(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    in_channels = 48
    out_channels = 24
    kernel_size = 3
    return [in_channels, out_channels, kernel_size]