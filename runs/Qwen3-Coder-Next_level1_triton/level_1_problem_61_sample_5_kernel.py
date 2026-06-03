import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    # Pointers to inputs and outputs
    x_ptr,  # Input tensor: (B, C_in, D, H, W)
    w_ptr,  # Weight tensor: (C_in, C_out, Kd, Kh, Kw)
    b_ptr,  # Bias tensor: (C_out,) or nullptr
    out_ptr,  # Output tensor: (B, C_out, D_out, H_out, W_out)
    # Dimensions
    B, C_in, C_out,
    D, H, W,
    D_out, H_out, W_out,
    Kd, Kh, Kw,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_pad_d, out_pad_h, out_pad_w,
    # Block sizes for tiling
    BLOCK_SIZE_C_out: tl.constexpr,
    BLOCK_SIZE_C_in: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Get program IDs
    pid_c_out = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Calculate output position
    out_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create meshgrid for output coordinates
    out_d_grid, out_h_grid, out_w_grid = tl.meshgrid(out_d, out_h, out_w)
    out_d_grid = tl.reshape(out_d_grid, [BLOCK_SIZE_D * BLOCK_SIZE_H * BLOCK_SIZE_W])
    out_h_grid = tl.reshape(out_h_grid, [BLOCK_SIZE_D * BLOCK_SIZE_H * BLOCK_SIZE_W])
    out_w_grid = tl.reshape(out_w_grid, [BLOCK_SIZE_D * BLOCK_SIZE_H * BLOCK_SIZE_W])
    
    # Compute corresponding input coordinates
    in_d = (out_d_grid - out_pad_d) // stride_d
    in_h = (out_h_grid - out_pad_h) // stride_h
    in_w = (out_w_grid - out_pad_w) // stride_w
    
    # Check if input coordinates are valid
    valid_mask = (
        (in_d >= 0) & (in_d < D) &
        (in_h >= 0) & (in_h < H) &
        (in_w >= 0) & (in_w < W)
    )
    
    # Offset calculations
    batch_offset = pid_b * (C_in * D * H * W)
    out_batch_offset = pid_b * (C_out * D_out * H_out * W_out)
    
    # Accumulate over input channels
    acc = tl.zeros([BLOCK_SIZE_D * BLOCK_SIZE_H * BLOCK_SIZE_W], dtype=tl.float32)
    
    # Loop over input channels in blocks
    for start_c_in in range(0, C_in, BLOCK_SIZE_C_in):
        c_in_range = start_c_in + tl.arange(0, BLOCK_SIZE_C_in)
        c_in_mask = c_in_range < C_in
        
        # Load input values
        in_d_indices = in_d * (H * W)
        in_h_indices = in_h * W
        in_w_indices = out_w_grid
        in_indices = batch_offset + c_in_range[:, None] * (D * H * W) + in_d_indices + in_h_indices + in_w_indices
        
        # Transpose input for better access pattern
        in_indices_t = tl.trans(in_indices)
        x_vals = tl.load(x_ptr + in_indices_t, mask=valid_mask & c_in_mask[:, None], other=0.0)
        
        # Load weights for this block of input channels
        # Weight layout: (C_in, C_out, Kd, Kh, Kw)
        weight_indices = c_in_range[:, None, None, None, None] * (C_out * Kd * Kh * Kw) + \
                        pid_c_out * (Kd * Kh * Kw) + \
                        tl.arange(0, Kd)[None, :, None, None, None] * (Kh * Kw) + \
                        tl.arange(0, Kh)[None, None, :, None, None] * Kw + \
                        tl.arange(0, Kw)[None, None, None, :, None]
        
        w_vals = tl.load(w_ptr + weight_indices, mask=c_in_mask[:, None, None, None, None])
        
        # Compute offset for valid positions
        d_offset = (out_d_grid - out_pad_d) % stride_d
        h_offset = (out_h_grid - out_pad_h) % stride_h
        w_offset = (out_w_grid - out_pad_w) % stride_w
        
        # Only accumulate if the position matches kernel alignment
        aligned_mask = (d_offset == 0) & (h_offset == 0) & (w_offset == 0)
        aligned_mask = tl.reshape(aligned_mask, [BLOCK_SIZE_D * BLOCK_SIZE_H * BLOCK_SIZE_W])
        
        # Reshape for broadcasting
        x_vals_reshaped = tl.reshape(x_vals, [BLOCK_SIZE_C_in, BLOCK_SIZE_D * BLOCK_SIZE_H * BLOCK_SIZE_W])
        w_vals_reshaped = tl.reshape(w_vals, [BLOCK_SIZE_C_in, BLOCK_SIZE_D * BLOCK_SIZE_H * BLOCK_SIZE_W])
        
        # Multiply and accumulate
        acc += tl.sum(x_vals_reshaped * w_vals_reshaped * aligned_mask[None, :], axis=0)
    
    # Convert to output precision and add bias if provided
    acc = acc.to(x_ptr.dtype.element_ty)
    
    # Add bias
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_c_out * BLOCK_SIZE_C_out + tl.arange(0, BLOCK_SIZE_C_out))
        acc += bias
    
    # Store output
    out_indices = out_batch_offset + pid_c_out * (D_out * H_out * W_out) + \
                  out_d_grid * (H_out * W_out) + out_h_grid * W_out + out_w_grid
    out_indices = tl.trans(out_indices)
    tl.store(out_ptr + out_indices, acc, mask=valid_mask)


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    groups: int = 1
) -> torch.Tensor:
    """
    Performs transposed 3D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        weight: Weight tensor of shape (in_channels, out_channels, kernel_depth, kernel_height, kernel_width)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride of the convolution
        padding: Padding applied to the input
        output_padding: Additional size added to one side of the output shape
        groups: Number of blocked connections (currently only groups=1 is supported)
    
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
    C_in2, C_out, Kd, Kh, Kw = weight.shape
    
    assert C_in == C_in2, "Input channels mismatch"
    assert groups == 1, "Only groups=1 is supported in this kernel"
    
    # Calculate output dimensions
    D_out = (D - 1) * stride - 2 * padding + output_padding + Kd
    H_out = (H - 1) * stride - 2 * padding + output_padding + Kh
    W_out = (W - 1) * stride - 2 * padding + output_padding + Kw
    
    # Create output tensor
    out = torch.empty((B, C_out, D_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Define block sizes for tiling
    BLOCK_SIZE_C_out = 16
    BLOCK_SIZE_C_in = 16
    BLOCK_SIZE_D = 4
    BLOCK_SIZE_H = 4
    BLOCK_SIZE_W = 4
    
    # Calculate grid dimensions
    grid_d = (D_out + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D
    grid_h = (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    
    # Launch the kernel
    conv_transpose3d_kernel[(
        C_out, B, grid_d, grid_h, grid_w
    )](
        x, weight, bias, out,
        B, C_in, C_out,
        D, H, W,
        D_out, H_out, W_out,
        Kd, Kh, Kw,
        stride, stride, stride,
        padding, padding, padding,
        output_padding, output_padding, output_padding,
        BLOCK_SIZE_C_out=BLOCK_SIZE_C_out,
        BLOCK_SIZE_C_in=BLOCK_SIZE_C_in,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the transposed 3D convolution model using Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same parameters as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Create the same ConvTranspose3d layer but we'll use our custom kernel
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, 
                                                  kernel_size=(kernel_size, kernel_size, kernel_size), 
                                                  stride=stride, padding=padding, 
                                                  output_padding=output_padding, groups=groups, 
                                                  bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using our optimized Triton kernel.
        """
        # Extract parameters from the original layer
        weight = self.conv_transpose3d.weight
        bias = self.conv_transpose3d.bias
        
        # Use our Triton kernel instead of the PyTorch implementation
        return triton_conv_transpose3d(
            x, weight, bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )