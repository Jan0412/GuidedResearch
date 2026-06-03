import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # Input tensor: (B, C_in, D, H, W)
    w_ptr,  # Weight tensor: (C_in, C_out, K_d, K_h, K_w)
    b_ptr,  # Bias tensor: (C_out,) or None
    out_ptr,  # Output tensor: (B, C_out, D_out, H_out, W_out)
    B: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    K_d: tl.constexpr,
    K_h: tl.constexpr,
    K_w: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    D_out: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr = 4,
    BLOCK_SIZE_H: tl.constexpr = 4,
    BLOCK_SIZE_W: tl.constexpr = 4,
    BLOCK_SIZE_C: tl.constexpr = 32,
):
    # Get program IDs for output dimensions
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Compute output position
    out_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for valid indices
    d_mask = out_d < D_out
    h_mask = out_h < H_out
    w_mask = out_w < W_out
    
    # Initialize accumulator
    output = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over input channels
    for c_in in range(0, C_in, BLOCK_SIZE_C):
        c_in_range = c_in + tl.arange(0, BLOCK_SIZE_C)
        c_in_mask = c_in_range < C_in
        
        # Create meshgrid for d, h, w
        d_grid, h_grid, w_grid = tl.meshgrid(out_d, out_h, out_w)
        d_grid = d_grid.reshape(BLOCK_SIZE_D * BLOCK_SIZE_H * BLOCK_SIZE_W, 1)
        h_grid = h_grid.reshape(BLOCK_SIZE_D * BLOCK_SIZE_H * BLOCK_SIZE_W, 1)
        w_grid = w_grid.reshape(BLOCK_SIZE_D * BLOCK_SIZE_H * BLOCK_SIZE_W, 1)
        
        # Compute corresponding input positions for transposed convolution
        # For transposed conv: input_pos = (output_pos - (kernel_pos - 1) * dilation - 1) // stride + padding
        # But it's easier to iterate over kernel positions and accumulate
        
        # Loop over kernel dimensions
        for kd in range(K_d):
            for kh in range(K_h):
                for kw in range(K_w):
                    # Compute input position from output position
                    # For transposed conv: output_pos = input_pos * stride - padding + kernel_pos * dilation
                    # => input_pos = (output_pos + padding - kernel_pos * dilation) / stride
                    
                    input_d = (out_d[:, None, None] * stride + kd * dilation - padding)
                    input_h = (out_h[None, :, None] * stride + kh * dilation - padding)
                    input_w = (out_w[None, None, :] * stride + kw * dilation - padding)
                    
                    # Check if input position is valid
                    d_valid = (input_d >= 0) & (input_d < D)
                    h_valid = (input_h >= 0) & (input_h < H)
                    w_valid = (input_w >= 0) & (input_w < W)
                    valid = d_valid & h_valid & w_valid
                    
                    # Compute 1D indices for input tensor
                    input_indices = (pid_b * (C_in * D * H * W) + 
                                    c_in_range[None, None, None, :] * (D * H * W) +
                                    input_d[:, :, :, None] * (H * W) +
                                    input_h[:, :, :, None] * W +
                                    input_w[:, :, :, None])
                    
                    # Compute 1D indices for weight tensor
                    # Weight shape: (C_in, C_out, K_d, K_h, K_w)
                    weight_indices = (c_in_range[:, None, None, None] * (C_out * K_d * K_h * K_w) +
                                     pid_c_out * (K_d * K_h * K_w) +
                                     kd * (K_h * K_w) +
                                     kh * K_w +
                                     kw)
                    
                    # Load input and weight values
                    x_shape = (BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C)
                    w_shape = (BLOCK_SIZE_C,)
                    
                    # Create mask for input load
                    valid_flat = valid.reshape(BLOCK_SIZE_D * BLOCK_SIZE_H * BLOCK_SIZE_W, 1)
                    d_flat = input_d.reshape(BLOCK_SIZE_D * BLOCK_SIZE_H * BLOCK_SIZE_W, 1)
                    h_flat = input_h.reshape(BLOCK_SIZE_D * BLOCK_SIZE_H * BLOCK_SIZE_W, 1)
                    w_flat = input_w.reshape(BLOCK_SIZE_D * BLOCK_SIZE_H * BLOCK_SIZE_W, 1)
                    
                    # Calculate flattened input indices
                    input_flattened = (pid_b * (C_in * D * H * W) + 
                                      c_in_range[None, :] * (D * H * W) +
                                      d_flat * (H * W) +
                                      h_flat * W +
                                      w_flat)
                    input_flattened = input_flattened.reshape(-1)
                    
                    # Load input
                    input_mask = valid_flat.reshape(-1)
                    x_vals = tl.load(x_ptr + input_flattened, mask=input_mask, other=0.0)
                    x_vals = x_vals.reshape(BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C)
                    
                    # Load weight
                    w_vals = tl.load(w_ptr + weight_indices, mask=c_in_mask[None, None, None, :], other=0.0)
                    
                    # Accumulate: output += x * w
                    output += tl.sum(x_vals * w_vals[:, :, :, :, None], axis=3)
    
    # Add bias if available
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_c_out)
        output += bias
    
    # Store output
    output = output.to(tl.float32)
    
    out_indices = (pid_b * (C_out * D_out * H_out * W_out) +
                  pid_c_out * (D_out * H_out * W_out) +
                  out_d[:, None, None] * (H_out * W_out) +
                  out_h[None, :, None] * W_out +
                  out_w[None, None, :])
    
    out_indices = out_indices.reshape(-1)
    output_flat = output.reshape(-1)
    
    out_mask = (d_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]).reshape(-1)
    
    tl.store(out_ptr + out_indices, output_flat, mask=out_mask)


def triton_conv_transpose3d(x, weight, bias, stride=1, padding=0, dilation=1):
    """
    Custom Triton kernel for 3D transposed convolution.
    """
    # Get dimensions
    B, C_in, D, H, W = x.shape
    C_out, _, K_d, K_h, K_w = weight.shape
    
    # Calculate output dimensions
    D_out = (D - 1) * stride - 2 * padding + dilation * (K_d - 1) + 1
    H_out = (H - 1) * stride - 2 * padding + dilation * (K_h - 1) + 1
    W_out = (W - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Allocate output tensor
    output = torch.empty(B, C_out, D_out, H_out, W_out, device=x.device, dtype=x.dtype)
    
    # Configure grid for kernel launch
    # We'll use a grid where each block processes a portion of the output
    BLOCK_SIZE_D = min(4, D_out)
    BLOCK_SIZE_H = min(4, H_out)
    BLOCK_SIZE_W = min(4, W_out)
    
    # Calculate grid dimensions
    grid_d = (D_out + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D
    grid_h = (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    
    # Launch kernel
    conv_transpose3d_kernel[grid_d, grid_h, grid_w](
        x, weight, bias, output,
        B, C_in, C_out,
        D, H, W,
        K_d, K_h, K_w,
        stride, padding, dilation,
        D_out, H_out, W_out,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=32 if C_in > 32 else C_in
    )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized version of the 3D transposed convolution model using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters and create the weight/bias tensors manually
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize weights using Xavier initialization
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size) * 
                                  math.sqrt(2.0 / (in_channels * kernel_size * kernel_size * kernel_size)))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D transposed convolution using Triton kernel.
        """
        return triton_conv_transpose3d(
            x, 
            self.weight, 
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )