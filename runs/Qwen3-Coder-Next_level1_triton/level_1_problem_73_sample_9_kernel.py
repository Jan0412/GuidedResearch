import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    # Pointers to tensors
    x_ptr,  # Input: (B, C_in, D, H, W)
    w_ptr,  # Weight: (C_in, C_out, kD, kH, kW)
    b_ptr,  # Bias: (C_out,) or None
    out_ptr,  # Output: (B, C_out, D_out, H_out, W_out)
    # Tensor dimensions
    B, C_in, D, H, W,
    C_out, kD, kH, kW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    output_pad_d, output_pad_h, output_pad_w,
    # Strides
    x_stride_b, x_stride_c, x_stride_d, x_stride_h, x_stride_w,
    w_stride_cin, w_stride_cout, w_stride_kd, w_stride_kh, w_stride_kw,
    out_stride_b, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    # Block sizes
    BLOCK_CIN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_COUT: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_cout = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Output position
    out_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    out_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    
    # Check bounds for output dimensions
    mask_d = out_d < (D - 1) * stride_d - 2 * pad_d + output_pad_d + kD
    mask_h = out_h < (H - 1) * stride_h - 2 * pad_h + output_pad_h + kH
    mask_w = out_w < (W - 1) * stride_w - 2 * pad_w + output_pad_w + kW
    
    # Initialize accumulator
    out_sum = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Loop over input channels
    for c_in_start in range(0, C_in, BLOCK_CIN):
        c_in = c_in_start + tl.arange(0, BLOCK_CIN)
        mask_cin = c_in < C_in
        
        # Compute corresponding input positions
        in_d = (out_d + pad_d - kD + 1) // stride_d
        in_h = (out_h + pad_h - kH + 1) // stride_h
        in_w = (out_w + pad_w - kW + 1) // stride_w
        
        # Check input bounds
        in_d_valid = (in_d >= 0) & (in_d < D)
        in_h_valid = (in_h >= 0) & (in_h < H)
        in_w_valid = (in_w >= 0) & (in_w < W)
        
        # Compute fractional positions in input (kD, kH, kW)
        k_d = out_d + pad_d - in_d * stride_d
        k_h = out_h + pad_h - in_h * stride_h
        k_w = out_w + pad_w - in_w * stride_w
        
        # Check kernel bounds
        k_d_valid = (k_d >= 0) & (k_d < kD)
        k_h_valid = (k_h >= 0) & (k_h < kH)
        k_w_valid = (k_w >= 0) & (k_w < kW)
        
        # Load input values
        x_offset = (pid_b * x_stride_b + 
                   c_in[:, None, None, None] * x_stride_c +
                   in_d[None, :, None, None] * x_stride_d +
                   in_h[None, None, :, None] * x_stride_h +
                   in_w[None, None, None, :] * x_stride_w)
        
        x_mask = (mask_cin[:, None, None, None] & 
                 in_d_valid[None, :, None, None] &
                 in_h_valid[None, None, :, None] &
                 in_w_valid[None, None, None, :])
        
        x_val = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)
        
        # Load weights
        w_offset = (c_in[:, None, None, None] * w_stride_cin +
                   pid_cout * w_stride_cout +
                   k_d[None, :, None, None] * w_stride_kd +
                   k_h[None, None, :, None] * w_stride_kh +
                   k_w[None, None, None, :] * w_stride_kw)
        
        w_mask = (mask_cin[:, None, None, None] &
                 k_d_valid[None, :, None, None] &
                 k_h_valid[None, None, :, None] &
                 k_w_valid[None, None, None, :])
        
        w_val = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)
        
        # Accumulate: x * w
        out_sum += tl.sum(x_val * w_val, axis=0)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_cout * BLOCK_COUT + tl.arange(0, BLOCK_COUT)[:1, None, None])
        out_sum += bias
    
    # Store result
    out_offset = (pid_b * out_stride_b +
                 pid_cout * out_stride_c +
                 out_d[None, :, None, None] * out_stride_d +
                 out_h[None, None, :, None] * out_stride_h +
                 out_w[None, None, None, :] * out_stride_w)
    
    out_mask = (mask_d[None, :, None, None] &
               mask_h[None, None, :, None] &
               mask_w[None, None, None, :])
    
    tl.store(out_ptr + out_offset, out_sum, mask=out_mask)


def triton_conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups):
    """
    Custom Triton implementation of 3D transposed convolution
    """
    # Get dimensions
    B, C_in, D, H, W = x.shape
    C_out = weight.shape[1]  # weight shape: (C_in, C_out, kD, kH, kW)
    kD, kH, kW = weight.shape[2], weight.shape[3], weight.shape[4]
    
    # Calculate output dimensions
    D_out = (D - 1) * stride[0] - 2 * padding[0] + output_padding[0] + kD
    H_out = (H - 1) * stride[1] - 2 * padding[1] + output_padding[1] + kH
    W_out = (W - 1) * stride[2] - 2 * padding[2] + output_padding[2] + kW
    
    # Prepare output tensor
    out = torch.empty(B, C_out, D_out, H_out, W_out, device=x.device, dtype=x.dtype)
    
    # Ensure contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Calculate strides
    x_stride_b, x_stride_c, x_stride_d, x_stride_h, x_stride_w = x.stride()
    w_stride_cin, w_stride_cout, w_stride_kd, w_stride_kh, w_stride_kw = weight.stride()
    out_stride_b, out_stride_c, out_stride_d, out_stride_h, out_stride_w = out.stride()
    
    # Block sizes for parallelization
    BLOCK_D = 4
    BLOCK_H = 8
    BLOCK_W = 16
    BLOCK_COUT = 16
    BLOCK_CIN = min(32, C_in)
    
    # Grid dimensions
    grid = (B, triton.cdiv(C_out, BLOCK_COUT), triton.cdiv(D_out, BLOCK_D), 
            triton.cdiv(H_out, BLOCK_H), triton.cdiv(W_out, BLOCK_W))
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        B, C_in, D, H, W,
        C_out, kD, kH, kW,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        x_stride_b, x_stride_c, x_stride_d, x_stride_h, x_stride_w,
        w_stride_cin, w_stride_cout, w_stride_kd, w_stride_kh, w_stride_kw,
        out_stride_b, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
        BLOCK_CIN=BLOCK_CIN,
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        BLOCK_COUT=BLOCK_COUT,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the 3D transposed convolution model using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, 
                                                   kernel_size=(kernel_size, kernel_size, kernel_size), 
                                                   stride=stride, padding=padding, 
                                                   groups=groups, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract parameters from the original layer
        weight = self.conv_transpose3d.weight
        bias = self.conv_transpose3d.bias
        
        # Call custom Triton implementation
        return triton_conv_transpose3d(
            x, weight, bias,
            stride=self.conv_transpose3d.stride,
            padding=self.conv_transpose3d.padding,
            output_padding=self.conv_transpose3d.output_padding,
            groups=self.conv_transpose3d.groups
        )