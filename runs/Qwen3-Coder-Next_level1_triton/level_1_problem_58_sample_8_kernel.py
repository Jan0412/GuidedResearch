import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    X_ptr,  # Input tensor: (B, C_in, D_in, H_in, W_in)
    W_ptr,  # Weight tensor: (C_in, C_out // groups, Kd, Kh, Kw)
    B_ptr,  # Bias tensor: (C_out,) or None
    Y_ptr,  # Output tensor: (B, C_out, D_out, H_out, W_out)
    B, C_in, C_out, groups,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    Kd, Kh, Kw,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_pad_d, out_pad_h, out_pad_w,
    # Block sizes for tiling
    BLOCK_B: tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
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
    
    # Compute batch index
    batch_idx = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    batch_mask = batch_idx < B
    
    # Compute output channel indices
    c_out_idx = pid_c_out * BLOCK_C_OUT + tl.arange(0, BLOCK_C_OUT)
    c_out_mask = c_out_idx < C_out
    
    # Compute output spatial indices
    d_idx = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_idx < D_out
    
    h_idx = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_idx < H_out
    
    w_idx = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = w_idx < W_out
    
    # Create meshgrid for output positions
    d_grid, h_grid, w_grid = tl.meshgrid(d_idx, h_idx, w_idx)
    d_grid = d_grid.reshape(BLOCK_D * BLOCK_H * BLOCK_W)
    h_grid = h_grid.reshape(BLOCK_D * BLOCK_H * BLOCK_W)
    w_grid = w_grid.reshape(BLOCK_D * BLOCK_H * BLOCK_W)
    
    # Compute corresponding input positions
    d_in = d_grid - stride_d * (Kd - 1) + stride_d * pad_d - out_pad_d
    h_in = h_grid - stride_h * (Kh - 1) + stride_h * pad_h - out_pad_h
    w_in = w_grid - stride_w * (Kw - 1) + stride_w * pad_w - out_pad_w
    
    # Create masks for valid input positions
    d_in_mask = (d_in >= 0) & (d_in < D_in)
    h_in_mask = (h_in >= 0) & (h_in < H_in)
    w_in_mask = (w_in >= 0) & (w_in < W_in)
    valid_mask = d_in_mask & h_in_mask & w_in_mask
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_B, BLOCK_C_OUT, BLOCK_D * BLOCK_H * BLOCK_W), dtype=tl.float32)
    
    # Loop over input channels
    for c_in_start in range(0, C_in, BLOCK_C_IN):
        c_in_idx = c_in_start + tl.arange(0, BLOCK_C_IN)
        c_in_mask_local = c_in_idx < C_in
        
        # Compute group indices
        group = c_in_idx // (C_in // groups)
        c_in_in_group = c_in_idx % (C_in // groups)
        
        # Load input data: shape [B, BLOCK_C_IN, BLOCK_D, BLOCK_H, BLOCK_W]
        # We need to handle the 5D indexing carefully
        for bc in range(BLOCK_B):
            b_idx = batch_idx[bc] if bc < BLOCK_B and pid_b * BLOCK_B + bc < B else 0
            for bi in range(BLOCK_D * BLOCK_H * BLOCK_W):
                d = d_in[bi]
                h = h_in[bi]
                w = w_in[bi]
                if d_in_mask[bi] and h_in_mask[bi] and w_in_mask[bi]:
                    input_offset = (
                        b_idx * (C_in * D_in * H_in * W_in) +
                        c_in_idx[:, None] * (D_in * H_in * W_in) +
                        d * (H_in * W_in) +
                        h * W_in +
                        w
                    ).flatten()
                    # Load input values
                    x = tl.load(
                        X_ptr + input_offset,
                        mask=(c_in_mask_local[:, None] & valid_mask[None, :]).flatten(),
                        other=0.0
                    ).reshape(BLOCK_C_IN, 1)
                else:
                    x = tl.zeros((BLOCK_C_IN, 1), dtype=tl.float32)
                
                # Load weights: shape [BLOCK_C_IN, BLOCK_C_OUT, Kd, Kh, Kw]
                # Weight index calculation for transposed conv
                # W[ci, co//g, kd, kh, kw] where ci = c_in_idx, co = c_out_idx
                for kd in range(Kd):
                    for kh in range(Kh):
                        for kw in range(Kw):
                            # Compute weight indices
                            w_offset = (
                                c_in_idx[:, None, None, None, None] * (C_out * Kd * Kh * Kw) +
                                (c_out_idx[None, :, None, None, None] // groups) * (Kd * Kh * Kw) +
                                kd * (Kh * Kw) +
                                kh * Kw +
                                kw
                            )
                            w_val = tl.load(
                                W_ptr + w_offset,
                                mask=(c_in_mask_local[:, None, None, None, None] & 
                                      (c_out_idx[None, :, None, None, None] < C_out) &
                                      (kd == Kd - 1 - (d_grid[bi] - (d_idx[None, :] + stride_d * kd - stride_d * pad_d + out_pad_d)) // stride_d) &
                                       (kh == Kh - 1 - (h_grid[bi] - (h_idx[None, :] + stride_h * kh - stride_h * pad_h + out_pad_h)) // stride_h) &
                                       (kw == Kw - 1 - (w_grid[bi] - (w_idx[None, :] + stride_w * kw - stride_w * pad_w + out_pad_w)) // stride_w)),
                                other=0.0
                            )
                            
                            # Accumulate: for transposed conv, we need to match the right kernel element
                            # The kernel element at position (kd, kh, kw) contributes to output position
                            # where d_in = d_out*stride - (Kd-1-kd) + pad - out_pad
                            if kd == Kd - 1 - (d_grid[bi] - (d_idx[None, :] + stride_d * kd - stride_d * pad_d + out_pad_d)) // stride_d and \
                               kh == Kh - 1 - (h_grid[bi] - (h_idx[None, :] + stride_h * kh - stride_h * pad_h + out_pad_h)) // stride_h and \
                               kw == Kw - 1 - (w_grid[bi] - (w_idx[None, :] + stride_w * kw - stride_w * pad_w + out_pad_w)) // stride_w:
                                acc += x * w_val[:, :, None]
    
    # Reshape accumulator
    acc = acc.reshape(BLOCK_B, BLOCK_C_OUT, BLOCK_D, BLOCK_H, BLOCK_W)
    
    # Add bias if present
    if B_ptr is not None:
        bias = tl.load(B_ptr + c_out_idx, mask=c_out_mask).reshape(1, BLOCK_C_OUT, 1, 1, 1)
        acc += bias
    
    # Store output
    for bc in range(BLOCK_B):
        b_idx = batch_idx[bc] if bc < BLOCK_B and pid_b * BLOCK_B + bc < B else 0
        for bo in range(BLOCK_C_OUT):
            c_out = c_out_idx[bo]
            if c_out < C_out:
                for bd in range(BLOCK_D):
                    d = d_idx[bd]
                    if d < D_out:
                        for bh in range(BLOCK_H):
                            h = h_idx[bh]
                            if h < H_out:
                                for bw in range(BLOCK_W):
                                    w = w_idx[bw]
                                    if w < W_out:
                                        output_offset = (
                                            b_idx * (C_out * D_out * H_out * W_out) +
                                            c_out * (D_out * H_out * W_out) +
                                            d * (H_out * W_out) +
                                            h * W_out +
                                            w
                                        )
                                        tl.store(Y_ptr + output_offset, acc[bc, bo, bd, bh, bw])


def triton_conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups):
    """
    Custom Triton implementation of ConvTranspose3d.
    """
    # Extract dimensions
    B, C_in, D_in, H_in, W_in = x.shape
    C_in_w, C_out_group, Kd, Kh, Kw = weight.shape
    C_out = C_in_w * groups
    
    # Calculate output dimensions
    D_out = (D_in - 1) * stride[0] - 2 * padding[0] + (Kd - 1) + output_padding[0] + 1
    H_out = (H_in - 1) * stride[1] - 2 * padding[1] + (Kh - 1) + output_padding[1] + 1
    W_out = (W_in - 1) * stride[2] - 2 * padding[2] + (Kw - 1) + output_padding[2] + 1
    
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Prepare output tensor
    y = torch.empty(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Configure kernel launch parameters
    BLOCK_B = 1
    BLOCK_C_OUT = 16
    BLOCK_C_IN = 8
    BLOCK_D = 4
    BLOCK_H = 4
    BLOCK_W = 4
    
    grid = (
        (B + BLOCK_B - 1) // BLOCK_B,
        (C_out + BLOCK_C_OUT - 1) // BLOCK_C_OUT,
        (D_out + BLOCK_D - 1) // BLOCK_D,
        (H_out + BLOCK_H - 1) // BLOCK_H,
        (W_out + BLOCK_W - 1) // BLOCK_W,
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, y,
        B, C_in, C_out, groups,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        Kd, Kh, Kw,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        BLOCK_B=BLOCK_B,
        BLOCK_C_OUT=BLOCK_C_OUT,
        BLOCK_C_IN=BLOCK_C_IN,
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for ConvTranspose3d.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Create the same convolution layer
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, 
                                                   stride=stride, padding=padding, 
                                                   output_padding=output_padding, 
                                                   groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the same parameters as the original layer
        return triton_conv_transpose3d(
            x,
            self.conv_transpose3d.weight,
            self.conv_transpose3d.bias,
            self.conv_transpose3d.stride,
            self.conv_transpose3d.padding,
            self.conv_transpose3d.output_padding,
            self.conv_transpose3d.groups
        )