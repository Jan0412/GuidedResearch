import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    # Pointers to inputs and outputs
    x_ptr,  # (batch_size, in_channels, D, H, W)
    w_ptr,  # (out_channels, in_channels // groups, Kd, Kh, Kw)
    b_ptr,  # (out_channels,) or None
    out_ptr,  # (batch_size, out_channels, D_out, H_out, W_out)
    # Dimensions
    B, C_in, C_out, groups,
    D_in, H_in, W_in,
    Kd, Kh, Kw,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    D_out, H_out, W_out,
    # Block sizes
    BLOCK_C_in: tl.constexpr,
    BLOCK_Kd: tl.constexpr,
    BLOCK_Kh: tl.constexpr,
    BLOCK_Kw: tl.constexpr,
    BLOCK_D_out: tl.constexpr,
    BLOCK_H_out: tl.constexpr,
    BLOCK_W_out: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    # Compute output position
    d_out = pid_d * BLOCK_D_out + tl.arange(0, BLOCK_D_out)
    h_out = pid_h * BLOCK_H_out + tl.arange(0, BLOCK_H_out)
    w_out = pid_w * BLOCK_W_out + tl.arange(0, BLOCK_W_out)
    
    # Create masks for output dimensions
    mask_d = d_out < D_out
    mask_h = h_out < H_out
    mask_w = w_out < W_out
    mask_dh = mask_d[:, None, None] & mask_h[None, :, None]
    mask = mask_dh[:, :, None] & mask_w[None, None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_D_out, BLOCK_H_out, BLOCK_W_out), dtype=tl.float32)
    
    # Compute group index and check if this output channel belongs to this group
    group_size_out = C_out // groups
    group_id = pid_c_out // group_size_out
    
    # Loop over input channels in groups
    for c_in_start in range(0, C_in, BLOCK_C_in):
        c_in_range = c_in_start + tl.arange(0, BLOCK_C_in)
        mask_c_in = c_in_range < C_in
        
        # Compute input channel indices for this group
        for g in range(groups):
            if g != group_id:
                continue
                
            # Compute input channel indices for this group and block
            c_in_g = g * (C_in // groups) + c_in_range
            
            # Load weights for this group
            w_kd = tl.arange(0, BLOCK_Kd)
            w_kh = tl.arange(0, BLOCK_Kh)
            w_kw = tl.arange(0, BLOCK_Kw)
            mask_kd = w_kd < Kd
            mask_kh = w_kh < Kh
            mask_kw = w_kw < Kw
            
            # Loop over kernel dimensions
            for kd in range(Kd):
                for kh in range(Kh):
                    for kw in range(Kw):
                        # Compute input positions
                        d_in = d_out * stride_d - pad_d + kd * dil_d
                        h_in = h_out * stride_h - pad_h + kh * dil_h
                        w_in = w_out * stride_w - pad_w + kw * dil_w
                        
                        # Masks for input positions
                        mask_d_in = d_in >= 0 & d_in < D_in
                        mask_h_in = h_in >= 0 & h_in < H_in
                        mask_w_in = w_in >= 0 & w_in < W_in
                        mask_in = mask_d_in[:, None, None] & mask_h_in[None, :, None] & mask_w_in[None, None, :]
                        
                        # Load input data
                        d_in_clamped = tl.maximum(tl.minimum(d_in, D_in - 1), 0)
                        h_in_clamped = tl.maximum(tl.minimum(h_in, H_in - 1), 0)
                        w_in_clamped = tl.maximum(tl.minimum(w_in, W_in - 1), 0)
                        
                        # Calculate input indices
                        x_indices = pid_b * (C_in * D_in * H_in * W_in) + \
                                   c_in_g[:, None, None, None] * (D_in * H_in * W_in) + \
                                   d_in_clamped[:, None, None] * (H_in * W_in) + \
                                   h_in_clamped[None, :, None] * W_in + \
                                   w_in_clamped[None, None, :]
                        
                        x_vals = tl.load(
                            x_ptr + x_indices,
                            mask=mask_in[:, :, :, None] & mask_c_in[None, None, None, :],
                            other=0.0
                        )
                        
                        # Load weights
                        w_indices = pid_c_out * (C_in // groups * Kd * Kh * Kw) + \
                                   (c_in_g - g * (C_in // groups)) * (Kd * Kh * Kw) + \
                                   (kd * Kh * Kw) + (kh * Kw) + kw
                        w_vals = tl.load(
                            w_ptr + w_indices,
                            mask=mask_c_in,
                            other=0.0
                        )
                        
                        # Expand dimensions for broadcasting
                        w_vals = w_vals[None, None, None, :]
                        acc += tl.sum(x_vals * w_vals, axis=3)
    
    # Add bias if available
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_c_out)
        acc += bias
    
    # Store result
    out_indices = pid_b * (C_out * D_out * H_out * W_out) + \
                 pid_c_out * (D_out * H_out * W_out) + \
                 d_out[:, None, None] * (H_out * W_out) + \
                 h_out[None, :, None] * W_out + \
                 w_out[None, None, :]
    
    tl.store(
        out_ptr + out_indices,
        acc,
        mask=mask
    )


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for 3D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), 
                                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check if input is on CUDA and contiguous
        if not x.is_cuda or not x.is_contiguous():
            return self.conv3d(x)
        
        # Extract parameters from the existing conv3d layer
        B, C_in, D_in, H_in, W_in = x.shape
        C_out, _, Kd, Kh, Kw = self.conv3d.weight.shape
        
        stride_d = stride_h = stride_w = self.conv3d.stride[0]
        pad_d = pad_h = pad_w = self.conv3d.padding[0]
        dil_d = dil_h = dil_w = self.conv3d.dilation[0]
        groups = self.conv3d.groups
        
        # Calculate output dimensions
        D_out = (D_in + 2 * pad_d - dil_d * (Kd - 1) - 1) // stride_d + 1
        H_out = (H_in + 2 * pad_h - dil_h * (Kh - 1) - 1) // stride_h + 1
        W_out = (W_in + 2 * pad_w - dil_w * (Kw - 1) - 1) // stride_w + 1
        
        # Create output tensor
        out = torch.empty(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
        
        # Kernel configuration
        BLOCK_C_in = 16
        BLOCK_Kd = 3
        BLOCK_Kh = 3
        BLOCK_Kw = 3
        BLOCK_D_out = 4
        BLOCK_H_out = 4
        BLOCK_W_out = 4
        
        # Grid configuration
        grid = (
            B,  # batch dimension
            C_out,  # output channels
            (D_out + BLOCK_D_out - 1) // BLOCK_D_out,
            (H_out + BLOCK_H_out - 1) // BLOCK_H_out,
            (W_out + BLOCK_W_out - 1) // BLOCK_W_out
        )
        
        # Launch kernel
        conv3d_kernel[grid](
            x, self.conv3d.weight, 
            self.conv3d.bias if self.conv3d.bias is not None else None,
            out,
            B, C_in, C_out, groups,
            D_in, H_in, W_in,
            Kd, Kh, Kw,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            dil_d, dil_h, dil_w,
            D_out, H_out, W_out,
            BLOCK_C_in=BLOCK_C_in,
            BLOCK_Kd=BLOCK_Kd,
            BLOCK_Kh=BLOCK_Kh,
            BLOCK_Kw=BLOCK_Kw,
            BLOCK_D_out=BLOCK_D_out,
            BLOCK_H_out=BLOCK_H_out,
            BLOCK_W_out=BLOCK_W_out
        )
        
        return out