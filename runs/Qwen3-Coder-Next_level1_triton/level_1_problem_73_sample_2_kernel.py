import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triton_conv_transpose3d_kernel(
    x_ptr,  # Input tensor: (B, C_in, D, H, W)
    w_ptr,  # Weight tensor: (C_in, C_out // G, Kd, Kh, Kw)
    b_ptr,  # Bias tensor: (C_out,)
    out_ptr,  # Output tensor: (B, C_out, D_out, H_out, W_out)
    B, C_in, C_out, G,  # Batch, channels, groups
    D, H, W,  # Input dimensions
    Kd, Kh, Kw,  # Kernel dimensions
    D_out, H_out, W_out,  # Output dimensions
    stride_d, stride_h, stride_w,  # Stride
    padding_d, padding_h, padding_w,  # Padding
    BLOCK_B: tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_KD: tl.constexpr,
    BLOCK_KH: tl.constexpr,
    BLOCK_KW: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    
    # Calculate output position
    out_idx = pid_b * (C_out * D_out * H_out * W_out) + pid_c_out * (D_out * H_out * W_out)
    
    # Process output in blocks
    c_out_offsets = pid_c_out * BLOCK_C_OUT + tl.arange(0, BLOCK_C_OUT)
    c_out_mask = c_out_offsets < C_out
    
    # For each output position
    for d_out in range(D_out):
        # Map output depth to input depth
        d_in = d_out - padding_d
        # Calculate kernel depth offset
        kd_offsets = tl.arange(0, BLOCK_KD)
        kd_mask = kd_offsets < Kd
        
        # Process height
        for h_out in range(H_out):
            h_in = h_out - padding_h
            kh_offsets = tl.arange(0, BLOCK_KH)
            kh_mask = kh_offsets < Kh
            
            # Process width
            for w_out in range(W_out):
                w_in = w_out - padding_w
                kw_offsets = tl.arange(0, BLOCK_KW)
                kw_mask = kw_offsets < Kw
                
                # Compute output index
                out_offset = out_idx + d_out * (H_out * W_out) + h_out * W_out + w_out
                
                # Accumulate over input channels and kernel dimensions
                acc = tl.zeros((BLOCK_C_OUT,), dtype=tl.float32)
                
                # Process input channels in blocks
                for c_in_start in range(0, C_in, BLOCK_C_IN):
                    c_in_offsets = c_in_start + tl.arange(0, BLOCK_C_IN)
                    c_in_mask = c_in_offsets < C_in
                    
                    # Process groups
                    for g in range(G):
                        g_start = g * (C_out // G)
                        g_end = (g + 1) * (C_out // G)
                        
                        # Check if current output channel is in this group
                        c_out_in_group = (c_out_offsets >= g_start) & (c_out_offsets < g_end) & c_out_mask
                        if tl.sum(c_out_in_group) > 0:
                            c_in_group = c_in_offsets
                            
                            # Calculate input position
                            if d_in >= 0 and d_in < D and h_in >= 0 and h_in < H and w_in >= 0 and w_in < W:
                                # Input index
                                x_offset = pid_b * (C_in * D * H * W) + c_in_group[:, None, None, None] * (D * H * W) + \
                                          d_in * (H * W) + h_in * W + w_in
                                
                                # Load input values
                                x_val = tl.load(x_ptr + x_offset, mask=c_in_mask[:, None, None, None], other=0.0)
                                
                                # Calculate weight indices
                                kd_range = d_in - d_out + Kd // 2 + kd_offsets[None, :, None, None]
                                kh_range = h_in - h_out + Kh // 2 + kh_offsets[None, None, :, None]
                                kw_range = w_in - w_out + Kw // 2 + kw_offsets[None, None, None, :]
                                
                                # Weight index
                                w_offset = c_in_group[:, None, None, None] * (C_out * Kd * Kh * Kw) + \
                                          c_out_offsets[None, :, None, None, None] * (Kd * Kh * Kw) + \
                                          kd_range * (Kh * Kw) + kh_range * Kw + kw_range
                                
                                # Load weights
                                w_val = tl.load(w_ptr + w_offset, mask=c_in_mask[:, None, None, None, None] & \
                                               (kd_range >= 0) & (kd_range < Kd) & \
                                               (kh_range >= 0) & (kh_range < Kh) & \
                                               (kw_range >= 0) & (kw_range < Kw), other=0.0)
                                
                                # Accumulate
                                acc += tl.sum(x_val[:, :, None, None, None] * w_val, axis=0)
                
                # Add bias if available
                if b_ptr is not None:
                    acc += tl.load(b_ptr + c_out_offsets, mask=c_out_mask)
                
                # Store result
                tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty), mask=c_out_mask)


def triton_conv_transpose3d(x, weight, bias, stride, padding, groups):
    """
    Performs 3D transposed convolution using Triton kernel.
    
    Args:
        x: Input tensor (B, C_in, D, H, W)
        weight: Weight tensor (C_in, C_out // G, Kd, Kh, Kw)
        bias: Bias tensor (C_out,) or None
        stride: Stride
        padding: Padding
        groups: Number of groups
    
    Returns:
        Output tensor (B, C_out, D_out, H_out, W_out)
    """
    B, C_in, D, H, W = x.shape
    C_in_w, C_out_g, Kd, Kh, Kw = weight.shape
    C_out = C_in_w * (C_out_g if C_out_g > 0 else 1)  # C_out = C_in_w * (C_out // G) * G / G
    
    # Calculate output dimensions
    D_out = (D - 1) * stride - 2 * padding + Kd
    H_out = (H - 1) * stride - 2 * padding + Kh
    W_out = (W - 1) * stride - 2 * padding + Kw
    
    # Prepare output tensor
    out = torch.empty((B, C_out, D_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Grid configuration
    BLOCK_B = 1
    BLOCK_C_OUT = 32
    BLOCK_KD = 3
    BLOCK_KH = 3
    BLOCK_KW = 3
    BLOCK_C_IN = 32
    
    grid = (B, (C_out + BLOCK_C_OUT - 1) // BLOCK_C_OUT)
    
    # Launch kernel
    triton_conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        B, C_in, C_out, groups,
        D, H, W,
        Kd, Kh, Kw,
        D_out, H_out, W_out,
        stride, stride, stride,
        padding, padding, padding,
        BLOCK_B=BLOCK_B,
        BLOCK_C_OUT=BLOCK_C_OUT,
        BLOCK_KD=BLOCK_KD,
        BLOCK_KH=BLOCK_KH,
        BLOCK_KW=BLOCK_KW,
        BLOCK_C_IN=BLOCK_C_IN,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the 3D transposed convolution with asymmetric input and square kernel.
    Uses custom Triton kernel for the convolution operation.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias
        
        # Initialize weights and bias
        # Note: PyTorch uses (in_channels, out_channels // groups, kD, kH, kW) format for conv_transpose3d
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize weights using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D transposed convolution.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Handle optional bias
        bias = self.bias if self.bias is not None else None
        
        # Use custom Triton kernel
        return triton_conv_transpose3d(
            x, weight, bias, 
            self.stride, self.padding, self.groups
        )


import math