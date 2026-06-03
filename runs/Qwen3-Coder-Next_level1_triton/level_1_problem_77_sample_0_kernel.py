import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # Input tensor: (B, C_in, D, H, W)
    w_ptr,  # Weight tensor: (C_in, C_out, Kd, Kh, Kw)
    b_ptr,  # Bias tensor: (C_out,) or None
    y_ptr,  # Output tensor: (B, C_out, D_out, H_out, W_out)
    B, C_in, C_out, 
    D, H, W,
    D_out, H_out, W_out,
    Kd, Kh, Kw,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr = 1,
    BLOCK_SIZE_COUT: tl.constexpr = 32,
    BLOCK_SIZE_CIN: tl.constexpr = 8,
    BLOCK_SIZE_D: tl.constexpr = 4,
    BLOCK_SIZE_H: tl.constexpr = 4,
    BLOCK_SIZE_W: tl.constexpr = 4,
    BLOCK_SIZE_KD: tl.constexpr = 3,
    BLOCK_SIZE_KH: tl.constexpr = 3,
    BLOCK_SIZE_KW: tl.constexpr = 3,
):
    # Program IDs
    pid_b = tl.program_id(0)  # Batch index
    pid_cout = tl.program_id(1)  # Output channel group
    pid_d = tl.program_id(2)  # Output depth position
    pid_h = tl.program_id(3)  # Output height position
    pid_w = tl.program_id(4)  # Output width position
    
    # Calculate output position
    out_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for output dimensions
    mask_d = out_d < D_out
    mask_h = out_h < H_out
    mask_w = out_w < W_out
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over input channels
    for c_in_start in range(0, C_in, BLOCK_SIZE_CIN):
        c_in_ids = c_in_start + tl.arange(0, BLOCK_SIZE_CIN)
        mask_cin = c_in_ids < C_in
        
        # Create mask for input position calculation
        # For transposed convolution: input_pos = (output_pos - kernel_pos + stride*padding) / stride
        # But more directly: output_pos = input_pos * stride + kernel_pos*(dilation-1) - padding
        
        # Calculate corresponding input positions for this output position
        # For each output position, we need to consider all input positions that contribute
        
        # Actually, for transposed conv: we iterate over kernel positions and input positions
        # output[d_out, h_out, w_out] += sum_{c_in, kd, kh, kw} input[d_in, h_in, w_in] * weight[c_in, c_out, kd, kh, kw]
        # where d_in = (d_out - (kd-1)*dil_d - 1 + pad_d) // stride_d + 1 (if valid)
        # This is complex, let's use the direct mapping:
        # d_in = (d_out - (kd-1)*dil_d - 1 + pad_d) // stride_d + 1
        # But only if the input position is valid and matches the stride
        
        # Alternative approach: for each kernel position, calculate which input position contributes
        for kd_start in range(0, Kd, BLOCK_SIZE_KD):
            kd_ids = kd_start + tl.arange(0, BLOCK_SIZE_KD)
            mask_kd = kd_ids < Kd
            
            for kh_start in range(0, Kh, BLOCK_SIZE_KH):
                kh_ids = kh_start + tl.arange(0, BLOCK_SIZE_KH)
                mask_kh = kh_ids < Kh
                
                for kw_start in range(0, Kw, BLOCK_SIZE_KW):
                    kw_ids = kw_start + tl.arange(0, BLOCK_SIZE_KW)
                    mask_kw = kw_ids < Kw
                    
                    # For each kernel position, calculate input positions
                    # d_in = (d_out - (kd-1)*dil_d - 1 + pad_d) // stride_d + 1
                    # But we need to handle the modulo condition
                    
                    # More efficient: compute d_in = (d_out - (kd-1)*dil_d - 1 + pad_d)
                    # and check if d_in is divisible by stride_d and in range
                    
                    # Calculate base input positions
                    d_in_base = pid_d * BLOCK_SIZE_D * stride_d - (kd_ids[None, None, :] - 1) * dil_d - 1 + pad_d
                    h_in_base = pid_h * BLOCK_SIZE_H * stride_h - (kh_ids[None, :] - 1) * dil_h - 1 + pad_h
                    w_in_base = pid_w * BLOCK_SIZE_W * stride_w - (kw_ids[:] - 1) * dil_w - 1 + pad_w
                    
                    # Add offsets
                    d_in_vals = d_in_base[None, :, :] + tl.arange(0, BLOCK_SIZE_D)[:, None, None] * stride_d
                    h_in_vals = h_in_base[:, None, :] + tl.arange(0, BLOCK_SIZE_H)[None, :, None] * stride_h
                    w_in_vals = w_in_base[:, :, None] + tl.arange(0, BLOCK_SIZE_W)[None, None, :] * stride_w
                    
                    # Check validity
                    mask_d_in = (d_in_vals >= 0) & (d_in_vals < D)
                    mask_h_in = (h_in_vals >= 0) & (h_in_vals < H)
                    mask_w_in = (w_in_vals >= 0) & (w_in_vals < W)
                    mask_valid = mask_d_in & mask_h_in & mask_w_in
                    
                    # Load input values
                    for i_cin in range(BLOCK_SIZE_CIN):
                        if c_in_start + i_cin < C_in:
                            c_in_id = c_in_start + i_cin
                            # Calculate input index
                            d_in = d_in_vals
                            h_in = h_in_vals
                            w_in = w_in_vals
                            
                            # Flatten indices for 5D tensor
                            # x_ptr offset = (b * C_in + c_in) * (D*H*W) + d * (H*W) + h * W + w
                            x_offset = ((pid_b * C_in + c_in_id) * (D * H * W) + 
                                       d_in * (H * W) + h_in * W + w_in)
                            
                            # Create mask for valid positions
                            x_mask = mask_valid & (d_in < D) & (h_in < H) & (w_in < W)
                            
                            # Load input - need to handle the fact that we have a 4D mask array
                            # We'll process element by element for simplicity in this implementation
                            # For better performance, we'd use vectorized loads and more complex masking
                            
                            # Actually, let's use a simpler approach that's more Triton-friendly
                            # Iterate over the output positions and check validity
                            for i_d in range(BLOCK_SIZE_D):
                                for i_h in range(BLOCK_SIZE_H):
                                    for i_w in range(BLOCK_SIZE_W):
                                        if (mask_d[i_d] and mask_h[i_h] and mask_w[i_w]):
                                            d_val = pid_d * BLOCK_SIZE_D + i_d
                                            h_val = pid_h * BLOCK_SIZE_H + i_h
                                            w_val = pid_w * BLOCK_SIZE_W + i_w
                                            
                                            # Calculate corresponding input position
                                            d_in_calc = d_val - (kd_ids - 1) * dil_d - 1 + pad_d
                                            h_in_calc = h_val - (kh_ids - 1) * dil_h - 1 + pad_h
                                            w_in_calc = w_val - (kw_ids - 1) * dil_w - 1 + pad_w
                                            
                                            # Check divisibility by stride
                                            valid_d = (d_in_calc >= 0) & (d_in_calc < D * stride_d) & ((d_in_calc % stride_d) == 0)
                                            valid_h = (h_in_calc >= 0) & (h_in_calc < H * stride_h) & ((h_in_calc % stride_h) == 0)
                                            valid_w = (w_in_calc >= 0) & (w_in_calc < W * stride_w) & ((w_in_calc % stride_w) == 0)
                                            
                                            d_in_final = d_in_calc // stride_d
                                            h_in_final = h_in_calc // stride_h
                                            w_in_final = w_in_calc // stride_w
                                            
                                            mask_final = valid_d & valid_h & valid_w & (d_in_final < D) & (h_in_final < H) & (w_in_final < W)
                                            
                                            # Load input value
                                            x_offset_val = ((pid_b * C_in + c_in_id) * (D * H * W) + 
                                                           d_in_final * (H * W) + h_in_final * W + w_in_final)
                                            x_val = tl.load(x_ptr + x_offset_val, mask=mask_final, other=0.0)
                                            
                                            # Load weight values for all kernel positions
                                            w_offset = ((c_in_id * C_out + pid_cout * BLOCK_SIZE_COUT) * (Kd * Kh * Kw) + 
                                                       kd_ids * (Kh * Kw) + kh_ids * Kw + kw_ids)
                                            w_vals = tl.load(w_ptr + w_offset, mask=mask_kd[None, None, :] & mask_kh[None, :, :] & mask_kw[:, :, :], other=0.0)
                                            
                                            # Accumulate
                                            for i_kd in range(BLOCK_SIZE_KD):
                                                for i_kh in range(BLOCK_SIZE_KH):
                                                    for i_kw in range(BLOCK_SIZE_KW):
                                                        if mask_kd[i_kd] and mask_kh[i_kh] and mask_kw[i_kw]:
                                                            # Only accumulate if the kernel position contributes
                                                            # We need to check if this kernel position and input position pair is valid
                                                            # This is getting complex, let's use a simpler approach
                                                            pass
    
    # Given the complexity of the direct approach, let's use a more standard implementation
    # that processes one output position at a time with proper masking
    
    # Let's rewrite with a cleaner approach
    # For each output position, accumulate contributions from all input positions and kernel positions
    
    # Calculate output position
    out_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    out_d = tl.where(out_d < D_out, out_d, D_out)
    out_h = tl.where(out_h < H_out, out_h, H_out)
    out_w = tl.where(out_w < W_out, out_w, W_out)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Iterate over input channels
    for c_in_idx in range(C_in):
        # Iterate over kernel positions
        for kd in range(Kd):
            for kh in range(Kh):
                for kw in range(Kw):
                    # Calculate input position that contributes to this output
                    # For transposed conv: output_pos = input_pos * stride + kernel_pos * (dilation-1) - padding
                    # So input_pos = (output_pos + padding - kernel_pos * (dilation-1)) / stride
                    
                    # Calculate input positions for all output positions in our block
                    in_d = (out_d[:, None, None] + pad_d - kd * dil_d) // stride_d
                    in_h = (out_h[None, :, None] + pad_h - kh * dil_h) // stride_h
                    in_w = (out_w[None, None, :] + pad_w - kw * dil_w) // stride_w
                    
                    # Check if input position is valid and if output position matches exactly
                    # (i.e., the division was exact)
                    valid_d = ((out_d[:, None, None] + pad_d - kd * dil_d) % stride_d == 0) & (in_d >= 0) & (in_d < D)
                    valid_h = ((out_h[None, :, None] + pad_h - kh * dil_h) % stride_h == 0) & (in_h >= 0) & (in_h < H)
                    valid_w = ((out_w[None, None, :] + pad_w - kw * dil_w) % stride_w == 0) & (in_w >= 0) & (in_w < W)
                    
                    mask_valid = valid_d & valid_h & valid_w
                    
                    # Load input values where valid
                    in_d_valid = tl.where(mask_valid, in_d, 0)
                    in_h_valid = tl.where(mask_valid, in_h, 0)
                    in_w_valid = tl.where(mask_valid, in_w, 0)
                    
                    # Calculate offset
                    x_offset = ((pid_b * C_in + c_in_idx) * (D * H * W) + 
                               in_d_valid * (H * W) + in_h_valid * W + in_w_valid)
                    
                    # Load input
                    x_val = tl.load(x_ptr + x_offset, mask=mask_valid, other=0.0)
                    
                    # Load weight
                    w_offset = ((c_in_idx * C_out + pid_cout * BLOCK_SIZE_COUT) * (Kd * Kh * Kw) + 
                               kd * (Kh * Kw) + kh * Kw + kw)
                    w_val = tl.load(w_ptr + w_offset, other=0.0)
                    
                    # Accumulate
                    accumulator += x_val * w_val
    
    # Add bias if present
    if HAS_BIAS:
        bias_offset = pid_cout * BLOCK_SIZE_COUT + tl.arange(0, BLOCK_SIZE_COUT)
        bias_mask = bias_offset < C_out
        bias_val = tl.load(b_ptr + bias_offset, mask=bias_mask, other=0.0)
        accumulator += bias_val[None, None, :]
    
    # Store result
    y_offset = ((pid_b * C_out + pid_cout * BLOCK_SIZE_COUT) * (D_out * H_out * W_out) + 
               out_d[:, None, None] * (H_out * W_out) + out_h[None, :, None] * W_out + out_w[None, None, :])
    
    y_mask = (mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :])
    tl.store(y_ptr + y_offset, accumulator, mask=y_mask)


def triton_conv_transpose3d(x, weight, bias=None, stride=1, padding=0, dilation=1, output_padding=0):
    """
    Performs 3D transposed convolution using Triton kernel.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Get dimensions
    B, C_in, D, H, W = x.shape
    C_in2, C_out, Kd, Kh, Kw = weight.shape
    assert C_in == C_in2, "Input channels must match"
    
    # Calculate output dimensions
    # For transposed convolution: D_out = (D-1)*stride - 2*padding + dilation*(Kd-1) + output_padding + 1
    D_out = (D - 1) * stride - 2 * padding + dilation * (Kd - 1) + output_padding + 1
    H_out = (H - 1) * stride - 2 * padding + dilation * (Kh - 1) + output_padding + 1
    W_out = (W - 1) * stride - 2 * padding + dilation * (Kw - 1) + output_padding + 1
    
    # Prepare output tensor
    y = torch.empty(B, C_out, D_out, H_out, W_out, device=x.device, dtype=x.dtype)
    
    # Define grid
    # We'll use: [batch, output_channel_groups, depth_blocks, height_blocks, width_blocks]
    BLOCK_SIZE_B = 1
    BLOCK_SIZE_COUT = 32
    BLOCK_SIZE_D = 1
    BLOCK_SIZE_H = 1
    BLOCK_SIZE_W = 1
    
    grid = (
        B,
        (C_out + BLOCK_SIZE_COUT - 1) // BLOCK_SIZE_COUT,
        (D_out + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D,
        (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W,
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, y,
        B, C_in, C_out,
        D, H, W,
        D_out, H_out, W_out,
        Kd, Kh, Kw,
        stride, stride, stride,
        padding, padding, padding,
        dilation, dilation, dilation,
        HAS_BIAS=bias is not None,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_COUT=BLOCK_SIZE_COUT,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Performs a 3D transposed convolution operation with square input and square kernel,
    and supports padding, dilation, and stride using optimized Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, bias=bias)
        
        # Store parameters for manual implementation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize with the same weights and bias as the original layer
        with torch.no_grad():
            self.weight = self.conv_transpose3d.weight.data.clone()
            if bias:
                self.bias = self.conv_transpose3d.bias.data.clone()
            else:
                self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D transposed convolution using optimized Triton kernel.
        """
        # Use our optimized Triton implementation
        return triton_conv_transpose3d(
            x, self.weight, self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation
        )