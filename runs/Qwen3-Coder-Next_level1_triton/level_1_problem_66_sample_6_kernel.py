import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def conv3d_kernel(
    x_ptr,  # Input tensor: (B, C_in, D, H, W)
    w_ptr,  # Weight tensor: (C_out, C_in, kD, kH, kW)
    b_ptr,  # Bias tensor: (C_out,) or None
    out_ptr,  # Output tensor: (B, C_out, D_out, H_out, W_out)
    B, C_in, D, H, W,  # Input dimensions
    C_out, kD, kH, kW,  # Weight dimensions
    stride_d, stride_h, stride_w,  # Strides
    pad_d, pad_h, pad_w,  # Padding
    dil_d, dil_h, dil_w,  # Dilation
    C_out_numel,  # Total elements in output
    BLOCK_SIZE: tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
    BLOCK_KD: tl.constexpr,
    BLOCK_KH: tl.constexpr,
    BLOCK_KW: tl.constexpr,
):
    # Get output indices
    c_out_block_start = tl.program_id(0) * BLOCK_C_OUT
    spatial_idx = tl.program_id(1) * BLOCK_SIZE
    
    # Compute output channel range
    c_out_offsets = c_out_block_start + tl.arange(0, BLOCK_C_OUT)
    c_out_mask = c_out_offsets < C_out
    
    # Compute spatial position from linear index
    total_spatial = D * H * W
    spatial_offsets = spatial_idx + tl.arange(0, BLOCK_SIZE)
    mask = spatial_offsets < total_spatial
    
    # Convert linear spatial index to (d, h, w)
    d = spatial_offsets // (H * W)
    h_w_rem = spatial_offsets % (H * W)
    h = h_w_rem // W
    w = h_w_rem % W
    
    # Compute output spatial coordinates (considering stride and padding)
    out_d = d
    out_h = h
    out_w = w
    
    # Calculate input spatial coordinates
    in_d = out_d * stride_d - pad_d + d * dil_d
    in_h = out_h * stride_h - pad_h + h * dil_h
    in_w = out_w * stride_w - pad_w + w * dil_w
    
    # Accumulate over channels and kernel dimensions
    output = tl.zeros((BLOCK_C_OUT,), dtype=tl.float32)
    
    # Loop over input channels
    for c_in_start in range(0, C_in, BLOCK_C_IN):
        c_in_offsets = c_in_start + tl.arange(0, BLOCK_C_IN)
        c_in_mask = c_in_offsets < C_in
        
        # Loop over kernel depth
        for kd_start in range(0, kD, BLOCK_KD):
            kd_offsets = kd_start + tl.arange(0, BLOCK_KD)
            kd_mask = kd_offsets < kD
            
            # Compute input depth positions for this kernel depth
            in_d_k = in_d - kd_offsets * dil_d
            d_valid = (in_d_k >= 0) & (in_d_k < D)
            
            # Loop over kernel height
            for kh_start in range(0, kH, BLOCK_KH):
                kh_offsets = kh_start + tl.arange(0, BLOCK_KH)
                kh_mask = kh_offsets < kH
                
                # Compute input height positions for this kernel height
                in_h_k = in_h - kh_offsets * dil_h
                h_valid = (in_h_k >= 0) & (in_h_k < H)
                
                # Loop over kernel width
                for kw_start in range(0, kW, BLOCK_KW):
                    kw_offsets = kw_start + tl.arange(0, BLOCK_KW)
                    kw_mask = kw_offsets < kW
                    
                    # Compute input width positions for this kernel width
                    in_w_k = in_w - kw_offsets * dil_w
                    w_valid = (in_w_k >= 0) & (in_w_k < W)
                    
                    # Combine validity masks
                    valid = d_valid[:, None, None] & h_valid[:, None] & w_valid
                    
                    # Load input values
                    # x_ptr shape: (B, C_in, D, H, W)
                    # Calculate base offset for batch=0
                    base_offset = c_in_offsets[None, :, None, None, None] * (D * H * W) + \
                                  in_d_k[:, None, None, None] * (H * W) + \
                                  in_h_k[:, :, None, None] * W + \
                                  in_w_k[:, :, :, None]
                    
                    # Transpose base_offset for proper indexing
                    base_offset = base_offset.permute(1, 0, 2, 3)
                    c_in_flat = c_in_offsets[:, None, None, None]
                    
                    # Compute actual indices
                    indices = c_in_flat * (D * H * W) + in_d_k[:, None, None, None] * (H * W) + \
                              in_h_k[:, :, None, None] * W + in_w_k[:, :, :, None]
                    
                    # Reshape for broadcasting
                    indices = indices.permute(1, 0, 2, 3)
                    valid_flat = valid.permute(1, 0, 2, 3)
                    
                    # Load input
                    x_offsets = indices.flatten()
                    x_mask_flat = valid_flat.flatten()
                    
                    # Simplified approach: load inputs and weights directly
                    # For efficiency, we'll use a more straightforward implementation
                    
                    # Compute weight indices: (C_out, C_in, kD, kH, kW)
                    c_out_k = c_out_offsets[:, None, None, None, None]
                    c_in_k = c_in_offsets[None, :, None, None, None]
                    kd_k = kd_offsets[None, None, :, None, None]
                    kh_k = kh_offsets[None, None, None, :, None]
                    kw_k = kw_offsets[None, None, None, None, :]
                    
                    w_indices = c_out_k * (C_in * kD * kH * kW) + \
                                c_in_k * (kD * kH * kW) + \
                                kd_k * (kH * kW) + \
                                kh_k * kW + \
                                kw_k
                    
                    w_offsets = w_indices.flatten()
                    
                    # Load weight values
                    w_vals = tl.load(w_ptr + w_offsets, mask=w_indices < (C_out * C_in * kD * kH * kW), other=0.0)
                    w_vals = w_vals.reshape(BLOCK_C_OUT, BLOCK_C_IN, BLOCK_KD, BLOCK_KH, BLOCK_KW)
                    
                    # Load input values for valid positions
                    # For the current spatial position and kernel offset
                    d_idx = out_d
                    h_idx = out_h
                    w_idx = out_w
                    in_d_idx = d_idx * stride_d - pad_d + kd_start * dil_d
                    in_h_idx = h_idx * stride_h - pad_h + kh_start * dil_h
                    in_w_idx = w_idx * stride_w - pad_w + kw_start * dil_w
                    
                    # Calculate input offset
                    input_offset = in_d_idx * (H * W) + in_h_idx * W + in_w_idx
                    
                    # Load input for all input channels
                    x_vals = tl.load(x_ptr + c_in_offsets * (D * H * W) + input_offset, 
                                   mask=c_in_mask, other=0.0)
                    
                    # Compute dot product
                    # w_vals shape: (BLOCK_C_OUT, BLOCK_C_IN, BLOCK_KD, BLOCK_KH, BLOCK_KW)
                    # For simplicity, only process valid kernel positions
                    if kd_start + BLOCK_KD <= kD and kh_start + BLOCK_KH <= kH and kw_start + BLOCK_KW <= kW:
                        # Full kernel
                        kernel_weight = w_vals
                        # Reshape for multiplication: (BLOCK_C_OUT, BLOCK_C_IN, BLOCK_KD, BLOCK_KH, BLOCK_KW)
                        # and (BLOCK_C_IN, 1, 1, 1, 1) -> broadcast
                        kernel_sum = tl.sum(kernel_weight * x_vals[None, :, None, None, None], axis=1)
                        output += tl.sum(kernel_sum, axis=[1, 2, 3])
                    else:
                        # Partial kernel - need to handle masks
                        for kd in range(BLOCK_KD):
                            for kh in range(BLOCK_KH):
                                for kw in range(BLOCK_KW):
                                    if kd_start + kd < kD and kh_start + kh < kH and kw_start + kw < kW:
                                        # Valid kernel position
                                        w_idx_k = c_out_offsets[:, None] * (C_in * kD * kH * kW) + \
                                                  c_in_offsets[None, :] * (kD * kH * kW) + \
                                                  (kd_start + kd) * (kH * kW) + \
                                                  (kh_start + kh) * kW + \
                                                  (kw_start + kw)
                                        w_vals_k = tl.load(w_ptr + w_idx_k, mask=c_out_mask[:, None] & c_in_mask[None, :], other=0.0)
                                        x_val = tl.load(x_ptr + c_in_offsets * (D * H * W) + 
                                                      (in_d + (kd_start + kd) * dil_d) * (H * W) + 
                                                      (in_h + (kh_start + kh) * dil_h) * W + 
                                                      (in_w + (kw_start + kw) * dil_w),
                                                      mask=c_in_mask, other=0.0)
                                        output += tl.sum(w_vals_k * x_val[None, :], axis=1)
    
    # Add bias if available
    if b_ptr is not None:
        bias = tl.load(b_ptr + c_out_offsets, mask=c_out_mask, other=0.0)
        output += bias
    
    # Store results
    if c_out_block_start + BLOCK_C_OUT <= C_out:
        # Full block
        out_offsets = c_out_offsets[:, None] * total_spatial + spatial_offsets[None, :]
        out_mask = c_out_mask[:, None] & mask[None, :]
        tl.store(out_ptr + out_offsets, output.T, mask=out_mask)
    else:
        # Handle edge case for last block
        out_offsets = c_out_offsets[:, None] * total_spatial + spatial_offsets[None, :]
        out_mask = c_out_mask[:, None] & mask[None, :]
        tl.store(out_ptr + out_offsets, output.T, mask=out_mask)


def triton_conv3d(x, weight, bias, stride, padding, dilation, groups):
    """
    Custom Triton implementation of 3D convolution
    """
    B, C_in, D, H, W = x.shape
    C_out, _, kD, kH, kW = weight.shape
    
    # Calculate output dimensions
    D_out = (D + 2 * padding[0] - dilation[0] * (kD - 1) - 1) // stride[0] + 1
    H_out = (H + 2 * padding[1] - dilation[1] * (kH - 1) - 1) // stride[1] + 1
    W_out = (W + 2 * padding[2] - dilation[2] * (kW - 1) - 1) // stride[2] + 1
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Prepare output tensor
    out = torch.empty(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Flatten spatial dimensions for easier processing
    total_spatial = D_out * H_out * W_out
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 128
    BLOCK_C_OUT = min(16, C_out)
    BLOCK_C_IN = min(8, C_in)
    BLOCK_KD = 3
    BLOCK_KH = 5
    BLOCK_KW = 7
    
    # Grid dimensions
    grid = lambda meta: (
        triton.cdiv(C_out, meta["BLOCK_C_OUT"]),
        triton.cdiv(total_spatial, meta["BLOCK_SIZE"]),
    )
    
    # Launch kernel
    conv3d_kernel[grid](
        x, weight, bias, out,
        B, C_in, D, H, W,
        C_out, kD, kH, kW,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        total_spatial * C_out,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_C_OUT=BLOCK_C_OUT,
        BLOCK_C_IN=BLOCK_C_IN,
        BLOCK_KD=BLOCK_KD,
        BLOCK_KH=BLOCK_KH,
        BLOCK_KW=BLOCK_KW,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model with custom Triton kernel for 3D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters for reference
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights (simple initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D convolution using Triton kernel.
        """
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Call our custom Triton convolution
        return triton_conv3d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )