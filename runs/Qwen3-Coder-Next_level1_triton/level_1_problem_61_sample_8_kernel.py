import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # Input tensor pointer (B, C_in, D, H, W)
    w_ptr,  # Weight tensor pointer (C_in, C_out, kD, kH, kW)
    b_ptr,  # Bias tensor pointer (C_out,) or None
    out_ptr,  # Output tensor pointer (B, C_out, D_out, H_out, W_out)
    B, C_in, C_out,
    D, H, W,
    D_out, H_out, W_out,
    kD, kH, kW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    output_pad_d, output_pad_h, output_pad_w,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_COUT: tl.constexpr,
    BLOCK_SIZE_CIN: tl.constexpr,
    BLOCK_SIZE_KD: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Get program IDs
    batch_id = tl.program_id(0)
    c_out_id = tl.program_id(1)
    
    # Calculate output position
    d_out = tl.program_id(2) // (H_out * W_out)
    rest = tl.program_id(2) % (H_out * W_out)
    h_out = rest // W_out
    w_out = rest % W_out
    
    # Compute the corresponding input position (accounting for stride and padding)
    d_in = d_out * stride_d - pad_d
    h_in = h_out * stride_h - pad_h
    w_in = w_out * stride_w - pad_w
    
    # Accumulator for the result
    acc = tl.zeros([BLOCK_SIZE_COUT], tl.float32)
    
    # Iterate over input channels and kernel dimensions
    for c_in_offset in range(0, C_in, BLOCK_SIZE_CIN):
        c_in_ids = c_in_offset + tl.arange(0, BLOCK_SIZE_CIN)
        c_in_mask = c_in_ids < C_in
        
        # Iterate over kernel depth
        for k_d in range(0, kD, BLOCK_SIZE_KD):
            k_d_ids = k_d + tl.arange(0, BLOCK_SIZE_KD)
            k_d_mask = k_d_ids < kD
            d_pos = d_in + k_d_ids * 1  # kernel dimension
            
            # Iterate over kernel height
            for k_h in range(0, kH, BLOCK_SIZE_KH):
                k_h_ids = k_h + tl.arange(0, BLOCK_SIZE_KH)
                k_h_mask = k_h_ids < kH
                h_pos = h_in + k_h_ids * 1
                
                # Iterate over kernel width
                for k_w in range(0, kW, BLOCK_SIZE_KW):
                    k_w_ids = k_w + tl.arange(0, BLOCK_SIZE_KW)
                    k_w_mask = k_w_ids < kW
                    w_pos = w_in + k_w_ids * 1
                    
                    # Create masks for valid input positions
                    d_mask = (d_pos >= 0) & (d_pos < D)
                    h_mask = (h_pos >= 0) & (h_pos < H)
                    w_mask = (w_pos >= 0) & (w_pos < W)
                    valid_mask = d_mask[:, None, None, None] & h_mask[None, :, None, None] & w_mask[None, None, :, None] & c_in_mask[None, None, None, :]
                    
                    # Compute offsets for input tensor
                    # Input shape: (B, C_in, D, H, W)
                    input_offsets = (
                        batch_id * (C_in * D * H * W) +
                        c_in_ids[None, None, None, :] * (D * H * W) +
                        d_pos[:, None, None, None] * (H * W) +
                        h_pos[None, :, None, None] * W +
                        w_pos[None, None, :, None]
                    )
                    
                    # Compute offsets for weight tensor
                    # Weight shape: (C_in, C_out, kD, kH, kW)
                    weight_offsets = (
                        c_in_ids[:, None, None, None] * (C_out * kD * kH * kW) +
                        c_out_id * (kD * kH * kW) +
                        k_d_ids[None, :, None, None] * (kH * kW) +
                        k_h_ids[None, None, :, None] * kW +
                        k_w_ids[None, None, None, :]
                    )
                    
                    # Load input values
                    x_val = tl.load(x_ptr + input_offsets, mask=valid_mask, other=0.0)
                    w_val = tl.load(w_ptr + weight_offsets, mask=c_in_mask[:, None, None, None] & k_d_mask[None, :, None, None] & k_h_mask[None, None, :, None] & k_w_mask[None, None, None, :], other=0.0)
                    
                    # Accumulate: output[b, c_out, d_out, h_out, w_out] += sum_{c_in, k_d, k_h, k_w} input[b, c_in, d_in+k_d, h_in+k_h, w_in+k_w] * weight[c_in, c_out, k_d, k_h, k_w]
                    acc += tl.sum(x_val * w_val, axis=[0, 1, 2, 3])
    
    # Store result
    out_offset = (
        batch_id * (C_out * D_out * H_out * W_out) +
        c_out_id * (D_out * H_out * W_out) +
        d_out * (H_out * W_out) +
        h_out * W_out +
        w_out
    )
    
    # Add bias if present
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + c_out_id)
        acc += bias_val
    
    tl.store(out_ptr + out_offset, acc[0])


class TritonConvTranspose3d(nn.Module):
    """
    Custom Triton implementation of ConvTranspose3d
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 output_padding=0, groups=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding, output_padding)
        self.groups = groups
        self.bias_flag = bias
        
        # Initialize weights (nn.init.kaiming_uniform_ is the default for ConvTranspose3d)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        # Get dimensions
        B, C_in, D, H, W = x.shape
        kD, kH, kW = self.kernel_size
        stride_d, stride_h, stride_w = self.stride
        pad_d, pad_h, pad_w = self.padding
        output_pad_d, output_pad_h, output_pad_w = self.output_padding
        
        # Calculate output dimensions (same as ConvTranspose3d)
        D_out = (D - 1) * stride_d - 2 * pad_d + (kD - 1) + output_pad_d + 1
        H_out = (H - 1) * stride_h - 2 * pad_h + (kH - 1) + output_pad_h + 1
        W_out = (W - 1) * stride_w - 2 * pad_w + (kW - 1) + output_pad_w + 1
        
        # Check input dimensions match
        assert C_in == self.in_channels, f"Expected {self.in_channels} channels, got {C_in}"
        
        # Prepare output tensor
        out = torch.empty(B, self.out_channels, D_out, H_out, W_out, device=x.device, dtype=x.dtype)
        
        # Set up block sizes (tunable parameters)
        BLOCK_SIZE_B = 1
        BLOCK_SIZE_COUT = 8
        BLOCK_SIZE_CIN = 16
        BLOCK_SIZE_KD = 3
        BLOCK_SIZE_KH = 3
        BLOCK_SIZE_KW = 3
        
        # Calculate grid dimensions
        grid = (B, self.out_channels, D_out * H_out * W_out)
        
        # Launch kernel
        conv_transpose3d_kernel[grid](
            x, self.weight, self.bias if self.bias is not None else None, out,
            B, self.in_channels, self.out_channels,
            D, H, W,
            D_out, H_out, W_out,
            kD, kH, kW,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            output_pad_d, output_pad_h, output_pad_w,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_COUT=BLOCK_SIZE_COUT,
            BLOCK_SIZE_CIN=BLOCK_SIZE_CIN,
            BLOCK_SIZE_KD=BLOCK_SIZE_KD,
            BLOCK_SIZE_KH=BLOCK_SIZE_KH,
            BLOCK_SIZE_KW=BLOCK_SIZE_KW,
        )
        
        return out


import math

class ModelNew(nn.Module):
    """
    Optimized Model using Triton kernel for transposed 3D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super().__init__()
        self.conv_transpose3d = TritonConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
            groups=groups, bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose3d(x)