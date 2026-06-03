import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose3d_kernel(
    input_ptr,  # Input tensor: (B, C_in, D, H, W)
    weight_ptr,  # Weight tensor: (C_in, C_out, Kd, Kh, Kw)
    bias_ptr,  # Bias tensor: (C_out,) or NULL
    output_ptr,  # Output tensor: (B, C_out, D_out, H_out, W_out)
    B, C_in, C_out,
    D, H, W,
    Kd, Kh, Kw,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_pad_d, out_pad_h, out_pad_w,
    D_out, H_out, W_out,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Program IDs for batch, output channels, and spatial dimensions
    batch_idx = tl.program_id(0)
    c_out_idx = tl.program_id(1)
    
    # Compute starting positions for the output spatial dimensions
    start_d = tl.program_id(2) * BLOCK_SIZE_D
    start_h = tl.program_id(3) * BLOCK_SIZE_H
    start_w = tl.program_id(4) * BLOCK_SIZE_W
    
    # Create ranges for the blocks
    offsets_d = start_d + tl.arange(0, BLOCK_SIZE_D)
    offsets_h = start_h + tl.arange(0, BLOCK_SIZE_H)
    offsets_w = start_w + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for valid indices
    mask_d = offsets_d < D_out
    mask_h = offsets_h < H_out
    mask_w = offsets_w < W_out
    
    # Broadcast masks
    mask_dhw = mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]
    
    # Output accumulator
    output = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Iterate over input channels
    for c_in_idx in range(0, C_in, BLOCK_SIZE_C_IN):
        # Load input block
        input_offsets = (
            batch_idx * C_in * D * H * W +
            (c_in_idx + tl.arange(0, BLOCK_SIZE_C_IN)[:, None, None, None]) * D * H * W +
            (offsets_d[None, :, None, None] * stride_d - pad_d + tl.arange(0, Kd)[:, None, None]) * H * W +
            (offsets_h[None, None, :, None] * stride_h - pad_h + tl.arange(0, Kh)[None, :, None]) * W +
            (offsets_w[None, None, None, :] * stride_w - pad_w + tl.arange(0, Kw)[None, None, :])
        )
        
        # Reshape input_offsets for proper indexing
        input_offsets = input_offsets.reshape(BLOCK_SIZE_C_IN, BLOCK_SIZE_D * BLOCK_SIZE_H * BLOCK_SIZE_W)
        mask_input = (
            (offsets_d[None, :, None, None] * stride_d - pad_d + tl.arange(0, Kd)[:, None, None]) >= 0 & 
            (offsets_d[None, :, None, None] * stride_d - pad_d + tl.arange(0, Kd)[:, None, None]) < D &
            (offsets_h[None, None, :, None] * stride_h - pad_h + tl.arange(0, Kh)[None, :, None]) >= 0 & 
            (offsets_h[None, None, :, None] * stride_h - pad_h + tl.arange(0, Kh)[None, :, None]) < H &
            (offsets_w[None, None, None, :] * stride_w - pad_w + tl.arange(0, Kw)[None, None, :]) >= 0 & 
            (offsets_w[None, None, None, :] * stride_w - pad_w + tl.arange(0, Kw)[None, None, :]) < W
        )
        mask_input = mask_input.reshape(BLOCK_SIZE_C_IN, BLOCK_SIZE_D * BLOCK_SIZE_H * BLOCK_SIZE_W)
        
        input_block = tl.load(
            input_ptr + input_offsets,
            mask=mask_input & (c_in_idx + tl.arange(0, BLOCK_SIZE_C_IN)[:, None] < C_in),
            other=0.0
        )
        
        # Load weight block
        weight_offsets = (
            (c_in_idx + tl.arange(0, BLOCK_SIZE_C_IN)[:, None, None, None]) * C_out * Kd * Kh * Kw +
            c_out_idx * Kd * Kh * Kw +
            tl.arange(0, Kd)[:, None, None] * Kh * Kw +
            tl.arange(0, Kh)[None, :, None] * Kw +
            tl.arange(0, Kw)[None, None, :]
        )
        
        weight_block = tl.load(
            weight_ptr + weight_offsets,
            mask=c_in_idx + tl.arange(0, BLOCK_SIZE_C_IN)[:, None, None, None] < C_in,
            other=0.0
        )
        
        # Compute convolution: output[d,h,w] += sum_cin(input[d+kd, h+kh, w+kw] * weight[cin, cout, kd, kh, kw])
        # Note: for transposed convolution, we're essentially doing:
        # output[c_out, d_out, h_out, w_out] = sum_{cin, kd, kh, kw} input[cin, d_out*stride_d - pad_d + kd, ...] * weight[cin, c_out, kd, kh, kw]
        
        # Reshape for multiplication
        input_reshaped = input_block.reshape(BLOCK_SIZE_C_IN, BLOCK_SIZE_D * BLOCK_SIZE_H * BLOCK_SIZE_W)
        weight_reshaped = weight_block.reshape(BLOCK_SIZE_C_IN, BLOCK_SIZE_D * BLOCK_SIZE_H * BLOCK_SIZE_W)
        
        # Accumulate
        output += tl.sum(input_reshaped * weight_reshaped, axis=0).reshape(BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W)
    
    # Add bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + c_out_idx)
        output += bias
    
    # Store output
    output_offsets = (
        batch_idx * C_out * D_out * H_out * W_out +
        c_out_idx * D_out * H_out * W_out +
        offsets_d[:, None, None] * H_out * W_out +
        offsets_h[None, :, None] * W_out +
        offsets_w[None, None, :]
    )
    
    tl.store(
        output_ptr + output_offsets,
        output.to(tl.float32),
        mask=mask_dhw
    )


def triton_conv_transpose3d(x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
    """
    Custom Triton implementation of ConvTranspose3d
    
    Args:
        x: Input tensor of shape (B, C_in, D, H, W)
        weight: Weight tensor of shape (C_in, C_out, Kd, Kh, Kw)
        bias: Optional bias tensor of shape (C_out,)
        stride: Stride of the convolution
        padding: Padding applied to the input
        output_padding: Additional size added to one side of the output shape
        groups: Number of groups (must be 1 for this implementation)
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    B, C_in, D, H, W = x.shape
    C_in_w, C_out, Kd, Kh, Kw = weight.shape
    
    # Calculate output dimensions
    D_out = (D - 1) * stride - 2 * padding + Kd + output_padding
    H_out = (H - 1) * stride - 2 * padding + Kh + output_padding
    W_out = (W - 1) * stride - 2 * padding + Kw + output_padding
    
    # Create output tensor
    output = torch.empty(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Set block sizes for optimization
    BLOCK_SIZE_D = 4
    BLOCK_SIZE_H = 4
    BLOCK_SIZE_W = 4
    BLOCK_SIZE_C_OUT = 16
    BLOCK_SIZE_C_IN = 16
    
    # Grid dimensions: (batch, output_channels, output_depth_blocks, output_height_blocks, output_width_blocks)
    grid = (
        B,
        C_out,
        (D_out + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D,
        (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, output,
        B, C_in, C_out,
        D, H, W,
        Kd, Kh, Kw,
        stride, stride, stride,
        padding, padding, padding,
        output_padding, output_padding, output_padding,
        D_out, H_out, W_out,
        BLOCK_SIZE_B=1,
        BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
        BLOCK_SIZE_C_IN=BLOCK_SIZE_C_IN,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for ConvTranspose3d.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Create the same ConvTranspose3d layer but we'll override the forward pass
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using our custom Triton kernel.
        """
        return triton_conv_transpose3d(
            x,
            self.conv_transpose3d.weight,
            self.conv_transpose3d.bias if self.conv_transpose3d.bias is not None else None,
            stride=self.conv_transpose3d.stride,
            padding=self.conv_transpose3d.padding,
            output_padding=self.conv_transpose3d.output_padding,
            groups=self.conv_transpose3d.groups
        )