import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,              # Input tensor: (B, C_in, H_in, W_in)
    w_ptr,              # Weight tensor: (C_in, C_out // G, K_h, K_w)
    b_ptr,              # Bias tensor: (C_out,) or None
    out_ptr,            # Output tensor: (B, C_out, H_out, W_out)
    B, C_in, C_out,     # Batch size, input channels, output channels
    H_in, W_in,         # Input height, width
    H_out, W_out,       # Output height, width
    K_h, K_w,           # Kernel height, width
    stride_h, stride_w, # Stride
    pad_h, pad_w,       # Padding
    dil_h, dil_w,       # Dilation
    G,                  # Groups
    C_in_per_group: tl.constexpr,
    C_out_per_group: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Compute output position
    batch_idx = tl.program_id(0)
    out_h_start = tl.program_id(1) * BLOCK_SIZE_H
    out_w_start = tl.program_id(2) * BLOCK_SIZE_W
    group_idx = tl.program_id(3)
    
    # Compute input channel range for this group
    c_in_start = group_idx * C_in_per_group
    c_out_start = group_idx * C_out_per_group
    
    # Create output tile coordinates
    out_h_offsets = out_h_start + tl.arange(0, BLOCK_SIZE_H)
    out_w_offsets = out_w_start + tl.arange(0, BLOCK_SIZE_W)
    out_h_mask = out_h_offsets < H_out
    out_w_mask = out_w_offsets < W_out
    
    # Initialize accumulator
    out_offsets = (
        batch_idx * C_out * H_out * W_out +
        out_h_offsets[:, None] * W_out * C_out +
        out_w_offsets[None, :] * C_out +
        tl.arange(0, C_out_per_group)[None, :]
    )
    
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, C_out_per_group), dtype=tl.float32)
    
    # Loop over input channels in the group
    for c_in_offset in range(0, C_in_per_group, BLOCK_SIZE_C):
        c_in_c = c_in_offset + tl.arange(0, BLOCK_SIZE_C)
        c_in_mask = c_in_c < C_in_per_group
        
        # Input position for this channel
        in_h = (out_h_offsets[:, None] - K_h + 1 + pad_h - (c_in_c[None, :, None] % 1) * 0) // stride_h
        in_w = (out_w_offsets[None, :] - K_w + 1 + pad_w - (c_in_c[None, :, None] % 1) * 0) // stride_w
        
        # For each input position, accumulate contributions from kernel positions
        # We'll iterate over kernel positions instead for better memory access
        for kh in range(0, K_h, BLOCK_SIZE_KH):
            kh_offsets = kh + tl.arange(0, BLOCK_SIZE_KH)
            kh_mask = kh_offsets < K_h
            
            for kw in range(0, K_w, BLOCK_SIZE_KW):
                kw_offsets = kw + tl.arange(0, BLOCK_SIZE_KW)
                kw_mask = kw_offsets < K_w
                
                # Compute input positions
                in_h_pos = out_h_offsets[:, None, None] - (K_h - 1 - kh_offsets[None, :, None]) * dil_h - pad_h
                in_w_pos = out_w_offsets[None, :, None] - (K_w - 1 - kw_offsets[None, None, :]) * dil_w - pad_w
                
                # Check if input positions are valid
                valid_in = (
                    (in_h_pos >= 0) & 
                    (in_h_pos < H_in) &
                    (in_w_pos >= 0) & 
                    (in_w_pos < W_in)
                )
                
                # Load input values: [BLOCK_SIZE_H, BLOCK_SIZE_C, BLOCK_SIZE_KW]
                in_h_idx = in_h_pos // 1
                in_w_idx = in_w_pos // 1
                
                # Transpose indexing for better locality
                x_block_offsets = (
                    batch_idx * C_in * H_in * W_in +
                    (c_in_start + c_in_c[None, :, None]) * H_in * W_in +
                    in_h_idx * W_in +
                    in_w_idx
                )
                
                x_val = tl.load(
                    x_ptr + x_block_offsets,
                    mask=valid_in & c_in_mask[:, None],
                    other=0.0
                )
                
                # Load weight values: [BLOCK_SIZE_C, BLOCK_SIZE_KH, BLOCK_SIZE_KW, C_out_per_group]
                w_block_offsets = (
                    (c_in_start + c_in_c[:, None, None]) * (C_out // G) * K_h * K_w +
                    kh_offsets[None, :, None] * K_w * (C_out // G) +
                    kw_offsets[None, None, :] * (C_out // G) +
                    tl.arange(0, C_out_per_group)[None, None, :]
                )
                
                w_val = tl.load(
                    w_ptr + w_block_offsets,
                    mask=c_in_mask[:, None, None] & kh_mask[:, None] & kw_mask[None, :],
                    other=0.0
                )
                
                # Compute contribution: x_val [H, C, KW] * w_val [C, KH, KW, C_out]
                # Accumulate over C dimension
                contrib = tl.sum(x_val[:, :, :, None] * w_val[:, None, :, :], axis=2)  # [H, KH, C_out]
                contrib = tl.sum(contrib, axis=1)  # [H, C_out]
                
                acc += contrib
                
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + c_out_start + tl.arange(0, C_out_per_group))
        acc += bias[None, None, :]
    
    # Store result
    tl.store(
        out_ptr + out_offsets,
        acc.to(out_ptr.dtype.element_ty),
        mask=out_h_mask[:, None, None] & out_w_mask[None, :, None]
    )


def triton_conv_transpose2d(x, weight, bias, stride, padding, dilation, groups):
    """
    Custom Triton implementation of 2D transposed convolution.
    """
    B, C_in, H_in, W_in = x.shape
    C_in2, C_out_per_g, K_h, K_w = weight.shape
    assert C_in2 == C_in, "Input channels mismatch"
    
    C_out = C_out_per_g * groups
    
    # Compute output dimensions
    H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (K_h - 1) + 1
    W_out = (W_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (K_w - 1) + 1
    
    # Allocate output tensor
    out = torch.empty(B, C_out, H_out, W_out, device=x.device, dtype=x.dtype)
    
    # Block sizes for tiling
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    BLOCK_SIZE_C = 8
    BLOCK_SIZE_KH = 3
    BLOCK_SIZE_KW = 3
    
    # Grid dimensions: (batch, H_tiles, W_tiles, groups)
    grid = (
        B,
        (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W,
        groups
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias, out,
        B, C_in, C_out,
        H_in, W_in,
        H_out, W_out,
        K_h, K_w,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        groups,
        C_in_per_group=C_in // groups,
        C_out_per_group=C_out // groups,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_KH=BLOCK_SIZE_KH,
        BLOCK_SIZE_KW=BLOCK_SIZE_KW,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), 
                 padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters for reconstruction
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create weight and bias parameters
        # nn.ConvTranspose2d weight shape: (in_channels, out_channels // groups, *kernel_size)
        # We need to match this for compatibility
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels // groups, *kernel_size)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using custom Triton kernel.
        """
        return triton_conv_transpose2d(
            x, self.weight, self.bias, 
            self.stride, self.padding, self.dilation, self.groups
        )