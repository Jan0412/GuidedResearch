import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    # Pointers to inputs and outputs
    x_ptr,  # (batch, in_channels, depth, height, width)
    w_ptr,  # (in_channels, out_channels//groups, kernel_size, kernel_size, kernel_size)
    b_ptr,  # (out_channels,) optional
    out_ptr,  # (batch, out_channels, depth_out, height_out, width_out)
    # Dimensions
    batch_size, in_channels, out_channels, 
    in_d, in_h, in_w,
    out_d, out_h, out_w,
    kernel_size, stride, padding, groups,
    # Block sizes
    BLOCK_BATCH: tl.constexpr,
    BLOCK_OUT_CH: tl.constexpr,
    BLOCK_IN_CH: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    pid_block_d = tl.program_id(2)
    pid_block_h = tl.program_id(3)
    pid_block_w = tl.program_id(4)
    
    # Calculate output position ranges
    out_d_start = pid_block_d * BLOCK_D
    out_h_start = pid_block_h * BLOCK_H
    out_w_start = pid_block_w * BLOCK_W
    
    # Output position indices
    out_d_idx = out_d_start + tl.arange(0, BLOCK_D)
    out_h_idx = out_h_start + tl.arange(0, BLOCK_H)
    out_w_idx = out_w_start + tl.arange(0, BLOCK_W)
    
    # Create meshgrid for output positions
    out_d_grid, out_h_grid, out_w_grid = tl.meshgrid(out_d_idx, out_h_idx, out_w_idx, indexing='ij')
    
    # Convert to 1D arrays for easier handling
    out_d_flat = out_d_grid.flatten()
    out_h_flat = out_h_grid.flatten()
    out_w_flat = out_w_grid.flatten()
    
    # Mask for valid output positions
    mask_d = out_d_flat < out_d
    mask_h = out_h_flat < out_h
    mask_w = out_w_flat < out_w
    mask = mask_d & mask_h & mask_w
    
    # Calculate corresponding input positions
    in_d_flat = (out_d_flat - padding) // stride
    in_h_flat = (out_h_flat - padding) // stride
    in_w_flat = (out_w_flat - padding) // stride
    
    # Check if input positions are within bounds
    mask_in_d = (in_d_flat >= 0) & (in_d_flat < in_d)
    mask_in_h = (in_h_flat >= 0) & (in_h_flat < in_h)
    mask_in_w = (in_w_flat >= 0) & (in_w_flat < in_w)
    mask_valid = mask & mask_in_d & mask_in_h & mask_in_w
    
    # Calculate stride offsets
    d_offset = (out_d_flat - padding) % stride
    h_offset = (out_h_flat - padding) % stride
    w_offset = (out_w_flat - padding) % stride
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_D * BLOCK_H * BLOCK_W,), dtype=tl.float32)
    
    # Process input channels in blocks
    for block_in_ch in range(0, in_channels, BLOCK_IN_CH):
        in_ch_idx = block_in_ch + tl.arange(0, BLOCK_IN_CH)
        in_ch_grid, _ = tl.meshgrid(in_ch_idx, tl.arange(0, BLOCK_D * BLOCK_H * BLOCK_W), indexing='ij')
        in_ch_flat = in_ch_grid.flatten()
        
        # Create mask for valid channels (handle last block)
        ch_mask = in_ch_flat < in_channels
        
        # Calculate input positions for this channel block
        in_d_pos = in_d_flat * (in_h * in_w) + in_h_flat * in_w + in_w_flat
        in_idx = pid_batch * (in_channels * in_d * in_h * in_w) + in_ch_flat * (in_d * in_h * in_w) + in_d_pos
        
        # Load input values
        x_vals = tl.load(x_ptr + in_idx, mask=mask_valid & ch_mask, other=0.0)
        
        # Process kernel positions
        for kd in range(kernel_size):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    # Check if this kernel position corresponds to the stride offset
                    is_match = (d_offset == kd) & (h_offset == kh) & (w_offset == kw)
                    
                    # Calculate weight indices
                    # Weight shape: (in_channels, out_channels//groups, kernel_size, kernel_size, kernel_size)
                    # For grouped convolution: groups = 4, so each group processes in_channels//groups output channels
                    
                    # Calculate group index for this input channel
                    group_idx = in_ch_flat // (in_channels // groups)
                    local_in_ch = in_ch_flat % (in_channels // groups)
                    
                    # Weight index: (local_in_ch, pid_out_ch, kd, kh, kw)
                    # But need to adjust for groups
                    w_idx = local_in_ch * (out_channels * kernel_size * kernel_size * kernel_size) + \
                           pid_out_ch * (kernel_size * kernel_size * kernel_size) + \
                           kd * (kernel_size * kernel_size) + kh * kernel_size + kw
                    
                    # Load weight
                    w_vals = tl.load(w_ptr + w_idx, mask=ch_mask, other=0.0)
                    
                    # Accumulate
                    acc += tl.where(mask_valid & is_match & ch_mask, x_vals * w_vals, 0.0)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_out_ch)
        acc += bias
    
    # Store results
    acc = acc.reshape((BLOCK_D, BLOCK_H, BLOCK_W))
    out_idx = pid_batch * (out_channels * out_d * out_h * out_w) + \
              pid_out_ch * (out_d * out_h * out_w) + \
              out_d_grid * (out_h * out_w) + out_h_grid * out_w + out_w_grid
    
    tl.store(out_ptr + out_idx, acc.to(x_ptr.dtype.element_ty), mask=mask)


def triton_conv_transpose3d(x, weight, bias=None, stride=1, padding=0, groups=1):
    """
    Custom Triton implementation of 3D transposed convolution.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    batch_size, in_channels, in_d, in_h, in_w = x.shape
    # Weight shape: (in_channels, out_channels//groups, kernel_size, kernel_size, kernel_size)
    out_channels = weight.shape[1] * groups
    
    # Calculate output dimensions
    out_d = (in_d - 1) * stride - 2 * padding + (kernel_size := weight.shape[2]) + max(0, 0)  # output_padding=0
    out_h = (in_h - 1) * stride - 2 * padding + kernel_size
    out_w = (in_w - 1) * stride - 2 * padding + kernel_size
    
    # Create output tensor
    out = torch.empty(batch_size, out_channels, out_d, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Configure grid
    BLOCK_BATCH = 1
    BLOCK_OUT_CH = 8
    BLOCK_IN_CH = 8
    BLOCK_D = 4
    BLOCK_H = 4
    BLOCK_W = 8
    
    # Grid dimensions
    grid = (
        batch_size,
        out_channels,
        (out_d + BLOCK_D - 1) // BLOCK_D,
        (out_h + BLOCK_H - 1) // BLOCK_H,
        (out_w + BLOCK_W - 1) // BLOCK_W
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding, groups,
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_OUT_CH=BLOCK_OUT_CH,
        BLOCK_IN_CH=BLOCK_IN_CH,
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the 3D transposed convolution model using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store original ConvTranspose3d layer for parameter access
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), 
                                                   stride=stride, padding=padding, groups=groups, bias=bias)
        # Mark that we'll use custom kernel instead of PyTorch implementation
        self.use_custom_kernel = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_custom_kernel:
            # Extract parameters from the original layer
            weight = self.conv_transpose3d.weight
            bias = self.conv_transpose3d.bias if self.conv_transpose3d.bias is not None else None
            
            # Get convolution parameters
            stride = self.conv_transpose3d.stride
            padding = self.conv_transpose3d.padding
            groups = self.conv_transpose3d.groups
            
            # Call custom Triton kernel
            return triton_conv_transpose3d(x, weight, bias, stride, padding, groups)
        else:
            # Fallback to PyTorch implementation
            return self.conv_transpose3d(x)