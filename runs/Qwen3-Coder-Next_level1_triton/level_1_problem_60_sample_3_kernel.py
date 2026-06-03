import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr,  # Input tensor pointer (N, C_in, W, H, D)
    w_ptr,  # Weight tensor pointer (C_out, C_in, k_w, k_h, k_d)
    b_ptr,  # Bias tensor pointer (C_out,) - optional, can be None
    out_ptr,  # Output tensor pointer (N, C_out, W_out, H_out, D_out)
    N, C_in, C_out, 
    W, H, D,  # Input dimensions
    W_out, H_out, D_out,  # Output dimensions
    k_w, k_h, k_d,  # Kernel dimensions
    stride_w, stride_h, stride_d,  # Strides
    pad_w, pad_h, pad_d,  # Padding
    dil_w, dil_h, dil_d,  # Dilation
    C_in_per_group: tl.constexpr,  # Channels per group
    C_out_per_group: tl.constexpr,  # Output channels per group
    BLOCK_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Get program IDs
    pid_n = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_d = tl.program_id(4)
    
    # Determine group index
    group_idx = pid_c_out // C_out_per_group
    c_out_start = group_idx * C_out_per_group
    c_out_offset = pid_c_out - c_out_start
    
    # Compute output position
    out_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    out_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    out_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    
    # Create masks for output dimensions
    w_mask = out_w < W_out
    h_mask = out_h < H_out
    d_mask = out_d < D_out
    w_h_mask = w_mask[:, None] & h_mask[None, :]
    w_h_d_mask = w_h_mask[:, :, None] & d_mask[None, None, :]
    
    # Compute input position corresponding to this output position
    in_w = out_w * stride_w - pad_w
    in_h = out_h * stride_h - pad_h
    in_d = out_d * stride_d - pad_d
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_W, BLOCK_H, BLOCK_D), dtype=tl.float32)
    
    # Loop over input channels
    for c_in in range(C_in_per_group):
        c_in_idx = c_in + group_idx * C_in_per_group
        
        # Loop over kernel dimensions
        for kw in range(k_w):
            kw_pos = kw * dil_w
            in_w_k = in_w + kw_pos
            w_k_mask = (in_w_k >= 0) & (in_w_k < W)
            
            for kh in range(k_h):
                kh_pos = kh * dil_h
                in_h_k = in_h + kh_pos
                h_k_mask = (in_h_k >= 0) & (in_h_k < H)
                
                for kd in range(k_d):
                    kd_pos = kd * dil_d
                    in_d_k = in_d + kd_pos
                    d_k_mask = (in_d_k >= 0) & (in_d_k < D)
                    
                    # Combine all masks
                    k_mask = w_k_mask[:, None, None] & h_k_mask[None, :, None] & d_k_mask[None, None, :]
                    
                    # Compute input indices
                    idx_w = tl.where(w_k_mask, in_w_k, 0)
                    idx_h = tl.where(h_k_mask, in_h_k, 0)
                    idx_d = tl.where(d_k_mask, in_d_k, 0)
                    
                    # Load input data: [N, C_in, W, H, D]
                    # We need to access the specific batch and channel
                    x_offset = (
                        pid_n * (C_in * W * H * D) +
                        c_in_idx * (W * H * D) +
                        idx_w[:, None, None] * (H * D) +
                        idx_h[None, :, None] * D +
                        idx_d[None, None, :]
                    )
                    x_vals = tl.load(
                        x_ptr + x_offset,
                        mask=k_mask & (idx_w[:, None, None] < W) & 
                               (idx_h[None, :, None] < H) & 
                               (idx_d[None, None, :] < D),
                        other=0.0
                    )
                    
                    # Load weight data: [C_out, C_in, k_w, k_h, k_d]
                    w_offset = (
                        pid_c_out * (C_in * k_w * k_h * k_d) +
                        c_in_idx * (k_w * k_h * k_d) +
                        kw * (k_h * k_d) +
                        kh * k_d +
                        kd
                    )
                    w_val = tl.load(w_ptr + w_offset)
                    
                    # Accumulate
                    acc += x_vals * w_val
    
    # Apply bias if provided
    if b_ptr is not None:
        b_offset = pid_c_out
        bias = tl.load(b_ptr + b_offset)
        acc += bias
    
    # Convert to output type and store
    acc = acc.to(x_ptr.dtype.element_ty)
    
    # Compute output offset
    out_offset = (
        pid_n * (C_out * W_out * H_out * D_out) +
        pid_c_out * (W_out * H_out * D_out) +
        out_w[:, None, None] * (H_out * D_out) +
        out_h[None, :, None] * D_out +
        out_d[None, None, :]
    )
    
    # Store result
    tl.store(
        out_ptr + out_offset,
        acc,
        mask=w_h_d_mask
    )


def triton_conv3d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Performs 3D convolution using Triton kernel.
    
    Args:
        x: Input tensor [N, C_in, W, H, D]
        weight: Weight tensor [C_out, C_in // groups, k_w, k_h, k_d]
        bias: Optional bias tensor [C_out]
        stride, padding, dilation, groups: Convolution parameters
        
    Returns:
        Output tensor [N, C_out, W_out, H_out, D_out]
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    N, C_in, W, H, D = x.shape
    C_out, _, k_w, k_h, k_d = weight.shape
    
    # Handle stride, padding, dilation as tuples or scalars
    if isinstance(stride, int):
        stride_w = stride_h = stride_d = stride
    else:
        stride_w, stride_h, stride_d = stride
        
    if isinstance(padding, int):
        pad_w = pad_h = pad_d = padding
    else:
        pad_w, pad_h, pad_d = padding
        
    if isinstance(dilation, int):
        dil_w = dil_h = dil_d = dilation
    else:
        dil_w, dil_h, dil_d = dilation
    
    # Calculate output dimensions
    W_out = (W + 2 * pad_w - dil_w * (k_w - 1) - 1) // stride_w + 1
    H_out = (H + 2 * pad_h - dil_h * (k_h - 1) - 1) // stride_h + 1
    D_out = (D + 2 * pad_d - dil_d * (k_d - 1) - 1) // stride_d + 1
    
    # Prepare output tensor
    out = torch.empty((N, C_out, W_out, H_out, D_out), dtype=x.dtype, device=x.device)
    
    # Determine block sizes (tunable parameters)
    BLOCK_W = 4
    BLOCK_H = 4
    BLOCK_D = 4
    BLOCK_C = 8  # Not used directly but could be used for channel blocking
    
    # Determine grid dimensions
    grid = (
        N,  # batch dimension
        C_out,  # output channels
        (W_out + BLOCK_W - 1) // BLOCK_W,  # width blocks
        (H_out + BLOCK_H - 1) // BLOCK_H,  # height blocks
        (D_out + BLOCK_D - 1) // BLOCK_D,  # depth blocks
    )
    
    # Launch kernel
    conv3d_kernel[grid](
        x, weight, bias, out,
        N, C_in, C_out,
        W, H, D,
        W_out, H_out, D_out,
        k_w, k_h, k_d,
        stride_w, stride_h, stride_d,
        pad_w, pad_h, pad_d,
        dil_w, dil_h, dil_d,
        C_in_per_group=C_in // groups,
        C_out_per_group=C_out // groups,
        BLOCK_W=BLOCK_W,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
        BLOCK_C=BLOCK_C,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model with Triton-based 3D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same Conv3d layer but we'll use our custom kernel
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=padding, dilation=dilation, 
                               groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, width, height, depth).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, width_out, height_out, depth_out).
        """
        # Use our custom Triton kernel for convolution
        return triton_conv3d(
            x,
            self.conv3d.weight,
            self.conv3d.bias if self.conv3d.bias is not None else None,
            stride=self.conv3d.stride,
            padding=self.conv3d.padding,
            dilation=self.conv3d.dilation,
            groups=self.conv3d.groups
        )