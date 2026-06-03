import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv3d_kernel(
    # Pointers to inputs and outputs
    x_ptr,  # [B, C_in, W, H, D]
    w_ptr,  # [C_out, C_in, K_w, K_h, K_d]
    b_ptr,  # [C_out] or None
    out_ptr,  # [B, C_out, W_out, H_out, D_out]
    # Dimensions
    B, C_in, W, H, D,
    C_out, K_w, K_h, K_d,
    stride_w, stride_h, stride_d,
    pad_w, pad_h, pad_d,
    dil_w, dil_h, dil_d,
    W_out, H_out, D_out,
    # Block sizes
    BLOCK_C_out: tl.constexpr,
    BLOCK_C_in: tl.constexpr,
    BLOCK_K_w: tl.constexpr,
    BLOCK_K_h: tl.constexpr,
    BLOCK_K_d: tl.constexpr,
):
    # Get program IDs for output tensor
    pid_c_out = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_d = tl.program_id(4)
    
    # Calculate output position
    out_w = pid_w
    out_h = pid_h
    out_d = pid_d
    
    # Calculate the starting input position for this output element
    in_w_start = out_w * stride_w - pad_w
    in_h_start = out_h * stride_h - pad_h
    in_d_start = out_d * stride_d - pad_d
    
    # Accumulator for the convolution result
    acc = tl.zeros((BLOCK_C_out,), dtype=tl.float32)
    
    # Iterate over input channels and kernel positions
    for c_in_offset in range(0, C_in, BLOCK_C_in):
        c_in_block = tl.arange(0, BLOCK_C_in)
        c_in_mask = c_in_block < (C_in - c_in_offset)
        c_in_idx = c_in_offset + c_in_block
        
        for k_w in range(0, K_w, BLOCK_K_w):
            for k_h in range(0, K_h, BLOCK_K_h):
                for k_d in range(0, K_d, BLOCK_K_d):
                    # Calculate kernel indices
                    k_w_idx = k_w + tl.arange(0, BLOCK_K_w)
                    k_h_idx = k_h + tl.arange(0, BLOCK_K_h)
                    k_d_idx = k_d + tl.arange(0, BLOCK_K_d)
                    
                    # Calculate input positions
                    in_w = in_w_start + k_w_idx * dil_w
                    in_h = in_h_start + k_h_idx * dil_h
                    in_d = in_d_start + k_d_idx * dil_d
                    
                    # Create masks for valid input positions
                    w_mask = (in_w >= 0) & (in_w < W)
                    h_mask = (in_h >= 0) & (in_h < H)
                    d_mask = (in_d >= 0) & (in_d < D)
                    kernel_mask = w_mask[:, None, None] & h_mask[None, :, None] & d_mask[None, None, :]
                    
                    # Load input values
                    x_offset = (
                        pid_b * (C_in * W * H * D) +
                        c_in_idx[:, None, None, None] * (W * H * D) +
                        in_w[None, :, None, None] * (H * D) +
                        in_h[None, None, :, None] * D +
                        in_d[None, None, None, :]
                    )
                    
                    # Reshape x_offset for proper indexing
                    x_offset_flat = x_offset.reshape(-1)
                    x_mask_flat = kernel_mask.reshape(-1)
                    
                    # Load input - need to handle the actual 4D indexing properly
                    # Simplified approach: iterate through kernel positions
                    for kw in range(BLOCK_K_w):
                        for kh in range(BLOCK_K_h):
                            for kd in range(BLOCK_K_d):
                                # Check if this kernel position is valid
                                if k_w + kw < K_w and k_h + kh < K_h and k_d + kd < K_d:
                                    in_w_pos = in_w_start + (k_w + kw) * dil_w
                                    in_h_pos = in_h_start + (k_h + kh) * dil_h
                                    in_d_pos = in_d_start + (k_d + kd) * dil_d
                                    
                                    # Check bounds
                                    if (in_w_pos >= 0 and in_w_pos < W and 
                                        in_h_pos >= 0 and in_h_pos < H and 
                                        in_d_pos >= 0 and in_d_pos < D):
                                        
                                        # Load input values for all c_in_idx
                                        x_offset_pos = (
                                            pid_b * (C_in * W * H * D) +
                                            c_in_idx * (W * H * D) +
                                            in_w_pos * (H * D) +
                                            in_h_pos * D +
                                            in_d_pos
                                        )
                                        x_vals = tl.load(x_ptr + x_offset_pos, mask=c_in_mask)
                                        
                                        # Load weight values for this kernel position
                                        w_offset = (
                                            (pid_c_out + tl.arange(0, BLOCK_C_out)[:, None, None, None]) * (C_in * K_w * K_h * K_d) +
                                            c_in_idx[None, :, None, None, None] * (K_w * K_h * K_d) +
                                            (k_w + kw) * (K_h * K_d) +
                                            (k_h + kh) * K_d +
                                            (k_d + kd)
                                        )
                                        w_vals = tl.load(w_ptr + w_offset, mask=(c_in_mask[None, :, None, None, None] & (pid_c_out + tl.arange(0, BLOCK_C_out)[:, None, None, None] < C_out)))
                                        
                                        # Reshape for multiplication
                                        x_vals_reshaped = x_vals[:, None, None, None]
                                        w_vals_reshaped = w_vals.reshape(BLOCK_C_out, BLOCK_C_in)
                                        
                                        # Accumulate
                                        acc += tl.sum(w_vals_reshaped * x_vals_reshaped, axis=1)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_c_out + tl.arange(0, BLOCK_C_out), mask=(pid_c_out + tl.arange(0, BLOCK_C_out) < C_out))
        acc += bias
    
    # Store result
    out_offset = (
        pid_b * (C_out * W_out * H_out * D_out) +
        (pid_c_out + tl.arange(0, BLOCK_C_out)[:, None, None, None]) * (W_out * H_out * D_out) +
        out_w * (H_out * D_out) +
        out_h * D_out +
        out_d
    )
    tl.store(out_ptr + out_offset, acc, mask=(pid_c_out + tl.arange(0, BLOCK_C_out)[:, None, None, None] < C_out))


def triton_conv3d(x, weight, bias, stride=1, padding=0, dilation=1, groups=1):
    """
    Performs 3D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape [B, C_in, W, H, D]
        weight: Weight tensor of shape [C_out, C_in, K_w, K_h, K_d]
        bias: Bias tensor of shape [C_out] or None
        stride, padding, dilation, groups: Convolution parameters
        
    Returns:
        Output tensor of shape [B, C_out, W_out, H_out, D_out]
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    if bias is not None:
        bias = bias.contiguous()
    
    B, C_in, W, H, D = x.shape
    C_out, _, K_w, K_h, K_d = weight.shape
    
    # Calculate output dimensions
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)
        
    W_out = (W + 2 * padding[0] - dilation[0] * (K_w - 1) - 1) // stride[0] + 1
    H_out = (H + 2 * padding[1] - dilation[1] * (K_h - 1) - 1) // stride[1] + 1
    D_out = (D + 2 * padding[2] - dilation[2] * (K_d - 1) - 1) // stride[2] + 1
    
    # Allocate output tensor
    out = torch.empty((B, C_out, W_out, H_out, D_out), dtype=x.dtype, device=x.device)
    
    # Configure grid and block sizes
    # Grid: (C_out_blocks, B, W_out, H_out, D_out)
    BLOCK_C_out = min(32, C_out)  # Adjust based on C_out
    BLOCK_C_in = min(16, C_in)
    BLOCK_K_w = 1  # Process kernel in loops
    BLOCK_K_h = 1
    BLOCK_K_d = 1
    
    grid = lambda meta: (
        triton.cdiv(C_out, meta["BLOCK_C_out"]),
        B,
        W_out,
        H_out,
        D_out
    )
    
    # Launch kernel
    conv3d_kernel[grid](
        x, weight, bias, out,
        B, C_in, W, H, D,
        C_out, K_w, K_h, K_d,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        W_out, H_out, D_out,
        BLOCK_C_out=BLOCK_C_out,
        BLOCK_C_in=BLOCK_C_in,
        BLOCK_K_w=BLOCK_K_w,
        BLOCK_K_h=BLOCK_K_h,
        BLOCK_K_d=BLOCK_K_d,
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters and create the convolution layer (we'll use its weights)
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use our Triton implementation instead of the standard conv3d
        return triton_conv3d(
            x, 
            self.conv3d.weight, 
            self.conv3d.bias if self.conv3d.bias is not None else None,
            stride=self.conv3d.stride[0] if isinstance(self.conv3d.stride, tuple) else self.conv3d.stride,
            padding=self.conv3d.padding[0] if isinstance(self.conv3d.padding, tuple) else self.conv3d.padding,
            dilation=self.conv3d.dilation[0] if isinstance(self.conv3d.dilation, tuple) else self.conv3d.dilation,
            groups=self.conv3d.groups
        )