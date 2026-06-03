import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    # Pointers to inputs and outputs
    x_ptr,  # [B, C_in, D, H, W]
    w_ptr,  # [C_out, C_in, K_d, K_h, K_w]
    b_ptr,  # [C_out] or None
    out_ptr,  # [B, C_out, D_out, H_out, W_out]
    # Dimensions
    B, C_in, D, H, W,
    C_out, 
    K_d, K_h, K_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    D_out, H_out, W_out,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_C_out: tl.constexpr,
    BLOCK_SIZE_C_in: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_K_d: tl.constexpr,
    BLOCK_SIZE_K_h: tl.constexpr,
    BLOCK_SIZE_K_w: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Compute base indices for output
    out_d = pid_d * BLOCK_SIZE_D
    out_h = pid_h * BLOCK_SIZE_H
    out_w = pid_w * BLOCK_SIZE_W
    
    # Compute input indices
    in_d_start = out_d * stride_d - pad_d
    in_h_start = out_h * stride_h - pad_h
    in_w_start = out_w * stride_w - pad_w
    
    # Accumulator for the convolution result
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_C_out), tl.float32)
    
    # Loop over input channels
    for c_in_offset in range(0, C_in, BLOCK_SIZE_C_in):
        c_in_idx = c_in_offset + tl.arange(0, BLOCK_SIZE_C_in)
        c_in_mask = c_in_idx < C_in
        
        # Loop over kernel depth
        for k_d_offset in range(0, K_d, BLOCK_SIZE_K_d):
            k_d_idx = k_d_offset + tl.arange(0, BLOCK_SIZE_K_d)
            k_d_mask = k_d_idx < K_d
            
            # Compute input d indices
            in_d_idx = in_d_start + k_d_idx * dil_d
            
            # Loop over kernel height
            for k_h_offset in range(0, K_h, BLOCK_SIZE_K_h):
                k_h_idx = k_h_offset + tl.arange(0, BLOCK_SIZE_K_h)
                k_h_mask = k_h_idx < K_h
                
                # Compute input h indices
                in_h_idx = in_h_start + k_h_idx * dil_h
                
                # Loop over kernel width
                for k_w_offset in range(0, K_w, BLOCK_SIZE_K_w):
                    k_w_idx = k_w_offset + tl.arange(0, BLOCK_SIZE_K_w)
                    k_w_mask = k_w_idx < K_w
                    
                    # Compute input w indices
                    in_w_idx = in_w_start + k_w_idx * dil_w
                    
                    # Create masks for valid input indices
                    d_mask = (in_d_idx >= 0) & (in_d_idx < D)
                    h_mask = (in_h_idx >= 0) & (in_h_idx < H)
                    w_mask = (in_w_idx >= 0) & (in_w_idx < W)
                    
                    # Reshape masks for broadcasting
                    d_mask_2d = d_mask[:, None] & d_mask[None, :]  # [BLOCK_SIZE_K_d, BLOCK_SIZE_B]
                    h_mask_2d = h_mask[:, None] & h_mask[None, :]  # [BLOCK_SIZE_K_h, BLOCK_SIZE_B]
                    w_mask_2d = w_mask[:, None] & w_mask[None, :]  # [BLOCK_SIZE_K_w, BLOCK_SIZE_B]
                    
                    # Combine masks
                    combined_mask = (
                        (in_d_idx[:, None, None] >= 0) & 
                        (in_d_idx[:, None, None] < D) &
                        (in_h_idx[None, :, None] >= 0) & 
                        (in_h_idx[None, :, None] < H) &
                        (in_w_idx[None, None, :] >= 0) & 
                        (in_w_idx[None, None, :] < W)
                    )
                    
                    # Load input data: [BLOCK_SIZE_B, BLOCK_SIZE_C_in, BLOCK_SIZE_K_d, BLOCK_SIZE_K_h, BLOCK_SIZE_K_w]
                    # For simplicity, we'll load with padding and handle out-of-bounds with masks
                    input_offsets = (
                        pid_b * (C_in * D * H * W) +
                        c_in_idx[:, None, None, None, None] * (D * H * W) +
                        (in_d_idx[:, None, None, None] * H * W) +
                        (in_h_idx[None, :, None, None] * W) +
                        (in_w_idx[None, None, :, None])  # Shape: [K_d, K_h, K_w, BLOCK_SIZE_C_in]
                    ).transpose(0, 3)  # [BLOCK_SIZE_C_in, K_d, K_h, K_w]
                    
                    input_data = tl.load(
                        x_ptr + input_offsets,
                        mask=combined_mask & c_in_mask[:, None, None, None],
                        other=0.0
                    )
                    
                    # Load weight data: [BLOCK_SIZE_C_out, BLOCK_SIZE_C_in, BLOCK_SIZE_K_d, BLOCK_SIZE_K_h, BLOCK_SIZE_K_w]
                    weight_offsets = (
                        pid_c_out * (C_in * K_d * K_h * K_w) +
                        c_in_idx[:, None, None, None, None] * (K_d * K_h * K_w) +
                        (k_d_idx[None, :, None, None, None] * K_h * K_w) +
                        (k_h_idx[None, None, :, None, None] * K_w) +
                        (k_w_idx[None, None, None, :, None])
                    ).transpose(0, 4)  # [BLOCK_SIZE_C_out, K_d, K_h, K_w, BLOCK_SIZE_C_in]
                    
                    weight_data = tl.load(
                        w_ptr + weight_offsets,
                        mask=combined_mask & c_in_mask[None, :, None, None, None],
                        other=0.0
                    )
                    
                    # Compute partial convolution result
                    # acc shape: [BLOCK_SIZE_B, BLOCK_SIZE_C_out]
                    # input_data shape: [BLOCK_SIZE_B, BLOCK_SIZE_C_in, K_d, K_h, K_w]
                    # weight_data shape: [BLOCK_SIZE_C_out, BLOCK_SIZE_C_in, K_d, K_h, K_w]
                    partial = tl.sum(input_data[None, :, :, :, :] * weight_data[:, :, :, :, :], axis=(1, 2, 3, 4))
                    acc += partial
    
    # Add bias if available
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_c_out)
        acc += bias
    
    # Store result
    out_offsets = (
        pid_b * (C_out * D_out * H_out * W_out) +
        pid_c_out * (D_out * H_out * W_out) +
        (out_d * H_out * W_out) +
        (out_h * W_out) +
        (out_w)
    )
    
    # Store the computed values
    tl.store(out_ptr + out_offsets, acc)


def triton_conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: tuple = (1, 1, 1),
    padding: tuple = (0, 0, 0),
    dilation: tuple = (1, 1, 1),
    groups: int = 1,
):
    """
    Triton-based 3D convolution implementation.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    assert groups == 1, "Only groups=1 is supported for simplicity."
    
    # Ensure contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C_in, D, H, W = x.shape
    C_out, _, K_d, K_h, K_w = weight.shape
    
    # Compute output dimensions
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    dil_d, dil_h, dil_w = dilation
    
    D_out = (D + 2 * pad_d - dil_d * (K_d - 1) - 1) // stride_d + 1
    H_out = (H + 2 * pad_h - dil_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (K_w - 1) - 1) // stride_w + 1
    
    # Create output tensor
    out = torch.empty(B, C_out, D_out, H_out, W_out, device=x.device, dtype=x.dtype)
    
    # Configure grid and block sizes
    # We'll use a 5D grid: [batch, output_channels, depth, height, width]
    # For simplicity, use reasonable block sizes
    BLOCK_SIZE_B = 1
    BLOCK_SIZE_C_out = 16
    BLOCK_SIZE_D = 2
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    
    # Calculate grid dimensions
    grid_d = (D_out + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D
    grid_h = (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    
    # Launch kernel
    conv3d_kernel[grid_d * grid_h * grid_w, BLOCK_SIZE_B, BLOCK_SIZE_C_out](
        x, weight, bias,
        out,
        B, C_in, D, H, W,
        C_out,
        K_d, K_h, K_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dil_d, dil_h, dil_w,
        D_out, H_out, W_out,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_C_out=BLOCK_SIZE_C_out,
        BLOCK_SIZE_C_in=4,  # Small tile for C_in
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_K_d=3,
        BLOCK_SIZE_K_h=5,
        BLOCK_SIZE_K_w=7,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized 3D convolution model using Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use our custom Triton kernel instead of PyTorch's native conv3d
        return triton_conv3d(
            x, 
            self.conv3d.weight,
            self.conv3d.bias if self.conv3d.bias is not None else None,
            stride=self.conv3d.stride,
            padding=self.conv3d.padding,
            dilation=self.conv3d.dilation,
            groups=self.conv3d.groups
        )