import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr,  # Input tensor: (B, C_in, D, H, W)
    w_ptr,  # Weight tensor: (C_out, C_in, Kd, Kh, Kw)
    b_ptr,  # Bias tensor: (C_out,)
    out_ptr,  # Output tensor: (B, C_out, D_out, H_out, W_out)
    B, C_in, C_out,
    D, H, W,
    Kd, Kh, Kw,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    D_out, H_out, W_out,
    BLOCK_SIZE_M: tl.constexpr,  # Output channels per block
    BLOCK_SIZE_N: tl.constexpr,  # Batch size per block
    BLOCK_SIZE_K: tl.constexpr,  # Input channels per block
    BLOCK_SIZE_D: tl.constexpr,  # Depth dimension per block
    BLOCK_SIZE_H: tl.constexpr,  # Height dimension per block
    BLOCK_SIZE_W: tl.constexpr,  # Width dimension per block
):
    # Program IDs for output dimensions
    pid_b = tl.program_id(0)  # Batch index
    pid_c = tl.program_id(1)  # Output channel index
    pid_d = tl.program_id(2)  # Depth position
    pid_h = tl.program_id(3)  # Height position
    pid_w = tl.program_id(4)  # Width position
    
    # Calculate output position
    out_d = pid_d * BLOCK_SIZE_D
    out_h = pid_h * BLOCK_SIZE_H
    out_w = pid_w * BLOCK_SIZE_W
    
    # Offset for output tensor
    out_offsets = tl.arange(0, BLOCK_SIZE_M)[:, None, None, None] * (H_out * W_out) + \
                  pid_b * (C_out * H_out * W_out) + \
                  out_d * (H_out * W_out) + out_h * W_out + out_w
    out_mask = (tl.arange(0, BLOCK_SIZE_M) < C_out) & \
               (out_d + tl.arange(0, BLOCK_SIZE_D)[None, :, None, None] < D_out) & \
               (out_h + tl.arange(0, BLOCK_SIZE_H)[None, None, :, None] < H_out) & \
               (out_w + tl.arange(0, BLOCK_SIZE_W)[None, None, None, :] < W_out)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for c_in_block in range((C_in + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K):
        c_in_start = c_in_block * BLOCK_SIZE_K
        c_in_offsets = c_in_start + tl.arange(0, BLOCK_SIZE_K)[None, :, None, None, None]
        c_in_mask = c_in_offsets < C_in
        
        # Loop over kernel depth
        for kd in range(Kd):
            d_in = out_d * stride_d + kd * dil_d - pad_d
            d_in_offsets = d_in + tl.arange(0, BLOCK_SIZE_D)[:, None, None]
            d_in_mask = (d_in_offsets >= 0) & (d_in_offsets < D)
            
            # Loop over kernel height
            for kh in range(Kh):
                h_in = out_h * stride_h + kh * dil_h - pad_h
                h_in_offsets = h_in + tl.arange(0, BLOCK_SIZE_H)[None, :, None]
                h_in_mask = (h_in_offsets >= 0) & (h_in_offsets < H)
                
                # Loop over kernel width
                for kw in range(Kw):
                    w_in = out_w * stride_w + kw * dil_w - pad_w
                    w_in_offsets = w_in + tl.arange(0, BLOCK_SIZE_W)[None, None, :]
                    w_in_mask = (w_in_offsets >= 0) & (w_in_offsets < W)
                    
                    # Load input data
                    x_offsets = c_in_offsets * (D * H * W) + \
                                d_in_offsets[:, None, None, None] * (H * W) + \
                                h_in_offsets[None, :, None, None] * W + \
                                w_in_offsets[None, None, :, None]
                    x_mask = c_in_mask & d_in_mask[:, None, None, None] & \
                             h_in_mask[None, :, None, None] & w_in_mask[None, None, :, None]
                    
                    x_val = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
                    
                    # Load weight data
                    w_offsets = (pid_c * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None, None, None]) * (C_in * Kd * Kh * Kw) + \
                                c_in_offsets * (Kd * Kh * Kw) + \
                                kd * (Kh * Kw) + kh * Kw + kw
                    w_mask = (pid_c * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) < C_out) & c_in_mask & \
                             (kd < Kd) & (kh < Kh) & (kw < Kw)
                    
                    w_val = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)
                    
                    # Accumulate convolution
                    acc += tl.sum(w_val[:, :, None, None, None] * x_val[None, :, :, :, :], axis=1)
    
    # Add bias if present
    if b_ptr is not None:
        bias_offsets = pid_c * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        bias_mask = bias_offsets < C_out
        bias_val = tl.load(b_ptr + bias_offsets, mask=bias_mask, other=0.0)
        acc += bias_val[:, None, None, None]
    
    # Store output
    out_offsets_final = (pid_c * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None, None, None]) * (D_out * H_out * W_out) + \
                        (out_d + tl.arange(0, BLOCK_SIZE_D)[None, :, None, None]) * (H_out * W_out) + \
                        (out_h + tl.arange(0, BLOCK_SIZE_H)[None, None, :, None]) * W_out + \
                        (out_w + tl.arange(0, BLOCK_SIZE_W)[None, None, None, :])
    out_mask_final = (pid_c * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) < C_out) & \
                     (out_d + tl.arange(0, BLOCK_SIZE_D)[None, :, None, None] < D_out) & \
                     (out_h + tl.arange(0, BLOCK_SIZE_H)[None, None, :, None] < H_out) & \
                     (out_w + tl.arange(0, BLOCK_SIZE_W)[None, None, None, :] < W_out)
    
    tl.store(out_ptr + out_offsets_final, acc, mask=out_mask_final)


def triton_conv3d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Triton-based 3D convolution implementation.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C_in, D, H, W = x.shape
    C_out, _, Kd, Kh, Kw = weight.shape
    
    # Calculate output dimensions
    D_out = (D + 2 * padding - dilation * (Kd - 1) - 1) // stride + 1
    H_out = (H + 2 * padding - dilation * (Kh - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (Kw - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((B, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    # Grid dimensions
    grid = (
        B,  # batch size
        (C_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,  # output channels
        (D_out + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D,  # depth
        (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,  # height
        (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W   # width
    )
    
    # Block sizes (tunable parameters for performance)
    BLOCK_SIZE_M = 8
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_K = 8
    BLOCK_SIZE_D = 4
    BLOCK_SIZE_H = 4
    BLOCK_SIZE_W = 4
    
    # Launch kernel
    conv3d_kernel[grid](
        x, weight, bias, out,
        B, C_in, C_out,
        D, H, W,
        Kd, Kh, Kw,
        stride, stride, stride,
        padding, padding, padding,
        dilation, dilation, dilation,
        D_out, H_out, W_out,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with square input and square kernel,
    optimized with custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Register buffers for parameters to maintain compatibility
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
        
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using Triton kernel.
        """
        return triton_conv3d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )