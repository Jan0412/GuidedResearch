import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    # Pointers to tensors
    x_ptr,  # Input: (B, C_in, D, H, W)
    w_ptr,  # Weight: (C_in, C_out, K_d, K_h, K_w)
    b_ptr,  # Bias: (C_out,) or None
    out_ptr,  # Output: (B, C_out, D_out, H_out, W_out)
    # Tensor dimensions
    B, C_in, C_out,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    K_d, K_h, K_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    # Block sizes for tiling
    BLOCK_B: tl.constexpr,
    BLOCK_C_out: tl.constexpr,
    BLOCK_C_in: tl.constexpr,
    BLOCK_K_d: tl.constexpr,
    BLOCK_K_h: tl.constexpr,
    BLOCK_K_w: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Get program IDs for output tensor indexing
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    # Calculate output position
    out_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    out_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    
    # Create masks for valid indices
    mask_d = out_d < D_out
    mask_h = out_h < H_out
    mask_w = out_w < W_out
    mask_3d = mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]
    
    # Initialize accumulator for output
    acc = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Loop over input channels and kernel dimensions
    for c_in in range(C_in):
        for kd in range(K_d):
            for kh in range(K_h):
                for kw in range(K_w):
                    # Calculate corresponding input position
                    in_d = (out_d - (kd * dil_d - pad_d)) // stride_d
                    in_h = (out_h - (kh * dil_h - pad_h)) // stride_h
                    in_w = (out_w - (kw * dil_w - pad_w)) // stride_w
                    
                    # Check if input position is valid
                    mask_in_d = (in_d >= 0) & (in_d < D_in) & mask_d[:, None, None]
                    mask_in_h = (in_h >= 0) & (in_h < H_in) & mask_h[None, :, None]
                    mask_in_w = (in_w >= 0) & (in_w < W_in) & mask_w[None, None, :]
                    mask_in = mask_in_d & mask_in_h & mask_in_w
                    
                    # Calculate pointers for input
                    offsets_in = (
                        pid_b * (C_in * D_in * H_in * W_in) +
                        c_in * (D_in * H_in * W_in) +
                        in_d[:, None, None] * (H_in * W_in) +
                        in_h[None, :, None] * W_in +
                        in_w[None, None, :]
                    )
                    
                    # Load input values (with masking)
                    x_val = tl.load(
                        x_ptr + offsets_in,
                        mask=mask_in,
                        other=0.0
                    )
                    
                    # Calculate weight pointer
                    w_offsets = (
                        c_in * (C_out * K_d * K_h * K_w) +
                        pid_c_out * (K_d * K_h * K_w) +
                        kd * (K_h * K_w) +
                        kh * K_w +
                        kw
                    )
                    w_val = tl.load(w_ptr + w_offsets)
                    
                    # Accumulate: x * w
                    acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        b_offsets = pid_c_out
        bias_val = tl.load(b_ptr + b_offsets)
        acc += bias_val
    
    # Store output
    out_offsets = (
        pid_b * (C_out * D_out * H_out * W_out) +
        pid_c_out * (D_out * H_out * W_out) +
        out_d[:, None, None] * (H_out * W_out) +
        out_h[None, :, None] * W_out +
        out_w[None, None, :]
    )
    
    tl.store(
        out_ptr + out_offsets,
        acc.to(x_ptr.dtype.element_ty),
        mask=mask_3d
    )


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1
) -> torch.Tensor:
    """
    Performs 3D transposed convolution using Triton kernel.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C_in, D_in, H_in, W_in = x.shape
    _, C_out, K_d, K_h, K_w = weight.shape
    
    # Calculate output dimensions
    D_out = (D_in - 1) * stride - 2 * padding + dilation * (K_d - 1) + 1
    H_out = (H_in - 1) * stride - 2 * padding + dilation * (K_h - 1) + 1
    W_out = (W_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1
    
    # Allocate output tensor
    out = torch.empty(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Define block sizes for tiling (tunable parameters)
    BLOCK_B = 1
    BLOCK_C_out = 16
    BLOCK_C_in = 4
    BLOCK_K_d = K_d
    BLOCK_K_h = K_h
    BLOCK_K_w = K_w
    BLOCK_D = 4
    BLOCK_H = 4
    BLOCK_W = 4
    
    # Calculate grid dimensions
    grid = lambda meta: (
        B,
        triton.cdiv(C_out, meta["BLOCK_C_out"]),
        triton.cdiv(D_out, meta["BLOCK_D"]),
        triton.cdiv(H_out, meta["BLOCK_H"]),
        triton.cdiv(W_out, meta["BLOCK_W"]),
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        B, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        K_d, K_h, K_w,
        stride, stride, stride,
        padding, padding, padding,
        dilation, dilation, dilation,
        BLOCK_B=BLOCK_B,
        BLOCK_C_out=BLOCK_C_out,
        BLOCK_C_in=BLOCK_C_in,
        BLOCK_K_d=BLOCK_K_d,
        BLOCK_K_h=BLOCK_K_h,
        BLOCK_K_w=BLOCK_K_w,
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for 3D transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias_flag = bias
        
        # Create weight and bias parameters (matching nn.ConvTranspose3d)
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
        
        # Initialize weights (matching PyTorch default initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation
        )


import math