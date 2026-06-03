import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 4, 'BLOCK_H': 4, 'BLOCK_W': 4, 'BLOCK_KC': 16}, num_warps=4),
        triton.Config({'BLOCK_D': 4, 'BLOCK_H': 4, 'BLOCK_W': 2, 'BLOCK_KC': 16}, num_warps=4),
        triton.Config({'BLOCK_D': 2, 'BLOCK_H': 4, 'BLOCK_W': 4, 'BLOCK_KC': 16}, num_warps=4),
        triton.Config({'BLOCK_D': 4, 'BLOCK_H': 2, 'BLOCK_W': 4, 'BLOCK_KC': 16}, num_warps=4),
    ],
    key=['D_out', 'H_out', 'W_out', 'kernel_size', 'dilation', 'stride'],
)
@triton.jit
def conv3d_kernel(
    x_ptr,  # Input tensor: (B, C_in, D, H, W)
    w_ptr,  # Weight tensor: (C_out, C_in, Kd, Kh, Kw)
    b_ptr,  # Bias tensor: (C_out,) or None
    y_ptr,  # Output tensor: (B, C_out, D_out, H_out, W_out)
    B, C_in, D, H, W,  # Input dimensions
    C_out, Kd, Kh, Kw,  # Weight dimensions
    D_out, H_out, W_out,  # Output dimensions
    stride_d, stride_h, stride_w,  # Stride
    pad_d, pad_h, pad_w,  # Padding
    dil_d, dil_h, dil_w,  # Dilation
    # Meta-parameters
    BLOCK_B: tl.constexpr,
    BLOCK_C_out: tl.constexpr,
    BLOCK_C_in: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_KC: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2) // (H_out * W_out)
    pid_h = (tl.program_id(2) // W_out) % H_out
    pid_w = tl.program_id(2) % W_out

    # Compute the starting indices in the output volume
    d_start = pid_d * stride_d - pad_d
    h_start = pid_h * stride_h - pad_h
    w_start = pid_w * stride_w - pad_w

    # Initialize accumulator
    acc = tl.zeros((BLOCK_C_out,), dtype=tl.float32)

    # Iterate over input channels and kernel positions
    for c_in_start in range(0, C_in, BLOCK_C_in):
        c_in_range = c_in_start + tl.arange(0, BLOCK_C_in)
        c_in_mask = c_in_range < C_in

        # Load input block: (BLOCK_B, BLOCK_C_in, BLOCK_D, BLOCK_H, BLOCK_W)
        # We'll process one batch element at a time for simplicity
        for bd in range(BLOCK_B):
            if pid_b + bd >= B:
                break
                
            for kd in range(Kd):
                d_idx = d_start + kd * dil_d
                d_valid = (d_idx >= 0) & (d_idx < D)
                d_offset = d_idx * (H * W) if d_valid else -1
                
                for kh in range(Kh):
                    h_idx = h_start + kh * dil_h
                    h_valid = (h_idx >= 0) & (h_idx < H)
                    h_offset = h_idx * W if h_valid else -1
                    
                    for kw in range(Kw):
                        w_idx = w_start + kw * dil_w
                        w_valid = (w_idx >= 0) & (w_idx < W)
                        w_offset = w_idx if w_valid else -1
                        
                        # Compute input offset
                        if d_valid and h_valid and w_valid:
                            in_offset = (pid_b + bd) * (C_in * D * H * W) + \
                                       c_in_range[:, None, None, None] * (D * H * W) + \
                                       d_offset * (H * W) + h_offset * W + w_offset
                            
                            # Load input values: shape (BLOCK_C_in, 1, 1, 1)
                            x_block = tl.load(
                                x_ptr + in_offset, 
                                mask=c_in_mask[:, None, None, None], 
                                other=0.0
                            )
                        else:
                            x_block = tl.zeros((BLOCK_C_in, 1, 1, 1), dtype=tl.float32)
                            
                        # Load weight values for this kernel position
                        w_offset = (pid_c_out * (C_in * Kd * Kh * Kw) + 
                                   c_in_range[:, None, None, None] * (Kd * Kh * Kw) +
                                   kd * (Kh * Kw) + kh * Kw + kw)
                        w_block = tl.load(
                            w_ptr + w_offset, 
                            mask=c_in_mask[:, None, None, None],
                            other=0.0
                        )
                        
                        # Accumulate
                        acc += tl.sum(x_block * w_block, axis=0)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_c_out)
        acc += bias

    # Store output
    y_offset = (pid_b * (C_out * D_out * H_out * W_out) + 
               pid_c_out * (D_out * H_out * W_out) + 
               pid_d * (H_out * W_out) + 
               pid_h * W_out + 
               pid_w)
    tl.store(y_ptr + y_offset, acc.to(y_ptr.dtype.element_ty))


def triton_conv3d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Performs 3D convolution using a Triton kernel.
    Assumes groups=1 and all tensors are contiguous on GPU.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    B, C_in, D, H, W = x.shape
    C_out, _, Kd, Kh, Kw = weight.shape
    
    # Compute output dimensions
    D_out = (D + 2 * padding - dilation * (Kd - 1) - 1) // stride + 1
    H_out = (H + 2 * padding - dilation * (Kh - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (Kw - 1) - 1) // stride + 1
    
    # Prepare output tensor
    y = torch.empty(B, C_out, D_out, H_out, W_out, device=x.device, dtype=x.dtype)
    
    # Define grid dimensions
    grid_b = 1  # Process one batch at a time for simplicity
    grid_c_out = C_out
    grid_dhw = D_out * H_out * W_out
    
    # Determine block sizes based on problem size
    # We'll use tunable block sizes via autotune
    BLOCK_B = 1
    BLOCK_C_out = 1
    BLOCK_C_in = min(16, C_in)
    
    # Launch kernel
    conv3d_kernel[grid_b, grid_c_out, grid_dhw](
        x, weight, bias, y,
        B, C_in, D, H, W,
        C_out, Kd, Kh, Kw,
        D_out, H_out, W_out,
        stride, stride, stride,
        padding, padding, padding,
        dilation, dilation, dilation,
        BLOCK_B=BLOCK_B,
        BLOCK_C_out=BLOCK_C_out,
        BLOCK_C_in=BLOCK_C_in,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with square input and square kernel,
    optimized with a custom Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), 
                               stride=stride, padding=padding, dilation=dilation, groups=groups, 
                               bias=bias)
        # Replace the forward method with our optimized version
        self.original_forward = self.conv3d.forward
        self.conv3d.forward = lambda x: triton_conv3d(
            x, self.conv3d.weight, 
            self.conv3d.bias if self.conv3d.bias is not None else None,
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=groups
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3d(x)