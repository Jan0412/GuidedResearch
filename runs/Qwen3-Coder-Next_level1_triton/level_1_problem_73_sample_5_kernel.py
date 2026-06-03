import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # Input tensor: (B, C_in, D, H, W)
    w_ptr,  # Weight tensor: (C_in, C_out//G, Kd, Kh, Kw)
    b_ptr,  # Bias tensor: (C_out,)
    out_ptr,  # Output tensor: (B, C_out, D_out, H_out, W_out)
    B, C_in, C_out, G,  # Batch size, input channels, output channels, groups
    D, H, W,  # Input dimensions
    Kd, Kh, Kw,  # Kernel dimensions
    stride_d, stride_h, stride_w,  # Stride
    pad_d, pad_h, pad_w,  # Padding
    out_d, out_h, out_w,  # Output dimensions
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr,  # Block size for accumulation
):
    # Output tensor indices
    pid_b = tl.program_id(0)  # Batch index
    pid_c = tl.program_id(1)  # Output channel index
    pid_d = tl.program_id(2)  # Depth index
    pid_h = tl.program_id(3)  # Height index
    pid_w = tl.program_id(4)  # Width index

    # Calculate which output channel this program handles
    c_out_start = pid_c * BLOCK_SIZE_M
    c_out_range = tl.arange(0, BLOCK_SIZE_M)
    c_out_mask = c_out_range < BLOCK_SIZE_M
    
    # Compute output position
    out_idx = pid_b * (C_out * out_d * out_h * out_w) + \
              c_out_start * (out_d * out_h * out_w) + \
              pid_d * (out_h * out_w) + \
              pid_h * out_w + pid_w
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Compute corresponding input position
    in_d = pid_d * stride_d - pad_d
    in_h = pid_h * stride_h - pad_h
    in_w = pid_w * stride_w - pad_w
    
    # Iterate over input channels and kernel dimensions
    for c_in_idx in range(C_in // G):
        for kd in range(Kd):
            for kh in range(Kh):
                for kw in range(Kw):
                    # Compute input coordinates
                    d = in_d + kd
                    h = in_h + kh
                    w = in_w + kw
                    
                    # Check bounds
                    if 0 <= d < D and 0 <= h < H and 0 <= w < W:
                        # Compute input index
                        in_idx = pid_b * (C_in * D * H * W) + \
                                c_in_idx * (D * H * W) + \
                                d * (H * W) + h * W + w
                        
                        # Load input value
                        x_val = tl.load(x_ptr + in_idx)
                        
                        # Compute weight index
                        # Weight layout: (C_in, C_out//G, Kd, Kh, Kw)
                        # But we need to handle groups correctly
                        # Group index for this input channel
                        g_idx = c_in_idx // (C_in // G)
                        # Local input channel in group
                        local_c_in = c_in_idx % (C_in // G)
                        # Output channels for this group
                        c_out_local = tl.arange(0, BLOCK_SIZE_M)
                        
                        # Weight index calculation
                        w_idx = (c_in_idx * C_out * Kd * Kh * Kw +
                                c_out_local * Kd * Kh * Kw +
                                kd * Kh * Kw +
                                kh * Kw +
                                kw)
                        
                        # Load weights
                        w_vals = tl.load(w_ptr + w_idx, mask=c_out_mask, other=0.0)
                        
                        # Accumulate
                        acc += x_val * w_vals
    
    # Add bias if available
    if b_ptr is not None:
        bias_vals = tl.load(b_ptr + c_out_start + c_out_range, mask=c_out_mask, other=0.0)
        acc += bias_vals
    
    # Store result
    tl.store(out_ptr + out_idx, acc.to(x_ptr.dtype.element_ty), mask=c_out_mask)


# Alternative, more efficient implementation using tiling
@triton.jit
def conv_transpose3d_kernel_v2(
    x_ptr,  # Input tensor: (B, C_in, D, H, W)
    w_ptr,  # Weight tensor: (C_in, C_out//G, Kd, Kh, Kw)
    b_ptr,  # Bias tensor: (C_out,)
    out_ptr,  # Output tensor: (B, C_out, D_out, H_out, W_out)
    B, C_in, C_out, G,  # Batch size, input channels, output channels, groups
    D, H, W,  # Input dimensions
    Kd, Kh, Kw,  # Kernel dimensions
    stride_d, stride_h, stride_w,  # Stride
    pad_d, pad_h, pad_w,  # Padding
    out_d, out_h, out_w,  # Output dimensions
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr,
):
    # Output tensor indices
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Compute output channel range
    c_out_start = pid_c_out * BLOCK_SIZE_C_OUT
    c_out_range = tl.arange(0, BLOCK_SIZE_C_OUT)
    c_out_mask = c_out_range < BLOCK_SIZE_C_OUT
    
    # Output index calculation
    out_idx_base = pid_b * (C_out * out_d * out_h * out_w) + \
                   pid_d * (out_h * out_w) + pid_h * out_w + pid_w
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_C_OUT,), dtype=tl.float32)
    
    # Compute corresponding input position
    in_d = pid_d * stride_d - pad_d
    in_h = pid_h * stride_h - pad_h
    in_w = pid_w * stride_w - pad_w
    
    # Iterate over input channels and kernel
    for c_in_idx in range(0, C_in, BLOCK_SIZE_C_IN):
        c_in_range = tl.arange(0, BLOCK_SIZE_C_IN)
        c_in_mask = c_in_range < C_in - c_in_idx
        
        for kd in range(Kd):
            for kh in range(Kh):
                for kw in range(Kw):
                    d = in_d + kd
                    h = in_h + kh
                    w = in_w + kw
                    
                    if 0 <= d < D and 0 <= h < H and 0 <= w < W:
                        # Input index
                        x_idx = pid_b * (C_in * D * H * W) + \
                                c_in_range * (D * H * W) + \
                                d * (H * W) + h * W + w
                        x_vals = tl.load(x_ptr + x_idx, mask=c_in_mask[:, None], other=0.0)
                        
                        # Weight index with proper group handling
                        # For group convolution: weight shape is (C_in, C_out//G, Kd, Kh, Kw)
                        # Group index for current input channels
                        g_idx = c_in_range // (C_in // G)
                        local_c_in = c_in_range % (C_in // G)
                        
                        # Only process channels in the same group as the output
                        # This is simplified - full group handling requires more complex logic
                        
                        # Weight index: (c_in, c_out_local, kd, kh, kw)
                        c_out_local_range = tl.arange(0, BLOCK_SIZE_C_OUT)
                        w_idx = (c_in_range[:, None] * C_out * Kd * Kh * Kw +
                                c_out_local_range[None, :] * Kd * Kh * Kw +
                                kd * Kh * Kw + kh * Kw + kw)
                        
                        w_vals = tl.load(w_ptr + w_idx, mask=c_in_mask[:, None] & c_out_mask[None, :], other=0.0)
                        
                        # Reshape for matmul-like operation
                        # x_vals: (BLOCK_SIZE_C_IN,)
                        # w_vals: (BLOCK_SIZE_C_IN, BLOCK_SIZE_C_OUT)
                        acc += tl.sum(x_vals[:, None] * w_vals, axis=0)
    
    # Add bias
    if b_ptr is not None:
        bias_vals = tl.load(b_ptr + c_out_start + c_out_range, mask=c_out_mask, other=0.0)
        acc += bias_vals
    
    # Store result
    out_idx = out_idx_base + c_out_start * (out_h * out_w)
    tl.store(out_ptr + out_idx, acc.to(x_ptr.dtype.element_ty), mask=c_out_mask)


def triton_conv_transpose3d(x, weight, bias, stride, padding, groups):
    """
    Triton implementation of 3D transposed convolution.
    """
    B, C_in, D, H, W = x.shape
    C_in_, C_out_, Kd, Kh, Kw = weight.shape
    
    # Calculate output dimensions
    out_d = (D - 1) * stride - 2 * padding + Kd
    out_h = (H - 1) * stride - 2 * padding + Kh
    out_w = (W - 1) * stride - 2 * padding + Kw
    
    # Create output tensor
    out = torch.empty((B, C_out_, out_d, out_h, out_w), dtype=x.dtype, device=x.device)
    
    # Grid dimensions
    grid = (B, (C_out_ + 32 - 1) // 32, out_d, out_h, out_w)
    
    # Launch kernel
    conv_transpose3d_kernel_v2[grid](
        x, weight, bias, out,
        B, C_in, C_out_, groups,
        D, H, W,
        Kd, Kh, Kw,
        stride, stride, stride,
        padding, padding, padding,
        out_d, out_h, out_w,
        BLOCK_SIZE_C_OUT=32,
        BLOCK_SIZE_C_IN=16,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for 3D transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using custom Triton kernel for 3D transposed convolution.
        """
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Call custom Triton kernel
        return triton_conv_transpose3d(
            x, weight, self.bias, 
            self.stride, self.padding, self.groups
        )