import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    # Pointers to inputs and outputs
    X_ptr,  # Input tensor: (B, C_in, D, H, W)
    W_ptr,  # Weight tensor: (C_in, C_out, Kd, Kh, Kw)
    B_ptr,  # Bias tensor: (C_out,) or None
    Y_ptr,  # Output tensor: (B, C_out, D_out, H_out, W_out)
    # Dimensions
    B, C_in, C_out,
    D, H, W,
    Kd, Kh, Kw,
    D_out, H_out, W_out,
    # Strides
    stride_x, stride_xc, stride xd, stride_xh, stride_xw,
    stride_w, stride_wc, stride_wd, stride_wh, stride_wk,
    stride_y, stride_yc, stride_yd, stride_yh, stride_yw,
    # Block sizes
    BLOCK_SIZE_CIN: tl.constexpr,
    BLOCK_SIZE_COUT: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Get program IDs
    pid_cout = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Compute output position
    out_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for output dimensions
    mask_d = out_d < D_out
    mask_h = out_h < H_out
    mask_w = out_w < W_out
    
    # Initialize accumulator for the output
    acc = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for start_cin in range(0, C_in, BLOCK_SIZE_CIN):
        c_in_offsets = start_cin + tl.arange(0, BLOCK_SIZE_CIN)
        mask_cin = c_in_offsets < C_in
        
        # Load input block: X[b, c_in, d_in, h_in, w_in]
        # Need to compute input positions from output positions
        for kd in range(Kd):
            d_in = out_d - kd
            mask_d_in = (d_in >= 0) & (d_in < D) & mask_d[:, None, None]
            
            for kh in range(Kh):
                h_in = out_h - kh
                mask_h_in = (h_in >= 0) & (h_in < H) & mask_d_in & mask_h[None, :, None]
                
                for kw in range(Kw):
                    w_in = out_w - kw
                    mask_w_in = (w_in >= 0) & (w_in < W) & mask_h_in & mask_w[None, None, :]
                    
                    # Compute indices for input
                    if start_cin == 0 and pid_cout == 0:
                        # Only compute once to avoid redundant work
                        # For simplicity, we'll handle masking differently
                        pass
                    
                    # Load weight: W[c_in, cout, kd, kh, kw]
                    w_offsets = (c_in_offsets[:, None, None, None] * stride_w +
                                pid_cout * stride_wc +
                                kd * stride_wd +
                                kh * stride_wh +
                                kw * stride_wk)
                    
                    mask_w_final = mask_cin[:, None, None, None]
                    w_vals = tl.load(W_ptr + w_offsets, mask=mask_w_final, other=0.0)
                    
                    # Load input: X[b, c_in, d_in, h_in, w_in]
                    # Compute input indices considering stride
                    d_in_block = d_in
                    h_in_block = h_in
                    w_in_block = w_in
                    
                    # Check bounds for input
                    valid_mask = (d_in_block >= 0) & (d_in_block < D) & \
                                (h_in_block >= 0) & (h_in_block < H) & \
                                (w_in_block >= 0) & (w_in_block < W)
                    
                    if tl.sum(valid_mask) > 0:
                        # Compute input pointer offset
                        x_offsets = (pid_b * stride_x +
                                    c_in_offsets[:, None, None, None] * stride_xc +
                                    d_in_block[None, :, :, :] * stride_xd +
                                    h_in_block[None, :, :, :] * stride_xh +
                                    w_in_block[None, :, :, :] * stride_xw)
                        
                        mask_x_final = mask_cin[:, None, None, None] & valid_mask[None, :, :, :]
                        x_vals = tl.load(X_ptr + x_offsets, mask=mask_x_final, other=0.0)
                        
                        # Accumulate: X * W
                        acc += tl.sum(x_vals * w_vals, axis=0)
    
    # Add bias if available
    if B_ptr is not None:
        bias = tl.load(B_ptr + pid_cout)
        acc += bias
    
    # Store result
    y_offsets = (pid_b * stride_y +
                pid_cout * stride_yc +
                out_d[:, None, None] * stride_yd +
                out_h[None, :, None] * stride_yh +
                out_w[None, None, :] * stride_yw)
    
    mask_y = mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]
    tl.store(Y_ptr + y_offsets, acc.to(Y_ptr.dtype.element_ty), mask=mask_y)


def triton_conv_transpose3d(x, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), output_padding=(0, 0, 0), groups=1):
    """
    Performs transposed 3D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (B, C_in, D, H, W)
        weight: Weight tensor of shape (C_in, C_out, Kd, Kh, Kw)
        bias: Optional bias tensor of shape (C_out,)
        stride: Stride tuple (stride_d, stride_h, stride_w)
        padding: Padding tuple (pad_d, pad_h, pad_w)
        output_padding: Output padding tuple (out_pad_d, out_pad_h, out_pad_w)
        groups: Number of groups (should be 1 for this implementation)
    
    Returns:
        Output tensor of shape (B, C_out, D_out, H_out, W_out)
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C_in, D, H, W = x.shape
    cin, cout, Kd, Kh, Kw = weight.shape
    
    # Calculate output dimensions
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    out_pad_d, out_pad_h, out_pad_w = output_padding
    
    D_out = (D - 1) * stride_d - 2 * pad_d + Kd + out_pad_d
    H_out = (H - 1) * stride_h - 2 * pad_h + Kh + out_pad_h
    W_out = (W - 1) * stride_w - 2 * pad_w + Kw + out_pad_w
    
    # Create output tensor
    y = torch.empty(B, cout, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Compute strides
    stride_x = x.stride(0)
    stride_xc = x.stride(1)
    stride_xd = x.stride(2)
    stride_xh = x.stride(3)
    stride_xw = x.stride(4)
    
    stride_w = weight.stride(0)
    stride_wc = weight.stride(1)
    stride_wd = weight.stride(2)
    stride_wh = weight.stride(3)
    stride_wk = weight.stride(4)
    
    stride_y = y.stride(0)
    stride_yc = y.stride(1)
    stride_yd = y.stride(2)
    stride_yh = y.stride(3)
    stride_yw = y.stride(4)
    
    # Grid dimensions for kernel launch
    # We parallelize over: C_out, batch, D, H, W
    # Use reasonable block sizes based on problem dimensions
    BLOCK_SIZE_COUT = min(cout, 32)
    BLOCK_SIZE_D = min(D_out, 8)
    BLOCK_SIZE_H = min(H_out, 8)
    BLOCK_SIZE_W = min(W_out, 8)
    
    grid = (
        triton.cdiv(cout, BLOCK_SIZE_COUT),
        B,
        triton.cdiv(D_out, BLOCK_SIZE_D),
        triton.cdiv(H_out, BLOCK_SIZE_H),
        triton.cdiv(W_out, BLOCK_SIZE_W),
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, y,
        B, C_in, cout,
        D, H, W,
        Kd, Kh, Kw,
        D_out, H_out, W_out,
        stride_x, stride_xc, stride_xd, stride_xh, stride_xw,
        stride_w, stride_wc, stride_wd, stride_wh, stride_wk,
        stride_y, stride_yc, stride_yd, stride_yh, stride_yw,
        BLOCK_SIZE_CIN=32,
        BLOCK_SIZE_COUT=BLOCK_SIZE_COUT,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for transposed 3D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), 
                 padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters for later use in forward
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias
        
        # Initialize weight and bias as in original ConvTranspose3d
        kernel_depth, kernel_width, kernel_height = kernel_size
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_depth, kernel_width, kernel_height))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize weight using kaiming uniform as in PyTorch
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )


# Import math for sqrt operations in parameter initialization
import math