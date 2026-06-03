import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    # Pointers to inputs and outputs
    x_ptr,  # Input tensor: (B, C_in, D, H, W)
    w_ptr,  # Weight tensor: (C_in, C_out, Kd, Kh, Kw)
    b_ptr,  # Bias tensor: (C_out,)
    out_ptr,  # Output tensor: (B, C_out, D_out, H_out, W_out)
    # Tensor dimensions
    B, C_in, C_out,
    D, H, W,
    Kd, Kh, Kw,
    D_out, H_out, W_out,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    output_pad_d, output_pad_h, output_pad_w,
    # Block sizes for tiling
    BLOCK_C_in: tl.constexpr,
    BLOCK_C_out: tl.constexpr,
    BLOCK_Kd: tl.constexpr,
    BLOCK_Kh: tl.constexpr,
    BLOCK_Kw: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Get program IDs
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
    
    # Calculate input positions from output positions
    # For transposed convolution: input_d = (out_d - output_pad_d) // stride_d
    # Only compute if the position corresponds to a valid input position
    in_d_start = (out_d - output_pad_d + pad_d) // stride_d
    in_h_start = (out_h - output_pad_h + pad_h) // stride_h
    in_w_start = (out_w - output_pad_w + pad_w) // stride_w
    
    # Calculate kernel positions for this output
    k_d_offsets = tl.arange(0, BLOCK_Kd)
    k_h_offsets = tl.arange(0, BLOCK_Kh)
    k_w_offsets = tl.arange(0, BLOCK_Kw)
    
    # Output accumulator
    acc = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Iterate over input channels
    for c_in_start in range(0, C_in, BLOCK_C_in):
        c_in_offsets = c_in_start + tl.arange(0, BLOCK_C_in)
        mask_c_in = c_in_offsets < C_in
        
        # Input indices: (B, C_in, D, H, W)
        in_d = in_d_start[:, None, None] + k_d_offsets[None, :, None, None] * stride_d - pad_d
        in_h = in_h_start[None, :, None, None] + k_h_offsets[None, None, :, None] * stride_h - pad_h
        in_w = in_w_start[None, None, :, None] + k_w_offsets[None, None, None, :] * stride_w - pad_w
        
        # Check if input position is valid
        valid_mask = (in_d >= 0) & (in_d < D) & (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W)
        
        # Load input data: (B, C_in, D, H, W)
        in_d_flat = in_d.flatten()
        in_h_flat = in_h.flatten()
        in_w_flat = in_w.flatten()
        mask_flat = valid_mask.flatten()[:, None] & mask_c_in[None, :]
        
        # Calculate input pointer offsets
        input_offsets = (
            pid_b * (C_in * D * H * W) +
            c_in_offsets[None, :] * (D * H * W) +
            in_d_flat[:, None] * (H * W) +
            in_h_flat[:, None] * W +
            in_w_flat[:, None]
        )
        
        # Load input values
        x = tl.load(x_ptr + input_offsets, mask=mask_flat, other=0.0)
        
        # Load weight data: (C_in, C_out, Kd, Kh, Kw)
        weight_offsets = (
            c_in_offsets[:, None, None, None, None] * (C_out * Kd * Kh * Kw) +
            pid_c_out * (Kd * Kh * Kw) +
            k_d_offsets[None, :, None, None, None] * (Kh * Kw) +
            k_h_offsets[None, None, :, None, None] * Kw +
            k_w_offsets[None, None, None, :, :]
        )
        
        w = tl.load(w_ptr + weight_offsets, mask=mask_c_in[:, None, None, None, None])
        
        # Reshape for multiplication
        x_reshaped = x.view(BLOCK_C_in, BLOCK_D * BLOCK_H * BLOCK_W)
        w_reshaped = w.view(BLOCK_C_in, BLOCK_D * BLOCK_H * BLOCK_W)
        
        # Accumulate
        acc += tl.sum(x_reshaped * w_reshaped, axis=0).view(BLOCK_D, BLOCK_H, BLOCK_W)
    
    # Apply bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_c_out)
        acc += bias
    
    # Store output
    out_offsets = (
        pid_b * (C_out * D_out * H_out * W_out) +
        pid_c_out * (D_out * H_out * W_out) +
        out_d[:, None, None] * (H_out * W_out) +
        out_h[None, :, None] * W_out +
        out_w[None, None, :]
    )
    
    tl.store(out_ptr + out_offsets, acc.to(x_ptr.dtype.element_ty), mask=mask_3d)


class TritonConvTranspose3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, 
                stride, padding, output_padding, dilation, groups):
        # Validate inputs
        assert x.dim() == 5, "Input must be 5D tensor (B, C_in, D, H, W)"
        assert weight.dim() == 5, "Weight must be 5D tensor (C_in, C_out, Kd, Kh, Kw)"
        
        # Extract dimensions
        B, C_in, D, H, W = x.shape
        C_in_w, C_out, Kd, Kh, Kw = weight.shape
        
        assert C_in == C_in_w, f"Input channels ({C_in}) must match weight input channels ({C_in_w})"
        
        # Calculate output dimensions
        stride_d, stride_h, stride_w = stride
        pad_d, pad_h, pad_w = padding
        output_pad_d, output_pad_h, output_pad_w = output_padding
        
        D_out = (D - 1) * stride_d - 2 * pad_d + Kd + output_pad_d
        H_out = (H - 1) * stride_h - 2 * pad_h + Kh + output_pad_h
        W_out = (W - 1) * stride_w - 2 * pad_w + Kw + output_pad_w
        
        # Create output tensor
        out = torch.empty(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
        
        # Configure kernel launch
        BLOCK_C_in = 16
        BLOCK_C_out = 16
        BLOCK_Kd = 3
        BLOCK_Kh = 3
        BLOCK_Kw = 3
        BLOCK_D = 4
        BLOCK_H = 4
        BLOCK_W = 4
        
        # Grid dimensions
        grid = (
            B,  # batch size
            triton.cdiv(C_out, BLOCK_C_out),  # output channels
            triton.cdiv(D_out, BLOCK_D),  # depth
            triton.cdiv(H_out, BLOCK_H),  # height
            triton.cdiv(W_out, BLOCK_W),  # width
        )
        
        # Launch kernel
        conv_transpose3d_kernel[grid](
            x, weight, bias,
            out,
            B, C_in, C_out,
            D, H, W,
            Kd, Kh, Kw,
            D_out, H_out, W_out,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            output_pad_d, output_pad_h, output_pad_w,
            BLOCK_C_in=BLOCK_C_in,
            BLOCK_C_out=BLOCK_C_out,
            BLOCK_Kd=BLOCK_Kd,
            BLOCK_Kh=BLOCK_Kh,
            BLOCK_Kw=BLOCK_Kw,
            BLOCK_D=BLOCK_D,
            BLOCK_H=BLOCK_H,
            BLOCK_W=BLOCK_W,
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        ctx.dilation = dilation
        ctx.groups = groups
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # For simplicity, fall back to PyTorch for backward pass
        x, weight = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        output_padding = ctx.output_padding
        dilation = ctx.dilation
        groups = ctx.groups
        
        # Use PyTorch's built-in backward
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv_transpose3d(
                x, grad_output, weight,
                stride=stride, padding=padding,
                output_padding=output_padding, groups=groups,
                dilation=dilation
            )
        
        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution with square input and square kernel using optimized Triton kernel.
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
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using optimized Triton kernel.
        """
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Call the custom Triton implementation
        return TritonConvTranspose3d.apply(
            x, self.weight, self.bias,
            (self.stride, self.stride, self.stride),
            (self.padding, self.padding, self.padding),
            (self.output_padding, self.output_padding, self.output_padding),
            (1, 1, 1),  # dilation
            self.groups
        )