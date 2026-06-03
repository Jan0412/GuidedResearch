import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Input tensor: (B, C_in, H, W)
    w_ptr,  # Weight tensor: (C_in, C_out // groups, kH, kW)
    b_ptr,  # Bias tensor: (C_out,) or nullptr
    out_ptr,  # Output tensor: (B, C_out, H_out, W_out)
    B, C_in, C_out, groups,
    H, W, kH, kW,
    stride, padding, output_padding,
    H_out, W_out,
    stride_h, stride_w,
    stride_in_h, stride_in_w,
    stride_out_h, stride_out_w,
    C_in_per_group, C_out_per_group,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Compute output position
    out_idx = pid_b * stride_out_h * H_out * W_out + pid_c_out * stride_out_w * W_out + pid_h * W_out + pid_w
    
    # Compute input channel group
    group_id = pid_c_out // C_out_per_group
    c_out_offset = pid_c_out % C_out_per_group
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_B,), dtype=tl.float32)
    
    # Loop over input channels in this group
    for c_in_start in range(0, C_in_per_group, BLOCK_SIZE_C_IN):
        c_in_idx = group_id * C_in_per_group + c_in_start + tl.arange(0, BLOCK_SIZE_C_IN)
        c_in_mask = c_in_idx < (group_id + 1) * C_in_per_group
        
        # For each input channel, accumulate contributions from all kernel positions
        for kh in range(kH):
            for kw in range(kW):
                # Compute corresponding input position
                h_in = pid_h - kh + padding - output_padding // 2
                w_in = pid_w - kw + padding - output_padding // 2
                
                # Check if this input position is valid
                valid_in = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                
                if valid_in:
                    # Compute input index
                    in_idx = pid_b * stride_in_h * H * W + c_in_idx[:, None] * stride_in_w * W + h_in * W + w_in
                    in_mask = c_in_mask[:, None] & (tl.arange(0, BLOCK_SIZE_B)[:, None] < B)
                    
                    # Compute weight index
                    w_idx = c_in_idx[:, None] * kW * kH * C_out + c_out_offset * kW * kH + kh * kW + kw
                    w_mask = c_in_mask[:, None] & (tl.arange(0, BLOCK_SIZE_C_OUT) < 1)
                    
                    # Load values
                    x_val = tl.load(x_ptr + in_idx, mask=in_mask, other=0.0)
                    w_val = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0)
                    
                    # Accumulate
                    acc += tl.sum(x_val * w_val, axis=0)
    
    # Add bias if present
    if b_ptr is not None:
        bias_idx = pid_c_out
        bias_val = tl.load(b_ptr + bias_idx)
        acc += bias_val
    
    # Store result
    out_mask = (tl.arange(0, BLOCK_SIZE_B) < B) & (pid_c_out < C_out) & (pid_h < H_out) & (pid_w < W_out)
    tl.store(out_ptr + out_idx, acc.to(x_ptr.dtype.element_ty), mask=out_mask)


class TritonConvTranspose2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, in_channels, out_channels, kernel_size, 
                stride, padding, output_padding, groups):
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        
        # Infer batch size and spatial dimensions from input
        B, C_in, H, W = x.shape
        kH, kW = kernel_size, kernel_size
        
        # Calculate output dimensions
        H_out = (H - 1) * stride - 2 * padding + output_padding + kH
        W_out = (W - 1) * stride - 2 * padding + output_padding + kW
        
        # Create output tensor
        out = torch.empty(B, out_channels, H_out, W_out, dtype=x.dtype, device=x.device)
        
        # Calculate strides
        stride_h = x.stride(0)
        stride_in_h = x.stride(1)
        stride_in_w = x.stride(3)
        stride_out_h = out.stride(0)
        stride_out_w = out.stride(3)
        
        # Parameters
        C_out = out_channels
        C_in_per_group = in_channels // groups
        C_out_per_group = out_channels // groups
        
        # Grid dimensions
        BLOCK_SIZE_B = 1
        BLOCK_SIZE_C_OUT = 16
        BLOCK_SIZE_K = 1
        BLOCK_SIZE_C_IN = 16
        
        grid = (
            (B + BLOCK_SIZE_B - 1) // BLOCK_SIZE_B,
            (C_out + BLOCK_SIZE_C_OUT - 1) // BLOCK_SIZE_C_OUT,
            (H_out + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K,
            (W_out + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K,
        )
        
        # Launch kernel
        conv_transpose2d_kernel[grid](
            x, weight, bias, out,
            B, C_in, C_out, groups,
            H, W, kH, kW,
            stride, padding, output_padding,
            H_out, W_out,
            stride, stride,
            stride_in_h, stride_in_w,
            stride_out_h, stride_out_w,
            C_in_per_group, C_out_per_group,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_C_IN=BLOCK_SIZE_C_IN,
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.params = (in_channels, out_channels, kernel_size, stride, padding, output_padding, groups)
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # This is a simplified implementation - full backward would require more kernels
        # For now, fallback to PyTorch's implementation for backward
        x, weight, bias = ctx.saved_tensors
        in_channels, out_channels, kernel_size, stride, padding, output_padding, groups = ctx.params
        
        return torch.nn.functional.conv_transpose2d(
            grad_output, weight, bias, stride, padding, 
            output_padding, groups, x.shape[2:4]
        ), None, None, None, None, None, None, None, None, None


def triton_conv_transpose2d(x, weight, bias, in_channels, out_channels, kernel_size, 
                            stride, padding, output_padding, groups):
    return TritonConvTranspose2d.apply(
        x, weight, bias, in_channels, out_channels, kernel_size,
        stride, padding, output_padding, groups
    )


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Initialize the parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels // groups, kernel_size, kernel_size)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            self.in_channels, self.out_channels, self.kernel_size,
            self.stride, self.padding, self.output_padding, self.groups
        )