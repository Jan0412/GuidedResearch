import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose3d_kernel(
    X,  # Input tensor: (B, C_in, D_in, H_in, W_in)
    K,  # Kernel tensor: (C_in, C_out // G, D_k, H_k, W_k)
    B,  # Bias tensor: (C_out,) or None
    Y,  # Output tensor: (B, C_out, D_out, H_out, W_out)
    # Dimensions
    B_size, C_in, D_in, H_in, W_in,
    C_out, D_out, H_out, W_out,
    D_k, H_k, W_k,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    output_padding_d, output_padding_h, output_padding_w,
    groups: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr = 4,
    BLOCK_SIZE_C_OUT: tl.constexpr = 32,
    BLOCK_SIZE_D: tl.constexpr = 8,
    BLOCK_SIZE_H: tl.constexpr = 8,
    BLOCK_SIZE_W: tl.constexpr = 8,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Calculate group and channel indices
    c_out_per_group = C_out // groups
    group_id = pid_c_out // c_out_per_group
    c_out_in_group = pid_c_out % c_out_per_group
    
    # Compute output position
    b = pid_b
    d_out = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    h_out = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w_out = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for output dimensions
    mask_d_out = d_out < D_out
    mask_h_out = h_out < H_out
    mask_w_out = w_out < W_out
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over input channels
    for c_in in range(C_in):
        # Compute corresponding input position for each output position
        # For transposed convolution: d_in = (d_out - output_padding_d - 1) // stride_d + 1
        # But we need to compute which input positions contribute to each output position
        
        # Compute kernel offset indices
        d_k_start = (d_out[:, None, None] - output_padding_d - 1 - (stride_d - 1)) // stride_d + 1
        h_k_start = (h_out[None, :, None] - output_padding_h - 1 - (stride_h - 1)) // stride_h + 1
        w_k_start = (w_out[None, None, :] - output_padding_w - 1 - (stride_w - 1)) // stride_w + 1
        
        # Adjust for padding: valid input positions
        d_in = d_k_start + tl.arange(0, BLOCK_SIZE_D)[:, None, None]
        h_in = h_k_start + tl.arange(0, BLOCK_SIZE_H)[None, :, None]
        w_in = w_k_start + tl.arange(0, BLOCK_SIZE_W)[None, None, :]
        
        # Kernel position indices
        d_k = tl.arange(0, BLOCK_SIZE_D)[:, None, None] - (d_k_start - tl.arange(0, BLOCK_SIZE_D)[:, None, None])
        h_k = tl.arange(0, BLOCK_SIZE_H)[None, :, None] - (h_k_start - tl.arange(0, BLOCK_SIZE_H)[None, :, None])
        w_k = tl.arange(0, BLOCK_SIZE_W)[None, None, :] - (w_k_start - tl.arange(0, BLOCK_SIZE_W)[None, None, :])
        
        # Simplify: compute directly which kernel indices correspond to each output position
        # For output position (d_out, h_out, w_out), input position (d_in, h_in, w_in) contributes
        # if d_out = d_in * stride_d - padding_d + d_k * (stride_d - 1) + output_padding_d
        # So d_k = d_out - d_in * stride_d + padding_d - output_padding_d
        
        # Actually, let's use the standard formula:
        # For transposed conv: out = conv(input, kernel)^T
        # out[b, c_out, d_out, h_out, w_out] = sum_{c_in, d_k, h_k, w_k} 
        #     input[b, c_in, d_in, h_in, w_in] * kernel[c_in, c_out, d_k, h_k, w_k]
        # where d_in = (d_out - d_k + stride_d - 1) // stride_d, etc.
        
        # More direct approach: iterate over kernel positions and find contributing input positions
        # For kernel position (d_k, h_k, w_k), it contributes to output positions:
        # d_out = d_in * stride_d + d_k - padding_d + output_padding_d
        
        # Let's compute directly: for output (d_out, h_out, w_out), which input (d_in, h_in, w_in)
        # and kernel (d_k, h_k, w_k) contribute?
        # d_out = d_in * stride_d - padding_d + d_k * (1) + output_padding_d
        # So d_in = (d_out + padding_d - output_padding_d - d_k) / stride_d
        
        # We'll iterate over kernel positions
        for d_k_idx in range(D_k):
            d_in_idx = (d_out[:, None, None] + padding_d - output_padding_d - d_k_idx) // stride_d
            mask_d_valid = (d_in_idx >= 0) & (d_in_idx < D_in) & ((d_out[:, None, None] + padding_d - output_padding_d - d_k_idx) % stride_d == 0)
            
            for h_k_idx in range(H_k):
                h_in_idx = (h_out[None, :, None] + padding_h - output_padding_h - h_k_idx) // stride_h
                mask_h_valid = (h_in_idx >= 0) & (h_in_idx < H_in) & ((h_out[None, :, None] + padding_h - output_padding_h - h_k_idx) % stride_h == 0)
                
                for w_k_idx in range(W_k):
                    w_in_idx = (w_out[None, None, :] + padding_w - output_padding_w - w_k_idx) // stride_w
                    mask_w_valid = (w_in_idx >= 0) & (w_in_idx < W_in) & ((w_out[None, None, :] + padding_w - output_padding_w - w_k_idx) % stride_w == 0)
                    
                    # Combined mask
                    mask_valid = mask_d_valid & mask_h_valid & mask_w_valid
                    
                    # Load input values
                    d_in_flat = d_in_idx[mask_valid]
                    h_in_flat = h_in_idx[mask_valid]
                    w_in_flat = w_in_idx[mask_valid]
                    
                    if tl.sum(mask_valid) > 0:
                        # Compute indices for input tensor
                        input_indices = (
                            b * (C_in * D_in * H_in * W_in) +
                            c_in * (D_in * H_in * W_in) +
                            d_in_flat * (H_in * W_in) +
                            h_in_flat * W_in +
                            w_in_flat
                        )
                        x_vals = tl.load(X + input_indices)
                        
                        # Compute indices for kernel tensor
                        # Kernel layout: (C_in, C_out // G, D_k, H_k, W_k)
                        kernel_indices = (
                            c_in * (c_out_per_group * D_k * H_k * W_k) +
                            c_out_in_group * (D_k * H_k * W_k) +
                            d_k_idx * (H_k * W_k) +
                            h_k_idx * W_k +
                            w_k_idx
                        )
                        k_val = tl.load(K + kernel_indices)
                        
                        # Accumulate
                        acc += tl.where(mask_valid, x_vals * k_val, 0.0)
    
    # Add bias if present
    if B is not None:
        bias_val = tl.load(B + pid_c_out)
        acc += bias_val
    
    # Store output
    y_indices = (
        b * (C_out * D_out * H_out * W_out) +
        pid_c_out * (D_out * H_out * W_out) +
        d_out[:, None, None] * (H_out * W_out) +
        h_out[None, :, None] * W_out +
        w_out[None, None, :]
    )
    tl.store(Y + y_indices, acc, mask=mask_d_out[:, None, None] & mask_h_out[None, :, None] & mask_w_out[None, None, :])


class ConvTranspose3dTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, output_padding, groups):
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        
        # Get dimensions
        B, C_in, D_in, H_in, W_in = x.shape
        C_in2, C_out_per_group, D_k, H_k, W_k = weight.shape
        assert C_in == C_in2, "Input channels must match"
        C_out = C_out_per_group * groups
        
        # Compute output dimensions
        D_out = (D_in - 1) * stride[0] - 2 * padding[0] + D_k + output_padding[0]
        H_out = (H_in - 1) * stride[1] - 2 * padding[1] + H_k + output_padding[1]
        W_out = (W_in - 1) * stride[2] - 2 * padding[2] + W_k + output_padding[2]
        
        # Create output tensor
        y = torch.empty(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
        
        # Define grid
        grid = lambda meta: (
            B,
            triton.cdiv(C_out, meta['BLOCK_SIZE_C_OUT']),
            triton.cdiv(D_out, meta['BLOCK_SIZE_D']),
            triton.cdiv(H_out, meta['BLOCK_SIZE_H']),
            triton.cdiv(W_out, meta['BLOCK_SIZE_W']),
        )
        
        # Launch kernel
        conv_transpose3d_kernel[grid](
            x, weight, bias, y,
            B, C_in, D_in, H_in, W_in,
            C_out, D_out, H_out, W_out,
            D_k, H_k, W_k,
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2],
            output_padding[0], output_padding[1], output_padding[2],
            groups,
            BLOCK_SIZE_B=4,
            BLOCK_SIZE_C_OUT=16,
            BLOCK_SIZE_D=4,
            BLOCK_SIZE_H=4,
            BLOCK_SIZE_W=4,
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        ctx.groups = groups
        
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        # This is a simplified implementation - in practice you'd want proper
        # backward kernels for training, but for inference-only we can return None
        x, weight = ctx.saved_tensors
        return None, None, None, None, None, None, None


class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution operation with asymmetric input and kernel sizes.
    Uses optimized Triton kernel for inference.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters but don't use the PyTorch implementation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Initialize weights (same as PyTorch)
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels // groups, *kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Kaiming uniform initialization similar to PyTorch
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using Triton kernel.
        """
        return ConvTranspose3dTriton.apply(
            x, self.weight, self.bias,
            self.stride, self.padding, self.output_padding, self.groups
        )