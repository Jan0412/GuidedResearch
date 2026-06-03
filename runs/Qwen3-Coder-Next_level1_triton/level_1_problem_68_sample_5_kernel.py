import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose3d_kernel(
    # Pointers to tensors
    x_ptr,  # [B, C_in, D_in, H_in, W_in]
    w_ptr,  # [C_in, C_out, K_d, K_h, K_w]
    b_ptr,  # [C_out] or None
    out_ptr,  # [B, C_out, D_out, H_out, W_out]
    # Tensor dimensions
    B, C_in, D_in, H_in, W_in,
    C_out, K_d, K_h, K_w,
    D_out, H_out, W_out,
    # Strides
    stride_x, stride_x_c, stride_x_d, stride_x_h, stride_x_w,
    stride_w, stride_w_c, stride_w_kd, stride_w_kh, stride_w_kw,
    stride_out, stride_out_c, stride_out_d, stride_out_h, stride_out_w,
    # Output padding and stride parameters
    pad_d, pad_h, pad_w,
    stride_d, stride_h, stride_w,
    output_pad_d, output_pad_h, output_pad_w,
    # Block sizes
    BLOCK_C_in: tl.constexpr,
    BLOCK_C_out: tl.constexpr,
    BLOCK_K_d: tl.constexpr,
    BLOCK_K_h: tl.constexpr,
    BLOCK_K_w: tl.constexpr,
    BLOCK_D_out: tl.constexpr,
    BLOCK_H_out: tl.constexpr,
    BLOCK_W_out: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Compute output positions
    out_d = pid_d * BLOCK_D_out + tl.arange(0, BLOCK_D_out)
    out_h = pid_h * BLOCK_H_out + tl.arange(0, BLOCK_H_out)
    out_w = pid_w * BLOCK_W_out + tl.arange(0, BLOCK_W_out)
    
    # Check bounds for output positions
    mask_d = out_d < D_out
    mask_h = out_h < H_out
    mask_w = out_w < W_out
    mask_dh = mask_d[:, None]
    mask_hw = mask_h[:, None] & mask_w[None, :]
    mask_dhw = mask_d[:, None, None] & mask_hw[None, :, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_D_out, BLOCK_H_out, BLOCK_W_out), dtype=tl.float32)
    
    # Iterate over input channels and kernel dimensions
    for c_in_start in range(0, C_in, BLOCK_C_in):
        c_in_range = c_in_start + tl.arange(0, BLOCK_C_in)
        mask_c_in = c_in_range < C_in
        
        for kd_start in range(0, K_d, BLOCK_K_d):
            kd_range = kd_start + tl.arange(0, BLOCK_K_d)
            mask_kd = kd_range < K_d
            
            for kh_start in range(0, K_h, BLOCK_K_h):
                kh_range = kh_start + tl.arange(0, BLOCK_K_h)
                mask_kh = kh_range < K_h
                
                for kw_start in range(0, K_w, BLOCK_K_w):
                    kw_range = kw_start + tl.arange(0, BLOCK_K_w)
                    mask_kw = kw_range < K_w
                    
                    # Compute input positions from output positions and kernel positions
                    # For transposed conv: input_pos = (output_pos - (kernel_pos - 1) + pad - output_pad) / stride
                    in_d = (out_d[:, None, None] - (kd_range[None, None, :] - 1) * stride_d + pad_d - output_pad_d) // stride_d
                    in_h = (out_h[None, :, None] - (kh_range[None, :, None] - 1) * stride_h + pad_h - output_pad_h) // stride_h
                    in_w = (out_w[None, None, :] - (kw_range[:, None, None] - 1) * stride_w + pad_w - output_pad_w) // stride_w
                    
                    # Check if input positions are valid
                    valid_in_d = (in_d >= 0) & (in_d < D_in)
                    valid_in_h = (in_h >= 0) & (in_h < H_in)
                    valid_in_w = (in_w >= 0) & (in_w < W_in)
                    valid_in = valid_in_d & valid_in_h & valid_in_w
                    
                    # Load input values: [BLOCK_D_out, BLOCK_H_out, BLOCK_W_out, BLOCK_C_in]
                    # We need to gather input values based on computed positions
                    x_offset_d = in_d * stride_x_d
                    x_offset_h = in_h * stride_x_h
                    x_offset_w = in_w * stride_x_w
                    x_offsets = x_offset_d[:, :, :, None] + x_offset_h[:, :, :, None] + x_offset_w[:, :, :, None] + c_in_range[None, None, None, :] * stride_x_c
                    
                    # Compute base pointer offset for batch
                    x_base = x_ptr + pid_b * stride_x
                    
                    # Load input with masking
                    x_vals = tl.zeros((BLOCK_D_out, BLOCK_H_out, BLOCK_W_out, BLOCK_C_in), dtype=tl.float32)
                    for i_c in range(BLOCK_C_in):
                        mask_c = mask_c_in[i_c]
                        if mask_c:
                            # Load with proper masking
                            for i_d in range(BLOCK_D_out):
                                for i_h in range(BLOCK_H_out):
                                    for i_w in range(BLOCK_W_out):
                                        if valid_in[i_d, i_h, i_w] and mask_d[i_d] and mask_h[i_h] and mask_w[i_w]:
                                            ptr = x_base + x_offsets[i_d, i_h, i_w, i_c]
                                            x_vals = tl.where(
                                                (mask_d[i_d] & mask_h[i_h] & mask_w[i_w])[:, :, :, None],
                                                tl.load(ptr),
                                                x_vals
                                            )
                    
                    # Load weights: [BLOCK_C_in, BLOCK_C_out, BLOCK_K_d, BLOCK_K_h, BLOCK_K_w]
                    # Weight layout: [C_in, C_out, K_d, K_h, K_w]
                    w_offsets = (c_in_range[:, None, None, None, None] * stride_w +
                                pid_c_out * stride_w_c +
                                kd_range[None, :, None, None, None] * stride_w_kd +
                                kh_range[None, None, :, None, None] * stride_w_kh +
                                kw_range[None, None, None, :, None] * stride_w_kw)
                    
                    # For simplicity, we'll load weights per channel
                    for i_c in range(BLOCK_C_in):
                        if mask_c_in[i_c]:
                            w_val = tl.load(w_ptr + w_offsets[i_c, :, :, :, :] + pid_c_out * stride_w_c)
                            # Reshape w_val to [BLOCK_C_out, BLOCK_K_d, BLOCK_K_h, BLOCK_K_w]
                            w_val = tl.reshape(w_val, (BLOCK_C_out, BLOCK_K_d, BLOCK_K_h, BLOCK_K_w))
                            
                            # Multiply and accumulate
                            # x_vals[i_d, i_h, i_w, i_c] * w_val[c_out, kd, kh, kw]
                            # Accumulate over kd, kh, kw, c_in
                            for i_d in range(BLOCK_D_out):
                                for i_h in range(BLOCK_H_out):
                                    for i_w in range(BLOCK_W_out):
                                        if valid_in[i_d, i_h, i_w] and mask_d[i_d] and mask_h[i_h] and mask_w[i_w]:
                                            for i_kd in range(BLOCK_K_d):
                                                for i_kh in range(BLOCK_K_h):
                                                    for i_kw in range(BLOCK_K_w):
                                                        if mask_kd[i_kd] and mask_kh[i_kh] and mask_kw[i_kw]:
                                                            acc = tl.where(
                                                                mask_dhw,
                                                                acc + x_vals[i_d, i_h, i_w, i_c] * w_val[:, i_kd, i_kh, i_kw],
                                                                acc
                                                            )
    
    # Store result
    out_offsets = (pid_b * stride_out +
                  pid_c_out * stride_out_c +
                  out_d[:, None, None] * stride_out_d +
                  out_h[None, :, None] * stride_out_h +
                  out_w[None, None, :] * stride_out_w)
    
    # Apply bias if present
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + pid_c_out * stride_w_c)
        acc = acc + bias_val
    
    # Store output
    tl.store(out_ptr + out_offsets, acc.to(x_ptr.dtype.element_ty), mask=mask_dhw)


class TritonConvTranspose3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, 
                stride, padding, output_padding, groups):
        # Save parameters for potential backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        ctx.groups = groups
        
        # Compute output shape
        B, C_in, D_in, H_in, W_in = x.shape
        C_out, C_in_grouped, K_d, K_h, K_w = weight.shape
        
        # For groups > 1, we'd need to adjust, but assuming groups=1 for now
        D_out = (D_in - 1) * stride[0] - 2 * padding[0] + (K_d - 1) + 1 + output_padding[0]
        H_out = (H_in - 1) * stride[1] - 2 * padding[1] + (K_h - 1) + 1 + output_padding[1]
        W_out = (W_in - 1) * stride[2] - 2 * padding[2] + (K_w - 1) + 1 + output_padding[2]
        
        # Create output tensor
        out = torch.empty(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
        
        # Grid configuration
        BLOCK_C_in = 8
        BLOCK_C_out = 8
        BLOCK_K_d = 3
        BLOCK_K_h = 3
        BLOCK_K_w = 3
        BLOCK_D_out = 4
        BLOCK_H_out = 4
        BLOCK_W_out = 4
        
        grid = (B, 
                triton.cdiv(C_out, BLOCK_C_out),
                triton.cdiv(D_out, BLOCK_D_out),
                triton.cdiv(H_out, BLOCK_H_out),
                triton.cdiv(W_out, BLOCK_W_out))
        
        # Strides
        stride_x = x.stride()
        stride_w = weight.stride()
        
        conv_transpose3d_kernel[grid](
            x, weight, bias,
            out,
            B, C_in, D_in, H_in, W_in,
            C_out, K_d, K_h, K_w,
            D_out, H_out, W_out,
            stride_x[0], stride_x[1], stride_x[2], stride_x[3], stride_x[4],
            stride_w[0], stride_w[1], stride_w[2], stride_w[3], stride_w[4],
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
            padding[0], padding[1], padding[2],
            stride[0], stride[1], stride[2],
            output_padding[0], output_padding[1], output_padding[2],
            BLOCK_C_in=BLOCK_C_in,
            BLOCK_C_out=BLOCK_C_out,
            BLOCK_K_d=BLOCK_K_d,
            BLOCK_K_h=BLOCK_K_h,
            BLOCK_K_w=BLOCK_K_w,
            BLOCK_D_out=BLOCK_D_out,
            BLOCK_H_out=BLOCK_H_out,
            BLOCK_W_out=BLOCK_W_out
        )
        
        return out


def triton_conv_transpose3d(x, weight, bias=None, stride=(1, 1, 1), 
                           padding=(0, 0, 0), output_padding=(0, 0, 0), groups=1):
    return TritonConvTranspose3d.apply(x, weight, bias, 
                                      stride, padding, output_padding, groups)


class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (kernel_depth, kernel_width, kernel_height), 
                             where kernel_width == kernel_height.
        stride (tuple, optional): Stride of the convolution. Defaults to (1, 1, 1).
        padding (tuple, optional): Padding applied to the input. Defaults to (0, 0, 0).
        output_padding (tuple, optional): Additional size added to one side of the output shape. Defaults to (0, 0, 0).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels, *kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming initialization for transposed convolution
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, width, height).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, width_out, height_out).
        """
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Use our custom Triton implementation
        return triton_conv_transpose3d(x, self.weight, self.bias,
                                      self.stride, self.padding, 
                                      self.output_padding, self.groups)