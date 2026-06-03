import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, depth, height, width)
    w_ptr,  # Weight tensor: (in_channels, out_channels, k_d, k_h, k_w)
    b_ptr,  # Bias tensor: (out_channels,) or None
    out_ptr,  # Output tensor: (batch, out_channels, out_d, out_h, out_w)
    # Dimensions
    batch_size, in_channels, out_channels,
    in_d, in_h, in_w,
    out_d, out_h, out_w,
    k_d, k_h, k_w,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    output_padding_d, output_padding_h, output_padding_w,
    # Block sizes for tiling
    BLOCK_BATCH: tl.constexpr,
    BLOCK_OUT_CH: tl.constexpr,
    BLOCK_IN_CH: tl.constexpr,
    BLOCK_KD: tl.constexpr,
    BLOCK_KH: tl.constexpr,
    BLOCK_KW: tl.constexpr,
    BLOCK_OUT_D: tl.constexpr,
    BLOCK_OUT_H: tl.constexpr,
    BLOCK_OUT_W: tl.constexpr,
):
    # Program IDs for output tensor
    pid_b = tl.program_id(0)
    pid_out_c = tl.program_id(1)
    pid_out_d = tl.program_id(2)
    pid_out_h = tl.program_id(3)
    pid_out_w = tl.program_id(4)
    
    # Create offsets for output tensor
    out_d_offset = pid_out_d * BLOCK_OUT_D
    out_h_offset = pid_out_h * BLOCK_OUT_H
    out_w_offset = pid_out_w * BLOCK_OUT_W
    
    out_d_range = tl.arange(0, BLOCK_OUT_D)
    out_h_range = tl.arange(0, BLOCK_OUT_H)
    out_w_range = tl.arange(0, BLOCK_OUT_W)
    
    out_d_mask = (out_d_offset + out_d_range) < out_d
    out_h_mask = (out_h_offset + out_h_range) < out_h
    out_w_mask = (out_w_offset + out_w_range) < out_w
    
    # Broadcast masks
    out_d_mask = out_d_mask[:, None, None]
    out_h_mask = out_h_mask[None, :, None]
    out_w_mask = out_w_mask[None, None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_OUT_D, BLOCK_OUT_H, BLOCK_OUT_W), dtype=tl.float32)
    
    # Iterate over input channels
    for ic in range(0, in_channels, BLOCK_IN_CH):
        in_c_range = ic + tl.arange(0, BLOCK_IN_CH)
        in_c_mask = in_c_range < in_channels
        
        # Iterate over kernel dimensions
        for kd in range(0, k_d, BLOCK_KD):
            for kh in range(0, k_h, BLOCK_KH):
                for kw in range(0, k_w, BLOCK_KW):
                    # Compute corresponding input positions
                    in_d_idx = (out_d_offset + out_d_range - (kd - padding_d) - output_padding_d) // stride_d
                    in_h_idx = (out_h_offset + out_h_range - (kh - padding_h) - output_padding_h) // stride_h
                    in_w_idx = (out_w_offset + out_w_range - (kw - padding_w) - output_padding_w) // stride_w
                    
                    # Check if input indices are within bounds
                    in_d_valid = (in_d_idx >= 0) & (in_d_idx < in_d)
                    in_h_valid = (in_h_idx >= 0) & (in_h_idx < in_h)
                    in_w_valid = (in_w_idx >= 0) & (in_w_idx < in_w)
                    
                    # Compute 1D indices for input tensor
                    # Layout: (batch, in_channels, depth, height, width)
                    in_batch_stride = in_channels * in_d * in_h * in_w
                    in_ch_stride = in_d * in_h * in_w
                    in_d_stride = in_h * in_w
                    in_h_stride = in_w
                    
                    # Calculate input indices
                    in_d_offset_actual = in_d_idx * in_d_stride
                    in_h_offset_actual = in_h_idx * in_h_stride
                    in_w_offset_actual = in_w_idx
                    
                    # Create masks for valid input positions
                    in_d_mask = in_d_valid[:, None, None]
                    in_h_mask = in_h_valid[None, :, None]
                    in_w_mask = in_w_valid[None, None, :]
                    combined_mask = in_d_mask & in_h_mask & in_w_mask
                    
                    # Load input values
                    in_batch_ptr = x_ptr + pid_b * in_batch_stride + in_c_range[:, None, None, None] * in_ch_stride
                    in_d_ptr = in_batch_ptr + in_d_offset_actual[None, :, None, None]
                    in_h_ptr = in_d_ptr + in_h_offset_actual[None, None, :, None]
                    in_w_ptr = in_h_ptr + in_w_offset_actual[None, None, None, :]
                    
                    # Reshape masks for input loading
                    in_c_mask_4d = in_c_mask[:, None, None, None]
                    mask_4d = combined_mask & in_c_mask_4d
                    
                    # Load input (with padding)
                    in_vals = tl.load(in_w_ptr, mask=mask_4d, other=0.0)
                    
                    # Load weights
                    # Weight layout: (in_channels, out_channels, k_d, k_h, k_w)
                    w_batch_stride = out_channels * k_d * k_h * k_w
                    w_ch_stride = k_d * k_h * k_w
                    w_kd_stride = k_h * k_w
                    w_kh_stride = k_w
                    
                    w_ptr_offset = ic * w_batch_stride + pid_out_c * w_ch_stride + \
                                  (kd + tl.arange(0, BLOCK_KD)[:, None, None, None]) * w_kd_stride + \
                                  (kh + tl.arange(0, BLOCK_KH)[None, :, None, None]) * w_kh_stride + \
                                  (kw + tl.arange(0, BLOCK_KW)[None, None, :, None])
                    
                    w_mask_4d = (kd + tl.arange(0, BLOCK_KD)[:, None, None, None] < k_d) & \
                               (kh + tl.arange(0, BLOCK_KH)[None, :, None, None] < k_h) & \
                               (kw + tl.arange(0, BLOCK_KW)[None, None, :, None] < k_w) & \
                               in_c_mask_4d
                    
                    w_vals = tl.load(w_ptr_offset, mask=w_mask_4d, other=0.0)
                    
                    # Accumulate: conv_transpose = sum over ic, kd, kh, kw of x[b,ic,*,*,*] * w[ic,oc,kd,kh,kw]
                    # Note: the kernel indices are reversed in transposed conv compared to regular conv
                    acc += tl.sum(in_vals * w_vals, axis=0)
    
    # Add bias if available
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_out_c)
        acc += bias
    
    # Store output
    out_batch_stride = out_channels * out_d * out_h * out_w
    out_ch_stride = out_d * out_h * out_w
    out_d_stride = out_h * out_w
    out_h_stride = out_w
    
    out_batch_ptr = out_ptr + pid_b * out_batch_stride + pid_out_c * out_ch_stride
    out_d_ptr = out_batch_ptr + out_d_offset * out_d_stride
    out_h_ptr = out_d_ptr + out_h_offset * out_h_stride
    out_w_ptr = out_h_ptr + out_w_offset * out_w_stride
    
    # Store with masks
    out_mask = out_d_mask & out_h_mask & out_w_mask
    tl.store(out_w_ptr, acc.to(x_ptr.dtype.element_ty), mask=out_mask)


def triton_conv_transpose3d(x, weight, bias, stride, padding, output_padding, dilation, groups):
    """
    Triton implementation of ConvTranspose3d.
    """
    # Extract dimensions
    batch_size, in_channels, in_d, in_h, in_w = x.shape
    out_channels, _, k_d, k_h, k_w = weight.shape
    
    # Calculate output dimensions
    out_d = (in_d - 1) * stride[0] - 2 * padding[0] + dilation[0] * (k_d - 1) + output_padding[0] + 1
    out_h = (in_h - 1) * stride[1] - 2 * padding[1] + dilation[1] * (k_h - 1) + output_padding[1] + 1
    out_w = (in_w - 1) * stride[2] - 2 * padding[2] + dilation[2] * (k_w - 1) + output_padding[2] + 1
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Create output tensor
    out = torch.empty(batch_size, out_channels, out_d, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Configure grid
    BLOCK_BATCH = 1
    BLOCK_OUT_CH = 16
    BLOCK_IN_CH = 16
    BLOCK_KD = 3
    BLOCK_KH = 3
    BLOCK_KW = 3
    BLOCK_OUT_D = 8
    BLOCK_OUT_H = 8
    BLOCK_OUT_W = 8
    
    grid = (
        (batch_size + BLOCK_BATCH - 1) // BLOCK_BATCH,
        (out_channels + BLOCK_OUT_CH - 1) // BLOCK_OUT_CH,
        (out_d + BLOCK_OUT_D - 1) // BLOCK_OUT_D,
        (out_h + BLOCK_OUT_H - 1) // BLOCK_OUT_H,
        (out_w + BLOCK_OUT_W - 1) // BLOCK_OUT_W,
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        k_d, k_h, k_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_OUT_CH=BLOCK_OUT_CH,
        BLOCK_IN_CH=BLOCK_IN_CH,
        BLOCK_KD=BLOCK_KD,
        BLOCK_KH=BLOCK_KH,
        BLOCK_KW=BLOCK_KW,
        BLOCK_OUT_D=BLOCK_OUT_D,
        BLOCK_OUT_H=BLOCK_OUT_H,
        BLOCK_OUT_W=BLOCK_OUT_W,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernels for ConvTranspose3d.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize with same parameters as original
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias
        
        # Create the weight and bias parameters (same as original ConvTranspose3d)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights (using Kaiming uniform as PyTorch does by default)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized transposed 3D convolution using Triton kernel.
        """
        # Ensure x is on the same device as weight
        x = x.to(self.weight.device)
        
        # Call Triton kernel
        return triton_conv_transpose3d(
            x, 
            self.weight, 
            self.bias, 
            stride=(self.stride, self.stride, self.stride),
            padding=(self.padding, self.padding, self.padding),
            output_padding=(self.output_padding, self.output_padding, self.output_padding),
            dilation=(1, 1, 1),
            groups=self.groups
        )


import math