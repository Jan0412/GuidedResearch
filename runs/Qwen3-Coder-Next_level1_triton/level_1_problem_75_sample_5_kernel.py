import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    # Pointers to input/output tensors
    x_ptr, y_ptr, w_ptr, b_ptr,
    # Dimensions
    batch_size, in_channels, out_channels, 
    in_h, in_w, out_h, out_w,
    kernel_h, kernel_w,
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    groups,
    # Strides for memory access
    x_batch_stride, x_channel_stride, x_h_stride, x_w_stride,
    y_batch_stride, y_channel_stride, y_h_stride, y_w_stride,
    w_out_channel_stride, w_in_channel_stride, w_h_stride, w_w_stride,
    # Block sizes
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_OUT_C: tl.constexpr, BLOCK_SIZE_IN_C: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c = tl.program_id(3)
    
    # Compute output coordinates
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    out_c = pid_c * BLOCK_SIZE_OUT_C + tl.arange(0, BLOCK_SIZE_OUT_C)
    
    # Create masks for valid indices
    h_mask = out_h < out_h
    w_mask = out_w < out_w
    c_mask = out_c < out_channels
    
    # Compute the group index and local channel indices
    group_size = out_channels // groups
    group_id = pid_c * BLOCK_SIZE_OUT_C // group_size
    local_out_c = out_c % group_size
    
    # Compute input channel range for this group
    in_c_per_group = in_channels // groups
    in_c_start = group_id * in_c_per_group
    in_c = in_c_start + tl.arange(0, BLOCK_SIZE_IN_C)
    in_c_mask = in_c < in_channels
    
    # Initialize accumulator for output
    output = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_OUT_C), dtype=tl.float32)
    
    # Iterate over input height
    for ih in range(in_h):
        # Compute corresponding output position for this input position
        oh = ih * stride_h - pad_h + ih * (dilation_h - 1)
        oh_offsets = oh + tl.arange(0, BLOCK_SIZE_H)
        oh_mask = (oh_offsets >= 0) & (oh_offsets < out_h)
        
        # Iterate over input width
        for iw in range(in_w):
            # Compute corresponding output position for this input position
            ow = iw * stride_w - pad_w + iw * (dilation_w - 1)
            ow_offsets = ow + tl.arange(0, BLOCK_SIZE_W)
            ow_mask = (ow_offsets >= 0) & (ow_offsets < out_w)
            
            # Load input values: shape (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_IN_C)
            x_oh = oh_offsets[:, None, None]
            x_ow = ow_offsets[None, :, None]
            x_ic = in_c[None, None, :]
            
            # Compute x indices
            x_indices = (pid_b * x_batch_stride + 
                        x_ic * x_channel_stride + 
                        x_oh * x_h_stride + 
                        x_ow * x_w_stride)
            
            # Load input
            x_val = tl.load(x_ptr + x_indices, 
                          mask=(x_ic < in_channels) & 
                               (x_oh < out_h) & 
                               (x_ow < out_w), 
                          other=0.0)
            
            # Iterate over kernel dimensions
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    # Compute corresponding output position for this kernel position
                    target_oh = oh_offsets[:, None, None] + kh * dilation_h
                    target_ow = ow_offsets[None, :, None] + kw * dilation_w
                    
                    # Compute kernel indices
                    kernel_ic = in_c[None, None, :]
                    kernel_oc = local_out_c[None, None, :] + tl.arange(0, BLOCK_SIZE_OUT_C)[:, None, None]
                    
                    # Load kernel values
                    kernel_indices = (kernel_oc * w_out_channel_stride + 
                                     kernel_ic * w_in_channel_stride + 
                                     kh * w_h_stride + 
                                     kw * w_w_stride)
                    
                    w_val = tl.load(w_ptr + kernel_indices, 
                                  mask=(kernel_ic < in_channels) & 
                                       (kernel_oc < group_size), 
                                  other=0.0)
                    
                    # Compute accumulation mask
                    acc_mask = ((target_oh >= 0) & (target_oh < out_h) & 
                              (target_ow >= 0) & (target_ow < out_w) &
                              (kernel_ic < in_channels) & 
                              (kernel_oc < group_size))
                    
                    # Accumulate
                    output += tl.where(acc_mask, 
                                      x_val * w_val, 
                                      tl.zeros_like(output))
    
    # Add bias if present
    if b_ptr is not None:
        b_val = tl.load(b_ptr + out_c, mask=c_mask)
        output += b_val[None, None, :]
    
    # Store output
    y_oh = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    y_ow = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    y_oc = pid_c * BLOCK_SIZE_OUT_C + tl.arange(0, BLOCK_SIZE_OUT_C)
    
    y_indices = (pid_b * y_batch_stride + 
                y_oc[None, None, :] * y_channel_stride + 
                y_oh[:, None, None] * y_h_stride + 
                y_ow[None, :, None] * y_w_stride)
    
    tl.store(y_ptr + y_indices, output, mask=(y_oh[:, None, None] < out_h) & 
             (y_ow[None, :, None] < out_w) & 
             (y_oc[None, None, :] < out_channels))


class TritonConvTranspose2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, 
                stride, padding, dilation, groups):
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        
        # Extract dimensions
        batch_size, in_channels, in_h, in_w = x.shape
        out_channels, _, kernel_h, kernel_w = weight.shape
        
        stride_h, stride_w = stride
        pad_h, pad_w = padding
        dilation_h, dilation_w = dilation
        groups = groups
        
        # Calculate output dimensions
        out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + 1
        out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1
        
        # Allocate output tensor
        y = torch.empty(batch_size, out_channels, out_h, out_w, 
                       dtype=x.dtype, device=x.device)
        
        # Grid dimensions
        BLOCK_SIZE_H = 8
        BLOCK_SIZE_W = 8
        BLOCK_SIZE_OUT_C = 8
        BLOCK_SIZE_IN_C = 8
        
        grid = (batch_size, 
                triton.cdiv(out_h, BLOCK_SIZE_H),
                triton.cdiv(out_w, BLOCK_SIZE_W),
                triton.cdiv(out_channels, BLOCK_SIZE_OUT_C))
        
        # Calculate strides
        x_batch_stride = x.stride(0)
        x_channel_stride = x.stride(1)
        x_h_stride = x.stride(2)
        x_w_stride = x.stride(3)
        
        y_batch_stride = y.stride(0)
        y_channel_stride = y.stride(1)
        y_h_stride = y.stride(2)
        y_w_stride = y.stride(3)
        
        w_out_channel_stride = weight.stride(0)
        w_in_channel_stride = weight.stride(1)
        w_h_stride = weight.stride(2)
        w_w_stride = weight.stride(3)
        
        # Launch kernel
        conv_transpose2d_kernel[grid](
            x, y, weight, bias,
            batch_size, in_channels, out_channels,
            in_h, in_w, out_h, out_w,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            dilation_h, dilation_w,
            groups,
            x_batch_stride, x_channel_stride, x_h_stride, x_w_stride,
            y_batch_stride, y_channel_stride, y_h_stride, y_w_stride,
            w_out_channel_stride, w_in_channel_stride, w_h_stride, w_w_stride,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_W=BLOCK_SIZE_W,
            BLOCK_SIZE_OUT_C=BLOCK_SIZE_OUT_C,
            BLOCK_SIZE_IN_C=BLOCK_SIZE_IN_C
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.input_size = (in_h, in_w)
        ctx.output_size = (out_h, out_w)
        
        return y


def triton_conv_transpose2d(x, weight, bias=None, stride=(1,1), 
                           padding=(0,0), dilation=(1,1), groups=1):
    return TritonConvTranspose2d.apply(x, weight, bias, stride, 
                                      padding, dilation, groups)


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), 
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, 
                       kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(x, self.weight, self.bias, 
                                      self.stride, self.padding, 
                                      self.dilation, self.groups)