import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch, in_channels, out_channels,
    h_in, w_in, h_out, w_out,
    k_size, stride, padding,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_oc, w_stride_kh, w_stride_kw,
    out_stride_b, out_stride_oc, out_stride_h, out_stride_w,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    # Program IDs
    pid_b_c = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Decompose pid_b_c into batch index and output channel index
    b = pid_b_c // out_channels
    oc = pid_b_c % out_channels
    
    # Calculate corresponding input channel for depthwise conv
    # For nn.Conv2d with groups=in_channels, each input channel maps to (out_channels // in_channels) output channels
    ic = oc // (out_channels // in_channels)

    # Output offsets
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offsets = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    
    # Mask for output boundaries
    out_mask = (h_offsets[:, None] < h_out) & (w_offsets[None, :] < w_out)

    # Accumulator for the convolution result
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Iterate over the kernel window
    for kh in range(k_size):
        for kw in range(k_size):
            # Compute input coordinates
            h_in_offsets = h_offsets * stride - padding + kh
            w_in_offsets = w_offsets * stride - padding + kw
            
            # Mask for input boundaries (padding)
            in_mask = (h_in_offsets[:, None] >= 0) & (h_in_offsets[:, None] < h_in) & \
                      (w_in_offsets[None, :] >= 0) & (w_in_offsets[None, :] < w_in)
            
            # Load input window
            # x_ptr + b*stride_b + ic*stride_c + h_off*stride_h + w_off*stride_w
            x_ptrs = x_ptr + b * x_stride_b + ic * x_stride_c + \
                     h_in_offsets[:, None] * x_stride_h + w_in_offsets[None, :] * x_stride_w
            
            x_vals = tl.load(x_ptrs, mask=in_mask, other=0.0)
            
            # Load weight for this specific channel and kernel position
            # weight_ptr + oc*stride_oc + kh*stride_kh + kw*stride_kw
            w_ptr_val = weight_ptr + oc * w_stride_oc + kh * w_stride_kh + kw * w_stride_kw
            w_val = tl.load(w_ptr_val)
            
            acc += x_vals * w_val

    # Add bias if available
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + oc)
        acc += bias_val

    # Store result
    out_ptrs = out_ptr + b * out_stride_b + oc * out_stride_oc + \
               h_offsets[:, None] * out_stride_h + w_offsets[None, :] * out_stride_w
    tl.store(out_ptrs, acc, mask=out_mask)


def triton_depthwise_conv2d(x, weight, bias, stride, padding):
    # Tensors must be contiguous and on CUDA
    x = x.contiguous()
    weight = weight.contiguous()
    
    batch, in_channels, h_in, w_in = x.shape
    out_channels, _, k_size, _ = weight.shape
    
    h_out = (h_in + 2 * padding - k_size) // stride + 1
    w_out = (w_in + 2 * padding - k_size) // stride + 1
    
    out = torch.empty((batch, out_channels, h_out, w_out), device=x.device, dtype=x.dtype)
    
    # Strides
    x_stride_b, x_stride_c, x_stride_h, x_stride_w = x.stride()
    w_stride_oc, w_stride_c, w_stride_kh, w_stride_kw = weight.stride()
    out_stride_b, out_stride_oc, out_stride_h, out_stride_w = out.stride()
    
    BLOCK_H = 16
    BLOCK_W = 16
    
    grid = (
        batch * out_channels, 
        (h_out + BLOCK_H - 1) // BLOCK_H, 
        (w_out + BLOCK_W - 1) // BLOCK_W
    )
    
    depthwise_conv2d_kernel[grid](
        x, weight, bias, out,
        batch, in_channels, out_channels,
        h_in, w_in, h_out, w_out,
        k_size, stride, padding,
        x_stride_b, x_stride_c, x_stride_h, x_stride_w,
        w_stride_oc, w_stride_kh, w_stride_kw,
        out_stride_b, out_stride_oc, out_stride_h, out_stride_w,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        # We keep the Conv2d layer to manage weights and bias, but we won't use its forward pass
        self.conv_params = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=(kernel_size, kernel_size), 
            stride=stride, padding=padding, 
            groups=in_channels, bias=bias
        )
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use Triton kernel instead of nn.Conv2d.forward
        return triton_depthwise_conv2d(
            x, 
            self.conv_params.weight, 
            self.conv_params.bias, 
            self.stride, 
            self.padding
        )