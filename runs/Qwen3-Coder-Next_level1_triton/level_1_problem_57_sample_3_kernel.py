import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose2d_kernel(
    # Pointers to tensors
    x_ptr,  # Input: (batch_size, in_channels, H, W)
    w_ptr,  # Weight: (in_channels, out_channels, kH, kW)
    b_ptr,  # Bias: (out_channels,) or None
    out_ptr,  # Output: (batch_size, out_channels, H_out, W_out)
    # Tensor dimensions
    batch_size, in_channels, out_channels,
    height, width,
    kernel_size,
    stride, padding, output_padding,
    # Strides
    x_batch_stride, x_channel_stride, x_height_stride, x_width_stride,
    w_in_channel_stride, w_out_channel_stride, w_kh_stride, w_kw_stride,
    out_batch_stride, out_channel_stride, out_height_stride, out_width_stride,
    # Block sizes
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr, BLOCK_SIZE_KW: tl.constexpr,
):
    # Get output tensor coordinates
    out_batch = tl.program_id(0)
    out_channel = tl.program_id(1)
    out_h_block = tl.program_id(2)
    out_w_block = tl.program_id(3)
    
    # Compute output spatial coordinates
    out_h = out_h_block * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = out_w_block * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for output coordinates
    out_h_mask = out_h < height * stride + output_padding + (kernel_size - 1 - padding * 2)
    out_w_mask = out_w < width * stride + output_padding + (kernel_size - 1 - padding * 2)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Iterate over input channels and kernel dimensions
    for in_channel in range(in_channels):
        for kh_start in range(0, kernel_size, BLOCK_SIZE_KH):
            kh = kh_start + tl.arange(0, BLOCK_SIZE_KH)
            kh_mask = kh < kernel_size
            
            for kw_start in range(0, kernel_size, BLOCK_SIZE_KW):
                kw = kw_start + tl.arange(0, BLOCK_SIZE_KW)
                kw_mask = kw < kernel_size
                
                # Compute input coordinates from output coordinates
                # For transposed convolution: input_h = (out_h - (kernel_size - 1 - kh)) // stride
                # More precisely: input_h = (out_h - (kernel_size - 1 - kh) + stride - 1) // stride
                # But we need to handle the relationship carefully
                input_h = (out_h - (kernel_size - 1 - kh) + stride - 1) // stride
                input_w = (out_w - (kernel_size - 1 - kw) + stride - 1) // stride
                
                # Check if input coordinates are valid
                h_valid = (input_h >= 0) & (input_h < height)
                w_valid = (input_w >= 0) & (input_w < width)
                valid_mask = h_valid & w_valid
                
                # Get input values
                x_offsets = (out_batch * x_batch_stride + 
                            in_channel * x_channel_stride + 
                            input_h[:, None] * x_height_stride + 
                            input_w[None, :] * x_width_stride)
                
                # Load input with mask
                x_val = tl.load(x_ptr + x_offsets, 
                               mask=valid_mask[:, :, None, None].any(axis=2).squeeze(2), 
                               other=0.0)
                
                # Get weight values
                w_offsets = (in_channel * w_in_channel_stride + 
                            out_channel * w_out_channel_stride + 
                            (kh[:, None] if BLOCK_SIZE_KH > 1 else kh) * w_kh_stride + 
                            (kw[None, :] if BLOCK_SIZE_KW > 1 else kw) * w_kw_stride)
                
                w_val = tl.load(w_ptr + w_offsets, mask=kh_mask[:, None] & kw_mask[None, :], other=0.0)
                
                # Accumulate
                if BLOCK_SIZE_H == 1:
                    acc += tl.sum(x_val * w_val, axis=1)
                elif BLOCK_SIZE_W == 1:
                    acc += tl.sum(x_val * w_val, axis=0)
                else:
                    acc += tl.dot(x_val, w_val, trans_a=False, trans_b=False)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_channel)
        acc += bias
    
    # Store result
    out_offsets = (out_batch * out_batch_stride + 
                  out_channel * out_channel_stride + 
                  out_h[:, None] * out_height_stride + 
                  out_w[None, :] * out_width_stride)
    
    tl.store(out_ptr + out_offsets, acc.to(tl.float32), mask=out_h_mask[:, None] & out_w_mask[None, :])


def triton_conv_transpose2d(x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
    """
    Triton implementation of ConvTranspose2d for the specific case in the model.
    Assumes square inputs and kernels.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size_h, kernel_size_w = weight.shape
    
    # For simplicity, assume square inputs and kernels
    assert kernel_size_h == kernel_size_w, "Only square kernels supported"
    assert height == width, "Only square inputs supported"
    
    kernel_size = kernel_size_h
    
    # Calculate output dimensions
    out_height = (height - 1) * stride - 2 * padding + (kernel_size - 1) + output_padding + 1
    out_width = (width - 1) * stride - 2 * padding + (kernel_size - 1) + output_padding + 1
    
    # Create output tensor
    out = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Strides for input tensor
    x_batch_stride = x.stride(0)
    x_channel_stride = x.stride(1)
    x_height_stride = x.stride(2)
    x_width_stride = x.stride(3)
    
    # Strides for weight tensor
    w_in_channel_stride = weight.stride(0)
    w_out_channel_stride = weight.stride(1)
    w_kh_stride = weight.stride(2)
    w_kw_stride = weight.stride(3)
    
    # Strides for output tensor
    out_batch_stride = out.stride(0)
    out_channel_stride = out.stride(1)
    out_height_stride = out.stride(2)
    out_width_stride = out.stride(3)
    
    # Grid dimensions
    # batch_size, out_channels, output_height_blocks, output_width_blocks
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_KH = 16
    BLOCK_SIZE_KW = 16
    
    grid = (batch_size, out_channels, 
            (out_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
            (out_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        height, width,
        kernel_size,
        stride, padding, output_padding,
        x_batch_stride, x_channel_stride, x_height_stride, x_width_stride,
        w_in_channel_stride, w_out_channel_stride, w_kh_stride, w_kw_stride,
        out_batch_stride, out_channel_stride, out_height_stride, out_width_stride,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_KH=BLOCK_SIZE_KH,
        BLOCK_SIZE_KW=BLOCK_SIZE_KW,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for transposed 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, 
            output_padding=output_padding, 
            groups=groups, bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Use Triton kernel for transposed convolution instead of PyTorch's implementation.
        """
        # Extract parameters from the original layer
        weight = self.conv_transpose2d.weight
        bias = self.conv_transpose2d.bias
        
        # Call our custom Triton implementation
        return triton_conv_transpose2d(
            x, weight, bias,
            stride=self.conv_transpose2d.stride[0],
            padding=self.conv_transpose2d.padding[0],
            output_padding=self.conv_transpose2d.output_padding[0],
            groups=self.conv_transpose2d.groups
        )