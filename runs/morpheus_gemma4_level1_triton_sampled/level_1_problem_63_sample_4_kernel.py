import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, in_channels, out_channels,
    height, width, 
    out_height, out_width,
    kernel_size, stride, padding, dilation, groups,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_oh = tl.program_id(2)
    pid_ow = tl.program_id(3)

    # Output offsets for the current tile
    oh_offsets = pid_oh * BLOCK_H + tl.arange(0, BLOCK_H)
    ow_offsets = pid_ow * BLOCK_W + tl.arange(0, BLOCK_W)

    # Mask for output boundaries
    mask_oh = oh_offsets < out_height
    mask_ow = ow_offsets < out_width

    # Weight offset base for this output channel
    # Weight shape: (out_channels, in_channels // groups, kh, kw)
    in_channels_per_group = in_channels // groups
    w_base = w_ptr + pid_oc * (in_channels_per_group * kernel_size * kernel_size)
    
    # Input channel starting index for this output channel
    ic_start = (pid_oc // (out_channels // groups)) * in_channels_per_group
    
    # Accumulator for the output tile
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Loop over input channels in the group and the kernel window
    for ic in range(0, in_channels_per_group):
        curr_ic = ic_start + ic
        for kh in range(0, kernel_size):
            for kw in range(0, kernel_size):
                # Calculate input coordinates based on stride, padding, and dilation
                ih = oh_offsets * stride - padding + kh * dilation
                iw = ow_offsets * stride - padding + kw * dilation
                
                # Mask for input boundaries (padding)
                mask_ih = (ih >= 0) & (ih < height)
                mask_iw = (iw >= 0) & (iw < width)
                
                # Final mask for the element in the tile
                mask = mask_oh[:, None] & mask_ow[None, :] & mask_ih[:, None] & mask_iw[None, :]
                
                # Safe offsets to prevent out-of-bounds memory access
                ih_safe = tl.where(mask_ih, ih, 0)
                iw_safe = tl.where(mask_iw, iw, 0)
                
                # Load input: x[batch, curr_ic, ih, iw]
                x_offset = (pid_batch * in_channels * height * width + 
                            curr_ic * height * width + 
                            ih_safe[:, None] * width + 
                            iw_safe[None, :])
                x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
                
                # Load weight: w[pid_oc, ic, kh, kw]
                w_val = tl.load(w_base + ic * kernel_size * kernel_size + kh * kernel_size + kw)
                
                acc += x_val * w_val

    # Add bias if provided
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + pid_oc)
        acc += bias_val

    # Store the result in the output tensor
    # out_ptr shape: (batch_size, out_channels, out_height, out_width)
    out_offset = (pid_batch * out_channels * out_height * out_width + 
                  pid_oc * out_height * out_width + 
                  oh_offsets[:, None] * out_width + 
                  ow_offsets[None, :])
    
    tl.store(out_ptr + out_offset, acc, mask=mask_oh[:, None] & mask_ow[None, :])

def triton_conv2d(x, weight, bias, stride, padding, dilation, groups):
    # Ensure tensors are contiguous and on GPU
    x = x.contiguous().float()
    weight = weight.contiguous().float()
    if bias is not None:
        bias = bias.contiguous().float()

    batch_size, in_channels, height, width = x.shape
    out_channels, in_channels_per_group, kh, kw = weight.shape
    
    # Calculate output dimensions
    out_height = (height + 2 * padding - dilation * (kh - 1) - 1) // stride + 1
    out_width = (width + 2 * padding - dilation * (kw - 1) - 1) // stride + 1
    
    out = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=torch.float32)
    
    BLOCK_H = 16
    BLOCK_W = 16
    
    grid = (
        batch_size, 
        out_channels, 
        (out_height + BLOCK_H - 1) // BLOCK_H, 
        (out_width + BLOCK_W - 1) // BLOCK_W
    )
    
    conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        height, width, 
        out_height, out_width,
        kh, stride, padding, dilation, groups,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
    )
    
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We use nn.Conv2d to manage weights and bias parameters
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), 
                                stride=stride, padding=padding, dilation=dilation, 
                                groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract parameters from the Conv2d layer
        weight = self.conv2d.weight
        bias = self.conv2d.bias
        stride = self.conv2d.stride[0]
        padding = self.conv2d.padding[0]
        dilation = self.conv2d.dilation[0]
        groups = self.conv2d.groups
        
        # Call the custom Triton implementation
        return triton_conv2d(x, weight, bias, stride, padding, dilation, groups)