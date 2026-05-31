import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose2d_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, in_channels, out_channels,
    h_in, w_in, h_out, w_out,
    kernel_size, stride, padding, groups,
    stride_h, stride_w, # Strides for output tensor
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_ic, w_stride_oc, w_stride_kh, w_stride_kw,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Output coordinates
    oh_start = pid_h * BLOCK_H
    ow_start = pid_w * BLOCK_W
    
    # Output channel and batch
    oc = pid_oc
    b = pid_b

    # Group logic for ConvTranspose2d
    # weight shape: (in_channels, out_channels // groups, kH, kW)
    oc_per_group = out_channels // groups
    ic_per_group = in_channels // groups
    group_id = oc // oc_per_group
    oc_in_group = oc % oc_per_group

    # Initialize accumulator
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Loop over input channels in the same group
    ic_start = group_id * ic_per_group
    for ic in range(ic_start, ic_start + ic_per_group):
        # Loop over kernel spatial dimensions
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Calculate corresponding input coordinates
                # oh = ih * stride - padding + kh  =>  ih = (oh + padding - kh) / stride
                # We use a range of oh (oh_start to oh_start + BLOCK_H)
                oh_offsets = oh_start + tl.arange(0, BLOCK_H)
                ow_offsets = ow_start + tl.arange(0, BLOCK_W)
                
                ih = (oh_offsets + padding - kh) / stride
                iw = (ow_offsets + padding - kw) / stride
                
                # Check if coordinates are valid (divisible by stride and within bounds)
                mask_h = (oh_offsets + padding - kh) % stride == 0
                mask_w = (ow_offsets + padding - kw) % stride == 0
                mask_bounds_h = (ih >= 0) & (ih < h_in)
                mask_bounds_w = (iw >= 0) & (iw < w_in)
                
                mask = mask_h & mask_w & mask_bounds_h & mask_bounds_w
                
                # Load input value: x[b, ic, ih, iw]
                # Convert ih, iw to integers for indexing
                ih_int = tl.cast(ih, tl.int32)
                iw_int = tl.cast(iw, tl.int32)
                
                x_offset = (b * x_stride_b + 
                            ic * x_stride_c + 
                            ih_int * x_stride_h + 
                            iw_int * x_stride_w)
                
                # We need to handle the 2D grid of (oh, ow)
                # To simplify, we can iterate through the block using offsets
                # but Triton prefers vectorized loads. 
                # Let's use a more efficient approach for the inner block:
                # Since we are computing a BLOCK_H x BLOCK_W tile:
                
                # Re-calculating offsets for the 2D tile
                oh_tile = oh_start + tl.arange(0, BLOCK_H)[:, None]
                ow_tile = ow_start + tl.arange(0, BLOCK_W)[None, :]
                
                ih_tile = (oh_tile + padding - kh) / stride
                iw_tile = (ow_tile + padding - kw) / stride
                
                mask_tile = ((oh_tile + padding - kh) % stride == 0) & \
                             ((ow_tile + padding - kw) % stride == 0) & \
                             (ih_tile >= 0) & (ih_tile < h_in) & \
                             (iw_tile >= 0) & (iw_tile < w_in)
                
                ih_tile_int = tl.cast(ih_tile, tl.int32)
                iw_tile_int = tl.cast(iw_tile, tl.int32)
                
                x_ptr_tile = x_ptr + (b * x_stride_b + ic * x_stride_c + 
                                      ih_tile_int * x_stride_h + iw_tile_int * x_stride_w)
                
                # Load weight: weight[ic, oc_in_group, kh, kw]
                w_ptr_val = weight_ptr + (ic * w_stride_ic + 
                                          oc_in_group * w_stride_oc + 
                                          kh * w_stride_kh + 
                                          kw * w_stride_kw)
                weight_val = tl.load(w_ptr_val)
                
                # Load input and accumulate
                x_val = tl.load(x_ptr_tile, mask=mask_tile, other=0.0)
                acc += x_val * weight_val

    # Add bias
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + oc)
        acc += bias_val

    # Store output
    oh_tile = oh_start + tl.arange(0, BLOCK_H)[:, None]
    ow_tile = ow_start + tl.arange(0, BLOCK_W)[None, :]
    out_mask = (oh_tile < h_out) & (ow_tile < w_out)
    
    out_offset = (b * (out_channels * h_out * w_out) + 
                  oc * (h_out * w_out) + 
                  oh_tile * w_out + 
                  ow_tile)
    
    tl.store(out_ptr + out_offset, acc, mask=out_mask)


def triton_conv_transpose2d(x, weight, bias, stride=1, padding=0, output_padding=0, groups=1):
    # Input shapes
    batch_size, in_channels, h_in, w_in = x.shape
    # weight shape: (in_channels, out_channels // groups, kH, kW)
    in_channels_w, oc_per_group, k_h, k_w = weight.shape
    out_channels = oc_per_group * groups

    # Output dimensions
    h_out = (h_in - 1) * stride - 2 * padding + k_h + output_padding
    w_out = (w_in - 1) * stride - 2 * padding + k_w + output_padding

    out = torch.empty((batch_size, out_channels, h_out, w_out), device=x.device, dtype=x.dtype)

    # Strides
    x_stride_b, x_stride_c, x_stride_h, x_stride_w = x.stride()
    w_stride_ic, w_stride_oc, w_stride_kh, w_stride_kw = weight.stride()

    BLOCK_H = 16
    BLOCK_W = 16

    grid = (batch_size, out_channels, 
            (h_out + BLOCK_H - 1) // BLOCK_H, 
            (w_out + BLOCK_W - 1) // BLOCK_W)

    conv_transpose2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        h_in, w_in, h_out, w_out,
        k_h, stride, padding, groups,
        0, 0, # dummy stride_h, stride_w
        x_stride_b, x_stride_c, x_stride_h, x_stride_w,
        w_stride_ic, w_stride_oc, w_stride_kh, w_stride_kw,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Use standard ConvTranspose2d for parameter management
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding, 
            groups=groups, bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are contiguous and on GPU
        x = x.contiguous()
        weight = self.conv_transpose2d.weight.contiguous()
        bias = self.conv_transpose2d.bias.contiguous() if self.conv_transpose2d.bias is not None else None
        
        return triton_conv_transpose2d(
            x, weight, bias, 
            stride=self.stride, 
            padding=self.padding, 
            output_padding=self.output_padding, 
            groups=self.groups
        )