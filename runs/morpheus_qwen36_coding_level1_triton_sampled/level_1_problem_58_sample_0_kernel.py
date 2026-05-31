import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    in_channels, out_channels, groups,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    output_padding_d, output_padding_h, output_padding_w,
    depth_in, height_in, width_in,
    depth_out, height_out, width_out,
    batch_size,
    BLOCK_SIZE_C: tl.constexpr
):
    # Flatten output index
    out_idx = tl.program_id(0)
    
    # Decode output coordinates
    # Layout: (B, C_out, D_out, H_out, W_out)
    stride_c0 = depth_out * height_out * width_out
    stride_d0 = height_out * width_out
    stride_h0 = width_out
    
    b = out_idx // (out_channels * stride_c0)
    rem = out_idx % (out_channels * stride_c0)
    c_out = rem // stride_c0
    rem = rem % stride_c0
    z = rem // stride_d0
    y = rem // stride_h0
    x = rem % stride_h0
    
    # Compute source coordinates for each kernel element
    # z_src = z * stride_d - padding_d + dz
    # Similarly for y and x
    
    acc = 0.0
    
    # Loop over input channels with grouping
    # For grouped transpose conv, c_out maps to group c_out // groups
    # and only c_in where c_in % groups == c_out % groups contribute
    # Weight shape: (C_out, C_in // groups, K_d, K_h, K_w)
    
    c_in_group = c_out // groups
    c_in_offset = c_out % groups
    
    # Iterate over the relevant input channels
    # We can loop over c_in_group and compute c_in directly
    # This avoids checking modulo every time
    
    # Load weight base pointer for this c_out and c_in_group
    # Weight strides: assume contiguous in C_out, then C_in//groups, then K dims
    # We need to compute offsets dynamically or assume layout
    # PyTorch ConvTranspose3d weight layout: (C_out, C_in//groups, K_d, K_h, K_w)
    # Strides are typically contiguous for last dims
    
    w_stride_c0 = kernel_d * kernel_h * kernel_w
    w_stride_c1 = kernel_w  # Assuming K_w is smallest, but generally stride is product of later dims
    # Actually, PyTorch layout is (C_out, C_in//groups, K_d, K_h, K_w)
    # Stride for C_in//groups is K_d * K_h * K_w
    # Stride for K_d is K_h * K_w
    # Stride for K_h is K_w
    # Stride for K_w is 1
    
    w_stride_c1 = kernel_d * kernel_h * kernel_w
    w_stride_d = kernel_h * kernel_w
    w_stride_h = kernel_w
    
    for c_in_g in range(0, in_channels // groups, BLOCK_SIZE_C):
        # c_in_g is the index in the weight tensor's C_in//groups dim
        # Corresponding c_in values: c_in = c_in_g * groups + c_in_offset
        # But we can just compute pointer offset for c_in_g
        
        w_ptr_base = w_ptr + c_out * w_stride_c0 + (c_in_g + c_in_offset) * w_stride_c1
        
        # Input pointer base for batch b and group
        # Input layout: (B, C_in, D_in, H_in, W_in)
        # Stride for C_in is D_in * H_in * W_in
        
        x_stride_c = depth_in * height_in * width_in
        x_ptr_base = x_ptr + b * (in_channels * x_stride_c) + (c_in_g * groups + c_in_offset) * x_stride_c
        
        # Loop over kernel dimensions
        # To optimize, we can unroll or use vectorized loads if possible
        # Here we iterate scalar for correctness and simplicity
        # In a real scenario, tiling kernel dims might be beneficial
        
        for dz in range(kernel_d):
            z_src = z * stride_d - padding_d + dz
            z_mask = (z_src >= 0) & (z_src < depth_in)
            
            if not z_mask:
                continue
                
            w_offset_d = dz * w_stride_d
            
            for dy in range(kernel_h):
                y_src = y * stride_h - padding_h + dy
                y_mask = (y_src >= 0) & (y_src < height_in)
                
                if not y_mask:
                    continue
                    
                w_offset_h = dy * w_stride_h
                
                for dx in range(kernel_w):
                    x_src = x * stride_w - padding_w + dx
                    x_mask = (x_src >= 0) & (x_src < width_in)
                    
                    if not x_mask:
                        continue
                        
                    w_offset_w = dx * 1  # K_w stride is 1
                    
                    # Load weight
                    w_val = tl.load(w_ptr_base + w_offset_d + w_offset_h + w_offset_w)
                    
                    # Load input
                    x_offset = z_src * x_stride_c + y_src * (height_in * width_in) + x_src * width_in
                    # x_ptr_base already includes c_in offset, so we just add spatial offsets
                    # Wait, x_ptr_base calculation above:
                    # x_ptr_base = x_ptr + b * (in_channels * x_stride_c) + (c_in_g * groups + c_in_offset) * x_stride_c
                    # This is correct for the specific c_in
                    
                    x_val = tl.load(x_ptr_base + z_src * x_stride_c + y_src * (height_in * width_in) + x_src * width_in, 
                                    mask=False, other=0.0) # Mask not needed here as we checked bounds
                    
                    acc += w_val * x_val

    # Add bias
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + c_out)
        
    # Store output
    out_offset = b * (out_channels * stride_c0) + c_out * stride_c0 + z * stride_d0 + y * stride_h0 + x
    tl.store(out_ptr + out_offset, acc)


def triton_conv_transpose3d(x, w, bias, stride, padding, output_padding, groups):
    assert x.is_cuda and w.is_cuda
    x = x.contiguous()
    w = w.contiguous()
    if bias is not None:
        bias = bias.contiguous()
        
    batch_size, in_channels, depth_in, height_in, width_in = x.shape
    out_channels, _, kernel_d, kernel_h, kernel_w = w.shape
    
    # Compute output dimensions
    depth_out = (depth_in - 1) * stride[0] - 2 * padding[0] + kernel_d + output_padding[0]
    height_out = (height_in - 1) * stride[1] - 2 * padding[1] + kernel_h + output_padding[1]
    width_out = (width_in - 1) * stride[2] - 2 * padding[2] + kernel_w + output_padding[2]
    
    out = torch.empty(batch_size, out_channels, depth_out, height_out, width_out, device=x.device, dtype=x.dtype)
    
    num_elements = batch_size * out_channels * depth_out * height_out * width_out
    BLOCK_SIZE_C = 1 # Tiling over input channels might be complex due to grouping and scattering
    
    grid = (num_elements,)
    
    conv_transpose3d_kernel[grid](
        x, w, bias, out,
        in_channels, out_channels, groups,
        kernel_d, kernel_h, kernel_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        depth_in, height_in, width_in,
        depth_out, height_out, width_out,
        batch_size,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x, 
            self.conv_transpose3d.weight, 
            self.conv_transpose3d.bias, 
            self.conv_transpose3d.stride, 
            self.conv_transpose3d.padding, 
            self.conv_transpose3d.output_padding, 
            self.conv_transpose3d.groups
        )