import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv3d_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, in_channels, out_channels,
    in_d, in_h, in_w,
    out_d, out_h, out_w,
    k_d, k_h, k_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    groups,
    BLOCK_SIZE: tl.constexpr,
):
    # The grid is (batch_size, out_channels, out_d, out_h, out_w)
    # We map the program ID to the output coordinates
    pid = tl.program_id(0)
    
    # Decompose pid into coordinates
    # Grid dimensions: [batch_size * out_channels * out_d * out_h * out_w]
    # To simplify, we'll use a 1D grid and manually decompose
    # However, for better performance in Triton, we usually use multi-dim grids.
    # But since we are implementing a general 3D conv, we'll use the provided grid mapping.
    
    # Using 1D pid for simplicity in this implementation
    # pid = n * (out_channels * out_d * out_h * out_w) + 
    #       oc * (out_d * out_h * out_w) + 
    #       od * (out_h * out_w) + 
    #       oh * (out_w) + 
    #       ow
    
    ow = pid % out_w
    rem = pid // out_w
    oh = rem % out_h
    rem = rem // out_h
    od = rem % out_d
    rem = rem // out_d
    oc = rem % out_channels
    n = rem // out_channels

    # Input channels per group
    in_channels_per_group = in_channels // groups
    group_id = oc // in_channels_per_group
    
    # Accumulator for the convolution result
    acc = 0.0

    # Loop over input channels in the group
    for ic in range(in_channels_per_group):
        # Loop over the kernel dimensions
        for kd in range(k_d):
            for kh in range(k_h):
                for kw in range(k_w):
                    # Calculate input coordinates with stride, padding, and dilation
                    id_val = od * stride_d - pad_d + kd * dil_d
                    ih_val = oh * stride_h - pad_h + kh * dil_h
                    iw_val = ow * stride_w - pad_w + kw * dil_w

                    # Boundary check for padding
                    if (id_val >= 0 and id_val < in_d and 
                        ih_val >= 0 and ih_val < in_h and 
                        iw_val >= 0 and iw_val < in_w):
                        
                        # Input tensor indexing: [n, c, d, h, w]
                        # c = group_id * (in_channels // groups) + ic
                        c_idx = group_id * in_channels_per_group + ic
                        x_offset = (n * in_channels * in_d * in_h * in_w) + \
                                   (c_idx * in_d * in_h * in_w) + \
                                   (id_val * in_h * in_w) + \
                                   (ih_val * in_w) + \
                                   iw_val
                        
                        # Weight tensor indexing: [oc, ic_per_group, kd, kh, kw]
                        w_offset = (oc * in_channels_per_group * k_d * k_h * k_w) + \
                                   (ic * k_d * k_h * k_w) + \
                                   (kd * k_h * k_w) + \
                                   (kh * k_w) + \
                                   kw
                        
                        x_val = tl.load(x_ptr + x_offset)
                        w_val = tl.load(weight_ptr + w_offset)
                        acc += x_val * w_val

    # Add bias if applicable
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + oc)

    # Store result: [n, oc, od, oh, ow]
    out_offset = (n * out_channels * out_d * out_h * out_w) + \
                 (oc * out_d * out_h * out_w) + \
                 (od * out_h * out_w) + \
                 (oh * out_w) + \
                 ow
    tl.store(out_ptr + out_offset, acc)

def triton_conv3d(x, weight, bias, stride, padding, dilation, groups):
    # Input shapes
    n, in_c, in_d, in_h, in_w = x.shape
    out_c, in_c_per_group, k_d, k_h, k_w = weight.shape
    
    s_d, s_h, s_w = stride
    p_d, p_h, p_w = padding
    d_d, d_h, d_w = dilation

    # Calculate output dimensions
    out_d = (in_d + 2 * p_d - d_d * (k_d - 1) - 1) // s_d + 1
    out_h = (in_h + 2 * p_h - d_h * (k_h - 1) - 1) // s_h + 1
    out_w = (in_w + 2 * p_w - d_w * (k_w - 1) - 1) // s_w + 1

    out = torch.empty((n, out_c, out_d, out_h, out_w), device=x.device, dtype=x.dtype)
    
    # Flatten the output dimensions for the grid
    total_elements = n * out_c * out_d * out_h * out_w
    grid = (total_elements,)

    conv3d_kernel[grid](
        x, weight, bias, out,
        n, in_c, out_c,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        k_d, k_h, k_w,
        s_d, s_h, s_w,
        p_d, p_h, p_w,
        d_d, d_h, d_w,
        groups,
        BLOCK_SIZE=1
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights and bias manually to match nn.Conv3d
        weight_shape = (out_channels, in_channels // groups, *kernel_size)
        self.weight = nn.Parameter(torch.randn(weight_shape))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are contiguous and on GPU
        x = x.contiguous().cuda().float()
        weight = self.weight.contiguous().cuda().float()
        bias = self.bias.contiguous().cuda().float() if self.bias is not None else None
        
        return triton_conv3d(
            x, weight, bias, 
            self.stride, self.padding, self.dilation, self.groups
        )