import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16, 'BLOCK_OC': 32, 'BLOCK_IC': 4}, num_warps=4),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 32, 'BLOCK_OC': 32, 'BLOCK_IC': 8}, num_warps=4),
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 8, 'BLOCK_OC': 16, 'BLOCK_IC': 4}, num_warps=8),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16, 'BLOCK_OC': 16, 'BLOCK_IC': 8}, num_warps=8),
    ],
    key=['batch', 'out_h', 'out_w', 'out_c', 'in_c'],
)
@triton.jit
def conv_transpose2d_kernel(
    # Pointers to tensors
    x_ptr,  # (batch, in_c, in_h, in_w)
    w_ptr,  # (in_c, out_c//groups, k_h, k_w)
    b_ptr,  # (out_c,) or None
    out_ptr,  # (batch, out_c, out_h, out_w)
    # Sizes
    batch, in_c, out_c, groups,
    in_h, in_w,
    out_h, out_w,
    k_h, k_w,
    stride_h, stride_w,
    padding_h, padding_w,
    output_pad_h, output_pad_w,
    dilation_h, dilation_w,
    # Strides
    stride_x_bc, stride_x_c, stride_x_h, stride_x_w,
    stride_w_ic, stride_w_oc, stride_w_kh, stride_w_kw,
    stride_b_c,
    stride_out_bc, stride_out_c, stride_out_h, stride_out_w,
    # Meta-parameters
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_IC: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Compute output coordinates
    out_h_idx = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w_idx = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    out_c_idx = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    
    # Create masks for valid indices
    mask_h = out_h_idx < out_h
    mask_w = out_w_idx < out_w
    mask_c = out_c_idx < out_c
    mask_hw = mask_h[:, None] & mask_w[None, :]
    mask_chw = mask_c[:, None, None] & mask_hw
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_OC, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Loop over input channels in groups
    num_ic_groups = in_c // groups
    group_offset = pid_oc * BLOCK_OC // (out_c // groups) * num_ic_groups
    
    # Iterate over input channels in blocks
    for ic_start in range(0, num_ic_groups, BLOCK_IC):
        ic_idx = ic_start + tl.arange(0, BLOCK_IC)
        mask_ic = ic_idx < num_ic_groups
        mask_ic_c = mask_ic[:, None] & mask_c[:, None]
        mask_ic_chw = mask_ic[:, None, None, None] & mask_c[:, None, None, None] & mask_hw[None, :, :, :]
        
        # Compute input channel indices for this group
        ic_global = ic_start + tl.arange(0, BLOCK_IC) + group_offset
        
        # Load input values (batch, ic, in_h, in_w)
        in_h_start = (out_h_idx[:, None] - (k_h - 1) * dilation_h + padding_h) // stride_h
        in_w_start = (out_w_idx[None, :] - (k_w - 1) * dilation_w + padding_w) // stride_w
        
        # Compute valid positions
        valid_h = (in_h_start >= 0) & (in_h_start < in_h) & (mask_h[:, None])
        valid_w = (in_w_start >= 0) & (in_w_start < in_w) & (mask_w[None, :])
        valid_hw = valid_h & valid_w
        
        # Compute relative positions within kernel
        k_h_pos = (out_h_idx[:, None] - in_h_start * stride_h - padding_h) // dilation_h
        k_w_pos = (out_w_idx[None, :] - in_w_start * stride_w - padding_w) // dilation_w
        
        # Create masks for kernel positions
        mask_kh = (k_h_pos >= 0) & (k_h_pos < k_h)
        mask_kw = (k_w_pos >= 0) & (k_w_pos < k_w)
        mask_khw = mask_kh & mask_kw
        
        # Compute final valid mask combining all constraints
        final_mask = mask_ic_chw & (mask_khw[None, :, :, :] & mask_hw[None, :, :, :])
        
        # Load input tensor
        in_h_idx = in_h_start * stride_h + padding_h
        in_w_idx = in_w_start * stride_w + padding_w
        
        # Compute actual input positions after accounting for dilation and stride
        x_offsets = (
            pid_b * stride_x_bc +
            (ic_global[:, None, None, None] * stride_x_c) +
            ((out_h_idx[:, None] - (k_h - 1) * dilation_h + padding_h) // stride_h)[None, :, :, None] * stride_x_h +
            ((out_w_idx[None, :] - (k_w - 1) * dilation_w + padding_w) // stride_w)[None, :, :, None] * stride_x_w
        )
        
        # Adjust for actual kernel positions
        x_offsets = (
            pid_b * stride_x_bc +
            (ic_global[:, None, None, None] * stride_x_c) +
            ((out_h_idx[:, None] - k_h_pos * dilation_h - padding_h) // stride_h)[None, :, :, None] * stride_x_h +
            ((out_w_idx[None, :] - k_w_pos * dilation_w - padding_w) // stride_w)[None, :, :, None] * stride_x_w
        )
        
        # Load input with proper masks
        x_val = tl.load(
            x_ptr + x_offsets,
            mask=final_mask,
            other=0.0
        )
        
        # Load weight tensor
        # Weight shape: (in_c, out_c//groups, k_h, k_w)
        w_offsets = (
            (ic_global[:, None, None, None] * stride_w_ic) +
            ((pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)[:, None, None, None]) * stride_w_oc) +
            (k_h_pos[None, :, :, :] * stride_w_kh) +
            (k_w_pos[None, :, :, :] * stride_w_kw)
        )
        
        w_val = tl.load(
            w_ptr + w_offsets,
            mask=mask_ic_c[:, :, None, None] & mask_khw[None, :, :, :],
            other=0.0
        )
        
        # Compute accumulation: x * w
        acc += tl.sum(x_val * w_val, axis=0)
    
    # Add bias if present
    if b_ptr is not None:
        b = tl.load(b_ptr + out_c_idx, mask=mask_c)
        acc += b[:, None, None]
    
    # Store result
    out_offsets = (
        pid_b * stride_out_bc +
        (out_c_idx[:, None, None] * stride_out_c) +
        (out_h_idx[None, :, None] * stride_out_h) +
        (out_w_idx[None, None, :] * stride_out_w)
    )
    
    tl.store(
        out_ptr + out_offsets,
        acc.to(x_ptr.dtype.element_ty),
        mask=mask_chw
    )


def triton_conv_transpose2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: tuple = (1, 1),
    padding: tuple = (0, 0),
    output_padding: tuple = (0, 0),
    dilation: tuple = (1, 1),
    groups: int = 1,
) -> torch.Tensor:
    """
    Triton implementation of ConvTranspose2d for FP32 tensors.
    """
    # Ensure inputs are contiguous and on CUDA
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch, in_c, in_h, in_w = x.shape
    _, out_c, k_h, k_w = weight.shape
    
    # Compute output dimensions
    out_h = (in_h - 1) * stride[0] - 2 * padding[0] + dilation[0] * (k_h - 1) + output_padding[0] + 1
    out_w = (in_w - 1) * stride[1] - 2 * padding[1] + dilation[1] * (k_w - 1) + output_padding[1] + 1
    
    # Allocate output tensor
    out = torch.empty(batch, out_c, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Compute strides
    stride_x = x.stride()
    stride_w = weight.stride()
    stride_b = bias.stride() if bias is not None else (0,)
    stride_out = out.stride()
    
    # Grid dimensions
    grid = (
        batch,
        triton.cdiv(out_c, 32),  # num OC blocks
        triton.cdiv(out_h, 16),  # num H blocks
        triton.cdiv(out_w, 16),  # num W blocks
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias, out,
        batch, in_c, out_c, groups,
        in_h, in_w,
        out_h, out_w,
        k_h, k_w,
        stride[0], stride[1],
        padding[0], padding[1],
        output_padding[0], output_padding[1],
        dilation[0], dilation[1],
        stride_x[0], stride_x[1], stride_x[2], stride_x[3],
        stride_w[0], stride_w[1], stride_w[2], stride_w[3],
        stride_b[0] if bias is not None else 0,
        stride_out[0], stride_out[1], stride_out[2], stride_out[3],
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for ConvTranspose2d.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), 
                 output_padding: tuple = (0, 0), dilation: tuple = (1, 1), 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters for reconstruction
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        
        # Create the weight and bias tensors (same as original ConvTranspose2d)
        # Note: We'll manually create these to match nn.ConvTranspose2d's initialization
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights (same as nn.ConvTranspose2d default initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using custom Triton kernel for ConvTranspose2d.
        """
        return triton_conv_transpose2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups
        )


import math