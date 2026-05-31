import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, Cin, H, W, Cout, Kh, Kw,
    Stride, Padding, Dilation, Groups,
    Ho, Wo,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
):
    # Program IDs
    pid_b_co = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Resolve batch index and output channel index
    b = pid_b_co // Cout
    co = pid_b_co % Cout

    # Output spatial offsets
    ho_start = pid_h * BLOCK_SIZE_H
    wo_start = pid_w * BLOCK_SIZE_W
    
    ho_range = ho_start + tl.arange(0, BLOCK_SIZE_H)
    wo_range = wo_start + tl.arange(0, BLOCK_SIZE_W)
    
    # Mask for output boundaries
    ho_mask = ho_range < Ho
    wo_mask = wo_range < Wo
    
    # Group logic
    out_channels_per_group = Cout // Groups
    in_channels_per_group = Cin // Groups
    group_id = co // out_channels_per_group
    ci_start = group_id * in_channels_per_group

    # Accumulator for the output block
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # Loop over kernel dimensions and input channels
    for kh in range(0, Kh):
        for kw in range(0, Kw):
            # Calculate input spatial indices for the current kernel element
            h_idx = ho_range[:, None] * Stride + kh * Dilation - Padding
            w_idx = wo_range[None, :] * Stride + kw * Dilation - Padding
            
            # Mask for input boundaries (padding)
            h_mask = (h_idx >= 0) & (h_idx < H)
            w_mask = (w_idx >= 0) & (w_idx < W)
            in_mask = h_mask & w_mask
            
            # Loop over input channels for this group
            for ci_local in range(0, in_channels_per_group):
                # Weight offset: [co, ci_local, kh, kw]
                # w shape: (Cout, Cin//Groups, Kh, Kw)
                w_offset = co * (in_channels_per_group * Kh * Kw) + \
                           ci_local * (Kh * Kw) + \
                           kh * Kw + kw
                w_val = tl.load(w_ptr + w_offset)

                # Input offset: [b, ci_start + ci_local, h_idx, w_idx]
                # x shape: (B, Cin, H, W)
                x_base_offset = b * (Cin * H * W) + \
                                (ci_start + ci_local) * (H * W)
                x_offset = x_base_offset + h_idx * W + w_idx
                
                # Load input with padding mask
                x_val = tl.load(x_ptr + x_offset, mask=in_mask, other=0.0)
                acc += x_val * w_val

    # Add bias if it exists
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + co)
        acc += bias_val

    # Store result with output boundary masks
    out_offset = b * (Cout * Ho * Wo) + \
                 co * (Ho * Wo) + \
                 ho_range[:, None] * Wo + \
                 wo_range[None, :]
    
    # Combine masks
    final_mask = ho_mask[:, None] & wo_mask[None, :]
    tl.store(out_ptr + out_offset, acc, mask=final_mask)


def triton_conv2d(x, weight, bias, stride=1, padding=0, dilation=1, groups=1):
    # Input shapes
    B, Cin, H, W = x.shape
    Cout, Cin_per_group, Kh, Kw = weight.shape
    
    # Calculate output dimensions
    Ho = (H + 2 * padding - dilation * (Kh - 1) - 1) // stride + 1
    Wo = (W + 2 * padding - dilation * (Kw - 1) - 1) // stride + 1
    
    # Ensure contiguous tensors
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    out = torch.empty((B, Cout, Ho, Wo), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE_H = 32
    BLOCK_SIZE_W = 32
    
    grid = (B * Cout, triton.cdiv(Ho, BLOCK_SIZE_H), triton.cdiv(Wo, BLOCK_SIZE_W))
    
    conv2d_kernel[grid](
        x, weight, bias, out,
        B, Cin, H, W, Cout, Kh, Kw,
        stride, padding, dilation, groups,
        Ho, Wo,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized 2D convolution operation using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Use nn.Conv2d to handle parameter initialization and storage
        self.conv_params = nn.Conv2d(
            in_channels, out_channels, (kernel_size, kernel_size), 
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract weights and bias from the Conv2d module
        weight = self.conv_params.weight
        bias = self.conv_params.bias if self.conv_params.bias is not None else None
        
        # Call the Triton-optimized convolution
        return triton_conv2d(
            x, weight, bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation, 
            groups=self.groups
        )