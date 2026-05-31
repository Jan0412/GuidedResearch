import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, in_channels, out_channels, height, width,
    h_out, w_out, kernel_size, stride, padding, dilation, groups,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw,
    stride_o_n, stride_o_co, stride_o_h, stride_o_w,
    BLOCK_W: tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
):
    # Program IDs
    pid_nh = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_co = tl.program_id(2)

    # Compute n and h from pid_nh
    n = pid_nh // h_out
    h = pid_nh % h_out

    # Compute offsets for width and output channels
    w_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    co_offs = pid_co * BLOCK_C_OUT + tl.arange(0, BLOCK_C_OUT)

    # Masks for boundaries
    mask_w = w_offs < w_out
    mask_co = co_offs < out_channels

    # Initialize accumulator
    # Shape: (BLOCK_W, BLOCK_C_OUT)
    acc = tl.zeros([BLOCK_W, BLOCK_C_OUT], dtype=tl.float32)

    in_channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups

    # Loop over input channels and kernel spatial dimensions
    for ci in range(0, in_channels_per_group):
        for kh in range(0, kernel_size):
            for kw in range(0, kernel_size):
                # Calculate input coordinates
                h_in = h * stride + kh * dilation - padding
                w_in = w_offs * stride + kw * dilation - padding
                
                # Input mask: check if current pixel is within bounds
                mask_x = (h_in >= 0) & (h_in < height) & (w_in >= 0) & (w_in < width)
                
                # Load input values for the current (n, ci, h_in, w_in)
                # We need to handle groups: the input channel depends on which output group we are in
                cg = co_offs // out_channels_per_group
                x_ptr_val = x_ptr + n * stride_x_n + (cg * groups + ci) * stride_x_c + h_in * stride_x_h + w_in * stride_x_w
                x_vals = tl.load(x_ptr_val, mask=mask_x & mask_w, other=0.0) # (BLOCK_W,)

                # Load weights for the current (co, ci, kh, kw)
                w_ptr_val = w_ptr + co_offs * stride_w_co + ci * stride_w_ci + kh * stride_w_kh + kw * stride_w_kw
                w_vals = tl.load(w_ptr_val, mask=mask_co, other=0.0) # (BLOCK_C_OUT,)

                # Outer product accumulation: (BLOCK_W, 1) * (1, BLOCK_C_OUT)
                acc += x_vals[:, None] * w_vals[None, :]

    # Add bias
    if b_ptr is not None:
        bias = tl.load(b_ptr + co_offs, mask=mask_co, other=0.0)
        acc += bias[None, :]

    # Store result to output tensor
    for i in range(BLOCK_W):
        w_val = w_offs + i
        if w_val < w_out:
            out_ptr_val = out_ptr + n * stride_o_n + co_offs * stride_o_co + h * stride_o_h + w_val * stride_o_w
            tl.store(out_ptr_val, acc[i, :], mask=mask_co)

def triton_conv2d(x, weight, bias, stride=1, padding=0, dilation=1, groups=1):
    # Input shapes
    n, c_in, h, w = x.shape
    c_out, c_in_per_group, kh, kw = weight.shape
    
    # Calculate output dimensions
    h_out = (h + 2 * padding - dilation * (kh - 1) - 1) // stride + 1
    w_out = (w + 2 * padding - dilation * (kw - 1) - 1) // stride + 1
    
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    out = torch.empty((n, c_out, h_out, w_out), device=x.device, dtype=x.dtype)
    
    # Strides
    stride_x_n, stride_x_c, stride_x_h, stride_x_w = x.stride()
    stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw = weight.stride()
    stride_o_n, stride_o_co, stride_o_h, stride_o_w = out.stride()
    
    # Tuning parameters
    BLOCK_W = 16
    BLOCK_C_OUT = 16
    
    # Grid
    grid = (n * h_out, triton.cdiv(w_out, BLOCK_W), triton.cdiv(c_out, BLOCK_C_OUT))
    
    conv2d_kernel[grid](
        x, weight, bias, out,
        n, c_in, c_out, h, w,
        h_out, w_out, kh, stride, padding, dilation, groups,
        stride_x_n, stride_x_c, stride_x_h, stride_x_w,
        stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw,
        stride_o_n, stride_o_co, stride_o_h, stride_o_w,
        BLOCK_W=BLOCK_W,
        BLOCK_C_OUT=BLOCK_C_OUT,
    )
    
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Use nn.Conv2d to initialize weights and bias
        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), 
                              stride=stride, padding=padding, dilation=dilation, 
                              groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract parameters from the nn.Conv2d layer
        weight = self.conv.weight
        bias = self.conv.bias if self.conv.bias is not None else None
        
        # Call the Triton kernel
        return triton_conv2d(
            x, weight, bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation, 
            groups=self.groups
        )