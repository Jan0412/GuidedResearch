import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Input tensor
    w_ptr,  # Weight tensor
    b_ptr,  # Bias tensor (optional)
    out_ptr,  # Output tensor
    N,  # Batch size
    C_in,  # Input channels
    H_in,  # Input height
    W_in,  # Input width
    C_out,  # Output channels
    H_out,  # Output height
    W_out,  # Output width
    K_h,  # Kernel height
    K_w,  # Kernel width
    stride_h,  # Stride height
    stride_w,  # Stride width
    pad_h,  # Padding height
    pad_w,  # Padding width
    output_pad_h,  # Output padding height
    output_pad_w,  # Output padding width
    dil_h,  # Dilation height
    dil_w,  # Dilation width
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_CIN: tl.constexpr,
    BLOCK_SIZE_COUT: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
    BLOCK_SIZE_OUT_H: tl.constexpr,
    BLOCK_SIZE_OUT_W: tl.constexpr,
):
    # Program IDs for output tensor
    pid_n = tl.program_id(0)  # batch index
    pid_cout = tl.program_id(1)  # output channel index
    pid_h = tl.program_id(2)  # output height block index
    pid_w = tl.program_id(3)  # output width block index
    
    # Calculate output tensor offsets
    out_offsets = pid_n * (C_out * H_out * W_out) + pid_cout * (H_out * W_out) + \
                  pid_h * BLOCK_SIZE_OUT_H * W_out + pid_w * BLOCK_SIZE_OUT_W
    
    # Create output tile mask
    out_h_range = pid_h * BLOCK_SIZE_OUT_H + tl.arange(0, BLOCK_SIZE_OUT_H)
    out_w_range = pid_w * BLOCK_SIZE_OUT_W + tl.arange(0, BLOCK_SIZE_OUT_W)
    out_h_mask = out_h_range < H_out
    out_w_mask = out_w_range < W_out
    out_mask = out_h_mask[:, None] * out_w_mask[None, :]
    
    # Initialize accumulator for output
    out accumulator = tl.zeros((BLOCK_SIZE_OUT_H, BLOCK_SIZE_OUT_W), tl.float32)
    
    # Iterate over input channels and kernel dimensions
    for c_in in range(C_in):
        for kh in range(K_h):
            for kw in range(K_w):
                # Calculate corresponding input position
                in_h = pid_h * BLOCK_SIZE_OUT_H + out_h_range
                in_w = pid_w * BLOCK_SIZE_OUT_W + out_w_range
                
                # Calculate input indices after accounting for stride, padding, and dilation
                in_h_idx = (in_h - pid_h * BLOCK_SIZE_OUT_H * stride_h - pad_h + kh * dil_h) // stride_h
                in_w_idx = (in_w - pid_w * BLOCK_SIZE_OUT_W * stride_w + pad_w + kw * dil_w) // stride_w
                
                # Check if input indices are valid
                valid_input = (in_h_idx >= 0) & (in_h_idx < H_in) & \
                              (in_w_idx >= 0) & (in_w_idx < W_in)
                
                # Calculate input pointer offset
                x_offset = pid_n * (C_in * H_in * W_in) + c_in * (H_in * W_in) + \
                          in_h_idx * W_in + in_w_idx
                
                # Load input and weight values
                x_val = tl.load(x_ptr + x_offset, mask=valid_input, other=0.0)
                w_val = tl.load(w_ptr + c_in * (K_h * K_w * C_out) + kh * (K_w * C_out) + kw * C_out + pid_cout)
                
                # Accumulate result
                accumulator += tl.where(valid_input, x_val * w_val, 0.0)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_cout)
        accumulator += bias
    
    # Store result
    tl.store(out_ptr + out_offsets, accumulator.to(x_ptr.dtype.element_ty), mask=out_mask)


def triton_conv_transpose2d(x, weight, bias=None, stride=(1, 1), padding=(0, 0), 
                           output_padding=(0, 0), dilation=(1, 1), groups=1):
    """
    Triton implementation of ConvTranspose2d
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract parameters
    N, C_in, H_in, W_in = x.shape
    C_out, C_in_group, K_h, K_w = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    output_pad_h, output_pad_w = output_padding
    dil_h, dil_w = dilation
    
    # Calculate output dimensions
    H_out = (H_in - 1) * stride_h - 2 * pad_h + dil_h * (K_h - 1) + output_pad_h + 1
    W_out = (W_in - 1) * stride_w - 2 * pad_w + dil_w * (K_w - 1) + output_pad_w + 1
    
    # Allocate output tensor
    out = torch.empty(N, C_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Define block sizes (tunable parameters for optimization)
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_COUT = 16
    BLOCK_SIZE_OUT_H = 8
    BLOCK_SIZE_OUT_W = 8
    
    # Calculate grid dimensions
    grid = (N, C_out, 
            triton.cdiv(H_out, BLOCK_SIZE_OUT_H), 
            triton.cdiv(W_out, BLOCK_SIZE_OUT_W))
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias, out,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        stride_h, stride_w,
        pad_h, pad_w,
        output_pad_h, output_pad_w,
        dil_h, dil_w,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_CIN=1,  # Not used in this implementation
        BLOCK_SIZE_COUT=BLOCK_SIZE_COUT,
        BLOCK_SIZE_KH=1,   # Not used in this implementation
        BLOCK_SIZE_KW=1,   # Not used in this implementation
        BLOCK_SIZE_OUT_H=BLOCK_SIZE_OUT_H,
        BLOCK_SIZE_OUT_W=BLOCK_SIZE_OUT_W,
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), 
                 output_padding: tuple = (0, 0), dilation: tuple = (1, 1), 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same parameters as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        
        # Create weight and bias parameters (same as nn.ConvTranspose2d)
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, kernel_size[0], kernel_size[1])
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            stride=self.stride, 
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups
        )