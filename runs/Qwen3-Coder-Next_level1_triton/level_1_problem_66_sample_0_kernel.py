import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv3d_kernel(
    input_ptr,  # Input tensor: (N, C_in, D, H, W)
    weight_ptr,  # Weight tensor: (C_out, C_in, Kd, Kh, Kw)
    bias_ptr,  # Bias tensor: (C_out,)
    output_ptr,  # Output tensor: (N, C_out, D_out, H_out, W_out)
    N, C_in, D, H, W,
    C_out, Kd, Kh, Kw,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    D_out, H_out, W_out,
    BLOCK_SIZE_C_out: tl.constexpr,
    BLOCK_SIZE_C_in: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_n = tl.program_id(0)  # batch index
    pid_c_out = tl.program_id(1)  # output channel index
    pid_d = tl.program_id(2)  # depth position in output
    pid_h = tl.program_id(3)  # height position in output
    pid_w = tl.program_id(4)  # width position in output
    
    # Calculate output spatial position
    out_d = pid_d
    out_h = pid_h
    out_w = pid_w
    
    # Calculate corresponding input spatial position
    in_d_start = out_d * stride_d - pad_d
    in_h_start = out_h * stride_h - pad_h
    in_w_start = out_w * stride_w - pad_w
    
    # Accumulator for the convolution result
    acc = tl.zeros((BLOCK_SIZE_C_out,), dtype=tl.float32)
    
    # Loop over output channels in block
    c_out_offset = pid_c_out * BLOCK_SIZE_C_out + tl.arange(0, BLOCK_SIZE_C_out)
    c_out_mask = c_out_offset < C_out
    
    # Load bias if available
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + c_out_offset, mask=c_out_mask, other=0.0)
        acc += bias_val
    
    # Loop over input channels
    for c_in in range(C_in):
        # Loop over kernel depth
        for kd in range(Kd):
            in_d = in_d_start + kd * dil_d
            # Check if in bounds
            if in_d >= 0 and in_d < D:
                # Loop over kernel height
                for kh in range(Kh):
                    in_h = in_h_start + kh * dil_h
                    if in_h >= 0 and in_h < H:
                        # Loop over kernel width
                        for kw in range(Kw):
                            in_w = in_w_start + kw * dil_w
                            if in_w >= 0 and in_w < W:
                                # Calculate indices
                                input_idx = pid_n * (C_in * D * H * W) + \
                                           c_in * (D * H * W) + \
                                           in_d * (H * W) + \
                                           in_h * W + \
                                           in_w
                                
                                weight_idx = c_out_offset[:, None] * (C_in * Kd * Kh * Kw) + \
                                            c_in * (Kd * Kh * Kw) + \
                                            kd * (Kh * Kw) + \
                                            kh * Kw + \
                                            kw
                                
                                # Load input value
                                input_val = tl.load(input_ptr + input_idx)
                                
                                # Load weight values
                                weight_vals = tl.load(weight_ptr + weight_idx, mask=c_out_mask[:, None], other=0.0)
                                
                                # Accumulate
                                acc += input_val * weight_vals[:, 0]
    
    # Store results
    output_idx = pid_n * (C_out * D_out * H_out * W_out) + \
                 c_out_offset * (D_out * H_out * W_out) + \
                 out_d * (H_out * W_out) + \
                 out_h * W_out + \
                 out_w
    
    tl.store(output_ptr + output_idx, acc, mask=c_out_mask)


def triton_conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride=(1, 1, 1),
    padding=(0, 0, 0),
    dilation=(1, 1, 1),
    groups=1
) -> torch.Tensor:
    """
    Triton-based 3D convolution implementation.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    assert groups == 1, "Only groups=1 is supported in this Triton implementation."
    
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    N, C_in, D, H, W = x.shape
    C_out, _, Kd, Kh, Kw = weight.shape
    
    # Calculate output dimensions
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    dil_d, dil_h, dil_w = dilation
    
    D_out = (D + 2 * pad_d - dil_d * (Kd - 1) - 1) // stride_d + 1
    H_out = (H + 2 * pad_h - dil_h * (Kh - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (Kw - 1) - 1) // stride_w + 1
    
    # Prepare output tensor
    output = torch.empty(N, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Grid dimensions
    grid = (N, (C_out + 63) // 64, D_out, H_out, W_out)
    
    # Launch kernel
    conv3d_kernel[grid](
        x, weight, bias, output,
        N, C_in, D, H, W,
        C_out, Kd, Kh, Kw,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dil_d, dil_h, dil_w,
        D_out, H_out, W_out,
        BLOCK_SIZE_C_out=64,
        BLOCK_SIZE_C_in=1,
        BLOCK_SIZE_K=1
    )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized Model using Triton kernel for 3D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), 
                 padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters for reference
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weight and bias parameters."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using Triton kernel.
        """
        return triton_conv3d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )