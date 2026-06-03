import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv3d_kernel(
    x_ptr,  # Input tensor: (N, C_in, D, H, W)
    w_ptr,  # Weight tensor: (C_out, C_in, Kd, Kh, Kw)
    b_ptr,  # Bias tensor: (C_out,)
    out_ptr,  # Output tensor: (N, C_out, D_out, H_out, W_out)
    N, C_in, C_out,
    D, H, W,
    Kd, Kh, Kw,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    D_out, H_out, W_out,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs for output tensor dimensions
    pid_n = tl.program_id(0)  # batch index
    pid_c_out = tl.program_id(1)  # output channel index
    pid_h = tl.program_id(2)  # spatial h index
    pid_w = tl.program_id(3)  # spatial w index
    pid_d = tl.program_id(4)  # spatial d index
    
    # Compute output position
    d_out = pid_d
    h_out = pid_h
    w_out = pid_w
    
    # Compute input position for the top-left corner of the kernel
    d_in = d_out * stride_d - pad_d
    h_in = h_out * stride_h - pad_h
    w_in = w_out * stride_w - pad_w
    
    # Accumulator for the convolution result
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over input channels
    for c_in_idx in range(C_in):
        # Loop over kernel dimensions
        for kd in range(Kd):
            d_kernel = d_in + kd * dil_d
            if d_kernel < 0 or d_kernel >= D:
                continue
                
            for kh in range(Kh):
                h_kernel = h_in + kh * dil_h
                if h_kernel < 0 or h_kernel >= H:
                    continue
                    
                for kw in range(Kw):
                    w_kernel = w_in + kw * dil_w
                    if w_kernel < 0 or w_kernel >= W:
                        continue
                    
                    # Load input value: x[pid_n, c_in_idx, d_kernel, h_kernel, w_kernel]
                    x_offset = (
                        pid_n * (C_in * D * H * W) +
                        c_in_idx * (D * H * W) +
                        d_kernel * (H * W) +
                        h_kernel * W +
                        w_kernel
                    )
                    x_val = tl.load(x_ptr + x_offset)
                    
                    # Load weight value: w[pid_c_out, c_in_idx, kd, kh, kw]
                    w_offset = (
                        pid_c_out * (C_in * Kd * Kh * Kw) +
                        c_in_idx * (Kd * Kh * Kw) +
                        kd * (Kh * Kw) +
                        kh * Kw +
                        kw
                    )
                    w_val = tl.load(w_ptr + w_offset)
                    
                    # Accumulate
                    acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        acc += tl.load(b_ptr + pid_c_out)
    
    # Store output
    out_offset = (
        pid_n * (C_out * D_out * H_out * W_out) +
        pid_c_out * (D_out * H_out * W_out) +
        d_out * (H_out * W_out) +
        h_out * W_out +
        w_out
    )
    tl.store(out_ptr + out_offset, acc)

def triton_conv3d(x, weight, bias, stride, padding, dilation):
    """Triton-based 3D convolution implementation"""
    # Get dimensions
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
    out = torch.empty((N, C_out, D_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Grid configuration: (batch_size, out_channels, H_out, W_out, D_out)
    # Using a reasonable block size for performance
    BLOCK_SIZE = 32
    
    grid = lambda meta: (
        N, 
        C_out, 
        (H_out + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"],
        (W_out + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"],
        (D_out + meta["BLOCK_SIZE_D"] - 1) // meta["BLOCK_SIZE_D"]
    )
    
    # Launch kernel with tiling parameters
    conv3d_kernel[grid](
        x, weight, bias, out,
        N, C_in, C_out,
        D, H, W,
        Kd, Kh, Kw,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dil_d, dil_h, dil_w,
        D_out, H_out, W_out,
        BLOCK_SIZE_M=1,
        BLOCK_SIZE_N=1,
        BLOCK_SIZE_K=1,
        BLOCK_SIZE_H=8,
        BLOCK_SIZE_W=8,
        BLOCK_SIZE_D=4,
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for 3D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias_flag = bias
        
        # Initialize weights (using same initialization as PyTorch Conv3d)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights and bias similar to PyTorch's Conv3d"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D convolution using Triton kernel.
        """
        return triton_conv3d(x, self.weight, self.bias, 
                            self.stride, self.padding, self.dilation)

# Import math for kaiming initialization
import math