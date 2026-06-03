import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel for 3D transposed convolution
@triton.jit
def conv_transpose3d_kernel(
    X,  # Input tensor: (N, C_in, D_in, H_in, W_in)
    W,  # Weight tensor: (C_in, C_out // groups, k_d, k_h, k_w)
    B,  # Bias tensor: (C_out,)
    Y,  # Output tensor: (N, C_out, D_out, H_out, W_out)
    N, C_in, C_out, groups,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    k_d, k_h, k_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    output_pad_d, output_pad_h, output_pad_w,
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs for output spatial dimensions and batch
    batch_idx = tl.program_id(0)
    out_d = tl.program_id(1)
    out_h = tl.program_id(2)
    out_w = tl.program_id(3)
    
    # Block for output channels
    c_out_offsets = tl.arange(0, BLOCK_SIZE_C_OUT)
    c_out_mask = c_out_offsets < C_out
    
    # Compute input position that contributes to this output position
    in_d = out_d * stride_d - pad_d + k_d - 1 - output_pad_d
    in_h = out_h * stride_h - pad_h + k_h - 1 - output_pad_h
    in_w = out_w * stride_w - pad_w + k_w - 1 - output_pad_w
    
    # Accumulator for each output channel
    acc = tl.zeros((BLOCK_SIZE_C_OUT,), dtype=tl.float32)
    
    # Iterate over input channels and kernel
    for c_in in range(0, C_in, BLOCK_SIZE_C_IN):
        c_in_offsets = c_in + tl.arange(0, BLOCK_SIZE_C_IN)
        c_in_mask = c_in_offsets < C_in
        
        # Check if input position is valid
        valid_input = (in_d >= 0) & (in_d < D_in) & (in_h >= 0) & (in_h < H_in) & (in_w >= 0) & (in_w < W_in)
        
        if valid_input:
            # Load input value
            x_idx = batch_idx * C_in * D_in * H_in * W_in + c_in_offsets * D_in * H_in * W_in + in_d * H_in * W_in + in_h * W_in + in_w
            x_val = tl.load(X + x_idx, mask=c_in_mask[:, None], other=0.0)
            
            # Iterate over kernel dimensions
            for kd in range(0, k_d, BLOCK_SIZE_K):
                kd_offsets = kd + tl.arange(0, BLOCK_SIZE_K)
                kd_mask = kd_offsets < k_d
                
                for kh in range(0, k_h, BLOCK_SIZE_K):
                    kh_offsets = kh + tl.arange(0, BLOCK_SIZE_K)
                    kh_mask = kh_offsets < k_h
                    
                    for kw in range(0, k_w, BLOCK_SIZE_K):
                        kw_offsets = kw + tl.arange(0, BLOCK_SIZE_K)
                        kw_mask = kw_offsets < k_w
                        
                        # Compute kernel indices
                        kernel_d = k_d - 1 - (in_d - (out_d * stride_d - pad_d + output_pad_d))
                        kernel_h = k_h - 1 - (in_h - (out_h * stride_h - pad_h + output_pad_h))
                        kernel_w = k_w - 1 - (in_w - (out_w * stride_w - pad_w + output_pad_w))
                        
                        # Load weights
                        w_idx = (c_in_offsets[:, None, None, None] * C_out * k_d * k_h * k_w + 
                                c_out_offsets[None, :, None, None] * k_d * k_h * k_w +
                                kernel_d * k_h * k_w + kernel_h * k_w + kernel_w)
                        w_val = tl.load(W + w_idx, mask=c_in_mask[:, None, None, None] & c_out_mask[None, :, None, None], other=0.0)
                        
                        # Accumulate
                        acc += tl.sum(x_val * w_val, axis=0)
    
    # Add bias if present
    if B is not None:
        b_val = tl.load(B + c_out_offsets, mask=c_out_mask, other=0.0)
        acc += b_val
    
    # Store result
    y_idx = batch_idx * C_out * D_out * H_out * W_out + c_out_offsets * D_out * H_out * W_out + out_d * H_out * W_out + out_h * W_out + out_w
    tl.store(Y + y_idx, acc.to(tl.float32), mask=c_out_mask)


class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution with a square input and an asymmetric kernel.
    Optimized with Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Validate inputs
        assert len(kernel_size) == 3, "kernel_size must be a tuple of length 3"
        assert len(stride) == 3, "stride must be a tuple of length 3"
        assert len(padding) == 3, "padding must be a tuple of length 3"
        assert len(output_padding) == 3, "output_padding must be a tuple of length 3"
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels // groups, *kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using Triton kernel.
        """
        # Ensure input is contiguous and on CUDA
        if not x.is_cuda:
            x = x.cuda()
        x = x.contiguous()
        
        # Get dimensions
        N, C_in, D_in, H_in, W_in = x.shape
        k_d, k_h, k_w = self.kernel_size
        stride_d, stride_h, stride_w = self.stride
        pad_d, pad_h, pad_w = self.padding
        output_pad_d, output_pad_h, output_pad_w = self.output_padding
        
        # Calculate output dimensions manually
        D_out = (D_in - 1) * stride_d - 2 * pad_d + (k_d - 1) + output_pad_d + 1
        H_out = (H_in - 1) * stride_h - 2 * pad_h + (k_h - 1) + output_pad_h + 1
        W_out = (W_in - 1) * stride_w - 2 * pad_w + (k_w - 1) + output_pad_w + 1
        
        # Create output tensor
        output = torch.empty(N, self.out_channels, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
        
        # Set up kernel launch parameters
        BLOCK_SIZE_C_OUT = 8
        BLOCK_SIZE_C_IN = 8
        BLOCK_SIZE_K = 3
        
        grid = (N, D_out, H_out, W_out)
        
        # Launch the kernel
        conv_transpose3d_kernel[grid](
            x, self.weight, self.bias, output,
            N, self.in_channels, self.out_channels, self.groups,
            D_in, H_in, W_in,
            D_out, H_out, W_out,
            k_d, k_h, k_w,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            output_pad_d, output_pad_h, output_pad_w,
            BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
            BLOCK_SIZE_C_IN=BLOCK_SIZE_C_IN,
            BLOCK_SIZE_K=BLOCK_SIZE_K
        )
        
        return output

import math