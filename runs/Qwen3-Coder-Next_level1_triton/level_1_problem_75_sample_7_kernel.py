import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Input tensor: (B, C_in, H_in, W_in)
    w_ptr,  # Weight tensor: (C_in, C_out // G, K_h, K_w)
    b_ptr,  # Bias tensor: (C_out,) or None
    y_ptr,  # Output tensor: (B, C_out, H_out, W_out)
    B: tl.constexpr,  # Batch size
    C_in: tl.constexpr,  # Input channels
    C_out: tl.constexpr,  # Output channels
    H_in: tl.constexpr,  # Input height
    W_in: tl.constexpr,  # Input width
    H_out: tl.constexpr,  # Output height
    W_out: tl.constexpr,  # Output width
    K_h: tl.constexpr,  # Kernel height
    K_w: tl.constexpr,  # Kernel width
    stride_h: tl.constexpr,  # Stride height
    stride_w: tl.constexpr,  # Stride width
    pad_h: tl.constexpr,  # Padding height
    pad_w: tl.constexpr,  # Padding width
    dil_h: tl.constexpr,  # Dilation height
    dil_w: tl.constexpr,  # Dilation width
    groups: tl.constexpr,  # Number of groups
    BLOCK_SIZE_C_OUT: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_C_IN: tl.constexpr,  # Block size for input channels
    BLOCK_SIZE_K_H: tl.constexpr,  # Block size for kernel height
    BLOCK_SIZE_K_W: tl.constexpr,  # Block size for kernel width
    BLOCK_SIZE_H_OUT: tl.constexpr,  # Block size for output height
    BLOCK_SIZE_W_OUT: tl.constexpr,  # Block size for output width
):
    # Program IDs for output tensor dimensions
    pid_b = tl.program_id(0)  # batch index
    pid_c_out = tl.program_id(1)  # output channel block index
    pid_h = tl.program_id(2)  # output height block index
    pid_w = tl.program_id(3)  # output width block index
    
    # Calculate output channel range for this program
    c_out_offsets = pid_c_out * BLOCK_SIZE_C_OUT + tl.arange(0, BLOCK_SIZE_C_OUT)
    c_out_mask = c_out_offsets < C_out
    
    # Calculate output spatial indices
    h_out_start = pid_h * BLOCK_SIZE_H_OUT
    w_out_start = pid_w * BLOCK_SIZE_W_OUT
    
    # Initialize accumulator for output
    acc = tl.zeros((BLOCK_SIZE_C_OUT,), tl.float32)
    
    # Iterate over input channels and kernel dimensions
    for c_in_block in range((C_in + BLOCK_SIZE_C_IN - 1) // BLOCK_SIZE_C_IN):
        c_in_start = c_in_block * BLOCK_SIZE_C_IN
        c_in_offsets = c_in_start + tl.arange(0, BLOCK_SIZE_C_IN)
        c_in_mask = c_in_offsets < C_in
        
        # Group index for this channel
        group_idx = c_in_start // (C_in // groups)
        
        # Iterate over kernel height
        for kh_block in range((K_h + BLOCK_SIZE_K_H - 1) // BLOCK_SIZE_K_H):
            kh_start = kh_block * BLOCK_SIZE_K_H
            kh_offsets = kh_start + tl.arange(0, BLOCK_SIZE_K_H)
            kh_mask = kh_offsets < K_h
            
            # Iterate over kernel width
            for kw_block in range((K_w + BLOCK_SIZE_K_W - 1) // BLOCK_SIZE_K_W):
                kw_start = kw_block * BLOCK_SIZE_K_W
                kw_offsets = kw_start + tl.arange(0, BLOCK_SIZE_K_W)
                kw_mask = kw_offsets < K_w
                
                # Calculate input spatial positions from output positions
                # For transposed convolution: H_in = (H_out - 1) * stride - 2*pad + dil*(K-1) + 1
                h_in_offsets = h_out_start * stride_h + kh_offsets * dil_h - pad_h
                w_in_offsets = w_out_start * stride_w + kw_offsets * dil_w - pad_w
                
                # Create masks for valid input positions
                h_in_mask = (h_in_offsets >= 0) & (h_in_offsets < H_in)
                w_in_mask = (w_in_offsets >= 0) & (w_in_offsets < W_in)
                
                # Load input values
                # x shape: (B, C_in, H_in, W_in)
                x_offset_h = h_in_offsets[:, None] * W_in + w_in_offsets[None, :]
                x_offset_c = c_in_offsets[None, :, None, None] * H_in * W_in + x_offset_h[None, None, :, :]
                x_offset_b = pid_b * C_in * H_in * W_in
                
                # Load weight values
                # w shape: (C_in, C_out // G, K_h, K_w)
                w_offset_c_out = c_out_offsets[:, None, None, None] % (C_out // groups)
                w_offset_group = group_idx * (C_out // groups)
                w_offset = w_offset_b = c_in_offsets[None, :, None, None] * (C_out // groups) * K_h * K_w + \
                                 w_offset_c_out * K_h * K_w + \
                                 kh_offsets[None, None, :, None] * K_w + \
                                 kw_offsets[None, None, None, :]
                
                # Compute convolution sum
                # We need to handle the broadcasting properly
                for i_h in range(min(BLOCK_SIZE_K_H, K_h - kh_start)):
                    for i_w in range(min(BLOCK_SIZE_K_W, K_w - kw_start)):
                        if kh_start + i_h < K_h and kw_start + i_w < K_w:
                            # Calculate input position for this kernel element
                            h_in = h_out_start * stride_h + (kh_start + i_h) * dil_h - pad_h
                            w_in = w_out_start * stride_w + (kw_start + i_w) * dil_w - pad_w
                            
                            if h_in >= 0 and h_in < H_in and w_in >= 0 and w_in < W_in:
                                # Load input at position
                                x_val = tl.load(
                                    x_ptr + pid_b * C_in * H_in * W_in + 
                                    c_in_offsets * H_in * W_in + 
                                    h_in * W_in + w_in,
                                    mask=c_in_mask,
                                    other=0.0
                                )
                                
                                # Load weight at position
                                w_val = tl.load(
                                    w_ptr + c_in_offsets * (C_out // groups) * K_h * K_w + 
                                    (c_out_offsets % (C_out // groups)) * K_h * K_w + 
                                    (kh_start + i_h) * K_w + (kw_start + i_w),
                                    mask=c_in_mask[:, None] & c_out_mask[None, :],
                                    other=0.0
                                )
                                
                                # Accumulate
                                acc += tl.sum(x_val[:, None] * w_val, axis=0)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + c_out_offsets, mask=c_out_mask, other=0.0)
        acc += bias
    
    # Store output
    y_offsets = (
        pid_b * C_out * H_out * W_out +
        c_out_offsets[:, None, None] * H_out * W_out +
        (h_out_start + tl.arange(0, BLOCK_SIZE_H_OUT)[None, :, None]) * W_out +
        (w_out_start + tl.arange(0, BLOCK_SIZE_W_OUT)[None, None, :])
    )
    
    # Create mask for output bounds
    h_out_mask = (h_out_start + tl.arange(0, BLOCK_SIZE_H_OUT)) < H_out
    w_out_mask = (w_out_start + tl.arange(0, BLOCK_SIZE_W_OUT)) < W_out
    mask = c_out_mask[:, None, None] & h_out_mask[None, :, None] & w_out_mask[None, None, :]
    
    tl.store(y_ptr + y_offsets, acc[:, None, None], mask=mask)


class TritonConvTranspose2d(nn.Module):
    """
    Custom implementation of 2D transposed convolution using Triton kernels.
    Handles asymmetric parameters, groups, padding, and dilation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), 
                 padding=(0, 0), dilation=(1, 1), groups=1, bias=False):
        super(TritonConvTranspose2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        
        # Calculate output shape dimensions
        self.kernel_h, self.kernel_w = self.kernel_size
        self.stride_h, self.stride_w = self.stride
        self.pad_h, self.pad_w = self.padding
        self.dil_h, self.dil_w = self.dilation
        
        # Weight tensor shape: (in_channels, out_channels // groups, kernel_h, kernel_w)
        self.weight = nn.Parameter(torch.zeros(in_channels, out_channels // groups, self.kernel_h, self.kernel_w))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        # Kaiming uniform initialization for transposed convolution
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        batch_size, _, H_in, W_in = x.shape
        
        # Calculate output dimensions
        H_out = (H_in - 1) * self.stride_h - 2 * self.pad_h + self.dil_h * (self.kernel_h - 1) + 1
        W_out = (W_in - 1) * self.stride_w - 2 * self.pad_w + self.dil_w * (self.kernel_w - 1) + 1
        
        # Allocate output tensor
        y = torch.empty(batch_size, self.out_channels, H_out, W_out, device=x.device, dtype=x.dtype)
        
        # Define block sizes for tiling
        BLOCK_SIZE_C_OUT = 32
        BLOCK_SIZE_C_IN = 8
        BLOCK_SIZE_K_H = 3
        BLOCK_SIZE_K_W = 5
        BLOCK_SIZE_H_OUT = 8
        BLOCK_SIZE_W_OUT = 8
        
        # Grid dimensions
        grid = (
            batch_size,  # B
            (self.out_channels + BLOCK_SIZE_C_OUT - 1) // BLOCK_SIZE_C_OUT,  # C_out blocks
            (H_out + BLOCK_SIZE_H_OUT - 1) // BLOCK_SIZE_H_OUT,  # H_out blocks
            (W_out + BLOCK_SIZE_W_OUT - 1) // BLOCK_SIZE_W_OUT   # W_out blocks
        )
        
        # Launch kernel
        conv_transpose2d_kernel[grid](
            x, self.weight, self.bias, y,
            B=batch_size,
            C_in=self.in_channels,
            C_out=self.out_channels,
            H_in=H_in,
            W_in=W_in,
            H_out=H_out,
            W_out=W_out,
            K_h=self.kernel_h,
            K_w=self.kernel_w,
            stride_h=self.stride_h,
            stride_w=self.stride_w,
            pad_h=self.pad_h,
            pad_w=self.pad_w,
            dil_h=self.dil_h,
            dil_w=self.dil_w,
            groups=self.groups,
            BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
            BLOCK_SIZE_C_IN=BLOCK_SIZE_C_IN,
            BLOCK_SIZE_K_H=BLOCK_SIZE_K_H,
            BLOCK_SIZE_K_W=BLOCK_SIZE_K_W,
            BLOCK_SIZE_H_OUT=BLOCK_SIZE_H_OUT,
            BLOCK_SIZE_W_OUT=BLOCK_SIZE_W_OUT
        )
        
        return y


import math


class ModelNew(nn.Module):
    """
    Optimized version of the Model using custom Triton kernels for transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), 
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = TritonConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, 
            dilation=dilation, groups=groups, 
            bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose2d(x)