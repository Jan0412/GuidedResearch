import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def im2col_kernel(
    x_ptr,              # Input tensor pointer (N, C, H, W)
    k_ptr,              # Kernel tensor pointer (OC, C, KH, KW)
    bias_ptr,           # Bias tensor pointer (OC,) - optional
    out_ptr,            # Output tensor pointer (N, OH, OW, OC)
    N, C, H, W,         # Input dimensions
    OC, KH, KW,         # Kernel dimensions
    stride_h, stride_w, # Stride
    pad_h, pad_w,       # Padding
    dil_h, dil_w,       # Dilation
    OH, OW,             # Output dimensions
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr,  # Block size for accumulation
):
    # Get program IDs
    pid_n = tl.program_id(0)  # batch index
    pid_oh = tl.program_id(1)  # output height index
    pid_ow = tl.program_id(2)  # output width index
    pid_oc_block = tl.program_id(3)  # output channel block index
    
    # Calculate output position
    out_h = pid_oh
    out_w = pid_ow
    
    # Calculate the starting position in the input for this output position
    in_h_start = out_h * stride_h - pad_h
    in_w_start = out_w * stride_w - pad_w
    
    # Create block for output channels
    oc_offsets = pid_oc_block * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    oc_mask = oc_offsets < OC
    
    # Create block for batch dimension
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_offsets < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    
    # Iterate over the kernel dimensions
    for kh in range(KH):
        in_h = in_h_start + kh * dil_h
        kh_offsets = kh * KW * C
        
        for kw in range(KW):
            in_w = in_w_start + kw * dil_w
            kw_offsets = kw * C
            
            # Check if this position is within input bounds
            valid_pos = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W)
            
            if not tl.static_bool(valid_pos):
                # For positions outside input bounds, use zero (padding)
                continue
                
            # Calculate input offset for this position
            in_offset = (n_offsets[:, None] * H * W * C + 
                        in_h * W * C + 
                        in_w * C + 
                        tl.arange(0, C)[None, :])
            
            # Load input values (N, C)
            x_block = tl.load(x_ptr + in_offset, 
                            mask=n_mask[:, None] & (tl.arange(0, C)[None, :] < C),
                            other=0.0)
            
            # Load kernel values (C, OC)
            kernel_offset = (oc_offsets[None, :] * C * KH * KW + 
                           tl.arange(0, C)[:, None] * KH * KW + 
                           kh_offsets + 
                           kw_offsets)
            k_block = tl.load(k_ptr + kernel_offset,
                            mask=oc_mask[None, :] & (tl.arange(0, C)[:, None] < C),
                            other=0.0)
            
            # Accumulate: x_block (N, C) @ k_block (C, OC) -> (N, OC)
            acc += tl.dot(x_block, k_block)
    
    # Add bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + oc_offsets, mask=oc_mask, other=0.0)
        acc += bias[None, :]
    
    # Store output
    out_offset = (pid_n * OH * OW * OC + 
                 out_h * OW * OC + 
                 out_w * OC + 
                 oc_offsets)
    tl.store(out_ptr + out_offset, acc, mask=n_mask[:, None] & oc_mask[None, :])


@triton.jit
def conv2d_kernel(
    x_ptr,              # Input tensor pointer (N, C, H, W)
    k_ptr,              # Kernel tensor pointer (OC, C, KH, KW)
    bias_ptr,           # Bias tensor pointer (OC,) - optional
    out_ptr,            # Output tensor pointer (N, OC, OH, OW)
    N, C, H, W,         # Input dimensions
    OC, KH, KW,         # Kernel dimensions
    stride_h, stride_w, # Stride
    pad_h, pad_w,       # Padding
    dil_h, dil_w,       # Dilation
    OH, OW,             # Output dimensions
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr,  # Block size for accumulation
):
    # Get program IDs
    pid_n = tl.program_id(0)  # batch index
    pid_oc_block = tl.program_id(1)  # output channel block index
    pid_oh = tl.program_id(2)  # output height index
    pid_ow = tl.program_id(3)  # output width index
    
    # Calculate output position
    out_h = pid_oh
    out_w = pid_ow
    
    # Calculate the starting position in the input for this output position
    in_h_start = out_h * stride_h - pad_h
    in_w_start = out_w * stride_w - pad_w
    
    # Create block for output channels
    oc_offsets = pid_oc_block * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    oc_mask = oc_offsets < OC
    
    # Create block for batch dimension
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_offsets < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    
    # Iterate over the kernel dimensions
    for kh in range(KH):
        in_h = in_h_start + kh * dil_h
        kh_offsets = kh * KW * C
        
        for kw in range(KW):
            in_w = in_w_start + kw * dil_w
            kw_offsets = kw * C
            
            # Check if this position is within input bounds
            valid_pos = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W)
            
            if not tl.static_bool(valid_pos):
                # For positions outside input bounds, use zero (padding)
                continue
                
            # Calculate input offset for this position
            in_offset = (n_offsets[:, None] * H * W * C + 
                        in_h * W * C + 
                        in_w * C + 
                        tl.arange(0, C)[None, :])
            
            # Load input values (N, C)
            x_block = tl.load(x_ptr + in_offset, 
                            mask=n_mask[:, None] & (tl.arange(0, C)[None, :] < C),
                            other=0.0)
            
            # Load kernel values (C, OC)
            kernel_offset = (oc_offsets[None, :] * C * KH * KW + 
                           tl.arange(0, C)[:, None] * KH * KW + 
                           kh_offsets + 
                           kw_offsets)
            k_block = tl.load(k_ptr + kernel_offset,
                            mask=oc_mask[None, :] & (tl.arange(0, C)[:, None] < C),
                            other=0.0)
            
            # Accumulate: x_block (N, C) @ k_block (C, OC) -> (N, OC)
            acc += tl.dot(x_block, k_block)
    
    # Add bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + oc_offsets, mask=oc_mask, other=0.0)
        acc += bias[None, :]
    
    # Store output in (N, OC, OH, OW) format
    out_offset = (pid_n * OC * OH * OW + 
                 oc_offsets[None, :] * OH * OW + 
                 out_h * OW + 
                 out_w)
    tl.store(out_ptr + out_offset, acc, mask=n_mask[:, None] & oc_mask[None, :])


def triton_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                 stride=1, padding=0, dilation=1, groups=1):
    """
    Performs 2D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (N, C, H, W)
        weight: Weight tensor of shape (OC, C, KH, KW)
        bias: Optional bias tensor of shape (OC,)
        stride: Stride of convolution
        padding: Padding applied to input
        dilation: Spacing between kernel elements
        groups: Number of blocked connections (not fully implemented for groups > 1)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Get dimensions
    N, C, H, W = x.shape
    OC, C_k, KH, KW = weight.shape
    
    # Calculate output dimensions
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
        
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding
        
    if isinstance(dilation, int):
        dil_h = dil_w = dilation
    else:
        dil_h, dil_w = dilation
    
    OH = (H + 2 * pad_h - dil_h * (KH - 1) - 1) // stride_h + 1
    OW = (W + 2 * pad_w - dil_w * (KW - 1) - 1) // stride_w + 1
    
    # Prepare output tensor
    out = torch.empty((N, OC, OH, OW), dtype=x.dtype, device=x.device)
    
    # Set block sizes (tunable parameters for performance)
    BLOCK_SIZE_M = 16  # output channels per block
    BLOCK_SIZE_N = 4   # batch per block
    BLOCK_SIZE_K = 16  # accumulation block size
    
    # Calculate grid dimensions
    grid = lambda meta: (
        N,                                  # batch
        (OC + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],  # output channel blocks
        OH,                                 # output height
        OW                                  # output width
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, out,
        N, C, H, W,
        OC, KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        OH, OW,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with a square input and an asymmetric kernel.
    Uses optimized Triton kernel for the convolution operation.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        # Kaiming initialization for convolutional layers
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel.
        """
        return triton_conv2d(x, self.weight, self.bias,
                           stride=self.stride, 
                           padding=self.padding, 
                           dilation=self.dilation, 
                           groups=self.groups)