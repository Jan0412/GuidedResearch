import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Tuple

# Triton kernel for 3D convolution with kernel_size x kernel_size x 1
@triton.jit
def conv3d_kernel(
    x_ptr,  # Input tensor: (B, C_in, H, W, D)
    w_ptr,  # Weight tensor: (C_out, C_in, K_h, K_w, 1)
    b_ptr,  # Bias tensor: (C_out,)
    y_ptr,  # Output tensor: (B, C_out, H_out, W_out, D)
    B: tl.constexpr,  # Batch size
    C_in: tl.constexpr,  # Input channels
    H: tl.constexpr,  # Input height
    W: tl.constexpr,  # Input width
    D: tl.constexpr,  # Input depth
    C_out: tl.constexpr,  # Output channels
    K_h: tl.constexpr,  # Kernel height
    K_w: tl.constexpr,  # Kernel width
    stride_h: tl.constexpr,  # Stride height
    stride_w: tl.constexpr,  # Stride width
    pad_h: tl.constexpr,  # Padding height
    pad_w: tl.constexpr,  # Padding width
    dil_h: tl.constexpr,  # Dilation height
    dil_w: tl.constexpr,  # Dilation width
    H_out: tl.constexpr,  # Output height
    W_out: tl.constexpr,  # Output width
    BLOCK_SIZE_C_out: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_K: tl.constexpr,  # Block size for kernel elements
):
    # Get batch, output height, output width, output channel indices
    batch_idx = tl.program_id(0)
    out_h_idx = tl.program_id(1)
    out_w_idx = tl.program_id(2)
    out_c_block_start = tl.program_id(3) * BLOCK_SIZE_C_out
    
    # Compute output channel offsets
    out_c_offsets = out_c_block_start + tl.arange(0, BLOCK_SIZE_C_out)
    out_c_mask = out_c_offsets < C_out
    
    # Compute input spatial coordinates with padding and dilation
    in_h = out_h_idx * stride_h - pad_h + tl.arange(0, K_h)[None, :] * dil_h
    in_w = out_w_idx * stride_w - pad_w + tl.arange(0, K_w)[None, :] * dil_w
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_C_out,), dtype=tl.float32)
    
    # Loop over input channels
    for in_c in range(C_in):
        # Loop over kernel height
        for kh in range(K_h):
            # Loop over kernel width
            for kw in range(K_w):
                # Check if input coordinates are valid
                h_valid = (in_h[kh, kw] >= 0) & (in_h[kh, kw] < H)
                w_valid = (in_w[kh, kw] >= 0) & (in_w[kh, kw] < W)
                valid = h_valid & w_valid
                
                # Get input value (0 if out of bounds)
                in_h_idx = tl.maximum(tl.minimum(in_h[kh, kw], H - 1), 0)
                in_w_idx = tl.maximum(tl.minimum(in_w[kh, kw], W - 1), 0)
                
                # Compute input pointer offset
                input_offset = (batch_idx * C_in * H * W * D + 
                               in_c * H * W * D + 
                               in_h_idx * W * D + 
                               in_w_idx * D)
                
                # Load input values for all depth slices
                x_vals = tl.load(x_ptr + input_offset + tl.arange(0, D),
                               mask=valid[:, None] & (tl.arange(0, D) < D),
                               other=0.0)
                
                # Load weight value
                weight_offset = (out_c_offsets[:, None, None, None] * C_in * K_h * K_w + 
                                in_c * K_h * K_w + 
                                kh * K_w + 
                                kw)
                w_val = tl.load(w_ptr + weight_offset, mask=out_c_mask[:, None, None])
                
                # Multiply and accumulate
                acc += tl.sum(x_vals * w_val, axis=[1, 2])
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_c_offsets, mask=out_c_mask)
        acc += bias
    
    # Store result
    y_offset = (batch_idx * C_out * H_out * W_out * D +
               out_c_offsets[:, None] * H_out * W_out * D +
               out_h_idx * W_out * D +
               out_w_idx * D)
    tl.store(y_ptr + y_offset, acc.to(x_ptr.dtype.element_ty), mask=out_c_mask[:, None] & (tl.arange(0, D) < D))


def triton_conv3d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                 stride: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (0, 0),
                 dilation: Tuple[int, int] = (1, 1)) -> torch.Tensor:
    """
    Performs 3D convolution with kernel_size x kernel_size x 1 using Triton.
    
    Args:
        x: Input tensor of shape (B, C_in, H, W, D)
        weight: Weight tensor of shape (C_out, C_in, K_h, K_w, 1)
        bias: Optional bias tensor of shape (C_out,)
        stride: Stride (stride_h, stride_w)
        padding: Padding (pad_h, pad_w)
        dilation: Dilation (dil_h, dil_w)
    
    Returns:
        Output tensor of shape (B, C_out, H_out, W_out, D)
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C_in, H, W, D = x.shape
    C_out, _, K_h, K_w, _ = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    
    # Compute output dimensions
    H_out = (H + 2 * pad_h - dil_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (K_w - 1) - 1) // stride_w + 1
    
    # Prepare output tensor
    y = torch.empty((B, C_out, H_out, W_out, D), dtype=x.dtype, device=x.device)
    
    # Set up kernel parameters
    BLOCK_SIZE_C_out = 16  # Tunable parameter
    BLOCK_SIZE_K = 8      # Not used directly but reserved for future optimizations
    
    # Grid dimensions: (batch, H_out, W_out, C_out_blocks)
    grid = (B, H_out, W_out, (C_out + BLOCK_SIZE_C_out - 1) // BLOCK_SIZE_C_out)
    
    # Launch kernel
    conv3d_kernel[grid](
        x, weight, bias, y,
        B, C_in, H, W, D, C_out, K_h, K_w,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
        H_out, W_out,
        BLOCK_SIZE_C_out, BLOCK_SIZE_K
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized 3D convolution model using custom Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Validate parameters (only groups=1 supported for this kernel)
        if groups != 1:
            raise ValueError("Triton kernel only supports groups=1")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D convolution using Triton kernel.
        """
        # Handle stride, padding, dilation as tuples for 2D spatial dimensions
        stride = (self.stride, self.stride)
        padding = (self.padding, self.padding)
        dilation = (self.dilation, self.dilation)
        
        # Call the Triton convolution kernel
        return triton_conv3d(x, self.weight, self.bias, stride, padding, dilation)