import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor pointer (B, C_in, H, W)
    w_ptr,  # Weight tensor pointer (C_out, C_in, K_h, K_w)
    b_ptr,  # Bias tensor pointer (C_out,) or None
    out_ptr,  # Output tensor pointer (B, C_out, H_out, W_out)
    B, C_in, H, W,  # Input dimensions
    C_out, K_h, K_w,  # Output channels and kernel dimensions
    stride_h, stride_w,  # Stride
    pad_h_top, pad_h_bottom, pad_w_left, pad_w_right,  # Padding
    dil_h, dil_w,  # Dilation
    H_out, W_out,  # Output dimensions
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr,  # Block size for input channels
    BLOCK_SIZE_P: tl.constexpr,  # Block size for output height
    BLOCK_SIZE_Q: tl.constexpr,  # Block size for output width
):
    # Get program IDs
    pid_c = tl.program_id(0)  # Output channel block
    pid_b = tl.program_id(1)  # Batch block
    pid_h = tl.program_id(2)  # Output height block
    pid_w = tl.program_id(3)  # Output width block
    
    # Calculate output position
    out_h_start = pid_h * BLOCK_SIZE_P
    out_w_start = pid_w * BLOCK_SIZE_Q
    out_c_start = pid_c * BLOCK_SIZE_M
    out_b_start = pid_b * BLOCK_SIZE_N
    
    # Create output coordinate offsets
    out_h_offsets = out_h_start + tl.arange(0, BLOCK_SIZE_P)
    out_w_offsets = out_w_start + tl.arange(0, BLOCK_SIZE_Q)
    out_c_offsets = out_c_start + tl.arange(0, BLOCK_SIZE_M)
    out_b_offsets = out_b_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for output dimensions
    out_h_mask = out_h_offsets < H_out
    out_w_mask = out_w_offsets < W_out
    out_c_mask = out_c_offsets < C_out
    out_b_mask = out_b_offsets < B
    
    # Initialize accumulator for convolution
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_P, BLOCK_SIZE_Q), dtype=tl.float32)
    
    # Convolution: iterate over input channels and kernel positions
    for c_in in range(C_in):
        for kh in range(K_h):
            for kw in range(K_w):
                # Calculate input coordinates
                in_h = out_h_start * stride_h + kh * dil_h - pad_h_top
                in_w = out_w_start * stride_w + kw * dil_w - pad_w_left
                
                # Input offsets
                in_h_offsets = in_h + tl.arange(0, BLOCK_SIZE_P)
                in_w_offsets = in_w + tl.arange(0, BLOCK_SIZE_Q)
                
                # Input masks
                in_h_mask = (in_h_offsets >= 0) & (in_h_offsets < H)
                in_w_mask = (in_w_offsets >= 0) & (in_w_offsets < W)
                input_mask = out_h_mask[:, None] & out_w_mask[None, :] & in_h_mask[:, None] & in_w_mask[None, :]
                
                # Load input values
                x_indices = (
                    out_b_offsets[:, None, None] * (C_in * H * W) +
                    c_in * (H * W) +
                    in_h_offsets[:, None, None] * W +
                    in_w_offsets[None, :, :]
                )
                x_vals = tl.load(
                    x_ptr + x_indices,
                    mask=input_mask[:, :, None],
                    other=0.0
                )
                
                # Load weight values
                w_indices = (
                    out_c_offsets[:, None, None] * (C_in * K_h * K_w) +
                    c_in * (K_h * K_w) +
                    kh * K_w + kw
                )
                w_vals = tl.load(
                    w_ptr + w_indices,
                    mask=out_c_mask[:, None, None]
                )
                
                # Accumulate convolution
                acc += w_vals[:, :, None] * x_vals
                
    # Add bias if present
    if b_ptr is not None:
        b_indices = out_c_offsets
        b_vals = tl.load(b_ptr + b_indices, mask=out_c_mask)
        acc += b_vals[:, None, None]
    
    # Store output
    out_indices = (
        out_b_offsets[:, None, None] * (C_out * H_out * W_out) +
        out_c_offsets[:, None, None] * (H_out * W_out) +
        out_h_offsets[None, :, None] * W_out +
        out_w_offsets[None, None, :]
    )
    
    out_mask = (
        out_b_mask[:, None, None] & 
        out_c_mask[:, None, None] & 
        out_h_mask[None, :, None] & 
        out_w_mask[None, None, :]
    )
    
    tl.store(out_ptr + out_indices, acc, mask=out_mask)


def triton_conv2d(x, weight, bias, stride, padding, dilation):
    """
    Triton-based 2D convolution implementation.
    
    Args:
        x: Input tensor of shape (B, C_in, H, W)
        weight: Weight tensor of shape (C_out, C_in, K_h, K_w)
        bias: Bias tensor of shape (C_out,) or None
        stride: Tuple (stride_h, stride_w)
        padding: Tuple (pad_h_top, pad_h_bottom, pad_w_left, pad_w_right)
        dilation: Tuple (dil_h, dil_w)
    """
    B, C_in, H, W = x.shape
    C_out, _, K_h, K_w = weight.shape
    stride_h, stride_w = stride
    pad_h_top, pad_h_bottom, pad_w_left, pad_w_right = padding
    dil_h, dil_w = dilation
    
    # Calculate output dimensions
    H_out = (H + pad_h_top + pad_h_bottom - dil_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W + pad_w_left + pad_w_right - dil_w * (K_w - 1) - 1) // stride_w + 1
    
    # Prepare output tensor
    out = torch.empty((B, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Configure block sizes for optimal performance
    BLOCK_SIZE_M = 16  # Output channels per block
    BLOCK_SIZE_N = 2   # Batch per block
    BLOCK_SIZE_P = 8   # Output height per block  
    BLOCK_SIZE_Q = 8   # Output width per block
    BLOCK_SIZE_K = 16  # Input channels per block (can be tuned)
    
    # Grid configuration: (C_out_blocks, B_blocks, H_out_blocks, W_out_blocks)
    grid = (
        (C_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (B + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
        (H_out + BLOCK_SIZE_P - 1) // BLOCK_SIZE_P,
        (W_out + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, out,
        B, C_in, H, W,
        C_out, K_h, K_w,
        stride_h, stride_w,
        pad_h_top, pad_h_bottom, pad_w_left, pad_w_right,
        dil_h, dil_w,
        H_out, W_out,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_P=BLOCK_SIZE_P,
        BLOCK_SIZE_Q=BLOCK_SIZE_Q
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized Model using Triton kernels for 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: int = 1, padding: tuple = (0, 0), 
                 dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters for reconstruction
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias_flag = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Handle padding format conversion
        # Input padding is (top/bottom, left/right) -> convert to (top, bottom, left, right)
        if isinstance(self.padding, int):
            pad_h = pad_w = self.padding
            padding_tuple = (pad_h, pad_h, pad_w, pad_w)
        elif len(self.padding) == 2:
            padding_tuple = (self.padding[0], self.padding[0], self.padding[1], self.padding[1])
        else:
            padding_tuple = self.padding
            
        # Handle stride format
        stride_tuple = (self.stride, self.stride) if isinstance(self.stride, int) else self.stride
        
        # Handle dilation format
        dilation_tuple = (self.dilation[0], self.dilation[1]) if isinstance(self.dilation, tuple) else (self.dilation, self.dilation)
        
        # Call Triton convolution
        return triton_conv2d(
            x, weight, self.bias, 
            stride_tuple, padding_tuple, dilation_tuple
        )