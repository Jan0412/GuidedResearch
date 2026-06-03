import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor pointer (N, C_in, H, W)
    w_ptr,  # Weight tensor pointer (C_out, C_in, K_h, K_w)
    b_ptr,  # Bias tensor pointer (C_out,) or None
    out_ptr,  # Output tensor pointer (N, C_out, H_out, W_out)
    N, C_in, C_out, H, W,  # Input dimensions
    K_h, K_w,  # Kernel dimensions
    stride_h, stride_w,  # Stride
    pad_h, pad_w,  # Padding
    dil_h, dil_w,  # Dilation
    H_out, W_out,  # Output dimensions
    BLOCK_N: tl.constexpr,
    BLOCK_C_out: tl.constexpr,
    BLOCK_C_in: tl.constexpr,
    BLOCK_K_h: tl.constexpr,
    BLOCK_K_w: tl.constexpr,
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    c_out_block = tl.program_id(1) * BLOCK_C_out
    out_h = tl.program_id(2)
    
    # Create output offsets for C_out dimension
    c_out_offsets = c_out_block + tl.arange(0, BLOCK_C_out)
    c_out_mask = c_out_offsets < C_out
    
    # Calculate input position corresponding to output position
    in_h = out_h * stride_h - pad_h
    
    # Iterate over W dimension in blocks
    for w_block_start in range(0, W_out, BLOCK_K_w):
        w_offsets = w_block_start + tl.arange(0, BLOCK_K_w)
        w_mask = w_offsets < W_out
        
        # Calculate output pointer base
        out_base = out_ptr + batch_idx * (C_out * H_out * W_out) + \
                   c_out_offsets[:, None] * (H_out * W_out) + \
                   out_h * W_out + w_offsets[None, :]
        
        # Initialize accumulator
        acc = tl.zeros((BLOCK_C_out, BLOCK_K_w), dtype=tl.float32)
        
        # Iterate over input channels
        for c_in_block_start in range(0, C_in, BLOCK_C_in):
            c_in_offsets = c_in_block_start + tl.arange(0, BLOCK_C_in)
            c_in_mask = c_in_offsets < C_in
            
            # Iterate over kernel height
            for kh in range(0, K_h, BLOCK_K_h):
                kh_offsets = kh + tl.arange(0, BLOCK_K_h)
                kh_mask = kh_offsets < K_h
                
                # Calculate input height positions
                in_h_kh = in_h + kh_offsets[:, None] * dil_h
                
                # Check if any kernel height positions are valid
                h_valid_mask = (in_h_kh >= 0) & (in_h_kh < H)
                
                # Iterate over kernel width
                for kw in range(0, K_w):
                    kw_offset = kw
                    in_w = w_offsets * stride_w - pad_w + kw_offset * dil_w
                    
                    # Create masks for valid positions
                    w_valid_mask = (in_w >= 0) & (in_w < W)
                    valid_mask = h_valid_mask & w_valid_mask[None, :]
                    
                    # Calculate input pointer offsets
                    x_offsets = batch_idx * (C_in * H * W) + \
                               (c_in_offsets[:, None, None] * (H * W)) + \
                               in_h_kh[:, :, None] * W + \
                               in_w[None, None, :]
                    
                    # Calculate weight pointer offsets
                    w_offsets_k = c_out_offsets[:, None, None] * (C_in * K_h * K_w) + \
                                 (c_in_offsets[None, :, None] * (K_h * K_w)) + \
                                 kh_offsets[None, None, :] * K_w + \
                                 kw
                    
                    # Load input values with masking
                    x_val = tl.load(x_ptr + x_offsets, 
                                   mask=(valid_mask & (c_in_offsets[None, :, None] < C_in)[None, :, :]) & 
                                        (kh_offsets[None, None, :] < K_h)[None, None, :], 
                                   other=0.0)
                    
                    # Load weight values with masking
                    w_val = tl.load(w_ptr + w_offsets_k,
                                   mask=(c_out_offsets[:, None, None] < C_out) & 
                                        (c_in_offsets[None, :, None] < C_in) & 
                                        (kh_offsets[None, None, :] < K_h) & 
                                        (kw_offset < K_w),
                                   other=0.0)
                    
                    # Accumulate convolution
                    acc += tl.sum(x_val * w_val, axis=1)
        
        # Add bias if present
        if b_ptr is not None:
            bias_offsets = c_out_offsets
            bias_val = tl.load(b_ptr + bias_offsets, mask=c_out_mask)
            acc += bias_val[:, None]
        
        # Store output
        tl.store(out_ptr + out_base, acc.to(x_ptr.dtype.element_ty), 
                mask=(c_out_mask[:, None] & w_mask[None, :]))


def triton_conv2d(x, weight, bias, stride, padding, dilation):
    """
    Custom Triton implementation of 2D convolution with support for:
    - Asymmetric kernel sizes
    - Asymmetric padding
    - Dilation
    - Asymmetric stride
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    N, C_in, H, W = x.shape
    C_out, _, K_h, K_w = weight.shape
    stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride)
    pad_h, pad_w = padding if isinstance(padding, tuple) else (padding, padding)
    dil_h, dil_w = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    
    # Calculate output dimensions
    H_out = (H + 2 * pad_h - dil_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (K_w - 1) - 1) // stride_w + 1
    
    # Allocate output tensor
    out = torch.empty((N, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Define block sizes for tiling
    BLOCK_N = 1
    BLOCK_C_out = min(64, C_out)
    BLOCK_C_in = min(32, C_in)
    BLOCK_K_h = min(3, K_h)
    BLOCK_K_w = 32
    
    # Grid dimensions: (batch, C_out blocks, H_out)
    grid = (N, (C_out + BLOCK_C_out - 1) // BLOCK_C_out, H_out)
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, out,
        N, C_in, C_out, H, W,
        K_h, K_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        H_out, W_out,
        BLOCK_N=BLOCK_N,
        BLOCK_C_out=BLOCK_C_out,
        BLOCK_C_in=BLOCK_C_in,
        BLOCK_K_h=BLOCK_K_h,
        BLOCK_K_w=BLOCK_K_w,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the convolution model using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), 
                 bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Initialize convolution weights and bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Create weight tensor with proper initialization
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 
                                               kernel_size[0], kernel_size[1]))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs optimized 2D convolution using Triton kernel.
        """
        return triton_conv2d(x, self.weight, self.bias, 
                            self.stride, self.padding, self.dilation)