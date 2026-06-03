import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor (batch, in_channels, H, W)
    w_ptr,  # Weight tensor (out_channels, in_channels, kernel_h, kernel_w)
    b_ptr,  # Bias tensor (out_channels,) or None
    out_ptr,  # Output tensor (batch, out_channels, out_h, out_w)
    batch_size, in_channels, out_channels,
    height, width, 
    out_h, out_w,
    kernel_h, kernel_w,
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr,  # Block size for in_channels
    BLOCK_H: tl.constexpr,       # Block size for output height
    BLOCK_W: tl.constexpr,       # Block size for output width
):
    # Get program IDs
    pid_batch = tl.program_id(1)
    pid_out_c = tl.program_id(0)
    
    # Compute output spatial indices
    pid_h = tl.program_id(2) // (out_w // BLOCK_W)
    pid_w = tl.program_id(2) % (out_w // BLOCK_W)
    
    # Compute output tensor offsets for this block
    out_offsets_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    out_offsets_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    
    # Create masks for valid output indices
    out_h_mask = out_offsets_h < out_h
    out_w_mask = out_offsets_w < out_w
    out_mask = out_h_mask[:, None] & out_w_mask[None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for k in range(0, in_channels, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < in_channels
        
        # Loop over kernel height
        for kh in range(kernel_h):
            # Compute input height index with dilation
            in_h = pid_h * stride_h - pad_h + kh * dilation_h
            
            # Loop over kernel width
            for kw in range(kernel_w):
                # Compute input width index with dilation
                in_w = pid_w * stride_w - pad_w + kw * dilation_w
                
                # Create masks for valid input indices
                in_h_mask = (in_h >= 0) & (in_h < height)
                in_w_mask = (in_w >= 0) & (in_w < width)
                
                # Load input data if valid
                if in_h_mask and in_w_mask:
                    x_offsets = (
                        pid_batch * (in_channels * height * width) +
                        k_offsets[:, None, None] * (height * width) +
                        in_h * width +
                        in_w
                    )
                    x_mask = k_mask[:, None, None] & out_mask
                    x_block = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
                else:
                    x_block = tl.zeros((BLOCK_SIZE_K, BLOCK_H, BLOCK_W), dtype=tl.float32)
                
                # Load weight data
                w_offsets = (
                    pid_out_c * (in_channels * kernel_h * kernel_w) +
                    k_offsets[:, None, None] * (kernel_h * kernel_w) +
                    kh * kernel_w +
                    kw
                )
                w_mask = k_mask[:, None, None]
                w_val = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)
                
                # Accumulate convolution
                acc += x_block * w_val[None, :, :, :]
    
    # Convert accumulator to output dtype
    acc = acc.to(out_ptr.dtype.element_ty)
    
    # Reduce across channel dimension (sum over in_channels)
    acc = tl.sum(acc, axis=1)  # Shape: (BLOCK_SIZE_M, BLOCK_H, BLOCK_W)
    
    # Add bias if present
    if b_ptr is not None:
        b_offsets = pid_out_c
        b_val = tl.load(b_ptr + b_offsets)
        acc += b_val
    
    # Store output
    out_offsets = (
        pid_batch * (out_channels * out_h * out_w) +
        pid_out_c * (out_h * out_w) +
        out_offsets_h[:, None] * out_w +
        out_offsets_w[None, :]
    )
    tl.store(out_ptr + out_offsets, acc, mask=out_mask)


def triton_conv2d(x, weight, bias, stride, padding, dilation, groups):
    """
    Custom Triton implementation of 2D convolution.
    Note: This implementation assumes groups=1 for simplicity (standard conv).
    For depthwise conv (groups=in_channels), a different kernel would be needed.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Compute output dimensions
    stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride)
    pad_h, pad_w = padding if isinstance(padding, tuple) else (padding, padding)
    dilation_h, dilation_w = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    
    out_h = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    
    # Create output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Set block sizes (tunable parameters)
    BLOCK_SIZE_M = 32  # Output channels per block
    BLOCK_SIZE_N = 1   # Batch size per block (fixed at 1 for simplicity)
    BLOCK_SIZE_K = 8   # Input channels per block
    BLOCK_H = 8        # Output height per block
    BLOCK_W = 8        # Output width per block
    
    # Compute grid dimensions
    grid = (
        triton.cdiv(out_channels, BLOCK_SIZE_M),  # Number of blocks for output channels
        batch_size,                               # Number of blocks for batch
        (triton.cdiv(out_h, BLOCK_H) * triton.cdiv(out_w, BLOCK_W))  # Number of spatial blocks
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        height, width, out_h, out_w,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the Model class using custom Triton convolution kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Validate groups parameter
        if self.groups != 1:
            raise NotImplementedError("Triton kernel only supports groups=1 for now")
        
        # Use custom Triton convolution
        return triton_conv2d(x, self.weight, self.bias, 
                            self.stride, self.padding, self.dilation, 
                            self.groups)