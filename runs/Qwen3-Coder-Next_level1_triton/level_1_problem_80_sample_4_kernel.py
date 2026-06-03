import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel for 2D convolution with dilation and padding support
@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, H, W)
    w_ptr,  # Weight tensor: (out_channels, in_channels, kH, kW)
    b_ptr,  # Bias tensor: (out_channels,) or None
    y_ptr,  # Output tensor: (batch, out_channels, H_out, W_out)
    batch_size, in_channels, out_channels,
    in_h, in_w, k_h, k_w,
    out_h, out_w,
    stride_h, stride_w,
    pad_h_top, pad_h_bottom, pad_w_left, pad_w_right,
    dil_h, dil_w,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output rows
    BLOCK_SIZE_N: tl.constexpr,  # Block size for output cols
    BLOCK_SIZE_K: tl.constexpr,  # Block size for reduction (in_channels)
):
    # Program IDs for output tensor
    pid_b = tl.program_id(0)  # batch index
    pid_m = tl.program_id(1)  # output row index
    pid_n = tl.program_id(2)  # output column index
    
    # Calculate output coordinates
    out_row = pid_m * BLOCK_SIZE_M
    out_col = pid_n * BLOCK_SIZE_N
    
    # Compute offsets for output tensor
    y_offsets = (pid_b * out_h * out_w * out_channels + 
                 tl.arange(0, BLOCK_SIZE_M)[:, None] * out_w * out_channels + 
                 tl.arange(0, BLOCK_SIZE_N)[None, :] * out_channels)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over channels and kernel positions
    for c in range(in_channels):
        # Loop over kernel height
        for kh in range(k_h):
            # Calculate input row with padding and dilation
            in_row = out_row * stride_h - pad_h_top + kh * dil_h
            
            # Loop over kernel width
            for kw in range(k_w):
                # Calculate input column with padding and dilation
                in_col = out_col * stride_w - pad_w_left + kw * dil_w
                
                # Check if input position is valid (within padded input)
                valid_row = (in_row >= 0) & (in_row < in_h)
                valid_col = (in_col >= 0) & (in_col < in_w)
                valid = valid_row & valid_col
                
                # Compute offsets for input tensor
                x_offsets = (pid_b * in_h * in_w * in_channels + 
                            in_row * in_w * in_channels + 
                            in_col * in_channels + 
                            c)
                
                # Load input value (0 if out of bounds)
                x_val = tl.load(x_ptr + x_offsets, mask=valid, other=0.0)
                
                # Compute offsets for weight tensor
                w_offsets = (tl.arange(0, BLOCK_SIZE_M)[:, None] * k_h * k_w * in_channels +
                            tl.arange(0, BLOCK_SIZE_N)[None, :] * k_w * in_channels +
                            kh * k_w * in_channels +
                            kw * in_channels +
                            c)
                
                # Load weight values
                w_val = tl.load(w_ptr + w_offsets)
                
                # Accumulate
                acc += x_val * w_val
    
    # Add bias if available
    if b_ptr is not None:
        b = tl.load(b_ptr + tl.arange(0, BLOCK_SIZE_N))
        acc += b[None, :]
    
    # Store result (convert to float32 for output)
    tl.store(y_ptr + y_offsets, acc.to(y_ptr.dtype.element_ty), 
             mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < out_h) & 
                  (tl.arange(0, BLOCK_SIZE_N)[None, :] < out_w))


def triton_conv2d(x, weight, bias, stride, padding, dilation):
    """
    Perform 2D convolution using Triton kernel.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    batch_size, in_channels, in_h, in_w = x.shape
    out_channels, _, k_h, k_w = weight.shape
    
    # Calculate output dimensions
    stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride)
    dil_h, dil_w = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    
    # Handle padding
    if isinstance(padding, tuple):
        if len(padding) == 2:
            pad_h_top = pad_h_bottom = padding[0]
            pad_w_left = pad_w_right = padding[1]
        elif len(padding) == 4:
            pad_h_top, pad_h_bottom, pad_w_left, pad_w_right = padding
        else:
            raise ValueError("Padding must be a tuple of 2 or 4 elements")
    else:
        pad_h_top = pad_h_bottom = pad_w_left = pad_w_right = padding
    
    # Calculate output dimensions
    out_h = (in_h + pad_h_top + pad_h_bottom - dil_h * (k_h - 1) - 1) // stride_h + 1
    out_w = (in_w + pad_w_left + pad_w_right - dil_w * (k_w - 1) - 1) // stride_w + 1
    
    # Create output tensor
    y = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Set block sizes for kernel
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32  # Can be tuned
    
    # Calculate grid dimensions
    grid = (batch_size, 
            triton.cdiv(out_h, BLOCK_SIZE_M), 
            triton.cdiv(out_w, BLOCK_SIZE_N))
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, y,
        batch_size, in_channels, out_channels,
        in_h, in_w, k_h, k_w,
        out_h, out_w,
        stride_h, stride_w,
        pad_h_top, pad_h_bottom, pad_w_left, pad_w_right,
        dil_h, dil_w,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of the 2D convolution model using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), 
                 bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel.
        """
        return triton_conv2d(x, self.weight, self.bias, 
                            self.stride, self.padding, self.dilation)