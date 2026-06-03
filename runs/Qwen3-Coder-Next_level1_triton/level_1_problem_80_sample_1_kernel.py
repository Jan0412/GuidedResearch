import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor pointer (batch, in_channels, H, W)
    w_ptr,  # Weight tensor pointer (out_channels, in_channels, kH, kW)
    b_ptr,  # Bias tensor pointer (out_channels) - can be None
    out_ptr,  # Output tensor pointer (batch, out_channels, H_out, W_out)
    batch_size, in_channels, out_channels,
    in_h, in_w, out_h, out_w,
    k_h, k_w,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output rows
    BLOCK_SIZE_N: tl.constexpr,  # Block size for output columns
    BLOCK_SIZE_K: tl.constexpr,  # Block size for reduction (in_channels)
):
    # Get program IDs for output grid
    pid_batch = tl.program_id(0)
    pid_out_h = tl.program_id(1)
    pid_out_w = tl.program_id(2)
    
    # Calculate output position
    out_h_idx = pid_out_h * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    out_w_idx = pid_out_w * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for valid indices
    mask_h = out_h_idx < out_h
    mask_w = out_w_idx < out_w
    mask = mask_h[:, None] & mask_w[None, :]
    
    # Initialize accumulator for output
    output = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Convolution loop over input channels and kernel dimensions
    for ic in range(in_channels):
        for kh in range(k_h):
            # Compute input H index with dilation and padding
            in_h_idx = pid_out_h * stride_h - pad_h + kh * dil_h
            
            # Check if this is a valid input position
            valid_in_h = (in_h_idx >= 0) & (in_h_idx < in_h)
            
            for kw in range(k_w):
                # Compute input W index with dilation and padding
                in_w_idx = pid_out_w * stride_w - pad_w + kw * dil_w
                
                # Check if this is a valid input position
                valid_in_w = (in_w_idx >= 0) & (in_w_idx < in_w)
                valid = valid_in_h[:, None] & valid_in_w[None, :]
                
                # Compute input tensor indices
                # Input shape: (batch, in_channels, in_h, in_w)
                # We need to gather data for all output positions and this kernel position
                
                # Calculate base offset for input
                in_offset = (pid_batch * in_channels * in_h * in_w + 
                            ic * in_h * in_w + 
                            in_h_idx[:, None] * in_w + 
                            in_w_idx[None, :])
                
                # Load input values with masking
                x_vals = tl.load(x_ptr + in_offset, mask=valid & mask, other=0.0)
                
                # Load weight values
                # Weight shape: (out_channels, in_channels, k_h, k_w)
                w_offset = (tl.arange(0, BLOCK_SIZE_M)[:, None, None] * 0 +  # Will be broadcast
                           ic * k_h * k_w + 
                           kh * k_w + 
                           kw)
                
                # Load weight (broadcast across output channels)
                w_val = tl.load(w_ptr + w_offset)
                
                # Accumulate: output += x * w
                output += tl.where(mask, x_vals * w_val, 0.0)
    
    # Add bias if provided
    if b_ptr is not None:
        bias_offset = tl.arange(0, BLOCK_SIZE_M)[:, None] * 0 +  # Broadcast across rows
        bias = tl.load(b_ptr + tl.arange(0, BLOCK_SIZE_M)[:, None], mask=mask_h[:, None])
        output += bias
    
    # Store output
    out_offset = (pid_batch * out_channels * out_h * out_w +
                 tl.arange(0, BLOCK_SIZE_M)[:, None] * out_h * out_w +
                 out_h_idx[:, None] * out_w +
                 out_w_idx[None, :])
    
    tl.store(out_ptr + out_offset, output.to(x_ptr.dtype.element_ty), mask=mask)


def triton_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                  stride=1, padding=(0, 0), dilation=(1, 1)) -> torch.Tensor:
    """
    Triton-based 2D convolution with support for dilation, padding, and asymmetric kernels.
    """
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    batch_size, in_channels, in_h, in_w = x.shape
    out_channels, _, k_h, k_w = weight.shape
    
    # Calculate output dimensions
    stride_h = stride_w = stride if isinstance(stride, int) else stride
    pad_h, pad_w = padding if isinstance(padding, tuple) else (padding, padding)
    dil_h, dil_w = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    
    out_h = (in_h + 2 * pad_h - dil_h * (k_h - 1) - 1) // stride_h + 1
    out_w = (in_w + 2 * pad_w - dil_w * (k_w - 1) - 1) // stride_w + 1
    
    # Create output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Define grid for kernel launch
    # Grid: (batch_size, out_h_blocks, out_w_blocks)
    BLOCK_SIZE_M = 8  # Output rows per block
    BLOCK_SIZE_N = 8  # Output columns per block
    
    grid = (batch_size, 
            (out_h + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
            (out_w + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        in_h, in_w, out_h, out_w,
        k_h, k_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=1  # We don't need this since we're looping explicitly
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model with Triton-based 2D convolution.
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
        self.bias_flag = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 2D convolution using Triton kernel.
        """
        return triton_conv2d(x, self.weight, self.bias,
                           stride=self.stride, 
                           padding=self.padding, 
                           dilation=self.dilation)