import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    # Pointers to inputs and outputs
    x_ptr,                # Input tensor (batch, in_channels, H, W)
    w_ptr,                # Weight tensor (out_channels, in_channels, kH, kW)
    b_ptr,                # Bias tensor (out_channels,) - can be None
    output_ptr,           # Output tensor (batch, out_channels, H_out, W_out)
    # Dimensions
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    in_h: tl.constexpr,
    in_w: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    k_h: tl.constexpr,
    k_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,  # Batch * out_h * out_w block size
    BLOCK_N: tl.constexpr,  # out_channels block size
    BLOCK_K: tl.constexpr,  # in_channels * k_h * k_w block size
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate total number of output positions (batch * out_h * out_w)
    total_out_positions = batch_size * out_h * out_w
    
    # Calculate batch, out_h, out_w from pid_m
    position_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    batch_idx = position_idx // (out_h * out_w)
    temp = position_idx % (out_h * out_w)
    out_h_idx = temp // out_w
    out_w_idx = temp % out_w
    
    # Create masks for valid positions
    mask_m = position_idx < total_out_positions
    
    # Compute the starting position in input for each output position
    # Account for padding, stride, and dilation
    in_h_start = out_h_idx * stride_h - pad_h
    in_w_start = out_w_idx * stride_w - pad_w
    
    # Output pointer offset for this block
    output_offsets = position_idx * out_h * out_w + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) * out_h * out_w
    output_mask = mask_m[:, None] & (tl.arange(0, BLOCK_N)[None, :] < out_channels)
    
    # Accumulator for the convolution result
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over in_channels and kernel dimensions
    for ic in range(in_channels):
        for kh in range(k_h):
            for kw in range(k_w):
                # Compute input position
                in_h_pos = in_h_start + kh * dilation_h
                in_w_pos = in_w_start + kw * dilation_w
                
                # Check if within input bounds
                valid_h = (in_h_pos >= 0) & (in_h_pos < in_h)
                valid_w = (in_w_pos >= 0) & (in_w_pos < in_w)
                valid = valid_h & valid_w
                
                # Calculate input offset
                input_offsets = batch_idx * (in_channels * in_h * in_w) + \
                               ic * (in_h * in_w) + \
                               in_h_pos * in_w + \
                               in_w_pos
                
                # Load input value (0 if out of bounds)
                x_val = tl.load(x_ptr + input_offsets, mask=mask_m & valid, other=0.0)
                
                # Load weight value
                weight_offsets = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) * (in_channels * k_h * k_w) + \
                                ic * (k_h * k_w) + \
                                kh * k_w + \
                                kw
                w_val = tl.load(w_ptr + weight_offsets, mask=(tl.arange(0, BLOCK_N) < out_channels), other=0.0)
                
                # Accumulate
                accumulator += x_val[:, None] * w_val[None, :]
    
    # Convert accumulator to float16 if needed for output
    accumulator = accumulator.to(output_ptr.dtype.element_ty)
    
    # Add bias if provided
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_n * BLOCK_N + tl.arange(0, BLOCK_N), mask=tl.arange(0, BLOCK_N) < out_channels)
        accumulator += bias[None, :]
    
    # Store output
    tl.store(output_ptr + output_offsets, accumulator, mask=output_mask)


def triton_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Triton implementation of 2D convolution using im2col + GEMM approach.
    Note: This simplified version only supports groups=1 for the kernel implementation.
    For groups > 1, we'd need additional logic or multiple kernel launches.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Extract dimensions
    batch_size, in_channels, in_h, in_w = x.shape
    out_channels, _, k_h, k_w = weight.shape
    
    # Calculate output dimensions
    out_h = (in_h + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, out_h, out_w, device=x.device, dtype=x.dtype)
    
    # Grid configuration
    total_out_positions = batch_size * out_h * out_w
    BLOCK_M = 128  # Tunable
    BLOCK_N = 64   # Tunable
    BLOCK_K = 32   # Tunable
    
    grid = (
        triton.cdiv(total_out_positions, BLOCK_M),
        triton.cdiv(out_channels, BLOCK_N),
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, output,
        batch_size, in_channels, out_channels,
        in_h, in_w, out_h, out_w,
        k_h, k_w,
        stride, stride,
        padding, padding,
        dilation, dilation,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
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
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check if groups > 1 - our current kernel only supports groups=1
        if self.groups != 1:
            # Fall back to PyTorch implementation for grouped convolutions
            return nn.functional.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        # Use our Triton kernel for standard convolution
        return triton_conv2d(x, self.weight, self.bias, 
                            stride=self.stride, 
                            padding=self.padding, 
                            dilation=self.dilation,
                            groups=self.groups)