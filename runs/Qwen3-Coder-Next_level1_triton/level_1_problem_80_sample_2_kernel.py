import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor
    w_ptr,  # Weight tensor
    b_ptr,  # Bias tensor (can be None)
    y_ptr,  # Output tensor
    batch_size, in_channels, out_channels,
    height, width,
    kernel_h, kernel_w,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    out_h, out_w,
    BLOCK_SIZE_M: tl.constexpr,  # Output tiles in height dimension
    BLOCK_SIZE_N: tl.constexpr,  # Output tiles in width dimension
    BLOCK_SIZE_K: tl.constexpr   # Channel dimension
):
    # Get program IDs for output tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate starting position for this block
    rm = pid_m * BLOCK_SIZE_M
    rn = pid_n * BLOCK_SIZE_N
    
    # Create output offsets
    output_offsets = (
        tl.arange(0, BLOCK_SIZE_M)[:, None] * out_w * out_channels +
        tl.arange(0, BLOCK_SIZE_N)[None, :] * out_channels +
        tl.arange(0, out_channels)[None, :]
    )
    output_mask = (
        (rm + tl.arange(0, BLOCK_SIZE_M)[:, None]) < out_h &
        (rn + tl.arange(0, BLOCK_SIZE_N)[None, :]) < out_w
    )
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N, out_channels), dtype=tl.float32)
    
    # Iterate over input channels and kernel dimensions
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Calculate input positions for this kernel element
            input_h = rm * stride_h - pad_h + kh * dil_h
            input_w = rn * stride_w - pad_w + kw * dil_w
            
            # Check if input positions are valid
            valid_h = (input_h >= 0) & (input_h < height)
            valid_w = (input_w >= 0) & (input_w < width)
            valid_mask = valid_h & valid_w
            
            # Calculate input offsets for all batch elements
            input_offsets = (
                tl.arange(0, batch_size)[:, None, None, None] * (height * width * in_channels) +
                (input_h * width + input_w) * in_channels +
                tl.arange(0, in_channels)[None, None, None, :]
            )
            
            # Load input data
            x = tl.load(x_ptr + input_offsets, mask=valid_mask[:, None, None, None], other=0.0)
            
            # Load weight for this kernel position
            weight_offsets = (
                tl.arange(0, out_channels)[None, None, None, :] +
                kh * kernel_w * out_channels +
                kw * out_channels +
                tl.arange(0, in_channels)[None, None, :, None]
            )
            w = tl.load(w_ptr + weight_offsets)
            
            # Compute accumulation: x * w -> accumulator
            # x shape: (batch, BLOCK_SIZE_M, BLOCK_SIZE_N, in_channels)
            # w shape: (out_channels, in_channels)
            # Result should be: (batch, BLOCK_SIZE_M, BLOCK_SIZE_N, out_channels)
            
            # Reshape for matrix multiplication
            x_reshaped = tl.reshape(x, (batch_size * BLOCK_SIZE_M * BLOCK_SIZE_N, in_channels))
            w_reshaped = tl.reshape(w, (in_channels, out_channels))
            
            # Matrix multiplication
            acc = tl.dot(x_reshaped, w_reshaped)
            acc = tl.reshape(acc, (batch_size, BLOCK_SIZE_M, BLOCK_SIZE_N, out_channels))
            
            # Accumulate
            accumulator += acc.to(tl.float32)
    
    # Add bias if present
    if b_ptr is not None:
        bias_offsets = tl.arange(0, out_channels)
        bias = tl.load(b_ptr + bias_offsets)
        accumulator += bias[None, None, None, :]
    
    # Store output
    y = accumulator.to(y_ptr.type.element_ty)
    tl.store(y_ptr + output_offsets, y, mask=output_mask)


def triton_conv2d(x, weight, bias, stride, padding, dilation):
    """Triton implementation of 2D convolution."""
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride)
    pad_h, pad_w = padding if isinstance(padding, tuple) else (padding, padding)
    dil_h, dil_w = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    
    # Calculate output dimensions
    out_h = (height + 2 * pad_h - dil_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (width + 2 * pad_w - dil_w * (kernel_w - 1) - 1) // stride_w + 1
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Allocate output tensor
    y = torch.empty(batch_size, out_channels, out_h, out_w, device=x.device, dtype=x.dtype)
    
    # Set block sizes for optimization
    BLOCK_SIZE_M = 4  # Output tiles in height
    BLOCK_SIZE_N = 8  # Output tiles in width
    BLOCK_SIZE_K = in_channels  # Channel dimension
    
    # Calculate grid dimensions
    grid = (
        triton.cdiv(out_h, BLOCK_SIZE_M),
        triton.cdiv(out_w, BLOCK_SIZE_N),
        1  # batch dimension
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, y,
        batch_size, in_channels, out_channels,
        height, width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        out_h, out_w,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized Model with Triton kernel for 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize weights using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using the Triton kernel.
        """
        return triton_conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation
        )