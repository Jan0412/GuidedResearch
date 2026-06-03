import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Input tensor (B, C_in, H, W)
    w_ptr,  # Weight tensor (C_in, C_out, K_h, K_w)
    b_ptr,  # Bias tensor (C_out,)
    y_ptr,  # Output tensor (B, C_out, H_out, W_out)
    batch_size, in_channels, out_channels,
    in_h, in_w, out_h, out_w,
    k_h, k_w,
    stride_h, stride_w,
    pad_h, pad_w,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for output spatial
    BLOCK_SIZE_K: tl.constexpr,  # Block size for input channels
):
    # Output tensor indices
    pid_batch = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Compute output position
    c_out_start = pid_c_out * BLOCK_SIZE_M
    h_start = pid_h * BLOCK_SIZE_N
    w_start = pid_w * BLOCK_SIZE_N
    
    # Create offset arrays for output
    c_out_offsets = c_out_start + tl.arange(0, BLOCK_SIZE_M)
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE_N)
    w_offsets = w_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for output bounds
    c_out_mask = c_out_offsets < out_channels
    h_mask = h_offsets < out_h
    w_mask = w_offsets < out_w
    
    # Initialize accumulator for output
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over input channels
    for c_in in range(in_channels):
        # Calculate corresponding input position for each output position
        # For transposed convolution: out_pos = in_pos * stride + (kernel_pos - padding)
        # So: in_pos = (out_pos - (kernel_pos - padding)) // stride
        
        # Kernel position offsets
        kh_offsets = tl.arange(0, k_h)
        kw_offsets = tl.arange(0, k_w)
        
        kh_grid, kw_grid = tl.meshgrid(kh_offsets, kw_offsets)
        kh_grid = kh_grid.flatten()
        kw_grid = kw_grid.flatten()
        
        # Compute input positions for this kernel position
        in_h_pos = (h_offsets[:, None, None] - (kh_grid[None, :, None] - pad_h)) // stride_h
        in_w_pos = (w_offsets[None, :, :] - (kw_grid[None, None, :] - pad_w)) // stride_w
        
        # Check if input positions are valid
        valid_mask = (in_h_pos >= 0) & (in_h_pos < in_h) & (in_w_pos >= 0) & (in_w_pos < in_w)
        
        # Load input values where valid
        in_h_valid = tl.where(valid_mask, in_h_pos, 0)
        in_w_valid = tl.where(valid_mask, in_w_pos, 0)
        
        # Compute input tensor offsets
        # Input layout: (B, C_in, H, W)
        in_batch_offset = pid_batch * in_channels * in_h * in_w
        in_c_offset = c_in * in_h * in_w
        in_offsets = in_batch_offset + in_c_offset + in_h_valid * in_w + in_w_valid
        
        # Load input values
        x_val = tl.load(x_ptr + in_offsets, mask=valid_mask, other=0.0)
        
        # Load corresponding weight values
        # Weight layout: (C_in, C_out, K_h, K_w)
        w_batch_offset = c_in * out_channels * k_h * k_w
        w_c_out_offset = c_out_offsets[:, None, None] * k_h * k_w
        w_kh_offset = kh_grid[None, :, None]
        w_kw_offset = kw_grid[None, None, :]
        w_offsets_full = w_batch_offset + w_c_out_offset + w_kh_offset + w_kw_offset
        
        # Load weights
        w_val = tl.load(w_ptr + w_offsets_full, mask=c_out_mask[:, None, None], other=0.0)
        
        # Accumulate: output += input * weight
        acc += x_val[None, :, :] * w_val
        
    # Add bias if present
    if b_ptr is not None:
        b_offsets = c_out_offsets
        b_val = tl.load(b_ptr + b_offsets, mask=c_out_mask, other=0.0)
        acc += b_val[:, None, None]
    
    # Store output
    y_batch_offset = pid_batch * out_channels * out_h * out_w
    y_c_out_offset = c_out_offsets[:, None, None] * out_h * out_w
    y_h_offset = h_offsets[None, :, None]
    y_w_offset = w_offsets[None, None, :]
    y_offsets = y_batch_offset + y_c_out_offset + y_h_offset * out_w + y_w_offset
    
    # Store result with proper masking
    tl.store(y_ptr + y_offsets, acc, mask=c_out_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :])


def triton_conv_transpose2d(x, weight, bias=None, stride=(1, 1), padding=(0, 0), output_padding=(0, 0), groups=1):
    """
    Triton implementation of 2D transposed convolution.
    
    Args:
        x: Input tensor (B, C_in, H, W)
        weight: Weight tensor (C_in, C_out, K_h, K_w)
        bias: Optional bias tensor (C_out,)
        stride: Stride (H, W)
        padding: Padding (H, W)
        output_padding: Output padding (H, W)
        groups: Number of groups (must be 1 for this implementation)
    """
    # Get dimensions
    batch_size, in_channels, in_h, in_w = x.shape
    _, out_channels, k_h, k_w = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    
    # Calculate output dimensions
    out_h = (in_h - 1) * stride_h - 2 * pad_h + k_h + output_padding[0]
    out_w = (in_w - 1) * stride_w - 2 * pad_w + k_w + output_padding[1]
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Allocate output tensor
    y = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Grid configuration
    # We'll use a 4D grid: (batch, output_channels, output_h, output_w)
    # For efficiency, we'll use tiling
    
    BLOCK_SIZE_M = 8  # Output channels per block
    BLOCK_SIZE_N = 8  # Spatial dimensions per block
    BLOCK_SIZE_K = 16  # Input channels per block (not used directly but for organization)
    
    # Calculate grid dimensions
    grid_batch = batch_size
    grid_c_out = (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_h = (out_h + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_w = (out_w + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    conv_transpose2d_kernel[grid_batch, grid_c_out, grid_h, grid_w](
        x, weight, bias, y,
        batch_size, in_channels, out_channels,
        in_h, in_w, out_h, out_w,
        k_h, k_w,
        stride_h, stride_w,
        pad_h, pad_w,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernels for transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same parameters as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_flag = bias
        
        # Initialize weight and bias (same as nn.ConvTranspose2d)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize weights and bias similar to nn.ConvTranspose2d."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding, output_padding=(0, 0), groups=1
        )