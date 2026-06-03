import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_im2col_kernel(
    x_ptr,  # Input tensor (B, C_in, L)
    w_ptr,  # Weight tensor (C_out, C_in, K)
    b_ptr,  # Bias tensor (C_out,)
    out_ptr,  # Output tensor (B, C_out, L_out)
    B, C_in, L,  # Input dimensions
    C_out, K,  # Weight dimensions
    stride, padding, dilation,  # Conv parameters
    L_out,  # Output length
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for output length
    BLOCK_SIZE_K: tl.constexpr,  # Block size for computation
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    batch_id = tl.program_id(2)  # Process one batch at a time per kernel launch
    
    # Calculate output channel and position ranges
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create mask for valid channels and positions
    mask_m = offsets_m < C_out
    mask_n = offsets_n < L_out
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over input channels and kernel size
    for c in range(C_in):
        for k in range(K):
            # Calculate input position with dilation and padding
            input_pos = offsets_n * stride + k * dilation - padding
            mask_pos = (input_pos >= 0) & (input_pos < L)
            
            # Calculate input pointer offset for this batch, channel, and position
            # Input layout: (batch, channel, length)
            input_offset = batch_id * C_in * L + c * L + input_pos
            
            # Load input values
            x_vals = tl.load(x_ptr + input_offset, mask=mask_pos & (input_pos < L), other=0.0)
            
            # Load weight value
            w_offset = c * K + k
            w_vals = tl.load(w_ptr + offsets_m * (C_in * K) + w_offset, mask=mask_m)
            
            # Accumulate: x * w
            acc += x_vals[None, :] * w_vals[:, None]
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + offsets_m, mask=mask_m)
        acc += bias[:, None]
    
    # Store result
    out_offset = batch_id * C_out * L_out + offsets_m[:, None] * L_out + offsets_n[None, :]
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offset, acc, mask=out_mask)


@triton.jit
def conv1d_im2col_kernel_fused(
    x_ptr,  # Input tensor (B, C_in, L)
    w_ptr,  # Weight tensor (C_out, C_in, K)
    b_ptr,  # Bias tensor (C_out,)
    out_ptr,  # Output tensor (B, C_out, L_out)
    B, C_in, L,  # Input dimensions
    C_out, K,  # Weight dimensions
    stride, padding, dilation,  # Conv parameters
    L_out,  # Output length
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    batch_id = tl.program_id(2)
    
    # Calculate output channel and position ranges
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create mask for valid channels and positions
    mask_m = offsets_m < C_out
    mask_n = offsets_n < L_out
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over input channels and kernel size
    for c in range(C_in):
        for k in range(K):
            # Calculate input position with dilation and padding
            input_pos = offsets_n * stride + k * dilation - padding
            mask_pos = (input_pos >= 0) & (input_pos < L)
            
            # Calculate input pointer offset for this batch, channel, and position
            input_offset = batch_id * C_in * L + c * L + input_pos
            
            # Load input values
            x_vals = tl.load(x_ptr + input_offset, mask=mask_pos & (input_pos < L), other=0.0)
            
            # Load weight value
            w_offset = c * K + k
            w_vals = tl.load(w_ptr + offsets_m * (C_in * K) + w_offset, mask=mask_m)
            
            # Accumulate: x * w
            acc += x_vals[None, :] * w_vals[:, None]
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + offsets_m, mask=mask_m)
        acc += bias[:, None]
    
    # Store result
    out_offset = batch_id * C_out * L_out + offsets_m[:, None] * L_out + offsets_n[None, :]
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offset, acc, mask=out_mask)


def triton_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
):
    """
    Triton implementation of 1D convolution using im2col approach.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, length)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_size)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride of convolution
        padding: Padding applied to input
        dilation: Spacing between kernel elements
        groups: Number of blocked connections (must be 1 for this implementation)
    """
    assert groups == 1, "Triton conv1d only supports groups=1"
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    
    # Ensure contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Get dimensions
    B, C_in, L = x.shape
    C_out, _, K = weight.shape
    
    # Calculate output length
    L_out = (L + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((B, C_out, L_out), dtype=x.dtype, device=x.device)
    
    # Set block sizes for optimization
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 8
    
    # Calculate grid dimensions
    # We use batch dimension as the third dimension for parallelization
    grid = (
        (C_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (L_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
        B,
    )
    
    # Launch kernel
    conv1d_im2col_kernel_fused[grid](
        x, weight, bias, out,
        B, C_in, L,
        C_out, K,
        stride, padding, dilation,
        L_out,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized 1D convolution using Triton kernels.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Store convolution parameters
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights (Xavier/Glorot initialization)
        fan_in = in_channels * kernel_size
        fan_out = out_channels * kernel_size
        std = torch.sqrt(torch.tensor(2.0 / (fan_in + fan_out)))
        with torch.no_grad():
            self.weight.normal_(0, std)
            if bias:
                self.bias.zero_()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 1D convolution using Triton kernel.
        """
        return triton_conv1d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )