import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv1d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, length)
    w_ptr,  # Weight tensor: (out_channels, in_channels, kernel_size)
    b_ptr,  # Bias tensor: (out_channels,)
    out_ptr,  # Output tensor: (batch, out_channels, output_length)
    batch_size, in_channels, out_channels, input_length, output_length, kernel_size,
    stride: tl.constexpr, padding: tl.constexpr, dilation: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,  # Batch size per block
    BLOCK_SIZE_N: tl.constexpr,  # Output channels per block
    BLOCK_SIZE_K: tl.constexpr,  # Input channels per block
    BLOCK_SIZE_L: tl.constexpr,  # Output positions per block
    BLOCK_SIZE_KW: tl.constexpr,  # Kernel width per block
):
    # Get program IDs
    pid_m = tl.program_id(0)  # Batch index
    pid_n = tl.program_id(1)  # Output channel index
    pid_l = tl.program_id(2)  # Output position index

    # Compute offsets
    batch_idx = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    out_channel_idx = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_pos = pid_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    
    # Create masks
    batch_mask = batch_idx < batch_size
    out_channel_mask = out_channel_idx < out_channels
    out_pos_mask = out_pos < output_length
    
    # Reshape masks for broadcasting
    batch_mask = batch_mask[:, None, None]
    out_channel_mask = out_channel_mask[None, :, None]
    out_pos_mask = out_pos_mask[None, None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_L), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for k in range(0, in_channels, BLOCK_SIZE_K):
        channel_offset = k + tl.arange(0, BLOCK_SIZE_K)
        channel_mask = channel_offset < in_channels
        channel_mask = channel_mask[None, None, :]
        
        # Load input: (batch, channel, pos)
        # Need to handle padding for input positions
        for kw in range(0, kernel_size, BLOCK_SIZE_KW):
            kernel_weight_offset = kw + tl.arange(0, BLOCK_SIZE_KW)
            kernel_weight_mask = kernel_weight_offset < kernel_size
            
            # Calculate input positions for each output position and kernel position
            # Input position = output_position * stride + kernel_position * dilation - padding
            input_pos = (out_pos[None, None, :] * stride + 
                        kernel_weight_offset[None, :, None] * dilation - 
                        padding)
            
            # Create input position mask
            input_pos_mask = (input_pos >= 0) & (input_pos < input_length)
            
            # Load input data with padding handling
            # We need to handle out-of-bounds positions by setting them to 0
            x = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_L), dtype=tl.float32)
            
            # Calculate indices for valid positions
            x_indices = batch_idx[:, None, None] * (in_channels * input_length) + \
                       channel_offset[None, :, None] * input_length + \
                       input_pos[None, None, :]
            
            # Load input with masking
            x_valid = tl.load(
                x_ptr + x_indices,
                mask=batch_mask & channel_mask & input_pos_mask,
                other=0.0
            )
            x = tl.where(batch_mask & channel_mask & input_pos_mask, x_valid, 0.0)
            
            # Load kernel weights: (out_channel, in_channel, kernel_pos)
            w_indices = out_channel_idx[None, :, None] * (in_channels * kernel_size) + \
                       channel_offset[None, None, :] * kernel_size + \
                       kernel_weight_offset[None, None, :]
            
            w = tl.load(
                w_ptr + w_indices,
                mask=out_channel_mask & channel_mask & kernel_weight_mask,
                other=0.0
            )
            
            # Accumulate convolution: sum over channel and kernel dimensions
            # x: (batch, channel, output_pos)
            # w: (out_channel, channel, kernel_pos)
            # Need to broadcast and multiply
            acc += tl.sum(x[None, :, :, :] * w[:, :, None, :], axis=2)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_channel_idx, mask=out_channel_mask)
        acc += bias[None, :, None]
    
    # Store output
    out_indices = (batch_idx[:, None, None] * (out_channels * output_length) + 
                  out_channel_idx[None, :, None] * output_length + 
                  out_pos[None, None, :])
    
    tl.store(
        out_ptr + out_indices,
        acc,
        mask=batch_mask & out_channel_mask & out_pos_mask
    )


def triton_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                  stride: int = 1, padding: int = 0, dilation: int = 1):
    """
    Optimized Triton implementation of 1D convolution.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, length)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_size)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride of convolution
        padding: Padding applied to input
        dilation: Spacing between kernel elements
    
    Returns:
        Output tensor of shape (batch_size, out_channels, output_length)
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, input_length = x.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, output_length, device=x.device, dtype=x.dtype)
    
    # Tunable parameters for block sizes
    BLOCK_SIZE_M = 8    # Batch size per block
    BLOCK_SIZE_N = 16   # Output channels per block
    BLOCK_SIZE_K = 8    # Input channels per block
    BLOCK_SIZE_L = 128  # Output positions per block
    BLOCK_SIZE_KW = 4   # Kernel width per block
    
    # Determine grid dimensions
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (out_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_l = (output_length + BLOCK_SIZE_L - 1) // BLOCK_SIZE_L
    
    grid = (grid_m, grid_n, grid_l)
    
    # Launch kernel
    conv1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, input_length, output_length, kernel_size,
        stride, padding, dilation,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
        BLOCK_SIZE_KW=BLOCK_SIZE_KW,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized 1D convolution using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        
        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs optimized 1D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        # Ensure input is on CUDA and contiguous
        x = x.cuda().contiguous() if x.is_cuda else x.contiguous()
        
        # Perform convolution using Triton kernel
        return triton_conv1d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding, dilation=self.dilation
        )