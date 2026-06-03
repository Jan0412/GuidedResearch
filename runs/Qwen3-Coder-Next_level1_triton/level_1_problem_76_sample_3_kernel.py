import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr,  # Input tensor: (batch_size, in_channels, length)
    w_ptr,  # Weight tensor: (out_channels, in_channels, kernel_size)
    b_ptr,  # Bias tensor: (out_channels,)
    out_ptr,  # Output tensor: (batch_size, out_channels, output_length)
    batch_size, in_channels, out_channels, length, output_length, kernel_size,
    stride, dilation,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch size
    BLOCK_SIZE_K: tl.constexpr,  # Block size for kernel elements
):
    # Get program IDs
    pid_m = tl.program_id(0)  # Output channel block index
    pid_n = tl.program_id(1)  # Batch block index
    pid_k = tl.program_id(2)  # Not used in this implementation
    
    # Calculate output channel range for this program
    out_channel_start = pid_m * BLOCK_SIZE_M
    out_channel_offsets = out_channel_start + tl.arange(0, BLOCK_SIZE_M)
    out_channel_mask = out_channel_offsets < out_channels
    
    # Calculate batch range for this program
    batch_start = pid_n * BLOCK_SIZE_N
    batch_offsets = batch_start + tl.arange(0, BLOCK_SIZE_N)
    batch_mask = batch_offsets < batch_size
    
    # Create meshgrid for batch and output channel indices
    batch_idx, out_channel_idx = tl.meshgrid(batch_offsets, out_channel_offsets)
    batch_idx = batch_idx.T
    out_channel_idx = out_channel_idx.T
    
    # Apply masks
    batch_mask = batch_mask[None, :] & batch_mask[:, None]
    out_channel_mask = out_channel_mask[None, :] & out_channel_mask[:, None]
    combined_mask = batch_mask & out_channel_mask
    
    # Initialize accumulator for output
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    
    # Loop over input channels
    for in_channel in range(in_channels):
        # Loop over kernel positions
        for k in range(0, kernel_size, BLOCK_SIZE_K):
            kernel_offset = k + tl.arange(0, BLOCK_SIZE_K)
            kernel_mask = kernel_offset < kernel_size
            
            # Compute input position for this kernel element
            # output_length = ((length - dilation * (kernel_size - 1) - 1) // stride) + 1
            # For output position j, input position = j * stride + dilation * k
            for out_pos in range(0, output_length, BLOCK_SIZE_M):
                out_pos_offset = out_pos + tl.arange(0, BLOCK_SIZE_M)
                out_pos_mask = out_pos_offset < output_length
                
                # Compute input positions: j * stride + dilation * k
                input_pos = out_pos_offset[None, :] * stride + dilation * kernel_offset[:, None]
                input_pos_mask = (input_pos >= 0) & (input_pos < length)
                
                # Load input: (batch_size, in_channels, length)
                # We need to index into the correct batch and input channel
                x_indices = batch_idx[:, :, None] * (in_channels * length) + \
                           in_channel * length + input_pos
                x_indices = x_indices.reshape(BLOCK_SIZE_N * BLOCK_SIZE_K, BLOCK_SIZE_M)
                input_pos_mask = input_pos_mask.reshape(BLOCK_SIZE_N * BLOCK_SIZE_K, BLOCK_SIZE_M)
                
                # Reshape for loading
                x_offsets = x_indices
                x_mask = (x_offsets < batch_size * in_channels * length) & input_pos_mask
                
                # Load input values: shape (BLOCK_SIZE_N * BLOCK_SIZE_K, BLOCK_SIZE_M)
                x_val = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
                x_val = x_val.reshape(BLOCK_SIZE_N, BLOCK_SIZE_K, BLOCK_SIZE_M)
                
                # Load kernel weights: shape (out_channels, in_channels, kernel_size)
                # Need to get weights for the current output channels and kernel positions
                w_indices = out_channel_idx[:, :, None] * (in_channels * kernel_size) + \
                           in_channel * kernel_size + kernel_offset[None, :, None]
                w_indices = w_indices.reshape(BLOCK_SIZE_N * BLOCK_SIZE_K, BLOCK_SIZE_M)
                w_mask = (w_indices < out_channels * in_channels * kernel_size)
                
                w_val = tl.load(w_ptr + w_indices, mask=w_mask, other=0.0)
                w_val = w_val.reshape(BLOCK_SIZE_N, BLOCK_SIZE_K, BLOCK_SIZE_M)
                
                # Accumulate: batch_idx x kernel_pos x out_pos
                # acc shape: (BLOCK_SIZE_N, BLOCK_SIZE_M)
                # x_val shape: (BLOCK_SIZE_N, BLOCK_SIZE_K, BLOCK_SIZE_M)
                # w_val shape: (BLOCK_SIZE_N, BLOCK_SIZE_K, BLOCK_SIZE_M)
                acc += tl.sum(x_val * w_val, axis=1)
    
    # Add bias if provided
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_channel_idx, mask=out_channel_mask, other=0.0)
        acc += bias
    
    # Store result
    out_indices = batch_idx * (out_channels * output_length) + \
                 out_channel_idx * output_length
    out_offsets = out_indices
    out_mask = (out_offsets < batch_size * out_channels * output_length) & combined_mask
    
    tl.store(out_ptr + out_offsets, acc.to(x_ptr.dtype.element_ty), mask=out_mask)


def triton_conv1d(x, weight, bias=None, stride=1, dilation=1):
    """
    Triton implementation of 1D convolution.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, length)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_size)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride of convolution
        dilation: Dilation factor
    
    Returns:
        Output tensor of shape (batch_size, out_channels, output_length)
    """
    batch_size, in_channels, length = x.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (length - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Allocate output tensor
    out = torch.empty(batch_size, out_channels, output_length, device=x.device, dtype=x.dtype)
    
    # Set block sizes - tuned for typical GPU architectures
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 8
    BLOCK_SIZE_K = 16
    
    # Grid dimensions: (out_channels blocks, batch blocks, kernel_position blocks)
    grid = (
        triton.cdiv(out_channels, BLOCK_SIZE_M),
        triton.cdiv(batch_size, BLOCK_SIZE_N),
        1  # We handle kernel position in the kernel
    )
    
    # Launch kernel
    conv1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, length, output_length, kernel_size,
        stride, dilation,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized 1D convolution using Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize weights and bias similar to nn.Conv1d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.stride = stride
        self.dilation = dilation
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv1d(x, self.weight, self.bias, self.stride, self.dilation)


# Import math for initialization
import math