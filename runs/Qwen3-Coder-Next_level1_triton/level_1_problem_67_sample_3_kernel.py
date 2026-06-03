import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr,  # Input tensor: (batch_size, in_channels, length)
    w_ptr,  # Weight tensor: (out_channels, in_channels // groups, kernel_size)
    b_ptr,  # Bias tensor: (out_channels,)
    out_ptr,  # Output tensor: (batch_size, out_channels, out_length)
    batch_size, in_channels, out_channels, length, out_length,
    kernel_size, stride, padding, dilation, groups,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for length dimension
    BLOCK_SIZE_K: tl.constexpr,  # Block size for in_channels * kernel_size
):
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    pid_length = tl.program_id(2)
    
    # Calculate output position
    out_pos = pid_length * BLOCK_SIZE_N
    out_ch_start = pid_out_ch * BLOCK_SIZE_M
    
    # Offset for this batch
    x_batch_offset = pid_batch * in_channels * length
    
    # Create ranges for output channels
    out_ch_offsets = out_ch_start + tl.arange(0, BLOCK_SIZE_M)
    out_ch_mask = out_ch_offsets < out_channels
    
    # Create ranges for output length
    length_offsets = out_pos + tl.arange(0, BLOCK_SIZE_N)
    length_mask = length_offsets < out_length
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over input channels and kernel positions
    # For 1D convolution: for each output position, we compute dot product of
    # (in_channels * kernel_size) weights with corresponding input values
    
    # Precompute the kernel positions for this convolution
    # For a given output position, the input positions are:
    # input_pos = out_pos * stride + kernel_pos * dilation - padding
    # where kernel_pos ranges from 0 to kernel_size-1
    
    # We'll process kernel positions one by one to avoid complex indexing
    for k_pos in range(kernel_size):
        # Calculate input positions for this kernel position
        input_pos = out_pos * stride + k_pos * dilation - padding
        
        # For each input channel
        for in_ch_start in range(0, in_channels, BLOCK_SIZE_K // kernel_size):
            in_ch_offsets_k = in_ch_start + tl.arange(0, BLOCK_SIZE_K // kernel_size)
            in_ch_mask_k = in_ch_offsets_k < in_channels
            
            # Load input values: x[batch, in_ch, input_pos + k_pos * dilation - padding]
            # We need to handle boundary conditions for input positions
            in_pos_offsets = input_pos + tl.zeros((BLOCK_SIZE_K // kernel_size, BLOCK_SIZE_N), dtype=tl.int32)
            
            # Broadcast for the channel dimension
            in_ch_broadcast = in_ch_offsets_k[:, None]
            in_pos_broadcast = in_pos_offsets[None, :]
            
            # Calculate actual input indices
            input_indices = x_batch_offset + in_ch_broadcast * length + in_pos_broadcast
            
            # Load input values with boundary check
            input_vals = tl.load(
                x_ptr + input_indices,
                mask=(in_ch_mask_k[:, None] & length_mask[None, :] & 
                      (in_pos_broadcast >= 0) & (in_pos_broadcast < length)),
                other=0.0
            )
            
            # Load corresponding weights: w[out_ch, in_ch, k_pos]
            # Weight shape: (out_channels, in_channels // groups, kernel_size)
            # For grouped convolution, we need to adjust indexing
            
            # Calculate weight indices
            # For simplicity, assume groups=1 first, then extend
            # Weight layout: [out_ch, in_ch_per_group, kernel_size]
            
            # We'll load weights for this kernel position
            w_offsets_out = out_ch_offsets
            w_offsets_in = in_ch_offsets_k
            w_offsets_k = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_K // kernel_size), k_pos, dtype=tl.int32)
            
            # Reshape for proper broadcasting
            w_indices = (w_offsets_out[:, None] * (in_channels // groups) * kernel_size + 
                        w_offsets_in[None, :] * kernel_size + 
                        w_offsets_k)
            
            # Load weights
            weights = tl.load(
                w_ptr + w_indices,
                mask=(out_ch_mask[:, None] & in_ch_mask_k[None, :]),
                other=0.0
            )
            
            # Perform accumulation: acc += input_vals * weights
            acc += tl.dot(input_vals, weights, transb=True)
    
    # Add bias if present
    if b_ptr is not None:
        bias_offsets = out_ch_offsets
        bias_vals = tl.load(b_ptr + bias_offsets, mask=out_ch_mask, other=0.0)
        acc += bias_vals[:, None]
    
    # Store output
    out_batch_offset = pid_batch * out_channels * out_length
    out_indices = (out_batch_offset + 
                   out_ch_offsets[:, None] * out_length + 
                   length_offsets[None, :])
    
    tl.store(
        out_ptr + out_indices,
        acc.to(x_ptr.dtype.element_ty),
        mask=(out_ch_mask[:, None] & length_mask[None, :])
    )


def triton_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                  stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1):
    """
    Triton-based 1D convolution.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, length)
        weight: Weight tensor of shape (out_channels, in_channels // groups, kernel_size)
        bias: Optional bias tensor of shape (out_channels,)
        stride, padding, dilation, groups: Convolution parameters
    
    Returns:
        Output tensor of shape (batch_size, out_channels, out_length)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, length = x.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    out_length = (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_length, device=x.device, dtype=x.dtype)
    
    # Define block sizes - tune these based on GPU capabilities
    BLOCK_SIZE_M = 16  # Output channels per block
    BLOCK_SIZE_N = 256  # Output length per block  
    BLOCK_SIZE_K = 256  # Input channels * kernel_size per block
    
    # Grid dimensions
    grid = (batch_size, 
            triton.cdiv(out_channels, BLOCK_SIZE_M),
            triton.cdiv(out_length, BLOCK_SIZE_N))
    
    # Launch kernel
    conv1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, length, out_length,
        kernel_size, stride, padding, dilation, groups,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for 1D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Create the weight and bias parameters manually since we'll use custom kernel
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Store convolution parameters
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights (using Kaiming uniform initialization like PyTorch)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv1d(x, self.weight, self.bias, 
                            self.stride, self.padding, self.dilation, self.groups)


import math