import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, length)
    w_ptr,  # Weight tensor: (out_channels, in_channels, kernel_size)
    b_ptr,  # Bias tensor: (out_channels,) or None
    out_ptr,  # Output tensor: (batch, out_channels, output_length)
    batch_size, in_channels, out_channels, input_length, output_length, kernel_size,
    stride, padding, dilation,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for output length
    BLOCK_SIZE_K: tl.constexpr,  # Block size for in_channels * kernel_size
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)  # Output channel block
    pid_n = tl.program_id(2)  # Output position block
    
    # Calculate offsets
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create mask for valid indices
    mask_m = offsets_m < out_channels
    mask_n = offsets_n < output_length
    
    # Initialize accumulator
    output = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over in_channels and kernel_size
    for k in range(0, in_channels * kernel_size, BLOCK_SIZE_K):
        # Map k to (ic, kw)
        k_block = k + offsets_k
        ic = k_block // kernel_size
        kw = k_block % kernel_size
        
        # Create masks for k
        mask_k = k_block < in_channels * kernel_size
        
        # Load weights: shape (out_channels, in_channels, kernel_size)
        # Transpose for GEMM: we want w[oc, ic, kw] -> w[oc, ic*kernel_size+kw]
        w_offsets = (offsets_m[:, None] * in_channels * kernel_size + 
                    ic[None, :] * kernel_size + kw[None, :])
        w = tl.load(w_ptr + w_offsets, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Load input using im2col: for each output position n, get the corresponding input window
        # input position for output n is: n * stride + kw * dilation - padding
        # We need to compute this for all n in the block
        input_positions = offsets_n[None, :] * stride + kw[:, None] * dilation - padding
        
        # Create mask for valid input positions
        input_mask = (input_positions >= 0) & (input_positions < input_length)
        
        # Load input: x[batch, ic, input_position]
        # We need to handle the batch dimension
        batch_offset = pid_batch * in_channels * input_length
        x_offsets = batch_offset + ic[:, None] * input_length + input_positions
        x_val = tl.load(x_ptr + x_offsets, mask=mask_k[:, None] & input_mask, other=0.0)
        
        # Accumulate: output[oc, n] += sum_{ic,kw} x[ic, pos] * w[oc, ic, kw]
        output += tl.dot(w, x_val, out_dtype=tl.float32)
    
    # Load bias if available
    if b_ptr is not None:
        bias = tl.load(b_ptr + offsets_m, mask=mask_m, other=0.0)
        output += bias[:, None]
    
    # Store output
    out_offsets = (pid_batch * out_channels * output_length + 
                  offsets_m[:, None] * output_length + offsets_n[None, :])
    tl.store(out_ptr + out_offsets, output, mask=mask_m[:, None] & mask_n[None, :])


def triton_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                 stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1):
    """
    Triton-based 1D convolution implementation.
    Supports only groups=1 for simplicity in this implementation.
    """
    assert groups == 1, "Triton conv1d only supports groups=1 for this implementation"
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    
    # Ensure contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, input_length = x.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, output_length, dtype=x.dtype, device=x.device)
    
    # Define block sizes (tunable for performance)
    BLOCK_SIZE_M = 16  # Output channels per block
    BLOCK_SIZE_N = 64  # Output positions per block  
    BLOCK_SIZE_K = 32  # in_channels * kernel_size per block
    
    # Grid: (batch, out_channels_blocks, output_length_blocks)
    grid = (
        batch_size,
        triton.cdiv(out_channels, BLOCK_SIZE_M),
        triton.cdiv(output_length, BLOCK_SIZE_N)
    )
    
    # Launch kernel
    conv1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, input_length, output_length, kernel_size,
        stride, padding, dilation,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized 1D convolution using Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Create the weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
        # Store convolution parameters
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv1d(x, self.weight, self.bias, 
                            self.stride, self.padding, self.dilation, self.groups)


import math