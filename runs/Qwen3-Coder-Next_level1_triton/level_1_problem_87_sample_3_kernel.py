import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def pointwise_conv_kernel(
    input_ptr,          # Input tensor pointer (batch, in_channels, height, width)
    weight_ptr,         # Weight tensor pointer (out_channels, in_channels, 1, 1)
    bias_ptr,           # Bias tensor pointer (out_channels,) or None
    output_ptr,         # Output tensor pointer (batch, out_channels, height, width)
    batch_size,         # Batch size
    in_channels,        # Number of input channels
    out_channels,       # Number of output channels
    spatial_size,       # height * width
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for spatial elements
    BLOCK_SIZE_K: tl.constexpr,  # Block size for input channels
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    pid_spatial = tl.program_id(2)
    
    # Calculate batch offset
    batch_offset = pid_batch * in_channels * spatial_size
    
    # Calculate output channel offset
    out_ch_offset = pid_out_ch * out_channels
    
    # Calculate spatial block offset
    spatial_start = pid_spatial * BLOCK_SIZE_N
    spatial_offsets = spatial_start + tl.arange(0, BLOCK_SIZE_N)
    spatial_mask = spatial_offsets < spatial_size
    
    # Create channel offsets for input (in_channels) and output (out_channels)
    in_ch_offsets = tl.arange(0, BLOCK_SIZE_K)
    out_ch_offsets = pid_out_ch * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for k in range(0, in_channels, BLOCK_SIZE_K):
        # Load input: shape (BLOCK_SIZE_M, BLOCK_SIZE_N) but we need to handle indexing properly
        # Input is [batch, in_ch, spatial] - flatten spatial dimension for easier indexing
        input_offsets = batch_offset + k * spatial_size + spatial_offsets
        input_mask = spatial_mask
        
        # Load weight for this channel block
        weight_offsets = out_ch_offsets[:, None] * in_channels + k + tl.arange(0, BLOCK_SIZE_K)[None, :]
        weight = tl.load(weight_ptr + weight_offsets, mask=(out_ch_offsets < out_channels)[:, None], other=0.0)
        
        # Load input values
        input_vals = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
        
        # Accumulate: for each output channel, multiply with corresponding input channel
        # Since weight is [out_ch, in_ch], and input is [in_ch, spatial], we need to handle this correctly
        # Actually, for 1x1 conv: output[out_ch, spatial] = sum_k weight[out_ch, k] * input[k, spatial]
        
        # Load single input value for this channel
        input_val = input_vals  # scalar per spatial position
        
        # Accumulate contribution from this input channel
        for m in range(BLOCK_SIZE_M):
            if out_ch_offsets[m] < out_channels:
                w_val = weight[m, k % BLOCK_SIZE_K] if k < in_channels - BLOCK_SIZE_K else tl.load(weight_ptr + out_ch_offsets[m] * in_channels + k)
                acc_val = acc[m] + w_val * input_val
                acc = tl.where(tl.arange(0, BLOCK_SIZE_M) == m, acc_val, acc)
    
    # Add bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + pid_out_ch * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M), 
                      mask=(pid_out_ch * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) < out_channels)
        acc = acc + bias
    
    # Store results
    output_offsets = (batch_offset + pid_out_ch * spatial_size + spatial_offsets)
    output_mask = spatial_mask
    tl.store(output_ptr + output_offsets, acc, mask=output_mask)


# A better implementation using a more straightforward approach
@triton.jit
def pointwise_conv_kernel_v2(
    input_ptr,          # Input tensor pointer (batch, in_channels, height, width)
    weight_ptr,         # Weight tensor pointer (out_channels, in_channels, 1, 1)
    bias_ptr,           # Bias tensor pointer (out_channels,) or None
    output_ptr,         # Output tensor pointer (batch, out_channels, height, width)
    batch_size,         # Batch size
    in_channels,        # Number of input channels
    out_channels,       # Number of output channels
    height,             # Height of the feature map
    width,              # Width of the feature map
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for spatial elements
    BLOCK_SIZE_K: tl.constexpr,  # Block size for input channels
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    pid_spatial = tl.program_id(2)
    
    # Calculate spatial index
    spatial_idx = pid_spatial * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    spatial_mask = spatial_idx < height * width
    
    # Calculate offsets for this batch
    batch_start = pid_batch * in_channels * height * width
    
    # Calculate output channel block
    out_ch_start = pid_out_ch * BLOCK_SIZE_M
    out_ch_offsets = out_ch_start + tl.arange(0, BLOCK_SIZE_M)
    out_ch_mask = out_ch_offsets < out_channels
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Perform the convolution: for each input channel, multiply and accumulate
    for k in range(0, in_channels, BLOCK_SIZE_K):
        # Load weight block: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        weight_offsets = out_ch_offsets[:, None] * in_channels + (k + tl.arange(0, BLOCK_SIZE_K))[None, :]
        weight_block = tl.load(weight_ptr + weight_offsets, 
                              mask=out_ch_mask[:, None], 
                              other=0.0)
        
        # Load input values for all channels in this block at this spatial position
        input_offsets = batch_start + (k + tl.arange(0, BLOCK_SIZE_K))[None, :] * (height * width) + spatial_idx[None, :]
        input_block = tl.load(input_ptr + input_offsets, 
                             mask=spatial_mask[None, :], 
                             other=0.0)
        
        # Accumulate: output[oc, sp] = sum_ic weight[oc, ic] * input[ic, sp]
        acc += tl.dot(weight_block, input_block, trans_a=False, trans_b=True)
    
    # Add bias if provided
    if bias_ptr is not None:
        bias_offsets = out_ch_offsets
        bias = tl.load(bias_ptr + bias_offsets, mask=out_ch_mask, other=0.0)
        acc = acc + bias
    
    # Store results
    output_offsets = batch_start + out_ch_offsets[:, None] * (height * width) + spatial_idx[None, :]
    tl.store(output_ptr + output_offsets, acc, mask=out_ch_mask[:, None] & spatial_mask[None, :])


def triton_pointwise_conv(input_tensor, weight, bias=None):
    """
    Performs pointwise 2D convolution using Triton kernel.
    
    Args:
        input_tensor: [batch, in_channels, height, width]
        weight: [out_channels, in_channels, 1, 1]
        bias: [out_channels] or None
    
    Returns:
        output: [batch, out_channels, height, width]
    """
    assert input_tensor.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight.shape[0]
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, height, width, 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate spatial size
    spatial_size = height * width
    
    # Kernel parameters
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    
    # Grid dimensions: (batch, out_channels_block, spatial_block)
    grid = lambda meta: (
        batch_size,
        triton.cdiv(out_channels, meta["BLOCK_SIZE_M"]),
        triton.cdiv(spatial_size, meta["BLOCK_SIZE_N"])
    )
    
    # Launch kernel
    pointwise_conv_kernel_v2[grid](
        input_tensor, weight, bias,
        output,
        batch_size, in_channels, out_channels, height, width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized version of the pointwise 2D convolution using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        # Remove the original conv layer and implement with Triton
        del self.conv1d
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the pointwise 2D convolution using Triton kernel.
        """
        return triton_pointwise_conv(x, self.weight, self.bias)