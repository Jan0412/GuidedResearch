import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose1d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, length)
    w_ptr,  # Weight tensor: (in_channels, out_channels, kernel_size)
    b_ptr,  # Bias tensor: (out_channels,) - can be None
    out_ptr,  # Output tensor: (batch, out_channels, out_length)
    batch_size, in_channels, out_channels, 
    input_length, kernel_size, 
    stride, padding, dilation,
    out_length,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for input channels
    BLOCK_SIZE_K: tl.constexpr,  # Block size for kernel elements
    BLOCK_SIZE_L: tl.constexpr,  # Block size for output length
):
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_out_c = tl.program_id(1)
    pid_out_l = tl.program_id(2)
    
    # Offset for this batch
    batch_offset = pid_batch * in_channels * input_length
    
    # Calculate output position
    out_l = pid_out_l * BLOCK_SIZE_L
    if out_l >= out_length:
        return
        
    # Compute the range of input positions that contribute to this output position
    # For transposed conv: out_pos = in_pos * stride + (kernel_pos - 1) * dilation - padding
    # So in_pos = (out_pos + padding - (kernel_pos - 1) * dilation) / stride
    
    # Process a tile of output length
    out_l_offsets = out_l + tl.arange(0, BLOCK_SIZE_L)
    out_l_mask = out_l_offsets < out_length
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_L, BLOCK_SIZE_M), dtype=tl.float32)
    
    # Loop over input channels
    for ic in range(0, in_channels, BLOCK_SIZE_N):
        ic_offsets = ic + tl.arange(0, BLOCK_SIZE_N)
        ic_mask = ic_offsets < in_channels
        
        # Load input data: shape (batch, in_channels, length)
        # We need to access positions that contribute to our output
        in_l_base = out_l_offsets * stride + padding
        # For each kernel position, calculate corresponding input position
        # But we'll handle kernel loop separately for efficiency
        
        # Process kernel elements
        for k in range(0, kernel_size, BLOCK_SIZE_K):
            k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
            k_mask = k_offsets < kernel_size
            
            # Calculate input positions for this kernel position
            # in_pos = (out_pos + padding - (kernel_pos - 1) * dilation) / stride
            # Only use integer positions that are valid
            dilation_k = (k_offsets - 0) * dilation  # (kernel_pos - 1) * dilation with kernel_pos starting at 1
            
            # Calculate input positions: (out_l_offsets + padding - dilation_k) / stride
            # But we need to handle the division properly
            in_l = (out_l_offsets * stride + padding - dilation_k[:, None]) // stride
            
            # Check bounds for input positions
            in_l_valid = (in_l >= 0) & (in_l < input_length)
            
            # Load input values: we need to handle the 3D indexing carefully
            # Flatten the batch and channel dimensions for easier indexing
            in_offsets = batch_offset + ic_offsets[None, :] * input_length + in_l
            
            # Load input with proper masking
            in_data = tl.load(
                x_ptr + in_offsets,
                mask=(ic_mask[None, :] & in_l_valid),
                other=0.0
            )
            
            # Load kernel weights: shape (in_channels, out_channels, kernel_size)
            # We want w[ic, oc, k] for our current tiles
            w_offsets = (
                ic_offsets[:, None] * (out_channels * kernel_size) + 
                pid_out_c * kernel_size + k_offsets[None, :]
            )
            
            # Load weights with masking
            w_data = tl.load(
                w_ptr + w_offsets,
                mask=(ic_mask[:, None] & k_mask[None, :]),
                other=0.0
            )
            
            # Compute accumulation: acc += input * weight
            # in_data shape: (BLOCK_SIZE_N, BLOCK_SIZE_L)
            # w_data shape: (BLOCK_SIZE_N, BLOCK_SIZE_K)
            # We want to accumulate over kernel positions and input channels
            
            # Reshape for matmul-like operation
            # For each output position l and output channel oc, we sum over ic and k
            # acc[l, oc] += sum_ic sum_k in[ic, l] * w[ic, oc, k] where l corresponds to position related to k
            
            # Actually, let's reorganize: for each output position and output channel, 
            # we sum over input channels and kernel positions
            
            # The correct computation is:
            # out[batch, oc, out_l] = sum_ic sum_k in[batch, ic, (out_l + padding - k*dilation)/stride] * w[ic, oc, k]
            # where division is integer division and only valid positions are used
            
            # Since we're processing BLOCK_SIZE_L output positions at once, we need:
            # acc[:, :] += in_data @ w_data^T but with proper alignment
            
            # Let me reconsider the indexing. The kernel index k in the loop corresponds to actual kernel positions.
            # For each output position out_l, and kernel position k, the contributing input position is:
            # in_pos = (out_l * stride + padding - k * dilation) / stride
            
            # Actually, let's simplify: since stride=1 in the test case, in_pos = out_l + padding - k*dilation
            # But for general stride, it's more complex
            
            # Given the complexity, let's use a simpler approach for the common case
            # and handle the general case with proper bounds checking
            
            # For efficiency, we'll compute the contribution to each output position
            # from the current kernel position k
            for lk in range(tl.minimum(BLOCK_SIZE_L, out_length - out_l)):
                # For output position out_l_offsets[lk]
                # Calculate which input position contributes for kernel position k_offsets[0]
                # (since we're in a loop over k, we need to handle each k separately)
                
                # This approach is getting too complex. Let's use a different strategy.
                pass
    
    # Actually, let me rewrite this kernel with a cleaner approach
    # For transposed convolution: output position depends on multiple input positions
    # and kernel weights
    
    # Simpler approach: for each output position, compute its value
    for lk in range(tl.minimum(BLOCK_SIZE_L, out_length - out_l)):
        out_pos = out_l + lk
        
        # For this output position, accumulate over input channels and kernel positions
        for ic in range(0, in_channels, BLOCK_SIZE_N):
            ic_offsets = ic + tl.arange(0, BLOCK_SIZE_N)
            ic_mask = ic_offsets < in_channels
            
            # For each kernel position
            for k in range(kernel_size):
                # Calculate input position: in_pos = (out_pos + padding - k*dilation) / stride
                in_pos = (out_pos * stride + padding - k * dilation)
                if stride > 1:
                    # Check if in_pos is divisible by stride
                    if in_pos % stride != 0:
                        continue
                    in_pos = in_pos // stride
                else:
                    in_pos = in_pos // stride  # stride=1 case
                    
                if in_pos < 0 or in_pos >= input_length:
                    continue
                
                # Load input: x[batch, ic, in_pos]
                x_offsets = batch_offset + ic_offsets * input_length + in_pos
                x_data = tl.load(x_ptr + x_offsets, mask=ic_mask, other=0.0)
                
                # Load weights: w[ic, pid_out_c, k]
                w_offsets = ic_offsets * (out_channels * kernel_size) + pid_out_c * kernel_size + k
                w_data = tl.load(w_ptr + w_offsets, mask=ic_mask, other=0.0)
                
                # Accumulate
                acc[lk, :] += x_data * w_data
    
    # Add bias if present
    if b_ptr is not None:
        b_offsets = pid_out_c
        bias = tl.load(b_ptr + b_offsets)
        acc += bias
    
    # Store result
    out_offsets = (
        pid_batch * out_channels * out_length + 
        pid_out_c * out_length + 
        out_l_offsets
    )
    out_mask = out_l_mask
    tl.store(out_ptr + out_offsets, acc.T, mask=out_mask)


def triton_conv_transpose1d(x, weight, bias, stride, padding, dilation):
    """
    Triton implementation of ConvTranspose1d
    
    x: (batch, in_channels, length)
    weight: (in_channels, out_channels, kernel_size)
    bias: (out_channels,) or None
    """
    batch_size, in_channels, input_length = x.shape
    _, out_channels, kernel_size = weight.shape
    
    # Calculate output length
    out_length = (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    
    # Allocate output tensor
    out = torch.empty(batch_size, out_channels, out_length, dtype=x.dtype, device=x.device)
    
    if out_length == 0:
        return out
    
    # Define block sizes
    BLOCK_SIZE_M = 16  # output channels per block
    BLOCK_SIZE_N = 16  # input channels per block
    BLOCK_SIZE_K = 8   # kernel elements per block
    BLOCK_SIZE_L = 32  # output length per block
    
    # Grid dimensions
    grid = (
        batch_size,  # batch
        triton.cdiv(out_channels, BLOCK_SIZE_M),  # output channels
        triton.cdiv(out_length, BLOCK_SIZE_L),    # output length
    )
    
    # Launch kernel
    conv_transpose1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        input_length, kernel_size,
        stride, padding, dilation,
        out_length,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for transposed 1D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters for reference
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.has_bias = bias
        
        # Create parameters matching nn.ConvTranspose1d
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Initialize weights (matching PyTorch's default initialization)
        self.reset_parameters()
    
    def reset_parameters(self):
        # Kaiming uniform initialization like PyTorch
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are contiguous and on GPU
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Call our Triton kernel
        return triton_conv_transpose1d(x, weight, self.bias, 
                                       self.stride, self.padding, self.dilation)