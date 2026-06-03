import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def transposed_conv2d_kernel(
    x_ptr,  # Input tensor: (B, C_in, H_in, W_in)
    w_ptr,  # Weight tensor: (C_in, C_out, K_h, K_w)
    b_ptr,  # Bias tensor: (C_out,)
    y_ptr,  # Output tensor: (B, C_out, H_out, W_out)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    output_padding: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for output spatial elements
    BLOCK_SIZE_K: tl.constexpr,  # Block size for input channels
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_c = tl.program_id(1)
    pid_spatial = tl.program_id(2)
    
    # Calculate output spatial position from pid_spatial
    # We'll map pid_spatial to (oh, ow) coordinates
    spatial_block_size = BLOCK_SIZE_N
    spatial_idx = pid_spatial * spatial_block_size
    
    # Unpack spatial index into oh, ow
    oh = spatial_idx // output_width
    ow = spatial_idx % output_width
    
    # Create output channel range
    out_c_offsets = pid_out_c * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    out_c_mask = out_c_offsets < out_channels
    
    # Calculate input spatial position for the top-left corner of the kernel
    # For transposed convolution: H_out = (H_in - 1) * stride - 2 * padding + (kernel_size - 1) + 1 + output_padding
    # So input position corresponding to output position (oh, ow) is:
    ih_base = oh - (kernel_size - 1) + padding
    iw_base = ow - (kernel_size - 1) + padding
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over input channels
    for ic in range(0, in_channels, BLOCK_SIZE_K):
        ic_offsets = ic + tl.arange(0, BLOCK_SIZE_K)
        ic_mask = ic_offsets < in_channels
        
        # Load input block: x[batch, ic, ih, iw]
        # For transposed conv, we need to sample multiple input positions
        # Based on kernel positions
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Calculate input position for this kernel element
                ih = ih_base + kh * stride
                iw = iw_base + kw * stride
                
                # Check if input position is valid
                in_bounds = (ih >= 0) & (ih < input_height) & (iw >= 0) & (iw < input_width)
                
                if tl.static_cast(tl.int1, in_bounds):
                    # Calculate input pointer offset
                    x_offset = (pid_batch * in_channels * input_height * input_width +
                               ic_offsets * input_height * input_width +
                               ih * input_width + iw)
                    
                    # Load input values
                    x_val = tl.load(x_ptr + x_offset, mask=ic_mask, other=0.0)
                    
                    # Calculate weight pointer offset for this kernel position
                    w_offset = (ic_offsets * out_channels * kernel_size * kernel_size +
                               out_c_offsets[:, None] * kernel_size * kernel_size +
                               kh * kernel_size + kw)
                    
                    # Load weight values
                    w_val = tl.load(w_ptr + w_offset, mask=ic_mask[None, :] & out_c_mask[:, None], other=0.0)
                    
                    # Accumulate: acc[out_c] += x[ic] * w[ic, out_c, kh, kw]
                    acc += tl.sum(x_val[:, None] * w_val, axis=0)
    
    # Add bias if provided
    if b_ptr is not None:
        bias_offsets = out_c_offsets
        bias_mask = out_c_mask
        bias_val = tl.load(b_ptr + bias_offsets, mask=bias_mask, other=0.0)
        acc += bias_val
    
    # Store result
    y_offset = (pid_batch * out_channels * output_height * output_width +
               out_c_offsets * output_height * output_width +
               oh * output_width + ow)
    
    tl.store(y_ptr + y_offset, acc.to(tl.float32), mask=out_c_mask)


def triton_transposed_conv2d(x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
    """
    Triton implementation of transposed 2D convolution.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, input_height, input_width = x.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height - 1) * stride - 2 * padding + (kernel_height - 1) + 1 + output_padding
    output_width = (input_width - 1) * stride - 2 * padding + (kernel_width - 1) + 1 + output_padding
    
    # Prepare output tensor
    y = torch.empty(batch_size, out_channels, output_height, output_width, dtype=x.dtype, device=x.device)
    
    # Tunable parameters
    BLOCK_SIZE_M = 8   # Output channels per block
    BLOCK_SIZE_N = 256  # Spatial elements per block
    BLOCK_SIZE_K = 32   # Input channels per block
    
    # Grid dimensions
    grid = (batch_size,
            triton.cdiv(out_channels, BLOCK_SIZE_M),
            triton.cdiv(output_height * output_width, BLOCK_SIZE_N))
    
    # Launch kernel
    transposed_conv2d_kernel[grid](
        x, weight, bias, y,
        batch_size, in_channels, out_channels,
        kernel_height, stride, padding, output_padding,
        input_height, input_width,
        output_height, output_width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for transposed 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters and create weight/bias tensors
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Initialize weight and bias as buffers (not parameters for simplicity)
        # In practice, these would be trained parameters
        self.register_buffer('weight', torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.register_buffer('bias', torch.empty(out_channels))
        else:
            self.bias = None
            
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized transposed 2D convolution using Triton kernel.
        """
        return triton_transposed_conv2d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )


# Import math for kaiming initialization
import math