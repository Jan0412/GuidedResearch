import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    # Pointers to inputs and outputs
    input_ptr, output_ptr, weight_ptr, bias_ptr,
    # Dimensions
    N, H, W, C_in, C_out, K,
    # Strides
    stride_input, stride_output, stride_weight,
    # Convolution parameters
    stride_h, stride_w, padding, dilation,
    # Block sizes
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C_in: tl.constexpr, BLOCK_SIZE_C_out: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs for output spatial dimensions and output channels
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c_out = tl.program_id(3)
    
    # Calculate output spatial coordinates
    h_out = pid_h
    w_out = pid_w
    
    # Calculate input spatial coordinates (accounting for stride and padding)
    h_in_start = h_out * stride_h - padding
    w_in_start = w_out * stride_w - padding
    
    # Initialize accumulator for this output position
    acc = tl.zeros((BLOCK_SIZE_C_out,), dtype=tl.float32)
    
    # Loop over input channels
    for c_in_start in range(0, C_in, BLOCK_SIZE_C_in):
        c_in = c_in_start + tl.arange(0, BLOCK_SIZE_C_in)
        c_in_mask = c_in < C_in
        
        # Loop over kernel height
        for kh in range(0, K, BLOCK_SIZE_K):
            kh_range = kh + tl.arange(0, BLOCK_SIZE_K)
            kh_mask = kh_range < K
            
            # Loop over kernel width
            for kw in range(0, K, BLOCK_SIZE_K):
                kw_range = kw + tl.arange(0, BLOCK_SIZE_K)
                kw_mask = kw_range < K
                
                # Calculate input coordinates for this kernel position
                h_coords = h_in_start + kh_range * dilation
                w_coords = w_in_start + kw_range * dilation
                
                # Create masks for valid input coordinates
                h_mask = (h_coords >= 0) & (h_coords < H)
                w_mask = (w_coords >= 0) & (w_coords < W)
                
                # Load input values
                # Input shape: (N, C_in, H, W)
                input_offsets = pid_n * stride_input + c_in[:, None, None] * stride_input // C_in + \
                               h_coords[None, :, None] * stride_input // H + \
                               w_coords[None, None, :] * stride_input // W
                input_offsets = input_offsets.flatten()
                input_mask = (c_in_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]).flatten()
                input_vals = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
                
                # Load weights
                # Weight shape: (C_out, C_in, K, K)
                weight_offsets = pid_c_out * stride_weight + c_in[None, :, None, None] * stride_weight // C_in + \
                                kh_range[None, None, :, None] * stride_weight // K + \
                                kw_range[None, None, None, :] * stride_weight // K
                weight_offsets = weight_offsets.flatten()
                weight_mask = (c_in_mask[None, :, None, None] & kh_mask[None, None, :, None] & kw_mask[None, None, None, :]).flatten()
                weight_vals = tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0)
                
                # Reshape for matrix multiplication
                input_vals = input_vals.reshape(BLOCK_SIZE_C_in, BLOCK_SIZE_K * BLOCK_SIZE_K)
                weight_vals = weight_vals.reshape(BLOCK_SIZE_C_in, BLOCK_SIZE_K * BLOCK_SIZE_K)
                
                # Accumulate: acc += input * weight^T
                acc += tl.sum(input_vals[:, :, None] * weight_vals[:, None, :], axis=0)
    
    # Add bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + pid_c_out)
        acc += bias
    
    # Store output
    # Output shape: (N, C_out, H_out, W_out)
    output_offsets = pid_n * stride_output + pid_c_out * stride_output // C_out + \
                    h_out * stride_output // H + \
                    w_out * stride_output // W
    tl.store(output_ptr + output_offsets, acc.to(tl.float32))


def triton_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Triton implementation of 2D convolution.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    assert groups == 1, "Groups != 1 not supported in this implementation"
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    N, C_in, H, W = x.shape
    C_out, _, K, _ = weight.shape
    
    # Calculate output dimensions
    H_out = (H + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    
    # Prepare output tensor
    output = torch.empty(N, C_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Calculate strides
    stride_input = x.stride(0)
    stride_output = output.stride(0)
    stride_weight = weight.stride(0)
    
    # Set block sizes (tunable parameters for optimization)
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    BLOCK_SIZE_C_in = min(16, C_in)
    BLOCK_SIZE_C_out = min(32, C_out)
    BLOCK_SIZE_K = min(3, K)
    
    # Grid dimensions: (batch, h_blocks, w_blocks, c_out_blocks)
    grid = lambda meta: (
        N,
        (H_out + meta['BLOCK_SIZE_H'] - 1) // meta['BLOCK_SIZE_H'],
        (W_out + meta['BLOCK_SIZE_W'] - 1) // meta['BLOCK_SIZE_W'],
        (C_out + meta['BLOCK_SIZE_C_out'] - 1) // meta['BLOCK_SIZE_C_out']
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        x, output, weight, bias,
        N, H, W, C_in, C_out, K,
        stride_input, stride_output, stride_weight,
        stride, stride, padding, dilation,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C_in=BLOCK_SIZE_C_in,
        BLOCK_SIZE_C_out=BLOCK_SIZE_C_out,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton convolution kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize weights and bias manually to use with our Triton kernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Register weight and bias as parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using our custom Triton kernel.
        """
        return triton_conv2d(x, self.weight, self.bias, 
                            stride=self.stride, padding=self.padding, 
                            dilation=self.dilation, groups=self.groups)