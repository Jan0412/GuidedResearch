import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv3d_kernel(
    # Pointers to inputs and outputs
    input_ptr, output_ptr, weight_ptr, bias_ptr,
    # Dimensions
    batch_size, in_channels, out_channels,
    depth, height, width,
    out_d, out_h, out_w,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dilation_d, dilation_h, dilation_w,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch dimension
    BLOCK_SIZE_K: tl.constexpr,  # Block size for accumulation
    # Flags
    HAS_BIAS: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)  # Output channel block
    pid_n = tl.program_id(1)  # Batch block
    pid_z = tl.program_id(2)  # Spatial position (combined D, H, W)
    
    # Calculate output spatial indices from pid_z
    # Layout: [out_d * out_h * out_w] for spatial positions
    total_spatial = out_d * out_h * out_w
    spatial_id = pid_z
    
    # Decompose spatial_id into d, h, w indices
    d_out = spatial_id // (out_h * out_w)
    rem = spatial_id % (out_h * out_w)
    h_out = rem // out_w
    w_out = rem % out_w
    
    # Calculate corresponding input spatial indices
    d_in = d_out * stride_d - pad_d
    h_in = h_out * stride_h - pad_h
    w_in = w_out * stride_w - pad_w
    
    # Offset for the batch
    batch_offset = pid_n * (in_channels * depth * height * width)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over input channels and kernel dimensions
    for c in range(in_channels):
        for kd in range(kernel_d):
            d = d_in + kd * dilation_d
            if d < 0 or d >= depth:
                continue
                
            for kh in range(kernel_h):
                h = h_in + kh * dilation_h
                if h < 0 or h >= height:
                    continue
                    
                for kw in range(kernel_w):
                    w = w_in + kw * dilation_w
                    if w < 0 or w >= width:
                        continue
                        
                    # Calculate input pointer offset
                    input_offset = batch_offset + c * (depth * height * width) + \
                                   d * (height * width) + h * width + w
                                   
                    # Load input value
                    input_val = tl.load(input_ptr + input_offset)
                    
                    # Calculate weight pointer offset
                    # Weight layout: [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
                    weight_offset = (pid_m * BLOCK_SIZE_M) * (in_channels * kernel_d * kernel_h * kernel_w) + \
                                   c * (kernel_d * kernel_h * kernel_w) + \
                                   kd * (kernel_h * kernel_w) + \
                                   kh * kernel_w + kw
                                   
                    # Load weights for the current output channel block
                    weight_vals = tl.load(weight_ptr + weight_offset + tl.arange(0, BLOCK_SIZE_M))
                    
                    # Accumulate: input * weight
                    accumulator += input_val * weight_vals
    
    # Store result
    output_offset = pid_n * (out_channels * out_d * out_h * out_w) + \
                   pid_m * BLOCK_SIZE_M * (out_d * out_h * out_w) + \
                   d_out * (out_h * out_w) + h_out * out_w + w_out
    
    if HAS_BIAS:
        bias_offset = pid_m * BLOCK_SIZE_M
        bias_vals = tl.load(bias_ptr + bias_offset)
        accumulator += bias_vals
    
    # Store output
    tl.store(output_ptr + output_offset, accumulator.to(tl.float32), mask=pid_m * BLOCK_SIZE_M < out_channels)

def triton_conv3d(x, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1)):
    """
    Triton-based 3D convolution implementation.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    batch_size, in_channels, depth, height, width = x.shape
    out_channels, _, kernel_d, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    dilation_d, dilation_h, dilation_w = dilation
    
    out_d = (depth + 2 * pad_d - dilation_d * (kernel_d - 1) - 1) // stride_d + 1
    out_h = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, out_d, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Set block sizes (tunable parameters)
    BLOCK_SIZE_M = 8  # Output channel block size
    BLOCK_SIZE_N = 1  # Batch block size
    BLOCK_SIZE_K = 16  # Not used in this implementation but kept for consistency
    
    # Grid dimensions
    grid = (
        (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,  # Output channel blocks
        batch_size,  # Batch blocks
        out_d * out_h * out_w  # Spatial position blocks
    )
    
    # Launch kernel
    conv3d_kernel[grid](
        x, output, weight, bias,
        batch_size, in_channels, out_channels,
        depth, height, width,
        out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        HAS_BIAS=bias is not None
    )
    
    return output


class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with asymmetric input and kernel sizes,
    optimized with Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), 
                 padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias_flag = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using Triton kernel.
        """
        return triton_conv3d(x, self.weight, self.bias, 
                           stride=self.stride, padding=self.padding, 
                           dilation=self.dilation)
    
    def extra_repr(self):
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, ' \
               f'kernel_size={self.kernel_size}, stride={self.stride}, ' \
               f'padding={self.padding}, dilation={self.dilation}, ' \
               f'groups={self.groups}, bias={self.bias_flag is not None}'