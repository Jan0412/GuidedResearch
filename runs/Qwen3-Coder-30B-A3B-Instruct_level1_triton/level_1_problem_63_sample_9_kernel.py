import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    groups,
    group_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_BIAS: tl.constexpr
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Calculate output dimensions
    num_output_elements = output_height * output_width
    
    # Each program processes one output element
    output_idx = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    channel_idx = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Mask for valid output indices
    mask_m = output_idx < num_output_elements
    mask_n = channel_idx < out_channels
    
    # Shared memory for input tile and weight tile
    input_tile = tl.shared_tensor(tl.float32, (BLOCK_SIZE_M, BLOCK_SIZE_K))
    weight_tile = tl.shared_tensor(tl.float32, (BLOCK_SIZE_K, BLOCK_SIZE_N))
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over groups
    for g in range(groups):
        # Calculate group offset
        group_offset_in = g * group_size
        group_offset_out = g * (out_channels // groups)
        
        # Loop over kernel elements
        for k in range(0, in_channels // groups * kernel_height * kernel_width, BLOCK_SIZE_K):
            # Load input tile
            input_row = output_idx // output_width
            input_col = output_idx % output_width
            
            # Calculate input coordinates
            input_row_start = input_row * stride_h - padding_h
            input_col_start = input_col * stride_w - padding_w
            
            # Load input data with proper indexing
            input_elements = []
            for i in range(BLOCK_SIZE_M):
                for j in range(BLOCK_SIZE_K):
                    if k + j < (in_channels // groups) * kernel_height * kernel_width:
                        # Calculate actual input coordinates
                        kh = (k + j) // (in_channels // groups * kernel_width)
                        kw = (k + j) % kernel_width
                        
                        input_row_actual = input_row_start + kh
                        input_col_actual = input_col_start + kw
                        
                        # Check bounds
                        if (input_row_actual >= 0 and input_row_actual < input_height and 
                            input_col_actual >= 0 and input_col_actual < input_width):
                            
                            # Calculate input index
                            input_idx = (pid_b * in_channels + group_offset_in + 
                                       (k + j) // (kernel_width * kernel_height)) * input_height * input_width + \
                                      input_row_actual * input_width + input_col_actual
                            input_elements.append(input_ptr[input_idx])
                        else:
                            input_elements.append(0.0)
            
            # Load weight tile
            weight_elements = []
            for i in range(BLOCK_SIZE_K):
                for j in range(BLOCK_SIZE_N):
                    if k + i < (in_channels // groups) * kernel_height * kernel_width and j < (out_channels // groups):
                        weight_idx = (group_offset_out + j) * (in_channels // groups) * kernel_height * kernel_width + (k + i)
                        weight_elements.append(weight_ptr[weight_idx])
            
            # Compute dot product
            for i in range(BLOCK_SIZE_M):
                for j in range(BLOCK_SIZE_N):
                    for l in range(BLOCK_SIZE_K):
                        if k + l < (in_channels // groups) * kernel_height * kernel_width:
                            acc[i, j] += input_elements[i * BLOCK_SIZE_K + l] * weight_elements[l * BLOCK_SIZE_N + j]
    
    # Apply bias if needed
    if USE_BIAS:
        for i in range(BLOCK_SIZE_M):
            for j in range(BLOCK_SIZE_N):
                if mask_m[i] and mask_n[j]:
                    acc[i, j] += bias_ptr[group_offset_out + j]
    
    # Store results
    for i in range(BLOCK_SIZE_M):
        for j in range(BLOCK_SIZE_N):
            if mask_m[i] and mask_n[j]:
                output_idx_global = (pid_b * out_channels + group_offset_out + j) * output_height * output_width + output_idx[i]
                output_ptr[output_idx_global] = acc[i, j]

def triton_conv2d(input_tensor, weight, bias, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """
    Triton implementation of 2D convolution
    """
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1) // stride[1] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    
    # Grid configuration
    grid_m = (output_height * output_width + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (out_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_b = batch_size
    
    grid = (grid_m, grid_n, grid_b)
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        groups,
        in_channels // groups,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        USE_BIAS=(bias is not None)
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with a square input and square kernel.
    Uses custom Triton kernels for optimization.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv2d(
            x, 
            self.weight, 
            self.bias, 
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            dilation=(self.dilation, self.dilation),
            groups=self.groups
        )

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, groups={self.groups}, "
            f"bias={self.bias is not None}"
        )