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
    input_stride_0, input_stride_1, input_stride_2, input_stride_3,
    weight_stride_0, weight_stride_1, weight_stride_2, weight_stride_3,
    output_stride_0, output_stride_1, output_stride_2, output_stride_3,
    batch_size, in_channels, out_channels, height, width, 
    kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    USE_BIAS: tl.constexpr
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_batch = tl.program_id(2)
    
    # Tile indices
    tile_m = pid_m * BLOCK_SIZE_M
    tile_n = pid_n * BLOCK_SIZE_N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension (input channels and kernel elements)
    for k in range(0, in_channels * kernel_height * kernel_width, BLOCK_SIZE_K):
        # Compute bounds for this tile
        k_end = min(k + BLOCK_SIZE_K, in_channels * kernel_height * kernel_width)
        
        # Load input tile (batch, in_channels, H, W)
        # For each kernel element, compute the appropriate offset
        input_tile = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
        weight_tile = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
        
        # Load weights (kernel_height * kernel_width * in_channels, out_channels)
        if k < in_channels * kernel_height * kernel_width:
            for i in range(BLOCK_SIZE_K):
                if k + i < in_channels * kernel_height * kernel_width:
                    # Extract channel, kernel_y, kernel_x from linear index
                    idx = k + i
                    c = idx % in_channels
                    ky = (idx // in_channels) % kernel_height
                    kx = (idx // (in_channels * kernel_height)) % kernel_width
                    
                    # Compute actual position in weight tensor
                    w_offset = ky * weight_stride_2 + kx * weight_stride_3 + c * weight_stride_1
                    for j in range(BLOCK_SIZE_N):
                        if j < out_channels:
                            weight_tile[i, j] = tl.load(weight_ptr + w_offset + j * weight_stride_0)
        
        # Load input tiles for convolution
        # We'll do this more carefully to handle the sliding window
        for i in range(BLOCK_SIZE_M):
            if tile_m + i < height:
                for j in range(BLOCK_SIZE_K):
                    if k + j < in_channels * kernel_height * kernel_width:
                        # Extract channel, kernel_y, kernel_x from linear index
                        idx = k + j
                        c = idx % in_channels
                        ky = (idx // in_channels) % kernel_height
                        kx = (idx // (in_channels * kernel_height)) % kernel_width
                        
                        # Compute actual position in input tensor
                        h = tile_m + i
                        w = tile_n + j
                        
                        # Apply padding and dilation
                        h_in = h * stride_h - pad_h + ky * dilation_h
                        w_in = w * stride_w - pad_w + kx * dilation_w
                        
                        # Check bounds
                        if 0 <= h_in < height and 0 <= w_in < width:
                            input_offset = pid_batch * input_stride_0 + c * input_stride_1 + h_in * input_stride_2 + w_in * input_stride_3
                            input_tile[i, j] = tl.load(input_ptr + input_offset)
                        else:
                            input_tile[i, j] = 0.0
        
        # Accumulate
        acc += tl.dot(input_tile, weight_tile)
    
    # Apply bias if needed
    if USE_BIAS:
        bias = tl.load(bias_ptr + pid_n * output_stride_1)
        acc += bias
    
    # Write result to global memory
    for i in range(BLOCK_SIZE_M):
        for j in range(BLOCK_SIZE_N):
            if tile_m + i < height and tile_n + j < width:
                output_offset = pid_batch * output_stride_0 + pid_n * output_stride_1 + (tile_m + i) * output_stride_2 + (tile_n + j) * output_stride_3
                tl.store(output_ptr + output_offset, acc[i, j])

def triton_conv2d(input_tensor, weight, bias, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """
    Custom Triton implementation of 2D convolution.
    """
    # Ensure tensors are contiguous and on GPU
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    out_height = (height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1) // stride[0] + 1
    out_width = (width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1) // stride[1] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, out_height, out_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes and group size
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Grid dimensions
    grid_m = (out_height + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (out_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_batch = batch_size
    
    # Create grid
    grid = (grid_m, grid_n, grid_batch)
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        batch_size, in_channels, out_channels, height, width,
        kernel_height, kernel_width, stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        USE_BIAS=(bias is not None)
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with a square input and square kernel.
    Optimized using custom Triton kernels.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
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
        
        # Initialize weights and biases
        if groups == 1:
            self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv2d(x, self.weight, self.bias, 
                           stride=(self.stride, self.stride),
                           padding=(self.padding, self.padding),
                           dilation=(self.dilation, self.dilation))