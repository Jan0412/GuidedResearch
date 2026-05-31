import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    input_height,
    input_width,
    output_height,
    output_width,
    in_channels,
    out_channels,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    batch_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Tile indices
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension (input channels and kernel elements)
    for k in range(0, in_channels * kernel_h * kernel_w, BLOCK_SIZE_K):
        # Compute tile boundaries
        k_limit = min(k + BLOCK_SIZE_K, in_channels * kernel_h * kernel_w)
        
        # Load input tile
        input_tile = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
        if pid_b < batch_size and m_offset < output_height * output_width:
            for i in range(BLOCK_SIZE_M):
                for j in range(BLOCK_SIZE_K):
                    if k + j < in_channels * kernel_h * kernel_w:
                        # Calculate input coordinates
                        out_y = (m_offset + i) // output_width
                        out_x = (m_offset + i) % output_width
                        
                        # Calculate kernel coordinates
                        kernel_idx = k + j
                        kernel_c = kernel_idx // (kernel_h * kernel_w)
                        kernel_y = (kernel_idx % (kernel_h * kernel_w)) // kernel_w
                        kernel_x = (kernel_idx % (kernel_h * kernel_w)) % kernel_w
                        
                        # Calculate input coordinates
                        in_y = out_y * stride_h - padding_h + kernel_y
                        in_x = out_x * stride_w - padding_w + kernel_x
                        
                        # Check bounds
                        if 0 <= in_y < input_height and 0 <= in_x < input_width:
                            input_val = tl.load(input_ptr + 
                                              pid_b * input_height * input_width * in_channels +
                                              in_y * input_width * in_channels +
                                              in_x * in_channels +
                                              kernel_c)
                            input_tile[i, j] = input_val
        
        # Load weight tile
        weight_tile = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
        if k < in_channels * kernel_h * kernel_w:
            for i in range(BLOCK_SIZE_K):
                for j in range(BLOCK_SIZE_N):
                    if k + i < in_channels * kernel_h * kernel_w and j < out_channels:
                        kernel_idx = k + i
                        kernel_c = kernel_idx // (kernel_h * kernel_w)
                        kernel_y = (kernel_idx % (kernel_h * kernel_w)) // kernel_w
                        kernel_x = (kernel_idx % (kernel_h * kernel_w)) % kernel_w
                        
                        weight_val = tl.load(weight_ptr + 
                                           j * kernel_h * kernel_w * in_channels +
                                           kernel_y * kernel_w * in_channels +
                                           kernel_x * in_channels +
                                           kernel_c)
                        weight_tile[i, j] = weight_val
        
        # Accumulate
        acc += tl.dot(input_tile, weight_tile)
    
    # Add bias
    if n_offset < out_channels:
        bias = tl.load(bias_ptr + n_offset)
        acc += bias
    
    # Write output
    if pid_b < batch_size and m_offset < output_height * output_width:
        for i in range(BLOCK_SIZE_M):
            for j in range(BLOCK_SIZE_N):
                if m_offset + i < output_height * output_width and n_offset + j < out_channels:
                    tl.store(output_ptr + 
                           pid_b * output_height * output_width * out_channels +
                           (m_offset + i) * out_channels +
                           (n_offset + j),
                           acc[i, j])

def triton_conv2d(input_tensor, weight, bias, stride=(1, 1), padding=(0, 0)):
    """
    Custom Triton implementation of 2D convolution
    """
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Extract dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - kernel_h) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - kernel_w) // stride[1] + 1
    
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
    
    # Launch kernel
    grid = (grid_m, grid_n, grid_b)
    
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        input_height,
        input_width,
        output_height,
        output_width,
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        batch_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        # Initialize weights properly
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        # Use our custom Triton convolution instead of default PyTorch
        x = triton_conv2d(x, self.conv1.weight, self.conv1.bias, 
                         stride=self.conv1.stride, 
                         padding=self.conv1.padding)
        return x