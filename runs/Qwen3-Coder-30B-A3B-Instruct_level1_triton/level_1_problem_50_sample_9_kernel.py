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
    GROUP_SIZE_M: tl.constexpr,
):
    # Get the program ID and determine which batch, output row, and output column this program handles
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    # Determine the tile indices for output dimensions
    tile_m_start = pid_m * BLOCK_SIZE_M
    tile_n_start = pid_n * BLOCK_SIZE_N
    
    # Check bounds for output dimensions
    if tile_m_start >= output_height or tile_n_start >= output_width:
        return
    
    # Loop over the K dimension (input channels and kernel elements)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Process all input channels and kernel elements
    for k in range(0, in_channels * kernel_h * kernel_w, BLOCK_SIZE_K):
        # Load input tile
        input_tile = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
        for i in range(BLOCK_SIZE_M):
            for j in range(BLOCK_SIZE_K):
                if k + j < in_channels * kernel_h * kernel_w:
                    # Compute the corresponding input coordinates
                    c = (k + j) // (kernel_h * kernel_w)
                    kh = (k + j) % (kernel_h * kernel_w) // kernel_w
                    kw = (k + j) % (kernel_h * kernel_w) % kernel_w
                    
                    # Calculate actual input coordinates
                    ih = tile_m_start * stride_h - padding_h + kh
                    iw = tile_n_start * stride_w - padding_w + kw
                    
                    # Check bounds
                    if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                        input_idx = pid_batch * input_height * input_width * in_channels + \
                                   ih * input_width * in_channels + \
                                   iw * in_channels + c
                        input_tile[i, j] = tl.load(input_ptr + input_idx, mask=True)
        
        # Load weight tile
        weight_tile = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
        for i in range(BLOCK_SIZE_K):
            for j in range(BLOCK_SIZE_N):
                if k + i < in_channels * kernel_h * kernel_w and j < out_channels:
                    # Compute the corresponding weight index
                    c = (k + i) // (kernel_h * kernel_w)
                    kh = (k + i) % (kernel_h * kernel_w) // kernel_w
                    kw = (k + i) % (kernel_h * kernel_w) % kernel_w
                    
                    weight_idx = j * in_channels * kernel_h * kernel_w + \
                                c * kernel_h * kernel_w + \
                                kh * kernel_w + kw
                    weight_tile[i, j] = tl.load(weight_ptr + weight_idx, mask=True)
        
        # Perform matrix multiplication
        acc += tl.dot(input_tile, weight_tile)
    
    # Apply bias
    bias_tile = tl.zeros((1, BLOCK_SIZE_N), dtype=tl.float32)
    for j in range(BLOCK_SIZE_N):
        if j < out_channels:
            bias_tile[0, j] = tl.load(bias_ptr + j, mask=True)
    
    acc += bias_tile
    
    # Write back results
    for i in range(BLOCK_SIZE_M):
        for j in range(BLOCK_SIZE_N):
            if tile_m_start + i < output_height and tile_n_start + j < output_width:
                output_idx = pid_batch * output_height * output_width * out_channels + \
                            (tile_m_start + i) * output_width * out_channels + \
                            (tile_n_start + j) * out_channels
                tl.store(output_ptr + output_idx + j, acc[i, j], mask=True)

def triton_conv2d(input_tensor, weight, bias, stride=(1, 1), padding=(0, 0)):
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Extract dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Compute output dimensions
    output_height = (input_height + 2 * padding[0] - kernel_h) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - kernel_w) // stride[1] + 1
    
    # Initialize output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Grid configuration
    grid = (
        batch_size,
        triton.cdiv(output_height, BLOCK_SIZE_M),
        triton.cdiv(output_width, BLOCK_SIZE_N)
    )
    
    # Launch kernel
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
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        # Initialize weights properly
        nn.init.kaiming_uniform_(self.conv1.weight, a=math.sqrt(5))
        fan_in = 3 * 11 * 11
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.conv1.bias, -bound, bound)

    def forward(self, x):
        x = triton_conv2d(x, self.conv1.weight, self.conv1.bias, 
                         stride=self.conv1.stride, padding=self.conv1.padding)
        return x