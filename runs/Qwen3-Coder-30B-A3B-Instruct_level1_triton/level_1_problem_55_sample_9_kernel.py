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
    input_row_stride,
    input_col_stride,
    weight_row_stride,
    weight_col_stride,
    output_row_stride,
    output_col_stride,
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
    dilation_h,
    dilation_w,
    groups,
    group_size_in,
    group_size_out,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get program IDs
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    m_block_id = tl.program_id(2)
    
    # Calculate output dimensions
    num_blocks_m = tl.cdiv(output_height, BLOCK_SIZE_M)
    num_blocks_n = tl.cdiv(output_width, BLOCK_SIZE_N)
    
    # Calculate which group this thread block belongs to
    group_id = out_channel_id // (out_channels // groups)
    
    # Initialize pointers for this block
    out_c_offset = out_channel_id * output_row_stride * output_col_stride
    input_batch_offset = batch_id * in_channels * input_row_stride * input_col_stride
    
    # Shared memory for tiles
    tile_a = tl.shared_tensor(tl.arange(0, BLOCK_SIZE_K), dtype=tl.float32)
    tile_b = tl.shared_tensor(tl.arange(0, BLOCK_SIZE_K), dtype=tl.float32)
    
    # Loop over K dimension
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Process each kernel position
    for kh in range(0, kernel_height):
        for kw in range(0, kernel_width):
            # Compute positions
            input_h_start = m_block_id * BLOCK_SIZE_M - padding_h + kh * dilation_h
            input_w_start = 0 - padding_w + kw * dilation_w
            
            # Load weights
            weight_offset = group_id * group_size_out * group_size_in * weight_row_stride * weight_col_stride
            weight_idx = (out_channel_id % group_size_out) * weight_row_stride * weight_col_stride
            weight_idx += (kh * weight_col_stride + kw)
            
            # Check if valid kernel position
            if input_h_start >= 0 and input_h_start < input_height and \
               input_w_start >= 0 and input_w_start < input_width:
                
                # Load input tile
                input_offset = input_batch_offset + (kh * weight_row_stride * weight_col_stride) + \
                              (kw * weight_col_stride)
                
                # Compute valid ranges
                h_start = max(0, input_h_start)
                h_end = min(input_height, input_h_start + BLOCK_SIZE_M)
                w_start = max(0, input_w_start)
                w_end = min(input_width, input_w_start + BLOCK_SIZE_N)
                
                # Perform computation
                for m in range(h_start, h_end):
                    for n in range(w_start, w_end):
                        # Calculate indices
                        input_idx = input_batch_offset + m * input_row_stride + n * input_col_stride
                        weight_idx = group_id * group_size_out * group_size_in * weight_row_stride * weight_col_stride
                        weight_idx += (out_channel_id % group_size_out) * weight_row_stride * weight_col_stride
                        weight_idx += kh * weight_col_stride + kw
                        
                        # Load values
                        input_val = tl.load(input_ptr + input_idx, mask=(m < input_height) & (n < input_width))
                        weight_val = tl.load(weight_ptr + weight_idx)
                        
                        # Accumulate
                        accumulator[m - h_start, n - w_start] += input_val * weight_val
    
    # Write results to global memory
    output_offset = batch_id * out_channels * output_row_stride * output_col_stride + \
                   out_channel_id * output_row_stride * output_col_stride
    
    for m in range(BLOCK_SIZE_M):
        for n in range(BLOCK_SIZE_N):
            if m_block_id * BLOCK_SIZE_M + m < output_height and n < output_width:
                output_idx = output_offset + (m_block_id * BLOCK_SIZE_M + m) * output_row_stride + n * output_col_stride
                tl.store(output_ptr + output_idx, accumulator[m, n])

def triton_conv2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """
    Triton implementation of 2D convolution using custom kernel
    """
    # Ensure inputs are contiguous and on GPU
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid dimensions
    num_blocks_m = triton.cdiv(output_height, BLOCK_SIZE_M)
    num_blocks_n = triton.cdiv(output_width, BLOCK_SIZE_N)
    num_groups = triton.cdiv(out_channels, GROUP_SIZE_M)
    
    # Launch kernel
    grid = (batch_size, out_channels, num_blocks_m)
    
    # Get strides
    input_row_stride = input_tensor.stride(2)
    input_col_stride = input_tensor.stride(3)
    weight_row_stride = weight.stride(2)
    weight_col_stride = weight.stride(3)
    output_row_stride = output.stride(2)
    output_col_stride = output.stride(3)
    
    # Call kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        input_row_stride,
        input_col_stride,
        weight_row_stride,
        weight_col_stride,
        output_row_stride,
        output_col_stride,
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
        dilation[0],
        dilation[1],
        groups,
        in_channels // groups,
        out_channels // groups,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized 2D convolution using Triton kernels.
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
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using Triton kernel implementation
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

# Note: This implementation uses a simplified approach for demonstration purposes.
# A production version would require more sophisticated optimizations and proper handling
# of all edge cases in the convolution operation.