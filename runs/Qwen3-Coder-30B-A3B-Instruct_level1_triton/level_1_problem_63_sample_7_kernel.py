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
    batch_size, in_channels, out_channels, input_height, input_width,
    kernel_height, kernel_width, output_height, output_width,
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    groups, bias_enabled,
    BLOCK_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    out_row_id = tl.program_id(2)
    
    # Calculate global thread index within block
    thread_id = tl.thread_id()
    
    # Shared memory for weights (for group-wise processing)
    shared_weight = tl.shared_ptr(weight_ptr, shape=(GROUPS_PER_BLOCK, kernel_height, kernel_width, in_channels // groups, out_channels // groups))
    
    # Process one output channel per block
    if out_channel_id >= out_channels:
        return
        
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over groups
    for g in range(0, groups, GROUPS_PER_BLOCK):
        # Process multiple groups per block
        group_offset = g
        group_count = min(GROUPS_PER_BLOCK, groups - g)
        
        # Load weights for this group
        for i in range(group_count):
            if i + g < groups:
                weight_base = weight_ptr + (i + g) * weight_stride_0
                for kh in range(kernel_height):
                    for kw in range(kernel_width):
                        for ic in range(in_channels // groups):
                            shared_weight[i, kh, kw, ic, 0] = tl.load(weight_base + kh * weight_stride_2 + kw * weight_stride_3 + ic * weight_stride_1)
        
        # Process output row
        for oh in range(out_row_id, output_height, tl.num_programs(2)):
            # Process multiple output pixels per thread
            for ow in range(thread_id, output_width, BLOCK_SIZE):
                if ow >= output_width:
                    continue
                    
                # Compute input coordinates
                ih_start = oh * stride_h - padding_h
                iw_start = ow * stride_w - padding_w
                
                # Accumulate across kernel and input channels
                for kh in range(kernel_height):
                    for kw in range(kernel_width):
                        for ic in range(in_channels // groups):
                            ih = ih_start + kh * dilation_h
                            iw = iw_start + kw * dilation_w
                            
                            # Check bounds
                            if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                                # Load input value
                                input_val = tl.load(input_ptr + 
                                                  batch_id * input_stride_0 +
                                                  (ic + group_offset * (in_channels // groups)) * input_stride_1 +
                                                  ih * input_stride_2 +
                                                  iw * input_stride_3)
                                
                                # Load weight
                                weight_val = tl.load(weight_ptr + 
                                                   (group_offset * (out_channels // groups) + out_channel_id % (out_channels // groups)) * weight_stride_0 +
                                                   kh * weight_stride_2 +
                                                   kw * weight_stride_3 +
                                                   ic * weight_stride_1)
                                
                                acc[thread_id] += input_val * weight_val
    
    # Reduce within block
    for i in range(BLOCK_SIZE // 2):
        acc[i] += acc[i + BLOCK_SIZE // 2]
    
    # Write result
    if thread_id == 0:
        output_base = output_ptr + batch_id * output_stride_0 + out_channel_id * output_stride_1 + out_row_id * output_stride_2
        for ow in range(output_width):
            val = acc[0]
            if bias_enabled:
                val += tl.load(bias_ptr + out_channel_id)
            tl.store(output_base + ow * output_stride_3, val)

def triton_conv2d(input_tensor, weight, bias, stride, padding, dilation, groups):
    """
    Custom Triton implementation of 2D convolution.
    """
    # Ensure tensors are on GPU
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE = 256
    GROUPS_PER_BLOCK = 4
    
    # Grid configuration
    grid = (
        batch_size,          # batch dimension
        out_channels,        # output channel dimension  
        (output_height + 7) // 8  # output rows (with 8 threads per block)
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        batch_size, in_channels, out_channels, input_height, input_width,
        kernel_height, kernel_width, output_height, output_width,
        stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
        groups, bias is not None,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUPS_PER_BLOCK=GROUPS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv2d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )