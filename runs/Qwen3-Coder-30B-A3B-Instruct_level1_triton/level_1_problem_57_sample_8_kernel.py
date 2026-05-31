import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
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
    groups,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_h_idx = tl.program_id(1)
    out_w_idx = tl.program_id(2)
    out_c_idx = tl.program_id(3)
    
    # Calculate output position
    out_h_start = out_h_idx * BLOCK_SIZE_H
    out_w_start = out_w_idx * BLOCK_SIZE_W
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE_H + 2*padding_h, BLOCK_SIZE_W + 2*padding_w))
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over groups
    for g in range(groups):
        # Calculate group-specific indices
        group_in_ch = in_channels // groups
        group_out_ch = out_channels // groups
        
        # Get weight slice for this group
        weight_group_ptr = weight_ptr + g * group_out_ch * group_in_ch * kernel_h * kernel_w
        
        # Process kernel elements
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate input coordinates
                input_h_start = out_h_start * stride_h - padding_h + kh
                input_w_start = out_w_start * stride_w - padding_w + kw
                
                # Load input tile
                if input_h_start >= 0 and input_h_start < input_height and input_w_start >= 0 and input_w_start < input_width:
                    # Load input data for this group
                    input_ptr_group = input_ptr + batch_idx * in_channels * input_height * input_width + \
                                      g * group_in_ch * input_height * input_width
                    
                    # Load input values into shared memory
                    for ih in range(BLOCK_SIZE_H):
                        for iw in range(BLOCK_SIZE_W):
                            h = input_h_start + ih
                            w = input_w_start + iw
                            if h >= 0 and h < input_height and w >= 0 and w < input_width:
                                val = tl.load(input_ptr_group + h * input_width + w)
                                shared_input[ih + padding_h][iw + padding_w] = val
                            else:
                                shared_input[ih + padding_h][iw + padding_w] = 0.0
                
                # Compute convolution for this kernel position
                for ih in range(BLOCK_SIZE_H):
                    for iw in range(BLOCK_SIZE_W):
                        # Get weight value
                        weight_val = tl.load(weight_group_ptr + 
                                           (out_c_idx % group_out_ch) * group_in_ch * kernel_h * kernel_w + 
                                           (kh * kernel_w + kw) * group_in_ch + 
                                           (out_c_idx // group_out_ch) * group_in_ch)
                        
                        # Accumulate
                        acc[ih][iw] += shared_input[ih + padding_h][iw + padding_w] * weight_val
    
    # Store output
    for ih in range(BLOCK_SIZE_H):
        for iw in range(BLOCK_SIZE_W):
            h = out_h_start + ih
            w = out_w_start + iw
            if h < output_height and w < output_width:
                output_ptr += batch_idx * out_channels * output_height * output_width + \
                             out_c_idx * output_height * output_width + \
                             h * output_width + w
                tl.store(output_ptr, acc[ih][iw])

def triton_conv_transpose2d(input_tensor, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
    """
    Triton implementation of ConvTranspose2d
    """
    # Extract dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height - 1) * stride - 2 * padding + kernel_h + output_padding
    output_width = (input_width - 1) * stride - 2 * padding + kernel_w + output_padding
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 8
    
    # Grid configuration
    grid = (
        batch_size,
        (output_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (output_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W,
        (out_channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        input_tensor,
        weight,
        output,
        input_height,
        input_width,
        output_height,
        output_width,
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        stride,
        stride,
        padding,
        padding,
        groups,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    
    # Add bias if provided
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            output_padding=self.output_padding, 
            groups=self.groups
        )