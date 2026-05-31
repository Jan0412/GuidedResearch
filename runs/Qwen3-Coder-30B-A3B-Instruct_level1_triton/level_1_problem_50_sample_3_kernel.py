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
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_OC: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_h_idx = tl.program_id(1)
    out_w_idx = tl.program_id(2)
    oc_block = tl.program_id(3)
    
    # Calculate output dimensions
    output_elements = output_height * output_width * out_channels
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE_H + 2*padding_h, BLOCK_SIZE_W + 2*padding_w, BLOCK_SIZE_C))
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_OC,), dtype=tl.float32)
    
    # Loop over input channels
    for c in range(0, in_channels, BLOCK_SIZE_C):
        # Load input tile with padding
        input_row = out_h_idx * stride_h - padding_h
        input_col = out_w_idx * stride_w - padding_w
        
        # Load input patch
        for ih in range(BLOCK_SIZE_H + 2*padding_h):
            for iw in range(BLOCK_SIZE_W + 2*padding_w):
                if (input_row + ih >= 0 and input_row + ih < input_height and 
                    input_col + iw >= 0 and input_col + iw < input_width):
                    # Check bounds for channel dimension
                    if c + tl.arange(0, BLOCK_SIZE_C) < in_channels:
                        idx = batch_idx * (input_height * input_width * in_channels) + \
                              (input_row + ih) * (input_width * in_channels) + \
                              (input_col + iw) * in_channels + c + tl.arange(0, BLOCK_SIZE_C)
                        shared_input[ih, iw, :] = tl.load(input_ptr + idx, mask=(c + tl.arange(0, BLOCK_SIZE_C) < in_channels))
                    else:
                        shared_input[ih, iw, :] = 0.0
                else:
                    shared_input[ih, iw, :] = 0.0
        
        # Load weights
        weight_offsets = oc_block * BLOCK_SIZE_OC * in_channels * kernel_h * kernel_w + \
                         tl.arange(0, BLOCK_SIZE_OC)[:, None, None] * (in_channels * kernel_h * kernel_w) + \
                         tl.arange(0, BLOCK_SIZE_C)[None, :, None] * (kernel_h * kernel_w) + \
                         tl.arange(0, kernel_h)[None, None, :] * kernel_w + \
                         tl.arange(0, kernel_w)[None, None, :]
        
        # Compute convolution
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                input_patch = shared_input[kh:kh+BLOCK_SIZE_H, kw:kw+BLOCK_SIZE_W, :]
                weight_vals = tl.load(weight_ptr + weight_offsets[:, kh, kw], mask=(oc_block * BLOCK_SIZE_OC + tl.arange(0, BLOCK_SIZE_OC) < out_channels))
                acc += tl.sum(input_patch * weight_vals, axis=(0, 1))
    
    # Add bias and store output
    if oc_block * BLOCK_SIZE_OC + tl.arange(0, BLOCK_SIZE_OC) < out_channels:
        bias_vals = tl.load(bias_ptr + oc_block * BLOCK_SIZE_OC + tl.arange(0, BLOCK_SIZE_OC))
        output_offset = batch_idx * (output_height * output_width * out_channels) + \
                       out_h_idx * (output_width * out_channels) + \
                       out_w_idx * out_channels + oc_block * BLOCK_SIZE_OC + tl.arange(0, BLOCK_SIZE_OC)
        tl.store(output_ptr + output_offset, acc + bias_vals, mask=(oc_block * BLOCK_SIZE_OC + tl.arange(0, BLOCK_SIZE_OC) < out_channels))

def triton_conv2d(input_tensor, weight, bias, stride=(1, 1), padding=(0, 0)):
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - kernel_h) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - kernel_w) // stride[1] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 32
    BLOCK_SIZE_OC = 32
    
    # Grid dimensions
    grid = (
        batch_size,          # batch dimension
        output_height,       # output height
        output_width,        # output width
        (out_channels + BLOCK_SIZE_OC - 1) // BLOCK_SIZE_OC  # output channels
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
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_OC=BLOCK_SIZE_OC
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        # Initialize weights and biases
        nn.init.kaiming_uniform_(self.conv1.weight, a=math.sqrt(5))
        fan_in = self.conv1.in_channels * self.conv1.kernel_size[0] * self.conv1.kernel_size[1]
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.conv1.bias, -bound, bound)

    def forward(self, x):
        # Replace the default conv2d with our Triton implementation
        return triton_conv2d(x, self.conv1.weight, self.conv1.bias, 
                            stride=self.conv1.stride, padding=self.conv1.padding)