import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, D, H, W)
    w_ptr,  # Weight tensor: (in_channels, out_channels, kD, kH, kW)
    b_ptr,  # Bias tensor: (out_channels,) or None
    out_ptr,  # Output tensor: (batch, out_channels, out_D, out_H, out_W)
    batch_size, in_channels, out_channels,
    depth, height, width,
    out_depth, out_height, out_width,
    kD, kH, kW,
    stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr,
):
    # Each block handles a portion of the output tensor
    pid_b = tl.program_id(0)  # batch index
    pid_oc = tl.program_id(1)  # output channel index
    pid_t = tl.program_id(2)  # thread index within block
    
    # Compute output position from pid_t
    # Flatten output spatial dimensions: out_D * out_H * out_W
    total_out_spatial = out_depth * out_height * out_width
    num_blocks = (total_out_spatial + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Determine which spatial position this thread processes
    spatial_idx = pid_t + (tl.program_id(2) // num_blocks) * BLOCK_SIZE
    if spatial_idx >= total_out_spatial:
        return
    
    # Decode spatial indices from flattened index
    out_d = spatial_idx // (out_height * out_width)
    rem = spatial_idx % (out_height * out_width)
    out_h = rem // out_width
    out_w = rem % out_width
    
    # Compute the starting position in input space
    # For transposed convolution: input_pos = output_pos - (kernel_pos - 1 - dilation*(kernel_size-1)) / stride + padding
    # More precisely: out_d = in_d * stride + (kD - 1 - dilation*(kD-1)) - 2*padding
    # So: in_d = (out_d + 2*padding - (kD - 1 - dilation*(kD-1))) / stride
    
    # Calculate the range of input positions that contribute to this output position
    # For each kernel position, compute corresponding input position
    sum_val = 0.0
    
    # Loop over input channels
    for ic in range(in_channels):
        # Loop over kernel positions
        for kd in range(kD):
            # Compute input depth index
            in_d = out_d - kd + dilation * (kD - 1) - padding
            # Adjust for stride
            in_d = (in_d + stride - 1) // stride  # ceiling division
            
            # Check if valid input depth
            if in_d >= 0 and in_d < depth:
                # Check if out_d matches exactly with stride computation
                if (in_d * stride + kd - dilation * (kD - 1) + padding) != out_d:
                    continue
                    
                for kh in range(kH):
                    in_h = out_h - kh + dilation * (kH - 1) - padding
                    in_h = (in_h + stride - 1) // stride
                    
                    if in_h >= 0 and in_h < height:
                        if (in_h * stride + kh - dilation * (kH - 1) + padding) != out_h:
                            continue
                            
                        for kw in range(kW):
                            in_w = out_w - kw + dilation * (kW - 1) - padding
                            in_w = (in_w + stride - 1) // stride
                            
                            if in_w >= 0 and in_w < width:
                                if (in_w * stride + kw - dilation * (kW - 1) + padding) != out_w:
                                    continue
                                
                                # Compute indices
                                x_idx = pid_b * (in_channels * depth * height * width) + \
                                        ic * (depth * height * width) + \
                                        in_d * (height * width) + \
                                        in_h * width + in_w
                                
                                w_idx = ic * (out_channels * kD * kH * kW) + \
                                        pid_oc * (kD * kH * kW) + \
                                        kd * (kH * kW) + \
                                        kh * kW + kw
                                
                                x_val = tl.load(x_ptr + x_idx)
                                w_val = tl.load(w_ptr + w_idx)
                                sum_val += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias_idx = pid_oc
        sum_val += tl.load(b_ptr + bias_idx)
    
    # Store result
    out_idx = pid_b * (out_channels * out_depth * out_height * out_width) + \
              pid_oc * (out_depth * out_height * out_width) + \
              out_d * (out_height * out_width) + \
              out_h * out_width + out_w
    tl.store(out_ptr + out_idx, sum_val)


def triton_conv_transpose3d(x, weight, bias, stride, padding, dilation):
    """
    Triton implementation of 3D transposed convolution.
    """
    batch_size, in_channels, depth, height, width = x.shape
    out_channels, in_channels_w, kD, kH, kW = weight.shape
    
    # Calculate output dimensions
    out_depth = (depth - 1) * stride - 2 * padding + dilation * (kD - 1) + 1
    out_height = (height - 1) * stride - 2 * padding + dilation * (kH - 1) + 1
    out_width = (width - 1) * stride - 2 * padding + dilation * (kW - 1) + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_depth, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Grid configuration
    BLOCK_SIZE = 256
    total_out_elements = batch_size * out_channels * out_depth * out_height * out_width
    num_blocks_per_channel = (total_out_elements // (batch_size * out_channels) + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Grid: (batch_size, out_channels, total_spatial_elements // BLOCK_SIZE + 1)
    grid = (batch_size, out_channels, (out_depth * out_height * out_width + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        depth, height, width,
        out_depth, out_height, out_width,
        kD, kH, kW,
        stride, padding, dilation,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized 3D transposed convolution using Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Register buffers for parameters (same as original)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        # Kaiming initialization similar to PyTorch's ConvTranspose3d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use our Triton kernel for the convolution
        return triton_conv_transpose3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)


# Import math for initialization
import math