import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    # Output tensor
    output_ptr,
    # Input tensor
    input_ptr,
    # Weight tensor
    weight_ptr,
    # Bias tensor (optional)
    bias_ptr,
    # Dimensions
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    groups,
    out_h,
    out_w,
    # Strides
    input_stride_b,
    input_stride_c,
    input_stride_h,
    input_stride_w,
    weight_stride_o,
    weight_stride_i,
    weight_stride_kh,
    weight_stride_kw,
    output_stride_b,
    output_stride_c,
    output_stride_h,
    output_stride_w,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr,  # Block size for intermediate computation
):
    # Get program IDs
    pid_batch = tl.program_id(1)
    pid_group = tl.program_id(0) // (tl.cdiv(out_channels // groups, BLOCK_SIZE_M))
    pid_c = tl.program_id(0) % (tl.cdiv(out_channels // groups, BLOCK_SIZE_M))
    
    # Calculate channel indices
    group_start = pid_group * (out_channels // groups)
    c_start = group_start + pid_c * BLOCK_SIZE_M
    c_range = tl.arange(0, BLOCK_SIZE_M)
    c_mask = c_start + c_range < group_start + (out_channels // groups)
    
    # Calculate batch index
    b = pid_batch
    
    # Iterate over output spatial positions
    for oh in range(out_h):
        for ow in range(out_w):
            # Accumulator for output
            acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
            
            # Iterate over input channels in the group
            for ic in range(in_channels // groups):
                # Calculate corresponding input position
                ih = oh - pad_h + ic * 0  # Will be updated below
                iw = ow - pad_w + ic * 0
                
                # For each kernel position
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        # Calculate input position considering dilation
                        input_h = (oh - pad_h + kh * dilation_h) // stride_h
                        input_w = (ow - pad_w + kw * dilation_w) // stride_w
                        
                        # Check if input position is valid
                        if input_h >= 0 and input_h < height and input_w >= 0 and input_w < width:
                            # Check if this kernel position maps to the current input position
                            if (oh - pad_h + kh * dilation_h) == input_h * stride_h and \
                               (ow - pad_w + kw * dilation_w) == input_w * stride_w:
                                # Load input value
                                input_val = tl.load(
                                    input_ptr + 
                                    b * input_stride_b + 
                                    (pid_group * (in_channels // groups) + ic) * input_stride_c + 
                                    input_h * input_stride_h + 
                                    input_w * input_stride_w,
                                    mask=(pid_group * (in_channels // groups) + ic) < (pid_group + 1) * (in_channels // groups)
                                )
                                
                                # Load weight value
                                weight_val = tl.load(
                                    weight_ptr + 
                                    (group_start + c_range) * weight_stride_o + 
                                    (pid_group * (in_channels // groups) + ic) * weight_stride_i + 
                                    kh * weight_stride_kh + 
                                    kw * weight_stride_kw,
                                    mask=c_mask
                                )
                                
                                # Accumulate
                                acc += input_val * weight_val
            
            # Add bias if provided
            if bias_ptr is not None:
                bias_val = tl.load(
                    bias_ptr + (group_start + c_range),
                    mask=c_mask
                )
                acc += bias_val
            
            # Store output
            tl.store(
                output_ptr + 
                b * output_stride_b + 
                (group_start + c_range) * output_stride_c + 
                oh * output_stride_h + 
                ow * output_stride_w,
                acc.to(tl.float32),
                mask=c_mask
            )


# For better performance, we'll use a more optimized version that processes multiple output positions
@triton.jit
def conv_transpose2d_kernel_optimized(
    # Output tensor
    output_ptr,
    # Input tensor
    input_ptr,
    # Weight tensor
    weight_ptr,
    # Bias tensor (optional)
    bias_ptr,
    # Dimensions
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    groups,
    out_h,
    out_w,
    # Strides
    input_stride_b,
    input_stride_c,
    input_stride_h,
    input_stride_w,
    weight_stride_o,
    weight_stride_i,
    weight_stride_kh,
    weight_stride_kw,
    output_stride_b,
    output_stride_c,
    output_stride_h,
    output_stride_w,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_OUT_H: tl.constexpr,  # Block size for output height
    BLOCK_SIZE_OUT_W: tl.constexpr,  # Block size for output width
):
    # Get program IDs
    pid_batch = tl.program_id(1)
    pid_group = tl.program_id(0) // (tl.cdiv(out_channels // groups, BLOCK_SIZE_M))
    pid_c = tl.program_id(0) % (tl.cdiv(out_channels // groups, BLOCK_SIZE_M))
    
    # Calculate channel indices
    group_start = pid_group * (out_channels // groups)
    c_start = group_start + pid_c * BLOCK_SIZE_M
    c_range = tl.arange(0, BLOCK_SIZE_M)
    c_mask = c_start + c_range < group_start + (out_channels // groups)
    
    # Calculate batch index
    b = pid_batch
    
    # Iterate over output spatial positions in blocks
    start_oh = tl.program_id(2) * BLOCK_SIZE_OUT_H
    start_ow = tl.program_id(3) * BLOCK_SIZE_OUT_W
    
    for oh_block in range(BLOCK_SIZE_OUT_H):
        oh = start_oh + oh_block
        if oh >= out_h:
            break
            
        for ow_block in range(BLOCK_SIZE_OUT_W):
            ow = start_ow + ow_block
            if ow >= out_w:
                break
                
            # Accumulator for output
            acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
            
            # Iterate over input channels in the group
            for ic in range(in_channels // groups):
                # For each kernel position
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        # Calculate input position considering dilation
                        input_h = (oh - pad_h + kh * dilation_h) // stride_h
                        input_w = (ow - pad_w + kw * dilation_w) // stride_w
                        
                        # Check if input position is valid
                        if input_h >= 0 and input_h < height and input_w >= 0 and input_w < width:
                            # Check if this kernel position maps to the current input position
                            if (oh - pad_h + kh * dilation_h) == input_h * stride_h and \
                               (ow - pad_w + kw * dilation_w) == input_w * stride_w:
                                # Load input value
                                input_val = tl.load(
                                    input_ptr + 
                                    b * input_stride_b + 
                                    (pid_group * (in_channels // groups) + ic) * input_stride_c + 
                                    input_h * input_stride_h + 
                                    input_w * input_stride_w,
                                    mask=(pid_group * (in_channels // groups) + ic) < (pid_group + 1) * (in_channels // groups)
                                )
                                
                                # Load weight value
                                weight_val = tl.load(
                                    weight_ptr + 
                                    (group_start + c_range) * weight_stride_o + 
                                    (pid_group * (in_channels // groups) + ic) * weight_stride_i + 
                                    kh * weight_stride_kh + 
                                    kw * weight_stride_kw,
                                    mask=c_mask
                                )
                                
                                # Accumulate
                                acc += input_val * weight_val
            
            # Add bias if provided
            if bias_ptr is not None:
                bias_val = tl.load(
                    bias_ptr + (group_start + c_range),
                    mask=c_mask
                )
                acc += bias_val
            
            # Store output
            tl.store(
                output_ptr + 
                b * output_stride_b + 
                (group_start + c_range) * output_stride_c + 
                oh * output_stride_h + 
                ow * output_stride_w,
                acc.to(tl.float32),
                mask=c_mask
            )


# Simpler, more efficient implementation using direct indexing
@triton.jit
def conv_transpose2d_kernel_simple(
    # Output tensor
    output_ptr,
    # Input tensor
    input_ptr,
    # Weight tensor
    weight_ptr,
    # Bias tensor (optional)
    bias_ptr,
    # Dimensions
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    groups,
    out_h,
    out_w,
    # Strides
    input_stride_b,
    input_stride_c,
    input_stride_h,
    input_stride_w,
    weight_stride_o,
    weight_stride_i,
    weight_stride_kh,
    weight_stride_kw,
    output_stride_b,
    output_stride_c,
    output_stride_h,
    output_stride_w,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
):
    # Get program IDs
    pid_batch = tl.program_id(1)
    pid_c = tl.program_id(0)  # Output channel block
    
    # Calculate channel indices
    c_start = pid_c * BLOCK_SIZE_M
    c_range = tl.arange(0, BLOCK_SIZE_M)
    c_mask = c_start + c_range < out_channels
    
    # Calculate batch index
    b = pid_batch
    
    # Iterate over output spatial positions
    for oh in range(out_h):
        for ow in range(out_w):
            # Accumulator for output
            acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
            
            # Iterate over groups
            for g in range(groups):
                group_in_channels = in_channels // groups
                group_out_channels = out_channels // groups
                
                # Iterate over input channels in this group
                for ic in range(group_in_channels):
                    # Calculate input position for this group
                    input_ic = g * group_in_channels + ic
                    
                    # For each kernel position
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            # Calculate input position considering dilation
                            input_h = (oh - pad_h + kh * dilation_h) // stride_h
                            input_w = (ow - pad_w + kw * dilation_w) // stride_w
                            
                            # Check if input position is valid
                            if input_h >= 0 and input_h < height and input_w >= 0 and input_w < width:
                                # Check if this kernel position maps to the current input position
                                if (oh - pad_h + kh * dilation_h) == input_h * stride_h and \
                                   (ow - pad_w + kw * dilation_w) == input_w * stride_w:
                                    # Load input value
                                    input_val = tl.load(
                                        input_ptr + 
                                        b * input_stride_b + 
                                        input_ic * input_stride_c + 
                                        input_h * input_stride_h + 
                                        input_w * input_stride_w
                                    )
                                    
                                    # Calculate weight indices
                                    out_c_idx = g * group_out_channels + (pid_c * BLOCK_SIZE_M) % group_out_channels
                                    weight_c_idx = (pid_c * BLOCK_SIZE_M) % group_out_channels
                                    
                                    # Load weight value (only if within bounds)
                                    weight_val = tl.load(
                                        weight_ptr + 
                                        out_c_idx * weight_stride_o + 
                                        input_ic * weight_stride_i + 
                                        kh * weight_stride_kh + 
                                        kw * weight_stride_kw,
                                        mask=(pid_c * BLOCK_SIZE_M + weight_c_idx) < out_channels
                                    )
                                    
                                    # Accumulate
                                    acc += input_val * weight_val
            
            # Add bias if provided
            if bias_ptr is not None:
                bias_val = tl.load(
                    bias_ptr + c_start + c_range,
                    mask=c_mask
                )
                acc += bias_val
            
            # Store output
            tl.store(
                output_ptr + 
                b * output_stride_b + 
                (c_start + c_range) * output_stride_c + 
                oh * output_stride_h + 
                ow * output_stride_w,
                acc.to(tl.float32),
                mask=c_mask
            )


# The most efficient implementation: process all output positions in parallel
@triton.jit
def conv_transpose2d_kernel_final(
    # Output tensor
    output_ptr,
    # Input tensor
    input_ptr,
    # Weight tensor
    weight_ptr,
    # Bias tensor (optional)
    bias_ptr,
    # Dimensions
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    groups,
    out_h,
    out_w,
    # Strides
    input_stride_b,
    input_stride_c,
    input_stride_h,
    input_stride_w,
    weight_stride_o,
    weight_stride_i,
    weight_stride_kh,
    weight_stride_kw,
    output_stride_b,
    output_stride_c,
    output_stride_h,
    output_stride_w,
):
    # Get program IDs
    pid_batch = tl.program_id(2)
    pid_c = tl.program_id(1)  # Output channel
    pid_oh_ow = tl.program_id(0)
    
    # Calculate output spatial position
    oh = pid_oh_ow // out_w
    ow = pid_oh_ow % out_w
    
    # Check bounds
    if pid_batch >= batch_size or pid_c >= out_channels or oh >= out_h or ow >= out_w:
        return
    
    # Calculate group info
    group_id = pid_c // (out_channels // groups)
    group_out_channels = out_channels // groups
    group_in_channels = in_channels // groups
    
    # Accumulator
    acc = 0.0
    
    # Iterate over input channels in the group
    for ic in range(group_in_channels):
        input_ic = group_id * group_in_channels + ic
        
        # For each kernel position
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate input position considering dilation
                input_h = (oh - pad_h + kh * dilation_h) // stride_h
                input_w = (ow - pad_w + kw * dilation_w) // stride_w
                
                # Check if input position is valid
                if input_h >= 0 and input_h < height and input_w >= 0 and input_w < width:
                    # Check if this kernel position maps to the current input position
                    if (oh - pad_h + kh * dilation_h) == input_h * stride_h and \
                       (ow - pad_w + kw * dilation_w) == input_w * stride_w:
                        # Load input value
                        input_val = tl.load(
                            input_ptr + 
                            pid_batch * input_stride_b + 
                            input_ic * input_stride_c + 
                            input_h * input_stride_h + 
                            input_w * input_stride_w
                        )
                        
                        # Load weight value
                        weight_val = tl.load(
                            weight_ptr + 
                            pid_c * weight_stride_o + 
                            input_ic * weight_stride_i + 
                            kh * weight_stride_kh + 
                            kw * weight_stride_kw
                        )
                        
                        # Accumulate
                        acc += input_val * weight_val
    
    # Add bias if provided
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + pid_c)
        acc += bias_val
    
    # Store output
    tl.store(
        output_ptr + 
        pid_batch * output_stride_b + 
        pid_c * output_stride_c + 
        oh * output_stride_h + 
        ow * output_stride_w,
        acc
    )


def triton_conv_transpose2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: tuple = (1, 1),
    padding: tuple = (0, 0),
    dilation: tuple = (1, 1),
    groups: int = 1,
):
    """
    Performs 2D transposed convolution using Triton kernel.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation
    
    # Calculate output dimensions
    out_h = (height - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + 1
    out_w = (width - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Calculate strides
    input_stride_b, input_stride_c, input_stride_h, input_stride_w = x.stride()
    weight_stride_o, weight_stride_i, weight_stride_kh, weight_stride_kw = weight.stride()
    output_stride_b, output_stride_c, output_stride_h, output_stride_w = out.stride()
    
    # Grid dimensions: [total_output_positions, out_channels, batch_size]
    total_out_positions = out_h * out_w
    grid = (total_out_positions, out_channels, batch_size)
    
    # Launch kernel
    conv_transpose2d_kernel_final[grid](
        out, x, weight, bias,
        batch_size, in_channels, out_channels,
        height, width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        groups,
        out_h, out_w,
        input_stride_b, input_stride_c, input_stride_h, input_stride_w,
        weight_stride_o, weight_stride_i, weight_stride_kh, weight_stride_kw,
        output_stride_b, output_stride_c, output_stride_h, output_stride_w,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the 2D transposed convolution with asymmetric input, 
    asymmetric kernel, grouped, padded, and dilated operations using Triton.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), 
                 padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create weight and bias parameters (same as nn.ConvTranspose2d)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size[0], kernel_size[1])
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize parameters similar to nn.ConvTranspose2d."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 2D transposed convolution.
        """
        # Handle case where x might not be contiguous
        x = x.contiguous()
        
        # Use our optimized Triton kernel
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )


# Import math for kaiming initialization
import math