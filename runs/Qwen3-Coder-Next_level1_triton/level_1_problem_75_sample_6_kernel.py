import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, height, width)
    w_ptr,  # Weight tensor: (in_channels, out_channels // groups, kH, kW)
    b_ptr,  # Bias tensor: (out_channels,)
    out_ptr,  # Output tensor: (batch, out_channels, out_h, out_w)
    batch_size, in_channels, out_channels,
    in_h, in_w, out_h, out_w,
    kH, kW, 
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    groups,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    
    # Calculate output position
    out_ch_start = pid_out_ch * BLOCK_SIZE_N
    out_ch_offsets = out_ch_start + tl.arange(0, BLOCK_SIZE_N)
    out_ch_mask = out_ch_offsets < out_channels
    
    # Output tensor offset for this batch
    out_batch_offset = pid_batch * out_channels * out_h * out_w
    
    # For each output position (oh, ow)
    for oh in range(out_h):
        for ow in range(out_w):
            # Initialize accumulator
            acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
            
            # Calculate input position corresponding to (oh, ow)
            # For transposed conv: in_h = (out_h - 1) * stride_h - 2*pad_h + dil_h*(kH-1) + 1
            # But we need to handle the convolution math correctly
            
            # Input coordinates that contribute to output (oh, ow)
            # in_h_pos = oh - dil_h*(kH-1) + pad_h - stride_h*something
            # Actually, for transposed conv: out = X * W, so 
            # out[oh, ow] = sum_{kh, kw} in[(oh - oh0 - stride_h*kh)/stride_h, ...] * W[...]
            # Let's compute which input positions contribute to output (oh, ow)
            
            # For transposed convolution:
            # out[oh, ow] = sum_{kh, kw} input[(oh - (kH-1-dil_h*kh) - pad_h)/stride_h, (ow - (kW-1-dil_w*kw) - pad_w)/stride_w] * weight[kh, kw, : :]
            # Only when indices are integers and in bounds
            
            # Iterate over input channels and kernel positions
            for in_ch in range(in_channels):
                # Group calculation
                group_idx = in_ch // (in_channels // groups)
                group_out_start = group_idx * (out_channels // groups)
                
                # Iterate over kernel positions
                for kh in range(kH):
                    for kw in range(kW):
                        # Calculate input position for this kernel element
                        # For transposed conv: input_pos = (output_pos - kernel_pos + padding - dilation*(kernel_size-1)) / stride
                        # But more accurately: output[oh, ow] += input[ih, iw] * weight[in_ch, out_ch, kh, kw]
                        # where ih = (oh - (kH-1-kh*dil_h) - pad_h) / stride_h, similarly for iw
                        
                        ih = (oh - (kH - 1 - kh * dil_h) - pad_h)
                        iw = (ow - (kW - 1 - kw * dil_w) - pad_w)
                        
                        # Check if this input position is valid (divisible by stride and in bounds)
                        if ih % stride_h == 0 and iw % stride_w == 0:
                            ih_strided = ih // stride_h
                            iw_strided = iw // stride_w
                            
                            # Check bounds
                            if 0 <= ih_strided < in_h and 0 <= iw_strided < in_w:
                                # Calculate input pointer offset
                                in_offset = (pid_batch * in_channels * in_h * in_w + 
                                            in_ch * in_h * in_w + 
                                            ih_strided * in_w + 
                                            iw_strided)
                                
                                # Load input value
                                x_val = tl.load(x_ptr + in_offset)
                                
                                # Calculate weight pointer offset
                                # Weight layout: [in_channels, out_channels // groups, kH, kW]
                                # But PyTorch uses [in_channels, out_channels // groups, kH, kW] for groups > 1
                                # Actually PyTorch uses: [in_channels, out_channels // groups, kH, kW]
                                # For grouped conv: weight[ic, oc_per_group, kh, kw]
                                # where oc_per_group = out_channels // groups
                                
                                # For current input channel, find corresponding output channels in same group
                                for oc_offset in range(BLOCK_SIZE_N):
                                    if out_ch_offsets[oc_offset] < out_channels:
                                        oc = out_ch_offsets[oc_offset]
                                        # Check if this output channel belongs to current group
                                        if group_idx == (oc // (out_channels // groups)):
                                            # Calculate weight offset
                                            # Weight index: [in_ch, oc % (out_channels // groups), kh, kw]
                                            oc_in_group = oc % (out_channels // groups)
                                            w_offset = (in_ch * (out_channels // groups) * kH * kW +
                                                       oc_in_group * kH * kW +
                                                       kh * kW +
                                                       kw)
                                            w_val = tl.load(w_ptr + w_offset)
                                            acc = acc + x_val * w_val * tl.cast(1, tl.float32)
            
            # Add bias if present
            if b_ptr is not None:
                for oc_offset in range(BLOCK_SIZE_N):
                    if out_ch_offsets[oc_offset] < out_channels:
                        bias_val = tl.load(b_ptr + out_ch_offsets[oc_offset])
                        acc = acc + bias_val
            
            # Store results
            for oc_offset in range(BLOCK_SIZE_N):
                if out_ch_offsets[oc_offset] < out_channels:
                    out_offset = (out_batch_offset + 
                                 out_ch_offsets[oc_offset] * out_h * out_w +
                                 oh * out_w + 
                                 ow)
                    tl.store(out_ptr + out_offset, acc[oc_offset])


# Optimized kernel using tiling for better performance
@triton.jit
def conv_transpose2d_kernel_optimized(
    x_ptr,  # Input tensor: (batch, in_channels, height, width)
    w_ptr,  # Weight tensor: (in_channels, out_channels // groups, kH, kW)
    b_ptr,  # Bias tensor: (out_channels,)
    out_ptr,  # Output tensor: (batch, out_channels, out_h, out_w)
    batch_size, in_channels, out_channels,
    in_h, in_w, out_h, out_w,
    kH, kW, 
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    groups,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    
    # Calculate output position
    out_ch_start = pid_out_ch * BLOCK_SIZE_N
    out_ch_offsets = out_ch_start + tl.arange(0, BLOCK_SIZE_N)
    out_ch_mask = out_ch_offsets < out_channels
    
    # Output tensor offset for this batch
    out_batch_offset = pid_batch * out_channels * out_h * out_w
    
    # For each output position (oh, ow)
    for oh in range(out_h):
        for ow in range(out_w):
            # Initialize accumulator
            acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
            
            # Iterate over input channels
            for in_ch in range(in_channels):
                # Group calculation
                group_idx = in_ch // (in_channels // groups)
                
                # Iterate over kernel positions
                for kh in range(kH):
                    for kw in range(kW):
                        # Calculate input position for this kernel element
                        ih = (oh - (kH - 1 - kh * dil_h) - pad_h)
                        iw = (ow - (kW - 1 - kw * dil_w) - pad_w)
                        
                        # Check if this input position is valid
                        if ih % stride_h == 0 and iw % stride_w == 0:
                            ih_strided = ih // stride_h
                            iw_strided = iw // stride_w
                            
                            # Check bounds
                            if 0 <= ih_strided < in_h and 0 <= iw_strided < in_w:
                                # Calculate input pointer offset
                                in_offset = (pid_batch * in_channels * in_h * in_w + 
                                            in_ch * in_h * in_w + 
                                            ih_strided * in_w + 
                                            iw_strided)
                                
                                # Load input value
                                x_val = tl.load(x_ptr + in_offset)
                                
                                # Calculate weight pointer offset
                                # Weight layout: [in_channels, out_channels // groups, kH, kW]
                                for oc_offset in range(BLOCK_SIZE_N):
                                    if out_ch_offsets[oc_offset] < out_channels:
                                        oc = out_ch_offsets[oc_offset]
                                        # Check if this output channel belongs to current group
                                        if group_idx == (oc // (out_channels // groups)):
                                            # Calculate weight offset
                                            oc_in_group = oc % (out_channels // groups)
                                            w_offset = (in_ch * (out_channels // groups) * kH * kW +
                                                       oc_in_group * kH * kW +
                                                       kh * kW +
                                                       kw)
                                            w_val = tl.load(w_ptr + w_offset)
                                            acc = acc + x_val * w_val
    
            # Add bias if present
            if b_ptr is not None:
                for oc_offset in range(BLOCK_SIZE_N):
                    if out_ch_offsets[oc_offset] < out_channels:
                        bias_val = tl.load(b_ptr + out_ch_offsets[oc_offset])
                        acc = acc + bias_val
            
            # Store results
            for oc_offset in range(BLOCK_SIZE_N):
                if out_ch_offsets[oc_offset] < out_channels:
                    out_offset = (out_batch_offset + 
                                 out_ch_offsets[oc_offset] * out_h * out_w +
                                 oh * out_w + 
                                 ow)
                    tl.store(out_ptr + out_offset, acc[oc_offset])


def triton_conv_transpose2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                           stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """
    Triton implementation of 2D transposed convolution.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Get dimensions
    batch_size, in_channels, in_h, in_w = x.shape
    out_channels = weight.shape[1] * groups  # weight shape: [in_channels, out_channels//groups, kH, kW]
    kH, kW = weight.shape[2], weight.shape[3]
    
    # Calculate output dimensions
    out_h = (in_h - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kH - 1) + 1
    out_w = (in_w - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kW - 1) + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Set up kernel parameters
    BLOCK_SIZE_M = 1
    BLOCK_SIZE_N = 32  # Can be tuned
    BLOCK_SIZE_K = 1
    
    # Calculate grid dimensions
    grid = (batch_size, (out_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    
    # Launch kernel
    conv_transpose2d_kernel_optimized[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        in_h, in_w, out_h, out_w,
        kH, kW,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        groups,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), 
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create weight and bias parameters (same as PyTorch's ConvTranspose2d)
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1])
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using custom Triton kernel.
        """
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation, 
            groups=self.groups
        )