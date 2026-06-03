import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    # Pointers to inputs and outputs
    x_ptr,  # [B, IC, H_in, W_in]
    w_ptr,  # [IC, OC//G, kH, kW] (transposed conv weight)
    b_ptr,  # [OC] or None
    out_ptr,  # [B, OC, H_out, W_out]
    # Dimensions
    B, IC, OC, G,  # Batch, input channels, output channels, groups
    H_in, W_in,  # Input height and width
    H_out, W_out,  # Output height and width
    kH, kW,  # Kernel height and width
    stride_h, stride_w,  # Stride
    pad_h, pad_w,  # Padding
    out_pad_h, out_pad_w,  # Output padding
    # Strides in memory
    stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_w_oc, stride_w_ic, stride_w_kh, stride_w_kw,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_KH: tl.constexpr,  # Block size for kernel height
    BLOCK_SIZE_KW: tl.constexpr,  # Block size for kernel width
    BLOCK_SIZE_OC: tl.constexpr,  # Block size for output channel dimension in matmul
    HAS_BIAS: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)  # Batch index
    pid_oc = tl.program_id(1)  # Output channel block index
    pid_h = tl.program_id(2)  # Output height index
    pid_w = tl.program_id(3)  # Output width index
    
    # Calculate the starting positions
    oc_start = pid_oc * BLOCK_SIZE_OC
    h_out_start = pid_h * stride_h - pad_h + out_pad_h // 2
    w_out_start = pid_w * stride_w - pad_w + out_pad_w // 2
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_OC), dtype=tl.float32)
    
    # For transposed convolution, each output position accumulates from input positions
    # based on the kernel positions
    for kh in range(kH):
        h_in = h_out_start + kh
        if h_in >= 0 and h_in < H_in:
            for kw in range(kW):
                w_in = w_out_start + kw
                if w_in >= 0 and w_in < W_in:
                    # Calculate input indices
                    offsets_x_b = pid_b * stride_x_b
                    offsets_x_c = tl.arange(0, BLOCK_SIZE_M)
                    offsets_x_h = h_in * stride_x_h
                    offsets_x_w = w_in * stride_x_w
                    
                    # Load input values: [BLOCK_SIZE_M, 1]
                    x_vals = tl.load(
                        x_ptr + offsets_x_b + offsets_x_c * stride_x_c + offsets_x_h + offsets_x_w,
                        mask=offsets_x_c < IC,
                        other=0.0
                    )
                    
                    # Calculate weight indices
                    # For transposed conv: weight shape is [IC, OC//G, kH, kW]
                    # But we're organizing by output channels, so we need to load weight[ic, oc, kh, kw]
                    # where oc = pid_oc * BLOCK_SIZE_OC + local_oc
                    
                    # Load weights for this kernel position
                    # weights[ic, oc, kh, kw] -> need to access for each ic and oc
                    
                    # Transpose weight for easier access: load [BLOCK_SIZE_M, BLOCK_SIZE_OC]
                    # Actually, let's do it differently - iterate over input channels
                    
                    # We need to compute: out[oc] += sum_ic x[ic] * w[ic, oc, kh, kw]
                    # So for each input channel ic, we need w[ic, oc_start:oc_start+BLOCK_SIZE_OC, kh, kw]
                    
                    # Load weights for all input channels in the block and output channels in the block
                    offsets_w_ic = tl.arange(0, BLOCK_SIZE_M)
                    offsets_w_oc = oc_start + tl.arange(0, BLOCK_SIZE_OC)
                    
                    # Compute weight pointer offset for this kernel position
                    w_offset_kh = kh * stride_w_kh
                    w_offset_kw = kw * stride_w_kw
                    
                    # Load weights: shape [BLOCK_SIZE_M, BLOCK_SIZE_OC]
                    w_vals = tl.load(
                        w_ptr + offsets_w_ic[:, None] * stride_w_ic + offsets_w_oc[None, :] * stride_w_oc + w_offset_kh + w_offset_kw,
                        mask=(offsets_w_ic < IC)[:, None] & (offsets_w_oc < OC)[None, :],
                        other=0.0
                    )
                    
                    # Compute accumulation: x_vals [BLOCK_SIZE_M, 1] * w_vals [BLOCK_SIZE_M, BLOCK_SIZE_OC]
                    # We need x_vals broadcasted properly
                    acc += tl.dot(x_vals[:, None], w_vals[None, :])
    
    # Store result with bias if present
    if HAS_BIAS:
        # Load bias: [OC]
        bias_vals = tl.load(b_ptr + tl.arange(0, BLOCK_SIZE_OC), mask=tl.arange(0, BLOCK_SIZE_OC) < OC)
        acc += bias_vals[None, :]
    
    # Convert to output type and store
    acc = acc.to(out_ptr.type.element_ty)
    
    # Store output
    offsets_out_b = pid_b * stride_out_b
    offsets_out_c = oc_start + tl.arange(0, BLOCK_SIZE_OC)
    offsets_out_h = pid_h * stride_out_h
    offsets_out_w = pid_w * stride_out_w
    
    tl.store(
        out_ptr + offsets_out_b + offsets_out_c * stride_out_c + offsets_out_h + offsets_out_w,
        acc,
        mask=offsets_out_c < OC
    )


# A more practical implementation using a different approach
# For transposed convolution, we can think of it as regular convolution with expanded input
# But for performance, we'll implement a direct kernel

@triton.jit
def conv_transpose2d_kernel_optimized(
    x_ptr,  # [B, IC, H_in, W_in]
    w_ptr,  # [IC, OC//G, kH, kW]
    b_ptr,  # [OC] or None
    out_ptr,  # [B, OC, H_out, W_out]
    B, IC, OC, G,
    H_in, W_in, H_out, W_out,
    kH, kW,
    stride_h, stride_w,
    pad_h, pad_w,
    out_pad_h, out_pad_w,
    stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_w_ic, stride_w_oc, stride_w_kh, stride_w_kw,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_OC: tl.constexpr,
    BLOCK_SIZE_IC: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # Calculate output position
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate output channel range
    oc_start = pid_oc * BLOCK_SIZE_OC
    oc_range = tl.arange(0, BLOCK_SIZE_OC) + oc_start
    
    # Calculate output position in input coordinates
    h_out = pid_h
    w_out = pid_w
    h_in_base = h_out * stride_h - pad_h + out_pad_h // 2
    w_in_base = w_out * stride_w - pad_w + out_pad_w // 2
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_OC,), dtype=tl.float32)
    
    # Iterate over input channels
    for ic_start in range(0, IC, BLOCK_SIZE_IC):
        ic_range = tl.arange(0, BLOCK_SIZE_IC) + ic_start
        mask_ic = ic_range < IC
        
        # Iterate over kernel positions
        for kh in range(kH):
            h_in = h_in_base + kh
            if h_in >= 0 and h_in < H_in:
                for kw in range(kW):
                    w_in = w_in_base + kw
                    if w_in >= 0 and w_in < W_in:
                        # Load input: [BLOCK_SIZE_IC]
                        x_offset = (pid_b * stride_x_b + 
                                   ic_range * stride_x_c + 
                                   h_in * stride_x_h + 
                                   w_in * stride_x_w)
                        x_vals = tl.load(x_ptr + x_offset, mask=mask_ic)
                        
                        # Load weights: [BLOCK_SIZE_IC, BLOCK_SIZE_OC]
                        w_offset = (ic_range[:, None] * stride_w_ic +
                                   oc_range[None, :] * stride_w_oc +
                                   kh * stride_w_kh +
                                   kw * stride_w_kw)
                        w_vals = tl.load(w_ptr + w_offset, 
                                        mask=(mask_ic[:, None] & (oc_range < OC)[None, :]))
                        
                        # Accumulate: sum_ic x[ic] * w[ic, oc]
                        acc += tl.sum(x_vals[:, None] * w_vals, axis=0)
    
    # Add bias
    if HAS_BIAS:
        bias_vals = tl.load(b_ptr + oc_range, mask=oc_range < OC)
        acc += bias_vals
    
    # Store output
    out_offset = (pid_b * stride_out_b + 
                 oc_range * stride_out_c + 
                 pid_h * stride_out_h + 
                 pid_w * stride_out_w)
    tl.store(out_ptr + out_offset, acc.to(out_ptr.type.element_ty), mask=oc_range < OC)


def triton_conv_transpose2d(x, weight, bias, stride, padding, output_padding, groups):
    """
    Triton implementation of transposed 2D convolution.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, IC, H_in, W_in = x.shape
    IC_, OC, kH, kW = weight.shape  # weight shape for conv_transpose: [IC, OC, kH, kW]
    
    # Calculate output dimensions
    H_out = (H_in - 1) * stride[0] - 2 * padding[0] + kH + output_padding[0]
    W_out = (W_in - 1) * stride[1] - 2 * padding[1] + kW + output_padding[1]
    
    # Create output tensor
    out = torch.empty(B, OC, H_out, W_out, device=x.device, dtype=x.dtype)
    
    # Calculate strides
    stride_x = x.stride()
    stride_w = weight.stride()
    stride_out = out.stride()
    
    # Convert to tuple if needed
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding)
    
    # Grid dimensions: [batch, output_channel_blocks, output_height, output_width]
    BLOCK_SIZE_B = 1
    BLOCK_SIZE_OC = 32
    BLOCK_SIZE_IC = 32
    
    grid = (
        B,  # batch
        (OC + BLOCK_SIZE_OC - 1) // BLOCK_SIZE_OC,  # output channel blocks
        H_out,  # output height
        W_out,  # output width
    )
    
    # Launch kernel
    conv_transpose2d_kernel_optimized[grid](
        x, weight, bias, out,
        B, IC, OC, groups,
        H_in, W_in, H_out, W_out,
        kH, kW,
        stride[0], stride[1],
        padding[0], padding[1],
        output_padding[0], output_padding[1],
        stride_x[0], stride_x[1], stride_x[2], stride_x[3],
        stride_w[0], stride_w[1], stride_w[2], stride_w[3],
        stride_out[0], stride_out[1], stride_out[2], stride_out[3],
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_OC=BLOCK_SIZE_OC,
        BLOCK_SIZE_IC=BLOCK_SIZE_IC,
        HAS_BIAS=bias is not None,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized Model using Triton kernel for transposed 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same parameters as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.has_bias = bias
        
        # Create weight and bias parameters (same as nn.ConvTranspose2d)
        # For conv_transpose2d, weight shape is [in_channels, out_channels//groups, kH, kW]
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        # Same initialization as nn.ConvTranspose2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using the Triton kernel.
        """
        # Convert stride, padding, output_padding to tuple if int
        stride = (self.stride, self.stride)
        padding = (self.padding, self.padding)
        output_padding = (self.output_padding, self.output_padding)
        
        return triton_conv_transpose2d(
            x, self.weight, self.bias, 
            stride, padding, output_padding, self.groups
        )


# Import math for initialization
import math