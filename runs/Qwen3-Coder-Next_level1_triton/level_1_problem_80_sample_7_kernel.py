import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def im2col_kernel(
    x_ptr,  # Input tensor pointer [N, C, H, W]
    col_ptr,  # Output column tensor pointer [N, C, kH, kW, OH, OW]
    N, C, H, W,  # Input dimensions
    kH, kW,  # Kernel height and width
    OH, OW,  # Output height and width
    stride_h, stride_w,  # Strides
    pad_h, pad_w,  # Padding
    dil_h, dil_w,  # Dilation
    BLOCK_SIZE: tl.constexpr,
):
    # Each block processes a portion of the output tensor
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_col = tl.program_id(2)
    
    # Compute output position from pid_col
    # pid_col encodes OH, OW, kH, kW indices
    # We'll use a flattened approach with BLOCK_SIZE elements per block
    
    # Calculate offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < (kH * kW * OH * OW)
    
    # For simplicity, we'll process one batch and channel per block
    # and iterate through spatial positions
    
    # Get base pointers for this batch and channel
    base_x = x_ptr + pid_n * (C * H * W) + pid_c * (H * W)
    
    # Compute output spatial position and kernel position from pid_col
    # pid_col ranges from 0 to (OH * OW * kH * kW - 1)
    spatial_idx = pid_col
    out_h = spatial_idx // (OW * kH * kW)
    rem = spatial_idx % (OW * kH * kW)
    out_w = rem // (kH * kW)
    kh = (rem % (kH * kW)) // kW
    kw = (rem % (kH * kW)) % kW
    
    # Calculate input position
    in_h = out_h * stride_h - pad_h + kh * dil_h
    in_w = out_w * stride_w - pad_w + kw * dil_w
    
    # Load input value if in bounds
    val = 0.0
    if 0 <= in_h < H and 0 <= in_w < W:
        offset = in_h * W + in_w
        val = tl.load(base_x + offset)
    
    # Store to column tensor
    col_offset = pid_n * (C * kH * kW * OH * OW) + \
                 pid_c * (kH * kW * OH * OW) + \
                 kh * (kW * OH * OW) + \
                 kw * (OH * OW) + \
                 out_h * OW + out_w
    tl.store(col_ptr + col_offset, val)


@triton.jit
def conv2d_fused_kernel(
    x_ptr,  # Input tensor [N, C, H, W]
    w_ptr,  # Weight tensor [OC, IC, kH, kW]
    b_ptr,  # Bias tensor [OC] or None
    out_ptr,  # Output tensor [N, OC, OH, OW]
    N, C, H, W,  # Input dimensions
    OC, IC, kH, kW,  # Output channels, input channels, kernel dimensions
    OH, OW,  # Output dimensions
    stride_h, stride_w,  # Strides
    pad_h, pad_w,  # Padding
    dil_h, dil_w,  # Dilation
    BLOCK_SIZE_M: tl.constexpr,  # Block size for N * OH * OW
    BLOCK_SIZE_N: tl.constexpr,  # Block size for OC
    BLOCK_SIZE_K: tl.constexpr,  # Block size for IC * kH * kW
):
    # Matmul-based convolution: reshape convolution to GEMM
    # X: [N*OH*OW, IC*kH*kW] (im2col)
    # W: [OC, IC*kH*kW] (reshaped)
    # Out: [N*OH*OW, OC]
    
    # Get program IDs
    pid_m = tl.program_id(0)  # For N*OH*OW
    pid_n = tl.program_id(1)  # For OC
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    # Compute offsets for input and weight
    # We'll process BLOCK_SIZE_M * BLOCK_SIZE_N elements
    
    # For simplicity, implement a straightforward im2col + matmul
    # This kernel processes one output element per program
    
    # Actually, let's do a more efficient fused approach
    # Each program computes one output element: out[n, oc, oh, ow]
    
    # Recalculate program IDs for element-wise computation
    total_elements = N * OC * OH * OW
    total_programs = tl.num_programs(0)
    
    # Compute which output element this program computes
    pid = tl.program_id(0)
    if pid >= total_elements:
        return
    
    # Decode pid into n, oc, oh, ow
    tmp = pid
    ow = tmp % OW
    tmp //= OW
    oh = tmp % OH
    tmp //= OH
    oc = tmp % OC
    tmp //= OC
    n = tmp
    
    # Compute bias
    if b_ptr is not None:
        acc = tl.load(b_ptr + oc)
    else:
        acc = 0.0
    
    # Compute convolution sum: sum over ic, kh, kw of x[n, ic, oh*sh + kh*dh - pad_h, ow*sw + kw*dw - pad_w] * w[oc, ic, kh, kw]
    for ic in range(IC):
        for kh in range(kH):
            for kw in range(kW):
                # Compute input position
                in_h = oh * stride_h + kh * dil_h - pad_h
                in_w = ow * stride_w + kw * dil_w - pad_w
                
                # Load input value if in bounds
                x_val = 0.0
                if 0 <= in_h < H and 0 <= in_w < W:
                    x_offset = n * (C * H * W) + ic * (H * W) + in_h * W + in_w
                    x_val = tl.load(x_ptr + x_offset)
                
                # Load weight value
                w_offset = oc * (IC * kH * kW) + ic * (kH * kW) + kh * kW + kw
                w_val = tl.load(w_ptr + w_offset)
                
                acc += x_val * w_val
    
    # Store result
    out_offset = n * (OC * OH * OW) + oc * (OH * OW) + oh * OW + ow
    tl.store(out_ptr + out_offset, acc)


def triton_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                 stride=1, padding=(0, 0), dilation=(1, 1)):
    """
    Triton-based 2D convolution implementation.
    
    Args:
        x: Input tensor of shape [N, C, H, W]
        weight: Weight tensor of shape [OC, IC, kH, kW]
        bias: Optional bias tensor of shape [OC]
        stride: Stride for convolution
        padding: Padding as (pad_h, pad_w)
        dilation: Dilation as (dil_h, dil_w)
    
    Returns:
        Output tensor of shape [N, OC, OH, OW]
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    N, C, H, W = x.shape
    OC, IC, kH, kW = weight.shape
    
    # Calculate output dimensions
    stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride)
    pad_h, pad_w = padding if isinstance(padding, tuple) else (padding, padding)
    dil_h, dil_w = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    
    OH = (H + 2 * pad_h - dil_h * (kH - 1) - 1) // stride_h + 1
    OW = (W + 2 * pad_w - dil_w * (kW - 1) - 1) // stride_w + 1
    
    # Prepare output tensor
    out = torch.empty(N, OC, OH, OW, dtype=x.dtype, device=x.device)
    
    # Total number of output elements
    total_elements = N * OC * OH * OW
    
    # Launch kernel with one program per output element
    # Use a reasonable block size for parallelism
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(total_elements, meta["BLOCK_SIZE"]),)
    
    # Launch the fused convolution kernel
    conv2d_fused_kernel[grid](x, weight, bias, out,
                              N, C, H, W,
                              OC, IC, kH, kW,
                              OH, OW,
                              stride_h, stride_w,
                              pad_h, pad_w,
                              dil_h, dil_w,
                              BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the Model with Triton-based 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), 
                 bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 2D convolution using Triton kernel.
        """
        return triton_conv2d(x, self.weight, self.bias, 
                            stride=self.stride, 
                            padding=self.padding, 
                            dilation=self.dilation)


import math