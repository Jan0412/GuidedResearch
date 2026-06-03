import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

# Triton kernel for im2col operation (extracting sliding blocks from input)
@triton.jit
def im2col_kernel(
    x_ptr,  # Input tensor pointer (N, C, H, W)
    col_ptr,  # Output column tensor pointer (N, C*KH*KW, OH*OW)
    n, c, h, w, kh, kw, oh, ow, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one element in the output column matrix
    total_elements = n * c * kh * kw * oh * ow
    pid = tl.program_id(0)
    
    # Calculate indices
    # We'll process in blocks for efficiency
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Decode the linear index into (n, c, kh, kw, oh, ow)
    temp = offsets
    ow_idx = temp % ow
    temp = temp // ow
    oh_idx = temp % oh
    temp = temp // oh
    kw_idx = temp % kw
    temp = temp // kw
    kh_idx = temp % kh
    temp = temp // kh
    c_idx = temp % c
    temp = temp // c
    n_idx = temp
    
    # Calculate input coordinates
    in_h = oh_idx * stride_h + kh_idx * dilation_h - pad_h
    in_w = ow_idx * stride_w + kw_idx * dilation_w - pad_w
    
    # Check bounds
    in_bounds = (in_h >= 0) & (in_h < h) & (in_w >= 0) & (in_w < w)
    
    # Calculate input pointer offset
    input_offset = n_idx * (c * h * w) + c_idx * (h * w) + in_h * w + in_w
    input_ptr = x_ptr + input_offset
    
    # Load value or zero if out of bounds
    val = tl.where(in_bounds, tl.load(input_ptr), 0.0)
    
    # Calculate output pointer offset
    col_offset = offsets
    tl.store(col_ptr + col_offset, val, mask=mask)

# Triton kernel for convolution using im2col + matrix multiplication
@triton.jit
def conv2d_kernel(
    col_ptr,  # Im2col tensor (N, C*KH*KW, OH*OW)
    weight_ptr,  # Weight tensor (OC, IC*KH*KW)
    bias_ptr,  # Bias tensor (OC,)
    out_ptr,  # Output tensor (N, OC, OH, OW)
    n, oc, ic_kh_kw, oh_ow,
    has_bias: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Matrix multiplication: col @ weight.T
    # col: (N * OH * OW, C*KH*KW)
    # weight: (OC, C*KH*KW)
    # out: (N * OH * OW, OC)
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create offsets for M dimension (N * OH * OW)
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # Create offsets for N dimension (OC)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # Create offsets for K dimension (C*KH*KW)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Masks
    m_mask = offsets_m < (n * oh_ow)
    n_mask = offsets_n < oc
    k_mask = offsets_k < ic_kh_kw
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, ic_kh_kw, BLOCK_SIZE_K):
        # Load col block: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        col_offset = (offsets_m[:, None] * ic_kh_kw + k + offsets_k[None, :])
        col_mask = (m_mask[:, None] & k_mask[None, :])
        col_val = tl.load(col_ptr + col_offset, mask=col_mask, other=0.0)
        
        # Load weight block: (BLOCK_SIZE_N, BLOCK_SIZE_K)
        weight_offset = (offsets_n[None, :] * ic_kh_kw + k + offsets_k[None, :])
        weight_mask = (n_mask[None, :] & k_mask[None, :])
        weight_val = tl.load(weight_ptr + weight_offset, mask=weight_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(col_val, weight_val.T)
    
    # Convert to output dtype
    acc = acc.to(tl.float32)
    
    # Add bias if available
    if has_bias:
        bias_val = tl.load(bias_ptr + offsets_n, mask=n_mask)
        acc += bias_val[None, :]
    
    # Store result
    out_offset = offsets_m[:, None] * oc + offsets_n[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_offset, acc, mask=out_mask)

# Triton kernel to reshape output from (N*OH*OW, OC) to (N, OC, OH, OW)
@triton.jit
def reshape_output_kernel(
    flat_out_ptr,  # Flattened output (N*OH*OW, OC)
    out_ptr,  # Reshaped output (N, OC, OH, OW)
    n, oc, oh, ow,
    BLOCK_SIZE: tl.constexpr,
):
    total_elements = n * oc * oh * ow
    pid = tl.program_id(0)
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Decode the linear index into (n, oc, oh, ow)
    temp = offsets
    ow_idx = temp % ow
    temp = temp // ow
    oh_idx = temp % oh
    temp = temp // oh
    oc_idx = temp % oc
    temp = temp // oc
    n_idx = temp
    
    # Calculate input offset (flattened)
    flat_offset = n_idx * (oh * ow) + oh_idx * ow + ow_idx
    
    # Calculate output offset (reshaped)
    out_offset = offsets
    
    # Load and store
    val = tl.load(flat_out_ptr + flat_offset)
    tl.store(out_ptr + out_offset, val, mask=mask)


def triton_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Performs 2D convolution using Triton kernels with im2col approach.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Get dimensions
    n, c, h, w = x.shape
    oc, ic, kh, kw = weight.shape
    
    # Calculate output dimensions
    out_h = (h + 2 * padding - dilation * (kh - 1) - 1) // stride + 1
    out_w = (w + 2 * padding - dilation * (kw - 1) - 1) // stride + 1
    
    # Calculate im2col dimensions
    col_size = c * kh * kw
    col_elements = n * out_h * out_w * col_size
    
    # Create im2col buffer
    col = torch.empty((col_elements,), dtype=x.dtype, device=x.device)
    
    # Configure im2col kernel
    BLOCK_SIZE_IM2COL = 256
    grid_im2col = (math.ceil(col_elements / BLOCK_SIZE_IM2COL),)
    
    im2col_kernel[grid_im2col](
        x, col,
        n, c, h, w, kh, kw, out_h, out_w,
        stride, stride, padding, padding, dilation, dilation,
        BLOCK_SIZE=BLOCK_SIZE_IM2COL
    )
    
    # Reshape col to (n * out_h * out_w, col_size)
    col_reshaped = col.view(n * out_h * out_w, col_size)
    
    # Prepare output tensor
    flat_out = torch.empty((n * out_h * out_w, oc), dtype=x.dtype, device=x.device)
    
    # Configure matrix multiplication kernel
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    
    grid_m = (n * out_h * out_w + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (oc + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_mm = (grid_m, grid_n)
    
    conv2d_kernel[grid_mm](
        col_reshaped, weight, bias, flat_out,
        n, oc, col_size, out_h * out_w,
        has_bias=bias is not None,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    # Reshape output to (n, oc, out_h, out_w)
    out = flat_out.view(n, out_h, out_w, oc)
    out = out.permute(0, 3, 1, 2).contiguous()
    
    return out


class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with asymmetric input and square kernel.
    Uses optimized Triton kernels for the convolution operation.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using optimized Triton kernels.
        """
        # Extract weights and bias from the original conv2d layer
        weight = self.conv2d.weight
        bias = self.conv2d.bias if self.conv2d.bias is not None else None
        
        # Get convolution parameters
        stride = self.conv2d.stride[0]
        padding = self.conv2d.padding[0]
        dilation = self.conv2d.dilation[0]
        groups = self.conv2d.groups
        
        # Perform convolution using Triton kernels
        return triton_conv2d(x, weight, bias, stride, padding, dilation, groups)