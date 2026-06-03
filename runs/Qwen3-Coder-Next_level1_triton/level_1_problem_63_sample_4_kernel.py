import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def im2col_kernel(
    x_ptr,  # Input tensor pointer (B, C, H, W)
    col_ptr,  # Output column matrix pointer
    batch_size, in_channels, height, width,
    kernel_size, stride, padding, dilation,
    out_h, out_w,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Compute output position
    batch_idx = tl.program_id(0)
    out_h_idx = tl.program_id(1)
    out_w_idx = tl.program_id(2)
    
    # Calculate base position in input
    h_offset = out_h_idx * stride - padding
    w_offset = out_w_idx * stride - padding
    
    # Calculate total elements in output column
    num_elements = in_channels * kernel_size * kernel_size
    
    # Process in blocks
    for block_start in range(0, num_elements, BLOCK_SIZE_M):
        offsets_m = block_start + tl.arange(0, BLOCK_SIZE_M)
        mask_m = offsets_m < num_elements
        
        # Convert linear index to (c, kh, kw)
        c = offsets_m // (kernel_size * kernel_size)
        kh_w = offsets_m % (kernel_size * kernel_size)
        kh = kh_w // kernel_size
        kw = kh_w % kernel_size
        
        # Compute input position
        h_pos = h_offset + kh * dilation
        w_pos = w_offset + kw * dilation
        
        # Check bounds
        valid = (h_pos >= 0) & (h_pos < height) & (w_pos >= 0) & (w_pos < width) & mask_m
        
        # Calculate input pointer offset
        input_offset = batch_idx * in_channels * height * width + c * height * width + h_pos * width + w_pos
        
        # Load value (0 if out of bounds)
        val = tl.load(x_ptr + input_offset, mask=valid, other=0.0)
        
        # Store in column matrix
        col_offset = (batch_idx * out_h * out_w + out_h_idx * out_w + out_w_idx) * num_elements + offsets_m
        tl.store(col_ptr + col_offset, val, mask=mask_m)


@triton.jit
def conv2d_gemm_kernel(
    a_ptr,  # Column matrix (B * out_h * out_w, in_c * k_h * k_w)
    b_ptr,  # Weight matrix (out_c, in_c * k_h * k_w)
    bias_ptr,  # Bias vector (out_c,)
    out_ptr,  # Output tensor (B, out_c, out_h, out_w)
    batch_size, out_h, out_w, out_channels, kernel_elements,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Compute output position
    batch_idx = tl.program_id(0)
    out_h_idx = tl.program_id(1)
    out_w_idx = tl.program_id(2)
    out_c_idx = tl.program_id(3)
    
    # Calculate row in matrix A
    row = batch_idx * out_h * out_w + out_h_idx * out_w + out_w_idx
    
    # Compute offsets for matrix multiplication C[row, out_c_idx] = A[row, :] @ B[out_c_idx, :].T
    # Process in blocks for efficiency
    sum_val = 0.0
    
    for k in range(0, kernel_elements, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offsets < kernel_elements
        
        # Load A[row, k_offsets]
        a_offset = row * kernel_elements + k_offsets
        a_val = tl.load(a_ptr + a_offset, mask=mask_k, other=0.0)
        
        # Load B[out_c_idx, k_offsets]
        b_offset = out_c_idx * kernel_elements + k_offsets
        b_val = tl.load(b_ptr + b_offset, mask=mask_k, other=0.0)
        
        # Accumulate product
        sum_val += tl.sum(a_val * b_val)
    
    # Add bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + out_c_idx)
        sum_val += bias
    
    # Store result
    out_offset = (batch_idx * out_channels * out_h * out_w + 
                  out_c_idx * out_h * out_w + out_h_idx * out_w + out_w_idx)
    tl.store(out_ptr + out_offset, sum_val)


def triton_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Optimized convolution using Triton kernels with im2col + GEMM approach.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Extract dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    out_h = (height + 2 * padding - dilation * (kernel_h - 1) - 1) // stride + 1
    out_w = (width + 2 * padding - dilation * (kernel_w - 1) - 1) // stride + 1
    
    # Create output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # im2col parameters
    kernel_elements = in_channels * kernel_h * kernel_w
    
    # Launch im2col kernel
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    grid_im2col = (batch_size, out_h, out_w)
    col = torch.empty(batch_size * out_h * out_w, kernel_elements, dtype=x.dtype, device=x.device)
    
    im2col_kernel[grid_im2col](
        x, col,
        batch_size, in_channels, height, width,
        kernel_h, stride, padding, dilation,
        out_h, out_w,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    # GEMM parameters
    BLOCK_SIZE_M = 8
    BLOCK_SIZE_N = 8
    BLOCK_SIZE_K = 32
    
    # For simplicity, process one output channel at a time
    grid_gemm = (batch_size, out_h, out_w, out_channels)
    
    conv2d_gemm_kernel[grid_gemm](
        col, weight.view(out_channels, -1),
        bias, out,
        batch_size, out_h, out_w, out_channels, kernel_elements,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized 2D convolution using Triton kernels with im2col + GEMM approach.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 2D convolution using Triton kernels.
        """
        return triton_conv2d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups
        )