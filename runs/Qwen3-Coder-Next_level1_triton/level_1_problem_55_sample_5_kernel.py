import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor (N, C, H, W)
    w_ptr,  # Weight tensor (C_out, C_in, K_h, K_w)
    b_ptr,  # Bias tensor (C_out,)
    out_ptr,  # Output tensor (N, C_out, H_out, W_out)
    N, C, H, W,  # Input dimensions
    C_out, C_in, K_h, K_w,  # Weight dimensions
    stride_h, stride_w,  # Stride
    pad_h, pad_w,  # Padding
    dil_h, dil_w,  # Dilation
    H_out, W_out,  # Output dimensions
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_KH: tl.constexpr,  # Block size for kernel height
    BLOCK_SIZE_KW: tl.constexpr,  # Block size for kernel width
    BLOCK_SIZE_C: tl.constexpr,  # Block size for input channels
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_channel = tl.program_id(1)
    
    # Skip if out of bounds
    if pid_batch >= N or pid_out_channel >= C_out:
        return
    
    # Compute output spatial position
    # We'll compute one output element per program for simplicity in this first version
    # Later optimizations could use larger blocks
    
    # Calculate the start position for this program
    out_h_start = tl.program_id(2) * BLOCK_SIZE_M
    out_w_start = tl.program_id(3) * BLOCK_SIZE_N
    
    # Compute the spatial indices
    out_h = out_h_start + tl.arange(0, BLOCK_SIZE_M)[:, None]
    out_w = out_w_start + tl.arange(0, BLOCK_SIZE_N)[None, :]
    
    # Check bounds for output spatial positions
    mask_hw = (out_h < H_out) & (out_w < W_out)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Compute convolution: for each kernel position
    for kh in range(K_h):
        for kw in range(K_w):
            # Compute input position
            in_h = out_h * stride_h - pad_h + kh * dil_h
            in_w = out_w * stride_w - pad_w + kw * dil_w
            
            # Mask for valid input positions
            mask_valid = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W)
            mask_combined = mask_hw & mask_valid
            
            # Load input values (N, C, H, W)
            # For batch dimension
            x_offsets_batch = pid_batch * (C * H * W)
            x_offsets_h = in_h * (C * W)
            x_offsets_w = in_w
            
            # Load input for all channels
            for c_start in range(0, C, BLOCK_SIZE_C):
                c_offsets = c_start + tl.arange(0, BLOCK_SIZE_C)
                mask_c = c_offsets < C
                
                # Compute full input offset
                x_offsets = x_offsets_batch + x_offsets_h + x_offsets_w + c_offsets * (H * W)
                
                # Load input
                x_val = tl.load(x_ptr + x_offsets, mask=mask_c[None, :] & mask_combined, other=0.0)
                
                # Load weight for this kernel position and output channel
                w_offsets = (pid_out_channel * (C_in * K_h * K_w) + 
                            c_offsets[:, None] * (K_h * K_w) + 
                            kh * K_w + kw)
                w_val = tl.load(w_ptr + w_offsets, mask=mask_c[:, None], other=0.0)
                
                # Accumulate
                acc += tl.dot(w_val, x_val, out_dtype=tl.float32)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_out_channel)
        acc += bias
    
    # Store output
    out_batch = pid_batch * (C_out * H_out * W_out)
    out_h_offset = out_h * (C_out * W_out)
    out_w_offset = out_w
    out_offsets = out_batch + out_h_offset + out_w_offset + pid_out_channel * (H_out * W_out)
    
    tl.store(out_ptr + out_offsets, acc.to(x_ptr.dtype.element_ty), mask=mask_hw)


# A more practical implementation using im2col + matmul approach for better performance
@triton.jit
def conv2d_im2col_kernel(
    x_ptr,  # Input tensor (N, C, H, W)
    w_ptr,  # Weight tensor (C_out, C_in, K_h, K_w)
    b_ptr,  # Bias tensor (C_out,)
    col_ptr,  # Temporary buffer for im2col (N * H_out * W_out, C_in * K_h * K_w)
    out_ptr,  # Output tensor (N, C_out, H_out, W_out)
    N, C, H, W,  # Input dimensions
    C_out, C_in, K_h, K_w,  # Weight dimensions
    stride_h, stride_w,  # Stride
    pad_h, pad_w,  # Padding
    dil_h, dil_w,  # Dilation
    H_out, W_out,  # Output dimensions
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output rows
    BLOCK_SIZE_N: tl.constexpr,  # Block size for output cols
    BLOCK_SIZE_K: tl.constexpr,  # Block size for kernel elements
):
    # im2col: convert patches to columns
    # For each (n, h_out, w_out), extract the kernel-sized patch
    
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    if pid_n >= N or pid_h >= H_out or pid_w >= W_out:
        return
    
    # Calculate input position for top-left corner of kernel
    in_h = pid_h * stride_h - pad_h
    in_w = pid_w * stride_w - pad_w
    
    # Compute column offset for this output position
    col_offset = (pid_n * H_out * W_out + pid_h * W_out + pid_w) * (C_in * K_h * K_w)
    
    # Fill the column with kernel values
    for kh in range(K_h):
        for kw in range(K_w):
            for c in range(C_in):
                # Calculate input position
                h_idx = in_h + kh * dil_h
                w_idx = in_w + kw * dil_w
                
                # Check bounds
                valid = (h_idx >= 0) and (h_idx < H) and (w_idx >= 0) and (w_idx < W)
                
                if valid:
                    # Input index
                    x_idx = pid_n * (C * H * W) + c * (H * W) + h_idx * W + w_idx
                    val = tl.load(x_ptr + x_idx)
                else:
                    val = 0.0
                
                # Column index
                col_idx = col_offset + (kh * K_w + kw) * C_in + c
                tl.store(col_ptr + col_idx, val)


@triton.jit
def matmul_kernel(
    a_ptr,  # (M, K)
    b_ptr,  # (K, N)
    c_ptr,  # (M, N)
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Standard matrix multiplication kernel
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    if pid_m >= M or pid_n >= N:
        return
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offsets < K
        
        # Load A block: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask_m = m_offsets < M
        a_offsets = m_offsets[:, None] * K + k_offsets[None, :]
        a = tl.load(a_ptr + a_offsets, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Load B block: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_n = n_offsets < N
        b_offsets = k_offsets[:, None] * N + n_offsets[None, :]
        b = tl.load(b_ptr + b_offsets, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        # Accumulate
        acc += tl.dot(a, b, out_dtype=tl.float32)
    
    # Store result
    m_offsets_final = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets_final = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_m_final = m_offsets_final < M
    mask_n_final = n_offsets_final < N
    c_offsets = m_offsets_final[:, None] * N + n_offsets_final[None, :]
    tl.store(c_ptr + c_offsets, acc.to(a_ptr.dtype.element_ty), mask=mask_m_final[:, None] & mask_n_final[None, :])


def conv2d_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> torch.Tensor:
    """
    Performs 2D convolution using Triton kernels.
    Uses im2col + matmul approach for efficiency.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Extract dimensions
    N, C, H, W = x.shape
    C_out, C_in, K_h, K_w = weight.shape
    
    # Calculate output dimensions
    H_out = (H + 2 * padding - dilation * (K_h - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (K_w - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(N, C_out, H_out, W_out, device=x.device, dtype=x.dtype)
    
    # im2col buffer
    col_buffer = torch.empty(N * H_out * W_out, C_in * K_h * K_w, device=x.device, dtype=x.dtype)
    
    # Configure kernel launch parameters
    # im2col kernel: one program per (batch, out_h, out_w)
    im2col_grid = (N, H_out, W_out)
    
    # matmul kernel: M = N * H_out * W_out, N = C_out, K = C_in * K_h * K_w
    M = N * H_out * W_out
    K = C_in * K_h * K_w
    N_matmul = C_out
    
    BLOCK_SIZE_M = 8
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 64
    
    matmul_grid = (
        (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (N_matmul + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    )
    
    # Launch im2col kernel
    conv2d_im2col_kernel[im2col_grid](
        x, weight, bias, col_buffer, out,
        N, C, H, W, C_out, C_in, K_h, K_w,
        stride, stride, padding, padding, dilation, dilation,
        H_out, W_out,
        BLOCK_SIZE_M=1, BLOCK_SIZE_N=1, BLOCK_SIZE_K=1
    )
    
    # Transpose weight for matmul: (C_out, K) -> (K, C_out)
    weight_t = weight.view(C_out, -1).t().contiguous()  # (K, C_out)
    
    # Launch matmul kernel
    matmul_kernel[matmul_grid](
        col_buffer, weight_t, out.view(N * C_out, H_out * W_out),
        M, N_matmul, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    # Add bias if present
    if bias is not None:
        bias_view = bias.view(1, C_out, 1, 1)
        out = out + bias_view
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model with Triton-based convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the weight and bias parameters directly
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create weight and bias tensors
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 2D convolution using Triton kernels.
        """
        return conv2d_triton(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )


# Import math for initialization
import math