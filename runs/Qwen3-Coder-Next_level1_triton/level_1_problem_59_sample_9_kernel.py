import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr,  # Input tensor pointer (N, C_in, H, W, D)
    w_ptr,  # Weight tensor pointer (C_out, C_in, K_h, K_w, K_d)
    b_ptr,  # Bias tensor pointer (C_out,)
    out_ptr,  # Output tensor pointer (N, C_out, H_out, W_out, D_out)
    N, C_in, H, W, D,  # Input dimensions
    C_out, K_h, K_w, K_d,  # Weight dimensions
    stride_h, stride_w, stride_d,  # Strides
    pad_h, pad_w, pad_d,  # Padding
    dil_h, dil_w, dil_d,  # Dilation
    # Output dimensions
    H_out, W_out, D_out,
    # Block sizes
    BLOCK_C_out: tl.constexpr,
    BLOCK_C_in: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,  # Block size for H_out
    BLOCK_N: tl.constexpr,  # Block size for W_out
    BLOCK_P: tl.constexpr,  # Block size for D_out
):
    # Program IDs
    pid_z = tl.program_id(0)  # Batch index
    pid_c = tl.program_id(1)  # Output channel block
    pid_h = tl.program_id(2)  # Output height block
    pid_w = tl.program_id(3)  # Output width block
    pid_d = tl.program_id(4)  # Output depth block
    
    # Calculate output positions
    out_h = pid_h * BLOCK_M
    out_w = pid_w * BLOCK_N
    out_d = pid_d * BLOCK_P
    
    # Initialize accumulator for output
    acc = tl.zeros((BLOCK_M, BLOCK_N, BLOCK_P), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for c_in_start in range(0, C_in, BLOCK_C_in):
        c_in_end = tl.minimum(c_in_start + BLOCK_C_in, C_in)
        
        # Loop over output channels in this block
        for c_out_offset in range(tl.minimum(BLOCK_C_out, C_out - pid_c * BLOCK_C_out)):
            c_out = pid_c * BLOCK_C_out + c_out_offset
            
            # Load weight block: (K_h, K_w, K_d, C_in_block)
            # We'll compute weights on the fly for efficiency
            for kh in range(K_h):
                for kw in range(K_w):
                    for kd in range(K_d):
                        # Calculate input position
                        in_h = out_h * stride_h + kh * dil_h - pad_h
                        in_w = out_w * stride_w + kw * dil_w - pad_w
                        in_d = out_d * stride_d + kd * dil_d - pad_d
                        
                        # Load input values for this kernel position
                        x_vals = tl.zeros((BLOCK_M, BLOCK_N, BLOCK_P), dtype=tl.float32)
                        
                        # Check bounds and load input
                        for m in range(BLOCK_M):
                            h_idx = in_h + m * stride_h
                            h_mask = (h_idx >= 0) & (h_idx < H)
                            for n in range(BLOCK_N):
                                w_idx = in_w + n * stride_w
                                w_mask = (w_idx >= 0) & (w_idx < W)
                                for p in range(BLOCK_P):
                                    d_idx = in_d + p * stride_d
                                    d_mask = (d_idx >= 0) & (d_idx < D)
                                    
                                    # Combined mask
                                    mask = h_mask & w_mask & d_mask
                                    
                                    if c_in_start == 0:
                                        # Load from input tensor
                                        input_ptr = x_ptr + pid_z * (C_in * H * W * D) + \
                                                   c_in_start * (H * W * D) + \
                                                   h_idx * (W * D) + \
                                                   w_idx * D + d_idx
                                        x_vals = tl.load(input_ptr, mask=mask, other=0.0)
                        
                        # Load weight value for this position
                        weight_ptr = w_ptr + c_out * (C_in * K_h * K_w * K_d) + \
                                    c_in_start * (K_h * K_w * K_d) + \
                                    kh * (K_w * K_d) + \
                                    kw * K_d + kd
                        weight_val = tl.load(weight_ptr)
                        
                        # Accumulate: acc += x * weight
                        acc += x_vals * weight_val
    
    # Add bias if provided
    if b_ptr is not None:
        bias_ptr = b_ptr + pid_c * BLOCK_C_out + tl.arange(0, BLOCK_C_out)
        bias = tl.load(bias_ptr, mask=tl.arange(0, BLOCK_C_out) < C_out, other=0.0)
        acc += bias[:, None, None]  # Broadcast bias to (BLOCK_M, BLOCK_N, BLOCK_P)
    
    # Store output
    out_ptr_offset = pid_z * (C_out * H_out * W_out * D_out) + \
                    pid_c * BLOCK_C_out * (H_out * W_out * D_out) + \
                    out_h * (W_out * D_out) + \
                    out_w * D_out + out_d
    
    # Store with proper masking
    mask_h = tl.arange(0, BLOCK_M) < H_out - out_h
    mask_w = tl.arange(0, BLOCK_N) < W_out - out_w
    mask_d = tl.arange(0, BLOCK_P) < D_out - out_d
    
    for m in range(BLOCK_M):
        for n in range(BLOCK_N):
            for p in range(BLOCK_P):
                if mask_h[m] and mask_w[n] and mask_d[p]:
                    output_ptr = out_ptr + out_ptr_offset + \
                                m * (W_out * D_out) + \
                                n * D_out + p
                    tl.store(output_ptr, acc[m, n, p])


def triton_conv3d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Triton implementation of 3D convolution.
    
    Args:
        x: Input tensor (N, C_in, H, W, D)
        weight: Weight tensor (C_out, C_in, K_h, K_w, K_d)
        bias: Optional bias tensor (C_out,)
        stride: Stride
        padding: Padding
        dilation: Dilation
        groups: Groups (currently only groups=1 supported)
    """
    if groups != 1:
        raise NotImplementedError("Groups != 1 not supported in Triton kernel")
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    N, C_in, H, W, D = x.shape
    C_out, _, K_h, K_w, K_d = weight.shape
    
    # Calculate output dimensions
    H_out = (H + 2 * padding - dilation * (K_h - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (K_w - 1) - 1) // stride + 1
    D_out = (D + 2 * padding - dilation * (K_d - 1) - 1) // stride + 1
    
    # Allocate output tensor
    out = torch.empty(N, C_out, H_out, W_out, D_out, dtype=x.dtype, device=x.device)
    
    # Kernel configuration
    BLOCK_C_out = 8  # Tune for your GPU
    BLOCK_C_in = 8   # Tune for your GPU
    BLOCK_M = 4      # Block size for H_out
    BLOCK_N = 4      # Block size for W_out
    BLOCK_P = 2      # Block size for D_out
    
    # Calculate grid dimensions
    grid = lambda meta: (
        N,  # Batch size
        (C_out + meta["BLOCK_C_out"] - 1) // meta["BLOCK_C_out"],  # Output channel blocks
        (H_out + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],          # Height blocks
        (W_out + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],          # Width blocks
        (D_out + meta["BLOCK_P"] - 1) // meta["BLOCK_P"],          # Depth blocks
    )
    
    # Launch kernel
    conv3d_kernel[grid](
        x, weight, bias, out,
        N, C_in, H, W, D,
        C_out, K_h, K_w, K_d,
        stride, stride, stride,
        padding, padding, padding,
        dilation, dilation, dilation,
        H_out, W_out, D_out,
        BLOCK_C_out=BLOCK_C_out,
        BLOCK_C_in=BLOCK_C_in,
        BLOCK_K=1,  # Not used in this implementation
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_P=BLOCK_P,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernels for 3D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same weights as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create the same Conv3d layer to get initialized weights
        self.conv3d = nn.Conv3d(in_channels, out_channels, 
                                (kernel_size, kernel_size, 1), 
                                stride=stride, padding=padding, 
                                dilation=dilation, groups=groups, 
                                bias=bias)
        
        # Store weight and bias references for Triton kernel
        self.weight = self.conv3d.weight
        self.bias = self.conv3d.bias if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D convolution using Triton kernel.
        """
        return triton_conv3d(x, self.weight, self.bias, 
                            stride=self.stride, 
                            padding=self.padding, 
                            dilation=self.dilation, 
                            groups=self.groups)