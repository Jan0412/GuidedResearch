import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Input tensor pointer (B, C_in, H_in, W_in)
    w_ptr,  # Weight tensor pointer (C_in, C_out, K_h, K_w)
    b_ptr,  # Bias tensor pointer (C_out,)
    out_ptr,  # Output tensor pointer (B, C_out, H_out, W_out)
    B, C_in, C_out, 
    H_in, W_in, 
    K_h, K_w,
    stride_h, stride_w,
    pad_h, pad_w,
    out_pad_h, out_pad_w,
    H_out, W_out,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr,  # Block size for input channels
    BLOCK_SIZE_H: tl.constexpr,  # Block size for output height
    BLOCK_SIZE_W: tl.constexpr,  # Block size for output width
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_h_out = tl.program_id(2)
    pid_w_out = tl.program_id(3)
    
    # Compute output position
    batch_idx = pid_batch
    out_c_idx = pid_c_out * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    out_h_idx = pid_h_out * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w_idx = pid_w_out * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create meshgrid for output positions
    out_h, out_w = tl.meshgrid(out_h_idx, out_w_idx)
    out_h = out_h.T
    out_w = out_w.T
    
    # Compute corresponding input positions
    in_h = (out_h - pad_h) // stride_h
    in_w = (out_w - pad_w) // stride_w
    
    # Check if input position is valid
    valid_mask = (in_h >= 0) & (in_h < H_in) & (in_w >= 0) & (in_w < W_in)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over input channels
    for c_in_idx in range(0, C_in, BLOCK_SIZE_K):
        c_in_block = c_in_idx + tl.arange(0, BLOCK_SIZE_K)
        
        # Load input data: shape [B, C_in, H_in, W_in]
        x_offset = (
            batch_idx * (C_in * H_in * W_in) +
            c_in_block[:, None, None] * (H_in * W_in) +
            in_h[None, :, :] * W_in +
            in_w[None, :, :]
        )
        x_mask = valid_mask[None, :, :] & (c_in_block[:, None, None] < C_in)
        x_val = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)
        
        # Load weights: shape [C_in, C_out, K_h, K_w]
        # For transposed conv, the kernel indices are computed as:
        # out_c = in_c * stride + kernel_offset - pad
        # kernel_h = out_h - in_h * stride + pad
        # kernel_w = out_w - in_w * stride + pad
        
        # Compute kernel positions
        kernel_h = out_h - in_h * stride_h + pad_h
        kernel_w = out_w - in_w * stride_w + pad_w
        
        # Check if kernel position is valid
        kernel_h_mask = (kernel_h >= 0) & (kernel_h < K_h)
        kernel_w_mask = (kernel_w >= 0) & (kernel_w < K_w)
        combined_mask = kernel_h_mask & kernel_w_mask
        
        # Load weights
        w_offset = (
            c_in_block[:, None, None, None] * (C_out * K_h * K_w) +
            out_c_idx[None, :, None, None] * (K_h * K_w) +
            kernel_h[None, None, :, :] * K_w +
            kernel_w[None, None, :, :]
        )
        w_mask = combined_mask[None, None, :, :] & (c_in_block[:, None, None, None] < C_in) & (out_c_idx[None, :, None, None] < C_out)
        w_val = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)
        
        # Compute contribution to accumulator
        # x_val has shape [BLOCK_SIZE_K, BLOCK_SIZE_H, BLOCK_SIZE_W]
        # w_val has shape [BLOCK_SIZE_K, BLOCK_SIZE_M, BLOCK_SIZE_H, BLOCK_SIZE_W]
        acc += tl.sum(x_val[:, None, :, :] * w_val, axis=0)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_c_idx)
        acc += bias[:, None, None]
    
    # Store result
    out_offset = (
        batch_idx * (C_out * H_out * W_out) +
        out_c_idx[:, None, None] * (H_out * W_out) +
        out_h[None, :, :] * W_out +
        out_w[None, :, :]
    )
    out_mask = (out_c_idx[:, None, None] < C_out) & valid_mask[None, :, :]
    tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty), mask=out_mask)


def triton_conv_transpose2d(x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
    """
    Performs transposed 2D convolution using Triton kernel.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    B, C_in, H_in, W_in = x.shape
    C_in2, C_out, K_h, K_w = weight.shape
    assert C_in == C_in2, "Input channels must match"
    
    # Calculate output dimensions
    H_out = (H_in - 1) * stride - 2 * padding + K_h + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + K_w + output_padding
    
    # Create output tensor
    out = torch.empty(B, C_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Set block sizes (tunable parameters)
    BLOCK_SIZE_M = 8   # Output channels per block
    BLOCK_SIZE_N = 1   # Batch per block (fixed to 1 for simplicity)
    BLOCK_SIZE_K = 8   # Input channels per block
    BLOCK_SIZE_H = 4   # Output height per block
    BLOCK_SIZE_W = 4   # Output width per block
    
    # Calculate grid dimensions
    grid = (
        B,  # batch
        (C_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,  # output channels
        (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,  # output height
        (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W,  # output width
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias, out,
        B, C_in, C_out,
        H_in, W_in,
        K_h, K_w,
        stride, stride,
        padding, padding,
        output_padding, output_padding,
        H_out, W_out,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the transposed 2D convolution model using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store the same parameters as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Initialize the weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized transposed 2D convolution using Triton kernel.
        """
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )