import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor: (B, C_in, H, W)
    w_ptr,  # Weight tensor: (C_out, C_in, K, K)
    b_ptr,  # Bias tensor: (C_out,) or None
    y_ptr,  # Output tensor: (B, C_out, H_out, W_out)
    B, C_in, H, W,
    C_out, K,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    H_out, W_out,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch dimension
    BLOCK_SIZE_K: tl.constexpr,  # Block size for input channels
    BLOCK_SIZE_P: tl.constexpr,  # Block size for height
    BLOCK_SIZE_Q: tl.constexpr,  # Block size for width
):
    # Get program IDs
    pid_m = tl.program_id(0)  # For output channels
    pid_n = tl.program_id(1)  # For batch
    pid_h = tl.program_id(2)  # For height blocks
    pid_w = tl.program_id(3)  # For width blocks
    
    # Calculate output position
    out_c_start = pid_m * BLOCK_SIZE_M
    batch_idx = pid_n * BLOCK_SIZE_N
    out_h_start = pid_h * BLOCK_SIZE_P
    out_w_start = pid_w * BLOCK_SIZE_Q
    
    # Offset pointers for batch
    x_ptr_batch = x_ptr + batch_idx * C_in * H * W
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_P, BLOCK_SIZE_Q), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for c_in_idx in range(0, C_in, BLOCK_SIZE_K):
        c_in_end = tl.minimum(c_in_idx + BLOCK_SIZE_K, C_in)
        c_in_block = c_in_end - c_in_idx
        
        # Load input block: (C_in_block, H_out*BLOCK_SIZE_P, W_out*BLOCK_SIZE_Q)
        # We'll compute offsets for the convolution window
        for kh in range(K):
            for kw in range(K):
                # Compute input spatial positions
                h_pos = out_h_start * stride_h + kh * dil_h - pad_h
                w_pos = out_w_start * stride_w + kw * dil_w - pad_w
                
                # Create offset arrays for the current kernel position
                offsets_h = tl.arange(0, BLOCK_SIZE_P)
                offsets_w = tl.arange(0, BLOCK_SIZE_Q)
                h_indices = h_pos + offsets_h * stride_h
                w_indices = w_pos + offsets_w * stride_w
                
                # Create masks
                mask_h = (h_indices >= 0) & (h_indices < H)
                mask_w = (w_indices >= 0) & (w_indices < W)
                mask = mask_h[:, None] & mask_w[None, :]
                
                # Load input values: shape (C_in_block, BLOCK_SIZE_P, BLOCK_SIZE_Q)
                # We need to load each input channel separately due to complex indexing
                for c_in_local in range(c_in_idx, c_in_end):
                    x_offset = c_in_local * H * W + h_indices[:, None] * W + w_indices[None, :]
                    x_vals = tl.load(x_ptr_batch + x_offset, mask=mask, other=0.0)
                    
                    # Load corresponding weight
                    w_offset = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None, None] * (C_in * K * K) + \
                               c_in_local * (K * K) + kh * K + kw
                    w_vals = tl.load(w_ptr + w_offset)
                    
                    # Accumulate: (BLOCK_SIZE_M, BLOCK_SIZE_P, BLOCK_SIZE_Q)
                    acc += x_vals[None, :, :] * w_vals[:, :, :]
    
    # Add bias if present
    if b_ptr is not None:
        b_offset = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None, None]
        bias = tl.load(b_ptr + b_offset)
        acc += bias
    
    # Store result
    y_offset = (batch_idx + pid_n * BLOCK_SIZE_N) * C_out * H_out * W_out + \
               (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None, None] * H_out * W_out + \
               (out_h_start + tl.arange(0, BLOCK_SIZE_P))[:, None] * W_out + \
               (out_w_start + tl.arange(0, BLOCK_SIZE_Q))[None, :]
    
    mask_y = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None, None] < C_out
    mask_y = mask_y & (out_h_start + tl.arange(0, BLOCK_SIZE_P))[:, None] < H_out
    mask_y = mask_y & (out_w_start + tl.arange(0, BLOCK_SIZE_Q))[None, :] < W_out
    
    tl.store(y_ptr + y_offset, acc, mask=mask_y)


class TritonConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, groups):
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        
        # Get dimensions
        B, C_in, H, W = x.shape
        C_out, _, K, _ = weight.shape
        stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride)
        pad_h, pad_w = padding if isinstance(padding, tuple) else (padding, padding)
        dil_h, dil_w = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        
        # Calculate output dimensions
        H_out = (H + 2 * pad_h - dil_h * (K - 1) - 1) // stride_h + 1
        W_out = (W + 2 * pad_w - dil_w * (K - 1) - 1) // stride_w + 1
        
        # Allocate output
        y = torch.empty((B, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
        
        # Define block sizes for optimization
        BLOCK_SIZE_M = 16  # Output channels per block
        BLOCK_SIZE_N = 1   # Batch per block
        BLOCK_SIZE_K = 8   # Input channels per block
        BLOCK_SIZE_P = 8   # Height per block
        BLOCK_SIZE_Q = 32  # Width per block (larger due to asymmetric input)
        
        # Grid dimensions
        grid = lambda meta: (
            triton.cdiv(C_out, meta['BLOCK_SIZE_M']),
            triton.cdiv(B, meta['BLOCK_SIZE_N']),
            triton.cdiv(H_out, meta['BLOCK_SIZE_P']),
            triton.cdiv(W_out, meta['BLOCK_SIZE_Q'])
        )
        
        # Launch kernel
        conv2d_kernel[grid](
            x, weight, bias, y,
            B, C_in, H, W,
            C_out, K,
            stride_h, stride_w,
            pad_h, pad_w,
            dil_h, dil_w,
            H_out, W_out,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_P=BLOCK_SIZE_P,
            BLOCK_SIZE_Q=BLOCK_SIZE_Q
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.input_size = (B, C_in, H, W)
        ctx.output_size = (B, C_out, H_out, W_out)
        
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        # This is a simplified implementation for inference-only
        # In production, you'd want proper backward implementation
        x, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            # For simplicity, fall back to PyTorch for backward
            grad_input = torch.nn.grad.conv2d_input(
                x.shape, weight, grad_output, ctx.stride, ctx.padding,
                ctx.dilation, ctx.groups
            )
        
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(
                x, weight.shape, grad_output, ctx.stride, ctx.padding,
                ctx.dilation, ctx.groups
            )
        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum([0, 2, 3])
        
        return grad_input, grad_weight, grad_bias, None, None, None, None


def triton_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return TritonConv2dFunction.apply(x, weight, bias, stride, padding, dilation, groups)


class ModelNew(nn.Module):
    """
    Optimized version of the 2D convolution model using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Register buffers to maintain compatibility
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create parameters manually since we're not using nn.Conv2d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 2D convolution using Triton kernel.
        """
        return triton_conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)