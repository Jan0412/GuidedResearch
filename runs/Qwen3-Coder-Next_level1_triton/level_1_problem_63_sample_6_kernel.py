import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, H, W)
    w_ptr,  # Weight tensor: (out_channels, in_channels, kH, kW)
    b_ptr,  # Bias tensor: (out_channels,) or None
    out_ptr,  # Output tensor: (batch, out_channels, H_out, W_out)
    batch_size, in_channels, out_channels,
    H, W, kH, kW,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    H_out, W_out,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr,  # Block size for in_channels
    BLOCK_H: tl.constexpr,       # Block height for output
    BLOCK_W: tl.constexpr,       # Block width for output
):
    # Program IDs
    pid_batch = tl.program_id(1)
    pid_out_c = tl.program_id(0)
    
    # Check bounds
    if pid_batch >= batch_size or pid_out_c >= out_channels:
        return
    
    # Calculate output position
    out_h_start = tl.program_id(2) * BLOCK_H
    out_w_start = 0  # We'll iterate over W in blocks
    
    # Allocate space for output values
    output = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Convolution loop over input channels
    for ic in range(in_channels):
        # Loop over output height
        for bh in range(BLOCK_H):
            out_h = out_h_start + bh
            # Check if this output position is valid
            if out_h < H_out:
                # Calculate input position
                in_h = out_h * stride_h - pad_h + bh * dil_h
                # Loop over output width
                for bw in range(BLOCK_W):
                    out_w = out_w_start + bw
                    if out_w < W_out:
                        in_w = out_w * stride_w - pad_w + bw * dil_w
                        # Load input value if in bounds
                        if 0 <= in_h < H and 0 <= in_w < W:
                            x_offset = pid_batch * (in_channels * H * W) + \
                                      ic * (H * W) + \
                                      in_h * W + in_w
                            x_val = tl.load(x_ptr + x_offset)
                        else:
                            x_val = 0.0
                        
                        # Load weight values
                        w_offset = pid_out_c * (in_channels * kH * kW) + \
                                  ic * (kH * kW) + \
                                  bh * kW + bw
                        w_val = tl.load(w_ptr + w_offset)
                        
                        # Accumulate
                        output = tl.where(
                            (out_h < H_out) & (out_w < W_out),
                            output + x_val * w_val,
                            output
                        )
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_out_c)
        output = output + bias
    
    # Store output
    for bh in range(BLOCK_H):
        out_h = out_h_start + bh
        if out_h < H_out:
            for bw in range(BLOCK_W):
                out_w = out_w_start + bw
                if out_w < W_out:
                    out_offset = pid_batch * (out_channels * H_out * W_out) + \
                                pid_out_c * (H_out * W_out) + \
                                out_h * W_out + out_w
                    tl.store(out_ptr + out_offset, output[bh, bw])


class TritonConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, groups):
        # Extract dimensions
        batch_size, in_channels, H, W = x.shape
        out_channels, _, kH, kW = weight.shape
        
        # Calculate output dimensions
        stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride)
        pad_h, pad_w = padding if isinstance(padding, tuple) else (padding, padding)
        dil_h, dil_w = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        
        H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) // stride_h + 1
        W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) // stride_w + 1
        
        # Allocate output tensor
        out = torch.empty((batch_size, out_channels, H_out, W_out), device=x.device, dtype=x.dtype)
        
        # Set up kernel launch parameters
        BLOCK_SIZE_M = 32  # Output channels per block
        BLOCK_SIZE_N = 4   # Batch per block
        BLOCK_SIZE_K = 16  # Input channels per block
        BLOCK_H = 8        # Output height per block
        BLOCK_W = 32       # Output width per block
        
        # Grid dimensions
        grid = lambda meta: (
            triton.cdiv(out_channels, meta['BLOCK_SIZE_M']),
            batch_size,
            triton.cdiv(H_out, meta['BLOCK_H'])
        )
        
        # Launch kernel
        conv2d_kernel[grid](
            x, weight, bias, out,
            batch_size, in_channels, out_channels,
            H, W, kH, kW,
            stride_h, stride_w,
            pad_h, pad_w,
            dil_h, dil_w,
            H_out, W_out,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_H=BLOCK_H,
            BLOCK_W=BLOCK_W
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # Implement backward pass using PyTorch for simplicity
        x, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        
        # Use PyTorch's built-in conv2d for gradient computation
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(x.shape, weight, grad_output, 
                                                   stride, padding, dilation, groups)
        
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(x, weight.shape, grad_output,
                                                     stride, padding, dilation, groups)
        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=[0, 2, 3])
        
        return grad_input, grad_weight, grad_bias, None, None, None, None


class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with a square input and square kernel.
    Uses custom Triton kernel for forward pass.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using custom Triton kernel.
        """
        return TritonConv2dFunction.apply(x, self.weight, self.bias, 
                                         self.stride, self.padding, 
                                         self.dilation, self.groups)