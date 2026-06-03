import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Input tensor: (B, C_in, H_in, W_in)
    w_ptr,  # Weight tensor: (C_in, C_out, K_h, K_w)
    b_ptr,  # Bias tensor: (C_out,) or None
    out_ptr,  # Output tensor: (B, C_out, H_out, W_out)
    B, C_in, C_out, K_h, K_w,
    H_in, W_in, H_out, W_out,
    stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w,
    BLOCK_SIZE_C_out: tl.constexpr,
    BLOCK_SIZE_C_in: tl.constexpr,
    BLOCK_SIZE_K_h: tl.constexpr,
    BLOCK_SIZE_K_w: tl.constexpr,
    BLOCK_SIZE_H_out: tl.constexpr,
    BLOCK_SIZE_W_out: tl.constexpr,
):
    # Program IDs for output dimensions
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1) // (BLOCK_SIZE_H_out * BLOCK_SIZE_W_out)
    pid_h_out = (tl.program_id(1) // BLOCK_SIZE_W_out) % BLOCK_SIZE_H_out
    pid_w_out = tl.program_id(1) % BLOCK_SIZE_W_out
    
    # Compute actual output coordinates
    h_out_start = pid_h_out * BLOCK_SIZE_H_out
    w_out_start = pid_w_out * BLOCK_SIZE_W_out
    
    # Create output tile indices
    h_offsets = h_out_start + tl.arange(0, BLOCK_SIZE_H_out)
    w_offsets = w_out_start + tl.arange(0, BLOCK_SIZE_W_out)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H_out, BLOCK_SIZE_W_out), dtype=tl.float32)
    
    # Loop over C_in, K_h, K_w to accumulate contributions
    for c_in in range(C_in):
        for k_h in range(K_h):
            for k_w in range(K_w):
                # Compute input position
                h_in = h_out_start * stride_h + k_h - pad_h
                w_in = w_out_start * stride_w + k_w - pad_w
                
                # Check if input position is valid
                valid_in = (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in)
                
                # Load input value if valid
                if valid_in:
                    x_offset = ((pid_b * C_in + c_in) * H_in + h_in) * W_in + w_in
                    x_val = tl.load(x_ptr + x_offset)
                else:
                    x_val = 0.0
                
                # Load weight value
                w_offset = ((c_in * C_out + pid_c_out) * K_h + k_h) * K_w + k_w
                w_val = tl.load(w_ptr + w_offset)
                
                # Accumulate
                acc += x_val * w_val
    
    # Add bias if available
    if b_ptr is not None:
        b_offset = pid_c_out
        bias = tl.load(b_ptr + b_offset)
        acc += bias
    
    # Store result
    for h_idx in range(BLOCK_SIZE_H_out):
        for w_idx in range(BLOCK_SIZE_W_out):
            h = h_out_start + h_idx
            w = w_out_start + w_idx
            if h < H_out and w < W_out:
                out_offset = ((pid_b * C_out + pid_c_out) * H_out + h) * W_out + w
                tl.store(out_ptr + out_offset, acc[h_idx, w_idx])


class TritonConvTranspose2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, output_padding, groups):
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        
        # Extract dimensions
        B, C_in, H_in, W_in = x.shape
        C_out, _, K_h, K_w = weight.shape
        stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride)
        pad_h, pad_w = padding if isinstance(padding, tuple) else (padding, padding)
        out_pad_h, out_pad_w = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        
        # Calculate output dimensions
        H_out = (H_in - 1) * stride_h - 2 * pad_h + K_h + out_pad_h
        W_out = (W_in - 1) * stride_w - 2 * pad_w + K_w + out_pad_w
        
        # Create output tensor
        out = torch.empty(B, C_out, H_out, W_out, dtype=x.dtype, device=x.device)
        
        # Configure kernel launch parameters
        BLOCK_SIZE_C_out = 1
        BLOCK_SIZE_C_in = 8
        BLOCK_SIZE_K_h = 3
        BLOCK_SIZE_K_w = 3
        BLOCK_SIZE_H_out = 8
        BLOCK_SIZE_W_out = 8
        
        grid = (B, (C_out * ((H_out + BLOCK_SIZE_H_out - 1) // BLOCK_SIZE_H_out) * 
                    ((W_out + BLOCK_SIZE_W_out - 1) // BLOCK_SIZE_W_out)))
        
        # Launch kernel
        conv_transpose2d_kernel[grid](
            x, weight, bias, out,
            B, C_in, C_out, K_h, K_w,
            H_in, W_in, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w,
            BLOCK_SIZE_C_out=BLOCK_SIZE_C_out,
            BLOCK_SIZE_C_in=BLOCK_SIZE_C_in,
            BLOCK_SIZE_K_h=BLOCK_SIZE_K_h,
            BLOCK_SIZE_K_w=BLOCK_SIZE_K_w,
            BLOCK_SIZE_H_out=BLOCK_SIZE_H_out,
            BLOCK_SIZE_W_out=BLOCK_SIZE_W_out,
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        ctx.groups = groups
        ctx.input_size = (B, C_in, H_in, W_in)
        ctx.output_size = (B, C_out, H_out, W_out)
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # This is a simplified implementation - for full training support, 
        # we would need to implement backward kernels as well
        x, weight, bias = ctx.saved_tensors
        
        # For now, just use PyTorch's native backward for simplicity
        # In production, you'd implement dedicated backward kernels
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.functional.conv_transpose2d(
                grad_output, weight, None, ctx.stride, ctx.padding,
                ctx.output_padding, ctx.groups
            )
        
        if ctx.needs_input_grad[1] and weight is not None:
            # Compute gradient w.r.t. weight using PyTorch
            # This would be optimized in a full implementation
            grad_weight = torch.empty_like(weight)
            # Use PyTorch's native implementation for backward
            grad_weight.copy_(torch.autograd.grad(
                outputs=grad_output, inputs=weight, grad_outputs=grad_output,
                retain_graph=False, create_graph=False
            )[0])
        
        if ctx.needs_input_grad[2] and bias is not None:
            grad_bias = torch.sum(grad_output, dim=(0, 2, 3))
        
        return grad_input, grad_weight, grad_bias, None, None, None, None


class ModelNew(nn.Module):
    """
    Optimized transposed 2D convolution using Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized transposed 2D convolution.
        """
        return TritonConvTranspose2d.apply(
            x, self.weight, self.bias, 
            self.stride, self.padding, self.output_padding, self.groups
        )