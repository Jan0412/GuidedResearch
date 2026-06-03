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
    B, C_in, C_out, 
    H_in, W_in,
    H_out, W_out,
    K_h, K_w,
    stride_h, stride_w,
    padding_h, padding_w,
    output_padding_h, output_padding_w,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr,  # Block size for accumulation (C_in)
):
    # Output tensor indices
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    out_w_idx = tl.program_id(3)
    
    # Compute input position that contributes to this output position
    # For transposed convolution: input_pos = (output_pos - kernel_pos + stride) // stride
    # But more directly: output[h_out, w_out] += input[h_in, w_in] * kernel[kh, kw]
    # where h_in = (h_out - kh + stride) // stride, similarly for w
    
    # Accumulator
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Iterate over input channels
    for in_c in range(C_in):
        # Iterate over kernel height
        for kh in range(K_h):
            # Compute input height index
            h_in = out_h_idx - kh + padding_h
            if h_in >= 0 and h_in < H_in and (h_in % stride_h == 0):
                h_in //= stride_h
                # Iterate over kernel width
                for kw in range(K_w):
                    # Compute input width index
                    w_in = out_w_idx - kw + padding_w
                    if w_in >= 0 and w_in < W_in and (w_in % stride_w == 0):
                        w_in //= stride_w
                        
                        # Load input value
                        x_offset = batch_idx * (C_in * H_in * W_in) + \
                                   in_c * (H_in * W_in) + \
                                   h_in * W_in + w_in
                        x_val = tl.load(x_ptr + x_offset)
                        
                        # Load weight value
                        w_offset = in_c * (C_out * K_h * K_w) + \
                                   out_c_idx * (K_h * K_w) + \
                                   kh * K_w + kw
                        w_val = tl.load(w_ptr + w_offset)
                        
                        # Accumulate
                        acc += x_val * w_val
    
    # Load bias if available
    if b_ptr is not None:
        bias_offset = out_c_idx
        acc += tl.load(b_ptr + bias_offset)
    
    # Store result
    out_offset = batch_idx * (C_out * H_out * W_out) + \
                 out_c_idx * (H_out * W_out) + \
                 out_h_idx * W_out + out_w_idx
    tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty))


class TritonConvTranspose2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, output_padding, groups):
        # Get dimensions
        B, C_in, H_in, W_in = x.shape
        C_out, C_in_group, K_h, K_w = weight.shape
        
        # Compute output dimensions
        H_out = (H_in - 1) * stride[0] - 2 * padding[0] + K_h + output_padding[0]
        W_out = (W_in - 1) * stride[1] - 2 * padding[1] + K_w + output_padding[1]
        
        # Create output tensor
        out = torch.empty((B, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
        
        # Launch kernel
        # Note: For simplicity, we use a 4D grid over batch, out_channels, height, width
        # In practice, we might want to optimize the block size
        def grid(meta):
            return (B, C_out, H_out, W_out)
        
        # For the kernel, we need to set up parameters
        conv_transpose2d_kernel[grid](
            x, weight, bias, out,
            B, C_in, C_out,
            H_in, W_in,
            H_out, W_out,
            K_h, K_w,
            stride[0], stride[1],
            padding[0], padding[1],
            output_padding[0], output_padding[1],
            BLOCK_SIZE_M=1,
            BLOCK_SIZE_N=1,
            BLOCK_SIZE_K=1,
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        ctx.groups = groups
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        output_padding = ctx.output_padding
        groups = ctx.groups
        
        B, C_in, H_in, W_in = x.shape
        C_out, _, K_h, K_w = weight.shape
        
        # Compute gradients
        # For transposed conv, backward pass involves:
        # 1. Gradient w.r.t. input: conv(grad_output, weight_t) - another transposed conv
        # 2. Gradient w.r.t. weight: conv(x_t, grad_output_t)
        
        # Simple implementation using PyTorch for backward for now
        # This maintains autograd functionality
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            # Gradient w.r.t. input is a regular convolution of grad_output with flipped weight
            grad_input = nn.functional.conv2d(
                grad_output, 
                weight.flip([2, 3]).permute(1, 0, 2, 3),
                stride=stride, padding=padding, groups=groups
            )
        
        if ctx.needs_input_grad[1]:
            # Gradient w.r.t. weight is a bit more complex
            # For simplicity, use PyTorch's implementation
            grad_weight = torch.empty_like(weight)
            # Using PyTorch's native implementation for weight gradient
            # This maintains correctness while using our custom forward
            grad_weight = nn.functional.conv2d(
                x.permute(1, 0, 2, 3),  # (C_in, B, H_in, W_in)
                grad_output.permute(1, 0, 2, 3),  # (C_out, B, H_out, W_out)
                stride=stride, padding=padding, groups=groups,
                output_padding=output_padding
            ).permute(1, 0, 2, 3)
        
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=[0, 2, 3])
        
        return grad_input, grad_weight, grad_bias, None, None, None, None


def triton_conv_transpose2d(x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
    # Convert parameters to tuples if needed
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding)
    
    return TritonConvTranspose2d.apply(x, weight, bias, stride, padding, output_padding, groups)


class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with asymmetric input and a square kernel.
    Uses custom Triton kernel for forward pass.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same way as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using custom Triton kernel.
        """
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.output_padding, self.groups
        )