import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Input tensor (B, C_in, H_in, W_in)
    w_ptr,  # Weight tensor (C_in, C_out, K_h, K_w)
    b_ptr,  # Bias tensor (C_out,)
    y_ptr,  # Output tensor (B, C_out, H_out, W_out)
    B: tl.constexpr,  # Batch size
    C_in: tl.constexpr,  # Input channels
    C_out: tl.constexpr,  # Output channels
    H_in: tl.constexpr,  # Input height
    W_in: tl.constexpr,  # Input width
    K_h: tl.constexpr,  # Kernel height
    K_w: tl.constexpr,  # Kernel width
    H_out: tl.constexpr,  # Output height
    W_out: tl.constexpr,  # Output width
    stride: tl.constexpr,
    padding: tl.constexpr,
    output_padding: tl.constexpr,
    groups: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr = 32,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr = 32,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr = 32,  # Block size for accumulation
):
    # Program IDs
    pid_c = tl.program_id(0)  # Output channel block
    pid_b = tl.program_id(1)  # Batch block
    pid_h = tl.program_id(2)  # Output height position
    pid_w = tl.program_id(3)  # Output width position
    
    # Calculate the starting positions in output tensor
    out_c_start = pid_c * BLOCK_SIZE_M
    out_h_start = pid_h * BLOCK_SIZE_M  # Reusing BLOCK_SIZE_M for simplicity
    out_w_start = pid_w * BLOCK_SIZE_M  # Reusing BLOCK_SIZE_M for simplicity
    
    # Create ranges for output channels, batch, height, width
    out_c_offsets = out_c_start + tl.arange(0, BLOCK_SIZE_M)
    batch_offset = pid_b
    out_h_offset = pid_h
    out_w_offset = pid_w
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Compute transposed convolution: for each output position, accumulate over input positions
    # For transposed conv: y[b, c_out, h_out, w_out] = sum_{k1, k2, c_in} x[b, c_in, h_out - k1*stride + padding, w_out - k2*stride + padding] * w[c_in, c_out, k1, k2]
    
    # Iterate over input channels
    for c_in_idx in range(0, C_in, BLOCK_SIZE_K):
        c_in_offsets = c_in_idx + tl.arange(0, BLOCK_SIZE_K)
        c_in_mask = c_in_offsets < C_in
        
        # Iterate over kernel positions
        for kh in range(K_h):
            for kw in range(K_w):
                # Compute corresponding input position
                h_in = out_h_start * stride + kh - padding
                w_in = out_w_start * stride + kw - padding
                
                # Check if input position is valid
                valid_h = (h_in >= 0) & (h_in < H_in)
                valid_w = (w_in >= 0) & (w_in < W_in)
                valid = valid_h & valid_w
                
                if valid:
                    # Load input: x[b, c_in, h_in, w_in]
                    x_offset = batch_offset * (C_in * H_in * W_in) + \
                              c_in_offsets * (H_in * W_in) + \
                              h_in * W_in + \
                              w_in
                    x_val = tl.load(x_ptr + x_offset, mask=c_in_mask & (c_in_offsets < C_in), other=0.0)
                    
                    # Load weights: w[c_in, c_out, kh, kw]
                    w_offset = c_in_offsets * (C_out * K_h * K_w) + \
                              out_c_offsets * (K_h * K_w) + \
                              kh * (K_w) + \
                              kw
                    w_val = tl.load(w_ptr + w_offset, mask=c_in_mask[:, None] & (out_c_offsets < C_out), other=0.0)
                    
                    # Accumulate: x_val * w_val
                    # x_val has shape (BLOCK_SIZE_K,), w_val has shape (BLOCK_SIZE_K, BLOCK_SIZE_M)
                    acc += tl.sum(x_val[:, None] * w_val, axis=0)
    
    # Apply bias if present
    if has_bias:
        b_val = tl.load(b_ptr + out_c_offsets, mask=out_c_offsets < C_out, other=0.0)
        acc += b_val
    
    # Store result
    y_offset = batch_offset * (C_out * H_out * W_out) + \
              out_c_offsets * (H_out * W_out) + \
              out_h_offset * W_out + \
              out_w_offset
    tl.store(y_ptr + y_offset, acc.to(x_ptr.dtype.element_ty), mask=out_c_offsets < C_out)


class TritonConvTranspose2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, output_padding, groups):
        # Get dimensions
        B, C_in, H_in, W_in = x.shape
        C_in_w, C_out, K_h, K_w = weight.shape
        
        # Calculate output dimensions (same as PyTorch's ConvTranspose2d)
        H_out = (H_in - 1) * stride - 2 * padding + K_h + output_padding
        W_out = (W_in - 1) * stride - 2 * padding + K_w + output_padding
        
        # Create output tensor
        y = torch.empty((B, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
        
        # Configure kernel launch parameters
        # We'll use 4D grid: [C_out_blocks, B, H_out_blocks, W_out_blocks]
        BLOCK_SIZE_M = 16  # Adjust for better performance
        BLOCK_SIZE_N = 16
        
        # Calculate number of blocks
        grid_c = (C_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        grid_b = B
        grid_h = (H_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        grid_w = (W_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        
        grid = (grid_c, grid_b, grid_h, grid_w)
        
        # Launch kernel
        conv_transpose2d_kernel[grid](
            x, weight, bias if bias is not None else torch.empty(1, device=x.device),
            y, B, C_in, C_out, H_in, W_in, K_h, K_w, H_out, W_out,
            stride, padding, output_padding, groups,
            has_bias=bias is not None,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=16,
        )
        
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        ctx.groups = groups
        ctx.input_shape = x.shape
        ctx.weight_shape = weight.shape
        
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        # For simplicity, we'll use PyTorch's built-in backward for gradients
        # This is a practical approach since implementing backward for transposed conv is complex
        x, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        output_padding = ctx.output_padding
        groups = ctx.groups
        
        # Use PyTorch's native backward for gradients
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            # Compute grad_input using PyTorch's conv_transpose2d with grad_output
            grad_input = torch.nn.grad.conv2d_input(
                x.shape, weight, grad_output, stride=stride, padding=padding,
                output_padding=output_padding, groups=groups
            )
        
        if ctx.needs_input_grad[1]:
            # Compute grad_weight using PyTorch's conv2d_weight
            grad_weight = torch.nn.grad.conv2d_weight(
                x, weight.shape, grad_output, stride=stride, padding=padding,
                groups=groups
            )
        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum([0, 2, 3])
        
        return grad_input, grad_weight, grad_bias, None, None, None, None


def triton_conv_transpose2d(x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
    return TritonConvTranspose2d.apply(x, weight, bias, stride, padding, output_padding, groups)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Create the parameters manually to use in our custom kernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Create weight and bias parameters (same shape as nn.ConvTranspose2d)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters (same initialization as PyTorch's default)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use our custom Triton implementation
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.output_padding, self.groups
        )