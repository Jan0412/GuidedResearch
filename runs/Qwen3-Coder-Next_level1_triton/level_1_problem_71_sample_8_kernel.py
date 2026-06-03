import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Input tensor (N, C_in, H_in, W_in)
    w_ptr,  # Weight tensor (C_in, C_out, K_h, K_w)
    b_ptr,  # Bias tensor (C_out,) or None
    y_ptr,  # Output tensor (N, C_out, H_out, W_out)
    N, C_in, C_out, H_in, W_in, K_h, K_w, 
    stride, padding, output_padding,
    H_out, W_out,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for batch dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_K: tl.constexpr,  # Block size for input channels
    BLOCK_H: tl.constexpr,       # Block size for output height
    BLOCK_W: tl.constexpr,       # Block size for output width
):
    # Program IDs
    pid_n = tl.program_id(0)  # batch index
    pid_c_out = tl.program_id(1)  # output channel index
    
    # Compute the range of output channels for this block
    c_out_start = pid_c_out * BLOCK_SIZE_N
    c_out_offsets = c_out_start + tl.arange(0, BLOCK_SIZE_N)
    c_out_mask = c_out_offsets < C_out
    
    # Compute the range of output height and width for this block
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    h_start = pid_h * BLOCK_H
    w_start = pid_w * BLOCK_W
    
    h_offsets = h_start + tl.arange(0, BLOCK_H)[:, None]
    w_offsets = w_start + tl.arange(0, BLOCK_W)[None, :]
    
    # Create mask for valid output positions
    h_mask = h_offsets < H_out
    w_mask = w_offsets < W_out
    combined_mask = h_mask & w_mask
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Loop over input channels
    for c_in in range(0, C_in, BLOCK_SIZE_K):
        c_in_offsets = c_in + tl.arange(0, BLOCK_SIZE_K)
        c_in_mask = c_in_offsets < C_in
        
        # Load input block: (BLOCK_H, BLOCK_W) for each input channel
        # Calculate corresponding input positions for this output position
        # For transposed conv: h_in = (h_out - output_padding - 1) // stride + padding
        # Actually, we iterate over output positions and accumulate from input
        
        # For efficiency, we'll compute the offset calculation differently
        # We'll compute which input positions contribute to which output positions
        
        # Process each position in the current output block
        for bh in range(BLOCK_H):
            for bw in range(BLOCK_W):
                if h_start + bh < H_out and w_start + bw < W_out:
                    # Calculate corresponding input position
                    h_in = (h_start + bh - output_padding) // stride
                    w_in = (w_start + bw - output_padding) // stride
                    
                    # Check if valid input position
                    if (h_start + bh - output_padding) % stride == 0 and \
                       (w_start + bw - output_padding) % stride == 0 and \
                       h_in >= 0 and h_in < H_in and w_in >= 0 and w_in < W_in:
                        
                        # Calculate input index
                        input_idx = pid_n * (C_in * H_in * W_in) + \
                                   c_in_offsets[:, None, None] * (H_in * W_in) + \
                                   h_in * W_in + w_in
                        input_mask = c_in_mask[:, None, None]
                        x_block = tl.load(x_ptr + input_idx, mask=input_mask, other=0.0)
                        
                        # Load weights: for transposed conv, weight shape is (C_in, C_out, K_h, K_w)
                        # We need weights for the specific kernel positions that map to this output
                        # For each input channel and output channel pair, we need the kernel position
                        # The kernel position is: h_k = h_out - stride*h_in - padding
                        h_k = h_start + bh - stride * h_in - padding
                        w_k = w_start + bw - stride * w_in - padding
                        
                        weight_idx = c_in_offsets[:, None] * (C_out * K_h * K_w) + \
                                    c_out_offsets[None, :] * (K_h * K_w) + \
                                    h_k * K_w + w_k
                        weight_mask = c_in_mask[:, None] & c_out_mask[None, :]
                        w_block = tl.load(w_ptr + weight_idx, mask=weight_mask, other=0.0)
                        
                        # Accumulate: x * w
                        acc += x_block * w_block[None, :, :]
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + c_out_offsets, mask=c_out_mask, other=0.0)
        acc += bias[:, None, None]
    
    # Store result
    output_idx = pid_n * (C_out * H_out * W_out) + \
                c_out_offsets[:, None, None] * (H_out * W_out) + \
                h_offsets[None, :, :] * W_out + w_offsets[None, :, :]
    output_mask = c_out_mask[:, None, None] & combined_mask[None, :, :]
    
    tl.store(y_ptr + output_idx, acc, mask=output_mask)


class TritonConvTranspose2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, output_padding):
        N, C_in, H_in, W_in = x.shape
        C_in2, C_out, K_h, K_w = weight.shape
        
        # Calculate output dimensions
        H_out = (H_in - 1) * stride - 2 * padding + output_padding + K_h
        W_out = (W_in - 1) * stride - 2 * padding + output_padding + K_w
        
        # Create output tensor
        y = torch.empty((N, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
        
        # Configure kernel launch parameters
        BLOCK_SIZE_N = 16  # Block size for output channels
        BLOCK_SIZE_K = 8   # Block size for input channels
        BLOCK_H = 8        # Block size for output height
        BLOCK_W = 8        # Block size for output width
        
        grid = (
            N,  # batch dimension
            (C_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,  # output channels
            (H_out + BLOCK_H - 1) // BLOCK_H,  # output height
            (W_out + BLOCK_W - 1) // BLOCK_W   # output width
        )
        
        conv_transpose2d_kernel[grid](
            x, weight, bias, y,
            N, C_in, C_out, H_in, W_in, K_h, K_w,
            stride, padding, output_padding,
            H_out, W_out,
            BLOCK_SIZE_M=1,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_H=BLOCK_H,
            BLOCK_W=BLOCK_W
        )
        
        ctx.save_for_backward(x, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        # For simplicity, fall back to PyTorch for backward pass
        x, weight = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        output_padding = ctx.output_padding
        
        # Use PyTorch's built-in backward for simplicity
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(
                x.shape, weight, grad_output, stride=stride, 
                padding=padding, output_padding=output_padding
            )
        
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(
                x, weight.shape, grad_output, stride=stride,
                padding=padding, output_padding=output_padding
            )
        
        if ctx.needs_input_grad[2] and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=[0, 2, 3])
        
        return grad_input, grad_weight, grad_bias, None, None, None


def triton_conv_transpose2d(x, weight, bias=None, stride=1, padding=0, output_padding=0):
    return TritonConvTranspose2d.apply(x, weight, bias, stride, padding, output_padding)


class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with asymmetric input and a square kernel.
    Optimized with Triton kernels for FP32 precision.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Validate parameters
        assert groups == 1, "Triton kernel only supports groups=1 for now"
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        # Kaiming initialization for transposed convolution
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using Triton kernel.
        """
        return triton_conv_transpose2d(
            x, self.weight, self.bias, 
            self.stride, self.padding, self.output_padding
        )


# Import math for initialization
import math