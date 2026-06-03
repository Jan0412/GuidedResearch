import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    # Pointers to tensors
    x_ptr,                # Input tensor (N, C_in, H, W)
    w_ptr,                # Weight tensor (C_out, C_in, K_h, K_w)
    b_ptr,                # Bias tensor (C_out,) or None
    out_ptr,              # Output tensor (N, C_out, H_out, W_out)
    # Tensor dimensions
    N, C_in, H, W,        # Input dimensions
    C_out,                # Output channels
    K_h, K_w,             # Kernel size
    stride_h, stride_w,   # Stride
    pad_h, pad_w,         # Padding
    dilation_h, dilation_w,  # Dilation
    # Output dimensions
    H_out, W_out,
    # Meta-parameters
    BLOCK_SIZE_N: tl.constexpr,    # Batch block size
    BLOCK_SIZE_C_OUT: tl.constexpr, # Output channel block size
    BLOCK_SIZE_C_IN: tl.constexpr,  # Input channel block size
    BLOCK_SIZE_H: tl.constexpr,    # Output height block size
    BLOCK_SIZE_W: tl.constexpr,    # Output width block size
):
    # Get program indices
    pid_n = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Compute output batch index
    batch_idx = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[:, None, None]
    batch_mask = batch_idx < N
    
    # Compute output channel indices
    c_out_idx = pid_c_out * BLOCK_SIZE_C_OUT + tl.arange(0, BLOCK_SIZE_C_OUT)[None, :, None]
    c_out_mask = c_out_idx < C_out
    
    # Compute output spatial indices
    h_idx = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)[None, None, :]
    w_idx = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)[None, None, :]
    
    # Initialize accumulator for convolution
    output = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C_OUT, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for c_in_block_start in range(0, C_in, BLOCK_SIZE_C_IN):
        c_in_block = c_in_block_start + tl.arange(0, BLOCK_SIZE_C_IN)[None, :, None, None]
        c_in_mask = c_in_block < C_in
        
        # Load input patch: [BLOCK_SIZE_N, BLOCK_SIZE_C_IN, BLOCK_SIZE_H, BLOCK_SIZE_W]
        # For each output position, we need to gather the corresponding input region
        # Calculate the starting input positions for each output position
        h_start = h_idx * stride_h - pad_h + c_in_block * 0  # Dummy for broadcasting
        w_start = w_idx * stride_w - pad_w + c_in_block * 0
        
        # Loop over kernel height
        for kh in range(K_h):
            h_input = h_start + kh * dilation_h
            h_input_mask = (h_input >= 0) & (h_input < H)
            
            # Loop over kernel width
            for kw in range(K_w):
                w_input = w_start + kw * dilation_w
                w_input_mask = (w_input >= 0) & (w_input < W)
                
                # Combined mask
                input_mask = batch_mask & c_in_mask & h_input_mask & w_input_mask
                
                # Compute input indices
                input_indices = (
                    batch_idx * (C_in * H * W) +
                    c_in_block * (H * W) +
                    h_input * W +
                    w_input
                )
                
                # Load input values
                x_block = tl.load(
                    x_ptr + input_indices,
                    mask=input_mask,
                    other=0.0
                )
                
                # Load corresponding weight values
                weight_indices = (
                    c_out_idx[:, :, None] * (C_in * K_h * K_w) +
                    c_in_block[None, :, :, None] * (K_h * K_w) +
                    kh * K_w +
                    kw
                )
                
                w_block = tl.load(
                    w_ptr + weight_indices,
                    mask=c_out_mask[:, :, None] & c_in_mask[None, :, :, None],
                    other=0.0
                )
                
                # Compute contribution to output
                # x_block shape: [BLOCK_SIZE_N, BLOCK_SIZE_C_IN, BLOCK_SIZE_H, BLOCK_SIZE_W]
                # w_block shape: [BLOCK_SIZE_C_OUT, BLOCK_SIZE_C_IN, 1, 1]
                # We need to broadcast and multiply
                x_expanded = x_block[None, :, :, :, :]  # [1, BLOCK_N, C_IN, BLOCK_H, BLOCK_W]
                w_expanded = w_block[:, :, None, None, None]  # [C_OUT, C_IN, 1, 1, 1]
                
                # Compute product and accumulate
                product = x_expanded * w_expanded  # [C_OUT, BLOCK_N, C_IN, BLOCK_H, BLOCK_W]
                output += tl.sum(product, axis=2)  # Sum over C_IN: [C_OUT, BLOCK_N, BLOCK_H, BLOCK_W]
    
    # Transpose output to desired shape: [BLOCK_N, BLOCK_C_OUT, BLOCK_H, BLOCK_W]
    output = output.transpose(0, 1)  # [BLOCK_C_OUT, BLOCK_N, BLOCK_H, BLOCK_W]
    output = output.transpose(1, 2)  # [BLOCK_C_OUT, BLOCK_H, BLOCK_N, BLOCK_W]
    output = output.transpose(2, 3)  # [BLOCK_C_OUT, BLOCK_H, BLOCK_W, BLOCK_N]
    output = output.transpose(0, 3)  # [BLOCK_N, BLOCK_H, BLOCK_W, BLOCK_C_OUT]
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + c_out_idx, mask=c_out_mask, other=0.0)
        output += bias[None, None, None, :]  # Broadcast to [BLOCK_N, BLOCK_H, BLOCK_W, BLOCK_C_OUT]
    
    # Transpose back to original layout
    output = output.transpose(0, 3)  # [BLOCK_C_OUT, BLOCK_H, BLOCK_W, BLOCK_N]
    output = output.transpose(2, 3)  # [BLOCK_C_OUT, BLOCK_H, BLOCK_N, BLOCK_W]
    output = output.transpose(1, 2)  # [BLOCK_C_OUT, BLOCK_N, BLOCK_H, BLOCK_W]
    output = output.transpose(0, 1)  # [BLOCK_N, BLOCK_C_OUT, BLOCK_H, BLOCK_W]
    
    # Compute output indices for storing
    out_indices = (
        batch_idx * (C_out * H_out * W_out) +
        c_out_idx[:, :, None] * (H_out * W_out) +
        h_idx[None, :, :] * W_out +
        w_idx[None, :, :]
    )
    
    out_mask = batch_mask & c_out_mask[:, :, None] & (h_idx[None, :, :] < H_out) & (w_idx[None, :, :] < W_out)
    
    # Store output
    tl.store(
        out_ptr + out_indices,
        output,
        mask=out_mask
    )


class TritonConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, groups):
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        
        # Extract dimensions
        N, C_in, H, W = x.shape
        C_out, _, K_h, K_w = weight.shape
        stride_h, stride_w = stride
        pad_h, pad_w = padding
        dilation_h, dilation_w = dilation
        
        # Compute output dimensions
        H_out = (H + 2 * pad_h - dilation_h * (K_h - 1) - 1) // stride_h + 1
        W_out = (W + 2 * pad_w - dilation_w * (K_w - 1) - 1) // stride_w + 1
        
        # Create output tensor
        out = torch.empty(N, C_out, H_out, W_out, dtype=x.dtype, device=x.device)
        
        # Define block sizes (tunable parameters for performance)
        BLOCK_SIZE_N = min(4, N)
        BLOCK_SIZE_C_OUT = 8
        BLOCK_SIZE_C_IN = 8
        BLOCK_SIZE_H = 8
        BLOCK_SIZE_W = 8
        
        # Define grid dimensions
        grid = (
            (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
            (C_out + BLOCK_SIZE_C_OUT - 1) // BLOCK_SIZE_C_OUT,
            (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
            (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
        )
        
        # Launch kernel
        conv2d_kernel[grid](
            x, weight, bias, out,
            N, C_in, H, W,
            C_out,
            K_h, K_w,
            stride_h, stride_w,
            pad_h, pad_w,
            dilation_h, dilation_w,
            H_out, W_out,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
            BLOCK_SIZE_C_IN=BLOCK_SIZE_C_IN,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_W=BLOCK_SIZE_W
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.input_size = (N, C_in, H, W)
        ctx.output_size = (N, C_out, H_out, W_out)
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # Standard PyTorch backward for convolution
        x, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        
        # Use PyTorch's built-in backward for simplicity
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(
                x.shape, weight, grad_output, stride, padding, 
                dilation, groups
            )
        
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(
                x, weight.shape, grad_output, stride, padding,
                dilation, groups
            )
        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum([0, 2, 3])
        
        return grad_input, grad_weight, grad_bias, None, None, None, None


class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with asymmetric input and kernel sizes.
    Optimized with Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Store convolution parameters
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return TritonConv2dFunction.apply(x, self.weight, self.bias, 
                                         self.stride, self.padding, 
                                         self.dilation, self.groups)


# Import math for kaiming initialization
import math