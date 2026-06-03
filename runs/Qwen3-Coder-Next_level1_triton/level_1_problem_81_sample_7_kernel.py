import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    # Pointers to inputs and outputs
    x_ptr,  # Input: (B, C_in, H_in, W_in)
    w_ptr,  # Weight: (C_in, C_out, K_h, K_w)
    b_ptr,  # Bias: (C_out,) or None
    out_ptr,  # Output: (B, C_out, H_out, W_out)
    # Dimensions
    batch_size, in_channels, out_channels,
    height_in, width_in,
    kernel_size,
    stride, padding, dilation,
    height_out, width_out,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr,  # Block size for input channels
    BLOCK_SIZE_H: tl.constexpr,  # Block size for output height
    BLOCK_SIZE_W: tl.constexpr,  # Block size for output width
):
    # Get program IDs
    pid_m = tl.program_id(0)  # For output channels
    pid_n = tl.program_id(1)  # For batch
    pid_h = tl.program_id(2)  # For output height
    pid_w = tl.program_id(3)  # For output width
    
    # Calculate output position
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create mask for valid output positions
    mask_h = out_h < height_out
    mask_w = out_w < width_out
    mask_hw = mask_h[:, None] & mask_w[None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Iterate over input channels
    for k in range(0, in_channels, BLOCK_SIZE_K):
        k_range = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_range < in_channels
        
        # Load input block: (B, C_in, H_in, W_in) -> (H_in, W_in)
        # Calculate input position from output position
        # For transposed conv: in_h = (out_h - stride + (kernel_size-1)*dilation + padding) // stride + 1
        # More precisely: out_h = (in_h - 1) * stride - 2*padding + dilation*(kernel_size-1) + 1
        # So: in_h = ((out_h - 1) - dilation*(kernel_size-1) + 2*padding) // stride + 1
        
        # Calculate the range of input positions that contribute to this output position
        # For each output position (out_h, out_w), it receives contributions from input positions
        # where: out_h = in_h * stride + k_h * dilation - padding (mod stride)
        # or equivalently: in_h = (out_h + padding - k_h * dilation) // stride
        
        # We'll compute contributions from all kernel positions
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Calculate corresponding input position
                in_h = (out_h[:, None] + padding - kh * dilation) // stride
                in_w = (out_w[None, :] + padding - kw * dilation) // stride
                
                # Check if this input position is valid
                valid_h = (in_h >= 0) & (in_h < height_in)
                valid_w = (in_w >= 0) & (in_w < width_in)
                valid = valid_h & valid_w
                
                # Check if kernel position corresponds to current output
                # The kernel position (kh, kw) contributes to output (out_h, out_w) from input (in_h, in_w)
                # where: out_h = in_h * stride + kh * dilation - padding
                # So we need: out_h == in_h * stride + kh * dilation - padding
                
                # Calculate the actual output position this kernel position contributes to
                expected_out_h = in_h * stride + kh * dilation - padding
                expected_out_w = in_w * stride + kw * dilation - padding
                
                # Only use contributions where the kernel position actually maps to our output
                correct_map_h = (expected_out_h == out_h[:, None])
                correct_map_w = (expected_out_w == out_w[None, :])
                correct_map = correct_map_h & correct_map_w
                
                # Combine masks
                total_mask = valid & correct_map
                
                # Load input values: shape (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_K)
                # We need to handle the indexing properly
                if k == 0 and kh == 0 and kw == 0:
                    # For the first iteration, initialize the accumulator properly
                    pass
                
                # Get input indices for the current kernel position
                # in_h and in_w are already calculated
                
                # Reshape for proper indexing
                in_h_flat = in_h[total_mask]
                in_w_flat = in_w[total_mask]
                valid_indices = tl.where(total_mask)
                
                # Load input values where valid
                if in_h_flat.numel() > 0:
                    # This is complex - we need to load specific elements
                    # Simplified approach: iterate over input positions that contribute
                    
                    # Instead, let's use a different approach: for each output position, 
                    # accumulate contributions from all input positions and kernel positions
                    
                    pass  # We'll implement a cleaner version below
    
    # Actually, let's implement a cleaner version of the transposed convolution kernel
    # The key insight is that for transposed conv, each output position receives contributions from
    # input positions based on the stride and dilation
    
    # Reset accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # For each output position, accumulate contributions
    # Output position: (pid_n, pid_m, out_h, out_w)
    # It receives contributions from input positions where:
    # out_h = in_h * stride + kh * dilation - padding
    # out_w = in_w * stride + kw * dilation - padding
    
    # So: in_h = (out_h + padding - kh * dilation) / stride
    #    in_w = (out_w + padding - kw * dilation) / stride
    
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            # Calculate input positions that contribute to this output position
            in_h_start = (out_h[:, None] + padding - kh * dilation) // stride
            in_w_start = (out_w[None, :] + padding - kw * dilation) // stride
            
            # Check which ones are valid
            valid_h = (in_h_start >= 0) & (in_h_start < height_in)
            valid_w = (in_w_start >= 0) & (in_w_start < width_in)
            valid = valid_h & valid_w
            
            # Check if the division is exact (no remainder)
            remainder_h = (out_h[:, None] + padding - kh * dilation) % stride
            remainder_w = (out_w[None, :] + padding - kw * dilation) % stride
            exact = (remainder_h == 0) & (remainder_w == 0)
            
            total_valid = valid & exact
            
            # Now we need to load input values at these positions
            # and weight values at (k_in, pid_m, kh, kw)
            
            # Get input positions
            in_h_pos = in_h_start[total_valid]
            in_w_pos = in_w_start[total_valid]
            
            # Iterate over input channels
            for k_in in range(in_channels):
                # Load weight: w[k_in, pid_m, kh, kw]
                w_offset = ((k_in * out_channels + pid_m) * kernel_size * kernel_size + 
                           kh * kernel_size + kw)
                w_ptr_offset = w_ptr + w_offset
                weight_val = tl.load(w_ptr_offset)
                
                # Load input: x[pid_n, k_in, in_h_pos, in_w_pos]
                if in_h_pos.numel() > 0:
                    # Calculate input offset
                    in_batch_offset = pid_n * in_channels * height_in * width_in
                    in_channel_offset = k_in * height_in * width_in
                    in_offset = in_batch_offset + in_channel_offset + in_h_pos * width_in + in_w_pos
                    
                    input_vals = tl.load(x_ptr + in_offset, mask=total_valid[total_valid])
                    
                    # Accumulate
                    # We need to scatter back to the accumulator
                    # This is tricky in Triton - let's use a different approach
                    
                    pass  # Placeholder
    
    # Given the complexity, let me provide a more practical implementation
    # that handles the transposed convolution properly
    
    # Actually, I'll implement a simpler but correct version
    # that processes one output position per program
    
    # Get the specific output position this program handles
    if pid_n >= batch_size:
        return
    if pid_m >= out_channels:
        return
    if pid_h * BLOCK_SIZE_H >= height_out:
        return
    if pid_w * BLOCK_SIZE_W >= width_out:
        return
    
    # Process this output position
    out_h_idx = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w_idx = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create meshgrid for output positions
    out_h_grid, out_w_grid = tl.meshgrid(out_h_idx, out_w_idx, indexing='ij')
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Accumulate over input channels and kernel positions
    for k_in in range(in_channels):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Calculate corresponding input position
                in_h = (out_h_grid + padding - kh * dilation) // stride
                in_w = (out_w_grid + padding - kw * dilation) // stride
                
                # Check if this is a valid contribution
                valid_h = (in_h >= 0) & (in_h < height_in)
                valid_w = (in_w >= 0) & (in_w < width_in)
                valid = valid_h & valid_w
                
                # Check exact division
                rem_h = (out_h_grid + padding - kh * dilation) % stride
                rem_w = (out_w_grid + padding - kw * dilation) % stride
                exact = (rem_h == 0) & (rem_w == 0)
                
                total_valid = valid & exact
                
                # Get input values
                if tl.sum(total_valid) > 0:
                    # Calculate input pointer offset
                    batch_offset = pid_n * in_channels * height_in * width_in
                    channel_offset = k_in * height_in * width_in
                    input_offsets = batch_offset + channel_offset + in_h * width_in + in_w
                    
                    # Load input values
                    input_vals = tl.load(x_ptr + input_offsets, mask=total_valid, other=0.0)
                    
                    # Get weight value
                    weight_offset = ((k_in * out_channels + pid_m) * kernel_size * kernel_size + 
                                    kh * kernel_size + kw)
                    weight_val = tl.load(w_ptr + weight_offset)
                    
                    # Accumulate
                    acc += tl.where(total_valid, input_vals * weight_val, 0.0)
    
    # Add bias if available
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_m)
        acc += bias
    
    # Store result
    out_batch_offset = pid_n * out_channels * height_out * width_out
    out_channel_offset = pid_m * height_out * width_out
    out_offsets = out_batch_offset + out_channel_offset + out_h_grid * width_out + out_w_grid
    
    tl.store(out_ptr + out_offsets, acc.to(x_ptr.dtype.element_ty), mask=total_valid)


class TritonConvTranspose2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation):
        # Get dimensions
        batch_size, in_channels, height_in, width_in = x.shape
        _, out_channels, kernel_size_h, kernel_size_w = weight.shape
        
        # Calculate output dimensions
        # For transposed convolution: 
        # H_out = (H_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
        # Since output_padding=0 in our case:
        height_out = (height_in - 1) * stride - 2 * padding + dilation * (kernel_size_h - 1) + 1
        width_out = (width_in - 1) * stride - 2 * padding + dilation * (kernel_size_w - 1) + 1
        
        # Create output tensor
        output = torch.empty(batch_size, out_channels, height_out, width_out, 
                            dtype=x.dtype, device=x.device)
        
        # Configure kernel launch
        BLOCK_SIZE_M = 8
        BLOCK_SIZE_N = 4
        BLOCK_SIZE_H = 8
        BLOCK_SIZE_W = 8
        
        grid_m = (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        grid_n = (batch_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        grid_h = (height_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
        grid_w = (width_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
        
        grid = (grid_m, grid_n, grid_h, grid_w)
        
        # Launch kernel
        conv_transpose2d_kernel[grid](
            x, weight, bias, output,
            batch_size, in_channels, out_channels,
            height_in, width_in,
            kernel_size_h,  # Using kernel_size_h for both dimensions
            stride, padding, dilation,
            height_out, width_out,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_W=BLOCK_SIZE_W
        )
        
        ctx.save_for_backward(x, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.input_size = (height_in, width_in)
        ctx.output_size = (height_out, width_out)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # For simplicity, we'll fall back to PyTorch's backward pass
        x, weight = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        
        # Use PyTorch's native backward for gradients
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(
                ctx.input_size + (x.shape[2], x.shape[3]), 
                weight, grad_output, stride=stride, 
                padding=padding, dilation=dilation, groups=1
            )
        
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(
                x, weight.shape, grad_output, stride=stride,
                padding=padding, dilation=dilation, groups=1
            )
        
        if ctx.needs_input_grad[2] and ctx.needs_input_grad[2] is not None:
            grad_bias = grad_output.sum([0, 2, 3])
        
        return grad_input, grad_weight, grad_bias, None, None, None


def triton_conv_transpose2d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """Wrapper function for the Triton-based transposed convolution."""
    return TritonConvTranspose2d.apply(x, weight, bias, stride, padding, dilation)


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for 2D transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Register buffers for weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Store convolution parameters
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(x, self.weight, self.bias, 
                                      self.stride, self.padding, self.dilation)


import math