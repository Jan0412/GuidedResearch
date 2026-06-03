import torch
import torch.nn as nn
import torch.nn.functional as F
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
    stride: tl.constexpr,
    padding: tl.constexpr,
    output_padding: tl.constexpr,
    groups: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Compute batch index
    batch_idx = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    batch_mask = batch_idx < B
    
    # Compute output channel indices
    c_out_idx = pid_c_out * BLOCK_SIZE_C_OUT + tl.arange(0, BLOCK_SIZE_C_OUT)
    c_out_mask = c_out_idx < C_out
    
    # Compute output spatial indices
    h_idx = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w_idx = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    h_mask = h_idx < H_out
    w_mask = w_idx < W_out
    
    # Create meshgrid for output positions
    h_grid, w_grid = tl.meshgrid(h_idx, w_idx)  # (BLOCK_SIZE_H, BLOCK_SIZE_W)
    h_grid = tl.reshape(h_grid, [BLOCK_SIZE_H * BLOCK_SIZE_W])
    w_grid = tl.reshape(w_grid, [BLOCK_SIZE_H * BLOCK_SIZE_W])
    
    # Output tensor strides
    out_stride_b = tl.load(out_ptr + 0)  # Will be computed in host code
    out_stride_c = tl.load(out_ptr + 1)
    out_stride_h = tl.load(out_ptr + 2)
    out_stride_w = tl.load(out_ptr + 3)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE_B, BLOCK_SIZE_C_OUT, BLOCK_SIZE_H, BLOCK_SIZE_W], dtype=tl.float32)
    
    # Compute transposed convolution
    # For transposed conv: output[b, c_out, h_out, w_out] = 
    #   sum_{c_in, k_h, k_w} x[b, c_in, h_out - k_h*stride + padding, w_out - k_w*stride + padding] * w[c_in, c_out, k_h, k_w]
    
    # Loop over input channels
    for c_in_start in range(0, C_in, BLOCK_SIZE_C_OUT):
        c_in_idx = c_in_start + tl.arange(0, BLOCK_SIZE_C_OUT)
        c_in_mask = c_in_idx < C_in
        
        # Compute corresponding input positions for each output position
        # h_in = h_out - k_h * stride + padding
        # w_in = w_out - k_w * stride + padding
        
        # Loop over kernel height
        for k_h_start in range(0, K_h, BLOCK_SIZE_K):
            k_h_idx = k_h_start + tl.arange(0, BLOCK_SIZE_K)
            k_h_mask = k_h_idx < K_h
            
            # Loop over kernel width
            for k_w_start in range(0, K_w, BLOCK_SIZE_K):
                k_w_idx = k_w_start + tl.arange(0, BLOCK_SIZE_K)
                k_w_mask = k_w_idx < K_w
                
                # Compute input positions
                h_in = h_grid[:, None, None] - k_h_idx[None, :, None] * stride + padding
                w_in = w_grid[:, None, None] - k_w_idx[None, None, :] * stride + padding
                
                # Mask for valid input positions
                h_in_mask = (h_in >= 0) & (h_in < H_in)
                w_in_mask = (w_in >= 0) & (w_in < W_in)
                valid_mask = h_in_mask & w_in_mask
                
                # Load input values: shape [BLOCK_SIZE_H*BLOCK_SIZE_W, BLOCK_SIZE_K, BLOCK_SIZE_K]
                x_offset = (batch_idx[:, None, None] * out_stride_b + 
                           c_in_idx[None, :, None] * out_stride_c + 
                           h_in * out_stride_h + 
                           w_in * out_stride_w)
                
                # Reshape for loading
                x_vals = tl.load(x_ptr + x_offset, mask=valid_mask & c_in_mask[None, :, None], other=0.0)
                
                # Load weight values: shape [BLOCK_SIZE_K, BLOCK_SIZE_K, BLOCK_SIZE_C_OUT]
                w_offset = (c_in_idx[None, :, None] * w_ptr + 
                           c_out_idx[None, None, :] * w_ptr + 
                           k_h_idx[:, None, None] * w_ptr + 
                           k_w_idx[None, :, None] * w_ptr)
                
                # Reshape for loading
                w_vals = tl.load(w_ptr + w_offset, mask=c_in_mask[None, :, None] & c_out_mask[None, None, :], other=0.0)
                
                # Compute outer product and accumulate
                # x_vals: [BLOCK_SIZE_H*BLOCK_SIZE_W, BLOCK_SIZE_K, BLOCK_SIZE_C_OUT]
                # w_vals: [BLOCK_SIZE_K, BLOCK_SIZE_K, BLOCK_SIZE_C_OUT]
                
                # Transpose for proper convolution computation
                x_reshaped = tl.reshape(x_vals, [BLOCK_SIZE_H * BLOCK_SIZE_W, BLOCK_SIZE_K, BLOCK_SIZE_K, BLOCK_SIZE_C_OUT])
                w_reshaped = tl.reshape(w_vals, [BLOCK_SIZE_K, BLOCK_SIZE_K, BLOCK_SIZE_C_OUT])
                
                # Accumulate: sum over k_h, k_w, c_in
                acc += tl.sum(x_reshaped * w_reshaped[None, :, :, :], axis=[1, 2])
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + c_out_idx, mask=c_out_mask)
        acc += bias[None, :, None, None]
    
    # Store results
    out_offset = (batch_idx[:, None, None] * out_stride_b + 
                 c_out_idx[None, :, None] * out_stride_c + 
                 h_grid[None, None, :] * out_stride_h + 
                 w_grid[None, None, :] * out_stride_w)
    
    out_vals = tl.reshape(acc, [BLOCK_SIZE_B * BLOCK_SIZE_C_OUT * BLOCK_SIZE_H * BLOCK_SIZE_W])
    out_mask = (batch_mask[:, None, None] & 
               c_out_mask[None, :, None] & 
               h_mask[None, None, :] & 
               w_mask[None, None, :])
    out_mask = tl.reshape(out_mask, [BLOCK_SIZE_B * BLOCK_SIZE_C_OUT * BLOCK_SIZE_H * BLOCK_SIZE_W])
    
    tl.store(out_ptr + out_offset, out_vals, mask=out_mask)


def triton_conv_transpose2d(x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
    """
    Perform transposed 2D convolution using Triton kernel.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Get dimensions
    B, C_in, H_in, W_in = x.shape
    _, C_out, K_h, K_w = weight.shape
    
    # Compute output dimensions (same as PyTorch's ConvTranspose2d)
    H_out = (H_in - 1) * stride - 2 * padding + (K_h - 1) + output_padding + 1
    W_out = (W_in - 1) * stride - 2 * padding + (K_w - 1) + output_padding + 1
    
    # Prepare output tensor
    out = torch.empty(B, C_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Compute strides for the kernel
    stride_b = out.stride(0)
    stride_c = out.stride(1)
    stride_h = out.stride(2)
    stride_w = out.stride(3)
    
    # Pass strides to kernel via pointer arithmetic trick
    # We'll embed them in the tensor data temporarily
    stride_tensor = torch.tensor([stride_b, stride_c, stride_h, stride_w], dtype=torch.int64, device=x.device)
    stride_ptr = stride_tensor.data_ptr()
    
    # Grid configuration
    BLOCK_SIZE_B = 1
    BLOCK_SIZE_C_OUT = 8
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    BLOCK_SIZE_K = 3  # kernel_size is typically small
    
    grid = lambda meta: (
        (B + meta['BLOCK_SIZE_B'] - 1) // meta['BLOCK_SIZE_B'],
        (C_out + meta['BLOCK_SIZE_C_OUT'] - 1) // meta['BLOCK_SIZE_C_OUT'],
        (H_out + meta['BLOCK_SIZE_H'] - 1) // meta['BLOCK_SIZE_H'],
        (W_out + meta['BLOCK_SIZE_W'] - 1) // meta['BLOCK_SIZE_W']
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias, out,
        B, C_in, C_out,
        H_in, W_in,
        H_out, W_out,
        K_h, K_w,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for transposed 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias
        
        # Create the weight and bias parameters manually
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size))
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