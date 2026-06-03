import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor pointer
    w_ptr,  # Weight tensor pointer
    b_ptr,  # Bias tensor pointer (can be None)
    out_ptr,  # Output tensor pointer
    batch_size,  # Batch size
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    in_h,  # Input height
    in_w,  # Input width
    out_h,  # Output height
    out_w,  # Output width
    kernel_h,  # Kernel height
    kernel_w,  # Kernel width
    stride_h,  # Stride height
    stride_w,  # Stride width
    pad_h,  # Padding height
    pad_w,  # Padding width
    dil_h,  # Dilation height
    dil_w,  # Dilation width
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output height
    BLOCK_SIZE_N: tl.constexpr,  # Block size for output width
    BLOCK_SIZE_K: tl.constexpr,  # Block size for channels
):
    # Program IDs for output tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Calculate output position
    out_row_start = pid_m * BLOCK_SIZE_M
    out_col_start = pid_n * BLOCK_SIZE_N
    
    # Check bounds
    out_row_offsets = out_row_start + tl.arange(0, BLOCK_SIZE_M)
    out_col_offsets = out_col_start + tl.arange(0, BLOCK_SIZE_N)
    
    out_row_mask = out_row_offsets < out_h
    out_col_mask = out_col_offsets < out_w
    
    # Create 2D mask for output
    out_mask = out_row_mask[:, None] & out_col_mask[None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Convolution loop over input channels and kernel positions
    for ic in range(in_channels):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate input position
                in_row = out_row_start * stride_h + kh * dil_h - pad_h
                in_col = out_col_start * stride_w + kw * dil_w - pad_w
                
                # Input offsets
                in_row_offsets = in_row + tl.arange(0, BLOCK_SIZE_M)
                in_col_offsets = in_col + tl.arange(0, BLOCK_SIZE_N)
                
                # Input mask
                in_row_mask = (in_row_offsets >= 0) & (in_row_offsets < in_h)
                in_col_mask = (in_col_offsets >= 0) & (in_col_offsets < in_w)
                in_mask = in_row_mask[:, None] & in_col_mask[None, :]
                
                # Compute input pointer offset for this batch, channel, row, col
                # x_ptr is (batch, in_channels, in_h, in_w)
                # Offset = batch * (in_channels * in_h * in_w) + ic * (in_h * in_w) + in_row * in_w + in_col
                x_offset = pid_b * (in_channels * in_h * in_w) + ic * (in_h * in_w)
                
                # Load input values with masking
                x_vals = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=x_ptr.dtype.element_ty)
                for i, r in enumerate(in_row_offsets):
                    for j, c in enumerate(in_col_offsets):
                        if in_mask[i, j]:
                            idx = x_offset + r * in_w + c
                            x_vals = tl.store(x_vals, x_vals + tl.load(x_ptr + idx, mask=in_mask[i, j]))
                
                # Load weight value
                # w_ptr is (out_channels, in_channels, kernel_h, kernel_w)
                # Offset for this kernel position
                w_offset = 0  # Will be computed for each output channel
                w_vals = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=w_ptr.dtype.element_ty)
                
                for oc in range(out_channels):
                    w_idx = oc * (in_channels * kernel_h * kernel_w) + ic * (kernel_h * kernel_w) + kh * kernel_w + kw
                    w_val = tl.load(w_ptr + w_idx)
                    
                    # Broadcast weight to output tile
                    w_broadcast = tl.broadcast_to(w_val, (BLOCK_SIZE_M, BLOCK_SIZE_N))
                    
                    # Accumulate
                    acc += x_vals * w_broadcast
                
                # Add bias if available
                if b_ptr is not None:
                    for oc in range(out_channels):
                        b_val = tl.load(b_ptr + oc)
                        b_broadcast = tl.broadcast_to(b_val, (BLOCK_SIZE_M, BLOCK_SIZE_N))
                        acc += x_vals * b_broadcast
    
    # Store results
    # out_ptr is (batch, out_channels, out_h, out_w)
    out_offset = pid_b * (out_channels * out_h * out_w)
    
    for oc in range(out_channels):
        out_idx = out_offset + oc * (out_h * out_w)
        
        # Compute output position for this channel
        out_row_offsets_oc = out_row_start + tl.arange(0, BLOCK_SIZE_M)
        out_col_offsets_oc = out_col_start + tl.arange(0, BLOCK_SIZE_N)
        
        out_row_mask_oc = out_row_offsets_oc < out_h
        out_col_mask_oc = out_col_offsets_oc < out_w
        out_mask_oc = out_row_mask_oc[:, None] & out_col_mask_oc[None, :]
        
        # Load accumulator for this channel
        out_val = tl.load(acc + oc * BLOCK_SIZE_M * BLOCK_SIZE_N, mask=out_mask_oc)
        
        # Store result
        tl.store(out_ptr + out_idx + out_row_offsets_oc[:, None] * out_w + out_col_offsets_oc[None, :], 
                 out_val, mask=out_mask_oc)


def triton_conv2d(x, weight, bias, stride, padding, dilation, groups):
    """
    Triton implementation of 2D convolution.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, height, width)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_h, kernel_w)
        bias: Bias tensor of shape (out_channels,) or None
        stride: Tuple (stride_h, stride_w)
        padding: Tuple (padding_h, padding_w)
        dilation: Tuple (dilation_h, dilation_w)
        groups: Number of groups (must be 1 for this implementation)
    """
    assert groups == 1, "Triton convolution only supports groups=1 for simplicity"
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    batch_size, in_channels, in_h, in_w = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    
    # Calculate output dimensions
    out_h = (in_h + 2 * pad_h - dil_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (in_w + 2 * pad_w - dil_w * (kernel_w - 1) - 1) // stride_w + 1
    
    # Allocate output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Define block sizes (tunable parameters)
    BLOCK_SIZE_M = 8
    BLOCK_SIZE_N = 8
    BLOCK_SIZE_K = 16  # Not used directly in this implementation but kept for extensibility
    
    # Define grid
    grid = lambda meta: (
        triton.cdiv(out_h, BLOCK_SIZE_M),
        triton.cdiv(out_w, BLOCK_SIZE_N),
        batch_size
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        in_h, in_w, out_h, out_w,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model with Triton-based convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), 
                 padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
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
        
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 2D convolution using Triton kernel.
        """
        return triton_conv2d(x, self.weight, self.bias, self.stride, self.padding, 
                            self.dilation, self.groups)


import math