import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,                # Input tensor pointer (N, C_in, H, W)
    w_ptr,                # Weight tensor pointer (C_out, C_in, K_h, K_w)
    b_ptr,                # Bias tensor pointer (C_out,)
    y_ptr,                # Output tensor pointer (N, C_out, H_out, W_out)
    n_elements,           # Total number of output elements
    # Tensor dimensions
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    in_height: tl.constexpr,
    in_width: tl.constexpr,
    out_height: tl.constexpr,
    out_width: tl.constexpr,
    kernel_height: tl.constexpr,
    kernel_width: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dil_h: tl.constexpr,
    dil_w: tl.constexpr,
    # Block sizes for tiling
    BLOCK_SIZE_N: tl.constexpr,  # Batch dimension block
    BLOCK_SIZE_COUT: tl.constexpr,  # Output channel block
    BLOCK_SIZE_CIN: tl.constexpr,  # Input channel block
    BLOCK_SIZE_H: tl.constexpr,  # Output height block
    BLOCK_SIZE_W: tl.constexpr,  # Output width block
    BLOCK_KH: tl.constexpr,  # Kernel height block
    BLOCK_KW: tl.constexpr,  # Kernel width block
):
    # Program IDs
    pid_n = tl.program_id(0)
    pid_cout = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Compute base offsets
    n_offset = pid_n * BLOCK_SIZE_N
    cout_offset = pid_cout * BLOCK_SIZE_COUT
    h_offset = pid_h * BLOCK_SIZE_H
    w_offset = pid_w * BLOCK_SIZE_W
    
    # Compute output coordinates
    h_idx = h_offset + tl.arange(0, BLOCK_SIZE_H)
    w_idx = w_offset + tl.arange(0, BLOCK_SIZE_W)
    cout_idx = cout_offset + tl.arange(0, BLOCK_SIZE_COUT)
    
    # Create masks for valid indices
    h_mask = h_idx < out_height
    w_mask = w_idx < out_width
    cout_mask = cout_idx < out_channels
    
    # Compute input coordinates for the top-left corner of the first output position
    h_in_start = h_idx * stride_h - pad_h
    w_in_start = w_idx * stride_w - pad_w
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_COUT), dtype=tl.float32)
    
    # Iterate over input channels
    for cin_start in range(0, in_channels, BLOCK_SIZE_CIN):
        cin_idx = cin_start + tl.arange(0, BLOCK_SIZE_CIN)
        cin_mask = cin_idx < in_channels
        
        # Iterate over kernel height
        for kh_start in range(0, kernel_height, BLOCK_KH):
            kh_idx = kh_start + tl.arange(0, BLOCK_KH)
            kh_mask = kh_idx < kernel_height
            
            # Iterate over kernel width
            for kw_start in range(0, kernel_width, BLOCK_KW):
                kw_idx = kw_start + tl.arange(0, BLOCK_KW)
                kw_mask = kw_idx < kernel_width
                
                # Compute input coordinates with dilation
                h_in = h_in_start[:, None, None] + kh_idx[None, :, None] * dil_h
                w_in = w_in_start[None, :, None] + kw_idx[None, None, :] * dil_w
                
                # Create input mask
                h_in_mask = (h_in >= 0) & (h_in < in_height)
                w_in_mask = (w_in >= 0) & (w_in < in_width)
                input_mask = h_in_mask & w_in_mask
                
                # Load input data with padding handling
                # We need to handle the case where input goes out of bounds
                h_in_clamped = tl.maximum(tl.minimum(h_in, in_height - 1), 0)
                w_in_clamped = tl.maximum(tl.minimum(w_in, in_width - 1), 0)
                
                # Compute linear offsets for input tensor
                input_offsets = (
                    n_offset * (in_channels * in_height * in_width) +
                    cin_idx[None, None, :, None, None] * (in_height * in_width) +
                    h_in_clamped[:, None, None, :, None] * in_width +
                    w_in_clamped[:, None, None, None, :]
                )
                
                # Reshape for broadcasting
                input_offsets = input_offsets.reshape(BLOCK_SIZE_H, BLOCK_KH, BLOCK_KW, BLOCK_SIZE_CIN)
                input_mask_reshaped = input_mask[:, None, None, :, None] & input_mask[None, :, :, None, :]
                
                # Load input values
                input_vals = tl.load(
                    x_ptr + input_offsets,
                    mask=input_mask_reshaped,
                    other=0.0
                )
                
                # Load weight data
                weight_offsets = (
                    cout_idx[None, None, None, :, None, None] * (in_channels * kernel_height * kernel_width) +
                    cin_idx[None, None, None, None, :, None] * (kernel_height * kernel_width) +
                    kh_idx[None, None, None, None, None, :] * kernel_width +
                    kw_idx[None, None, None, None, None, :]
                )
                weight_offsets = weight_offsets.reshape(BLOCK_SIZE_COUT, BLOCK_SIZE_CIN, BLOCK_KH, BLOCK_KW)
                
                # Load weights
                weights = tl.load(
                    w_ptr + weight_offsets,
                    mask=cout_mask[:, None, None, None] & cin_mask[None, :, None, None] & 
                         kh_mask[None, None, :, None] & kw_mask[None, None, None, :],
                    other=0.0
                )
                
                # Compute accumulation: acc += input * weight
                # Reshape for broadcasting: (H, KH, KW, Cin) * (Cout, Cin, KH, KW) -> (H, W, Cout)
                # Use einsum-like computation
                for kh in range(BLOCK_KH):
                    for kw in range(BLOCK_KW):
                        if kh_start + kh < kernel_height and kw_start + kw < kernel_width:
                            # Extract slices for this kernel position
                            input_slice = input_vals[:, kh, kw, :]  # (BLOCK_SIZE_H, BLOCK_SIZE_CIN)
                            weight_slice = weights[:, :, kh, kw]    # (BLOCK_SIZE_COUT, BLOCK_SIZE_CIN)
                            
                            # Compute outer product accumulation
                            # acc[h, w, cout] += sum_cin input[h, cin] * weight[cout, cin]
                            acc += tl.dot(
                                input_slice, 
                                tl.trans(weight_slice),
                                allow_tf32=True
                            )
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + cout_idx, mask=cout_mask, other=0.0)
        acc += bias[None, None, :]
    
    # Store result
    y_offsets = (
        n_offset * (out_channels * out_height * out_width) +
        cout_idx[None, None, :] * (out_height * out_width) +
        h_idx[:, None, None] * out_width +
        w_idx[None, :, None]
    )
    
    # Reshape acc to match output shape
    acc_reshaped = acc.reshape(BLOCK_SIZE_H * BLOCK_SIZE_W, BLOCK_SIZE_COUT)
    
    # Create output mask
    y_mask = h_mask[:, None, None] & w_mask[None, :, None] & cout_mask[None, None, :]
    y_mask_reshaped = y_mask.reshape(BLOCK_SIZE_H * BLOCK_SIZE_W, BLOCK_SIZE_COUT)
    
    tl.store(
        y_ptr + y_offsets.reshape(BLOCK_SIZE_H * BLOCK_SIZE_W, BLOCK_SIZE_COUT),
        acc_reshaped,
        mask=y_mask_reshaped
    )


def triton_conv2d(x, weight, bias, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """
    Triton-based 2D convolution implementation.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, height, width)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_height, kernel_width)
        bias: Optional bias tensor of shape (out_channels,)
        stride, padding, dilation, groups: Convolution parameters
    
    Returns:
        Output tensor of shape (batch_size, out_channels, out_height, out_width)
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    
    # Calculate output dimensions
    out_height = (in_height + 2 * pad_h - dil_h * (kernel_height - 1) - 1) // stride_h + 1
    out_width = (in_width + 2 * pad_w - dil_w * (kernel_width - 1) - 1) // stride_w + 1
    
    # Prepare output tensor
    y = torch.empty(batch_size, out_channels, out_height, out_width, dtype=x.dtype, device=x.device)
    
    # Set up block sizes for tiling (tuned for performance)
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_COUT = 16
    BLOCK_SIZE_CIN = 16
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    BLOCK_KH = 3
    BLOCK_KW = 3
    
    # Compute grid dimensions
    grid_n = (batch_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_cout = (out_channels + BLOCK_SIZE_COUT - 1) // BLOCK_SIZE_COUT
    grid_h = (out_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (out_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    
    # Launch kernel
    conv2d_kernel[grid_n, grid_cout, grid_h, grid_w](
        x, weight, bias, y,
        batch_size * out_channels * out_height * out_width,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        in_height=in_height,
        in_width=in_width,
        out_height=out_height,
        out_width=out_width,
        kernel_height=kernel_height,
        kernel_width=kernel_width,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dil_h=dil_h,
        dil_w=dil_w,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_COUT=BLOCK_SIZE_COUT,
        BLOCK_SIZE_CIN=BLOCK_SIZE_CIN,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_KH=BLOCK_KH,
        BLOCK_KW=BLOCK_KW,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized 2D convolution using Triton kernels.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of two integers representing the height and width of the convolution kernel.
        stride (tuple, optional): Tuple of two integers representing the stride in the height and width dimensions. Defaults to (1, 1).
        padding (tuple, optional): Tuple of two integers representing the padding in the height and width dimensions. Defaults to (0, 0).
        dilation (tuple, optional): Tuple of two integers representing the dilation in the height and width dimensions. Defaults to (1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), 
                 padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Register weights as parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Store convolution parameters
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        import math
        return triton_conv2d(x, self.weight, self.bias, 
                           stride=self.stride, 
                           padding=self.padding, 
                           dilation=self.dilation, 
                           groups=self.groups)