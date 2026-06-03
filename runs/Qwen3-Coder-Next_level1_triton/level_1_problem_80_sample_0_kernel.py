import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def conv2d_kernel(
    # Pointers to inputs and outputs
    x_ptr,  # (B, C_in, H, W)
    w_ptr,  # (C_out, C_in, K_h, K_w)
    b_ptr,  # (C_out,) or None
    out_ptr,  # (B, C_out, H_out, W_out)
    # Dimensions
    B, C_in, H, W,
    C_out, K_h, K_w,
    stride_h, stride_w,
    pad_h_top, pad_h_bottom, pad_w_left, pad_w_right,
    dil_h, dil_w,
    H_out, W_out,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch size
    BLOCK_SIZE_K: tl.constexpr,  # Block size for input channels
    BLOCK_SIZE_P: tl.constexpr,  # Block size for output height
    BLOCK_SIZE_Q: tl.constexpr,  # Block size for output width
):
    # Program IDs
    pid_m = tl.program_id(0)  # For output channels
    pid_n = tl.program_id(1)  # For batch
    pid_p = tl.program_id(2)  # For output height
    pid_q = tl.program_id(3)  # For output width

    # Compute start indices for the block
    out_channel_start = pid_m * BLOCK_SIZE_M
    batch_start = pid_n * BLOCK_SIZE_N
    out_h_start = pid_p * BLOCK_SIZE_P
    out_w_start = pid_q * BLOCK_SIZE_Q

    # Create output offsets for this block
    out_offsets_h = tl.arange(0, BLOCK_SIZE_P)
    out_offsets_w = tl.arange(0, BLOCK_SIZE_Q)
    out_offsets_h, out_offsets_w = tl.meshgrid(out_offsets_h, out_offsets_w)
    out_offsets_h = out_offsets_h.T.flatten()
    out_offsets_w = out_offsets_w.T.flatten()

    # Compute input coordinates corresponding to output position
    in_h = out_h_start + out_offsets_h - pad_h_top
    in_w = out_w_start + out_offsets_w - pad_w_left
    in_h = in_h * dil_h
    in_w = in_w * dil_w

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_P * BLOCK_SIZE_Q,), dtype=tl.float32)

    # Loop over input channels
    for k in range(0, C_in, BLOCK_SIZE_K):
        k_start = k
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < C_in

        # Loop over kernel height and width
        for kh in range(K_h):
            in_h_kh = in_h + kh * dil_h
            h_valid = (in_h_kh >= 0) & (in_h_kh < H)
            
            for kw in range(K_w):
                in_w_kw = in_w + kw * dil_w
                w_valid = (in_w_kw >= 0) & (in_w_kw < W)
                
                # Create combined mask
                valid_mask = h_valid & w_valid & k_mask
                
                # Load input values: (BLOCK_SIZE_P * BLOCK_SIZE_Q, BLOCK_SIZE_K)
                # For efficiency, we load the input values as a 2D block
                # First, gather input indices
                h_indices = in_h_kh * W + in_w_kw
                h_indices = h_indices[None, :] * C_in + k_offsets[:, None]
                h_indices = h_indices.T.flatten()  # (BLOCK_SIZE_P * BLOCK_SIZE_Q * BLOCK_SIZE_K)
                
                # Load input
                x_block = tl.load(
                    x_ptr + h_indices,
                    mask=valid_mask[:, None] & k_mask[None, :],
                    other=0.0
                )
                x_block = tl.trans(x_block)  # (BLOCK_SIZE_K, BLOCK_SIZE_P * BLOCK_SIZE_Q)
                
                # Load weights
                w_indices = (out_channel_start + tl.arange(0, BLOCK_SIZE_M)[:, None] * C_in * K_h * K_w +
                            k_offsets[None, :] * K_h * K_w +
                            kh * K_w + kw)
                w_block = tl.load(w_ptr + w_indices, mask=k_mask[None, :] & (tl.arange(0, BLOCK_SIZE_M)[:, None] < C_out), other=0.0)
                
                # Compute dot product
                acc += tl.sum(x_block * w_block, axis=0)

    # Add bias if present
    if b_ptr is not None:
        bias_offsets = out_channel_start + tl.arange(0, BLOCK_SIZE_M)
        bias_mask = bias_offsets < C_out
        bias = tl.load(b_ptr + bias_offsets, mask=bias_mask, other=0.0)
        acc += bias[:, None].flatten()

    # Store result
    out_offsets = (batch_start + tl.arange(0, BLOCK_SIZE_N)[:, None] * C_out * H_out * W_out +
                   out_channel_start[None, :] * H_out * W_out +
                   (out_h_start + out_offsets_h[None, :]) * W_out +
                   (out_w_start + out_offsets_w[None, :]))
    out_offsets = out_offsets.flatten()
    
    acc = acc[None, :].flatten()
    tl.store(out_ptr + out_offsets, acc, mask=(batch_start + tl.arange(0, BLOCK_SIZE_N)[:, None] < B) & 
             (out_channel_start + tl.arange(0, BLOCK_SIZE_M)[None, :] < C_out) &
             (out_h_start + out_offsets_h[None, :] < H_out) &
             (out_w_start + out_offsets_w[None, :] < W_out))


def triton_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: tuple = (1, 1),
    padding: tuple = (0, 0),
    dilation: tuple = (1, 1),
):
    """
    Triton implementation of 2D convolution.
    
    Args:
        x: Input tensor of shape (B, C_in, H, W)
        weight: Weight tensor of shape (C_out, C_in, K_h, K_w)
        bias: Optional bias tensor of shape (C_out,)
        stride: Tuple (stride_h, stride_w)
        padding: Tuple (pad_h_top, pad_w_left) or (pad_h, pad_w) for symmetric padding
        dilation: Tuple (dilation_h, dilation_w)
        
    Returns:
        Output tensor of shape (B, C_out, H_out, W_out)
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C_in, H, W = x.shape
    C_out, _, K_h, K_w = weight.shape
    
    # Parse stride, padding, and dilation
    stride_h, stride_w = stride
    if isinstance(padding, int):
        pad_h_top = pad_h_bottom = pad_w_left = pad_w_right = padding
    elif isinstance(padding, tuple):
        if len(padding) == 2:
            pad_h_top = pad_h_bottom = padding[0]
            pad_w_left = pad_w_right = padding[1]
        elif len(padding) == 4:
            pad_h_top, pad_h_bottom, pad_w_left, pad_w_right = padding
        else:
            raise ValueError("Padding tuple must be of length 2 or 4")
    else:
        raise ValueError("Padding must be an int or tuple")
    
    dil_h, dil_w = dilation
    
    # Calculate output dimensions
    H_out = (H + pad_h_top + pad_h_bottom - dil_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W + pad_w_left + pad_w_right - dil_w * (K_w - 1) - 1) // stride_w + 1
    
    # Allocate output tensor
    out = torch.empty((B, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Define block sizes for the kernel
    BLOCK_SIZE_M = 16  # output channels per block
    BLOCK_SIZE_N = 2   # batch size per block
    BLOCK_SIZE_K = 8   # input channels per block
    BLOCK_SIZE_P = 8   # output height per block
    BLOCK_SIZE_Q = 8   # output width per block
    
    # Calculate grid dimensions
    grid_m = (C_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (B + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_p = (H_out + BLOCK_SIZE_P - 1) // BLOCK_SIZE_P
    grid_q = (W_out + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q
    
    # Launch the kernel
    conv2d_kernel[grid_m, grid_n, grid_p * grid_q](
        x, weight, bias, out,
        B, C_in, H, W,
        C_out, K_h, K_w,
        stride_h, stride_w,
        pad_h_top, pad_h_bottom, pad_w_left, pad_w_right,
        dil_h, dil_w,
        H_out, W_out,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_P=BLOCK_SIZE_P,
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the Model using Triton kernels for 2D convolution.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width). 
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (tuple, optional): Padding applied to the input (top/bottom, left/right). Defaults to (0, 0).
        dilation (tuple, optional): Spacing between kernel elements (height, width). Defaults to (1, 1).
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        # Store original Conv2d for initialization
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Extract parameters from the Conv2d layer
        weight = self.conv2d.weight
        bias = self.conv2d.bias if self.conv2d.bias is not None else None
        
        # Parse parameters
        stride = self.conv2d.stride
        if isinstance(stride, int):
            stride = (stride, stride)
        
        padding = self.conv2d.padding
        if isinstance(padding, int):
            padding = (padding, padding)
        
        dilation = self.conv2d.dilation
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        
        # Call our Triton implementation
        return triton_conv2d(x, weight, bias, stride, padding, dilation)