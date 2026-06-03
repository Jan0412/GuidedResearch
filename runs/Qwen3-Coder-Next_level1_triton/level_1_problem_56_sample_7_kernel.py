import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    # Pointers to tensors
    x_ptr,  # Input tensor: (N, C_in, H, W)
    w_ptr,  # Weight tensor: (C_out, C_in // groups, K_h, K_w)
    b_ptr,  # Bias tensor: (C_out,) or None
    out_ptr,  # Output tensor: (N, C_out, H_out, W_out)
    # Dimensions
    batch_size, in_channels, out_channels, groups,
    in_h, in_w, out_h, out_w,
    k_h, k_w,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    # Pointers to meta-info for groups
    weight_start_ptr,  # Pointer to array storing weight starting indices for each group
    # Meta-parameters
    BLOCK_N: tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
    BLOCK_KH: tl.constexpr,
    BLOCK_KW: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Get program IDs
    pid_n = tl.program_id(0)  # batch index
    pid_c_out = tl.program_id(1)  # output channel index
    pid_h = tl.program_id(2)  # height tile index
    pid_w = tl.program_id(3)  # width tile index
    
    # Get group for this output channel
    group_id = pid_c_out // (out_channels // groups)
    
    # Define ranges for output tile
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_out_offsets = pid_c_out * BLOCK_C_OUT + tl.arange(0, BLOCK_C_OUT)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offsets = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    
    # Create masks for valid indices
    n_mask = n_offsets < batch_size
    c_out_mask = c_out_offsets < out_channels
    h_mask = h_offsets < out_h
    w_mask = w_offsets < out_w
    
    # Initialize accumulator for output
    output = tl.zeros((BLOCK_N, BLOCK_C_OUT, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Loop over input channels in groups
    for c_in_start in range(0, in_channels, BLOCK_C_IN):
        c_in_offsets = c_in_start + tl.arange(0, BLOCK_C_IN)
        c_in_mask = c_in_offsets < in_channels
        
        # Load input tile: x[pid_n, c_in_start:c_in_start+BLOCK_C_IN, h:h+BLOCK_H, w:w+BLOCK_W]
        # But need to account for padding and dilation
        # For each position in output tile, compute corresponding input positions
        
        # Compute the input h and w positions for this output position
        h_in_offsets = pid_h * BLOCK_H * stride_h + tl.arange(0, BLOCK_H)[None, :, None, None] * stride_h - pad_h
        w_in_offsets = pid_w * BLOCK_W * stride_w + tl.arange(0, BLOCK_W)[None, None, :, None] * stride_w - pad_w
        
        # Reshape for broadcasting with kernel dimensions
        for kh_start in range(0, k_h, BLOCK_KH):
            kh_offsets = kh_start + tl.arange(0, BLOCK_KH)
            kh_mask = kh_offsets < k_h
            
            for kw_start in range(0, k_w, BLOCK_KW):
                kw_offsets = kw_start + tl.arange(0, BLOCK_KW)
                kw_mask = kw_offsets < k_w
                
                # Compute actual input h, w indices
                h_in = h_in_offsets + kh_offsets[None, :, None, None] * dil_h
                w_in = w_in_offsets + kw_offsets[None, None, :, None] * dil_w
                
                # Create masks for valid input indices
                h_in_mask = (h_in >= 0) & (h_in < in_h)
                w_in_mask = (w_in >= 0) & (w_in < in_w)
                combined_mask = h_in_mask & w_in_mask & c_in_mask[None, None, None, :]
                
                # Load input values
                # Input shape: (batch_size, in_channels, in_h, in_w)
                # We need to gather x[pid_n, c_in, h_in, w_in]
                # But this is tricky in Triton - better to use gather or compute indices explicitly
                
                # Alternative approach: use loops for kernel dimensions
                # For each kernel position
                for kh in range(kh_start, min(kh_start + BLOCK_KH, k_h)):
                    for kw in range(kw_start, min(kw_start + BLOCK_KW, k_w)):
                        # Compute input positions
                        h_in = pid_h * BLOCK_H * stride_h + tl.arange(0, BLOCK_H)[None, :, None, None] * stride_h + kh * dil_h - pad_h
                        w_in = pid_w * BLOCK_W * stride_w + tl.arange(0, BLOCK_W)[None, None, :, None] * stride_w + kw * dil_w - pad_w
                        
                        # Create masks
                        h_in_mask = (h_in >= 0) & (h_in < in_h)
                        w_in_mask = (w_in >= 0) & (w_in < in_w)
                        combined_mask = h_in_mask & w_in_mask
                        
                        # Get input indices
                        input_indices = (
                            n_offsets[:, None, None, None] * (in_channels * in_h * in_w) +
                            c_in_offsets[None, None, None, :] * (in_h * in_w) +
                            h_in * in_w +
                            w_in
                        )
                        
                        # Reshape combined_mask for broadcasting
                        mask_4d = combined_mask[:, :, :, None] & c_in_mask[None, None, None, :]
                        
                        # Load input
                        x_vals = tl.load(
                            x_ptr + input_indices,
                            mask=mask_4d,
                            other=0.0
                        )
                        
                        # Load weight
                        weight_indices = (
                            c_out_offsets[:, None, None, None] * (in_channels * k_h * k_w) +
                            (c_in_offsets - c_in_start)[None, None, None, :] * (k_h * k_w) +
                            (kh - kh_start) * k_w +
                            (kw - kw_start)
                        )
                        
                        w_vals = tl.load(
                            w_ptr + weight_indices,
                            mask=c_out_mask[:, None, None, None] & c_in_mask[None, None, None, :],
                            other=0.0
                        )
                        
                        # Accumulate: output += x * w
                        output += x_vals * w_vals
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + c_out_offsets, mask=c_out_mask, other=0.0)
        output += bias[None, :, None, None]
    
    # Store output
    # Output shape: (batch_size, out_channels, out_h, out_w)
    out_indices = (
        n_offsets[:, None, None, None] * (out_channels * out_h * out_w) +
        c_out_offsets[None, :, None, None] * (out_h * out_w) +
        h_offsets[None, None, :, None] * out_w +
        w_offsets[None, None, None, :]
    )
    
    out_mask = (
        n_mask[:, None, None, None] &
        c_out_mask[None, :, None, None] &
        h_mask[None, None, :, None] &
        w_mask[None, None, None, :]
    )
    
    # Reshape output for storage
    output_flat = output.reshape(BLOCK_N * BLOCK_C_OUT * BLOCK_H * BLOCK_W)
    out_indices_flat = out_indices.reshape(BLOCK_N * BLOCK_C_OUT * BLOCK_H * BLOCK_W)
    out_mask_flat = out_mask.reshape(BLOCK_N * BLOCK_C_OUT * BLOCK_H * BLOCK_W)
    
    tl.store(out_ptr + out_indices_flat, output_flat, mask=out_mask_flat)


def triton_conv2d(x, weight, bias, stride, padding, dilation, groups):
    """
    Triton implementation of 2D convolution.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, height, width)
        weight: Weight tensor of shape (out_channels, in_channels // groups, k_h, k_w)
        bias: Bias tensor of shape (out_channels,) or None
        stride: Tuple (stride_h, stride_w)
        padding: Tuple (pad_h, pad_w)
        dilation: Tuple (dil_h, dil_w)
        groups: Number of groups
        
    Returns:
        Output tensor of shape (batch_size, out_channels, out_h, out_w)
    """
    batch_size, in_channels, in_h, in_w = x.shape
    out_channels, _, k_h, k_w = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    
    # Calculate output dimensions
    out_h = (in_h + 2 * pad_h - dil_h * (k_h - 1) - 1) // stride_h + 1
    out_w = (in_w + 2 * pad_w - dil_w * (k_w - 1) - 1) // stride_w + 1
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Allocate output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Define block sizes for tiling
    BLOCK_N = 1  # batch size per block
    BLOCK_C_OUT = 8  # output channels per block
    BLOCK_C_IN = 8  # input channels per block (should divide in_channels or groups)
    BLOCK_KH = 2  # kernel height per block
    BLOCK_KW = 2  # kernel width per block
    BLOCK_H = 8  # output height per block
    BLOCK_W = 8  # output width per block
    
    # Compute grid dimensions
    grid = (
        (batch_size + BLOCK_N - 1) // BLOCK_N,
        (out_channels + BLOCK_C_OUT - 1) // BLOCK_C_OUT,
        (out_h + BLOCK_H - 1) // BLOCK_H,
        (out_w + BLOCK_W - 1) // BLOCK_W,
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, groups,
        in_h, in_w, out_h, out_w,
        k_h, k_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        # For groups, we need to pass weight offsets
        # Simplified: assume groups=1 for now, or pass appropriate offsets
        # In a full implementation, would need to handle group offsets properly
        tl.full((1,), 0, tl.int64),  # placeholder for weight_start_ptr
        BLOCK_N=BLOCK_N,
        BLOCK_C_OUT=BLOCK_C_OUT,
        BLOCK_C_IN=BLOCK_C_IN,
        BLOCK_KH=BLOCK_KH,
        BLOCK_KW=BLOCK_KW,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for 2D convolution.
    
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
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create the convolution layer parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        import math
        
        # Ensure input is on GPU
        if not x.is_cuda:
            x = x.cuda()
        
        # Call the Triton kernel
        return triton_conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )