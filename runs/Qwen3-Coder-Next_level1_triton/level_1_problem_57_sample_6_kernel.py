import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Input tensor
    w_ptr,  # Weight tensor
    b_ptr,  # Bias tensor (optional)
    y_ptr,  # Output tensor
    batch_size,  # B
    in_channels,  # C_in
    out_channels,  # C_out
    input_h,  # H_in
    input_w,  # W_in
    kernel_size,  # K
    stride,  # S
    padding,  # P
    output_padding,  # OP
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate output dimensions
    output_h = (input_h - 1) * stride - 2 * padding + output_padding + kernel_size
    output_w = (input_w - 1) * stride - 2 * padding + output_padding + kernel_size
    
    # Calculate the starting position for this block
    h_start = pid_h * BLOCK_H
    w_start = pid_w * BLOCK_W
    
    # Create offsets for output spatial dimensions
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    
    # Create masks for output
    h_mask = h_offsets < output_h
    w_mask = w_offsets < output_w
    hw_mask = h_mask[:, None] & w_mask[None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Iterate over input channels and kernel positions
    for c_in in range(0, in_channels, BLOCK_C):
        c_in_offsets = c_in + tl.arange(0, BLOCK_C)
        c_in_mask = c_in_offsets < in_channels
        
        # Load input block
        x_offset_h = (h_start + stride) // stride  # This needs to be adjusted based on stride
        x_offset_w = (w_start + stride) // stride
        
        # For each output position, calculate corresponding input position
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Calculate input coordinates that contribute to this output position
                h_in = h_start + kh - kernel_size + 1 + padding
                w_in = w_start + kw - kernel_size + 1 + padding
                
                # Adjust for stride
                h_in_strided = h_in // stride
                w_in_strided = w_in // stride
                
                # Check if valid input position
                h_in_valid = (h_in_strided >= 0) & (h_in_strided < input_h)
                w_in_valid = (w_in_strided >= 0) & (w_in_strided < input_w)
                valid_mask = hw_mask & h_in_valid[:, None] & w_in_valid[None, :]
                
                # Calculate actual input indices
                input_h_idx = h_in_strided
                input_w_idx = w_in_strided
                
                # Load input values
                x_ptrs = x_ptr + pid_b * (in_channels * input_h * input_w) + \
                         c_in_offsets[:, None, None] * (input_h * input_w) + \
                         input_h_idx[None, :, None] * input_w + \
                         input_w_idx[None, None, :]
                
                # Need to handle broadcasting for BLOCK_H, BLOCK_W
                # Simplified approach: use 3D indexing
                x_block = tl.zeros((BLOCK_C, BLOCK_H, BLOCK_W), dtype=tl.float32)
                
                # For each channel in BLOCK_C
                for i, c in enumerate(c_in_offsets):
                    if c < in_channels:
                        x_val = tl.load(
                            x_ptr + pid_b * (in_channels * input_h * input_w) + 
                            c * (input_h * input_w) + 
                            input_h_idx * input_w + 
                            input_w_idx,
                            mask=valid_mask,
                            other=0.0
                        )
                        x_block = tl.where(c_in_offsets[:, None, None] == c, 
                                          tl.broadcast_to(x_val[None, :, :], 
                                                        (1, BLOCK_H, BLOCK_W)), 
                                          x_block)
                
                # Load weight value for this kernel position and output channel
                w_val = tl.load(
                    w_ptr + pid_c_out * (in_channels * kernel_size * kernel_size) + 
                    c_in_offsets * (kernel_size * kernel_size) + 
                    kh * kernel_size + kw,
                    mask=c_in_mask,
                    other=0.0
                )
                
                # Accumulate: x * w
                acc += tl.sum(x_block * w_val[:, None, None], axis=0)
    
    # Add bias if present
    if b_ptr is not None:
        b_val = tl.load(b_ptr + pid_c_out)
        acc += b_val
    
    # Store result
    y_ptrs = y_ptr + pid_b * (out_channels * output_h * output_w) + \
             pid_c_out * (output_h * output_w) + \
             h_offsets[:, None] * output_w + \
             w_offsets[None, :]
    
    tl.store(y_ptrs, acc, mask=hw_mask)


def triton_conv_transpose2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                            stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1):
    """
    Triton implementation of ConvTranspose2d
    """
    # Get dimensions
    batch_size, in_channels, input_h, input_w = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    output_h = (input_h - 1) * stride - 2 * padding + output_padding + kernel_h
    output_w = (input_w - 1) * stride - 2 * padding + output_padding + kernel_w
    
    # Prepare output tensor
    y = torch.empty(batch_size, out_channels, output_h, output_w, device=x.device, dtype=x.dtype)
    
    # Grid configuration
    # For large tensors, use tiling to avoid race conditions
    BLOCK_H = 32
    BLOCK_W = 32
    BLOCK_C = 16
    
    grid_h = (output_h + BLOCK_H - 1) // BLOCK_H
    grid_w = (output_w + BLOCK_W - 1) // BLOCK_W
    
    # Launch kernel
    conv_transpose2d_kernel[
        (batch_size, out_channels, grid_h, grid_w)
    ](
        x, weight, bias, y,
        batch_size, in_channels, out_channels,
        input_h, input_w,
        kernel_h,  # Assuming square kernel
        stride, padding, output_padding,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        BLOCK_C=BLOCK_C,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
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
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        
        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using custom Triton kernel.
        """
        return triton_conv_transpose2d(x, self.weight, self.bias,
                                       stride=self.stride, padding=self.padding,
                                       output_padding=self.output_padding, groups=self.groups)