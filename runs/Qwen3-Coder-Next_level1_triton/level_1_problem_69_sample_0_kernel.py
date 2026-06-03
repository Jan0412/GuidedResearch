import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Input tensor: (B, C_in, H_in, W_in)
    w_ptr,  # Weight tensor: (C_in, C_out, K_h, K_w)
    b_ptr,  # Bias tensor: (C_out,) or None
    out_ptr,  # Output tensor: (B, C_out, H_out, W_out)
    batch_size, in_channels, out_channels,
    height_in, width_in,
    kernel_h, kernel_w,
    stride_h, stride_w,
    padding_h, padding_w,
    output_padding_h, output_padding_w,
    dilation_h, dilation_w,
    grid_h, grid_w,  # Output height and width
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch size
    BLOCK_SIZE_K: tl.constexpr,  # Block size for input channels
    BLOCK_SIZE_DH: tl.constexpr,  # Block size for output height
    BLOCK_SIZE_DW: tl.constexpr,  # Block size for output width
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate output position
    out_h = pid_h * BLOCK_SIZE_DH + tl.arange(0, BLOCK_SIZE_DH)
    out_w = pid_w * BLOCK_SIZE_DW + tl.arange(0, BLOCK_SIZE_DW)
    
    # Create masks for output positions
    mask_h = out_h < grid_h
    mask_w = out_w < grid_w
    mask_hw = mask_h[:, None] & mask_w[None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_DH, BLOCK_SIZE_DW), dtype=tl.float32)
    
    # Compute input position for each output position
    # For transposed convolution: out_h = in_h * stride_h + (k_h - 1) * dilation_h - padding_h
    # So in_h = (out_h + padding_h - (k_h - 1) * dilation_h) / stride_h
    
    for k_h in range(kernel_h):
        in_h = (out_h[:, None] + padding_h - k_h * dilation_h) // stride_h
        mask_in_h = (in_h >= 0) & (in_h < height_in) & mask_h[:, None]
        
        for k_w in range(kernel_w):
            in_w = (out_w[None, :] + padding_w - k_w * dilation_w) // stride_w
            mask_in_w = (in_w >= 0) & (in_w < width_in) & mask_w[None, :]
            
            # Calculate input indices
            in_h_valid = in_h * mask_in_h  # Valid positions only
            in_w_valid = in_w * mask_in_w
            
            # Compute input pointer offset
            # Input: (batch, in_c, in_h, in_w)
            input_offset = (
                pid_batch * (in_channels * height_in * width_in) +
                tl.arange(0, in_channels)[:, None, None] * (height_in * width_in) +
                in_h_valid[None, :, :] * width_in +
                in_w_valid[None, :, :]
            )
            input_offset = tl.flatten(input_offset)
            mask_input = (
                tl.arange(0, in_channels)[:, None, None] < in_channels &
                mask_in_h[None, :, :] &
                mask_in_w[None, :, :]
            )
            mask_input = tl.flatten(mask_input)
            
            # Load input values
            x_offset = x_ptr + input_offset
            x_vals = tl.load(x_offset, mask=mask_input, other=0.0)
            x_vals = tl.reshape(x_vals, (in_channels, BLOCK_SIZE_DH, BLOCK_SIZE_DW))
            
            # Compute weight pointer offset
            # Weight: (in_c, out_c, k_h, k_w)
            weight_offset = (
                tl.arange(0, in_channels)[:, None, None] * (out_channels * kernel_h * kernel_w) +
                pid_out_c * (kernel_h * kernel_w) +
                k_h * kernel_w +
                k_w
            )
            weight_offset = tl.flatten(weight_offset)
            w_vals = tl.load(w_ptr + weight_offset, mask=mask_input, other=0.0)
            w_vals = tl.reshape(w_vals, (in_channels, BLOCK_SIZE_DH, BLOCK_SIZE_DW))
            
            # Accumulate: sum over in_c dimension
            acc += tl.sum(x_vals * w_vals, axis=0)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_out_c)
        acc += bias
    
    # Store result
    out_offset = (
        pid_batch * (out_channels * grid_h * grid_w) +
        pid_out_c * (grid_h * grid_w) +
        out_h[:, None] * grid_w +
        out_w[None, :]
    )
    mask_out = mask_hw
    tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty), mask=mask_out)


class TritonConvTranspose2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, output_padding, dilation, groups):
        # Save parameters for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        ctx.dilation = dilation
        ctx.groups = groups
        
        # Calculate output dimensions
        B, C_in, H_in, W_in = x.shape
        C_out, _, K_h, K_w = weight.shape
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        output_padding_h, output_padding_w = output_padding
        dilation_h, dilation_w = dilation
        
        H_out = (H_in - 1) * stride_h - 2 * padding_h + output_padding_h + (K_h - 1) * dilation_h + 1
        W_out = (W_in - 1) * stride_w - 2 * padding_w + output_padding_w + (K_w - 1) * dilation_w + 1
        
        # Create output tensor
        out = torch.empty((B, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
        
        # Configure kernel launch parameters
        BLOCK_SIZE_M = 8  # Output channels per block
        BLOCK_SIZE_N = 4  # Batch size per block (we process one batch at a time)
        BLOCK_SIZE_K = 8  # Input channels per block
        BLOCK_SIZE_DH = 8  # Output height per block
        BLOCK_SIZE_DW = 8  # Output width per block
        
        grid = (
            B,  # Batch size
            triton.cdiv(C_out, BLOCK_SIZE_M),  # Output channels
            triton.cdiv(H_out, BLOCK_SIZE_DH),  # Output height
            triton.cdiv(W_out, BLOCK_SIZE_DW)   # Output width
        )
        
        # Launch kernel
        conv_transpose2d_kernel[grid](
            x, weight, bias, out,
            B, C_in, C_out,
            H_in, W_in,
            K_h, K_w,
            stride_h, stride_w,
            padding_h, padding_w,
            output_padding_h, output_padding_w,
            dilation_h, dilation_w,
            H_out, W_out,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_DH=BLOCK_SIZE_DH,
            BLOCK_SIZE_DW=BLOCK_SIZE_DW
        )
        
        return out


def triton_conv_transpose2d(x, weight, bias=None, stride=(1, 1), padding=(0, 0), 
                           output_padding=(0, 0), dilation=(1, 1), groups=1):
    return TritonConvTranspose2dFunction.apply(x, weight, bias, stride, padding, 
                                              output_padding, dilation, groups)


class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution operation with asymmetric input and kernel size.
    Uses optimized Triton kernel for forward pass.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), 
                 output_padding: tuple = (0, 0), dilation: tuple = (1, 1), 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weight and bias
        K_h, K_w = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, K_h, K_w))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using Triton kernel.
        """
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            self.stride, self.padding,
            self.output_padding, self.dilation,
            self.groups
        )