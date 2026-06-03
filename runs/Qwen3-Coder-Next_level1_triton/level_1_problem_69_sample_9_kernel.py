import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, H_in, W_in)
    w_ptr,  # Weight tensor: (in_channels, out_channels, kH, kW)
    b_ptr,  # Bias tensor: (out_channels,) or None
    y_ptr,  # Output tensor: (batch, out_channels, H_out, W_out)
    batch_size, in_channels, out_channels,
    H_in, W_in, H_out, W_out,
    kH, kW,
    stride_h, stride_w,
    padding_h, padding_w,
    output_padding_h, output_padding_w,
    dilation_h, dilation_w,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr,  # Block size for input channels
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_c = tl.program_id(1)
    
    # Skip if batch index is out of bounds
    if pid_batch >= batch_size:
        return
    
    # Output channel block start
    out_c_start = pid_out_c * BLOCK_SIZE_M
    out_c_offsets = out_c_start + tl.arange(0, BLOCK_SIZE_M)
    out_c_mask = out_c_offsets < out_channels
    
    # Compute output height and width positions
    # For transposed convolution: output position corresponds to input position * stride + ...
    # We need to iterate over all input positions that contribute to each output position
    
    # Create output tensor pointer offset
    y_ptr_batch = y_ptr + pid_batch * out_channels * H_out * W_out
    
    # Process each output position
    for oh in range(H_out):
        for ow in range(W_out):
            # Compute the corresponding input position
            ih = (oh - output_padding_h - padding_h) // stride_h
            iw = (ow - output_padding_w - padding_w) // stride_w
            
            # Check if this input position is valid
            if ih >= 0 and ih < H_in and iw >= 0 and iw < W_in:
                # Check if the kernel position aligns properly
                residual_h = (oh - output_padding_h - padding_h) % stride_h
                residual_w = (ow - output_padding_w - padding_w) % stride_w
                
                # For valid transposed convolution, residual must be 0
                if residual_h == 0 and residual_w == 0:
                    kernel_h = ih * stride_h + padding_h - oh + output_padding_h + dilation_h * (kH - 1)
                    kernel_w = iw * stride_w + padding_w - ow + output_padding_w + dilation_w * (kW - 1)
                    
                    # Accumulate contributions from all input channels
                    acc = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
                    
                    for ic in range(in_channels):
                        # Load weight: (in_c, out_c, kH, kW)
                        w_idx = ic * out_channels * kH * kW + out_c_offsets * kH * kW + kernel_h * kW + kernel_w
                        w_val = tl.load(w_ptr + w_idx, mask=out_c_mask, other=0.0)
                        
                        # Load input: (batch, in_c, H_in, W_in)
                        x_idx = pid_batch * in_channels * H_in * W_in + ic * H_in * W_in + ih * W_in + iw
                        x_val = tl.load(x_ptr + x_idx)
                        
                        # Accumulate
                        acc += w_val * x_val
                    
                    # Add bias if available
                    if b_ptr is not None:
                        b_idx = out_c_offsets
                        b_val = tl.load(b_ptr + b_idx, mask=out_c_mask, other=0.0)
                        acc += b_val
                    
                    # Store result
                    y_idx = pid_batch * out_channels * H_out * W_out + out_c_offsets * H_out * W_out + oh * W_out + ow
                    tl.store(y_ptr + y_idx, acc, mask=out_c_mask)


class TritonConvTranspose2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, output_padding, dilation, groups):
        # Validate parameters
        assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
        assert groups == 1, "Only groups=1 is supported for now."
        
        # Save parameters for backward pass (if needed)
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        ctx.dilation = dilation
        ctx.groups = groups
        
        # Get dimensions
        batch_size, in_channels, H_in, W_in = x.shape
        out_channels, _, kH, kW = weight.shape
        
        # Compute output dimensions
        H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kH - 1) + output_padding[0] + 1
        W_out = (W_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kW - 1) + output_padding[1] + 1
        
        # Create output tensor
        y = torch.empty(batch_size, out_channels, H_out, W_out, dtype=x.dtype, device=x.device)
        
        # Set block sizes (tunable)
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 1
        BLOCK_SIZE_K = 16
        
        # Grid dimensions
        grid = (batch_size, triton.cdiv(out_channels, BLOCK_SIZE_M))
        
        # Launch kernel
        conv_transpose2d_kernel[grid](
            x, weight, bias, y,
            batch_size, in_channels, out_channels,
            H_in, W_in, H_out, W_out,
            kH, kW,
            stride[0], stride[1],
            padding[0], padding[1],
            output_padding[0], output_padding[1],
            dilation[0], dilation[1],
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        
        return y


def triton_conv_transpose2d(x, weight, bias=None, stride=(1, 1), padding=(0, 0), 
                           output_padding=(0, 0), dilation=(1, 1), groups=1):
    return TritonConvTranspose2d.apply(x, weight, bias, stride, padding, output_padding, dilation, groups)


class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution operation with asymmetric input and kernel size.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of integers representing the kernel size (height, width).
        stride (tuple, optional): Tuple of integers representing the stride of the convolution. Defaults to (1, 1).
        padding (tuple, optional): Tuple of integers representing the padding applied to the input. Defaults to (0, 0).
        output_padding (tuple, optional): Tuple of integers representing the additional size added to one side of the output shape. Defaults to (0, 0).
        dilation (tuple, optional): Tuple of integers representing the spacing between kernel elements. Defaults to (1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
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
        
        # Initialize weights
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        
        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        # Kaiming initialization for conv transpose
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            self.stride, self.padding,
            self.output_padding, self.dilation,
            self.groups
        )