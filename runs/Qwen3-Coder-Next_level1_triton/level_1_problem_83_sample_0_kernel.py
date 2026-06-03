import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_asymmetric_kernel(
    x_ptr,  # Input tensor pointer (batch, channels, height, width)
    w_ptr,  # Weight tensor pointer (channels, kernel_height, kernel_width)
    b_ptr,  # Bias pointer (channels,) or None
    out_ptr,  # Output tensor pointer
    batch_size, in_channels, height, width,
    kernel_h, kernel_w,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    out_h, out_w,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
):
    # Get batch and channel indices
    bc_id = tl.program_id(0)
    batch_id = bc_id // in_channels
    channel_id = bc_id % in_channels
    
    # Get output position
    out_h_id = tl.program_id(1)
    out_w_id = tl.program_id(2)
    
    # Compute input position corresponding to this output
    in_h_start = out_h_id * stride_h - pad_h
    in_w_start = out_w_id * stride_w - pad_w
    
    # Initialize accumulator
    acc = 0.0
    
    # Loop over kernel height (only kernel_h, kernel_w is always 1 in our case)
    for kh in range(KERNEL_H):
        in_h = in_h_start + kh * dil_h
        # Check bounds for height
        if 0 <= in_h < height:
            # For kernel width, it's always 1 in our case, so we only need one iteration
            in_w = in_w_start  # Since kernel_w = 1
            
            # Check bounds for width
            if 0 <= in_w < width:
                # Compute input index
                input_offset = (batch_id * in_channels * height * width + 
                               channel_id * height * width + 
                               in_h * width + in_w)
                
                # Compute weight index
                weight_offset = (channel_id * kernel_h * kernel_w + 
                                kh * kernel_w + 0)  # Only kh, kernel_w is always 1
                
                # Load values
                x_val = tl.load(x_ptr + input_offset)
                w_val = tl.load(w_ptr + weight_offset)
                
                # Accumulate
                acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias_offset = channel_id
        acc += tl.load(b_ptr + bias_offset)
    
    # Compute output index
    out_offset = (batch_id * in_channels * out_h * out_w + 
                 channel_id * out_h * out_w + 
                 out_h_id * out_w + out_w_id)
    
    # Store result
    tl.store(out_ptr + out_offset, acc)


def triton_depthwise_conv2d_asymmetric(x, weight, bias=None, 
                                       stride=(1, 1), padding=(0, 0), 
                                       dilation=(1, 1)):
    """
    Performs asymmetric depthwise 2D convolution (kernel_size, 1) using Triton.
    
    Args:
        x: Input tensor (batch_size, in_channels, height, width)
        weight: Weight tensor (in_channels, kernel_size, 1)
        bias: Optional bias tensor (in_channels,)
        stride, padding, dilation: Convolution parameters
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    batch_size, in_channels, height, width = x.shape
    _, _, kernel_h, kernel_w = weight.shape
    
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    
    # Calculate output dimensions
    out_h = (height + 2 * pad_h - dil_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (width + 2 * pad_w - dil_w * (kernel_w - 1) - 1) // stride_w + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, in_channels, out_h, out_w, device=x.device, dtype=x.dtype)
    
    # Grid dimensions: (batch_size * in_channels, out_h, out_w)
    grid = (batch_size * in_channels, out_h, out_w)
    
    # Launch kernel
    BLOCK_SIZE_H = 1  # Not used for tiling in this implementation
    BLOCK_SIZE_W = 1  # Not used for tiling in this implementation
    
    depthwise_conv2d_asymmetric_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, height, width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        out_h, out_w,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        KERNEL_H=kernel_h,
        KERNEL_W=kernel_w,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernels for asymmetric depthwise convolution.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Create the Conv2d layer to initialize weights
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size, 1), 
                               stride=stride, padding=padding, dilation=dilation, 
                               groups=in_channels, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the asymmetric depthwise 2D convolution using optimized Triton kernel.
        """
        # Ensure input is on CUDA
        if not x.is_cuda:
            x = x.cuda()
        
        # Get the weight and bias from the original conv2d layer
        weight = self.conv2d.weight
        bias = self.conv2d.bias if self.conv2d.bias is not None else None
        
        # Convert to proper format for our kernel
        # The weight from nn.Conv2d is in format (out_channels, in_channels_per_group, kernel_height, kernel_width)
        # Since groups=in_channels, in_channels_per_group=1, so weight is (in_channels, 1, kernel_size, 1)
        # We need to reshape it to (in_channels, kernel_size, 1) to match our kernel expectation
        weight_reshaped = weight.view(self.in_channels, self.kernel_size, 1)
        
        # Call our Triton kernel
        return triton_depthwise_conv2d_asymmetric(
            x, weight_reshaped, bias,
            stride=(self.stride, 1), 
            padding=(self.padding, 0), 
            dilation=(self.dilation, 1)
        )