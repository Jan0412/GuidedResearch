import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # Input tensor (batch, in_channels, depth, height, width)
    w_ptr,  # Weight tensor (in_channels, out_channels, k_d, k_h, k_w)
    b_ptr,  # Bias tensor (out_channels) - can be None
    out_ptr,  # Output tensor (batch, out_channels, out_d, out_h, out_w)
    batch_size, in_channels, out_channels,
    depth, height, width,
    out_depth, out_height, out_width,
    k_d, k_h, k_w,
    stride: tl.constexpr,
    padding: tl.constexpr,
    output_padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate global indices for output tensor
    # Output tensor is (batch, out_channels, out_d, out_h, out_w)
    # We'll parallelize over batch, out_channels, and spatial positions
    
    # Linear index to work on
    pid = tl.program_id(0)
    
    # Total number of output elements
    total_elements = batch_size * out_channels * out_depth * out_height * out_width
    
    if pid >= total_elements:
        return
    
    # Decode linear index to (b, oc, od, oh, ow)
    temp = pid
    ow = temp % out_width
    temp //= out_width
    oh = temp % out_height
    temp //= out_height
    od = temp % out_depth
    temp //= out_depth
    oc = temp % out_channels
    temp //= out_channels
    b = temp
    
    # Compute starting point in input for this output position
    # For transposed convolution, input position = output_position * stride - padding + offset
    # But we need to compute which input positions contribute to this output position
    
    # Accumulator for the convolution result
    acc = 0.0
    
    # Iterate over kernel dimensions
    for kd in range(k_d):
        for kh in range(k_h):
            for kw in range(k_w):
                # Compute corresponding input position
                id_ = od - kd * stride
                ih = oh - kh * stride
                iw = ow - kw * stride
                
                # Check if input position is within bounds
                if (id_ >= 0 and id_ < depth and 
                    ih >= 0 and ih < height and 
                    iw >= 0 and iw < width):
                    
                    # Compute input pointer offset
                    input_offset = (b * in_channels * depth * height * width +
                                   0 * depth * height * width +  # in_channels offset will be added below
                                   id_ * height * width +
                                   ih * width +
                                   iw)
                    
                    # Compute weight pointer offset
                    weight_offset = (0 * out_channels * k_d * k_h * k_w +
                                    oc * k_d * k_h * k_w +
                                    kd * k_h * k_w +
                                    kh * k_w +
                                    kw)
                    
                    # Iterate over input channels
                    for ic in range(in_channels):
                        # Load input value
                        input_val = tl.load(x_ptr + input_offset + ic * depth * height * width)
                        # Load weight value
                        weight_val = tl.load(w_ptr + weight_offset + ic * out_channels * k_d * k_h * k_w)
                        # Accumulate
                        acc += input_val * weight_val
    
    # Add bias if present
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + oc)
        acc += bias_val
    
    # Store result
    tl.store(out_ptr + pid, acc)


def triton_conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups):
    """
    Performs transposed 3D convolution using Triton kernel.
    """
    # Get shapes
    batch_size, in_channels, depth, height, width = x.shape
    out_channels, _, k_d, k_h, k_w = weight.shape
    
    # Compute output shape
    out_depth = (depth - 1) * stride - 2 * padding + (k_d - 1) + output_padding + 1
    out_height = (height - 1) * stride - 2 * padding + (k_h - 1) + output_padding + 1
    out_width = (width - 1) * stride - 2 * padding + (k_w - 1) + output_padding + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_depth, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Total number of output elements
    total_elements = batch_size * out_channels * out_depth * out_height * out_width
    
    # Launch kernel with a reasonable block size
    BLOCK_SIZE = 128
    grid = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        depth, height, width,
        out_depth, out_height, out_width,
        k_d, k_h, k_w,
        stride, padding, output_padding,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model with Triton-based transposed 3D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store original module parameters but replace the convolution operation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.use_bias = bias
        
        # Initialize weights using the same logic as nn.ConvTranspose3d
        # Use Kaiming uniform initialization similar to PyTorch
        k = 1.0 / (in_channels * kernel_size * kernel_size * kernel_size)
        bound = k ** 0.5
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        nn.init.uniform_(self.weight, -bound, bound)
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized transposed 3D convolution.
        """
        return triton_conv_transpose3d(
            x, self.weight, self.bias, 
            self.stride, self.padding, self.output_padding, self.groups
        )