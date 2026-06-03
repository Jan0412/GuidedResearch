import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # Input tensor pointer
    w_ptr,  # Weight tensor pointer
    b_ptr,  # Bias tensor pointer (can be None)
    out_ptr,  # Output tensor pointer
    batch_size,  # B
    in_channels,  # C_in
    out_channels,  # C_out
    depth, height, width,  # Input dimensions
    out_depth, out_height, out_width,  # Output dimensions
    kernel_size,  # K
    stride, padding, output_padding,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_CIN: tl.constexpr,
    BLOCK_SIZE_COUT: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_cout = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Create ranges for output dimensions
    out_d_start = pid_d * BLOCK_SIZE_D
    out_d_offsets = out_d_start + tl.arange(0, BLOCK_SIZE_D)
    out_d_mask = out_d_offsets < out_depth
    
    out_h_start = pid_h * BLOCK_SIZE_H
    out_h_offsets = out_h_start + tl.arange(0, BLOCK_SIZE_H)
    out_h_mask = out_h_offsets < out_height
    
    out_w_start = pid_w * BLOCK_SIZE_W
    out_w_offsets = out_w_start + tl.arange(0, BLOCK_SIZE_W)
    out_w_mask = out_w_offsets < out_width
    
    # Create range for output channels
    cout_offsets = pid_cout * BLOCK_SIZE_COUT + tl.arange(0, BLOCK_SIZE_COUT)
    cout_mask = cout_offsets < out_channels
    
    # Create range for input channels
    cin_offsets = tl.arange(0, BLOCK_SIZE_CIN)
    cin_mask = cin_offsets < in_channels
    
    # Initialize accumulator
    output_sum = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_COUT), dtype=tl.float32)
    
    # Iterate over input channels
    for c_in in range(0, in_channels, BLOCK_SIZE_CIN):
        # Calculate input indices that contribute to each output position
        # For transposed convolution: in_d = (out_d - output_padding) // stride
        # Only process if the input index is valid
        in_d_start = (out_d_offsets - output_padding + stride - 1) // stride
        in_d_mask = ((out_d_offsets - output_padding) % stride == 0) & (in_d_start >= 0) & (in_d_start < depth)
        
        in_h_start = (out_h_offsets - output_padding + stride - 1) // stride
        in_h_mask = ((out_h_offsets - output_padding) % stride == 0) & (in_h_start >= 0) & (in_h_start < height)
        
        in_w_start = (out_w_offsets - output_padding + stride - 1) // stride
        in_w_mask = ((out_w_offsets - output_padding) % stride == 0) & (in_w_start >= 0) & (in_w_start < width)
        
        # Get the kernel indices: k = out - in*stride + padding
        k_d_offsets = out_d_offsets - in_d_start * stride + padding
        k_h_offsets = out_h_offsets - in_h_start * stride + padding
        k_w_offsets = out_w_offsets - in_w_start * stride + padding
        
        # Load input values for this channel
        x_batch_ptr = x_ptr + (pid_b * in_channels * depth * height * width + 
                               c_in * depth * height * width)
        x_offsets = (in_d_start[:, None, None, None] * height * width +
                     in_h_start[None, :, None, None] * width +
                     in_w_start[None, None, :, None])
        x_mask = (in_d_mask[:, None, None, None] & 
                  in_h_mask[None, :, None, None] & 
                  in_w_mask[None, None, :, None] &
                  cin_mask[None, None, None, :])
        
        # Reshape x for broadcasting
        x_val = tl.load(x_batch_ptr + x_offsets, mask=x_mask, other=0.0)
        
        # Load weights for these input channels and output channels
        # Weight shape: (in_channels, out_channels, K, K, K)
        w_batch_ptr = w_ptr + (c_in * out_channels * kernel_size * kernel_size * kernel_size +
                               cout_offsets[None, :, None, None, None] * kernel_size * kernel_size * kernel_size)
        
        # Compute output for each kernel position
        for k_d in range(kernel_size):
            for k_h in range(kernel_size):
                for k_w in range(kernel_size):
                    # Check if kernel indices are valid
                    k_d_valid = (k_d_offsets[None, None, None, :] == k_d) & (k_d_offsets >= 0)
                    k_h_valid = (k_h_offsets[None, None, :, None] == k_h) & (k_h_offsets >= 0)
                    k_w_valid = (k_w_offsets[None, :, None, None] == k_w) & (k_w_offsets >= 0)
                    
                    # Load weights
                    w_offsets = (k_d * kernel_size * kernel_size +
                                 k_h * kernel_size +
                                 k_w)
                    w_val = tl.load(w_batch_ptr + w_offsets, mask=cout_mask[None, :, None, None, None], other=0.0)
                    
                    # Compute contribution
                    contrib = x_val * w_val
                    output_sum += tl.sum(contrib, axis=3)  # Sum over input channels
                    
    # Apply bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + cout_offsets, mask=cout_mask, other=0.0)
        output_sum += bias[None, None, None, :]
    
    # Store result
    out_batch_ptr = out_ptr + (pid_b * out_channels * out_depth * out_height * out_width +
                               cout_offsets[None, None, None, :] * out_depth * out_height * out_width)
    out_offsets = (out_d_offsets[:, None, None, None] * out_height * out_width +
                   out_h_offsets[None, :, None, None] * out_width +
                   out_w_offsets[None, None, :, None])
    out_mask = (out_d_mask[:, None, None, None] & 
                out_h_mask[None, :, None, None] & 
                out_w_mask[None, None, :, None] &
                cout_mask[None, None, None, :])
    
    tl.store(out_batch_ptr + out_offsets, output_sum, mask=out_mask)


def triton_conv_transpose3d(x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
    """
    Triton implementation of 3D transposed convolution.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    batch_size, in_channels, depth, height, width = x.shape
    out_channels, _, kernel_size, _, _ = weight.shape
    
    # Calculate output dimensions
    out_depth = (depth - 1) * stride - 2 * padding + (kernel_size - 1) + output_padding + 1
    out_height = (height - 1) * stride - 2 * padding + (kernel_size - 1) + output_padding + 1
    out_width = (width - 1) * stride - 2 * padding + (kernel_size - 1) + output_padding + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_depth, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Grid dimensions
    grid = lambda meta: (
        batch_size,
        triton.cdiv(out_channels, meta["BLOCK_SIZE_COUT"]),
        triton.cdiv(out_depth, meta["BLOCK_SIZE_D"]),
        triton.cdiv(out_height, meta["BLOCK_SIZE_H"]),
        triton.cdiv(out_width, meta["BLOCK_SIZE_W"]),
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        depth, height, width,
        out_depth, out_height, out_width,
        kernel_size,
        stride, padding, output_padding,
        BLOCK_SIZE_D=4,
        BLOCK_SIZE_H=4,
        BLOCK_SIZE_W=4,
        BLOCK_SIZE_CIN=16,
        BLOCK_SIZE_COUT=8,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution with square input and square kernel using Triton kernel.
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
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using Triton kernel.
        """
        return triton_conv_transpose3d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding, 
            output_padding=self.output_padding, groups=self.groups
        )