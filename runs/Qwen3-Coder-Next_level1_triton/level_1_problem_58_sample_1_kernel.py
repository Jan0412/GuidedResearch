import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # Input tensor pointer (N, C_in, D_in, H_in, W_in)
    w_ptr,  # Weight tensor pointer (C_in, C_out // groups, k_d, k_h, k_w)
    b_ptr,  # Bias tensor pointer (C_out,)
    out_ptr,  # Output tensor pointer (N, C_out, D_out, H_out, W_out)
    N, C_in, C_out, groups,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    k_d, k_h, k_w,
    s_d, s_h, s_w,  # strides
    p_d, p_h, p_w,  # padding
    op_d, op_h, op_w,  # output_padding
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs for batch and output channel
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1) * BLOCK_SIZE_C_OUT
    # Output spatial positions
    out_d = tl.program_id(2)
    out_h = tl.program_id(3)
    out_w = tl.program_id(4)
    
    # Compute input channel group
    c_per_group = C_in // groups
    group_idx = out_c_idx // (C_out // groups)
    out_c_idx_in_group = out_c_idx % (C_out // groups)
    
    # Calculate corresponding input position for the first output element
    # For transposed conv: input_d = (out_d - op_d - k_d + 1 + p_d + stride_d * input_d) / stride_d
    # But more straightforward: for each output position, we accumulate contributions from input positions
    # that would map to this output position
    
    # Accumulator for the result
    acc = tl.zeros((BLOCK_SIZE_C_OUT,), dtype=tl.float32)
    
    # Loop over input channels in the group
    for c_in_offset in range(0, c_per_group, BLOCK_SIZE_K):
        c_in = group_idx * c_per_group + c_in_offset
        
        # Compute the input position that contributes to (batch_idx, out_d, out_h, out_w)
        # For transposed convolution: out_d = s_d * in_d + k_d - 1 - p_d + op_d
        # So in_d = (out_d - k_d + 1 + p_d - op_d) / s_d
        in_d_start = (out_d - k_d + 1 + p_d - op_d) // s_d
        in_h_start = (out_h - k_h + 1 + p_h - op_w) // s_h  # Note: using p_h here
        in_w_start = (out_w - k_w + 1 + p_w - op_w) // s_w
        
        # Check if this is a valid input position
        valid_input = (in_d_start >= 0) & (in_d_start < D_in) & \
                      (in_h_start >= 0) & (in_h_start < H_in) & \
                      (in_w_start >= 0) & (in_w_start < W_in)
        
        if valid_input:
            # Calculate the exact kernel position
            k_d_offset = (out_d - s_d * in_d_start - p_d + op_d)
            k_h_offset = (out_h - s_h * in_h_start - p_h + op_h)
            k_w_offset = (out_w - s_w * in_w_start - p_w + op_w)
            
            # Load input value
            in_idx = batch_idx * (C_in * D_in * H_in * W_in) + \
                     c_in * (D_in * H_in * W_in) + \
                     in_d_start * (H_in * W_in) + \
                     in_h_start * W_in + in_w_start
            x_val = tl.load(x_ptr + in_idx, mask=valid_input, other=0.0)
            
            # Load weight values for all output channels in this block
            # Weight layout: (C_in, C_out // groups, k_d, k_h, k_w)
            w_idx_base = c_in * ((C_out // groups) * k_d * k_h * k_w) + \
                         out_c_idx_in_group * (k_d * k_h * k_w) + \
                         k_d_offset * (k_h * k_w) + \
                         k_h_offset * k_w + k_w_offset
            
            # Load weights for the current kernel position
            w_offset = tl.arange(0, BLOCK_SIZE_C_OUT)
            w_idx = w_idx_base + w_offset * (k_d * k_h * k_w * (C_out // groups))
            # Simplified: we need to handle the weight layout properly
            # Actually, for PyTorch ConvTranspose3d, weight shape is (in_channels, out_channels // groups, *kernel_size)
            # So we need to adjust the indexing
            
            # Let's rewrite weight indexing properly
            # For each output channel, we need weight[c_in, c_out_group_idx, k_d, k_h, k_w]
            # where c_out_group_idx = out_c_idx + w_offset - group_idx * (C_out // groups)
            c_out_group_idx = out_c_idx_in_group + w_offset
            w_idx = (c_in * (C_out // groups) + c_out_group_idx) * (k_d * k_h * k_w) + \
                    k_d_offset * (k_h * k_w) + k_h_offset * k_w + k_w_offset
            
            # Transpose the weight to match conv_transpose layout
            # Actually, for ConvTranspose3d, the weight is stored as (C_in, C_out // groups, k_d, k_h, k_w)
            # So our indexing is correct
            
            w_vals = tl.load(w_ptr + w_idx, mask=(w_offset < (C_out // groups)) & valid_input, other=0.0)
            
            # Accumulate: x_val * w_vals
            acc += x_val * w_vals
    
    # Add bias if present
    if b_ptr is not None:
        bias_idx = out_c_idx + tl.arange(0, BLOCK_SIZE_C_OUT)
        bias_vals = tl.load(b_ptr + bias_idx, mask=bias_idx < C_out, other=0.0)
        acc += bias_vals
    
    # Store result
    out_c_range = out_c_idx + tl.arange(0, BLOCK_SIZE_C_OUT)
    out_mask = out_c_range < C_out
    
    out_idx = (batch_idx * C_out + out_c_range) * (D_out * H_out * W_out) + \
              out_d * (H_out * W_out) + out_h * W_out + out_w
    
    tl.store(out_ptr + out_idx, acc.to(x_ptr.dtype.element_ty), mask=out_mask)


def triton_conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups):
    """
    Performs transposed 3D convolution using Triton kernel.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    N, C_in, D_in, H_in, W_in = x.shape
    C_in_w, C_out_per_group, k_d, k_h, k_w = weight.shape
    C_out = C_in_w * C_out_per_group * groups // C_in_w  # This should be groups * C_out_per_group
    
    # Compute output dimensions
    D_out = (D_in - 1) * stride[0] - 2 * padding[0] + (k_d - 1) * stride[0] + output_padding[0] + 1
    H_out = (H_in - 1) * stride[1] - 2 * padding[1] + (k_h - 1) * stride[1] + output_padding[1] + 1
    W_out = (W_in - 1) * stride[2] - 2 * padding[2] + (k_w - 1) * stride[2] + output_padding[2] + 1
    
    # Create output tensor
    out = torch.empty(N, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Configure kernel launch parameters
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_C_OUT = 16  # Tune this based on GPU
    BLOCK_SIZE_K = 8       # For input channel blocking
    
    grid = (
        N,  # batch dimension
        (C_out + BLOCK_SIZE_C_OUT - 1) // BLOCK_SIZE_C_OUT,  # output channels
        D_out,  # depth
        H_out,  # height
        W_out   # width
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        N, C_in, C_out, groups,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        k_d, k_h, k_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution operation with asymmetric input and kernel sizes,
    optimized with custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), 
                 padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, 
                 bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias
        
        # Create weight and bias parameters (same as nn.ConvTranspose3d)
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights (mimic PyTorch's default initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using custom Triton kernel.
        """
        return triton_conv_transpose3d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.output_padding, self.groups
        )