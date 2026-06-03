import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # Input tensor pointer (B, C_in, D_in, H_in, W_in)
    w_ptr,  # Weight tensor pointer (C_in, C_out // G, Kd, Kh, Kw)
    bias_ptr,  # Bias tensor pointer (C_out,) or None
    out_ptr,  # Output tensor pointer (B, C_out, D_out, H_out, W_out)
    B: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    G: tl.constexpr,
    D_in: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    D_out: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    Kd: tl.constexpr,
    Kh: tl.constexpr,
    Kw: tl.constexpr,
    S_d: tl.constexpr,
    S_h: tl.constexpr,
    S_w: tl.constexpr,
    P_d: tl.constexpr,
    P_h: tl.constexpr,
    P_w: tl.constexpr,
    OP_d: tl.constexpr,
    OP_h: tl.constexpr,
    OP_w: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    c_out_block = tl.program_id(1)
    
    # Calculate output position
    # Each thread handles one output element
    c_out_idx = c_out_block * BLOCK_SIZE_C_OUT + tl.arange(0, BLOCK_SIZE_C_OUT)
    
    # Output tensor indices
    d_out = tl.arange(0, 1)  # We'll iterate through depth
    h_out = tl.arange(0, 1)  # We'll iterate through height
    w_out = tl.arange(0, 1)  # We'll iterate through width
    
    # Create mask for c_out_idx
    c_out_mask = c_out_idx < C_out
    
    # Calculate corresponding input indices
    # For transposed convolution: D_in = (D_out - 1 - OP_d + 2*P_d - Kd) // S_d + 1
    # So: D_out = (D_in - 1) * S_d + OP_d + Kd - 2*P_d
    # We compute the output indices directly
    
    # For each output position, we need to accumulate contributions from input and kernel
    # The kernel index is computed from output and input positions
    
    # Process in a loop over depth, height, width for simplicity
    for d in range(D_out):
        for h in range(H_out):
            for w in range(W_out):
                # Initialize accumulator for this output position
                acc = tl.zeros((BLOCK_SIZE_C_OUT,), dtype=tl.float32)
                
                # Calculate the corresponding input position range
                # For transposed convolution: input_d = (output_d - 1) * stride + kernel_d - padding - output_padding
                # But more precisely: input_d = (output_d - (Kd - 1) + P_d) // S_d
                # And kernel_d = (output_d - (Kd - 1) + P_d) % S_d + (Kd - 1) - ((output_d - (Kd - 1) + P_d) // S_d) * S_d
                
                # Actually, let's use the standard formula:
                # input_d = (output_d - OP_d - Kd + 1 + 2*P_d + S_d - 1) // S_d
                # This simplifies to: input_d = (output_d - OP_d + P_d) // S_d
                
                input_d = (d - OP_d + P_d) // S_d
                kernel_d = (d - OP_d + P_d) % S_d + (Kd - 1) - ((d - OP_d + P_d) // S_d) * S_d
                
                input_h = (h - OP_h + P_h) // S_h
                kernel_h = (h - OP_h + P_h) % S_h + (Kh - 1) - ((h - OP_h + P_h) // S_h) * S_h
                
                input_w = (w - OP_w + P_w) // S_w
                kernel_w = (w - OP_w + P_w) % S_w + (Kw - 1) - ((w - OP_w + P_w) // S_w) * S_w
                
                # Check bounds for input position
                if input_d >= 0 and input_d < D_in and input_h >= 0 and input_h < H_in and input_w >= 0 and input_w < W_in:
                    # Now accumulate over input channels and groups
                    for c_in_group_start in range(0, C_in, BLOCK_SIZE_C_IN):
                        c_in_group = c_in_group_start + tl.arange(0, BLOCK_SIZE_C_IN)
                        c_in_mask = c_in_group < C_in
                        
                        # Calculate group index for this input channel
                        group_idx = c_in_group // (C_in // G)
                        c_out_group_start = group_idx * (C_out // G)
                        
                        # Get input value
                        x_offset = batch_idx * (C_in * D_in * H_in * W_in) + \
                                  c_in_group * (D_in * H_in * W_in) + \
                                  input_d * (H_in * W_in) + \
                                  input_h * W_in + \
                                  input_w
                        x_vals = tl.load(x_ptr + x_offset, mask=c_in_mask[:, None], other=0.0)
                        
                        # Get weight value
                        # Weight shape: (C_in, C_out // G, Kd, Kh, Kw)
                        w_offset = c_in_group[:, None] * (C_out * Kd * Kh * Kw) + \
                                  (c_out_idx[None, :] - c_out_group_start[:, None]) * (Kd * Kh * Kw) + \
                                  kernel_d * (Kh * Kw) + \
                                  kernel_h * Kw + \
                                  kernel_w
                        w_vals = tl.load(w_ptr + w_offset, mask=c_in_mask[:, None] & c_out_mask[None, :], other=0.0)
                        
                        # Accumulate
                        acc += tl.sum(x_vals * w_vals, axis=0)
                
                # Add bias if available
                if bias_ptr is not None:
                    bias_vals = tl.load(bias_ptr + c_out_idx, mask=c_out_mask, other=0.0)
                    acc += bias_vals
                
                # Store result
                out_offset = batch_idx * (C_out * D_out * H_out * W_out) + \
                            c_out_idx * (D_out * H_out * W_out) + \
                            d * (H_out * W_out) + \
                            h * W_out + \
                            w
                tl.store(out_ptr + out_offset, acc.to(tl.float32), mask=c_out_mask)


def triton_conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups):
    """
    Performs 3D transposed convolution using Triton kernel.
    """
    B, C_in, D_in, H_in, W_in = x.shape
    C_in_, C_out_per_group, Kd, Kh, Kw = weight.shape
    C_out = C_in_ * C_out_per_group * groups // C_in_  # C_out = C_out_per_group * groups
    
    # Calculate output dimensions
    D_out = (D_in - 1) * stride[0] - 2 * padding[0] + Kd + output_padding[0]
    H_out = (H_in - 1) * stride[1] - 2 * padding[1] + Kh + output_padding[1]
    W_out = (W_in - 1) * stride[2] - 2 * padding[2] + Kw + output_padding[2]
    
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Create output tensor
    out = torch.empty(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Configure kernel launch parameters
    BLOCK_SIZE_B = 1
    BLOCK_SIZE_C_OUT = 16
    BLOCK_SIZE_C_IN = 8
    BLOCK_SIZE_K = 8
    
    grid = (B, triton.cdiv(C_out, BLOCK_SIZE_C_OUT))
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        B, C_in, C_out, groups,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        Kd, Kh, Kw,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
        BLOCK_SIZE_C_IN=BLOCK_SIZE_C_IN,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the 3D transposed convolution using custom Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
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
        
        # Create weight and bias parameters (same as ConvTranspose3d)
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D transposed convolution using Triton kernel.
        """
        # Check if we have CUDA
        if not x.is_cuda:
            # Fall back to PyTorch for non-CUDA
            return nn.functional.conv_transpose3d(x, self.weight, self.bias, 
                                                  self.stride, self.padding, 
                                                  self.output_padding, self.groups)
        
        # Use custom Triton kernel for CUDA
        return triton_conv_transpose3d(
            x, self.weight, self.bias, 
            self.stride, self.padding, 
            self.output_padding, self.groups
        )
    
    def extra_repr(self):
        return '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, output_padding={output_padding}, groups={groups}, bias={bias_flag}'.format(**self.__dict__)


# Import math for initialization
import math