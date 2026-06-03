import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    # Pointers to tensors
    x_ptr,  # Input tensor: (batch, in_channels, D_in, H_in, W_in)
    w_ptr,  # Weight tensor: (in_channels, out_channels, D_k, H_k, W_k)
    b_ptr,  # Bias tensor: (out_channels,) or None
    out_ptr,  # Output tensor: (batch, out_channels, D_out, H_out, W_out)
    # Tensor dimensions
    batch_size, in_channels, out_channels,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    D_k, H_k, W_k,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_pad_d, out_pad_h, out_pad_w,
    # Block sizes for tiling
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for input channels
    BLOCK_SIZE_KD: tl.constexpr,  # Block size for kernel depth
    BLOCK_SIZE_KH: tl.constexpr,  # Block size for kernel height
    BLOCK_SIZE_KW: tl.constexpr,  # Block size for kernel width
):
    # Get program IDs
    pid_b = tl.program_id(0)  # batch index
    pid_d = tl.program_id(1)  # output depth index
    pid_h = tl.program_id(2)  # output height index
    pid_w = tl.program_id(3)  # output width index
    pid_c = tl.program_id(4)  # output channel block index

    # Compute output channel range for this block
    out_c_start = pid_c * BLOCK_SIZE_M
    out_c_offsets = out_c_start + tl.arange(0, BLOCK_SIZE_M)
    out_c_mask = out_c_offsets < out_channels

    # Initialize accumulator for output values
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    # Iterate over input channels
    for ic in range(in_channels):
        # Compute corresponding input positions
        in_d = pid_d * stride_d - pad_d + ic * 0  # ic is just a placeholder here
        in_h = pid_h * stride_h - pad_h
        in_w = pid_w * stride_w - pad_w

        # Compute kernel offset positions
        for kd in range(D_k):
            d_offset = pid_d * stride_d - pad_d + kd
            if d_offset >= 0 and d_offset < D_in:
                in_d = d_offset
                for kh in range(H_k):
                    h_offset = pid_h * stride_h - pad_h + kh
                    if h_offset >= 0 and h_offset < H_in:
                        in_h = h_offset
                        for kw in range(W_k):
                            w_offset = pid_w * stride_w - pad_w + kw
                            if w_offset >= 0 and w_offset < W_in:
                                in_w = w_offset

                                # Load input value
                                x_idx = pid_b * (in_channels * D_in * H_in * W_in) + \
                                        ic * (D_in * H_in * W_in) + \
                                        in_d * (H_in * W_in) + \
                                        in_h * W_in + in_w
                                x_val = tl.load(x_ptr + x_idx, mask=True)

                                # Load weight value
                                w_idx = ic * (out_channels * D_k * H_k * W_k) + \
                                        pid_c * BLOCK_SIZE_M * (D_k * H_k * W_k) + \
                                        kd * (H_k * W_k) + \
                                        kh * W_k + kw
                                w_val = tl.load(w_ptr + w_idx, mask=out_c_mask, other=0.0)

                                # Accumulate
                                acc += x_val * w_val

    # Add bias if present
    if b_ptr is not None:
        b_idx = out_c_offsets
        b_val = tl.load(b_ptr + b_idx, mask=out_c_mask, other=0.0)
        acc += b_val

    # Store result
    out_idx = pid_b * (out_channels * D_out * H_out * W_out) + \
              out_c_offsets * (D_out * H_out * W_out) + \
              pid_d * (H_out * W_out) + \
              pid_h * W_out + pid_w
    tl.store(out_ptr + out_idx, acc.to(x_ptr.dtype.element_ty), mask=out_c_mask)


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: tuple = (1, 1, 1),
    padding: tuple = (0, 0, 0),
    output_padding: tuple = (0, 0, 0),
    groups: int = 1,
):
    """
    Performs transposed 3D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch, in_channels, D_in, H_in, W_in)
        weight: Weight tensor of shape (in_channels, out_channels, D_k, H_k, W_k)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride tuple (stride_d, stride_h, stride_w)
        padding: Padding tuple (pad_d, pad_h, pad_w)
        output_padding: Output padding tuple (out_pad_d, out_pad_h, out_pad_w)
        groups: Number of groups (should be 1 for this implementation)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    assert groups == 1, "This implementation only supports groups=1"
    
    # Ensure contiguous tensors
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Get dimensions
    batch_size, in_channels, D_in, H_in, W_in = x.shape
    _, out_channels, D_k, H_k, W_k = weight.shape
    
    # Calculate output dimensions (same as PyTorch's ConvTranspose3d)
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    out_pad_d, out_pad_h, out_pad_w = output_padding
    
    D_out = (D_in - 1) * stride_d - 2 * pad_d + D_k + out_pad_d
    H_out = (H_in - 1) * stride_h - 2 * pad_h + H_k + out_pad_h
    W_out = (W_in - 1) * stride_w - 2 * pad_w + W_k + out_pad_w
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, D_out, H_out, W_out, 
                     dtype=x.dtype, device=x.device)
    
    # Define block sizes for tiling
    BLOCK_SIZE_M = 8  # Output channel block size
    BLOCK_SIZE_N = 32  # Input channel block size (not used in this kernel but kept for extensibility)
    BLOCK_SIZE_KD = 3  # Kernel depth block size
    BLOCK_SIZE_KH = 5  # Kernel height block size
    BLOCK_SIZE_KW = 7  # Kernel width block size
    
    # Define grid dimensions
    grid = lambda meta: (
        batch_size,
        D_out,
        H_out,
        W_out,
        (out_channels + meta['BLOCK_SIZE_M'] - 1) // meta['BLOCK_SIZE_M']
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        D_k, H_k, W_k,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_pad_d, out_pad_h, out_pad_w,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_KD=BLOCK_SIZE_KD,
        BLOCK_SIZE_KH=BLOCK_SIZE_KH,
        BLOCK_SIZE_KW=BLOCK_SIZE_KW,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for transposed 3D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, 
                 bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same way as original
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Create the weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights and bias."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized transposed 3D convolution.
        """
        return triton_conv_transpose3d(
            x, 
            self.weight, 
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )