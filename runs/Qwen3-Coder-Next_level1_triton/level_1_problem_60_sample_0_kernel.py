import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr,  # Input tensor pointer (N, C_in, W, H, D)
    w_ptr,  # Weight tensor pointer (C_out, C_in, k_w, k_h, k_d)
    b_ptr,  # Bias tensor pointer (C_out,)
    out_ptr,  # Output tensor pointer (N, C_out, W_out, H_out, D_out)
    n_elements,  # Total number of output elements
    N: tl.constexpr,  # Batch size
    C_in: tl.constexpr,  # Input channels
    C_out: tl.constexpr,  # Output channels
    W: tl.constexpr,  # Input width
    H: tl.constexpr,  # Input height
    D: tl.constexpr,  # Input depth
    W_out: tl.constexpr,  # Output width
    H_out: tl.constexpr,  # Output height
    D_out: tl.constexpr,  # Output depth
    k_w: tl.constexpr,  # Kernel width
    k_h: tl.constexpr,  # Kernel height
    k_d: tl.constexpr,  # Kernel depth
    stride_w: tl.constexpr,  # Stride in width
    stride_h: tl.constexpr,  # Stride in height
    stride_d: tl.constexpr,  # Stride in depth
    pad_w: tl.constexpr,  # Padding in width
    pad_h: tl.constexpr,  # Padding in height
    pad_d: tl.constexpr,  # Padding in depth
    dil_w: tl.constexpr,  # Dilation in width
    dil_h: tl.constexpr,  # Dilation in height
    dil_d: tl.constexpr,  # Dilation in depth
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate output tensor index
    out_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = out_idx < n_elements
    
    # Extract coordinates from flattened index
    # Layout: N, C_out, W_out, H_out, D_out
    d_out_idx = out_idx % D_out
    h_out_idx = (out_idx // D_out) % H_out
    w_out_idx = (out_idx // (D_out * H_out)) % W_out
    c_out_idx = (out_idx // (D_out * H_out * W_out)) % C_out
    n_idx = out_idx // (D_out * H_out * W_out * C_out)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE], tl.float32)
    
    # Add bias if available
    if b_ptr is not None:
        bias = tl.load(b_ptr + c_out_idx, mask=mask)
        acc += bias.to(tl.float32)
    
    # Convolution: iterate over input channels and kernel dimensions
    for c_in in range(C_in):
        for kw in range(k_w):
            for kh in range(k_h):
                for kd in range(k_d):
                    # Calculate input coordinates with padding and dilation
                    w_in = w_out_idx * stride_w + kw * dil_w - pad_w
                    h_in = h_out_idx * stride_h + kh * dil_h - pad_h
                    d_in = d_out_idx * stride_d + kd * dil_d - pad_d
                    
                    # Check bounds
                    in_bounds = (w_in >= 0) & (w_in < W) & (h_in >= 0) & (h_in < H) & (d_in >= 0) & (d_in < D)
                    
                    # Calculate input and weight indices
                    x_offset = ((n_idx * C_in + c_in) * W * H * D + 
                               w_in * H * D + h_in * D + d_in)
                    w_offset = ((c_out_idx * C_in + c_in) * k_w * k_h * k_d + 
                               kw * k_h * k_d + kh * k_d + kd)
                    
                    # Load values with bounds checking
                    x_val = tl.load(x_ptr + x_offset, mask=in_bounds & mask, other=0.0)
                    w_val = tl.load(w_ptr + w_offset, mask=mask, other=0.0)
                    
                    # Accumulate
                    acc += x_val * w_val
    
    # Store result
    tl.store(out_ptr + out_idx, acc, mask=mask)


def triton_conv3d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                  stride=1, padding=0, dilation=1, groups=1):
    """
    Triton implementation of 3D convolution.
    
    Args:
        x: Input tensor of shape (N, C_in, W, H, D)
        weight: Weight tensor of shape (C_out, C_in, k_w, k_h, k_d)
        bias: Optional bias tensor of shape (C_out,)
        stride: Stride for convolution
        padding: Padding for convolution
        dilation: Dilation for convolution
        groups: Groups for convolution (must be 1 for this implementation)
    
    Returns:
        Output tensor of shape (N, C_out, W_out, H_out, D_out)
    """
    assert groups == 1, "Group convolution not supported in this implementation"
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    
    # Ensure contiguous tensors
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Extract dimensions
    N, C_in, W, H, D = x.shape
    C_out, _, k_w, k_h, k_d = weight.shape
    
    # Calculate output dimensions
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)
    
    stride_w, stride_h, stride_d = stride
    pad_w, pad_h, pad_d = padding
    dil_w, dil_h, dil_d = dilation
    
    W_out = (W + 2 * pad_w - dil_w * (k_w - 1) - 1) // stride_w + 1
    H_out = (H + 2 * pad_h - dil_h * (k_h - 1) - 1) // stride_h + 1
    D_out = (D + 2 * pad_d - dil_d * (k_d - 1) - 1) // stride_d + 1
    
    # Prepare output tensor
    out = torch.empty((N, C_out, W_out, H_out, D_out), dtype=x.dtype, device=x.device)
    
    # Calculate total output elements
    n_elements = N * C_out * W_out * H_out * D_out
    BLOCK_SIZE = 256
    
    # Determine grid size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    conv3d_kernel[grid](
        x, weight, bias, out,
        n_elements,
        N=N, C_in=C_in, C_out=C_out,
        W=W, H=H, D=D,
        W_out=W_out, H_out=H_out, D_out=D_out,
        k_w=k_w, k_h=k_h, k_d=k_d,
        stride_w=stride_w, stride_h=stride_h, stride_d=stride_d,
        pad_w=pad_w, pad_h=pad_h, pad_d=pad_d,
        dil_w=dil_w, dil_h=dil_h, dil_d=dil_d,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized 3D convolution model using custom Triton kernel.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (kernel_width, kernel_height, kernel_depth).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters for forward pass
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 
                                              self.kernel_size[0], self.kernel_size[1], 
                                              self.kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, width, height, depth).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, width_out, height_out, depth_out).
        """
        return triton_conv3d(x, self.weight, self.bias,
                            stride=self.stride, padding=self.padding,
                            dilation=self.dilation, groups=self.groups)