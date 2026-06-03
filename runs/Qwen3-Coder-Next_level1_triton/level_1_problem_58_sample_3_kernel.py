import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # Input tensor pointer (B, C_in, D_in, H_in, W_in)
    w_ptr,  # Weight tensor pointer (C_in, C_out, D_k, H_k, W_k)
    b_ptr,  # Bias tensor pointer (C_out,) or None
    out_ptr,  # Output tensor pointer (B, C_out, D_out, H_out, W_out)
    B, C_in, D_in, H_in, W_in,  # Input dimensions
    C_out, D_out, H_out, W_out,  # Output dimensions
    D_k, H_k, W_k,  # Kernel dimensions
    stride_d, stride_h, stride_w,  # Stride
    pad_d, pad_h, pad_w,  # Padding
    out_pad_d, out_pad_h, out_pad_w,  # Output padding
    BLOCK_SIZE_C_in: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C_out: tl.constexpr,
):
    # Get program IDs
    batch_id = tl.program_id(0)
    c_out_id = tl.program_id(1)
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)
    
    # Calculate output position
    out_idx = batch_id * (C_out * D_out * H_out * W_out) + \
              c_out_id * (D_out * H_out * W_out) + \
              d_out * (H_out * W_out) + \
              h_out * W_out + \
              w_out
    
    # Compute accumulation for this output position
    acc = 0.0
    
    # Bias contribution if bias exists
    if b_ptr is not None:
        acc = tl.load(b_ptr + c_out_id)
    
    # Iterate over input channels and kernel positions
    for c_in in range(C_in):
        for kd in range(D_k):
            for kh in range(H_k):
                for kw in range(W_k):
                    # Calculate corresponding input position
                    d_in = (d_out - kd + pad_d) // stride_d
                    h_in = (h_out - kh + pad_h) // stride_h
                    w_in = (w_out - kw + pad_w) // stride_w
                    
                    # Check if input position is valid
                    if (d_out - kd + pad_d) % stride_d == 0 and \
                       (h_out - kh + pad_h) % stride_h == 0 and \
                       (w_out - kw + pad_w) % stride_w == 0 and \
                       0 <= d_in < D_in and \
                       0 <= h_in < H_in and \
                       0 <= w_in < W_in:
                        
                        # Calculate input index
                        x_idx = batch_id * (C_in * D_in * H_in * W_in) + \
                                c_in * (D_in * H_in * W_in) + \
                                d_in * (H_in * W_in) + \
                                h_in * W_in + \
                                w_in
                        
                        # Calculate weight index
                        w_idx = c_in * (C_out * D_k * H_k * W_k) + \
                                c_out_id * (D_k * H_k * W_k) + \
                                kd * (H_k * W_k) + \
                                kh * W_k + \
                                kw
                        
                        # Load and multiply
                        x_val = tl.load(x_ptr + x_idx)
                        w_val = tl.load(w_ptr + w_idx)
                        acc += x_val * w_val
    
    # Store result
    tl.store(out_ptr + out_idx, acc)


class TritonConvTranspose3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, 
                stride, padding, output_padding, groups):
        # Extract parameters from inputs
        B, C_in, D_in, H_in, W_in = x.shape
        _, C_out, D_k, H_k, W_k = weight.shape
        stride_d, stride_h, stride_w = stride
        pad_d, pad_h, pad_w = padding
        out_pad_d, out_pad_h, out_pad_w = output_padding
        
        # Calculate output dimensions
        D_out = (D_in - 1) * stride_d - 2 * pad_d + D_k + out_pad_d
        H_out = (H_in - 1) * stride_h - 2 * pad_h + H_k + out_pad_h
        W_out = (W_in - 1) * stride_w - 2 * pad_w + W_k + out_pad_w
        
        # Create output tensor
        out = torch.empty(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
        
        # Configure kernel launch parameters
        grid = (B, C_out, D_out, H_out, W_out)
        
        # Launch kernel (using small block sizes for simplicity, can be optimized further)
        conv_transpose3d_kernel[grid](
            x, weight, bias, out,
            B, C_in, D_in, H_in, W_in,
            C_out, D_out, H_out, W_out,
            D_k, H_k, W_k,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            out_pad_d, out_pad_h, out_pad_w,
            BLOCK_SIZE_C_in=1,
            BLOCK_SIZE_D=1,
            BLOCK_SIZE_H=1,
            BLOCK_SIZE_W=1,
            BLOCK_SIZE_C_out=1
        )
        
        ctx.save_for_backward(x, weight)
        ctx.conv_params = (stride, padding, output_padding)
        
        return out


def triton_conv_transpose3d(x, weight, bias=None, stride=(1,1,1), 
                            padding=(0,0,0), output_padding=(0,0,0), groups=1):
    return TritonConvTranspose3d.apply(x, weight, bias, stride, padding, output_padding, groups)


class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution operation with asymmetric input and kernel sizes.
    Uses custom Triton kernel for optimization.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
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
        
        # Initialize weights and bias (same as nn.ConvTranspose3d)
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, *kernel_size) / 
            (in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2]) ** 0.5
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Register buffers for parameters
        self.register_buffer('stride_buffer', torch.tensor(stride))
        self.register_buffer('padding_buffer', torch.tensor(padding))
        self.register_buffer('output_padding_buffer', torch.tensor(output_padding))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using Triton kernel.
        """
        return triton_conv_transpose3d(
            x, self.weight, self.bias, 
            tuple(self.stride_buffer.tolist()),
            tuple(self.padding_buffer.tolist()),
            tuple(self.output_padding_buffer.tolist()),
            self.groups
        )