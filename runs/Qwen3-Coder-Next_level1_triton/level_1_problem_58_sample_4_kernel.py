import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    # Pointers to input/output tensors
    X,  # Input tensor: (B, C_in, D_in, H_in, W_in)
    W,  # Weight tensor: (C_in, C_out // groups, D_k, H_k, W_k)
    B,  # Bias tensor: (C_out,) or None
    Y,  # Output tensor: (B, C_out, D_out, H_out, W_out)
    # Tensor dimensions
    C_in, C_out, groups,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    D_k, H_k, W_k,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_pad_d, out_pad_h, out_pad_w,
    # Block sizes for tiling
    BLOCK_SIZE_C_in: tl.constexpr,
    BLOCK_SIZE_C_out: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Calculate output position
    out_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create mask for valid indices
    mask_d = out_d < D_out
    mask_h = out_h < H_out
    mask_w = out_w < W_out
    mask = mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]
    
    # Compute output offset for this thread
    out_offset = (
        pid_batch * (C_out * D_out * H_out * W_out) +
        pid_c_out * (D_out * H_out * W_out) +
        out_d[:, None, None] * (H_out * W_out) +
        out_h[None, :, None] * W_out +
        out_w[None, None, :]
    )
    out_offset = tl.reshape(out_offset, [BLOCK_SIZE_D * BLOCK_SIZE_H * BLOCK_SIZE_W])
    mask_flat = tl.reshape(mask, [BLOCK_SIZE_D * BLOCK_SIZE_H * BLOCK_SIZE_W])
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE_D * BLOCK_SIZE_H * BLOCK_SIZE_W], dtype=tl.float32)
    
    # Compute input position from output position (transposed convolution property)
    # For transposed convolution: out_pos = in_pos * stride + (kernel_pos - 1 - pad)
    # So: in_pos = (out_pos - (kernel_pos - 1 - pad)) // stride
    
    # Iterate over input channels and kernel positions
    for c_in_start in range(0, C_in, BLOCK_SIZE_C_in):
        c_in_range = c_in_start + tl.arange(0, BLOCK_SIZE_C_in)
        c_in_mask = c_in_range < C_in
        
        for k_d in range(D_k):
            for k_h in range(H_k):
                for k_w in range(W_k):
                    # Compute corresponding input position
                    in_d = (out_d - (k_d - pad_d)) // stride_d
                    in_h = (out_h - (k_h - pad_h)) // stride_h
                    in_w = (out_w - (k_w - pad_w)) // stride_w
                    
                    # Check if input position is valid
                    valid_mask = (
                        (in_d >= 0) & (in_d < D_in) & 
                        (in_h >= 0) & (in_h < H_in) &
                        (in_w >= 0) & (in_w < W_in)
                    )
                    
                    # Compute input offset
                    in_offset = (
                        pid_batch * (C_in * D_in * H_in * W_in) +
                        c_in_range[:, None, None, None] * (D_in * H_in * W_in) +
                        in_d[None, :, None, None] * (H_in * W_in) +
                        in_h[None, None, :, None] * W_in +
                        in_w[None, None, None, :]
                    )
                    
                    # Compute weight offset: W[c_in, c_out_group, k_d, k_h, k_w]
                    c_out_group = pid_c_out // (C_out // groups)
                    weight_offset = (
                        c_in_range[:, None, None, None] * (C_out * D_k * H_k * W_k) +
                        c_out_group * (C_out // groups * D_k * H_k * W_k) +
                        k_d * (C_out * H_k * W_k) +
                        pid_c_out * (H_k * W_k) +
                        k_h * (C_out * W_k) +
                        pid_c_out * W_k +
                        k_w
                    )
                    
                    # Load input and weight
                    X_val = tl.load(X + in_offset, mask=valid_mask[:, :, :, :] & c_in_mask[:, None, None, None])
                    W_val = tl.load(W + weight_offset, mask=c_in_mask[:, None, None, None])
                    
                    # Accumulate
                    acc += tl.sum(X_val * W_val, axis=0)
    
    # Add bias if present
    if B is not None:
        bias = tl.load(B + pid_c_out)
        acc += bias
    
    # Store result
    tl.store(Y + out_offset, acc, mask=mask_flat)


class TritonConvTranspose3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, output_padding, groups):
        # Get dimensions
        B, C_in, D_in, H_in, W_in = x.shape
        C_in_w, C_out_group, D_k, H_k, W_k = weight.shape
        C_out = C_in_w * C_out_group
        
        # Compute output dimensions
        D_out = (D_in - 1) * stride[0] - 2 * padding[0] + (D_k - 1) + output_padding[0] + 1
        H_out = (H_in - 1) * stride[1] - 2 * padding[1] + (H_k - 1) + output_padding[1] + 1
        W_out = (W_in - 1) * stride[2] - 2 * padding[2] + (W_k - 1) + output_padding[2] + 1
        
        # Allocate output tensor
        y = torch.empty((B, C_out, D_out, H_out, W_out), dtype=x.dtype, device=x.device)
        
        # Kernel configuration
        BLOCK_SIZE_C_in = min(32, C_in)
        BLOCK_SIZE_C_out = min(32, C_out)
        BLOCK_SIZE_D = min(4, D_out)
        BLOCK_SIZE_H = min(4, H_out)
        BLOCK_SIZE_W = min(4, W_out)
        
        # Grid configuration
        grid = (
            B,  # batch dimension
            triton.cdiv(C_out, BLOCK_SIZE_C_out),  # output channels
            triton.cdiv(D_out, BLOCK_SIZE_D),  # depth
            triton.cdiv(H_out, BLOCK_SIZE_H),  # height
            triton.cdiv(W_out, BLOCK_SIZE_W),  # width
        )
        
        # Launch kernel
        conv_transpose3d_kernel[grid](
            x, weight, bias, y,
            C_in, C_out, groups,
            D_in, H_in, W_in,
            D_out, H_out, W_out,
            D_k, H_k, W_k,
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2],
            output_padding[0], output_padding[1], output_padding[2],
            BLOCK_SIZE_C_in=BLOCK_SIZE_C_in,
            BLOCK_SIZE_C_out=BLOCK_SIZE_C_out,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_W=BLOCK_SIZE_W,
        )
        
        ctx.save_for_backward(x, weight)
        ctx.params = (stride, padding, output_padding, groups)
        
        return y


def triton_conv_transpose3d(x, weight, bias=None, stride=(1, 1, 1), 
                           padding=(0, 0, 0), output_padding=(0, 0, 0), groups=1):
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using custom Triton kernel.
        """
        return triton_conv_transpose3d(
            x, self.weight, self.bias, 
            self.stride, self.padding, self.output_padding, self.groups
        )