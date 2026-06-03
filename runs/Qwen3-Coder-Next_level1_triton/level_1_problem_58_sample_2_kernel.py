import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose3d_kernel(
    # Pointers to tensors
    x_ptr,           # Input tensor: (B, C_in, D_in, H_in, W_in)
    w_ptr,           # Weight tensor: (C_in, C_out, D_k, H_k, W_k)
    b_ptr,           # Bias tensor: (C_out,) - can be None
    out_ptr,         # Output tensor: (B, C_out, D_out, H_out, W_out)
    # Tensor dimensions
    B, C_in, D_in, H_in, W_in,
    C_out, D_k, H_k, W_k,
    D_out, H_out, W_out,
    # Strides
    stride_d, stride_h, stride_w,
    # Padding
    pad_d, pad_h, pad_w,
    # Output padding
    out_pad_d, out_pad_h, out_pad_w,
    # Block sizes
    BLOCK_C_IN: tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Calculate output position
    out_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    out_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    
    # Create masks for output dimensions
    mask_d = out_d < D_out
    mask_h = out_h < H_out
    mask_w = out_w < W_out
    mask_3d = mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for offset_c_in in range(0, C_in, BLOCK_C_IN):
        c_in_range = offset_c_in + tl.arange(0, BLOCK_C_IN)
        mask_c_in = c_in_range < C_in
        
        # Process each kernel position
        for kd in range(D_k):
            for kh in range(H_k):
                for kw in range(W_k):
                    # Calculate corresponding input position
                    in_d = (out_d - kd + pad_d) // stride_d
                    in_h = (out_h - kh + pad_h) // stride_h
                    in_w = (out_w - kw + pad_w) // stride_w
                    
                    # Check if input positions are valid
                    mask_in_d = (in_d >= 0) & (in_d < D_in) & (out_d - kd + pad_d == in_d * stride_d)
                    mask_in_h = (in_h >= 0) & (in_h < H_in) & (out_h - kh + pad_h == in_h * stride_h)
                    mask_in_w = (in_w >= 0) & (in_w < W_in) & (out_w - kw + pad_w == in_w * stride_w)
                    
                    # Create input mask
                    mask_in = mask_in_d[:, None, None] & mask_in_h[None, :, None] & mask_in_w[None, None, :]
                    mask_combined = mask_3d & mask_in & mask_c_in[None, None, :]
                    
                    # Load input values
                    in_offsets = (
                        pid_b * (C_in * D_in * H_in * W_in) +
                        c_in_range[None, None, :] * (D_in * H_in * W_in) +
                        in_d[:, None, None] * (H_in * W_in) +
                        in_h[None, :, None] * W_in +
                        in_w[None, None, :]
                    )
                    x_vals = tl.load(x_ptr + in_offsets, mask=mask_combined, other=0.0)
                    
                    # Load weight values
                    w_offsets = (
                        c_in_range[:, None, None] * (C_out * D_k * H_k * W_k) +
                        pid_c_out * (D_k * H_k * W_k) +
                        kd * (H_k * W_k) +
                        kh * W_k +
                        kw
                    )
                    w_vals = tl.load(w_ptr + w_offsets, mask=mask_c_in[:, None, None], other=0.0)
                    
                    # Accumulate: x * w
                    acc += tl.sum(x_vals * w_vals, axis=2)
    
    # Add bias if available
    if b_ptr is not None:
        bias_offsets = pid_c_out
        bias_val = tl.load(b_ptr + bias_offsets)
        acc += bias_val
    
    # Store result
    out_offsets = (
        pid_b * (C_out * D_out * H_out * W_out) +
        pid_c_out * (D_out * H_out * W_out) +
        out_d[:, None, None] * (H_out * W_out) +
        out_h[None, :, None] * W_out +
        out_w[None, None, :]
    )
    tl.store(out_ptr + out_offsets, acc.to(x_ptr.dtype.element_ty), mask=mask_3d)


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: tuple = (1, 1, 1),
    padding: tuple = (0, 0, 0),
    output_padding: tuple = (0, 0, 0),
    groups: int = 1
) -> torch.Tensor:
    """
    Triton implementation of ConvTranspose3d
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    assert groups == 1, "Only groups=1 is supported in this Triton kernel."
    
    # Ensure contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C_in, D_in, H_in, W_in = x.shape
    C_in_w, C_out, D_k, H_k, W_k = weight.shape
    
    # Validate channel dimensions
    assert C_in == C_in_w, f"Input channels mismatch: {C_in} vs {C_in_w}"
    
    # Calculate output dimensions
    D_out = (D_in - 1) * stride[0] - 2 * padding[0] + D_k + output_padding[0]
    H_out = (H_in - 1) * stride[1] - 2 * padding[1] + H_k + output_padding[1]
    W_out = (W_in - 1) * stride[2] - 2 * padding[2] + W_k + output_padding[2]
    
    # Create output tensor
    out = torch.empty(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Set block sizes (tunable parameters)
    BLOCK_C_IN = min(32, C_in)
    BLOCK_C_OUT = 1
    BLOCK_D = min(4, D_out)
    BLOCK_H = min(4, H_out)
    BLOCK_W = min(8, W_out)
    
    # Calculate grid dimensions
    grid = (
        B,                          # batch dimension
        C_out,                      # output channels
        (D_out + BLOCK_D - 1) // BLOCK_D,
        (H_out + BLOCK_H - 1) // BLOCK_H,
        (W_out + BLOCK_W - 1) // BLOCK_W
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        B, C_in, D_in, H_in, W_in,
        C_out, D_k, H_k, W_k,
        D_out, H_out, W_out,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        BLOCK_C_IN=BLOCK_C_IN,
        BLOCK_C_OUT=BLOCK_C_OUT,
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the transposed 3D convolution model using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Create the ConvTranspose3d layer to get the weights and bias
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, 
            output_padding=output_padding, 
            groups=groups, bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using Triton kernel.
        """
        # Use our optimized Triton implementation
        return triton_conv_transpose3d(
            x,
            self.conv_transpose3d.weight,
            self.conv_transpose3d.bias,
            stride=self.conv_transpose3d.stride,
            padding=self.conv_transpose3d.padding,
            output_padding=self.conv_transpose3d.output_padding,
            groups=self.conv_transpose3d.groups
        )