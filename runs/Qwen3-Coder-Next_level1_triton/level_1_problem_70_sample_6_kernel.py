import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    # Pointers to tensors
    x_ptr,  # Input tensor (B, C_in, D, H, W)
    w_ptr,  # Weight tensor (C_in, C_out, K, K, K)
    b_ptr,  # Bias tensor (C_out,) or None
    out_ptr,  # Output tensor (B, C_out, D_out, H_out, W_out)
    # Tensor dimensions
    B, C_in, D, H, W,
    C_out, K, 
    D_out, H_out, W_out,
    stride_d, stride_h, stride_w,
    output_padding_d, output_padding_h, output_padding_w,
    # Block sizes for tiling
    BLOCK_B: tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Get program IDs for batch, output channels, and spatial positions
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Calculate starting positions in output
    out_d = pid_d * BLOCK_D
    out_h = pid_h * BLOCK_H
    out_w = pid_w * BLOCK_W
    
    # Initialize accumulator for the output
    acc = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Iterate over input channels and kernel positions
    for c_in in range(C_in):
        for k_d in range(K):
            # Calculate corresponding input position
            in_d = out_d + k_d - 1  # because stride=1 and padding=0 for transposed conv
            if in_d >= 0 and in_d < D:
                for k_h in range(K):
                    in_h = out_h + k_h - 1
                    if in_h >= 0 and in_h < H:
                        for k_w in range(K):
                            in_w = out_w + k_w - 1
                            if in_w >= 0 and in_w < W:
                                # Calculate pointers
                                x_offset = ((pid_b * C_in * D * H * W) + 
                                           (c_in * D * H * W) + 
                                           (in_d * H * W) + 
                                           (in_h * W) + 
                                           in_w)
                                w_offset = ((c_in * C_out * K * K * K) + 
                                           (pid_c_out * K * K * K) + 
                                           (k_d * K * K) + 
                                           (k_h * K) + 
                                           k_w)
                                
                                # Load values
                                x_val = tl.load(x_ptr + x_offset)
                                w_val = tl.load(w_ptr + w_offset)
                                
                                # Accumulate
                                acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_c_out)
        acc += bias
    
    # Store results
    for d_offset in range(BLOCK_D):
        for h_offset in range(BLOCK_H):
            for w_offset in range(BLOCK_W):
                d = out_d + d_offset
                h = out_h + h_offset
                w = out_w + w_offset
                
                if (d < D_out and h < H_out and w < W_out and
                    pid_b < B):
                    out_offset = ((pid_b * C_out * D_out * H_out * W_out) +
                                 (pid_c_out * D_out * H_out * W_out) +
                                 (d * H_out * W_out) +
                                 (h * W_out) +
                                 w)
                    tl.store(out_ptr + out_offset, acc[d_offset, h_offset, w_offset])


def triton_conv_transpose3d(x, weight, bias=None, stride=1, padding=0, output_padding=0, dilation=1, groups=1):
    """
    Triton implementation of ConvTranspose3d.
    Assumes stride=1, padding=0, dilation=1, groups=1 for simplicity.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Extract dimensions
    B, C_in, D, H, W = x.shape
    C_in_, C_out, K, K2, K3 = weight.shape
    assert C_in == C_in_, f"Channel mismatch: {C_in} vs {C_in_}"
    assert K == K2 == K3, "Kernel must be cubic"
    K = K
    
    # Calculate output shape
    D_out = (D - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1
    H_out = (H - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1
    W_out = (W - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1
    
    # Create output tensor
    out = torch.empty(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Configure grid
    # Use reasonable block sizes for the given dimensions
    BLOCK_B = 1
    BLOCK_C_OUT = 8
    BLOCK_D = 8
    BLOCK_H = 8
    BLOCK_W = 8
    BLOCK_K = 3
    
    # Grid dimensions
    grid = (
        (B + BLOCK_B - 1) // BLOCK_B,
        (C_out + BLOCK_C_OUT - 1) // BLOCK_C_OUT,
        (D_out + BLOCK_D - 1) // BLOCK_D,
        (H_out + BLOCK_H - 1) // BLOCK_H,
        (W_out + BLOCK_W - 1) // BLOCK_W,
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        B, C_in, D, H, W,
        C_out, K,
        D_out, H_out, W_out,
        stride, stride, stride,
        output_padding, output_padding, output_padding,
        BLOCK_B=BLOCK_B,
        BLOCK_C_OUT=BLOCK_C_OUT,
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        BLOCK_K=BLOCK_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for ConvTranspose3d.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, output_padding: int = 0, dilation: int = 1, groups: int = 1, 
                 bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the convolution layer with the same parameters
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), 
                                                stride=stride, padding=padding, output_padding=output_padding, 
                                                dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using Triton kernel instead of PyTorch implementation.
        """
        # Extract parameters from the original layer
        weight = self.conv_transpose3d.weight
        bias = self.conv_transpose3d.bias if self.conv_transpose3d.bias is not None else None
        
        # Use our Triton implementation
        return triton_conv_transpose3d(
            x, 
            weight, 
            bias,
            stride=self.conv_transpose3d.stride,
            padding=self.conv_transpose3d.padding,
            output_padding=self.conv_transpose3d.output_padding,
            dilation=self.conv_transpose3d.dilation,
            groups=self.conv_transpose3d.groups
        )