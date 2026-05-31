import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
    batch_size, depth, width, height,
    depth_out, width_out, height_out,
    BLOCK_D: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_H: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_KD: tl.constexpr, BLOCK_KW: tl.constexpr, BLOCK_KH: tl.constexpr,
):
    # Program ID corresponds to a specific output element
    pid = tl.program_id(0)
    
    # Calculate output indices
    oh = pid % height_out
    ow = (pid // height_out) % width_out
    od = (pid // (height_out * width_out)) % depth_out
    oc = (pid // (height_out * width_out * depth_out)) % out_channels
    b = pid // (height_out * width_out * depth_out * out_channels)
    
    # Calculate input region bounds
    d_start = od * stride - padding
    w_start = ow * stride - padding
    h_start = oh * stride - padding
    
    # Load weights for the current output channel
    # Weights shape: (out_channels, in_channels/groups, kernel_size, kernel_size, kernel_size)
    # We need weights for output channel oc
    w_ptr_oc = w_ptr + oc * in_channels // groups * kernel_size * kernel_size * kernel_size
    
    # Load bias if present
    bias_val = 0.0
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + oc)
    
    # Initialize accumulator
    acc = 0.0
    
    # Iterate over groups, input channels, and kernel dimensions
    for g in range(groups):
        for c in range(in_channels // groups):
            for kd in range(kernel_size):
                for kw in range(kernel_size):
                    for kh in range(kernel_size):
                        # Calculate input coordinates
                        d = d_start + kd * dilation
                        w = w_start + kw * dilation
                        h = h_start + kh * dilation
                        
                        # Check bounds
                        if d >= 0 and d < depth and w >= 0 and w < width and h >= 0 and h < height:
                            # Load input value
                            x_ptr_idx = b * in_channels * depth * width * height + \
                                        (c + g * (in_channels // groups)) * depth * width * height + \
                                        d * width * height + w * height + h
                            x_val = tl.load(x_ptr + x_ptr_idx)
                            
                            # Load weight value
                            w_ptr_idx = c * kernel_size * kernel_size * kernel_size + \
                                        kd * kernel_size * kernel_size + \
                                        kw * kernel_size + kh
                            w_val = tl.load(w_ptr_oc + w_ptr_idx)
                            
                            # Accumulate product
                            acc += x_val * w_val
    
    # Add bias and store result
    out_val = acc + bias_val
    out_ptr_idx = b * out_channels * depth_out * width_out * height_out + \
                  oc * depth_out * width_out * height_out + \
                  od * width_out * height_out + \
                  ow * height_out + oh
    tl.store(out_ptr + out_ptr_idx, out_val)


def triton_conv3d(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
                  in_channels: int, out_channels: int, kernel_size: int,
                  stride: int = 1, padding: int = 0, dilation: int = 1,
                  groups: int = 1, bias: bool = False) -> torch.Tensor:
    """
    Triton implementation of 3D convolution.
    """
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    if bias:
        b = b.contiguous()
    
    # Calculate output dimensions
    depth_out = (x.shape[2] + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    width_out = (x.shape[3] + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    height_out = (x.shape[4] + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((x.shape[0], out_channels, depth_out, width_out, height_out), dtype=x.dtype, device=x.device)
    
    # Grid configuration: one program per output element
    grid_size = x.shape[0] * out_channels * depth_out * width_out * height_out
    
    # Launch kernel
    conv3d_kernel[grid_size](
        x, w, b if bias else None, out,
        in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
        x.shape[0], x.shape[2], x.shape[3], x.shape[4],
        depth_out, width_out, height_out,
        BLOCK_D=1, BLOCK_W=1, BLOCK_H=1,
        BLOCK_C=1, BLOCK_KD=1, BLOCK_KW=1, BLOCK_KH=1,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized 3D convolution model using custom Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using Triton kernel.
        """
        w = self.conv3d.weight
        b = self.conv3d.bias if self.bias else None
        return triton_conv3d(
            x, w, b,
            self.conv3d.in_channels, self.conv3d.out_channels, self.kernel_size,
            self.stride, self.padding, self.dilation, self.groups, self.bias
        )