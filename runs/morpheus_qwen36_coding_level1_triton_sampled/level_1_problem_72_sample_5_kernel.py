import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    bias_ptr,
    in_channels,
    out_channels,
    groups,
    kernel_size_d,
    kernel_size_h,
    kernel_size_w,
    stride_d,
    stride_h,
    stride_w,
    padding_d,
    padding_h,
    padding_w,
    output_padding_d,
    output_padding_h,
    output_padding_w,
    input_shape_d,
    input_shape_h,
    input_shape_w,
    output_shape_d,
    output_shape_h,
    output_shape_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of output elements
    pid = tl.program_id(0)
    num_elements = output_shape_d * output_shape_h * output_shape_w
    
    # Iterate over output elements in this block
    for i in range(BLOCK_SIZE):
        idx = pid * BLOCK_SIZE + i
        if idx >= num_elements:
            break
        
        # Decode output coordinates
        rem = idx
        w_out = rem % output_shape_w
        rem //= output_shape_w
        h_out = rem % output_shape_h
        d_out = rem // output_shape_h
        
        # Batch index (assuming batch size is 1 for simplicity in grid, 
        # but we need to handle batch. We'll assume batch is handled by grid or loop.
        # For a general kernel, we should include batch in the index or grid.
        # Let's assume the grid covers B * C_out * D_out * H_out * W_out.
        # But to keep grid simple, let's handle batch in the kernel or assume batch=1.
        # Given the example, we should support batch.
        # We'll compute batch from idx if we flatten everything.
        # However, flattening everything makes index math complex.
        # Better: Grid over (B, C_out, D_out, H_out, W_out) is too large.
        # Grid over 1D index of (B * C_out * D_out * H_out * W_out).
        
        # Re-decode with batch
        # Total output elements per batch = C_out * D_out * H_out * W_out
        # But we can just compute batch from the flat index if we structure the grid that way.
        # Let's assume the grid is launched with size = B * C_out * D_out * H_out * W_out.
        # Then:
        # batch = idx // (C_out * D_out * H_out * W_out)
        # rest = idx % (C_out * D_out * H_out * W_out)
        # c_out = rest // (D_out * H_out * W_out)
        # rest2 = rest % (D_out * H_out * W_out)
        # w_out = rest2 % W_out
        # rest3 = rest2 // W_out
        # h_out = rest3 % H_out
        # d_out = rest3 // H_out
        
        # This decoding is expensive inside the kernel.
        # Optimization: Use tl.program_id with multiple dimensions if possible,
        # or accept the cost. For FP32 speedup, correctness is key.
        # We'll use a 1D grid and decode.
        
        # However, to avoid complex decoding, we can assume the grid is launched
        # such that each block handles a chunk of the flattened tensor.
        # We'll compute coordinates.
        
        # Let's compute batch, c_out, d_out, h_out, w_out from idx
        # We need strides or shapes.
        # We'll pass output_shape as a tuple or separate dims.
        # We passed separate dims.
        
        # Compute batch
        batch = idx // (out_channels * output_shape_d * output_shape_h * output_shape_w)
        idx_local = idx % (out_channels * output_shape_d * output_shape_h * output_shape_w)
        
        c_out = idx_local // (output_shape_d * output_shape_h * output_shape_w)
        idx_local2 = idx_local % (output_shape_d * output_shape_h * output_shape_w)
        
        w_out = idx_local2 % output_shape_w
        idx_local3 = idx_local2 // output_shape_w
        h_out = idx_local3 % output_shape_h
        d_out = idx_local3 // output_shape_h
        
        # Compute output pointer
        # out_ptr is contiguous? Usually yes.
        # out shape: (B, C_out, D_out, H_out, W_out)
        # offset = ((batch * out_channels + c_out) * output_shape_d + d_out) * output_shape_h + h_out) * output_shape_w + w_out
        out_offset = (((batch * out_channels + c_out) * output_shape_d + d_out) * output_shape_h + h_out) * output_shape_w + w_out
        out_ptr_local = out_ptr + out_offset
        
        # Compute sum
        sum_val = 0.0
        
        # Groups
        in_channels_per_group = in_channels // groups
        group_idx = c_out // in_channels_per_group
        c_out_in_group = c_out % in_channels_per_group
        
        # Iterate over kernel dimensions and input channels
        for k_d in range(kernel_size_d):
            for k_h in range(kernel_size_h):
                for k_w in range(kernel_size_w):
                    # Input coordinates
                    in_d = d_out * stride_d - padding_d + k_d
                    in_h = h_out * stride_h - padding_h + k_h
                    in_w = w_out * stride_w - padding_w + k_w
                    
                    # Check input bounds
                    if in_d < 0 or in_d >= input_shape_d or in_h < 0 or in_h >= input_shape_h or in_w < 0 or in_w >= input_shape_w:
                        continue
                    
                    # Iterate over input channels in the group
                    for c_in_offset in range(in_channels_per_group):
                        c_in = group_idx * in_channels_per_group + c_in_offset
                        
                        # Load weight
                        # w shape: (C_out, C_in, K_d, K_h, K_w)
                        w_offset = ((c_out * in_channels + c_in) * kernel_size_d + k_d) * kernel_size_h + k_h) * kernel_size_w + k_w
                        w_val = tl.load(w_ptr + w_offset)
                        
                        # Load input
                        # x shape: (B, C_in, D, H, W)
                        x_offset = ((batch * in_channels + c_in) * input_shape_d + in_d) * input_shape_h + in_h) * input_shape_w + in_w
                        x_val = tl.load(x_ptr + x_offset)
                        
                        sum_val += x_val * w_val
        
        # Add bias if present
        if bias_ptr is not None:
            bias_offset = batch * out_channels + c_out
            bias_val = tl.load(bias_ptr + bias_offset)
            sum_val += bias_val
        
        # Store result
        tl.store(out_ptr_local, sum_val)


def triton_conv_transpose3d(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor = None, stride=(1,1,1), padding=(0,0,0), output_padding=(0,0,0), groups=1) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda
    x = x.contiguous()
    w = w.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, in_d, in_h, in_w = x.shape
    out_channels, _, k_d, k_h, k_w = w.shape
    
    # Compute output shape
    out_d = (in_d - 1) * stride[0] - 2 * padding[0] + k_d + output_padding[0]
    out_h = (in_h - 1) * stride[1] - 2 * padding[1] + k_h + output_padding[1]
    out_w = (in_w - 1) * stride[2] - 2 * padding[2] + k_w + output_padding[2]
    
    out = torch.empty((batch_size, out_channels, out_d, out_h, out_w), dtype=x.dtype, device=x.device)
    
    total_elements = batch_size * out_channels * out_d * out_h * out_w
    BLOCK_SIZE = 128
    
    grid = lambda meta: ((total_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    conv_transpose3d_kernel[grid](
        x, w, out, bias,
        in_channels, out_channels, groups,
        k_d, k_h, k_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        
        # We replace the forward with the Triton kernel
        # The weights are stored in self.conv_transpose3d
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv_transpose3d.weight
        b = self.conv_transpose3d.bias if self.bias else None
        
        return triton_conv_transpose3d(
            x, w, b, 
            stride=self.stride, 
            padding=self.padding, 
            output_padding=self.output_padding, 
            groups=self.groups
        )


def get_inputs():
    batch_size = 8
    in_channels = 32
    depth = 12
    height = 24
    width = 48
    x = torch.rand(batch_size, in_channels, depth, height, width).cuda()
    return [x]


def get_init_inputs():
    in_channels = 32
    out_channels = 32
    kernel_size = (3, 5, 7)
    stride = (2, 2, 2)
    padding = (1, 2, 3)
    output_padding = (1, 1, 1)
    groups = 4
    bias = False
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias]