import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    in_depth,
    in_width,
    in_height,
    kernel_depth,
    kernel_width,
    kernel_height,
    stride_d,
    stride_w,
    stride_h,
    padding_d,
    padding_w,
    padding_h,
    output_padding_d,
    output_padding_w,
    output_padding_h,
    BLOCK_SIZE_IN: tl.constexpr,
    BLOCK_SIZE_OUT: tl.constexpr,
):
    # Grid covers the output tensor elements
    out_idx = tl.program_id(0)
    
    # Decode output index to 5D coordinates
    temp = out_idx
    h_out = temp % in_height
    temp //= in_height
    w_out = temp % in_width
    temp //= in_width
    d_out = temp % in_depth
    temp //= in_depth
    c_out = temp % out_channels
    b = temp // out_channels
    
    # Compute the starting position in the input tensor
    start_d = d_out * stride_d - padding_d
    start_w = w_out * stride_w - padding_w
    start_h = h_out * stride_h - padding_h
    
    # Initialize accumulator
    acc = 0.0
    
    # Loop over input channels and kernel dimensions
    for c_in in tl.range(0, in_channels, BLOCK_SIZE_IN):
        for k_d in tl.range(0, kernel_depth, BLOCK_SIZE_IN):
            for k_w in tl.range(0, kernel_width, BLOCK_SIZE_IN):
                for k_h in tl.range(0, kernel_height, BLOCK_SIZE_IN):
                    # Calculate input coordinates
                    d_in = start_d + k_d
                    w_in = start_w + k_w
                    h_in = start_h + k_h
                    
                    # Check bounds
                    mask_d = (d_in >= 0) & (d_in < in_depth)
                    mask_w = (w_in >= 0) & (w_in < in_width)
                    mask_h = (h_in >= 0) & (h_in < in_height)
                    mask = mask_d & mask_w & mask_h
                    
                    if mask:
                        # Load input value
                        input_offset = ((b * in_channels + c_in + c_in) * in_depth + d_in) * in_width * in_height + w_in * in_height + h_in
                        input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
                        
                        # Load weight value
                        weight_offset = (c_out * in_channels + c_in + c_in) * kernel_depth * kernel_width * kernel_height + (k_d * kernel_width + k_w) * kernel_height + k_h
                        weight_val = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0)
                        
                        acc += input_val * weight_val
    
    # Store the result
    out_offset = ((b * out_channels + c_out) * in_depth + d_out) * in_width * in_height + w_out * in_height + h_out
    tl.store(output_ptr + out_offset, acc)


def triton_conv_transpose3d(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple = (1, 1, 1),
    padding: tuple = (0, 0, 0),
    output_padding: tuple = (0, 0, 0),
) -> torch.Tensor:
    assert input.is_cuda and weight.is_cuda
    input = input.contiguous()
    weight = weight.contiguous()
    
    batch_size, in_channels, in_depth, in_width, in_height = input.shape
    out_channels, _, kernel_depth, kernel_width, kernel_height = weight.shape
    
    # Calculate output dimensions
    out_depth = (in_depth - 1) * stride[0] - 2 * padding[0] + kernel_depth + output_padding[0]
    out_width = (in_width - 1) * stride[1] - 2 * padding[1] + kernel_width + output_padding[1]
    out_height = (in_height - 1) * stride[2] - 2 * padding[2] + kernel_height + output_padding[2]
    
    output = torch.empty((batch_size, out_channels, out_depth, out_width, out_height), dtype=input.dtype, device=input.device)
    
    # Grid size
    grid = (output.numel(),)
    
    # Launch kernel
    BLOCK_SIZE_IN = 1
    BLOCK_SIZE_OUT = 1
    
    conv_transpose3d_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        in_depth=in_depth,
        in_width=in_width,
        in_height=in_height,
        kernel_depth=kernel_depth,
        kernel_width=kernel_width,
        kernel_height=kernel_height,
        stride_d=stride[0],
        stride_w=stride[1],
        stride_h=stride[2],
        padding_d=padding[0],
        padding_w=padding[1],
        padding_h=padding[2],
        output_padding_d=output_padding[0],
        output_padding_w=output_padding[1],
        output_padding_h=output_padding[2],
        BLOCK_SIZE_IN=BLOCK_SIZE_IN,
        BLOCK_SIZE_OUT=BLOCK_SIZE_OUT,
    )
    
    return output


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
        
        # Initialize weight tensor
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = triton_conv_transpose3d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)
        return output