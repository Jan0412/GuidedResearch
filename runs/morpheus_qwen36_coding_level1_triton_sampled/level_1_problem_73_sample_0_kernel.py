import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, out_ptr,
    stride, padding, kernel_size,
    in_channels, out_channels, groups,
    batch_size, depth_in, height_in, width_in,
    depth_out, height_out, width_out,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Flatten output dimensions for 1D grid
    out_elements = batch_size * out_channels * depth_out * height_out * width_out
    pid = tl.program_id(0)
    
    if pid < out_elements:
        # Decode coordinates
        wo = pid % width_out
        ho = (pid // width_out) % height_out
        do = (pid // (width_out * height_out)) % depth_out
        oc = (pid // (width_out * height_out * depth_out)) % out_channels
        b = pid // (width_out * height_out * depth_out * out_channels)
        
        # Calculate input channel range for this output channel
        channels_per_group = in_channels // groups
        group_id = oc // channels_per_group
        ic_start = group_id * channels_per_group
        ic_end = ic_start + channels_per_group
        
        acc = 0.0
        
        # Iterate over input channels and kernel dimensions
        for ic in range(ic_start, ic_end):
            for dk in tl.range(0, kernel_size, BLOCK_SIZE_K):
                for dh in tl.range(0, kernel_size, BLOCK_SIZE_K):
                    for dw in tl.range(0, kernel_size, BLOCK_SIZE_K):
                        di = do - padding + dk
                        dh_in = ho - padding + dh
                        dw_in = wo - padding + dw
                        
                        # Bounds check
                        if di >= 0 and di < depth_in and dh_in >= 0 and dh_in < height_in and dw_in >= 0 and dw_in < width_in:
                            x_offset = b * in_channels * depth_in * height_in * width_in + \
                                      ic * depth_in * height_in * width_in + \
                                      di * height_in * width_in + dh_in * width_in + dw_in
                            w_offset = oc * in_channels * kernel_size * kernel_size * kernel_size + \
                                      ic * kernel_size * kernel_size * kernel_size + \
                                      dk * kernel_size * kernel_size + dh * kernel_size + dw
                            
                            x_val = tl.load(x_ptr + x_offset)
                            w_val = tl.load(w_ptr + w_offset)
                            acc += x_val * w_val
        
        # Store result
        out_offset = b * out_channels * depth_out * height_out * width_out + \
                    oc * depth_out * height_out * width_out + \
                    do * height_out * width_out + ho * width_out + wo
        tl.store(out_ptr + out_offset, acc)


def triton_conv_transpose3d(x: torch.Tensor, w: torch.Tensor, stride: int, padding: int, kernel_size: int, groups: int) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    
    batch_size, in_channels, depth_in, height_in, width_in = x.shape
    out_channels, _, _, _, _ = w.shape
    
    # Calculate output dimensions
    depth_out = (depth_in - 1) * stride - 2 * padding + kernel_size
    height_out = (height_in - 1) * stride - 2 * padding + kernel_size
    width_out = (width_in - 1) * stride - 2 * padding + kernel_size
    
    out = torch.empty(batch_size, out_channels, depth_out, height_out, width_out, device=x.device, dtype=x.dtype)
    
    out_elements = batch_size * out_channels * depth_out * height_out * width_out
    BLOCK_SIZE_C = 1
    BLOCK_SIZE_K = 1
    
    grid = lambda meta: (triton.cdiv(out_elements, meta["BLOCK_SIZE"]),)
    
    conv_transpose3d_kernel[grid](
        x, w, out,
        stride, padding, kernel_size,
        in_channels, out_channels, groups,
        batch_size, depth_in, height_in, width_in,
        depth_out, height_out, width_out,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = triton_conv_transpose3d(x, self.weight, self.stride, self.padding, self.kernel_size, self.groups)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1)
        return out