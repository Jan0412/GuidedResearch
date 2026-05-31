import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    in_channels, out_channels, kernel_h, kernel_w,
    stride_h, stride_w, padding_h, padding_w,
    dilation_h, dilation_w, groups,
    height, width, height_out, width_out,
    BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # 2D block indices for output spatial dimensions
    block_x = tl.program_id(0)
    block_y = tl.program_id(1)
    
    # Block offsets within the output tile
    offsets_x = block_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offsets_y = block_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    
    # Mask for valid output positions
    mask_x = offsets_x < width_out
    mask_y = offsets_y < height_out
    mask = mask_x[:, None] & mask_y[None, :]
    
    # Initialize accumulator for the output tile
    acc = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.float32)
    
    # Loop over groups
    for g in range(groups):
        # Loop over input channels within the group
        for c_off in range(0, in_channels // groups, BLOCK_SIZE_C):
            c_offsets = c_off + tl.arange(0, BLOCK_SIZE_C)
            mask_c = c_offsets < (in_channels // groups)
            
            # Load input tile with padding handling
            # Input coordinates corresponding to output tile
            x_start = block_x * BLOCK_SIZE_X * stride_w - padding_w
            y_start = block_y * BLOCK_SIZE_Y * stride_h - padding_h
            
            # Create coordinate grids for input loading
            # We need to map output tile to input tile including halo
            # Input tile size: (kernel_h + 2*padding_h) x (kernel_w + 2*padding_w)
            # But we can compute offsets dynamically
            
            # For each element in the output tile, compute input coordinates
            # and load the corresponding input patch
            
            # Accumulate over kernel dimensions
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    # Input coordinates for this kernel position
                    # y_in = y_start + kh * dilation_h
                    # x_in = x_start + kw * dilation_w
                    
                    # We can vectorize this by creating offset grids
                    y_in_offsets = y_start + kh * dilation_h + tl.arange(0, BLOCK_SIZE_Y) * stride_h
                    x_in_offsets = x_start + kw * dilation_w + tl.arange(0, BLOCK_SIZE_X) * stride_w
                    
                    # Load input tile
                    # Shape: (BLOCK_SIZE_Y, BLOCK_SIZE_X)
                    # Need to handle padding by masking
                    y_mask = (y_in_offsets[:, None] >= 0) & (y_in_offsets[:, None] < height)
                    x_mask = (x_in_offsets[None, :] >= 0) & (x_in_offsets[None, :] < width)
                    patch_mask = y_mask & x_mask
                    
                    # Load input values
                    # x_ptr layout: (N, C, H, W)
                    # We are processing one batch at a time in this simplified version
                    # For full batch support, we'd need to loop or use 3D grid
                    # Here we assume batch=1 for simplicity in kernel, or handle batch in wrapper
                    # To keep it general, we'll assume the kernel is launched per batch element
                    # or we handle batch index in the grid
                    
                    # For this implementation, we'll handle batch in the grid launch
                    # So x_ptr points to current batch
                    
                    x_val = tl.load(
                        x_ptr + (c_offsets[None, None] * height * width + 
                                 y_in_offsets[:, None] * width + 
                                 x_in_offsets[None, :]),
                        mask=patch_mask & mask_c[None, None],
                        other=0.0
                    )
                    
                    # Load corresponding weights
                    # w_ptr layout: (Out, In//G, K_h, K_w)
                    w_val = tl.load(
                        w_ptr + (g * (out_channels // groups) * in_channels + 
                                 c_offsets[None, None] * kernel_h * kernel_w + 
                                 kh * kernel_w + kw),
                        mask=mask_c[None, None],
                        other=0.0
                    )
                    
                    # Accumulate
                    acc += x_val * w_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_offsets = tl.arange(0, BLOCK_SIZE_Y)[:, None] * out_channels + tl.arange(0, BLOCK_SIZE_X)[None, :]
        bias = tl.load(bias_ptr + bias_offsets, mask=mask, other=0.0)
        acc += bias
    
    # Store output
    out_offsets = (block_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y))[:, None] * width_out + \
                  (block_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X))[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), 
                 padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h, self.kernel_w = kernel_size
        self.stride_h, self.stride_w = stride
        self.padding_h, self.padding_w = padding
        self.dilation_h, self.dilation_w = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias
        self.register_buffer('weight', torch.empty(out_channels, in_channels // groups, self.kernel_h, self.kernel_w))
        if bias:
            self.register_buffer('bias', torch.empty(out_channels))
        else:
            self.bias = None
            
        # Initialize parameters (simple random init for demonstration)
        with torch.no_grad():
            fan_in = in_channels // groups * self.kernel_h * self.kernel_w
            std = (2.0 / fan_in) ** 0.5
            self.weight.normal_(0, std)
            if bias is not None:
                self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_ch, height, width = x.shape
        assert in_ch == self.in_channels, f"Expected input channels {self.in_channels}, got {in_ch}"
        
        # Compute output dimensions
        height_out = (height + 2 * self.padding_h - self.dilation_h * (self.kernel_h - 1) - 1) // self.stride_h + 1
        width_out = (width + 2 * self.padding_w - self.dilation_w * (self.kernel_w - 1) - 1) // self.stride_w + 1
        
        out = torch.empty((batch_size, self.out_channels, height_out, width_out), device=x.device, dtype=x.dtype)
        
        # Kernel launch configuration
        BLOCK_SIZE_X = 16
        BLOCK_SIZE_Y = 16
        BLOCK_SIZE_C = 8
        
        grid = (
            triton.cdiv(width_out, BLOCK_SIZE_X),
            triton.cdiv(height_out, BLOCK_SIZE_Y),
        )
        
        # Launch kernel for each batch element
        for b in range(batch_size):
            x_ptr = x[b]
            out_ptr = out[b]
            
            conv2d_kernel[grid](
                x_ptr, self.weight, self.bias, out_ptr,
                self.in_channels, self.out_channels, self.kernel_h, self.kernel_w,
                self.stride_h, self.stride_w, self.padding_h, self.padding_w,
                self.dilation_h, self.dilation_w, self.groups,
                height, width, height_out, width_out,
                BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_C
            )
            
        return out


def get_inputs():
    batch_size = 8
    in_channels = 64
    out_channels = 128
    kernel_size = (5, 7)
    height = 512
    width = 256
    x = torch.rand(batch_size, in_channels, height, width, device='cuda')
    return [x]


def get_init_inputs():
    return [64, 128, (5, 7)]