import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    n_channels,
    out_channels,
    length,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (N * OC, L_out_blocks)
    pid = tl.program_id(0)
    num_l_blocks = tl.program_id(1)
    
    # Decode n and oc from pid
    n = pid // out_channels
    oc = pid % out_channels
    
    # Calculate offsets for output block
    l_out_offsets = tl.arange(0, BLOCK_N) + num_l_blocks * BLOCK_N
    mask_l = l_out_offsets < n_elements
    
    # Calculate base pointers
    # x shape: (N, C, L)
    # w shape: (OC, IC//groups, K)
    # out shape: (N, OC, L_out)
    
    base_x = x_ptr + n * n_channels * length
    base_w = w_ptr + oc * (n_channels // groups) * kernel_size
    base_out = out_ptr + n * out_channels * n_elements + oc * n_elements
    
    # Bias
    if b_ptr is not None:
        bias = tl.load(b_ptr + oc)
    else:
        bias = 0.0
        
    # Accumulator
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    
    # Loop over input channels
    IC_per_group = n_channels // groups
    OC_per_group = out_channels // groups
    group = oc // OC_per_group
    
    c_offsets = tl.arange(0, BLOCK_K)
    k_offsets = tl.arange(0, kernel_size)
    
    for c_start in range(0, n_channels, BLOCK_K):
        c_offsets_block = c_offsets + c_start
        
        # Check if c_offsets are within bounds
        mask_c = c_offsets_block < n_channels
        
        # Determine group and local_ic for w indexing
        # group is fixed for oc
        # local_ic = c % IC_per_group
        local_ic = c_offsets_block % IC_per_group
        
        # w index: oc * IC_per_group * K + local_ic * K + k
        w_offsets = base_w + local_ic[:, None] * kernel_size + k_offsets[None, :]
        w_mask = mask_c[:, None]
        
        # x index: n * C * L + c * L + l_pos
        # l_pos = l_out * stride + k * dilation
        l_pos = l_out_offsets[:, None] * stride + k_offsets[None, :] * dilation
        
        # Mask for valid input positions (padding check)
        mask_x_valid = (l_pos >= padding) & (l_pos < length + padding)
        mask_x = mask_x_valid & mask_c[:, None]
        
        x_offsets = base_x + c_offsets_block[:, None] * length + l_pos[None, :]
        
        # Load
        x_vals = tl.load(x_offsets, mask=mask_x, other=0.0)
        w_vals = tl.load(w_offsets, mask=w_mask, other=0.0)
        
        # Accumulate
        acc += tl.sum(x_vals * w_vals, axis=0)
        
    # Add bias
    acc += bias
    
    # Store
    tl.store(base_out + l_out_offsets, acc, mask=mask_l)


def triton_conv1d(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, 
                  in_channels: int, out_channels: int, kernel_size: int, 
                  stride: int = 1, padding: int = 0, dilation: int = 1, 
                  groups: int = 1) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda
    x = x.contiguous()
    w = w.contiguous()
    if b is not None:
        b = b.contiguous()
        
    batch_size, _, length = x.shape
    length_out = (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    out = torch.empty((batch_size, out_channels, length_out), dtype=x.dtype, device=x.device)
    
    n_elements = length_out
    BLOCK_K = 64
    BLOCK_N = 64
    
    grid = (batch_size * out_channels, triton.cdiv(n_elements, BLOCK_N))
    
    conv1d_kernel[grid](
        x, w, b, out, n_elements, in_channels, out_channels, length, 
        kernel_size, stride, padding, dilation, groups,
        BLOCK_M=1, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias as in nn.Conv1d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv1d(
            x, self.weight, self.bias,
            self.in_channels, self.out_channels, self.kernel_size,
            self.stride, self.padding, self.dilation, self.groups
        )