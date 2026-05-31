import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def maxpool3d_kernel(
    x_ptr, 
    out_ptr, 
    N, C, D, H, W, 
    Od, Oh, Ow, 
    S, P, Dil, K,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Decompose program ID
    pid = tl.program_id(0)
    
    num_blocks_w = (Ow + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    ow_block_id = pid % num_blocks_w
    rem = pid // num_blocks_w
    
    oh = rem % Oh
    rem = rem // Oh
    
    od = rem % Od
    rem = rem // Od
    
    c = rem % C
    n = rem // C

    # Pointers and strides
    S_N = C * D * H * W
    S_C = D * H * W
    S_D = H * W
    S_H = W
    
    S_out_N = C * Od * Oh * Ow
    S_out_C = Od * Oh * Ow
    S_out_D = Oh * Ow
    S_out_H = Ow

    # Output width offsets
    ow_start = ow_block_id * BLOCK_SIZE_W
    ow_offsets = ow_start + tl.arange(0, BLOCK_SIZE_W)
    mask_ow = ow_offsets < Ow

    # Initialize max values to negative infinity
    cur_max = tl.full((BLOCK_SIZE_W,), -float('inf'), dtype=tl.float32)

    # Iterate over kernel dimensions
    for kd in range(K):
        id_val = od * S + kd * Dil - P
        mask_id = (id_val >= 0) & (id_val < D)
        
        for kh in range(K):
            ih_val = oh * S + kh * Dil - P
            mask_ih = (ih_val >= 0) & (ih_val < H)
            
            for kw in range(K):
                iw_vals = ow_offsets * S + kw * Dil - P
                mask_iw = (iw_vals >= 0) & (iw_vals < W)
                
                # Combine masks
                final_mask = mask_ow & mask_id & mask_ih & mask_iw
                
                # Calculate input offsets
                # x_ptr + n*S_N + c*S_C + id*S_D + ih*S_H + iw
                input_offsets = (n * S_N + c * S_C + 
                                 id_val * S_D + 
                                 ih_val * S_H + 
                                 iw_vals)
                
                # Load values and update max
                vals = tl.load(x_ptr + input_offsets, mask=final_mask, other=-float('inf'))
                cur_max = tl.maximum(cur_max, vals)

    # Store results
    out_offsets = (n * S_out_N + c * S_out_C + 
                   od * S_out_D + 
                   oh * S_out_H + 
                   ow_offsets)
    tl.store(out_ptr + out_offsets, cur_max, mask=mask_ow)


def triton_maxpool3d(x, kernel_size, stride, padding, dilation, ceil_mode):
    # Ensure input is on CUDA and contiguous
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()
    
    N, C, D, H, W = x.shape
    
    # Calculate output dimensions
    def calc_out_dim(dim, k, s, p, d, ceil):
        out_dim = (dim + 2 * p - d * (k - 1) - 1)
        if ceil:
            return (out_dim + s - 1) // s + 1
        else:
            return out_dim // s + 1

    Od = calc_out_dim(D, kernel_size, stride, padding, dilation, ceil_mode)
    Oh = calc_out_dim(H, kernel_size, stride, padding, dilation, ceil_mode)
    Ow = calc_out_dim(W, kernel_size, stride, padding, dilation, ceil_mode)
    
    out = torch.empty((N, C, Od, Oh, Ow), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE_W = 16
    num_blocks_w = (Ow + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    grid = (N * C * Od * Oh * num_blocks_w,)

    maxpool3d_kernel[grid](
        x, out, 
        N, C, D, H, W, 
        Od, Oh, Ow, 
        stride, padding, dilation, kernel_size, 
        BLOCK_SIZE_W=BLOCK_SIZE_W
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 3D using custom Triton kernels.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices
        
        if return_indices:
            raise NotImplementedError("return_indices=True is not supported in this Triton implementation.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 3D to the input tensor using Triton.
        """
        return triton_maxpool3d(
            x, 
            self.kernel_size, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.ceil_mode
        )