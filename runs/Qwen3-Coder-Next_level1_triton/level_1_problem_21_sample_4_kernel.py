import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute sigmoid using the stable formula: 1 / (1 + exp(-x))
    # For numerical stability, we use: 
    #   sigmoid(x) = 0.5 + 0.5 * tanh(x/2) for large |x|
    #   or use exp(-|x|) to avoid overflow
    # Here we use a stable implementation:
    # sigmoid(x) = 1 / (1 + exp(-x)) but avoid overflow for large positive x
    x_abs = tl.abs(x)
    exp_neg_x_abs = tl.exp(-x_abs)
    # Use formula: sigmoid(x) = 1 / (1 + exp(-x)) = 
    #   for x >= 0: 1 / (1 + exp(-x))
    #   for x < 0: exp(x) / (1 + exp(x))
    # Or use: sigmoid(x) = 0.5 * (1 + tanh(x/2)) for better numerical stability
    # But Triton's tanh may not be as fast as direct exp, so we use the exp-based approach
    # For numerical stability:
    #   if x >= 0: sigmoid(x) = 1 / (1 + exp(-x))
    #   if x < 0:  sigmoid(x) = exp(x) / (1 + exp(x))
    # Let's use a single expression that avoids overflow:
    one = tl.full((1,), 1.0, dtype=tl.float32)
    neg_x = -x
    exp_neg_x = tl.exp(neg_x)
    # To avoid overflow in exp_neg_x for large negative x, we use:
    #   if x is very negative, exp(-x) becomes very large, so 1/(1+exp(-x)) ~ 0
    #   if x is very positive, exp(-x) ~ 0, so 1/(1+exp(-x)) ~ 1
    # We'll use the safe formulation:
    #   sigmoid(x) = 1 / (1 + exp(-x)) but clamp exp(-x) to avoid overflow
    # However, for FP32, exp(-700) is effectively 0 and exp(700) overflows, but we can use:
    #   exp_neg_x = tl.exp(tl.minimum(neg_x, 80.0))  # clamp to avoid overflow
    # But Triton's tl.exp is fast and handles clamping internally in many cases.
    # Actually, the most efficient and stable way in Triton is:
    #   sigmoid(x) = 0.5 * (1 + tl.tanh(0.5 * x))
    # because tanh is well-behaved and implemented efficiently.
    # Let's use the tanh-based formulation for stability and speed.
    tanh_val = tl.tanh(0.5 * x)
    out = 0.5 * (one + tanh_val)
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_sigmoid(x: torch.Tensor):
    """
    This function wraps the Triton kernel call for sigmoid.
    It:
      1. Ensures the input is contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 512  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    sigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Sigmoid activation using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Sigmoid activation to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Sigmoid applied, same shape as input.
        """
        return triton_sigmoid(x)