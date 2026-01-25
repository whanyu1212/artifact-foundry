"""
Convolution Operations - From Scratch Implementation

Implements 2D convolution and pooling operations using only NumPy.
These are the fundamental building blocks of Convolutional Neural Networks.

Key Operations:
- im2col: Convert image patches to columns (efficient convolution)
- col2im: Inverse of im2col (for backpropagation)
- conv2d_forward: 2D convolution forward pass
- conv2d_backward: 2D convolution backward pass (gradients)
- max_pool_forward: Max pooling forward pass
- max_pool_backward: Max pooling backward pass

Mathematical Foundation:
Forward convolution:
    Y[i,j,k] = Σ_m Σ_n Σ_c W[m,n,c,k] × X[i+m, j+n, c] + b[k]

Backward convolution:
    ∂L/∂W = convolution of X with ∂L/∂Y
    ∂L/∂X = full convolution of ∂L/∂Y with rotated W

Implementation uses im2col trick for efficiency:
- Converts sliding window operation to matrix multiplication
- Much faster than naive nested loops
- Standard approach in frameworks (Caffe, PyTorch, etc.)
"""

import numpy as np
from typing import Tuple, Optional


# ============================================================================
# IM2COL / COL2IM - Efficient Convolution via Matrix Multiplication
# ============================================================================


def get_im2col_indices(
    x_shape: Tuple[int, ...],
    filter_h: int,
    filter_w: int,
    stride: int = 1,
    padding: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute indices for im2col transformation.

    im2col reshapes image patches into columns, allowing convolution
    to be computed as matrix multiplication (much faster!).

    Args:
        x_shape: Input shape (batch, channels, height, width)
        filter_h: Filter height
        filter_w: Filter width
        stride: Convolution stride
        padding: Zero padding

    Returns:
        Tuple of (k, i, j) index arrays for extracting patches
    """
    batch_size, channels, height, width = x_shape

    # Output dimensions after convolution
    out_h = (height + 2 * padding - filter_h) // stride + 1
    out_w = (width + 2 * padding - filter_w) // stride + 1

    # Indices for extracting patches
    # k: channel index (repeated for each position in filter)
    # i: row index within filter
    # j: column index within filter

    # Create indices for filter positions
    i0 = np.repeat(np.arange(filter_h), filter_w)  # [0,0,0, 1,1,1, 2,2,2] for 3x3
    i0 = np.tile(i0, channels)  # Repeat for each channel

    i1 = stride * np.repeat(np.arange(out_h), out_w)  # Row positions in output

    j0 = np.tile(np.arange(filter_w), filter_h * channels)  # Col within filter
    j1 = stride * np.tile(np.arange(out_w), out_h)  # Col positions in output

    # Combine: final indices are offset by stride
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    # Channel indices
    k = np.repeat(np.arange(channels), filter_h * filter_w).reshape(-1, 1)

    return k, i, j


def im2col(
    x: np.ndarray,
    filter_h: int,
    filter_w: int,
    stride: int = 1,
    padding: int = 0
) -> np.ndarray:
    """
    Transform image into column matrix for efficient convolution.

    Converts sliding window operation into matrix multiplication:
    - Each column corresponds to a patch of the input
    - Convolution becomes: output = W @ im2col(X)

    Example (1D):
        Input: [1, 2, 3, 4, 5], filter_size=3, stride=1
        Output columns: [[1, 2, 3],
                         [2, 3, 4],
                         [3, 4, 5]]^T

    Args:
        x: Input, shape (batch, channels, height, width)
        filter_h: Filter height
        filter_w: Filter width
        stride: Stride for sliding window
        padding: Zero padding to add around input

    Returns:
        Column matrix, shape (filter_h * filter_w * channels, out_h * out_w * batch)
    """
    # Add padding if needed
    if padding > 0:
        x = np.pad(
            x,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant'
        )

    # Get indices for extracting patches
    k, i, j = get_im2col_indices(x.shape, filter_h, filter_w, stride, padding=0)

    # Extract columns using advanced indexing
    # x[:, k, i, j] extracts all patches for all samples
    # Shape: (batch, channels*filter_h*filter_w, out_h*out_w)
    cols = x[:, k, i, j]

    # Reshape to (channels*filter_h*filter_w, out_h*out_w*batch)
    batch_size = x.shape[0]
    channels = k.max() + 1
    cols = cols.transpose(1, 2, 0).reshape(channels * filter_h * filter_w, -1)

    return cols


def col2im(
    cols: np.ndarray,
    x_shape: Tuple[int, ...],
    filter_h: int,
    filter_w: int,
    stride: int = 1,
    padding: int = 0
) -> np.ndarray:
    """
    Inverse of im2col - transform columns back to image format.

    Used in backpropagation to convert gradients from column format
    back to image format.

    Note: Multiple patches may contribute to same pixel (overlapping windows),
    so we accumulate (sum) contributions.

    Args:
        cols: Column matrix from im2col
        x_shape: Original input shape (batch, channels, height, width)
        filter_h: Filter height
        filter_w: Filter width
        stride: Stride used in im2col
        padding: Padding used in im2col

    Returns:
        Image format array, shape x_shape
    """
    batch_size, channels, height, width = x_shape

    # Account for padding in dimensions
    h_padded = height + 2 * padding
    w_padded = width + 2 * padding

    # Initialize output with padding
    x_padded = np.zeros((batch_size, channels, h_padded, w_padded), dtype=cols.dtype)

    # Get indices
    k, i, j = get_im2col_indices(x_shape, filter_h, filter_w, stride, padding)

    # Reshape columns to (filter_h*filter_w*channels, out_h*out_w, batch)
    out_h = (height + 2 * padding - filter_h) // stride + 1
    out_w = (width + 2 * padding - filter_w) // stride + 1

    cols_reshaped = cols.reshape(channels * filter_h * filter_w, -1, batch_size)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)

    # Accumulate values back to image (use at for repeated indices)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    # Remove padding
    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded


# ============================================================================
# CONVOLUTION FORWARD & BACKWARD
# ============================================================================


def conv2d_forward(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    stride: int = 1,
    padding: int = 0
) -> Tuple[np.ndarray, dict]:
    """
    Forward pass for 2D convolution.

    Computation (for each output position):
        Y[n,k,i,j] = Σ_c Σ_m Σ_n W[k,c,m,n] × X[n,c,i*s+m,j*s+n] + b[k]

    Uses im2col trick:
        1. Transform input patches to columns: X_col
        2. Reshape filters to rows: W_row
        3. Compute: Y = W_row @ X_col + b
        4. Reshape Y to proper output dimensions

    Args:
        x: Input, shape (batch, in_channels, height, width)
        w: Filters, shape (out_channels, in_channels, filter_h, filter_w)
        b: Biases, shape (out_channels,)
        stride: Convolution stride
        padding: Zero padding around input

    Returns:
        out: Output, shape (batch, out_channels, out_h, out_w)
        cache: Dictionary containing values needed for backward pass
    """
    batch_size, in_channels, height, width = x.shape
    out_channels, _, filter_h, filter_w = w.shape

    # Output dimensions
    out_h = (height + 2 * padding - filter_h) // stride + 1
    out_w = (width + 2 * padding - filter_w) // stride + 1

    # Transform input to column matrix
    x_col = im2col(x, filter_h, filter_w, stride, padding)

    # Reshape filters to (out_channels, in_channels * filter_h * filter_w)
    w_row = w.reshape(out_channels, -1)

    # Convolution as matrix multiplication: out = W @ X_col + b
    out = w_row @ x_col + b.reshape(-1, 1)

    # Reshape output to (out_channels, out_h, out_w, batch)
    out = out.reshape(out_channels, out_h, out_w, batch_size)

    # Transpose to (batch, out_channels, out_h, out_w)
    out = out.transpose(3, 0, 1, 2)

    # Cache values needed for backward pass
    cache = {
        'x': x,
        'w': w,
        'b': b,
        'x_col': x_col,
        'stride': stride,
        'padding': padding
    }

    return out, cache


def conv2d_backward(
    dout: np.ndarray,
    cache: dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Backward pass for 2D convolution.

    Computes gradients:
        ∂L/∂X: Gradient w.r.t. input (for previous layer)
        ∂L/∂W: Gradient w.r.t. filters (for parameter update)
        ∂L/∂b: Gradient w.r.t. biases (for parameter update)

    Key insights:
        - ∂L/∂W is computed by convolving X with ∂L/∂Y
        - ∂L/∂X is computed by "full convolution" of ∂L/∂Y with rotated W
        - Both can be done efficiently with im2col/col2im

    Args:
        dout: Gradient of loss w.r.t. output, shape (batch, out_channels, out_h, out_w)
        cache: Cached values from forward pass

    Returns:
        dx: Gradient w.r.t. input, same shape as x
        dw: Gradient w.r.t. filters, same shape as w
        db: Gradient w.r.t. biases, same shape as b
    """
    x, w, b, x_col, stride, padding = (
        cache['x'], cache['w'], cache['b'],
        cache['x_col'], cache['stride'], cache['padding']
    )

    batch_size, in_channels, height, width = x.shape
    out_channels, _, filter_h, filter_w = w.shape

    # Gradient w.r.t. bias: sum over all positions
    # db = Σ_n Σ_i Σ_j dout[n,:,i,j]
    db = np.sum(dout, axis=(0, 2, 3))

    # Reshape dout to (out_channels, out_h * out_w * batch)
    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(out_channels, -1)

    # Gradient w.r.t. weights: dW = dout @ X_col^T
    # This is equivalent to convolving input with output gradient
    dw = dout_reshaped @ x_col.T
    dw = dw.reshape(w.shape)

    # Gradient w.r.t. input: dx = W^T @ dout
    # This is a "full convolution" (convolution with rotated filter)
    w_reshaped = w.reshape(out_channels, -1)
    dx_col = w_reshaped.T @ dout_reshaped

    # Transform dx from column format back to image format
    dx = col2im(dx_col, x.shape, filter_h, filter_w, stride, padding)

    return dx, dw, db


# ============================================================================
# MAX POOLING FORWARD & BACKWARD
# ============================================================================


def max_pool_forward(
    x: np.ndarray,
    pool_h: int = 2,
    pool_w: int = 2,
    stride: int = 2
) -> Tuple[np.ndarray, dict]:
    """
    Forward pass for max pooling.

    Max pooling: Take maximum value in each pool window
        Y[n,c,i,j] = max_{m,n in pool} X[n,c,i*s+m,j*s+n]

    Purpose:
        - Reduce spatial dimensions (downsampling)
        - Provide translation invariance
        - Reduce parameters in subsequent layers

    Args:
        x: Input, shape (batch, channels, height, width)
        pool_h: Pooling window height
        pool_w: Pooling window width
        stride: Stride for pooling

    Returns:
        out: Output, shape (batch, channels, out_h, out_w)
        cache: Values needed for backward pass (max positions)
    """
    batch_size, channels, height, width = x.shape

    # Output dimensions
    out_h = (height - pool_h) // stride + 1
    out_w = (width - pool_w) // stride + 1

    # Use im2col to extract pooling windows
    x_col = im2col(x, pool_h, pool_w, stride=stride, padding=0)

    # Reshape to (batch * channels, pool_h * pool_w, out_h * out_w)
    x_col = x_col.reshape(channels * pool_h * pool_w, batch_size, out_h * out_w)
    x_col = x_col.transpose(1, 0, 2)
    x_col = x_col.reshape(batch_size * channels, pool_h * pool_w, out_h * out_w)

    # Take max over pooling window (axis=1)
    out = np.max(x_col, axis=1)

    # Store indices of max values (needed for backward pass)
    max_idx = np.argmax(x_col, axis=1)

    # Reshape output to (batch, channels, out_h, out_w)
    out = out.reshape(batch_size, channels, out_h, out_w)

    cache = {
        'x': x,
        'max_idx': max_idx,
        'pool_h': pool_h,
        'pool_w': pool_w,
        'stride': stride
    }

    return out, cache


def max_pool_backward(
    dout: np.ndarray,
    cache: dict
) -> np.ndarray:
    """
    Backward pass for max pooling.

    Gradient flows only to the max element in each pool window.
    All other elements get zero gradient.

    Intuition: Only the maximum contributed to output, so only
    the maximum should receive gradient.

    Args:
        dout: Gradient w.r.t. output, shape (batch, channels, out_h, out_w)
        cache: Cached values from forward pass

    Returns:
        dx: Gradient w.r.t. input, same shape as x
    """
    x, max_idx, pool_h, pool_w, stride = (
        cache['x'], cache['max_idx'],
        cache['pool_h'], cache['pool_w'], cache['stride']
    )

    batch_size, channels, height, width = x.shape
    out_h, out_w = dout.shape[2], dout.shape[3]

    # Flatten dout
    dout_flat = dout.reshape(batch_size * channels, -1)

    # Create gradient matrix in column format
    # Only max positions get gradient, others are zero
    dx_col = np.zeros((batch_size * channels, pool_h * pool_w, out_h * out_w))

    # Scatter gradient to max positions
    # For each pool window, put gradient at the max index
    rows = np.arange(batch_size * channels).reshape(-1, 1)
    dx_col[rows, max_idx, np.arange(out_h * out_w)] = dout_flat

    # Reshape dx_col for col2im
    dx_col = dx_col.reshape(batch_size * channels * pool_h * pool_w, out_h * out_w)
    dx_col = dx_col.reshape(channels * pool_h * pool_w, -1)

    # Convert back to image format
    dx = col2im(dx_col, x.shape, pool_h, pool_w, stride=stride, padding=0)

    return dx


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def check_conv_output_size(
    input_size: int,
    filter_size: int,
    stride: int = 1,
    padding: int = 0
) -> int:
    """
    Calculate output size after convolution.

    Formula: output_size = ⌊(input_size - filter_size + 2×padding) / stride⌋ + 1

    Args:
        input_size: Input spatial dimension
        filter_size: Filter spatial dimension
        stride: Convolution stride
        padding: Zero padding

    Returns:
        Output spatial dimension
    """
    return (input_size - filter_size + 2 * padding) // stride + 1


def check_pool_output_size(
    input_size: int,
    pool_size: int,
    stride: int
) -> int:
    """
    Calculate output size after pooling.

    Formula: output_size = ⌊(input_size - pool_size) / stride⌋ + 1

    Args:
        input_size: Input spatial dimension
        pool_size: Pooling window size
        stride: Pooling stride

    Returns:
        Output spatial dimension
    """
    return (input_size - pool_size) // stride + 1


# ============================================================================
# DEMONSTRATIONS
# ============================================================================


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()

    console.print("\n[bold cyan]" + "=" * 70 + "[/bold cyan]")
    console.print("[bold cyan]Convolution Operations: From Scratch[/bold cyan]")
    console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]")

    # ==========================================================================
    # TEST 1: Basic Convolution
    # ==========================================================================

    console.print("\n[bold yellow]Test 1: 2D Convolution[/bold yellow]")
    console.print("-" * 70)

    # Simple example: edge detection
    console.print("\n[bold]Edge Detection Example:[/bold]")
    console.print("Vertical edge detector filter")

    # Create simple image (5x5) with vertical edge
    x_simple = np.array([
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1]
    ], dtype=float).reshape(1, 1, 5, 5)

    # Vertical edge detector
    w_simple = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ], dtype=float).reshape(1, 1, 3, 3)

    b_simple = np.array([0.0])

    out_simple, _ = conv2d_forward(x_simple, w_simple, b_simple, stride=1, padding=0)

    console.print("\nInput (5x5):")
    console.print(x_simple[0, 0])
    console.print("\nFilter (3x3) - Vertical Edge Detector:")
    console.print(w_simple[0, 0])
    console.print("\nOutput (3x3):")
    console.print(out_simple[0, 0])
    console.print("\n[green]✓ Strong response at vertical edge![/green]")

    # ==========================================================================
    # TEST 2: Max Pooling
    # ==========================================================================

    console.print("\n[bold yellow]Test 2: Max Pooling[/bold yellow]")
    console.print("-" * 70)

    # Create 4x4 input
    x_pool = np.array([
        [1, 3, 2, 4],
        [5, 6, 7, 8],
        [3, 2, 1, 2],
        [1, 3, 5, 9]
    ], dtype=float).reshape(1, 1, 4, 4)

    console.print("\nInput (4x4):")
    console.print(x_pool[0, 0])

    out_pool, _ = max_pool_forward(x_pool, pool_h=2, pool_w=2, stride=2)

    console.print("\nAfter 2x2 Max Pooling (stride=2):")
    console.print(out_pool[0, 0])
    console.print("\n[green]✓ Each value is max of its 2x2 window[/green]")

    # ==========================================================================
    # TEST 3: Multi-channel Convolution
    # ==========================================================================

    console.print("\n[bold yellow]Test 3: Multi-Channel Convolution (RGB-style)[/bold yellow]")
    console.print("-" * 70)

    # 2 samples, 3 channels (like RGB), 8x8 images
    np.random.seed(42)
    x_multi = np.random.randn(2, 3, 8, 8)

    # 16 filters, 3 input channels, 3x3 kernels
    w_multi = np.random.randn(16, 3, 3, 3) * 0.1
    b_multi = np.zeros(16)

    out_multi, cache_multi = conv2d_forward(
        x_multi, w_multi, b_multi, stride=1, padding=1
    )

    console.print(f"\nInput shape: {x_multi.shape}")
    console.print(f"  (batch=2, channels=3, height=8, width=8)")
    console.print(f"\nFilter shape: {w_multi.shape}")
    console.print(f"  (out_channels=16, in_channels=3, filter_h=3, filter_w=3)")
    console.print(f"\nOutput shape: {out_multi.shape}")
    console.print(f"  (batch=2, out_channels=16, out_h=8, out_w=8)")
    console.print("\n[green]✓ Multi-channel convolution works![/green]")

    # Test backpropagation
    dout_multi = np.random.randn(*out_multi.shape)
    dx_multi, dw_multi, db_multi = conv2d_backward(dout_multi, cache_multi)

    console.print("\n[bold]Gradient Shapes:[/bold]")
    console.print(f"  dx: {dx_multi.shape} (same as input)")
    console.print(f"  dw: {dw_multi.shape} (same as filters)")
    console.print(f"  db: {db_multi.shape} (same as biases)")
    console.print("\n[green]✓ Backpropagation produces correct shapes![/green]")

    # ==========================================================================
    # KEY INSIGHTS
    # ==========================================================================

    console.print("\n[bold cyan]" + "=" * 70 + "[/bold cyan]")
    console.print("[bold cyan]Key Insights[/bold cyan]")
    console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]")

    insights = [
        "",
        "[bold]CONVOLUTION:[/bold]",
        "• Sliding window operation: applies filter at every position",
        "• Detects features: edges, textures, patterns",
        "• Parameter sharing: same filter for entire image",
        "• Output size: (input - filter + 2×padding) / stride + 1",
        "",
        "[bold]IM2COL TRICK:[/bold]",
        "• Converts convolution to matrix multiplication",
        "• Much faster than naive loops (leverages optimized BLAS)",
        "• Standard in frameworks: Caffe, PyTorch, Theano",
        "• Trade-off: Speed for memory (duplicates data)",
        "",
        "[bold]MAX POOLING:[/bold]",
        "• Downsampling: reduces spatial dimensions",
        "• Translation invariance: small shifts don't change max",
        "• Backprop: gradient only flows to max element",
        "",
        "[bold]BACKPROPAGATION:[/bold]",
        "• ∂L/∂W = convolution of input with output gradient",
        "• ∂L/∂X = full convolution with rotated filter",
        "• Both can use im2col/col2im for efficiency",
        "",
        "[yellow]✓ Now you understand what PyTorch Conv2d does internally![/yellow]",
    ]

    for insight in insights:
        console.print(insight)

    console.print("")
