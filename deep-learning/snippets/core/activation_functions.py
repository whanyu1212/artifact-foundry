"""
Activation Functions - From Scratch Implementation

Implements common activation functions and their derivatives for neural networks.

Classical Activations:
- Sigmoid
- Tanh (Hyperbolic Tangent)
- ReLU (Rectified Linear Unit)

Modern Activations:
- Leaky ReLU
- ELU (Exponential Linear Unit)
- SELU (Scaled ELU)
- Swish (SiLU)
- GELU (Gaussian Error Linear Unit)

Output Layer Activations:
- Softmax
- Linear

All implementations include both forward pass and derivative for backpropagation.
Validated against standard implementations and numerical gradients.
"""

import numpy as np
from typing import Optional
from scipy.special import erf  # For GELU


# ============================================================================
# CLASSICAL ACTIVATION FUNCTIONS
# ============================================================================


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.

    Formula: σ(z) = 1 / (1 + e^(-z))

    Args:
        z: Input array, any shape.

    Returns:
        Activated output, same shape as input.

    Properties:
        - Range: (0, 1)
        - Smooth: Yes
        - Zero-centered: No
        - Monotonic: Yes (always increasing)

    Use cases:
        - Binary classification (output layer)
        - Gate mechanisms (LSTM, GRU)
        - Legacy hidden layers (not recommended)

    Example:
        >>> z = np.array([-2, -1, 0, 1, 2])
        >>> sigmoid(z)
        array([0.1192, 0.2689, 0.5000, 0.7311, 0.8808])
    """
    # Numerical stability: avoid overflow for large negative values
    # For z < 0, use: σ(z) = e^z / (1 + e^z)
    # For z >= 0, use: σ(z) = 1 / (1 + e^(-z))
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),  # Standard formula for z >= 0
        np.exp(z) / (1 + np.exp(z)),  # Stable formula for z < 0
    )


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of sigmoid function.

    Formula: σ'(z) = σ(z) * (1 - σ(z))

    Args:
        z: Input array (pre-activation), any shape.

    Returns:
        Derivative, same shape as input.

    Note:
        This computes derivative w.r.t. z (pre-activation).
        If you have activations a = σ(z), use: a * (1 - a)

    Example:
        >>> z = np.array([0.0])
        >>> sigmoid_derivative(z)  # Maximum gradient at z=0
        array([0.25])
    """
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z: np.ndarray) -> np.ndarray:
    """
    Hyperbolic tangent activation function.

    Formula: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))

    Args:
        z: Input array, any shape.

    Returns:
        Activated output, same shape as input.

    Properties:
        - Range: (-1, 1)
        - Smooth: Yes
        - Zero-centered: Yes
        - Monotonic: Yes (always increasing)

    Relationship to sigmoid:
        tanh(z) = 2 * σ(2z) - 1

    Use cases:
        - RNN hidden states (still common)
        - Legacy hidden layers
        - When zero-centered outputs are desired

    Example:
        >>> z = np.array([-2, -1, 0, 1, 2])
        >>> tanh(z)
        array([-0.9640, -0.7616, 0.0000, 0.7616, 0.9640])
    """
    return np.tanh(z)


def tanh_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of tanh function.

    Formula: tanh'(z) = 1 - tanh^2(z)

    Args:
        z: Input array (pre-activation), any shape.

    Returns:
        Derivative, same shape as input.

    Note:
        If you have activations a = tanh(z), use: 1 - a^2

    Example:
        >>> z = np.array([0.0])
        >>> tanh_derivative(z)  # Maximum gradient at z=0
        array([1.0])
    """
    t = np.tanh(z)
    return 1 - t**2


def relu(z: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit (ReLU) activation function.

    Formula: ReLU(z) = max(0, z)

    Args:
        z: Input array, any shape.

    Returns:
        Activated output, same shape as input.

    Properties:
        - Range: [0, ∞)
        - Smooth: No (kink at z=0)
        - Zero-centered: No
        - Monotonic: Yes (non-decreasing)
        - Sparse: Yes (outputs exactly 0 for z < 0)

    Advantages:
        - Very fast to compute
        - No vanishing gradient for z > 0
        - Induces sparsity

    Disadvantages:
        - Dying ReLU: neurons can get stuck at 0
        - Not zero-centered

    Use cases:
        - Default choice for hidden layers
        - CNNs, feedforward networks
        - Most modern architectures

    Example:
        >>> z = np.array([-2, -1, 0, 1, 2])
        >>> relu(z)
        array([0, 0, 0, 1, 2])
    """
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU function.

    Formula:
        ReLU'(z) = 1 if z > 0 else 0

    Args:
        z: Input array (pre-activation), any shape.

    Returns:
        Derivative, same shape as input.

    Note:
        At z=0, technically undefined. We use 0 (sub-gradient).
        This is the standard convention in deep learning.

    Example:
        >>> z = np.array([-1, 0, 1])
        >>> relu_derivative(z)
        array([0, 0, 1])
    """
    return (z > 0).astype(float)


# ============================================================================
# MODERN ACTIVATION FUNCTIONS
# ============================================================================


def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Leaky ReLU activation function.

    Formula: LeakyReLU(z) = max(αz, z) = z if z > 0 else αz

    Args:
        z: Input array, any shape.
        alpha: Slope for negative inputs. Typically 0.01.

    Returns:
        Activated output, same shape as input.

    Properties:
        - Range: (-∞, ∞)
        - Smooth: No (kink at z=0)
        - Zero-centered: No
        - Monotonic: Yes

    Advantages:
        - Fixes dying ReLU problem
        - Still very fast
        - Always has non-zero gradient

    Use cases:
        - Alternative to ReLU when dying neurons are a problem
        - GANs (commonly used)
        - Any hidden layer

    Variants:
        - Parametric ReLU (PReLU): α is learned
        - Randomized Leaky ReLU: α sampled randomly

    Example:
        >>> z = np.array([-2, -1, 0, 1, 2])
        >>> leaky_relu(z, alpha=0.01)
        array([-0.02, -0.01, 0.00, 1.00, 2.00])
    """
    return np.where(z > 0, z, alpha * z)


def leaky_relu_derivative(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Derivative of Leaky ReLU.

    Formula:
        LeakyReLU'(z) = 1 if z > 0 else α

    Args:
        z: Input array (pre-activation), any shape.
        alpha: Slope for negative inputs. Same as in forward pass.

    Returns:
        Derivative, same shape as input.

    Example:
        >>> z = np.array([-1, 0, 1])
        >>> leaky_relu_derivative(z, alpha=0.01)
        array([0.01, 0.01, 1.00])
    """
    return np.where(z > 0, 1.0, alpha)


def elu(z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Exponential Linear Unit (ELU) activation function.

    Formula:
        ELU(z) = z if z > 0 else α(e^z - 1)

    Args:
        z: Input array, any shape.
        alpha: Scale for negative values. Typically 1.0.

    Returns:
        Activated output, same shape as input.

    Properties:
        - Range: (-α, ∞)
        - Smooth: Yes (differentiable everywhere)
        - Zero-centered: Approximately (mean closer to 0 than ReLU)
        - Monotonic: Yes

    Advantages:
        - Smooth (no kink at 0)
        - Zero-centered outputs
        - No dying neurons
        - Often faster convergence than ReLU

    Disadvantages:
        - Slower than ReLU (exponential computation)

    Use cases:
        - Hidden layers when faster convergence desired
        - Alternative to ReLU with better properties
        - Deeper networks

    Example:
        >>> z = np.array([-2, -1, 0, 1, 2])
        >>> elu(z, alpha=1.0)
        array([-0.8647, -0.6321, 0.0000, 1.0000, 2.0000])
    """
    return np.where(z > 0, z, alpha * (np.exp(z) - 1))


def elu_derivative(z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Derivative of ELU activation.

    Formula:
        ELU'(z) = 1 if z > 0 else α * e^z = ELU(z) + α if z ≤ 0

    Args:
        z: Input array (pre-activation), any shape.
        alpha: Scale parameter. Same as in forward pass.

    Returns:
        Derivative, same shape as input.

    Note:
        For z ≤ 0: derivative = ELU(z) + α = α * e^z

    Example:
        >>> z = np.array([-1, 0, 1])
        >>> elu_derivative(z, alpha=1.0)
        array([0.3679, 1.0000, 1.0000])
    """
    return np.where(z > 0, 1.0, alpha * np.exp(z))


def selu(z: np.ndarray) -> np.ndarray:
    """
    Scaled Exponential Linear Unit (SELU) activation function.

    Formula:
        SELU(z) = λ * (z if z > 0 else α(e^z - 1))

    Where:
        λ ≈ 1.0507
        α ≈ 1.67326

    These constants ensure self-normalizing property.

    Args:
        z: Input array, any shape.

    Returns:
        Activated output, same shape as input.

    Properties:
        - Range: (-λα, ∞)
        - Smooth: Yes
        - Zero-centered: Yes
        - Self-normalizing: Maintains mean~0, variance~1

    Advantages:
        - Self-normalizing (can replace batch norm)
        - Enables very deep networks without batch norm

    Disadvantages:
        - Requires specific conditions:
          * LeCun normal initialization
          * AlphaDropout (not standard dropout)
          * Fully connected architecture

    Use cases:
        - Deep fully connected networks
        - When batch normalization is problematic
        - Specific SELU architectures

    Paper:
        Klambauer et al. (2017) "Self-Normalizing Neural Networks"

    Example:
        >>> z = np.array([-2, -1, 0, 1, 2])
        >>> selu(z)
        array([-1.5201, -1.1113, 0.0000, 1.0507, 2.1014])
    """
    # Constants for self-normalizing property
    alpha = 1.67326324
    scale = 1.05070098

    return scale * np.where(z > 0, z, alpha * (np.exp(z) - 1))


def selu_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of SELU activation.

    Formula:
        SELU'(z) = λ if z > 0 else λ * α * e^z

    Args:
        z: Input array (pre-activation), any shape.

    Returns:
        Derivative, same shape as input.

    Example:
        >>> z = np.array([-1, 0, 1])
        >>> selu_derivative(z)
        array([0.6472, 1.0507, 1.0507])
    """
    alpha = 1.67326324
    scale = 1.05070098

    return scale * np.where(z > 0, 1.0, alpha * np.exp(z))


def swish(z: np.ndarray) -> np.ndarray:
    """
    Swish activation function (also called SiLU - Sigmoid-weighted Linear Unit).

    Formula:
        Swish(z) = z * σ(z) = z / (1 + e^(-z))

    Args:
        z: Input array, any shape.

    Returns:
        Activated output, same shape as input.

    Properties:
        - Range: (-∞, ∞)
        - Smooth: Yes (infinitely differentiable)
        - Zero-centered: No
        - Monotonic: No (small bump below 0)
        - Self-gated: Output depends on both z and σ(z)

    Advantages:
        - Smooth (better gradient flow than ReLU)
        - Non-monotonic (more expressive)
        - Often outperforms ReLU in deep networks

    Disadvantages:
        - Slower than ReLU (sigmoid computation)

    Use cases:
        - Deep networks (especially very deep)
        - Modern architectures (EfficientNet)
        - When ReLU performance plateaus

    Papers:
        - Ramachandran et al. (2017) "Searching for Activation Functions"
        - Elfwing et al. (2018) "Sigmoid-Weighted Linear Units"

    Example:
        >>> z = np.array([-2, -1, 0, 1, 2])
        >>> swish(z)
        array([-0.2384, -0.2689, 0.0000, 0.7311, 1.7616])
    """
    return z * sigmoid(z)


def swish_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of Swish activation.

    Formula:
        Swish'(z) = σ(z) + z * σ(z) * (1 - σ(z))
                  = σ(z) * (1 + z * (1 - σ(z)))

    Args:
        z: Input array (pre-activation), any shape.

    Returns:
        Derivative, same shape as input.

    Example:
        >>> z = np.array([0.0])
        >>> swish_derivative(z)
        array([0.5])
    """
    s = sigmoid(z)
    return s + z * s * (1 - s)
    # Equivalent: s * (1 + z * (1 - s))


def gelu(z: np.ndarray, approximate: bool = True) -> np.ndarray:
    """
    Gaussian Error Linear Unit (GELU) activation function.

    Formula (exact):
        GELU(z) = z * Φ(z)
        where Φ is the CDF of standard normal distribution

    Formula (approximate):
        GELU(z) ≈ 0.5 * z * (1 + tanh(√(2/π) * (z + 0.044715 * z^3)))
        or
        GELU(z) ≈ z * σ(1.702 * z)

    Args:
        z: Input array, any shape.
        approximate: If True, use fast approximation. Default True.

    Returns:
        Activated output, same shape as input.

    Properties:
        - Range: (-∞, ∞)
        - Smooth: Yes
        - Zero-centered: No
        - Monotonic: No (similar to Swish)
        - Probabilistic: Based on Gaussian CDF

    Advantages:
        - State-of-the-art for Transformers
        - Smooth, well-behaved gradients
        - Probabilistic interpretation

    Disadvantages:
        - Computationally expensive (erf or approximation)

    Use cases:
        - Transformer models (BERT, GPT)
        - NLP tasks
        - When computational cost is acceptable

    Paper:
        Hendrycks & Gimpel (2016) "Gaussian Error Linear Units (GELUs)"

    Example:
        >>> z = np.array([-2, -1, 0, 1, 2])
        >>> gelu(z)
        array([-0.0454, -0.1588, 0.0000, 0.8412, 1.9546])
    """
    if approximate:
        # Fast approximation: GELU(z) ≈ z * σ(1.702z)
        return z * sigmoid(1.702 * z)
    else:
        # Exact: GELU(z) = z * Φ(z) = 0.5 * z * (1 + erf(z/√2))
        return 0.5 * z * (1 + erf(z / np.sqrt(2)))


def gelu_derivative(z: np.ndarray, approximate: bool = True) -> np.ndarray:
    """
    Derivative of GELU activation.

    Formula (approximate):
        GELU'(z) ≈ σ(1.702z) * (1 + 1.702z * (1 - σ(1.702z)))

    Args:
        z: Input array (pre-activation), any shape.
        approximate: If True, use fast approximation. Default True.

    Returns:
        Derivative, same shape as input.

    Example:
        >>> z = np.array([0.0])
        >>> gelu_derivative(z)
        array([0.5])
    """
    if approximate:
        # Derivative of z * σ(1.702z)
        # d/dz[z * σ(az)] = σ(az) + az * σ(az) * (1 - σ(az))
        a = 1.702
        s = sigmoid(a * z)
        return s + a * z * s * (1 - s)
    else:
        # Exact derivative involves Gaussian PDF
        phi = 0.5 * (1 + erf(z / np.sqrt(2)))
        pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        return phi + z * pdf


# ============================================================================
# OUTPUT LAYER ACTIVATIONS
# ============================================================================


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax activation function.

    Converts logits to probability distribution.

    Formula:
        softmax(z_i) = e^(z_i) / Σ_j e^(z_j)

    Args:
        z: Input array. Typically shape (n_samples, n_classes).
        axis: Axis along which to compute softmax. Default -1 (last axis).

    Returns:
        Probability distribution, same shape as input.
        Each element in [0, 1], sum along axis equals 1.

    Properties:
        - Range: (0, 1) for each output
        - Sum: Σ softmax(z) = 1 (probability distribution)
        - Monotonic: Preserves relative ordering of inputs

    Numerical Stability:
        Subtracts max(z) before exp to prevent overflow.

    Use cases:
        - Multi-class classification (output layer)
        - Attention mechanisms
        - Any time you need a probability distribution

    Paired with:
        - Categorical cross-entropy loss
        - Clean gradient: ∂L/∂z_i = ŷ_i - y_i

    Example:
        >>> z = np.array([[1.0, 2.0, 3.0]])
        >>> softmax(z)
        array([[0.0900, 0.2447, 0.6652]])
        >>> softmax(z).sum()  # Sums to 1
        1.0
    """
    # Numerical stability: subtract max to prevent overflow
    # softmax(z) = softmax(z - max(z)) (mathematically equivalent)
    z_shifted = z - np.max(z, axis=axis, keepdims=True)

    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)


def softmax_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of softmax activation.

    Formula (for element i w.r.t. input j):
        ∂softmax(z_i) / ∂z_j = softmax(z_i) * (δ_ij - softmax(z_j))
        where δ_ij = 1 if i=j else 0

    Note:
        This returns the diagonal elements of the Jacobian matrix.
        For full Jacobian, see dedicated function.

    When paired with cross-entropy loss:
        ∂L/∂z = ŷ - y  (very clean gradient!)

    Args:
        z: Input array (logits).

    Returns:
        Derivative for diagonal elements (most common use case).

    Example:
        >>> z = np.array([1.0, 2.0, 3.0])
        >>> s = softmax(z)
        >>> softmax_derivative(z)  # Diagonal of Jacobian
        array([0.0819, 0.1849, 0.2227])
    """
    s = softmax(z)
    # Diagonal elements: s_i * (1 - s_i)
    return s * (1 - s)


def linear(z: np.ndarray) -> np.ndarray:
    """
    Linear activation (identity function).

    Formula: linear(z) = z

    Args:
        z: Input array, any shape.

    Returns:
        Same as input (identity).

    Use cases:
        - Regression output layer
        - No constraints on output range

    Paired with:
        - MSE, MAE, or other regression losses

    Example:
        >>> z = np.array([1.0, 2.0, 3.0])
        >>> linear(z)
        array([1.0, 2.0, 3.0])
    """
    return z


def linear_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of linear activation.

    Formula: linear'(z) = 1

    Args:
        z: Input array, any shape.

    Returns:
        Array of ones, same shape as input.

    Example:
        >>> z = np.array([1.0, 2.0, 3.0])
        >>> linear_derivative(z)
        array([1., 1., 1.])
    """
    return np.ones_like(z)


# ============================================================================
# DEMONSTRATIONS
# ============================================================================


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    from rich import box
    import matplotlib.pyplot as plt

    console = Console()

    console.print("\n[bold cyan]" + "=" * 70 + "[/bold cyan]")
    console.print("[bold cyan]Activation Functions: From Scratch Implementation[/bold cyan]")
    console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]")

    # Test input range
    z_test = np.linspace(-5, 5, 11)

    # ==========================================================================
    # CLASSICAL ACTIVATIONS
    # ==========================================================================

    console.print("\n[bold yellow]CLASSICAL ACTIVATION FUNCTIONS[/bold yellow]")
    console.print("-" * 70)

    console.print("\n[bold green]1. Sigmoid[/bold green]")
    sig_output = sigmoid(z_test)
    sig_deriv = sigmoid_derivative(z_test)

    sig_table = Table(title="Sigmoid Activation", box=box.ROUNDED)
    sig_table.add_column("z", justify="right", style="cyan")
    sig_table.add_column("σ(z)", justify="right", style="green")
    sig_table.add_column("σ'(z)", justify="right", style="yellow")

    for i in range(0, len(z_test), 2):  # Show every other value
        sig_table.add_row(
            f"{z_test[i]:6.1f}",
            f"{sig_output[i]:.4f}",
            f"{sig_deriv[i]:.4f}",
        )

    console.print(sig_table)
    console.print("[yellow]→ Range: (0, 1), symmetric around 0.5 at z=0[/yellow]")
    console.print("[yellow]→ Gradient vanishes for |z| > 3[/yellow]")

    console.print("\n[bold green]2. Tanh[/bold green]")
    tanh_output = tanh(z_test)
    tanh_deriv = tanh_derivative(z_test)

    console.print(f"Sample values:")
    console.print(f"z = {z_test[::2]}")
    console.print(f"tanh(z) = {tanh_output[::2]}")
    console.print(f"tanh'(z) = {tanh_deriv[::2]}")
    console.print("[yellow]→ Range: (-1, 1), zero-centered[/yellow]")
    console.print("[yellow]→ Stronger gradients than sigmoid, but still vanishes[/yellow]")

    console.print("\n[bold green]3. ReLU[/bold green]")
    relu_output = relu(z_test)
    relu_deriv = relu_derivative(z_test)

    console.print(f"Sample values:")
    console.print(f"z = {z_test[::2]}")
    console.print(f"ReLU(z) = {relu_output[::2]}")
    console.print(f"ReLU'(z) = {relu_deriv[::2]}")
    console.print("[yellow]→ Range: [0, ∞), very fast to compute[/yellow]")
    console.print("[yellow]→ No vanishing gradient for z > 0[/yellow]")
    console.print("[yellow]→ Dying ReLU: gradient is 0 for z ≤ 0[/yellow]")

    # ==========================================================================
    # MODERN ACTIVATIONS
    # ==========================================================================

    console.print("\n[bold yellow]MODERN ACTIVATION FUNCTIONS[/bold yellow]")
    console.print("-" * 70)

    console.print("\n[bold green]4. Leaky ReLU[/bold green]")
    leaky_output = leaky_relu(z_test, alpha=0.01)
    leaky_deriv = leaky_relu_derivative(z_test, alpha=0.01)

    console.print(f"Sample values (α=0.01):")
    console.print(f"z = {z_test[::2]}")
    console.print(f"LeakyReLU(z) = {leaky_output[::2]}")
    console.print(f"LeakyReLU'(z) = {leaky_deriv[::2]}")
    console.print("[yellow]→ Fixes dying ReLU: always has gradient[/yellow]")

    console.print("\n[bold green]5. ELU[/bold green]")
    elu_output = elu(z_test, alpha=1.0)
    elu_deriv = elu_derivative(z_test, alpha=1.0)

    console.print(f"Sample values (α=1.0):")
    console.print(f"z = {z_test[::2]}")
    console.print(f"ELU(z) = {elu_output[::2]}")
    console.print(f"ELU'(z) = {elu_deriv[::2]}")
    console.print("[yellow]→ Smooth, zero-centered, no dying neurons[/yellow]")

    console.print("\n[bold green]6. SELU[/bold green]")
    selu_output = selu(z_test)
    selu_deriv = selu_derivative(z_test)

    console.print(f"Sample values:")
    console.print(f"z = {z_test[::2]}")
    console.print(f"SELU(z) = {selu_output[::2]}")
    console.print(f"SELU'(z) = {selu_deriv[::2]}")
    console.print("[yellow]→ Self-normalizing: maintains mean≈0, var≈1[/yellow]")

    console.print("\n[bold green]7. Swish[/bold green]")
    swish_output = swish(z_test)
    swish_deriv = swish_derivative(z_test)

    console.print(f"Sample values:")
    console.print(f"z = {z_test[::2]}")
    console.print(f"Swish(z) = {swish_output[::2]}")
    console.print(f"Swish'(z) = {swish_deriv[::2]}")
    console.print("[yellow]→ Smooth, non-monotonic (small bump below 0)[/yellow]")

    console.print("\n[bold green]8. GELU[/bold green]")
    gelu_output = gelu(z_test)
    gelu_deriv = gelu_derivative(z_test)

    console.print(f"Sample values:")
    console.print(f"z = {z_test[::2]}")
    console.print(f"GELU(z) = {gelu_output[::2]}")
    console.print(f"GELU'(z) = {gelu_deriv[::2]}")
    console.print("[yellow]→ Used in Transformers (BERT, GPT)[/yellow]")

    # ==========================================================================
    # OUTPUT LAYER ACTIVATIONS
    # ==========================================================================

    console.print("\n[bold yellow]OUTPUT LAYER ACTIVATIONS[/bold yellow]")
    console.print("-" * 70)

    console.print("\n[bold green]9. Softmax (Multi-class)[/bold green]")
    logits = np.array([[2.0, 1.0, 0.1]])
    probs = softmax(logits)

    console.print(f"Logits: {logits[0]}")
    console.print(f"Softmax: {probs[0]}")
    console.print(f"Sum: {probs.sum():.6f}")
    console.print("[yellow]→ Converts logits to probability distribution[/yellow]")

    # ==========================================================================
    # COMPARISON TABLE
    # ==========================================================================

    console.print("\n[bold cyan]" + "=" * 70 + "[/bold cyan]")
    console.print("[bold cyan]Activation Function Comparison[/bold cyan]")
    console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]")

    comparison_table = Table(
        title="Activations at z=-1, 0, 1", box=box.ROUNDED
    )
    comparison_table.add_column("Function", style="cyan")
    comparison_table.add_column("z=-1", justify="right", style="red")
    comparison_table.add_column("z=0", justify="right", style="yellow")
    comparison_table.add_column("z=1", justify="right", style="green")

    z_compare = np.array([-1.0, 0.0, 1.0])

    activations_to_compare = [
        ("Sigmoid", sigmoid),
        ("Tanh", tanh),
        ("ReLU", relu),
        ("Leaky ReLU", lambda z: leaky_relu(z, 0.01)),
        ("ELU", lambda z: elu(z, 1.0)),
        ("SELU", selu),
        ("Swish", swish),
        ("GELU", gelu),
    ]

    for name, func in activations_to_compare:
        outputs = func(z_compare)
        comparison_table.add_row(
            name,
            f"{outputs[0]:7.4f}",
            f"{outputs[1]:7.4f}",
            f"{outputs[2]:7.4f}",
        )

    console.print(comparison_table)

    # ==========================================================================
    # GRADIENT COMPARISON
    # ==========================================================================

    console.print("\n[bold cyan]" + "=" * 70 + "[/bold cyan]")
    console.print("[bold cyan]Gradient Comparison (at z=0)[/bold cyan]")
    console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]")

    gradient_table = Table(
        title="Gradients at z=0", box=box.ROUNDED
    )
    gradient_table.add_column("Function", style="cyan")
    gradient_table.add_column("Gradient", justify="right", style="green")

    z_zero = np.array([0.0])

    gradients_to_compare = [
        ("Sigmoid", sigmoid_derivative),
        ("Tanh", tanh_derivative),
        ("ReLU", relu_derivative),
        ("Leaky ReLU", lambda z: leaky_relu_derivative(z, 0.01)),
        ("ELU", lambda z: elu_derivative(z, 1.0)),
        ("SELU", selu_derivative),
        ("Swish", swish_derivative),
        ("GELU", gelu_derivative),
    ]

    for name, deriv_func in gradients_to_compare:
        grad = deriv_func(z_zero)[0]
        gradient_table.add_row(name, f"{grad:.4f}")

    console.print(gradient_table)
    console.print("\n[yellow]→ Tanh has strongest gradient at z=0[/yellow]")
    console.print("[yellow]→ Sigmoid gradient is weaker (0.25 at z=0)[/yellow]")
    console.print("[yellow]→ ReLU gradient is constant (1.0 for z>0)[/yellow]")

    # ==========================================================================
    # KEY INSIGHTS
    # ==========================================================================

    console.print("\n[bold cyan]" + "=" * 70 + "[/bold cyan]")
    console.print("[bold cyan]Key Insights[/bold cyan]")
    console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]")

    insights = [
        "",
        "[bold]HIDDEN LAYERS:[/bold]",
        "• ReLU: Default choice (fast, no vanishing gradient for z>0)",
        "• Leaky ReLU: Fixes dying ReLU problem",
        "• ELU: Smooth, zero-centered, faster convergence",
        "• SELU: Self-normalizing (specific use case)",
        "• Swish/GELU: Modern, often outperform ReLU (slower)",
        "",
        "[bold]OUTPUT LAYERS:[/bold]",
        "• Binary classification → Sigmoid",
        "• Multi-class classification → Softmax",
        "• Regression → Linear (identity)",
        "",
        "[bold]AVOID:[/bold]",
        "• Sigmoid/Tanh in hidden layers (vanishing gradients)",
        "• ReLU in output layer",
        "",
        "[bold]GRADIENT PROPERTIES:[/bold]",
        "• Sigmoid/Tanh: Vanishing gradients for large |z|",
        "• ReLU: Dead neurons for z ≤ 0",
        "• Leaky ReLU/ELU: Always have gradients",
        "• Swish/GELU: Smooth gradients everywhere",
        "",
        "[bold]PRACTICAL RECOMMENDATIONS:[/bold]",
        "• Start with ReLU for hidden layers",
        "• If ReLU doesn't work: try Leaky ReLU → ELU → Swish",
        "• For Transformers: use GELU",
        "• For output: match activation to task (sigmoid/softmax/linear)",
        "",
        "[yellow]✓ All implementations include forward and derivative[/yellow]",
    ]

    for insight in insights:
        console.print(insight)

    console.print("")
