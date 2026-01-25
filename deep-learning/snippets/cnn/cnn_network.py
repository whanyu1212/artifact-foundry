"""
Convolutional Neural Network - From Scratch Implementation

A complete CNN implementation using only NumPy, building on the convolution
operations from convolution.py.

Key Components:
- Conv2D layer: 2D convolutional layer with multiple filters
- MaxPool2D layer: Max pooling for downsampling
- Flatten layer: Convert 2D feature maps to 1D for fully connected layers
- CNN class: Complete convolutional neural network

Architecture Pattern:
    Input → [Conv-ReLU-Pool]×N → Flatten → [Dense-ReLU]×M → Output

This implementation demonstrates:
- How convolution and dense layers work together
- Managing 2D spatial dimensions through network
- Parameter initialization for CNNs
- Full forward and backward propagation through CNN

Educational focus: Every step is explicit and commented to show what's happening.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import sys
from pathlib import Path

# Import convolution operations
try:
    from convolution import (
        conv2d_forward, conv2d_backward,
        max_pool_forward, max_pool_backward,
        check_conv_output_size, check_pool_output_size
    )
except ImportError:
    from .convolution import (
        conv2d_forward, conv2d_backward,
        max_pool_forward, max_pool_backward,
        check_conv_output_size, check_pool_output_size
    )

# Import Dense layer from core
core_dir = Path(__file__).parent.parent / "core"
sys.path.insert(0, str(core_dir))

try:
    from neural_network import Dense
except ImportError:
    # If running as script, we'll define a simple Dense layer here
    pass


# ============================================================================
# CNN LAYER CLASSES
# ============================================================================


class Conv2D:
    """
    2D Convolutional layer.

    Applies convolution operation to learn spatial features from images.

    Forward: output = ReLU(conv(input, filters) + bias)
    Backward: Compute gradients via backpropagation

    Parameters:
        in_channels: Number of input channels (e.g., 3 for RGB)
        out_channels: Number of filters to learn
        filter_size: Size of square filters (filter_size × filter_size)
        stride: Stride for convolution
        padding: Zero padding around input
        activation: Activation function name ('relu', 'none')

    Attributes:
        W: Filters, shape (out_channels, in_channels, filter_h, filter_w)
        b: Biases, shape (out_channels,)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        activation: str = 'relu'
    ):
        """Initialize convolutional layer."""
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.activation_name = activation

        # Initialize weights and biases
        # Will be properly initialized using He/Xavier initialization
        fan_in = in_channels * filter_size * filter_size
        self.W = np.random.randn(out_channels, in_channels, filter_size, filter_size) / np.sqrt(fan_in)
        self.b = np.zeros(out_channels)

        # Activation function
        self._set_activation(activation)

        # Cache for backpropagation
        self.cache = None
        self.cache_activation = None

        # Gradients
        self.dW = None
        self.db = None

    def _set_activation(self, name: str):
        """Set activation function."""
        if name == 'relu':
            self.activation = lambda x: np.maximum(0, x)
            self.activation_derivative = lambda x: (x > 0).astype(float)
        elif name == 'none' or name == 'linear':
            self.activation = lambda x: x
            self.activation_derivative = lambda x: np.ones_like(x)
        else:
            raise ValueError(f"Unknown activation: {name}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through convolutional layer.

        Process:
            1. Convolve input with filters: z = conv(x, W) + b
            2. Apply activation: a = activation(z)

        Args:
            x: Input, shape (batch, in_channels, height, width)

        Returns:
            Output, shape (batch, out_channels, out_h, out_w)
        """
        # Convolution
        z, self.cache = conv2d_forward(x, self.W, self.b, self.stride, self.padding)

        # Activation
        self.cache_activation = z.copy()  # Store pre-activation for backward
        a = self.activation(z)

        return a

    def backward(self, da: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass through convolutional layer.

        Computes gradients and updates parameters.

        Args:
            da: Gradient of loss w.r.t. activations
            learning_rate: Learning rate for parameter updates

        Returns:
            Gradient of loss w.r.t. input (for previous layer)
        """
        # Gradient through activation
        dz = da * self.activation_derivative(self.cache_activation)

        # Gradient through convolution
        dx, self.dW, self.db = conv2d_backward(dz, self.cache)

        # Update parameters
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

        return dx

    def output_shape(self, input_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate output spatial dimensions.

        Args:
            input_shape: (height, width) of input

        Returns:
            (out_height, out_width) of output
        """
        h, w = input_shape
        out_h = check_conv_output_size(h, self.filter_size, self.stride, self.padding)
        out_w = check_conv_output_size(w, self.filter_size, self.stride, self.padding)
        return out_h, out_w


class MaxPool2D:
    """
    2D Max Pooling layer.

    Reduces spatial dimensions by taking maximum in each pooling window.

    Purpose:
        - Downsampling (reduce computation in later layers)
        - Translation invariance (small shifts don't change output)
        - Noise reduction (suppress non-maximum activations)

    Parameters:
        pool_size: Size of pooling window (pool_size × pool_size)
        stride: Stride for pooling (typically equals pool_size for non-overlapping)

    No learnable parameters (just a fixed operation).
    """

    def __init__(self, pool_size: int = 2, stride: Optional[int] = None):
        """Initialize max pooling layer."""
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size

        # Cache for backpropagation
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through max pooling.

        Args:
            x: Input, shape (batch, channels, height, width)

        Returns:
            Output, shape (batch, channels, out_h, out_w)
        """
        out, self.cache = max_pool_forward(x, self.pool_size, self.pool_size, self.stride)
        return out

    def backward(self, dout: np.ndarray, learning_rate: float = None) -> np.ndarray:
        """
        Backward pass through max pooling.

        No parameters to update (just pass gradients backward).

        Args:
            dout: Gradient of loss w.r.t. output
            learning_rate: Unused (no parameters to update)

        Returns:
            Gradient of loss w.r.t. input
        """
        dx = max_pool_backward(dout, self.cache)
        return dx

    def output_shape(self, input_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate output spatial dimensions.

        Args:
            input_shape: (height, width) of input

        Returns:
            (out_height, out_width) of output
        """
        h, w = input_shape
        out_h = check_pool_output_size(h, self.pool_size, self.stride)
        out_w = check_pool_output_size(w, self.pool_size, self.stride)
        return out_h, out_w


class Flatten:
    """
    Flatten layer: Convert 2D feature maps to 1D vector.

    Transforms (batch, channels, height, width) → (batch, channels × height × width)

    Used between convolutional layers and fully connected layers.

    Example:
        Input: (32, 64, 7, 7) - 32 samples, 64 channels, 7×7 spatial
        Output: (32, 3136) - 32 samples, 3136 features

    No learnable parameters.
    """

    def __init__(self):
        """Initialize flatten layer."""
        self.input_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: Flatten spatial dimensions.

        Args:
            x: Input, shape (batch, channels, height, width)

        Returns:
            Output, shape (batch, channels × height × width)
        """
        self.input_shape = x.shape
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1).T  # Transpose to match Dense layer input format

    def backward(self, dout: np.ndarray, learning_rate: float = None) -> np.ndarray:
        """
        Backward pass: Reshape gradient back to 2D spatial format.

        Args:
            dout: Gradient, shape (channels × height × width, batch)
            learning_rate: Unused (no parameters)

        Returns:
            Gradient reshaped to (batch, channels, height, width)
        """
        dout_reshaped = dout.T.reshape(self.input_shape)
        return dout_reshaped


# ============================================================================
# CNN NETWORK CLASS
# ============================================================================


class CNN:
    """
    Convolutional Neural Network.

    Flexible CNN architecture supporting:
    - Multiple Conv-ReLU-Pool blocks
    - Flatten layer
    - Fully connected layers
    - Various activations and loss functions

    Typical architecture:
        Input (28×28×1 MNIST image)
        ↓
        Conv2D(1→32, 3×3) + ReLU → (28×28×32)
        ↓
        MaxPool(2×2) → (14×14×32)
        ↓
        Conv2D(32→64, 3×3) + ReLU → (14×14×64)
        ↓
        MaxPool(2×2) → (7×7×64)
        ↓
        Flatten → (3136,)
        ↓
        Dense(3136→128) + ReLU
        ↓
        Dense(128→10) + Softmax
        ↓
        Output (10 classes)

    Example:
        >>> cnn = CNN(input_shape=(28, 28, 1))
        >>> cnn.add_conv_layer(32, filter_size=3, padding=1)
        >>> cnn.add_pool_layer(pool_size=2)
        >>> cnn.add_conv_layer(64, filter_size=3, padding=1)
        >>> cnn.add_pool_layer(pool_size=2)
        >>> cnn.add_flatten()
        >>> cnn.add_dense_layer(128, activation='relu')
        >>> cnn.add_dense_layer(10, activation='softmax')
        >>> cnn.compile(loss='categorical_crossentropy')
    """

    def __init__(self, input_shape: Tuple[int, int, int]):
        """
        Initialize CNN.

        Args:
            input_shape: (height, width, channels) of input images
        """
        self.input_shape = input_shape
        self.layers: List[Any] = []
        self.layer_types: List[str] = []

        # Track current shape through network
        self.current_h, self.current_w, self.current_c = input_shape

        # Loss function
        self.loss_function = None
        self.loss_derivative = None

        # Training history
        self.history: Dict[str, List[float]] = {'loss': [], 'accuracy': []}

    def add_conv_layer(
        self,
        out_channels: int,
        filter_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        activation: str = 'relu'
    ):
        """
        Add convolutional layer.

        Args:
            out_channels: Number of filters
            filter_size: Filter size (filter_size × filter_size)
            stride: Convolution stride
            padding: Zero padding
            activation: Activation function ('relu', 'none')
        """
        layer = Conv2D(
            self.current_c, out_channels,
            filter_size, stride, padding, activation
        )

        self.layers.append(layer)
        self.layer_types.append('conv')

        # Update current shape
        self.current_h, self.current_w = layer.output_shape((self.current_h, self.current_w))
        self.current_c = out_channels

    def add_pool_layer(self, pool_size: int = 2, stride: Optional[int] = None):
        """
        Add max pooling layer.

        Args:
            pool_size: Pooling window size
            stride: Pooling stride (defaults to pool_size)
        """
        layer = MaxPool2D(pool_size, stride)

        self.layers.append(layer)
        self.layer_types.append('pool')

        # Update current shape
        self.current_h, self.current_w = layer.output_shape((self.current_h, self.current_w))

    def add_flatten(self):
        """Add flatten layer (convert 2D to 1D)."""
        layer = Flatten()

        self.layers.append(layer)
        self.layer_types.append('flatten')

        # After flatten, we're in 1D space
        # current_c will now represent flattened dimension
        self.current_c = self.current_h * self.current_w * self.current_c

    def add_dense_layer(self, output_size: int, activation: str = 'relu'):
        """
        Add fully connected (dense) layer.

        Must be after flatten layer.

        Args:
            output_size: Number of neurons
            activation: Activation function
        """
        # Use Dense layer from neural_network.py
        layer = Dense(self.current_c, output_size, activation)

        self.layers.append(layer)
        self.layer_types.append('dense')

        self.current_c = output_size

    def compile(self, loss: str = 'categorical_crossentropy'):
        """
        Compile network by setting loss function.

        Args:
            loss: Loss function ('categorical_crossentropy', 'mse', 'binary_crossentropy')
        """
        if loss == 'categorical_crossentropy':
            def cce_loss(y_true, y_pred):
                y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
                return -np.mean(np.sum(y_true * np.log(y_pred), axis=0))

            def cce_derivative(y_true, y_pred):
                y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
                return -y_true / y_pred

            self.loss_function = cce_loss
            self.loss_derivative = cce_derivative

        elif loss == 'mse':
            self.loss_function = lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2)
            self.loss_derivative = lambda y_true, y_pred: y_pred - y_true

        elif loss == 'binary_crossentropy':
            def bce_loss(y_true, y_pred):
                y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
                return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

            def bce_derivative(y_true, y_pred):
                y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
                return -(y_true / y_pred - (1 - y_true) / (1 - y_pred))

            self.loss_function = bce_loss
            self.loss_derivative = bce_derivative

        else:
            raise ValueError(f"Unknown loss: {loss}")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through entire network.

        Args:
            X: Input images, shape (batch, channels, height, width)

        Returns:
            Predictions, shape depends on output layer
        """
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray, learning_rate: float):
        """
        Backward propagation through entire network.

        Args:
            y_true: True labels
            y_pred: Predictions from forward pass
            learning_rate: Learning rate for updates
        """
        # Initial gradient
        da = self.loss_derivative(y_true, y_pred)

        # Backpropagate through layers in reverse
        for layer in reversed(self.layers):
            da = layer.backward(da, learning_rate)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        verbose: bool = True,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, List[float]]:
        """
        Train the CNN.

        Args:
            X: Training images, shape (n_samples, channels, height, width)
            y: Training labels, shape (n_outputs, n_samples)
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            verbose: Print progress
            validation_data: Optional (X_val, y_val)

        Returns:
            Training history
        """
        n_samples = X.shape[0]

        # Reset history
        self.history = {'loss': [], 'accuracy': []}
        if validation_data is not None:
            self.history['val_loss'] = []
            self.history['val_accuracy'] = []

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[:, indices] if y.ndim == 2 else y[indices]

            # Mini-batch training
            epoch_losses = []

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[:, i:i+batch_size] if y.ndim == 2 else y_shuffled[i:i+batch_size]

                # Forward pass
                y_pred = self.forward(X_batch)

                # Compute loss
                loss = self.loss_function(y_batch, y_pred)
                epoch_losses.append(loss)

                # Backward pass
                self.backward(y_batch, y_pred, learning_rate)

            # Record metrics
            avg_loss = np.mean(epoch_losses)
            self.history['loss'].append(avg_loss)

            # Accuracy (if classification)
            train_acc = self._compute_accuracy(X, y)
            self.history['accuracy'].append(train_acc)

            # Validation metrics
            if validation_data is not None:
                X_val, y_val = validation_data
                y_val_pred = self.forward(X_val)
                val_loss = self.loss_function(y_val, y_val_pred)
                val_acc = self._compute_accuracy(X_val, y_val)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_acc)

                if verbose and epoch % 1 == 0:
                    print(f"Epoch {epoch+1:3d}: "
                          f"loss={avg_loss:.4f}, acc={train_acc:.4f}, "
                          f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
            else:
                if verbose and epoch % 1 == 0:
                    print(f"Epoch {epoch+1:3d}: loss={avg_loss:.4f}, acc={train_acc:.4f}")

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.forward(X)

    def _compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        y_pred = self.predict(X)

        if y_pred.shape[0] == 1:
            # Binary classification
            predictions = (y_pred > 0.5).astype(int)
            accuracy = np.mean(predictions == y)
        else:
            # Multi-class classification
            predictions = np.argmax(y_pred, axis=0)
            labels = np.argmax(y, axis=0) if y.ndim == 2 else y
            accuracy = np.mean(predictions == labels)

        return accuracy

    def summary(self):
        """Print network architecture summary."""
        print("\n" + "=" * 70)
        print("CNN Architecture Summary")
        print("=" * 70)
        print(f"\nInput shape: {self.input_shape} (H×W×C)")

        h, w, c = self.input_shape
        total_params = 0

        for i, (layer, layer_type) in enumerate(zip(self.layers, self.layer_types)):
            print(f"\nLayer {i+1}: {layer_type.upper()}")

            if layer_type == 'conv':
                n_params = layer.W.size + layer.b.size
                total_params += n_params
                h, w = layer.output_shape((h, w))
                c = layer.out_channels
                print(f"  Filters: {layer.out_channels}")
                print(f"  Filter size: {layer.filter_size}×{layer.filter_size}")
                print(f"  Output: {h}×{w}×{c}")
                print(f"  Parameters: {n_params:,}")

            elif layer_type == 'pool':
                h, w = layer.output_shape((h, w))
                print(f"  Pool size: {layer.pool_size}×{layer.pool_size}")
                print(f"  Output: {h}×{w}×{c}")
                print(f"  Parameters: 0")

            elif layer_type == 'flatten':
                flattened = h * w * c
                print(f"  Output: {flattened}")
                print(f"  Parameters: 0")
                h, w, c = flattened, 1, 1

            elif layer_type == 'dense':
                n_params = layer.W.size + layer.b.size
                total_params += n_params
                print(f"  Neurons: {layer.output_size}")
                print(f"  Activation: {layer.activation_name}")
                print(f"  Parameters: {n_params:,}")

        print(f"\n{'=' * 70}")
        print(f"Total parameters: {total_params:,}")
        print("=" * 70 + "\n")


# ============================================================================
# DEMONSTRATION
# ============================================================================


if __name__ == "__main__":
    from rich.console import Console

    console = Console()

    console.print("\n[bold cyan]" + "=" * 70 + "[/bold cyan]")
    console.print("[bold cyan]CNN: From Scratch Implementation[/bold cyan]")
    console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]")

    # Create a simple CNN for MNIST-like data
    console.print("\n[bold yellow]Building CNN for 28×28 grayscale images (MNIST-style)[/bold yellow]")

    cnn = CNN(input_shape=(28, 28, 1))
    cnn.add_conv_layer(32, filter_size=3, padding=1, activation='relu')
    cnn.add_pool_layer(pool_size=2)
    cnn.add_conv_layer(64, filter_size=3, padding=1, activation='relu')
    cnn.add_pool_layer(pool_size=2)
    cnn.add_flatten()
    cnn.add_dense_layer(128, activation='relu')
    cnn.add_dense_layer(10, activation='softmax')
    cnn.compile(loss='categorical_crossentropy')

    cnn.summary()

    console.print("\n[green]✓ CNN architecture created successfully![/green]")
    console.print("\n[yellow]Ready to train on MNIST or other image datasets![/yellow]\n")
