"""
Neural Network - From Scratch Implementation

A complete multi-layer perceptron (MLP) implementation using only NumPy.
Demonstrates the fundamental mechanics of neural networks without any
deep learning frameworks.

Key Components:
- Dense (fully connected) layers with flexible activation functions
- Forward propagation: computing predictions
- Backpropagation: computing gradients via chain rule
- Weight initialization strategies (Xavier, He)
- Training loop with mini-batch gradient descent
- Loss functions (MSE, Binary/Categorical Cross-Entropy)
- Model evaluation and prediction

Mathematical Foundation:
- Forward: a^(l) = σ(W^(l) · a^(l-1) + b^(l))
- Backward: δ^(l) = (W^(l+1))^T · δ^(l+1) ⊙ σ'(z^(l))
- Update: W := W - α · ∂L/∂W

This implementation prioritizes clarity and educational value over performance.
Each step of forward/backward propagation is explicitly shown with comments.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Dict, Any
import copy


# ============================================================================
# LAYER CLASSES
# ============================================================================


class Dense:
    """
    Fully connected (dense) layer.

    A dense layer performs: output = activation(W · input + b)
    where W is the weight matrix and b is the bias vector.

    Parameters:
        input_size: Number of input features
        output_size: Number of neurons in this layer
        activation: Activation function name ('relu', 'sigmoid', 'tanh', etc.)

    Attributes:
        W: Weight matrix, shape (output_size, input_size)
        b: Bias vector, shape (output_size, 1)
        activation_name: Name of activation function
        activation: Forward activation function
        activation_derivative: Derivative of activation function

    Cache (stored during forward pass, used in backward pass):
        z: Pre-activation values
        a_prev: Input from previous layer

    Gradients (computed during backward pass):
        dW: Gradient of loss w.r.t. weights
        db: Gradient of loss w.r.t. biases
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: str = 'relu'
    ):
        """Initialize dense layer with random weights."""
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation

        # Initialize weights and biases
        # Will be properly initialized by the network using init_weights()
        self.W = np.random.randn(output_size, input_size) * 0.01
        self.b = np.zeros((output_size, 1))

        # Set activation functions
        self._set_activation(activation)

        # Cache for backpropagation
        self.z = None  # Pre-activation: z = W·a_prev + b
        self.a_prev = None  # Input from previous layer

        # Gradients
        self.dW = None
        self.db = None

    def _set_activation(self, name: str) -> None:
        """Set activation function and its derivative."""
        # Import activation functions from the activation_functions module
        # Handle both relative and absolute imports
        try:
            from .activation_functions import (
                relu, relu_derivative,
                sigmoid, sigmoid_derivative,
                tanh, tanh_derivative,
                leaky_relu, leaky_relu_derivative,
                elu, elu_derivative,
                swish, swish_derivative,
                gelu, gelu_derivative,
                linear, linear_derivative,
                softmax, softmax_derivative
            )
        except ImportError:
            from activation_functions import (
                relu, relu_derivative,
                sigmoid, sigmoid_derivative,
                tanh, tanh_derivative,
                leaky_relu, leaky_relu_derivative,
                elu, elu_derivative,
                swish, swish_derivative,
                gelu, gelu_derivative,
                linear, linear_derivative,
                softmax, softmax_derivative
            )

        activations = {
            'relu': (relu, relu_derivative),
            'sigmoid': (sigmoid, sigmoid_derivative),
            'tanh': (tanh, tanh_derivative),
            'leaky_relu': (
                lambda z: leaky_relu(z, 0.01),
                lambda z: leaky_relu_derivative(z, 0.01)
            ),
            'elu': (
                lambda z: elu(z, 1.0),
                lambda z: elu_derivative(z, 1.0)
            ),
            'swish': (swish, swish_derivative),
            'gelu': (gelu, gelu_derivative),
            'linear': (linear, linear_derivative),
            'softmax': (softmax, softmax_derivative),
        }

        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")

        self.activation, self.activation_derivative = activations[name]

    def forward(self, a_prev: np.ndarray) -> np.ndarray:
        """
        Forward propagation through the layer.

        Computation:
            z = W · a_prev + b  (linear transformation)
            a = σ(z)            (activation)

        Args:
            a_prev: Activations from previous layer, shape (input_size, batch_size)

        Returns:
            a: Activations of this layer, shape (output_size, batch_size)
        """
        # Cache input for backpropagation
        self.a_prev = a_prev

        # Linear transformation: z = W·a + b
        # W: (output_size, input_size)
        # a_prev: (input_size, batch_size)
        # z: (output_size, batch_size)
        self.z = np.dot(self.W, a_prev) + self.b

        # Apply activation function
        a = self.activation(self.z)

        return a

    def backward(self, da: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward propagation through the layer.

        Computes gradients and updates parameters.

        Chain rule:
            dz = da ⊙ σ'(z)              (element-wise with activation derivative)
            dW = (1/m) · dz · a_prev^T   (gradient w.r.t. weights)
            db = (1/m) · sum(dz)         (gradient w.r.t. biases)
            da_prev = W^T · dz           (gradient to pass to previous layer)

        Args:
            da: Gradient of loss w.r.t. activations, shape (output_size, batch_size)
            learning_rate: Learning rate for parameter updates

        Returns:
            da_prev: Gradient to pass to previous layer, shape (input_size, batch_size)
        """
        batch_size = da.shape[1]

        # Compute dz: gradient w.r.t. pre-activation
        # dz = da ⊙ σ'(z)
        dz = da * self.activation_derivative(self.z)

        # Compute gradients w.r.t. parameters
        # dW = (1/m) · dz · a_prev^T
        self.dW = (1 / batch_size) * np.dot(dz, self.a_prev.T)

        # db = (1/m) · sum(dz, axis=1)
        # keepdims=True preserves shape (output_size, 1)
        self.db = (1 / batch_size) * np.sum(dz, axis=1, keepdims=True)

        # Compute gradient w.r.t. previous layer activations
        # da_prev = W^T · dz
        da_prev = np.dot(self.W.T, dz)

        # Update parameters using gradient descent
        # W := W - α · dW
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

        return da_prev


# ============================================================================
# NEURAL NETWORK CLASS
# ============================================================================


class NeuralNetwork:
    """
    Multi-layer perceptron (feedforward neural network).

    A flexible neural network that supports:
    - Arbitrary number of layers with different activations
    - Multiple loss functions
    - Weight initialization strategies
    - Mini-batch gradient descent
    - Training history tracking

    Architecture:
        Input → Dense(h1) → ... → Dense(hn) → Output

    Example:
        >>> # Binary classification network
        >>> nn = NeuralNetwork()
        >>> nn.add_layer(Dense(2, 4, activation='relu'))
        >>> nn.add_layer(Dense(4, 1, activation='sigmoid'))
        >>> nn.compile(loss='binary_crossentropy', init='he')
        >>> nn.fit(X_train, y_train, epochs=100, batch_size=32)
    """

    def __init__(self):
        """Initialize empty neural network."""
        self.layers: List[Dense] = []
        self.loss_function = None
        self.loss_derivative = None
        self.history: Dict[str, List[float]] = {
            'loss': [],
            'accuracy': []
        }

    def add_layer(self, layer: Dense) -> None:
        """
        Add a layer to the network.

        Args:
            layer: Dense layer instance
        """
        self.layers.append(layer)

    def compile(self, loss: str = 'mse', init: str = 'xavier') -> None:
        """
        Compile the network by setting loss function and initializing weights.

        Args:
            loss: Loss function name ('mse', 'binary_crossentropy', 'categorical_crossentropy')
            init: Weight initialization strategy ('xavier', 'he', 'random')
        """
        # Set loss function
        self._set_loss_function(loss)

        # Initialize weights
        self._init_weights(init)

    def _set_loss_function(self, name: str) -> None:
        """Set the loss function and its derivative."""
        if name == 'mse':
            # Mean Squared Error: L = (1/2m) Σ(y - ŷ)²
            self.loss_function = lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2)
            # Derivative: ∂L/∂ŷ = -(y - ŷ) = ŷ - y
            self.loss_derivative = lambda y_true, y_pred: y_pred - y_true

        elif name == 'binary_crossentropy':
            # Binary Cross-Entropy: L = -(1/m) Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
            def bce_loss(y_true, y_pred):
                # Clip predictions to avoid log(0)
                y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
                return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

            # Derivative: ∂L/∂ŷ = -(y/ŷ - (1-y)/(1-ŷ))
            # Simplified when using sigmoid output: ŷ - y
            def bce_derivative(y_true, y_pred):
                y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
                return -(y_true / y_pred - (1 - y_true) / (1 - y_pred))

            self.loss_function = bce_loss
            self.loss_derivative = bce_derivative

        elif name == 'categorical_crossentropy':
            # Categorical Cross-Entropy: L = -(1/m) Σ Σ y_i · log(ŷ_i)
            def cce_loss(y_true, y_pred):
                y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
                return -np.mean(np.sum(y_true * np.log(y_pred), axis=0))

            # Derivative: ∂L/∂ŷ = -y/ŷ
            # Simplified when using softmax output: ŷ - y
            def cce_derivative(y_true, y_pred):
                y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
                return -y_true / y_pred

            self.loss_function = cce_loss
            self.loss_derivative = cce_derivative

        else:
            raise ValueError(f"Unknown loss function: {name}")

    def _init_weights(self, strategy: str) -> None:
        """
        Initialize weights using specified strategy.

        Strategies:
            - xavier (Glorot): W ~ N(0, sqrt(2/(n_in + n_out)))
              Good for sigmoid/tanh activations

            - he: W ~ N(0, sqrt(2/n_in))
              Good for ReLU activations

            - random: W ~ N(0, 0.01)
              Simple random initialization
        """
        for layer in self.layers:
            n_in = layer.input_size
            n_out = layer.output_size

            if strategy == 'xavier':
                # Xavier/Glorot initialization
                limit = np.sqrt(2.0 / (n_in + n_out))
                layer.W = np.random.randn(n_out, n_in) * limit

            elif strategy == 'he':
                # He initialization (good for ReLU)
                limit = np.sqrt(2.0 / n_in)
                layer.W = np.random.randn(n_out, n_in) * limit

            elif strategy == 'random':
                # Simple random initialization
                layer.W = np.random.randn(n_out, n_in) * 0.01

            else:
                raise ValueError(f"Unknown initialization strategy: {strategy}")

            # Biases always initialized to zero
            layer.b = np.zeros((n_out, 1))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through entire network.

        Sequentially applies each layer's forward pass.

        Args:
            X: Input data, shape (input_size, batch_size)

        Returns:
            Final predictions, shape (output_size, batch_size)
        """
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray, learning_rate: float) -> None:
        """
        Backward propagation through entire network.

        Computes gradients via chain rule and updates all parameters.

        Args:
            y_true: True labels, shape (output_size, batch_size)
            y_pred: Predictions from forward pass, shape (output_size, batch_size)
            learning_rate: Learning rate for gradient descent
        """
        # Compute initial gradient: ∂L/∂ŷ
        da = self.loss_derivative(y_true, y_pred)

        # Backpropagate through layers in reverse order
        for layer in reversed(self.layers):
            da = layer.backward(da, learning_rate)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        verbose: bool = True,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, List[float]]:
        """
        Train the neural network.

        Uses mini-batch gradient descent to optimize network parameters.

        Args:
            X: Training data, shape (n_features, n_samples)
            y: Training labels, shape (n_outputs, n_samples)
            epochs: Number of complete passes through training data
            batch_size: Number of samples per gradient update
            learning_rate: Step size for gradient descent
            verbose: Whether to print training progress
            validation_data: Optional (X_val, y_val) for validation tracking

        Returns:
            Training history dictionary with 'loss' and 'accuracy' lists
        """
        n_samples = X.shape[1]

        # Reset history
        self.history = {'loss': [], 'accuracy': []}
        if validation_data is not None:
            self.history['val_loss'] = []
            self.history['val_accuracy'] = []

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[:, indices]
            y_shuffled = y[:, indices]

            # Mini-batch training
            epoch_losses = []

            for i in range(0, n_samples, batch_size):
                # Get mini-batch
                X_batch = X_shuffled[:, i:i+batch_size]
                y_batch = y_shuffled[:, i:i+batch_size]

                # Forward pass
                y_pred = self.forward(X_batch)

                # Compute loss
                loss = self.loss_function(y_batch, y_pred)
                epoch_losses.append(loss)

                # Backward pass and update
                self.backward(y_batch, y_pred, learning_rate)

            # Record average loss for epoch
            avg_loss = np.mean(epoch_losses)
            self.history['loss'].append(avg_loss)

            # Compute accuracy on full training set
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

                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch:3d}: "
                          f"loss={avg_loss:.4f}, acc={train_acc:.4f}, "
                          f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
            else:
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch:3d}: loss={avg_loss:.4f}, acc={train_acc:.4f}")

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Input data, shape (n_features, n_samples)

        Returns:
            Predictions, shape (n_outputs, n_samples)
        """
        return self.forward(X)

    def _compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy.

        For binary classification: predictions > 0.5 → class 1
        For multi-class: argmax of predictions

        Args:
            X: Input data
            y: True labels

        Returns:
            Accuracy as a float in [0, 1]
        """
        y_pred = self.predict(X)

        # Binary classification
        if y_pred.shape[0] == 1:
            predictions = (y_pred > 0.5).astype(int)
            accuracy = np.mean(predictions == y)
        # Multi-class classification
        else:
            predictions = np.argmax(y_pred, axis=0)
            labels = np.argmax(y, axis=0)
            accuracy = np.mean(predictions == labels)

        return accuracy

    def summary(self) -> None:
        """Print network architecture summary."""
        print("\n" + "=" * 60)
        print("Neural Network Summary")
        print("=" * 60)

        total_params = 0

        for i, layer in enumerate(self.layers):
            n_params = layer.W.size + layer.b.size
            total_params += n_params

            print(f"\nLayer {i+1}: Dense")
            print(f"  Input size:  {layer.input_size}")
            print(f"  Output size: {layer.output_size}")
            print(f"  Activation:  {layer.activation_name}")
            print(f"  Parameters:  {n_params:,} (W: {layer.W.shape}, b: {layer.b.shape})")

        print(f"\n{'=' * 60}")
        print(f"Total parameters: {total_params:,}")
        print("=" * 60 + "\n")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and test sets.

    Args:
        X: Features, shape (n_features, n_samples)
        y: Labels, shape (n_outputs, n_samples)
        test_size: Proportion of data for test set
        random_state: Random seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[1]
    n_test = int(n_samples * test_size)

    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train = X[:, train_indices]
    X_test = X[:, test_indices]
    y_train = y[:, train_indices]
    y_test = y[:, test_indices]

    return X_train, X_test, y_train, y_test


def one_hot_encode(y: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Convert integer labels to one-hot encoded vectors.

    Args:
        y: Integer labels, shape (n_samples,)
        n_classes: Number of classes

    Returns:
        One-hot encoded labels, shape (n_classes, n_samples)

    Example:
        >>> y = np.array([0, 1, 2, 1])
        >>> one_hot_encode(y, n_classes=3)
        array([[1., 0., 0., 0.],
               [0., 1., 0., 1.],
               [0., 0., 1., 0.]])
    """
    n_samples = y.shape[0]
    one_hot = np.zeros((n_classes, n_samples))
    one_hot[y, np.arange(n_samples)] = 1
    return one_hot


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
    console.print("[bold cyan]Neural Network: From Scratch Implementation[/bold cyan]")
    console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]")

    # ==========================================================================
    # EXAMPLE 1: Binary Classification (XOR Problem)
    # ==========================================================================

    console.print("\n[bold yellow]Example 1: Binary Classification (XOR Problem)[/bold yellow]")
    console.print("-" * 70)

    console.print("\n[bold]The XOR Problem:[/bold]")
    console.print("Classic non-linearly separable problem that requires hidden layers.")
    console.print("Input: 2 binary values, Output: XOR result")

    # Generate XOR dataset
    X_xor = np.array([[0, 0, 1, 1],
                      [0, 1, 0, 1]])
    y_xor = np.array([[0, 1, 1, 0]])  # XOR truth table

    console.print(f"\nDataset:")
    xor_table = Table(box=box.ROUNDED)
    xor_table.add_column("X1", justify="center", style="cyan")
    xor_table.add_column("X2", justify="center", style="cyan")
    xor_table.add_column("XOR", justify="center", style="green")

    for i in range(X_xor.shape[1]):
        xor_table.add_row(
            str(X_xor[0, i]),
            str(X_xor[1, i]),
            str(int(y_xor[0, i]))
        )

    console.print(xor_table)

    # Build network
    console.print("\n[bold]Network Architecture:[/bold]")
    console.print("Input (2) → Hidden (4, ReLU) → Output (1, Sigmoid)")

    nn_xor = NeuralNetwork()
    nn_xor.add_layer(Dense(2, 4, activation='relu'))
    nn_xor.add_layer(Dense(4, 1, activation='sigmoid'))
    nn_xor.compile(loss='binary_crossentropy', init='he')

    nn_xor.summary()

    # Train
    console.print("[bold]Training...[/bold]")
    history_xor = nn_xor.fit(
        X_xor, y_xor,
        epochs=1000,
        batch_size=4,
        learning_rate=0.1,
        verbose=False
    )

    # Evaluate
    y_pred_xor = nn_xor.predict(X_xor)

    console.print("\n[bold green]Results:[/bold green]")
    results_table = Table(title="XOR Predictions", box=box.ROUNDED)
    results_table.add_column("X1", justify="center", style="cyan")
    results_table.add_column("X2", justify="center", style="cyan")
    results_table.add_column("True", justify="center", style="yellow")
    results_table.add_column("Predicted", justify="center", style="green")
    results_table.add_column("Probability", justify="center", style="magenta")

    for i in range(X_xor.shape[1]):
        pred_class = int(y_pred_xor[0, i] > 0.5)
        results_table.add_row(
            str(X_xor[0, i]),
            str(X_xor[1, i]),
            str(int(y_xor[0, i])),
            str(pred_class),
            f"{y_pred_xor[0, i]:.4f}"
        )

    console.print(results_table)

    final_acc_xor = nn_xor._compute_accuracy(X_xor, y_xor)
    console.print(f"\n[bold]Final Training Accuracy: {final_acc_xor:.2%}[/bold]")
    console.print(f"[bold]Final Loss: {history_xor['loss'][-1]:.4f}[/bold]")

    # ==========================================================================
    # EXAMPLE 2: Multi-class Classification (Synthetic Dataset)
    # ==========================================================================

    console.print("\n[bold yellow]Example 2: Multi-class Classification[/bold yellow]")
    console.print("-" * 70)

    console.print("\n[bold]Synthetic 3-class Dataset:[/bold]")
    console.print("3 clusters in 2D space, 60 samples per class")

    # Generate synthetic multi-class data
    np.random.seed(42)

    # Class 0: centered at (1, 1)
    X_class0 = np.random.randn(2, 60) * 0.5 + np.array([[1], [1]])

    # Class 1: centered at (-1, 1)
    X_class1 = np.random.randn(2, 60) * 0.5 + np.array([[-1], [1]])

    # Class 2: centered at (0, -1)
    X_class2 = np.random.randn(2, 60) * 0.5 + np.array([[0], [-1]])

    X_multi = np.hstack([X_class0, X_class1, X_class2])
    y_multi_int = np.array([0]*60 + [1]*60 + [2]*60)
    y_multi = one_hot_encode(y_multi_int, n_classes=3)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_multi, y_multi, test_size=0.2, random_state=42
    )

    console.print(f"\nTraining set: {X_train.shape[1]} samples")
    console.print(f"Test set: {X_test.shape[1]} samples")

    # Build network with softmax output
    console.print("\n[bold]Network Architecture:[/bold]")
    console.print("Input (2) → Hidden1 (8, ReLU) → Hidden2 (8, ReLU) → Output (3, Softmax)")

    nn_multi = NeuralNetwork()
    nn_multi.add_layer(Dense(2, 8, activation='relu'))
    nn_multi.add_layer(Dense(8, 8, activation='relu'))
    nn_multi.add_layer(Dense(8, 3, activation='softmax'))  # Softmax for multi-class
    nn_multi.compile(loss='categorical_crossentropy', init='he')

    # Train
    console.print("\n[bold]Training...[/bold]")
    history_multi = nn_multi.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        learning_rate=0.01,
        verbose=False,
        validation_data=(X_test, y_test)
    )

    # Evaluate
    train_acc_multi = nn_multi._compute_accuracy(X_train, y_train)
    test_acc_multi = nn_multi._compute_accuracy(X_test, y_test)

    console.print(f"\n[bold green]Results:[/bold green]")
    console.print(f"Training Accuracy: {train_acc_multi:.2%}")
    console.print(f"Test Accuracy: {test_acc_multi:.2%}")
    console.print(f"Final Training Loss: {history_multi['loss'][-1]:.4f}")
    console.print(f"Final Validation Loss: {history_multi['val_loss'][-1]:.4f}")

    # ==========================================================================
    # EXAMPLE 3: Regression Problem
    # ==========================================================================

    console.print("\n[bold yellow]Example 3: Regression (Sine Wave)[/bold yellow]")
    console.print("-" * 70)

    console.print("\n[bold]Task:[/bold]")
    console.print("Learn to approximate f(x) = sin(x) using a neural network")

    # Generate sine wave data
    X_reg = np.linspace(-np.pi, np.pi, 100).reshape(1, -1)
    y_reg = np.sin(X_reg)

    # Add some noise
    y_reg += np.random.randn(1, 100) * 0.1

    # Build regression network
    console.print("\n[bold]Network Architecture:[/bold]")
    console.print("Input (1) → Hidden (10, Tanh) → Output (1, Linear)")

    nn_reg = NeuralNetwork()
    nn_reg.add_layer(Dense(1, 10, activation='tanh'))
    nn_reg.add_layer(Dense(10, 1, activation='linear'))
    nn_reg.compile(loss='mse', init='xavier')

    # Train
    console.print("\n[bold]Training...[/bold]")
    history_reg = nn_reg.fit(
        X_reg, y_reg,
        epochs=500,
        batch_size=10,
        learning_rate=0.01,
        verbose=False
    )

    # Evaluate
    y_pred_reg = nn_reg.predict(X_reg)
    mse = np.mean((y_reg - y_pred_reg) ** 2)

    console.print(f"\n[bold green]Results:[/bold green]")
    console.print(f"Final MSE: {mse:.6f}")
    console.print(f"Final Loss: {history_reg['loss'][-1]:.6f}")

    # ==========================================================================
    # KEY INSIGHTS
    # ==========================================================================

    console.print("\n[bold cyan]" + "=" * 70 + "[/bold cyan]")
    console.print("[bold cyan]Key Insights: How Neural Networks Work[/bold cyan]")
    console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]")

    insights = [
        "",
        "[bold]FORWARD PROPAGATION:[/bold]",
        "• Each layer: a^(l) = σ(W^(l) · a^(l-1) + b^(l))",
        "• Information flows from input → hidden → output",
        "• Non-linear activations enable learning complex patterns",
        "",
        "[bold]BACKPROPAGATION:[/bold]",
        "• Chain rule: ∂L/∂W^(l) = ∂L/∂a^(L) · ∂a^(L)/∂z^(L) · ... · ∂z^(l)/∂W^(l)",
        "• Gradients computed in reverse: output → hidden → input",
        "• Each layer computes: dW, db (parameter gradients), da_prev (for previous layer)",
        "",
        "[bold]WEIGHT INITIALIZATION:[/bold]",
        "• Xavier: Good for sigmoid/tanh (sqrt(2/(n_in + n_out)))",
        "• He: Good for ReLU (sqrt(2/n_in))",
        "• Proper initialization prevents vanishing/exploding gradients",
        "",
        "[bold]LOSS FUNCTIONS:[/bold]",
        "• MSE: Regression tasks",
        "• Binary Cross-Entropy: Binary classification",
        "• Categorical Cross-Entropy: Multi-class classification",
        "",
        "[bold]TRAINING PROCESS:[/bold]",
        "• Mini-batch gradient descent: balance speed and stability",
        "• Learning rate: controls step size (too large → divergence, too small → slow)",
        "• Epochs: complete passes through training data",
        "",
        "[bold]ARCHITECTURE CHOICES:[/bold]",
        "• More layers: can learn more complex patterns (but harder to train)",
        "• More neurons: more capacity (but risk overfitting)",
        "• Activation functions: ReLU for hidden, sigmoid/softmax for output",
        "",
        "[yellow]✓ This implementation shows the core mechanics without abstractions[/yellow]",
        "[yellow]✓ Every step of forward/backward propagation is explicit[/yellow]",
        "[yellow]✓ Ready to extend with: momentum, dropout, batch norm, etc.[/yellow]",
    ]

    for insight in insights:
        console.print(insight)

    console.print("")
