"""
Simple Neural Network Examples

Quick examples showing how to use the from-scratch neural network
for common tasks.
"""

import numpy as np
import sys
from pathlib import Path

# Add core directory to path
core_dir = Path(__file__).parent.parent / "core"
sys.path.insert(0, str(core_dir))

from neural_network import NeuralNetwork, Dense, train_test_split, one_hot_encode


def example_binary_classification():
    """Example: Binary classification on a simple dataset."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Binary Classification")
    print("=" * 60)

    # Create simple binary classification dataset
    # Class 0: points near origin
    # Class 1: points far from origin
    np.random.seed(42)

    X_class0 = np.random.randn(2, 50) * 0.5
    X_class1 = np.random.randn(2, 50) * 0.5 + 2

    X = np.hstack([X_class0, X_class1])
    y = np.hstack([np.zeros((1, 50)), np.ones((1, 50))])

    # Build network
    nn = NeuralNetwork()
    nn.add_layer(Dense(2, 8, activation='relu'))
    nn.add_layer(Dense(8, 1, activation='sigmoid'))
    nn.compile(loss='binary_crossentropy', init='he')

    print("\nNetwork:")
    nn.summary()

    # Train
    print("Training...")
    history = nn.fit(X, y, epochs=100, batch_size=32, learning_rate=0.01, verbose=False)

    # Evaluate
    accuracy = nn._compute_accuracy(X, y)
    print(f"\nFinal Accuracy: {accuracy:.2%}")
    print(f"Final Loss: {history['loss'][-1]:.4f}")


def example_regression():
    """Example: Regression - learning a quadratic function."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Regression")
    print("=" * 60)

    # Create dataset: y = x^2
    np.random.seed(42)
    X = np.linspace(-2, 2, 100).reshape(1, -1)
    y = X ** 2 + np.random.randn(1, 100) * 0.1  # Add noise

    # Build network
    nn = NeuralNetwork()
    nn.add_layer(Dense(1, 16, activation='relu'))
    nn.add_layer(Dense(16, 16, activation='relu'))
    nn.add_layer(Dense(16, 1, activation='linear'))
    nn.compile(loss='mse', init='he')

    print("\nNetwork:")
    nn.summary()

    # Train
    print("Training...")
    history = nn.fit(X, y, epochs=200, batch_size=32, learning_rate=0.01, verbose=False)

    # Evaluate
    y_pred = nn.predict(X)
    mse = np.mean((y - y_pred) ** 2)

    print(f"\nFinal MSE: {mse:.6f}")
    print(f"Final Loss: {history['loss'][-1]:.6f}")

    # Show some predictions
    print("\nSample predictions:")
    for i in [0, 25, 50, 75, 99]:
        print(f"  x={X[0,i]:5.2f}, true={y[0,i]:5.2f}, pred={y_pred[0,i]:5.2f}")


def example_multiclass():
    """Example: Multi-class classification."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Multi-class Classification")
    print("=" * 60)

    # Create 4-class dataset
    np.random.seed(42)

    # 4 clusters in corners
    X_c0 = np.random.randn(2, 40) * 0.3 + np.array([[1], [1]])
    X_c1 = np.random.randn(2, 40) * 0.3 + np.array([[-1], [1]])
    X_c2 = np.random.randn(2, 40) * 0.3 + np.array([[-1], [-1]])
    X_c3 = np.random.randn(2, 40) * 0.3 + np.array([[1], [-1]])

    X = np.hstack([X_c0, X_c1, X_c2, X_c3])
    y_int = np.array([0]*40 + [1]*40 + [2]*40 + [3]*40)
    y = one_hot_encode(y_int, n_classes=4)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build network
    nn = NeuralNetwork()
    nn.add_layer(Dense(2, 16, activation='relu'))
    nn.add_layer(Dense(16, 16, activation='relu'))
    nn.add_layer(Dense(16, 4, activation='softmax'))
    nn.compile(loss='categorical_crossentropy', init='he')

    print(f"\nData:")
    print(f"  Training: {X_train.shape[1]} samples")
    print(f"  Test: {X_test.shape[1]} samples")

    print("\nNetwork:")
    nn.summary()

    # Train
    print("Training...")
    history = nn.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        learning_rate=0.01,
        verbose=False,
        validation_data=(X_test, y_test)
    )

    # Evaluate
    train_acc = nn._compute_accuracy(X_train, y_train)
    test_acc = nn._compute_accuracy(X_test, y_test)

    print(f"\nTraining Accuracy: {train_acc:.2%}")
    print(f"Test Accuracy: {test_acc:.2%}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NEURAL NETWORK FROM SCRATCH - SIMPLE EXAMPLES")
    print("=" * 60)

    example_binary_classification()
    example_regression()
    example_multiclass()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60 + "\n")
