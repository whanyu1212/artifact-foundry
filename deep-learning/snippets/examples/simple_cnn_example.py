"""
Simple CNN Example - Synthetic Data

Demonstrates CNN from scratch on a simple synthetic dataset.

Task: Classify images by position of bright square
- Class 0: Bright square in top-left
- Class 1: Bright square in top-right
- Class 2: Bright square in bottom-left
- Class 3: Bright square in bottom-right

This toy problem demonstrates:
- CNN can learn spatial patterns
- Convolution detects local features
- Pooling provides translation invariance
- Full CNN training pipeline
"""

import numpy as np
import sys
from pathlib import Path

# Add CNN directory to path
cnn_dir = Path(__file__).parent.parent / "cnn"
sys.path.insert(0, str(cnn_dir))

from cnn_network import CNN


def generate_square_dataset(n_samples: int = 1000, img_size: int = 16, square_size: int = 4):
    """
    Generate synthetic dataset with squares in different quadrants.

    Args:
        n_samples: Number of samples to generate
        img_size: Size of square images
        square_size: Size of bright square

    Returns:
        X: Images, shape (n_samples, 1, img_size, img_size)
        y: Labels, shape (4, n_samples) - one-hot encoded
    """
    X = np.zeros((n_samples, 1, img_size, img_size))
    y = np.zeros((4, n_samples))

    half = img_size // 2

    for i in range(n_samples):
        # Randomly choose quadrant
        quadrant = np.random.randint(0, 4)

        # Determine position based on quadrant
        if quadrant == 0:  # Top-left
            row, col = 2, 2
        elif quadrant == 1:  # Top-right
            row, col = 2, half + 2
        elif quadrant == 2:  # Bottom-left
            row, col = half + 2, 2
        else:  # Bottom-right
            row, col = half + 2, half + 2

        # Add bright square
        X[i, 0, row:row+square_size, col:col+square_size] = 1.0

        # Add some noise
        X[i, 0] += np.random.randn(img_size, img_size) * 0.1

        # One-hot encode label
        y[quadrant, i] = 1

    return X, y


def one_hot_decode(y: np.ndarray) -> np.ndarray:
    """Convert one-hot to class indices."""
    return np.argmax(y, axis=0)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CNN FROM SCRATCH - SYNTHETIC DATA EXAMPLE")
    print("=" * 70)

    # ==========================================================================
    # GENERATE DATA
    # ==========================================================================

    print("\n[1/4] Generating synthetic dataset...")

    np.random.seed(42)

    # Generate training and test data
    X_train, y_train = generate_square_dataset(n_samples=800, img_size=16)
    X_test, y_test = generate_square_dataset(n_samples=200, img_size=16)

    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    print(f"  Image size: {X_train.shape[2]}×{X_train.shape[3]}")
    print(f"  Classes: 4 (square position quadrants)")

    # Show sample images
    print("\n  Sample images (first 4 training samples):")
    for i in range(4):
        label = one_hot_decode(y_train[:, i:i+1])[0]
        print(f"    Sample {i}: Label={label} (quadrant)")

    # ==========================================================================
    # BUILD CNN
    # ==========================================================================

    print("\n[2/4] Building CNN architecture...")

    cnn = CNN(input_shape=(16, 16, 1))

    # Smaller network for this simple task
    cnn.add_conv_layer(16, filter_size=3, padding=1, activation='relu')
    cnn.add_pool_layer(pool_size=2)
    cnn.add_conv_layer(32, filter_size=3, padding=1, activation='relu')
    cnn.add_pool_layer(pool_size=2)
    cnn.add_flatten()
    cnn.add_dense_layer(32, activation='relu')
    cnn.add_dense_layer(4, activation='softmax')

    cnn.compile(loss='categorical_crossentropy')

    cnn.summary()

    # ==========================================================================
    # TRAIN CNN
    # ==========================================================================

    print("\n[3/4] Training CNN...")
    print("  (This may take a minute on CPU)\n")

    history = cnn.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        learning_rate=0.01,
        verbose=True,
        validation_data=(X_test, y_test)
    )

    # ==========================================================================
    # EVALUATE
    # ==========================================================================

    print("\n[4/4] Evaluating CNN...")

    # Predictions on test set
    y_pred = cnn.predict(X_test)
    predictions = np.argmax(y_pred, axis=0)
    labels = one_hot_decode(y_test)

    # Accuracy
    test_acc = np.mean(predictions == labels)

    print(f"\nFinal Test Accuracy: {test_acc:.2%}")

    # Show some predictions
    print("\nSample Predictions:")
    print("  Sample | True | Predicted | Correct?")
    print("  " + "-" * 40)
    for i in range(10):
        true_label = labels[i]
        pred_label = predictions[i]
        correct = "✓" if pred_label == true_label else "✗"
        print(f"     {i:2d}  |  {true_label}   |     {pred_label}     |    {correct}")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nTask: Classify square position (4 quadrants)")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"\nFinal Results:")
    print(f"  Training accuracy: {history['accuracy'][-1]:.2%}")
    print(f"  Test accuracy: {test_acc:.2%}")

    if test_acc >= 0.90:
        print("\n✓ CNN successfully learned spatial patterns!")
        print("  The network learned to detect which quadrant contains the bright square.")
    elif test_acc >= 0.70:
        print("\n⚠ CNN learned some patterns but could improve")
        print("  Try training longer or adjusting hyperparameters.")
    else:
        print("\n✗ CNN didn't learn well")
        print("  This simple task should be easy - check implementation or hyperparameters.")

    print("\n" + "=" * 70)
    print("CNN FROM SCRATCH WORKS! 🎉")
    print("=" * 70)
    print("\nYou've now implemented and trained a CNN from scratch using only NumPy!")
    print("Key achievements:")
    print("  ✓ Implemented 2D convolution (forward & backward)")
    print("  ✓ Implemented max pooling (forward & backward)")
    print("  ✓ Built complete CNN architecture")
    print("  ✓ Trained on spatial classification task")
    print("\nNext steps:")
    print("  - Try on real MNIST dataset")
    print("  - Experiment with different architectures")
    print("  - Visualize learned filters")
    print("  - Compare to PyTorch implementation")
    print("")
