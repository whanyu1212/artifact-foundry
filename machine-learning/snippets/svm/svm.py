"""
Support Vector Machine (SVM) Classifier

Implements linear and kernel SVM for binary classification using a simplified
primal formulation with gradient descent.

Mathematical Foundation:
- Objective: min ½||w||² + C Σᵢ max(0, 1 - yᵢ(w^T xᵢ + b))
- Hinge loss: L(y, ŷ) = max(0, 1 - y·ŷ)
- Kernel trick: K(x, x') = φ(x)^T φ(x')

Algorithm (Linear SVM):
1. Initialize weights w and bias b
2. For each epoch:
   - For each sample (xᵢ, yᵢ):
     - If yᵢ(w^T xᵢ + b) < 1: Update w and b (violated margin)
     - Else: Only regularize w (correct with margin)
3. Decision: sign(w^T x + b)

Note: Production SVM uses dual formulation with SMO algorithm. This is a
simplified educational implementation demonstrating core concepts.

Time Complexity:
- Training: O(n·d·epochs) for linear, O(n²·epochs) for kernel
- Prediction: O(d) for linear, O(n_sv·d) for kernel
"""

import numpy as np
from typing import Literal, Optional, Callable


class SVM:
    """
    Support Vector Machine for binary classification.

    Implements linear and kernel SVM using hinge loss with L2 regularization.
    Uses (sub)gradient descent for optimization.

    Attributes:
        kernel (str): Kernel type ('linear', 'poly', 'rbf').
        C (float): Regularization parameter (inverse of regularization strength).
        gamma (float): Kernel coefficient for 'rbf' and 'poly'.
        degree (int): Degree for polynomial kernel.
        coef0 (float): Independent term for polynomial kernel.
        learning_rate (float): Step size for gradient descent.
        n_epochs (int): Number of training epochs.
        w_ (np.ndarray): Weights for linear SVM, shape (n_features,).
        b_ (float): Bias term.
        alpha_ (np.ndarray): Dual coefficients for kernel SVM, shape (n_samples,).
        X_train_ (np.ndarray): Stored training features for kernel SVM.
        y_train_ (np.ndarray): Stored training labels.

    Mathematical Notes:
        Primal formulation:
        min_{w,b} ½||w||² + C Σᵢ ξᵢ
        s.t. yᵢ(w^T xᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0

        Unconstrained (hinge loss):
        min_{w,b} ½||w||² + C Σᵢ max(0, 1 - yᵢ(w^T xᵢ + b))

    Important:
        - Labels must be {-1, +1} (not {0, 1})
        - Always scale features before training
        - Use cross-validation to tune C and gamma
    """

    def __init__(
        self,
        kernel: Literal["linear", "poly", "rbf"] = "linear",
        C: float = 1.0,
        gamma: Optional[float] = None,
        degree: int = 3,
        coef0: float = 0.0,
        learning_rate: float = 0.001,
        n_epochs: int = 1000,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize SVM classifier.

        Args:
            kernel: Kernel type.
                'linear': K(x, x') = x^T x'
                'poly': K(x, x') = (gamma·x^T x' + coef0)^degree
                'rbf': K(x, x') = exp(-gamma·||x - x'||²)
            C: Regularization parameter (positive float).
                Large C: Hard margin (fewer violations, risk overfitting)
                Small C: Soft margin (more violations, larger margin)
            gamma: Kernel coefficient. If None, defaults to 1/n_features.
                For RBF: controls width (large gamma = narrow, complex boundary)
            degree: Degree for polynomial kernel (only used if kernel='poly').
            coef0: Independent term in polynomial kernel.
            learning_rate: Step size for gradient descent.
            n_epochs: Number of passes over training data.
            random_state: Random seed for reproducibility.

        Raises:
            ValueError: If C <= 0 or degree < 1.
        """
        if C <= 0:
            raise ValueError("C must be positive")
        if degree < 1:
            raise ValueError("degree must be at least 1")

        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state

        # Learned parameters
        self.w_: Optional[np.ndarray] = None
        self.b_: float = 0.0
        self.alpha_: Optional[np.ndarray] = None
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVM":
        """
        Fit SVM classifier using (sub)gradient descent.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training labels {-1, +1}, shape (n_samples,).

        Returns:
            self: Fitted classifier instance.

        Algorithm:
            For each epoch:
                Shuffle training data
                For each sample (xᵢ, yᵢ):
                    If margin violated (yᵢ·ŷᵢ < 1):
                        Update: w -= lr·(w - C·yᵢ·xᵢ)
                                b -= lr·(-C·yᵢ)
                    Else:
                        Regularize: w -= lr·w

        Notes:
            - Linear kernel: Direct optimization of w and b
            - Kernel methods: Maintain dual coefficients (simplified)
        """
        if not np.all(np.isin(y, [-1, 1])):
            raise ValueError("Labels must be -1 or +1")

        n_samples, n_features = X.shape

        # Set gamma default if not specified
        if self.gamma is None:
            self.gamma = 1.0 / n_features

        # Store training data for kernel methods
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

        # Random state for shuffling
        rng = np.random.RandomState(self.random_state)

        if self.kernel == "linear":
            # Linear SVM: Optimize weights directly
            self._fit_linear(X, y, rng)
        else:
            # Kernel SVM: Simplified dual approach
            self._fit_kernel(X, y, rng)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Args:
            X: Features to predict, shape (n_samples, n_features).

        Returns:
            predictions: Predicted labels {-1, +1}, shape (n_samples,).

        Formula:
            Linear: ŷ = sign(w^T x + b)
            Kernel: ŷ = sign(Σᵢ αᵢyᵢ K(xᵢ, x) + b)
        """
        if self.w_ is None and self.alpha_ is None:
            raise RuntimeError("Model must be fitted before prediction")

        decision_values = self.decision_function(X)
        return np.sign(decision_values)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function values (signed distance to hyperplane).

        Args:
            X: Features, shape (n_samples, n_features).

        Returns:
            decision_values: Signed distances, shape (n_samples,).
                Positive: predicted as +1
                Negative: predicted as -1
                Magnitude: confidence (distance from boundary)
        """
        if self.kernel == "linear":
            # Linear: w^T x + b
            return X @ self.w_ + self.b_
        else:
            # Kernel: Σᵢ αᵢyᵢ K(xᵢ, x) + b
            n_samples = X.shape[0]
            decision = np.zeros(n_samples)

            for i in range(n_samples):
                # Compute kernel with all training points
                kernel_values = self._compute_kernel(self.X_train_, X[i : i + 1])
                # Weighted sum using dual coefficients
                decision[i] = np.sum(self.alpha_ * self.y_train_ * kernel_values.ravel())

            return decision + self.b_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy.

        Args:
            X: Test features, shape (n_samples, n_features).
            y: True labels {-1, +1}, shape (n_samples,).

        Returns:
            accuracy: Fraction of correct predictions.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def _fit_linear(self, X: np.ndarray, y: np.ndarray, rng: np.random.RandomState) -> None:
        """
        Fit linear SVM using subgradient descent.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training labels, shape (n_samples,).
            rng: Random number generator for shuffling.

        Algorithm:
            Minimize: ½||w||² + C Σᵢ max(0, 1 - yᵢ(w^T xᵢ + b))

            Subgradient:
            - If yᵢ(w^T xᵢ + b) < 1 (margin violation):
                ∂L/∂w = w - C·yᵢ·xᵢ
                ∂L/∂b = -C·yᵢ
            - Else (correct with margin):
                ∂L/∂w = w
                ∂L/∂b = 0
        """
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.w_ = np.zeros(n_features)
        self.b_ = 0.0

        # Training loop
        for epoch in range(self.n_epochs):
            # Shuffle data for stochastic gradient descent
            indices = rng.permutation(n_samples)

            for idx in indices:
                x_i = X[idx]
                y_i = y[idx]

                # Compute margin: yᵢ(w^T xᵢ + b)
                margin = y_i * (np.dot(self.w_, x_i) + self.b_)

                # Check if margin is violated (hinge loss > 0)
                if margin < 1:
                    # Violated: update both w and b
                    # Gradient: w - C·yᵢ·xᵢ for w, -C·yᵢ for b
                    self.w_ -= self.learning_rate * (self.w_ - self.C * y_i * x_i)
                    self.b_ -= self.learning_rate * (-self.C * y_i)
                else:
                    # Not violated: only regularize w
                    # Gradient: w for w, 0 for b
                    self.w_ -= self.learning_rate * self.w_

    def _fit_kernel(self, X: np.ndarray, y: np.ndarray, rng: np.random.RandomState) -> None:
        """
        Fit kernel SVM using simplified dual approach.

        For educational purposes, we use a simplified approach:
        - Maintain dual coefficients alpha_i for each training point
        - Update coefficients based on margin violations
        - Use kernel evaluations for predictions

        Note: This is a simplified version. Production kernel SVM uses
        quadratic programming to solve the dual problem.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training labels, shape (n_samples,).
            rng: Random number generator.
        """
        n_samples = X.shape[0]

        # Initialize dual coefficients (one per training sample)
        self.alpha_ = np.zeros(n_samples)
        self.b_ = 0.0

        # Precompute kernel matrix for efficiency
        # K[i,j] = K(xᵢ, xⱼ)
        K = self._compute_kernel(X, X)

        # Training loop
        for epoch in range(self.n_epochs):
            indices = rng.permutation(n_samples)

            for idx in indices:
                # Decision function for sample idx: Σⱼ αⱼyⱼ K(xⱼ, xᵢ) + b
                decision = np.sum(self.alpha_ * y * K[:, idx]) + self.b_

                # Margin for sample idx: yᵢ·decision
                margin = y[idx] * decision

                # Update rule (simplified Pegasos-like update for kernel SVM)
                if margin < 1:
                    # Margin violated: increase alpha_i
                    self.alpha_[idx] += self.learning_rate * self.C
                else:
                    # Margin satisfied: decay alpha_i (regularization)
                    self.alpha_[idx] *= (1 - self.learning_rate)

                # Clip alpha to [0, C] (box constraint from dual formulation)
                self.alpha_ = np.clip(self.alpha_, 0, self.C)

        # Compute bias from support vectors (simplified approach)
        # Support vectors: samples with 0 < alpha < C
        sv_mask = (self.alpha_ > 1e-5) & (self.alpha_ < self.C - 1e-5)
        if np.any(sv_mask):
            # Average bias over support vectors
            sv_indices = np.where(sv_mask)[0]
            bias_sum = 0.0
            for idx in sv_indices:
                decision_no_bias = np.sum(self.alpha_ * y * K[:, idx])
                bias_sum += y[idx] - decision_no_bias
            self.b_ = bias_sum / len(sv_indices)

    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute kernel matrix K[i,j] = K(X1[i], X2[j]).

        Args:
            X1: First set of samples, shape (n1, n_features).
            X2: Second set of samples, shape (n2, n_features).

        Returns:
            K: Kernel matrix, shape (n1, n2).

        Kernels:
            Linear: K(x, x') = x^T x'
            Polynomial: K(x, x') = (gamma·x^T x' + coef0)^degree
            RBF: K(x, x') = exp(-gamma·||x - x'||²)
        """
        if self.kernel == "linear":
            # Linear kernel: K(x, x') = x^T x'
            # Matrix form: X1 @ X2^T
            return X1 @ X2.T

        elif self.kernel == "poly":
            # Polynomial: K(x, x') = (gamma·x^T x' + coef0)^degree
            linear_kernel = X1 @ X2.T
            return (self.gamma * linear_kernel + self.coef0) ** self.degree

        elif self.kernel == "rbf":
            # RBF (Gaussian): K(x, x') = exp(-gamma·||x - x'||²)
            # Compute squared Euclidean distances efficiently:
            # ||x - x'||² = ||x||² + ||x'||² - 2x^T x'

            # Shape: (n1,) - sum of squares for each row
            X1_sq = np.sum(X1**2, axis=1, keepdims=True)
            # Shape: (n2,) - sum of squares for each row
            X2_sq = np.sum(X2**2, axis=1, keepdims=True)

            # Squared distances: (n1, n2)
            sq_dists = X1_sq + X2_sq.T - 2 * (X1 @ X2.T)

            # RBF kernel: exp(-gamma·distance²)
            return np.exp(-self.gamma * sq_dists)

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def get_support_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get support vectors (for kernel SVM).

        Returns:
            X_sv: Support vector features, shape (n_sv, n_features).
            y_sv: Support vector labels, shape (n_sv,).

        Notes:
            Support vectors: training points with alpha_i > threshold.
            These are the points that define the decision boundary.
        """
        if self.alpha_ is None:
            raise RuntimeError("Only available for kernel SVM after fitting")

        # Support vectors: alpha > small threshold
        sv_mask = self.alpha_ > 1e-5
        return self.X_train_[sv_mask], self.y_train_[sv_mask]


if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_circles, load_breast_cancer
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report
    from rich.console import Console
    from rich.table import Table
    from rich import box
    import matplotlib.pyplot as plt

    console = Console()

    console.print("\n[bold cyan]" + "=" * 70)
    console.print("Support Vector Machine: Linear & Kernel SVM")
    console.print("=" * 70 + "[/bold cyan]")

    # Example 1: Linear SVM on Linearly Separable Data
    console.print("\n[bold yellow]1. Linear SVM (Linearly Separable Data)[/bold yellow]")
    console.print("-" * 70)

    # Generate linearly separable data
    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=42,
    )
    y = 2 * y - 1  # Convert {0,1} to {-1,+1}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Scale features (CRITICAL for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train linear SVM
    svm_linear = SVM(kernel="linear", C=1.0, learning_rate=0.001, n_epochs=1000)
    svm_linear.fit(X_train_scaled, y_train)

    console.print(f"Training Accuracy: [green]{svm_linear.score(X_train_scaled, y_train):.4f}[/green]")
    console.print(f"Test Accuracy: [green]{svm_linear.score(X_test_scaled, y_test):.4f}[/green]")
    console.print(f"Learned weights shape: {svm_linear.w_.shape}")
    console.print(f"Bias term: {svm_linear.b_:.4f}")

    # Example 2: Effect of C (Regularization)
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("2. Effect of Regularization Parameter C")
    console.print("=" * 70 + "[/bold cyan]")

    c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    c_table = Table(title="Regularization Impact", box=box.ROUNDED)
    c_table.add_column("C", justify="center", style="cyan")
    c_table.add_column("Train Acc", justify="right", style="green")
    c_table.add_column("Test Acc", justify="right", style="yellow")
    c_table.add_column("Interpretation", style="white")

    for c_val in c_values:
        svm_c = SVM(kernel="linear", C=c_val, learning_rate=0.001, n_epochs=1000)
        svm_c.fit(X_train_scaled, y_train)

        train_acc = svm_c.score(X_train_scaled, y_train)
        test_acc = svm_c.score(X_test_scaled, y_test)

        if c_val < 1:
            interp = "Soft margin (large)"
        elif c_val == 1:
            interp = "Balanced"
        else:
            interp = "Hard margin (small)"

        c_table.add_row(f"{c_val:.2f}", f"{train_acc:.4f}", f"{test_acc:.4f}", interp)

    console.print(c_table)
    console.print("\n[yellow]→ Small C:[/yellow] Larger margin, more violations (regularization)")
    console.print("[yellow]→ Large C:[/yellow] Smaller margin, fewer violations (risk overfitting)")

    # Example 3: RBF Kernel for Non-Linear Data
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("3. RBF Kernel SVM (Non-Linear Data)")
    console.print("=" * 70 + "[/bold cyan]")

    # Generate non-linearly separable data (circles)
    X_circles, y_circles = make_circles(n_samples=200, noise=0.1, factor=0.3, random_state=42)
    y_circles = 2 * y_circles - 1  # Convert to {-1, +1}

    X_train_nl, X_test_nl, y_train_nl, y_test_nl = train_test_split(
        X_circles, y_circles, test_size=0.3, random_state=42
    )

    # Scale
    scaler_nl = StandardScaler()
    X_train_nl_scaled = scaler_nl.fit_transform(X_train_nl)
    X_test_nl_scaled = scaler_nl.transform(X_test_nl)

    # Compare Linear vs RBF
    kernels_table = Table(title="Kernel Comparison (Non-Linear Data)", box=box.ROUNDED)
    kernels_table.add_column("Kernel", style="cyan")
    kernels_table.add_column("Train Acc", justify="right", style="green")
    kernels_table.add_column("Test Acc", justify="right", style="yellow")

    for kernel_type in ["linear", "rbf"]:
        svm_kernel = SVM(
            kernel=kernel_type, C=1.0, gamma=0.5, learning_rate=0.01, n_epochs=500
        )
        svm_kernel.fit(X_train_nl_scaled, y_train_nl)

        train_acc = svm_kernel.score(X_train_nl_scaled, y_train_nl)
        test_acc = svm_kernel.score(X_test_nl_scaled, y_test_nl)

        kernels_table.add_row(kernel_type.upper(), f"{train_acc:.4f}", f"{test_acc:.4f}")

    console.print(kernels_table)
    console.print("\n[yellow]→ Linear kernel:[/yellow] Fails on non-linear boundaries")
    console.print("[yellow]→ RBF kernel:[/yellow] Handles non-linear patterns effectively")

    # Example 4: Real-World Dataset (Breast Cancer)
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("4. Real Dataset (Breast Cancer)")
    console.print("=" * 70 + "[/bold cyan]")

    cancer = load_breast_cancer()
    X_cancer, y_cancer = cancer.data, cancer.target
    y_cancer = 2 * y_cancer - 1  # Convert to {-1, +1}

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cancer, y_cancer, test_size=0.3, random_state=42
    )

    # Scale
    scaler_c = StandardScaler()
    X_train_c_scaled = scaler_c.fit_transform(X_train_c)
    X_test_c_scaled = scaler_c.transform(X_test_c)

    # Train SVM
    svm_cancer = SVM(kernel="rbf", C=1.0, gamma=0.01, learning_rate=0.01, n_epochs=500)
    svm_cancer.fit(X_train_c_scaled, y_train_c)

    console.print(f"Dataset: {X_cancer.shape[0]} samples, {X_cancer.shape[1]} features")
    console.print(f"Training Accuracy: [green]{svm_cancer.score(X_train_c_scaled, y_train_c):.4f}[/green]")
    console.print(f"Test Accuracy: [green]{svm_cancer.score(X_test_c_scaled, y_test_c):.4f}[/green]")

    # Example 5: Support Vectors
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("5. Support Vectors (RBF Kernel)")
    console.print("=" * 70 + "[/bold cyan]")

    # Train RBF SVM and examine support vectors
    svm_rbf = SVM(kernel="rbf", C=1.0, gamma=0.5, learning_rate=0.01, n_epochs=500)
    svm_rbf.fit(X_train_nl_scaled, y_train_nl)

    X_sv, y_sv = svm_rbf.get_support_vectors()

    console.print(f"Total training samples: {len(X_train_nl_scaled)}")
    console.print(f"Number of support vectors: [yellow]{len(X_sv)}[/yellow]")
    console.print(f"Percentage: [yellow]{100 * len(X_sv) / len(X_train_nl_scaled):.1f}%[/yellow]")
    console.print("\n[yellow]→ Support vectors:[/yellow] Points that define the decision boundary")
    console.print("[yellow]→ Sparse solution:[/yellow] Only fraction of training data needed")

    # Example 6: Validation Against Scikit-Learn
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("6. Validation Against Scikit-Learn (Linear SVM)")
    console.print("=" * 70 + "[/bold cyan]")

    # Our implementation
    our_svm = SVM(kernel="linear", C=1.0, learning_rate=0.001, n_epochs=1000)
    our_svm.fit(X_train_scaled, y_train)
    our_acc = our_svm.score(X_test_scaled, y_test)

    # Scikit-learn (using SGDClassifier for fair comparison with our GD approach)
    from sklearn.linear_model import SGDClassifier

    sklearn_svm = SGDClassifier(loss="hinge", alpha=1 / (len(X_train_scaled) * 1.0), max_iter=1000)
    sklearn_svm.fit(X_train_scaled, y_train)
    sklearn_acc = sklearn_svm.score(X_test_scaled, y_test)

    comparison_table = Table(title="Implementation Comparison", box=box.ROUNDED)
    comparison_table.add_column("Implementation", style="cyan")
    comparison_table.add_column("Accuracy", justify="right", style="green")

    comparison_table.add_row("Our SVM (GD)", f"{our_acc:.4f}")
    comparison_table.add_row("Scikit-Learn (SGD)", f"{sklearn_acc:.4f}")

    console.print(comparison_table)
    console.print("\n[yellow]Note:[/yellow] Our implementation uses simplified primal formulation.")
    console.print("[yellow]Production SVM:[/yellow] Uses dual formulation with SMO algorithm (faster, exact).")

    # Key insights
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("Key Insights")
    console.print("=" * 70 + "[/bold cyan]")

    insights = [
        "• SVM finds maximum margin hyperplane (best separation)",
        "• Soft margin (C parameter) allows violations for non-separable data",
        "• ALWAYS scale features before SVM (distance-based)",
        "• Linear SVM: Fast, interpretable, works for linearly separable data",
        "• Kernel trick: Handles non-linear boundaries without explicit mapping",
        "• RBF kernel: Most versatile (Gaussian, infinite-dimensional space)",
        "• Support vectors: Only boundary points determine decision",
        "• Large C → hard margin (risk overfitting), Small C → soft margin",
        "• Large gamma → complex boundary, Small gamma → smooth boundary",
        "",
        "[yellow]→ Best for:[/yellow] High-dimensional data, clear margin, text/image classification",
        "[yellow]→ Avoid when:[/yellow] Very large datasets (slow), need probabilities, noisy data",
    ]

    for insight in insights:
        console.print(insight)

    console.print("")
