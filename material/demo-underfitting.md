# Demo Underfitting vs Overfitting

---

### Python Code

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# 1. Generate synthetic data based on a Cosine function
np.random.seed(42)  # For reproducible results


def true_function(X):
    return np.cos(1.5 * np.pi * X)


# 30 random data points with some added Gaussian noise
n_samples = 30
X = np.sort(np.random.rand(n_samples))
y = true_function(X) + np.random.randn(n_samples) * 0.15

# Continuous X-values for plotting smooth curves
X_test = np.linspace(0, 1, 200)

# 2. Define the polynomial degrees to compare
# Degree 1 = Underfitting, Degree 4 = Good Fit, Degree 15 = Overfitting
degrees = [1, 4, 15]
titles = [
    "Underfitting (Degree 1)",
    "Optimal Fit (Degree 4)",
    "Overfitting (Degree 15)",
]

plt.figure(figsize=(15, 5))

for i, degree in enumerate(degrees):
    ax = plt.subplot(1, 3, i + 1)

    # Build a pipeline: convert features to polynomial, then fit linear regression
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline(
        [
            ("polynomial_features", polynomial_features),
            ("linear_regression", linear_regression),
        ]
    )

    # Train model
    pipeline.fit(X[:, np.newaxis], y)

    # Predict values
    y_pred = pipeline.predict(X_test[:, np.newaxis])

    # Plot results
    plt.plot(
        X_test,
        true_function(X_test),
        label="True Function",
        color="green",
        linestyle="--",
    )
    plt.plot(
        X_test, y_pred, label="Model Prediction", color="red", linewidth=2
    )
    plt.scatter(
        X,
        y,
        edgecolor="black",
        facecolor="royalblue",
        s=35,
        label="Training Data",
    )

    plt.xlabel("X")
    plt.ylabel("y")
    plt.xlim((0, 1))
    plt.ylim((-1.5, 1.5))
    plt.title(titles[i], fontsize=13, fontweight="bold")
    plt.legend(loc="lower left")

plt.tight_layout()
plt.show()
```

---

### Code Explanation

#### 1. Data Generation
* **`true_function(X)`**: We choose a non-linear target function: $y = \cos(1.5 \pi X)$. 
* **Noise**: Real-world data is never clean. We add Gaussian noise (`np.random.randn(...) * 0.15`) to the 30 sampled points so the points bounce slightly around the true curve.

#### 2. Machine Learning Pipeline (`Pipeline`)
* We use a `Pipeline` from `sklearn` combining:
  * **`PolynomialFeatures(degree)`**: Transforms a single input feature $X$ into powers up to the chosen degree (e.g., for degree 2, it creates $[X, X^2]$).
  * **`LinearRegression()`**: Fits standard linear regression onto these transformed features.

#### 3. Breakdown of the 3 Cases

| Degree | Label | What is Happening? | Machine Learning Term |
| :--- | :--- | :--- | :--- |
| **Degree 1** | **Underfitting** | The model is a straight line ($y = mx + b$). It is too simple to capture the curved trend of the data. It performs poorly on both training and test data. | **High Bias**: Strong incorrect assumptions about the data. |
| **Degree 4** | **Optimal Fit** | The model has enough flexibility (a 4th-order polynomial) to capture the true underlying wave shape without being tricked by the noise. | **Balanced**: Low bias and low variance. Good generalization. |
| **Degree 15** | **Overfitting** | The model has 15 degrees of freedom. It passes through almost every training dot, fitting the random noise rather than the trend. Notice how it swings wildly at the boundaries. | **High Variance**: Overly sensitive to the specific training points. Fails on new data. |