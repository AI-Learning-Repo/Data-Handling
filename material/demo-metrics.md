# Demo: Regression Metrics

### Step 1: Metric Definitions and a Simplified Explanation of $R^2$

#### 1. Mathematical Definitions of Regression Metrics

Let $y_i$ be the actual value, $\hat{y}_i$ be the predicted value, $\bar{y}$ be the mean of actual values, and $n$ be the number of samples.

* **Mean Absolute Error (MAE):** The average magnitude of absolute errors. It treats all errors proportionally.
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

* **Mean Squared Error (MSE):** The average of the squared errors. Squaring penalizes large deviations much more severely than small ones.
  $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

* **Root Mean Squared Error (RMSE):** The square root of MSE. It brings the error unit back to the original unit of the target variable while maintaining the sensitivity to large errors.
  $$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

* **Coefficient of Determination ($R^2$):** Measures the proportion of variance in the target variable explained by the model.
  $$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

---

#### 2. A Simplified Explanation of How $R^2$ Works

Think of $R^2$ as a comparison between your model and a **"baseline model" that knows nothing and always predicts the average ($\bar{y}$)**:

```
Total Variance (SS_tot):   Errors made if you simply guess the average every time.
Residual Variance (SS_res): Errors left over after using your trained model.
```

* **If your model is useless:** $\text{SS}_{\text{res}} \approx \text{SS}_{\text{tot}} \implies R^2 \approx 1 - 1 = \mathbf{0.0}$ (0% of variance explained).
* **If your model is perfect:** $\text{SS}_{\text{res}} = 0 \implies R^2 = 1 - 0 = \mathbf{1.0}$ (100% of variance explained).
* **If your model is worse than guessing the mean:** $\text{SS}_{\text{res}} > \text{SS}_{\text{tot}} \implies R^2 < \mathbf{0}$ (negative $R^2$).

---

### Step 2: Environment Setup and Data Generation



```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set seed for reproducible results
np.random.seed(42)
n_samples = 40

# Generate clean linear data
X = np.linspace(1, 10, n_samples).reshape(-1, 1)
y_clean = 2.5 * X.ravel() + np.random.normal(0, 2.0, n_samples)

# Create an identical copy and inject two extreme outliers
y_outlier = y_clean.copy()
y_outlier[10] += 25.0  # Positive extreme outlier
y_outlier[30] -= 25.0  # Negative extreme outlier

print("Datasets generated successfully.")
```

---

### Step 3: Model Fitting and Metric Computation



We fit two linear regression models:
1. One on the standard (clean) dataset.
2. One on the dataset containing the two extreme outliers.

We compute **MAE**, **MSE**, **RMSE**, and **$R^2$** for both cases using `scikit-learn`.



```python
# Helper function to compute all metrics
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

# 1. Fit on clean data
model_clean = LinearRegression()
model_clean.fit(X, y_clean)
y_pred_clean = model_clean.predict(X)
mae_c, mse_c, rmse_c, r2_c = compute_metrics(y_clean, y_pred_clean)

# 2. Fit on outlier data
model_outlier = LinearRegression()
model_outlier.fit(X, y_outlier)
y_pred_outlier = model_outlier.predict(X)
mae_o, mse_o, rmse_o, r2_o = compute_metrics(y_outlier, y_pred_outlier)

# Print comparative table
print(f"{'Metric':<8} | {'Clean Data':<12} | {'With Outliers':<12} | {'Relative Increase'}")
print("-" * 55)
print(f"{'MAE':<8} | {mae_c:<12.3f} | {mae_o:<12.3f} | {mae_o / mae_c:.2f}x")
print(f"{'MSE':<8} | {mse_c:<12.3f} | {mse_o:<12.3f} | {mse_o / mse_c:.2f}x")
print(f"{'RMSE':<8} | {rmse_c:<12.3f} | {rmse_o:<12.3f} | {rmse_o / rmse_c:.2f}x")
print(f"{'R-squared':<8} | {r2_c:<12.3f} | {r2_o:<12.3f} | ---")
```

---

### Step 4: Visualizing Residuals and Outlier Sensitivity



The red vertical dashed lines represent the **residuals** ($y_i - \hat{y}_i$). Notice how the outliers create exceptionally long residual lines, which disproportionately expand MSE and RMSE due to the squaring operation.



```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Clean Data
axes[0].scatter(X, y_clean, color='tab:blue', label='Actual Data', zorder=3)
axes[0].plot(X, y_pred_clean, color='black', linewidth=2, label='Regression Line')
# Draw residual lines
axes[0].vlines(X, y_clean, y_pred_clean, colors='gray', linestyle='--', alpha=0.6, label='Residuals')
axes[0].set_title(f"Clean Dataset\nMAE: {mae_c:.2f} | MSE: {mse_c:.2f} | RMSE: {rmse_c:.2f} | R²: {r2_c:.2f}")
axes[0].set_xlabel("Feature (X)")
axes[0].set_ylabel("Target (y)")
axes[0].legend()
axes[0].grid(True, linestyle=':', alpha=0.6)

# Plot 2: Outlier Data
axes[1].scatter(X, y_outlier, color='tab:blue', label='Actual Data', zorder=3)
# Highlight the two outliers
axes[1].scatter(X[[10, 30]], y_outlier[[10, 30]], color='crimson', s=100, zorder=4, label='Injected Outliers')
axes[1].plot(X, y_pred_outlier, color='black', linewidth=2, label='Regression Line')
# Draw residual lines
axes[1].vlines(X, y_outlier, y_pred_outlier, colors='crimson', linestyle='--', alpha=0.6, label='Residuals')
axes[1].set_title(f"Dataset with Outliers\nMAE: {mae_o:.2f} | MSE: {mse_o:.2f} | RMSE: {rmse_o:.2f} | R²: {r2_o:.2f}")
axes[1].set_xlabel("Feature (X)")
axes[1].set_ylabel("Target (y)")
axes[1].legend()
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()
```

---

### Step 5: Summary of Behavioral Differences



#### Key Insights

1. **Why RMSE Leans Toward Outliers:**
   * In MAE, an error of $25$ contributes exactly $25$ to the summation.
   * In MSE, an error of $25$ contributes $25^2 = 625$ to the summation.
   * Because of this quadratic penalty, RMSE increases far more aggressively than MAE when outliers are present (as shown in the ratio comparison table).

2. **When to Use Which:**
   * **Use MAE:** When you want a steady metric where every error counts equally, or when the data naturally contains noisy outliers that should not dominate model evaluation.
   * **Use RMSE:** When large errors are especially costly in real-world terms (e.g., severe under-predictions in inventory or financial risk) and must be heavily penalized.
   * **Use $R^2$:** When communicating model performance to stakeholders, as it is standardized (scale-free) and directly indicates how much variance the model captures.



---

### Concept Explanation: Why MSE Moves Closer to Outliers and MAE Does Not

In linear regression, the choice of loss function determines how the model balances errors across all data points during training.



#### 1. Why MSE Loss Moves the Model Closer to Outliers (Figure 9)

> https://developers.google.com/machine-learning/crash-course/linear-regression/loss

**Mean Squared Error (MSE / $L_2$ Loss):**
$$\mathcal{L}_{\text{MSE}} = (y - \hat{y})^2$$

* **Quadratic Penalty:** MSE squares the error ($e = y - \hat{y}$). 
  * An error of $2$ contributes $2^2 = 4$ to the total loss.
  * An error of $10$ contributes $10^2 = 100$ to the total loss.
* **Gradient Mechanics:** The derivative of $(y - \hat{y})^2$ with respect to the error is:
  $$\frac{\partial}{\partial \hat{y}}\left[(y - \hat{y})^2\right] = -2(y - \hat{y})$$
  The pull (gradient) that a data point exerts on the regression line is **proportional to its distance** from the line. 
* **The Result:** An extreme outlier creates an enormous gradient. To minimize the overall sum of squares, the model is forced to tilt or shift toward that outlier, sacrificing accuracy on several well-behaved points just to reduce the single large squared penalty.



#### 2. Why MAE Loss Keeps the Model Farther from Outliers (Figure 10)

> https://developers.google.com/machine-learning/crash-course/linear-regression/loss

**Mean Absolute Error (MAE / $L_1$ Loss):**
$$\mathcal{L}_{\text{MAE}} = |y - \hat{y}|$$

* **Linear Penalty:** MAE takes the absolute difference without squaring it.
  * An error of $2$ contributes $2$ to the total loss.
  * An error of $10$ contributes $10$ to the total loss.
* **Gradient Mechanics:** The derivative of $|y - \hat{y}|$ with respect to the error is:
  $$\frac{\partial}{\partial \hat{y}}\left[|y - \hat{y}|\right] = -\text{sign}(y - \hat{y}) = \begin{cases} -1, & \text{if } y > \hat{y} \\ +1, & \text{if } y < \hat{y} \end{cases}$$
  The pull that any data point exerts on the regression line is **constant ($\pm 1$)**, regardless of whether the point is $2$ units away or $100$ units away.
* **The Result:** Twenty typical points pulling downward with a gradient of $-1$ each will easily overpower a single extreme outlier pulling upward with a gradient of $+1$. The model stays anchored to the majority trend and largely ignores the outlier.

> **Intuitive Analogy:** MSE is like computing the **mean** (sensitive to extreme values), whereas MAE is like computing the **median** (robust to extreme values).



### Colab Implementation: Visualizing MSE vs. MAE Loss

In `scikit-learn`:
* `LinearRegression()` minimizes **MSE ($L_2$ Loss)**.
* `QuantileRegressor(quantile=0.5, alpha=0)` minimizes **MAE ($L_1$ Loss)** (as the 50th percentile corresponds to the median / minimum absolute deviations).

#### Code Cell

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, QuantileRegressor

# 1. Generate clean linear data
np.random.seed(42)
n_samples = 30
X = np.linspace(1, 10, n_samples).reshape(-1, 1)
y = 1.8 * X.ravel() + np.random.normal(0, 1.2, n_samples)

# 2. Introduce a single extreme outlier
# We place it at X ~ 3 with a very high Y value
y_with_outlier = y.copy()
y_with_outlier[6] += 25.0

# 3. Fit Model 1: MSE Loss (Standard Ordinary Least Squares)
model_mse = LinearRegression()
model_mse.fit(X, y_with_outlier)
y_pred_mse = model_mse.predict(X)

# 4. Fit Model 2: MAE Loss (Quantile Regression at 50th percentile, alpha=0 removes regularization)
model_mae = QuantileRegressor(quantile=0.5, alpha=0.0)
model_mae.fit(X, y_with_outlier)
y_pred_mae = model_mae.predict(X)

# 5. Visualization
plt.figure(figsize=(10, 6))

# Plot inliers and outlier
plt.scatter(X, y_with_outlier, color='tab:blue', s=50, label='Normal Observations', alpha=0.8)
plt.scatter(X[6], y_with_outlier[6], color='red', s=150, edgecolors='black', linewidth=1.5,
            label='Single Outlier', zorder=5)

# Plot both fitted lines
plt.plot(X, y_pred_mse, color='red', linestyle='--', linewidth=2.5,
         label='Fitted with MSE Loss (Tilted toward outlier)')
plt.plot(X, y_pred_mae, color='green', linestyle='-', linewidth=2.5,
         label='Fitted with MAE Loss (Robust against outlier)')

# Labeling and aesthetics
plt.title("Effect of Outliers on Model Fitting: MSE Loss vs. MAE Loss", fontsize=13)
plt.xlabel("Input Feature (X)", fontsize=11)
plt.ylabel("Target (y)", fontsize=11)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
```



### What the Plot Shows

1. **The Green Line (MAE Loss):**
   * Stays aligned directly through the center of the regular data points.
   * Because the outlier's directional pull is bounded to $+1$, it cannot overpower the collective pull of the remaining $29$ points.

2. **The Red Dashed Line (MSE Loss):**
   * Rotates and shifts upward toward the red point.
   * Because the error on that single point is squared ($25^2 = 625$), the optimizer accepts small error increases on all other points in order to cut that massive squared penalty down.

---
To ensure the outlier cannot be missed or cut off at the graph borders, this version:
1. **Adjusts axis margins (`plt.ylim`)** so the outlier is completely inside the frame.
2. **Adds a red arrow and callout text** pointing directly to the outlier.
3. **Draws dashed residual lines** showing the exact distance from the outlier to each fitted line.

### Colab Code Cell

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, QuantileRegressor

# 1. Generate clean linear data (1D arrays for clean indexing)
np.random.seed(42)
n_samples = 25
x = np.linspace(1, 10, n_samples)
y = 2.0 * x + np.random.normal(0, 1.0, size=n_samples)

# 2. Select index 7 and inject a prominent outlier
outlier_idx = 7
y_with_outlier = y.copy()
y_with_outlier[outlier_idx] += 22.0  # Push point far above the linear trend

# Convert X to 2D for scikit-learn: shape (n_samples, 1)
X_2d = x.reshape(-1, 1)

# 3. Model 1: MSE Loss (Linear Regression / OLS)
model_mse = LinearRegression()
model_mse.fit(X_2d, y_with_outlier)
y_pred_mse = model_mse.predict(X_2d)

# 4. Model 2: MAE Loss (Quantile Regression at 50th percentile)
model_mae = QuantileRegressor(quantile=0.5, alpha=0.0)
model_mae.fit(X_2d, y_with_outlier)
y_pred_mae = model_mae.predict(X_2d)

# ---------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------
plt.figure(figsize=(11, 7))

# Plot normal inlier points
inlier_mask = np.arange(n_samples) != outlier_idx
plt.scatter(x[inlier_mask], y_with_outlier[inlier_mask], 
            color='tab:blue', s=60, label='Normal Points (Inliers)', zorder=3)

# Plot the single outlier prominently
plt.scatter(x[outlier_idx], y_with_outlier[outlier_idx], 
            color='red', s=180, edgecolors='black', linewidth=1.5, 
            label='Outlier Point', zorder=5)

# Plot the two regression lines
plt.plot(x, y_pred_mse, color='crimson', linestyle='--', linewidth=2.5,
         label='MSE Loss Line (Pulled towards outlier)')
plt.plot(x, y_pred_mae, color='green', linestyle='-', linewidth=2.5,
         label='MAE Loss Line (Stays with normal points)')

# Draw residual lines from outlier to both models
plt.vlines(x=x[outlier_idx], ymin=y_pred_mse[outlier_idx], ymax=y_with_outlier[outlier_idx],
           color='crimson', linestyle=':', linewidth=2, label='Error to MSE line')
plt.vlines(x=x[outlier_idx], ymin=y_pred_mae[outlier_idx], ymax=y_pred_mse[outlier_idx],
           color='green', linestyle=':', linewidth=2, label='Extra error tolerated by MAE')

# Add an arrow pointing directly to the outlier
plt.annotate(
    'Outlier Point\n(High Residual)',
    xy=(x[outlier_idx], y_with_outlier[outlier_idx]),
    xytext=(x[outlier_idx] + 1.0, y_with_outlier[outlier_idx] + 2.0),
    arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=8),
    fontsize=10,
    fontweight='bold',
    bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.6)
)

# Expand Y-limits so the outlier is not at the edge of the plot
plt.ylim(bottom=min(y_with_outlier) - 3, top=max(y_with_outlier) + 8)

plt.title("Google ML Crash Course Concept:\nMSE Loss Pulls Toward Outliers vs. MAE Loss Resists Them", fontsize=13)
plt.xlabel("Feature (x)", fontsize=11)
plt.ylabel("Target (y)", fontsize=11)
plt.legend(loc='upper left', framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
```


### What to Observe in This Plot

* **The Yellow Callout Box:** Points directly to the red outlier at coordinates $(x \approx 3.6, y \approx 30)$.
* **The Red Line (MSE):** Tilts upward toward the red point. Because MSE squares the distance, it sacrifices the fit of all other points to shorten the red dashed vertical line.
* **The Green Line (MAE):** Passes straight through the blue points. Because MAE only penalizes distance linearly, the single outlier does not have enough pull to drag the line away from the 24 blue points.


---

No single metric tells the complete story of a regression model. When used together, **$R^2$, MAE, and RMSE** function as a diagnostic dashboard that evaluates three distinct dimensions of model quality:

* **$R^2$:** *Relative explanatory power* (scale-free benchmark).

### 1. Comparing MAE and RMSE: Diagnosing the Error Distribution

Because both MAE and RMSE share the original unit of the target variable ($y$), comparing their values directly reveals how errors are distributed:

$$\text{MAE} \le \text{RMSE}$$

* **When $\text{RMSE} \approx \text{MAE}$:**
  * The errors across all data points are roughly the same magnitude.
  * The model makes steady, consistent errors without severe surprises.
* **When $\text{RMSE} \gg \text{MAE}$:**
  * The model predicts most points accurately (keeping MAE low), but occasionally makes rare, massive mistakes.
  * Because RMSE squares each residual before averaging, large errors pull RMSE upward drastically. 

> **Diagnostic Insight:** If a model has $\text{MAE} = \$5$ and $\text{RMSE} = \$6$, the error profile is uniform. If $\text{MAE} = \$5$ and $\text{RMSE} = \$28$, the model is reliable on average, but suffers from severe edge-case failures.


### 2. Pairing $R^2$ with MAE & RMSE: Explanatory Power vs. Practical Utility

$R^2$ is dimensionless (independent of the target's unit of measurement), whereas MAE and RMSE are absolute:

* **$R^2$ tells you how much signal you extracted:** It benchmarks your model against a trivial baseline (always predicting the mean $\bar{y}$). It allows you to compare models across entirely different datasets or domains.
* **MAE and RMSE tell you operational viability:** A business or engineering team cannot make operational decisions based on a percentage of variance alone; they need to know the actual deviation in dollars, degrees, or hours.

#### The "High $R^2$, High Error" Paradox
* If you predict company revenues spanning from $\$100,000$ to $\$10,000,000,000$, the target has massive variance ($\text{SS}_{\text{tot}}$).
* A model can easily score **$R^2 = 0.98$**, yet have an **MAE of $\$500,000$**. 
* The metric pairing clarifies that while the model captures the overall macro trend well, it may still be too imprecise for fine-grained budget decisions.

#### The "Low $R^2$, Low Error" Paradox
* If you predict human body temperature (which varies in a narrow band around $37^\circ\text{C}$), the target variance is very small.
* A model might only score **$R^2 = 0.25$**, yet have an **MAE of $0.15^\circ\text{C}$**.
* The metric pairing reveals that although the model explains little variation beyond the mean, the practical error is small enough for many medical applications.

### 3. Diagnostic Matrix for Model Evaluation

| Metric Pattern | Interpretation | Recommended Action |
| :--- | :--- | :--- |
| **High $R^2$, Low MAE, Low RMSE** | **Ideal Model:** Strong explanatory power, low average error, and no extreme mistakes. | Model is ready for validation/deployment. |
| **High $R^2$, Low MAE, High RMSE** | **Tail-Risk Model:** The model fits the vast majority of data well, but fails drastically on a small subset of samples. | Inspect the worst predictions; check for unhandled edge cases or data entry anomalies. |
| **Low $R^2$, Low MAE, Low RMSE** | **Low-Variance Target:** The model does not beat the baseline by much, but total deviation is practically negligible. | Consider whether complex modeling is necessary, or if a simple rule/mean suffices. |
| **Low $R^2$, High MAE, High RMSE** | **Underperforming Model:** The model fails to capture the underlying pattern and produces large errors. | Re-engineer features, try non-linear architectures, or acquire more informative predictors. |

---

### Summary Workflow

1. **Check $R^2$ first:** Confirm that the model is genuinely learning patterns beyond simply guessing the mean ($R^2 > 0$).
2. **Read MAE for intuition:** Use MAE to understand the day-to-day error magnitude in real-world units.
3. **Compare RMSE against MAE:** Look for a large divergence between RMSE and MAE to evaluate whether the model has dangerous tail-risk errors.