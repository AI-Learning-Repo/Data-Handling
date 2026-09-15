# Demo: Pearson correlation

---

### Step 1: Theoretical Overview and Formula


#### Pearson Correlation Coefficient ($r$)

The Pearson correlation coefficient measures the direction and strength of a linear association between two continuous variables. 

Given two variables, $X$ and $Y$, with $n$ sample observations, the sample correlation coefficient $r$ is calculated as:

$$r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}$$

* **Range:** $[-1, 1]$
* **$r = 1$:** Perfect positive linear relationship.
* **$r = -1$:** Perfect negative linear relationship.
* **$r = 0$:** Absence of any linear association.

In `scikit-learn`, Pearson's $r$ is implemented via `sklearn.feature_selection.r_regression`.

---

### Step 2: Environment Setup and Data Generation



We begin by importing the necessary libraries and generating synthetic data representing three scenarios:
1. A feature with a linear relationship to a target.
2. Two features with a linear relationship to each other.
3. A feature with a non-linear (quadratic) relationship to a target.



```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import r_regression

# Ensure reproducible results
np.random.seed(42)
n_samples = 300

# 1. Base input feature
X1 = np.random.uniform(-3, 3, n_samples)

# 2. Target linearly dependent on X1
y_linear = 2.0 * X1 + np.random.normal(0, 1.0, n_samples)

# 3. Second feature linearly dependent on X1 (collinear features)
X2 = 0.85 * X1 + np.random.normal(0, 0.6, n_samples)

# 4. Feature and Target with a non-linear (quadratic) relationship
X3 = np.random.uniform(-3, 3, n_samples)
y_nonlinear = (X3 ** 2) + np.random.normal(0, 0.4, n_samples)

print("Dataset successfully generated.")
```

---

<details>
<summary><b>Click to expand: How the synthetic linear and non-linear data was generated</b></summary>

### Reproducibility and Sample Size
```python
np.random.seed(42)
n_samples = 300
```
* `np.random.seed(42)` sets the starting point for the random number generator, ensuring that the generated numbers and plots are identical every time the code runs.
* `n_samples = 300` defines the dataset size. 300 points provide enough statistical stability to observe correlation trends clearly without cluttering scatter plots.


### Base Input Feature ($X_1$)
```python
X1 = np.random.uniform(-3, 3, n_samples)
```
* Draws 300 numbers from a **uniform continuous distribution** between $-3$ and $3$. 
* This provides a balanced spread of positive and negative values centered at $0$.

### Linear Feature-Target Relationship ($y_{\text{linear}}$)
```python
y_linear = 2.0 * X1 + np.random.normal(0, 1.0, n_samples)
```
* Follows the standard linear equation: 
  $$y = m \cdot X + c + \epsilon$$
* **Slope ($m = 2.0$):** For every 1-unit increase in $X_1$, $y$ increases by 2 units on average.
* **Gaussian Noise ($\epsilon$):** `np.random.normal(0, 1.0, n_samples)` adds random scatter drawn from a normal distribution with mean $0$ and standard deviation $1.0$. Without this noise, the data would fall on a deterministic straight line ($r = 1.0$). Adding controlled noise yields a realistic correlation ($r \approx 0.90$).


### Linear Feature-Feature Relationship ($X_2$)
```python
X2 = 0.85 * X1 + np.random.normal(0, 0.6, n_samples)
```
* Generates a second feature directly derived from $X_1$ using the same linear model approach.
* Because $X_2$ moves in tandem with $X_1$, this simulates **multicollinearity** (redundancy between inputs).


### Non-Linear Relationship ($y_{\text{nonlinear}}$)
```python
X3 = np.random.uniform(-3, 3, n_samples)
y_nonlinear = (X3 ** 2) + np.random.normal(0, 0.4, n_samples)
```
* Follows a quadratic function:
  $$y = X_3^2 + \epsilon$$
* **Why this produces $r \approx 0$:**
  * For $X_3 < 0$, as $X_3$ increases toward zero, $y$ decreases (negative trend).
  * For $X_3 > 0$, as $X_3$ increases away from zero, $y$ increases (positive trend).
  * Because the parabola is symmetric around $0$, the negative slope on the left cancels out the positive slope on the right. 
  * Pearson's formula tries to fit a single straight line through the data and finds an average slope of zero, demonstrating its inability to detect non-linear dependencies.

</details>



---

### Step 3: Feature-Target Linear Relationship


When assessing candidate features for regression tasks, computing $r$ against the target variable indicates whether a linear model will readily pick up the signal.


```python
# Reshape X1 to 2D array as required by scikit-learn
r_target = r_regression(X1.reshape(-1, 1), y_linear)[0]

print(f"Feature-Target Correlation (r): {r_target:.4f}")

# Visualization
plt.figure(figsize=(6, 4))
plt.scatter(X1, y_linear, alpha=0.7, color='steelblue')
plt.title(f"Feature vs Target (Linear)\nr = {r_target:.2f}")
plt.xlabel("Input Feature (X1)")
plt.ylabel("Target (y_linear)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
```

---

### Step 4: Feature-Feature Linear Relationship (Multicollinearity)


Computing $r$ between pairs of input features helps detect multicollinearity. When two features exhibit a high correlation, they provide redundant information, which can inflate standard errors in linear models.


```python
# Compute correlation between X1 and X2
r_features = r_regression(X1.reshape(-1, 1), X2)[0]

print(f"Feature-Feature Correlation (r): {r_features:.4f}")

# Visualization
plt.figure(figsize=(6, 4))
plt.scatter(X1, X2, alpha=0.7, color='seagreen')
plt.title(f"Feature vs Feature (Multicollinearity)\nr = {r_features:.2f}")
plt.xlabel("Feature 1 (X1)")
plt.ylabel("Feature 2 (X2)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
```

---

### Step 5: Limitation of $r$ Under Non-Linearity

A fundamental limitation of Pearson's $r$ is that it only measures **linear** relationships. If the relationship is strong but non-linear (e.g., quadratic, exponential, periodic), Pearson's $r$ may yield values close to zero.


```python
# Compute correlation between X3 and y_nonlinear
r_nonlin = r_regression(X3.reshape(-1, 1), y_nonlinear)[0]

print(f"Non-linear Relationship Correlation (r): {r_nonlin:.4f}")

# Visualization
plt.figure(figsize=(6, 4))
plt.scatter(X3, y_nonlinear, alpha=0.7, color='firebrick')
plt.title(f"Non-Linear Relationship\nr = {r_nonlin:.2f}")
plt.xlabel("Input Feature (X3)")
plt.ylabel("Target (y_nonlinear)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
```

---

### Step 6: Summary and Findings

#### Observations

1. **Feature-Target ($r \approx 0.85 - 0.95$):** Reflects a strong linear relationship. The feature serves as an effective linear predictor.
2. **Feature-Feature ($r \approx 0.80 - 0.90$):** Indicates substantial redundancy between features, signaling multicollinearity that may require dimensionality reduction or feature removal.
3. **Non-linear Relationship ($r \approx 0.00$):** Although the target is deterministically defined by $y = X_3^2$, the negative slope for $X_3 < 0$ and the positive slope for $X_3 > 0$ cancel each other out.

**Conclusion:** A low Pearson correlation does not imply an absence of dependency. When non-linear dynamics are suspected, non-parametric alternatives such as Spearman's rank correlation or Mutual Information (`sklearn.feature_selection.mutual_info_regression`) should be used.

----

### `r_regression` vs `corr`

Both calculate Pearson’s $r$, but they belong to different Python libraries and serve slightly different purposes:

| Function / Method | Library | Input Format | Primary Use Case |
| :--- | :--- | :--- | :--- |
| **`r_regression`** | **`scikit-learn`** | 2D matrix of features $X$, 1D target vector $y$ | Designed for **feature selection**; computes $r$ between every column in $X$ and the target $y$ in one call. |
| **`.corr()`** | **`pandas`** | DataFrame | Computes a full pairwise correlation matrix between all columns in a table. |

---

Using Pearson’s correlation coefficient ($r$) before training a regression model is **a standard, valuable first step in Exploratory Data Analysis (EDA)**, but it **should not be used as the sole criterion for feature selection**. 

While it provides quick insight into linear associations, relying exclusively on $r$ to keep or discard features can lead to flawed modeling decisions.

---

### Where Pearson’s $r$ Is Useful Before Modeling

#### 1. Identifying Strong Linear Predictors (Feature vs. Target)
Computing $r$ between each numerical feature and the target variable helps establish a baseline ranking of direct linear associations. If a feature shows $|r| > 0.8$, you know immediately that a standard linear model will capture significant variance from that feature alone.

#### 2. Diagnosing Multicollinearity Early (Feature vs. Feature)
Generating a correlation heatmap among the input features reveals redundancies. If two features, $X_1$ and $X_2$, have a correlation near $\pm 1.0$, they provide virtually identical information. In linear models (e.g., Ordinary Least Squares), multicollinearity causes:
* Unstable coefficient estimates.
* Inflated standard errors and unreliable $p$-values.
* Difficulties in determining which variable is truly driving the prediction.

#### 3. Low Computational Cost
Pearson’s $r$ is fast to calculate ($O(n)$ per variable pair), making it efficient to compute across hundreds of variables before investing time in expensive model training.

---

### Why Relying on Pearson’s $r$ Alone Is Risky

#### 1. It Discards Valid Non-Linear Features
Pearson’s $r$ exclusively measures **straight-line** associations. If an input feature has a deterministic non-linear relationship with the target (such as a parabola $y = X^2$, an exponential curve, or a seasonal cycle), its Pearson $r$ can be close to $0$. 
* **The Risk:** If you automatically drop features with low $r$, you will discard variables that non-linear models (like Random Forests, Gradient Boosted Trees, or Polynomial Regressors) could use effectively.

#### 2. Bivariate Checks Ignore Multivariate Synergy
Pearson’s $r$ is strictly **bivariate** (it evaluates one feature against the target in isolation). A regression model, by contrast, is **multivariate** (it evaluates the effect of a feature *conditional on all other features*).
* A feature may have a very low correlation with the target on its own, but become highly predictive when combined with other features.
* **Example:** In predicting house prices, "lot frontage" alone may correlate weakly with price. However, combined with "lot depth," it defines total surface area, which is highly predictive.

#### 3. It Cannot Detect Complex Interactions
Pearson's $r$ cannot account for interaction effects (e.g., $X_1 \times X_2$). Two variables may individually correlate weakly with the target, yet their product or ratio might explain most of the target's variance.

#### 4. Vulnerability to Outliers
A single extreme observation can either:
* Artificially inflate the correlation coefficient of an otherwise uncorrelated feature.
* Drag a genuinely strong correlation down to near zero.

---

### Recommended Best-Practice Workflow

Instead of using Pearson’s $r$ as a pass/fail filter, integrate it into a broader diagnostic process:


1. **Calculate Pearson's $r$ for initial intuition:** Identify which features have simple, direct linear relationships with the target and check for collinear feature pairs.
2. **Complement with Visuals:** Use scatter plots for top and bottom correlated features to verify whether low $r$ is due to true randomness or an undetected non-linear curve (Anscombe's Quartet effect).
3. **Use Mutual Information (MI):** Run non-parametric metrics such as `mutual_info_regression` from `scikit-learn`. Unlike Pearson's $r$, Mutual Information captures non-linear relationships and entropy-based dependencies.
4. **Let Regularization Decide:** Rather than manually dropping features based on $r$, feed the full set of candidate features into regularized models (such as **LASSO / $L_1$ penalty**) or tree-based algorithms. These methods automatically weigh, penalize, or eliminate features in a multivariate context.