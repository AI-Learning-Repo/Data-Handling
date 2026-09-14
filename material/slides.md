### Introduction to Regression Analysis

---

### Theory (Part 1)

> From Correlations to Simple & Multiple Linear Regression

---

### Agenda
* **Case Study:** The Auto MPG Problem & Feature Representation
* **Exploratory Analysis:** Visualizations & Pearson Correlation ($r$)
* **Modeling:** Simple & Multiple Linear Regression
* **Evaluation:** MAE, RMSE, and $R^2$
* **Generalization:** Train/Test Splits, Underfitting & Overfitting

---

### What is Supervised Regression?
* **Supervised Learning:** Learning a mapping from labeled historical examples:
  $$X \text{ (Features)} \longrightarrow y \text{ (Target)}$$
* **Classification vs. Regression:**
  * **Classification:** Predicts discrete categories (e.g., spam vs. not spam, survival 0/1).
  * **Regression:** Predicts continuous numerical values (e.g., salary, temperature, **fuel efficiency**).

---

### Case Study: The Auto MPG Dataset
* **Goal:** Predict vehicle fuel efficiency based on automobile characteristics.
* **Target ($y$):** `mpg` (Miles Per Gallon).
* **Candidate Features ($X$):**
  * `weight`, `displacement`, `cylinders`, `horsepower`, `acceleration`, `model_year`.
* **Important Note on Missing Values:**
  * `horsepower` contains missing values; Activity 1 uses complete features (`weight`, `displacement`, `cylinders`) to keep the focus on modeling fundamentals.

---

### Understanding the Target: MPG
* **MPG = Miles Per Gallon:**
  * Measures distance traveled per unit volume of fuel.
  * **Higher is better** (40 MPG is more efficient than 20 MPG).
* **Metric Equivalents:**
  * Distance per volume: $1\text{ MPG} \approx 0.425\text{ km/L}$ (higher is better).
  * Volume per distance: $\text{L}/100\text{ km}$ (lower is better).

---

### Feature Matrix ($X$) vs. Target Vector ($y$)
* In scikit-learn, data structure shapes matter:
* **$X$ (Features):** Must be a **2D** table (`DataFrame` or 2D array):
  ```python
  X = df[["weight"]]  # Double brackets -> shape (N, 1)
  ```
* **$y$ (Target):** Must be a **1D** sequence (`Series` or 1D array):
  ```python
  y = df["mpg"]       # Single brackets -> shape (N,)
  ```

---

### Visualizing Relationships: Scatter Plots
* Before writing modeling code, always plot your data:
  ```python
  plt.scatter(df["weight"], df["mpg"])
  ```
* **Look for the overall trend:**
  * As vehicle weight increases, MPG generally decreases.
  * This indicates an **inverse (negative) relationship**.
* Points do not fall on a single straight line because fuel efficiency depends on multiple factors.

---

### Pearson Correlation Coefficient ($r$)
* Measures the **strength and direction of a linear relationship** between two numerical variables.
* **Range:** $-1.0 \le r \le +1.0$
* **Interpretation:**
  * $+1.0$: Perfect positive linear association (both increase together).
  * $-1.0$: Perfect negative linear association (one increases, other decreases).
  * $\approx 0.0$: No **linear** association.

---

### How Pearson $r$ Works
<!-- .slide: style="font-size: 90%;" -->
* Sample formula:
  $$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$
* **Deviation Logic:**
  * Observations on the same side of the mean yield $(+)(+)$ or $(-)(-)$, resulting in positive products.
  * Observations on opposite sides yield $(+)(-)$, driving the correlation negative.
  * In Auto MPG: $r(\text{weight}, \text{mpg}) \approx -0.83$ (strong negative linear association).

---

### Crucial Limitation: Linearity
* Pearson $r$ **only** detects straight-line relationships.
* Consider $y = x^2$ for $x \in [-3, 3]$:
  * $y$ is completely and deterministically dependent on $x$.
  * Yet, Pearson $r = 0.0$!
  * Positive and negative slopes cancel each other out.
* **Golden Rule:** $r \approx 0$ means *no linear relationship*, not *no relationship*. Always visualize!

---

### Correlation Rules of Thumb
* **Correlation $\neq$ Causation:**
  * High correlation proves association, not that changing $X$ directly causes a change in $y$.
* **Multicollinearity:**
  * Features often correlate with one another (e.g., `weight` and `displacement`).
  * Useful to check because redundant features complicate coefficient interpretation.
* **Feature Selection:**
  * Never select features based solely on high correlation with the target.

---

### Simple Linear Regression
* Uses **one feature** to fit the best straight line:
  $$\hat{y} = b_0 + b_1 x$$
  $$\widehat{\text{MPG}} = b_0 + b_1(\text{weight})$$
* **$\hat{y}$ (y-hat):** Predicted target value.
* **$b_1$ (Slope / Coefficient):** Predicted change in $y$ for a 1-unit increase in $x$.
* **$b_0$ (Intercept):** Predicted value of $y$ when $x = 0$.

---

### Interpreting the Parameters
<!-- .slide: style="font-size: 90%;" -->
* Suppose the fitted model is:
  $$\widehat{\text{MPG}} = 46.2 - 0.0076 \times (\text{weight})$$
* **Slope ($-0.0076$):**
  * Adding 1 pound of vehicle weight decreases predicted fuel efficiency by $0.0076\text{ MPG}$.
  * Units matter: changing pounds to kilograms changes the numerical coefficient.
* **Intercept ($46.2$):** The theoretical MPG of a car with 0 weight (physically impossible, but mathematically necessary).

---

### Multiple Linear Regression
* Extends the model to two or more input features:
  $$\hat{y} = b_0 + b_1 x_1 + b_2 x_2 + \dots + b_p x_p$$
  $$\widehat{\text{MPG}} = b_0 + b_1(\text{weight}) + b_2(\text{displacement}) + b_3(\text{cylinders})$$
* **Still a linear model:** The prediction remains a linear combination of features and weights.
* **Key Benefit:** Allows the model to use complementary information simultaneously.

---

### Interpreting Multiple Regression Coefficients
<!-- .slide: style="font-size: 85%;" -->
* In multiple regression, each coefficient represents the effect of that feature **while holding all other included features constant**:
  > "For two cars with identical displacement and cylinder count, each additional pound of weight is associated with a change of $b_1$ in MPG."
* Adding more features does not automatically make the model better on unseen data—features can introduce noise or redundancy.

---

### The scikit-learn API Workflow
```python
from sklearn.linear_model import LinearRegression

# 1. Instantiate the estimator
model = LinearRegression()

# 2. Fit (estimate optimal parameters b0, b1 from training data)
model.fit(X_train, y_train)

# 3. Predict (apply learned equation to new observations)
y_pred = model.predict(X_test)

# 4. Inspect learned parameters
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
```

---

### Why Split Data? (Generalization)
<!-- .slide: style="font-size: 85%;" -->
* The primary goal of machine learning is **generalization**:
  > Learning patterns that perform well on **new, unseen data**, not just memorizing training examples.
* Evaluating a model on the data used to train it produces an overly optimistic, biased score.
* We partition data into:
  * **Training Set (~80%):** Used by `.fit()` to learn parameters.
  * **Test Set (~20%):** Held back; used strictly for final evaluation.

---

### Train/Test Split in Code
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```
* `test_size=0.20`: Reserves 20% of samples for evaluation.
* `random_state=42`: Sets a fixed random seed so that the split is reproducible across runs.

---

### Evaluation Metrics: MAE & RMSE
<!-- .slide: style="font-size: 75%;" -->
* Regression predictions are continuous numbers; accuracy percentages do not apply.
* **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{n}\sum |y_i - \hat{y}_i|$$
  * Measures average error magnitude; directly interpretable in MPG units.
* **Root Mean Squared Error (RMSE):**
  $$\text{RMSE} = \sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}$$
  * Squaring errors penalizes large mistakes heavily; retains original target units.

---

### Evaluation Metrics: $R^2$ Score
<!-- .slide: style="font-size: 75%;" -->
* **Coefficient of Determination ($R^2$):**
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
* Compares model performance against a naive baseline that always predicts the mean target $\bar{y}$.
* **Scale:**
  * $R^2 = 1.0$: Perfect predictions.
  * $R^2 = 0.0$: Performs no better than the simple mean.
  * $R^2 < 0.0$: Performs worse than the mean baseline.
* *$R^2$ is NOT classification accuracy (e.g., $R^2 = 0.65$ does NOT mean "65% correct").*

---

### Diagnostic Plot: Actual vs. Predicted
* Scatter plot of actual values ($y_{\text{test}}$) vs. predicted values ($\hat{y}_{\text{pred}}$):
  ```python
  plt.scatter(y_test, y_pred)
  plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--")
  ```
* **The $45^\circ$ Diagonal Line:**
  * Represents perfect predictions ($\hat{y} = y$).
  * Points close to the line indicate small residuals.
  * Points far away represent large prediction errors.

---

### Underfitting vs. Overfitting
* **Underfitting (High Bias):**
  * Model is too simple to capture true patterns (e.g., fitting a straight line to a curve).
  * **Symptom:** High training error, high test error.
* **Overfitting (High Variance):**
  * Model is overly complex and memorizes training noise.
  * **Symptom:** Very low training error, but high/worse test error.
* **Good Generalization:** Low training error and low test error with a minimal performance gap.

---

### Model Complexity & The Trade-Off

> Increasing model complexity always lowers training error, but test error degrades past optimal complexity.

---

### Preventing Data Leakage
* **Data Leakage:** When information from the test set influences model training.
* Preprocessing transformations (scaling, imputing, feature generation) must learn parameters **exclusively from the training set**:
  ```python
  # CORRECT:
  X_train, X_test, y_train, y_test = train_test_split(X, y)
  
  # Fit transformer ONLY on training data
  X_train_transformed = transformer.fit_transform(X_train)
  # Apply learned transform to test data
  X_test_transformed = transformer.transform(X_test)
  ```

---

### Overview of Activity 1
```
1. Load Auto MPG dataset via ucimlrepo
                 ↓
2. Inspect features, target, and distributions (info(), describe())
                 ↓
3. Visualize relationship: scatter plot (weight vs. mpg)
                 ↓
4. Calculate Pearson correlation (corr())
                 ↓
5. Train Simple Linear Regression (weight ➔ mpg)
                 ↓
6. Evaluate baseline with MAE, RMSE, and R²
                 ↓
7. Train Multiple Linear Regression (weight + displacement + cylinders)
                 ↓
8. Compare models on the SAME test set
```

---

> .

---

### Theory: Part 2

> From Simple Lines to Polynomial Curves

---


### The Problem: Predicting Taxi Fares
* **Supervised Learning:** Predict a continuous numerical value (`FARE`).
* **Input Features ($X$):**
  * `TRIP_MILES`: Trip distance
  * `TRIP_MINUTES`: Trip duration (`TRIP_SECONDS / 60`)
* **Target ($y$):**
  * `FARE`: Cost in dollars ($)
* **Goal:** Train a model that learns the mapping $X \rightarrow y$ and generalizes to unseen rides.

---

### Feature Representation
* **$X$ (Feature Matrix):** Must be **2D** (shape: `[n_samples, n_features]`).
  * Double brackets return a DataFrame: `df[["TRIP_MILES"]]` $\rightarrow$ shape `(N, 1)`.
  * Single brackets return a Series: `df["TRIP_MILES"]` $\rightarrow$ shape `(N,)` (causes errors in estimators).
* **$y$ (Target Vector):** Must be **1D** (shape: `[n_samples]`).
  * Selected with single brackets: `df["FARE"]`.

---

### The Modeling Progression
* We build complexity step-by-step:
```
1. Simple Linear Regression     (1 feature: miles)
              ↓
2. Multiple Linear Regression   (2 features: miles + minutes)
              ↓
3. Polynomial Regression        (Engineered curves & interactions)
              ↓
4. scikit-learn Pipeline        (Packaging workflow cleanly)
```

---

### Model 1: Simple Linear Regression
<!-- .slide: style="font-size: 85%;" -->
* Uses one predictor:
  $$\widehat{\text{FARE}} = b_0 + b_1(\text{TRIP\\_MILES})$$
* **$b_1$ (Slope):** Estimated fare increase for each additional mile driven.
* **$b_0$ (Intercept):** Predicted fare when distance is 0 (base fare / pickup fee).
* **Fitting Objective (OLS):** Minimizes the sum of squared residuals:
  $$\sum (y_i - \hat{y}_i)^2$$

---

### Model 2: Multiple Linear Regression
<!-- .slide: style="font-size: 85%;" -->
* Adds trip duration as a second predictor:
  $$\widehat{\text{FARE}} = b_0 + b_1(\text{TRIP\\_MILES}) + b_2(\text{TRIP\\_MINUTES})$$
* **Still a linear model:** Linear combination of parameters.
* **Interpretation:**
  * $b_1$: Effect of 1 extra mile, **holding duration constant**.
  * $b_2$: Effect of 1 extra minute, **holding distance constant**.
* Captures real-world variance: Two trips with the same mileage can have different fares due to traffic delays.

---

### Model 3: Polynomial Regression
* What if the relationship between distance and fare is curved?
* Add higher-degree powers of the feature:
  $$\hat{y} = b_0 + b_1 x + b_2 x^2$$
* **Key Concept:** This is **still fitted with `LinearRegression`**!
  * It is nonlinear with respect to input $x$.
  * It is strictly **linear with respect to parameters** $b_0, b_1, b_2$.

---

### How `PolynomialFeatures` Transforms Data
<!-- .slide: style="font-size: 85%;" -->
* With **1 feature** ($x$) at `degree=2`:
  $$[x] \longrightarrow [x, \; x^2]$$
* With **2 features** ($x_1, x_2$) at `degree=2`:
  $$[x_1, x_2] \longrightarrow [x_1, \; x_2, \; x_1^2, \; x_1 x_2, \; x_2^2]$$
* **Interaction Term ($x_1 x_2$):** Allows the effect of distance to depend on trip duration.
* Note: Set `include_bias=False` because `LinearRegression` already fits an intercept.

---

### The Golden Rule of Preprocessing
* **Fit on Train, Transform on Test:**
```python
# 1. Learn transformation structure from TRAINING data only
X_train_poly = poly.fit_transform(X_train)

# 2. Apply already-learned transformation to TEST data
X_test_poly = poly.transform(X_test)
```
* Never call `fit_transform()` on the test set (prevents data leakage).

---

### The Manual Workflow vs. Pipeline
* **Manual Approach:**
```
Raw Data ➔ poly.fit_transform() ➔ model.fit() ➔ poly.transform() ➔ model.predict()
```
* **Pipeline Approach:**
```python
pipeline = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("model", LinearRegression())
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```
* `Pipeline` automates transformations during both fitting and inference without altering the underlying math.

---

### Model Evaluation: The Metric Triad
<!-- .slide: style="font-size: 75%;" -->
* **MAE (Mean Absolute Error):**
  $$\text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i|$$
  * Direct business interpretation in dollars ($).
* **RMSE (Root Mean Squared Error):**
  $$\text{RMSE} = \sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$$
  * Heavily penalizes rare, large prediction errors.
* **$R^2$ (Coefficient of Determination):**
  * Proportion of variance explained compared to predicting the simple mean $\bar{y}$.

---

### Diagnostic Plots: Checking Predictions
<!-- .slide: style="font-size: 85%;" -->
* **Actual vs. Predicted Plot:**
  * Plot $y_{\text{test}}$ vs. $\hat{y}_{\text{pred}}$.
  * Points should cluster tightly around the $45^\circ$ reference line ($y = \hat{y}$).
* **Residual Plot ($y - \hat{y}$):**
  * Residuals should be randomly dispersed around 0.
  * Systematic curvature indicates missing non-linear structure.
  * Funnel shapes indicate heteroscedasticity (error variance changes with fare size).

---

### Hyperparameter Tuning: Polynomial Degree
<!-- .slide: style="font-size: 85%;" -->
* Experimenting with model complexity (Degrees 1 through 4):

| Degree | Model Nature | Risk / Observation |
| :---: | :--- | :--- |
| **1** | Simple straight line | Possible **underfitting** (high train error, high test error) |
| **2** | Quadratic curve | Often optimal balance (low train error, low test error) |
| **3–4+** | Highly flexible curve | Risk of **overfitting** (train error drops, test error spikes) |

* **Lesson:** Training error monotonically decreases as degree increases, but test error degrades past optimal complexity.

---

### Does Feature Scaling Matter Here?
* **Ordinary Least Squares (`LinearRegression`):**
  * Scaling (`StandardScaler`) is **not strictly required**.
  * OLS automatically compensates by adjusting coefficient scales.
* **When Scaling IS Required:**
  1. Regularized regression (Ridge, Lasso, ElasticNet).
  2. Distance-based models (KNN, SVM).
  3. High-degree polynomials experiencing numerical instability.

---

### Beyond Basic OLS: Regularization (Preview)
<!-- .slide: style="font-size: 75%;" -->
* What if we have many correlated features and want to prevent overfitting?
* **Ridge Regression (L2 Penalty):**
  $$\text{Loss} = \text{OLS Error} + \lambda \sum b_j^2$$
  * Shrinks coefficients close to zero; stabilizes correlated features.
* **Lasso Regression (L1 Penalty):**
  $$\text{Loss} = \text{OLS Error} + \lambda \sum |b_j|$$
  * Forces non-informative coefficients exactly to 0 (automatic feature selection).

---

### Beyond Linear Models: Tree Regressors (Preview)
* **Decision Tree Regressors:**
  * Splits feature space into rectangular regions using yes/no thresholds.
  * Captures non-linearities and interactions naturally without polynomial feature engineering.
* **Ensemble Methods:**
  * **Random Forest:** Averages predictions of multiple decorrelated trees to reduce variance.
  * **Gradient Boosting:** Builds trees sequentially, each correcting the residuals of previous trees.

---

### Overview of Activity 2
```
1. Load Chicago Taxi dataset & derive TRIP_MINUTES
2. Exploratory Data Analysis & Correlation Matrix
3. Hold out 20% test data (train_test_split, random_state=42)
4. Train & evaluate Model 1 (Simple Linear: miles)
5. Train & evaluate Model 2 (Multiple Linear: miles + minutes)
6. Train & evaluate Model 3 (Polynomial degree 2 manually)
7. Run Degree Experiment (1 to 4) to visualize overfitting
8. Wrap Model 3 into a scikit-learn Pipeline
```

