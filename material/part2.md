# Theory: Regression with scikit-learn

## Introduction

Regression is a supervised machine-learning problem in which the model learns to predict a **continuous numerical value** from one or more input variables.

In this lab, the value we want to predict is the **taxi fare**. The fare is a numerical quantity, so regression is an appropriate family of methods.

The central idea is simple:

```text
input features  ──> regression model ──> predicted numerical value
```

For the taxi problem, examples of input features are:

- `TRIP_MILES`: distance of the trip
- `TRIP_MINUTES`: duration of the trip

and the target is:

- `FARE`: the taxi fare

The important machine-learning vocabulary is:

| Term | Meaning |
|---|---|
| Feature | An input variable used by the model |
| Target | The value we want the model to predict |
| Observation / sample | One row of data |
| Model | A mathematical function that maps features to a prediction |
| Training | The process of learning model parameters from training data |
| Prediction | The value produced by a trained model |
| Evaluation | Measuring how well predictions agree with known target values |

The lab deliberately builds models step by step so that these concepts remain visible rather than being hidden behind a single high-level command.

---

# 1. From a real-world question to a regression problem

Suppose a taxi company wants to estimate a customer's fare before the trip is completed.

We can express the problem as:

> Given information about a taxi trip, can we predict the fare?

This becomes a supervised learning problem because the training data contains examples for which the correct answer, the fare, is already known.

Conceptually, the dataset looks like this:

| TRIP_MILES | TRIP_MINUTES | FARE |
|---:|---:|---:|
| 2.1 | 8.0 | 9.50 |
| 5.4 | 17.0 | 18.20 |
| 10.2 | 31.0 | 31.80 |

The model tries to learn a relationship between the input features and the target.

The important point is that the model does **not** memorize a formula that we manually give it. Instead, we provide examples and let the learning algorithm estimate parameters from those examples.

---

# 2. Features and target variables

A regression model distinguishes between:

- **X**: the input features
- **y**: the target

For simple linear regression using distance:

```python
X = training_df[["TRIP_MILES"]]
y = training_df["FARE"]
```

Here:

```text
X = TRIP_MILES
 y = FARE
```

For multiple linear regression:

```python
X = training_df[["TRIP_MILES", "TRIP_MINUTES"]]
y = training_df["FARE"]
```

Now the model has two inputs.

The distinction is important because the target is the value we are trying to predict. It must not accidentally be included among the input features.

### Why does scikit-learn often use double brackets for X?

A single column such as:

```python
training_df["TRIP_MILES"]
```

is normally a pandas `Series`, which is one-dimensional.

Using:

```python
training_df[["TRIP_MILES"]]
```

returns a two-dimensional `DataFrame` containing one feature. Scikit-learn estimators generally expect `X` in the form:

```text
number of samples × number of features
```

So even with one feature, the structure is conceptually:

```text
[[x1],
 [x2],
 [x3],
 ...]
```

while `y` is usually one-dimensional:

```text
[y1, y2, y3, ...]
```

---

# 3. Train and test data

A model needs data on which to learn and separate data on which to evaluate itself.

We normally divide the available examples into at least two groups:

```text
                 complete dataset
                       |
              -------------------
              |                 |
          training set       test set
              |                 |
          learn model       evaluate model
```

The **training set** is used to estimate model parameters.

The **test set** is held back until after training. It is intended to represent data that the model has not seen during fitting.

This separation matters because evaluating a model on the same examples that were used to train it can give an overly optimistic impression of performance.

## 3.1 `train_test_split`

A common scikit-learn implementation is:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)
```

`test_size=0.20` means that approximately 20% of the samples are reserved for testing and the remaining 80% are used for training.

`random_state=42` makes the split reproducible. The particular number 42 has no mathematical significance here; it is simply a fixed seed.

### Why split before fitting?

Consider this sequence:

```text
1. split the data
2. fit the model using training data
3. predict using test data
4. calculate metrics using test targets
```

The model must not learn from `X_test` and `y_test` before evaluation.

This principle becomes especially important when preprocessing or feature engineering has parameters that are learned from the data. Such transformations should also be fitted using the training data and then applied to the test data.

---

# 4. Correlation and an initial look at relationships

Before choosing a model, it is useful to explore whether the variables appear to be related.

For two numerical variables, one familiar tool is the **Pearson correlation coefficient**.

## 4.1 Pearson Correlation Coefficient (`r`)

Pearson's correlation coefficient, usually written as `r`, measures the **strength and direction of a linear relationship between two numerical variables**.

Its value is between -1 and +1:

```text
r = +1     perfect positive linear relationship
r =  0     no linear correlation
r = -1     perfect negative linear relationship
```

The formula is:

$$
r = \frac{\operatorname{cov}(X,Y)}{\sigma_X\sigma_Y}
$$

where:

- `cov(X,Y)` is the covariance between `X` and `Y`;
- `σX` is the standard deviation of `X`;
- `σY` is the standard deviation of `Y`.

An equivalent sample formula is:

$$
r = \frac{\sum_i (x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_i (x_i-\bar{x})^2}\sqrt{\sum_i (y_i-\bar{y})^2}}
$$

You do not normally calculate this by hand in the lab. Pandas can calculate correlations with:

```python
training_df[["TRIP_MILES", "TRIP_MINUTES", "FARE"]].corr()
```

### 4.2 How to interpret `r`

Suppose the correlation between `TRIP_MILES` and `FARE` is positive and fairly large.

That suggests:

> Longer trips tend to have higher fares, and the relationship has a substantial linear component.

It does **not** prove that distance is the only useful predictor, and it does not prove that a linear regression model will be perfect.

Similarly, correlation does not tell us that one variable causes another.

For example:

```text
high correlation  ≠  causation
```

Correlation is best treated as an exploratory tool and an initial clue about relationships in the data.

### 4.3 Correlation does not detect every useful relationship

Pearson's `r` specifically measures linear association.

Imagine a relationship shaped approximately like a U:

```text
      y
      | *       *
      |  *     *
      |   *   *
      |    * *
      |     *
      +------------ x
```

There is a strong systematic relationship, but the ordinary Pearson correlation can be close to zero because positive and negative slopes occur in different parts of the curve.

Therefore:

> A low Pearson correlation does not always mean that there is no useful relationship.

This is one reason visualization and nonlinear modeling can be important.

---

# 5. Simple linear regression

Simple linear regression uses **one feature** to predict a continuous target.

The model has the form:

$$
\hat{y} = b_0 + b_1x
$$

where:

- `x` is the input feature;
- `ŷ` is the predicted target;
- `b0` is the intercept;
- `b1` is the coefficient or slope.

For the taxi example, we might use:

$$
\widehat{FARE} = b_0 + b_1(TRIP\_MILES)
$$

The model is trying to find the line that gives the best fit to the training observations according to the ordinary least-squares objective.

---

## 5.1 What does the coefficient mean?

Suppose the fitted model is:

$$
\widehat{FARE} = 3.20 + 2.10(TRIP\_MILES)
$$

Then:

- the intercept is `3.20`;
- the coefficient for `TRIP_MILES` is `2.10`.

The coefficient means that, according to this model, increasing trip distance by one mile is associated with an increase of about 2.10 fare units, on average, all else being equal within the model.

Be careful with the word **associated**. A regression coefficient describes the model's fitted relationship; it is not automatically a causal statement.

### What is the intercept?

The intercept is the predicted value when every input feature is zero.

For the taxi example, that would correspond to a zero-mile trip, which may not be a realistic trip. Therefore the intercept can be mathematically important even when its real-world interpretation is limited.

---

# 6. Ordinary least squares: what the linear regression model is trying to do

For each training example, the model produces a prediction `ŷi`.

The residual, or prediction error, is:

$$
 e_i = y_i - \hat{y}_i
$$

The ordinary least-squares method chooses the coefficients to minimize the **sum of squared residuals**:

$$
\sum_i (y_i - \hat{y}_i)^2
$$

The squaring has an important effect:

- positive and negative errors do not cancel;
- larger errors receive disproportionately more weight.

The fitted line is therefore a compromise that minimizes this squared-error objective for the training data.

---

# 7. Multiple linear regression

Multiple linear regression extends linear regression to **two or more input features**.

With two features the equation is:

$$
\hat{y} = b_0 + b_1x_1 + b_2x_2
$$

For the taxi problem:

$$
\widehat{FARE}
=
b_0
+ b_1(TRIP\_MILES)
+ b_2(TRIP\_MINUTES)
$$

The key conceptual point is:

> Multiple linear regression is still a linear model.

Having several features does not make the model nonlinear.

It is called **multiple** linear regression because the model uses multiple input variables.

## 7.1 Interpreting coefficients in a multiple model

Suppose the model is:

$$
\widehat{FARE}
= 2.5 + 1.7(TRIP\_MILES) + 0.4(TRIP\_MINUTES)
$$

The interpretation of `1.7` is approximately:

> For a fixed trip duration, increasing trip distance by one mile is associated with a 1.7-unit increase in predicted fare.

The phrase **holding the other included features constant** is important.

Likewise, the duration coefficient describes the model's change in prediction for an additional minute while keeping the other input features fixed.

In real data, features can be related to each other, so coefficient interpretation should be done carefully.

---

# 8. Why add another feature?

A single feature rarely contains all the information relevant to a real-world prediction.

For taxi fares, distance is obviously important, but trip duration can also matter. Two trips of identical distance can take different amounts of time because of traffic, road conditions, or routing.

Adding another feature may allow the model to explain variation that was previously treated as unexplained error.

However, more features are not automatically better.

A feature should be useful because it improves generalization or provides relevant information, not simply because it is available.

---

# 9. Polynomial regression

A straight line cannot represent every relationship.

For example, suppose the relationship between a feature `x` and a target is curved. A linear model has the form:

$$
\hat{y}=b_0+b_1x
$$

A polynomial model might include a squared term:

$$
\hat{y}=b_0+b_1x+b_2x^2
$$

or a cubic term:

$$
\hat{y}=b_0+b_1x+b_2x^2+b_3x^3
$$

This gives the model more flexibility.

## 9.1 Polynomial regression is still fitted with a linear estimator

This point is important.

Polynomial regression is often described as a nonlinear regression technique because the resulting curve can be nonlinear in `x`.

However, the model is **linear in its parameters** `b0`, `b1`, `b2`, ... . That means ordinary linear regression can still be used after transforming the input features.

For one feature, the transformation is conceptually:

```text
x  ->  [x, x², x³, ...]
```

Then `LinearRegression` is fitted to those transformed columns.

scikit-learn provides `PolynomialFeatures` to perform this transformation.

---

# 10. What `PolynomialFeatures` actually does

Suppose the original feature is:

```text
TRIP_MILES
```

With degree 2, polynomial feature generation includes terms based on:

```text
TRIP_MILES
TRIP_MILES²
```

With two original features, such as `TRIP_MILES` and `TRIP_MINUTES`, degree 2 can also produce an interaction term:

```text
TRIP_MILES × TRIP_MINUTES
```

Conceptually, degree 2 can therefore create:

```text
1
TRIP_MILES
TRIP_MINUTES
TRIP_MILES²
TRIP_MILES * TRIP_MINUTES
TRIP_MINUTES²
```

The exact column names and ordering are provided by the transformer.

The reason the lab performs this transformation explicitly before introducing `Pipeline` is pedagogical: students should see that a polynomial model consists of two separate ideas:

1. transform the input features;
2. fit a regression estimator to the transformed features.

---

# 11. Model complexity

The three models studied in the lab increase in flexibility:

```text
Simple linear regression
        |
        | add another feature
        v
Multiple linear regression
        |
        | add polynomial terms
        v
Polynomial regression
```

A more flexible model can represent more complicated patterns, but greater flexibility also creates a greater risk of fitting random variation rather than the underlying relationship.

This leads directly to **underfitting and overfitting**.

---

# 12. Underfitting and overfitting

## 12.1 Underfitting

A model **underfits** when it is too simple to capture important patterns in the data.

Imagine that the true relationship is curved but we force a straight line onto it. The model may systematically miss the structure.

A typical pattern is:

```text
training error: relatively high
validation/test error: relatively high
```

The model has not learned enough of the useful structure.

### Example

Suppose fare increases with distance in a relationship that is not well approximated by one straight line. A very simple linear model may have difficulty capturing the pattern.

Adding a suitable feature or polynomial term can sometimes reduce underfitting.

---

## 12.2 Overfitting

A model **overfits** when it learns the training data too closely, including noise or unusual details that do not generalize to new observations.

A highly flexible polynomial model illustrates the idea well.

With increasing degree, the fitted curve may bend sharply to pass close to individual training observations:

```text
low complexity       high complexity
      |                     |
   smooth fit           very wiggly fit
      |                     |
   general pattern       training noise
```

A typical pattern is:

```text
training error: very low
validation/test error: noticeably higher
```

This gap between training performance and performance on unseen data is a warning sign.

---

# 13. The bias-variance intuition

Underfitting and overfitting can be understood through the ideas of **bias** and **variance**.

A very simple model can have:

- high bias: it makes strong simplifying assumptions;
- relatively low variance: it is not very sensitive to small changes in the training data.

A very flexible model can have:

- lower bias: it can represent complicated patterns;
- higher variance: it can become sensitive to the particular training examples.

Conceptually:

```text
model complexity  --->

low                         high
complexity                  complexity

underfitting      useful region       overfitting
```

There is often a useful middle ground in which the model is complex enough to capture meaningful structure but not so flexible that it memorizes noise.

This is one reason why a higher polynomial degree is not automatically better.

---

# 14. How to recognize underfitting and overfitting in the lab

Students should not decide that a model is better merely because its training score is better.

Instead, compare performance on data that were not used for fitting.

A useful pattern to examine is:

| Situation | Training performance | Test performance | Interpretation |
|---|---|---|---|
| Underfitting | Poor | Poor | Model is too simple |
| Good fit | Good | Good | Reasonable generalization |
| Overfitting | Very good | Much worse | Model may be fitting noise |

There is no universal numerical threshold that says exactly when a model is overfitting. The diagnosis depends on the data, metric, problem, and model.

---

# 15. Regression evaluation metrics

A regression model produces numerical predictions, so we need appropriate numerical error measures.

The lab uses three metrics:

- MAE
- RMSE
- R²

No single metric gives the complete picture.

---

## 15.1 Mean Absolute Error (MAE)

MAE is the average absolute difference between the real target and the prediction:

$$
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

For example, if the absolute errors are:

```text
2, 4, 1, 3
```

then:

```text
MAE = (2 + 4 + 1 + 3) / 4 = 2.5
```

The advantage of MAE is that it is easy to interpret in the same units as the target.

If the target is measured in currency units, the MAE is also expressed in currency units.

MAE treats all absolute errors proportionally. An error of 10 is twice as large as an error of 5 in the MAE calculation.

---

## 15.2 Mean Squared Error (MSE)

MSE is:

$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

Squaring makes larger errors count more heavily.

For example:

```text
error = 2  -> squared error = 4
error = 10 -> squared error = 100
```

This is useful when large mistakes should receive substantially more penalty.

However, because the error is squared, MSE is not expressed in the original target units.

---

## 15.3 Root Mean Squared Error (RMSE)

RMSE is the square root of MSE:

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

Taking the square root returns the metric to the same units as the target.

RMSE is therefore often easier to discuss than MSE while still giving larger errors extra influence.

For a taxi fare problem, RMSE can be interpreted roughly as the typical scale of prediction error, although it is not exactly the same thing as an average absolute error.

---

## 15.4 R²: coefficient of determination

R² measures how much of the variation in the target is explained by the model relative to a baseline that predicts the mean target value.

One common definition is:

$$
R^2 = 1 - \frac{\sum_i(y_i-\hat{y}_i)^2}{\sum_i(y_i-\bar{y})^2}
$$

where `ȳ` is the mean of the target values.

An R² value of:

- `1.0` means perfect prediction on the evaluated data;
- `0` means the model performs like the mean-prediction baseline under this definition;
- a negative value means the model performs worse than that baseline on the evaluated data.

R² is useful for comparing explanatory performance, but it should not be interpreted as the percentage of individual predictions that are correct.

---

# 16. Why use several metrics?

Consider two models:

```text
Model A: lower MAE, slightly higher RMSE
Model B: higher MAE, lower RMSE
```

The difference may suggest that one model makes many moderate errors while the other makes fewer but more extreme errors.

Looking at several metrics can reveal such differences.

A sensible evaluation asks:

1. How large are the typical errors?
2. Are there large mistakes that deserve attention?
3. How much variation does the model explain?
4. Does performance hold on unseen data?

---

# 17. Actual versus predicted plots

A useful regression visualization is a scatter plot with:

- actual target values on the x-axis;
- predicted values on the y-axis.

Conceptually, the ideal relationship is:

```text
predicted
   |
   |        /
   |      /
   |    /
   |  /
   | /
   +-------------- actual
```

More precisely, perfect predictions would lie on the diagonal:

$$
\hat{y}=y
$$

So we often draw a reference line:

```python
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "--")
```

Points close to the diagonal indicate predictions close to the true values.

Systematic patterns away from the diagonal can reveal problems that a single summary metric may hide.

---

# 18. Residuals

A residual is:

$$
residual = y - \hat{y}
$$

Residual plots help us inspect whether the model's errors appear random or show systematic structure.

A useful idealized pattern is:

```text
residual
   |
 2 |    .  .
 1 | .     .   .
 0 |----------------------
-1 |   .  .   .
-2 |      .
   +---------------------- feature
```

We generally do not want a strong curved, funnel-shaped, or otherwise systematic pattern.

For example, a curved residual pattern can indicate that a linear model is missing nonlinear structure. That observation can motivate polynomial regression.

---

# 19. Feature scaling: when does it matter?

Feature scaling means transforming numerical features so that their numerical scales are more comparable.

A common transformation is standardization:

$$
 z = \frac{x-\mu}{\sigma}
$$

where:

- `μ` is the mean of the feature in the training data;
- `σ` is the standard deviation of the feature in the training data.

In scikit-learn, `StandardScaler` performs this transformation.

### Why scaling is sometimes important

Scaling is particularly important for models whose behavior depends on the scale of the features, including many regularized linear models and distance-based methods.

Ordinary least-squares `LinearRegression` does not require standardized features in the same way for the purpose of fitting predictions. Its coefficients are simply expressed in the units of the original features.

Polynomial features can also introduce very different numerical magnitudes, especially at high degrees, so scaling can become useful for numerical stability and for certain downstream estimators.

### A key rule: fit preprocessing on training data

If a scaler is used:

```python
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Do not fit the scaler separately on the test set. Otherwise, information from the test set influences the transformation used for evaluation.

---

# 20. Why the lab performs steps separately before using Pipeline

A machine-learning workflow can contain several operations:

```text
raw features
    |
    v
feature transformation
    |
    v
regression model
    |
    v
prediction
```

When these are written separately, students can see each object and each transformation.

A `Pipeline` combines sequential operations into one estimator-like object.

For example, conceptually:

```python
Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("model", LinearRegression()),
])
```

The pipeline does not create a new machine-learning algorithm. It organizes several steps into one workflow and helps ensure that transformations are applied consistently.

The lab therefore introduces `Pipeline` **after** the manual version. Once students understand:

```text
transform -> fit -> predict
```

they can understand what the pipeline is doing rather than treating it as a black box.

---

# 21. What is regularization?

Regularization is a general strategy for controlling model complexity by adding a penalty to the objective used during training.

The intuition is:

> Do not only ask the model to fit the training data. Also discourage unnecessarily large or complex parameter values.

Regularization can improve generalization, especially when features are numerous, correlated, noisy, or when a model is otherwise prone to overfitting.

There are two important regularization ideas related to this lab extension:

- **L2 regularization**
- **L1 regularization**

The main lab does not require these algorithms, but understanding them helps students see what comes after ordinary linear regression.

---

# 22. L2 regularization

L2 regularization adds a penalty proportional to the **sum of squared coefficients**.

For a generic linear regression model, the idea can be represented as:

$$
\text{loss} = \text{prediction error} + \lambda\sum_j b_j^2
$$

where `λ` controls the strength of the penalty.

The penalty encourages the coefficients to remain smaller in magnitude.

### Why square the coefficients?

Large coefficients are penalized more strongly because they are squared.

For example:

```text
coefficient = 2  -> square = 4
coefficient = 10 -> square = 100
```

L2 regularization therefore discourages extreme coefficient values without generally forcing coefficients exactly to zero.

---

# 23. Other regression algorithms

Linear and polynomial regression are useful starting points, but they are not the only regression methods.

Different problems have different characteristics, and different algorithms make different assumptions about the data.

The following methods are useful next steps:

| Algorithm | Main idea |
|---|---|
| Ridge Regression | Linear regression with **L2 regularization** |
| Lasso Regression | Linear regression with **L1 regularization**; can encourage coefficients to become exactly zero, which can perform feature selection |
| ElasticNet | Combines **L1 and L2 regularization** |
| Decision Tree Regressor | Learns a nonlinear, rule-based partitioning of the feature space |
| Random Forest Regressor | Ensemble of many decision trees whose predictions are combined |
| Gradient Boosting Regressor | Builds an ensemble sequentially, with later models focusing on reducing errors made by earlier ones |

These models extend the ideas learned in the current lab rather than replacing them.



### Ridge Regression: L2 regularization

Ridge regression uses an ordinary linear model with an L2 penalty.

Conceptually:

$$
\text{objective}
=
\text{sum of squared errors}
+
\lambda\sum_j b_j^2
$$

The effect is to shrink coefficients toward zero.

Ridge is often useful when:

- many features contribute to the prediction;
- features are correlated;
- ordinary least-squares coefficients are unstable or very large;
- we want regularization but do not want coefficients aggressively removed.

A larger regularization strength generally produces more shrinkage.

### Lasso Regression: L1 regularization and feature selection

Lasso adds a penalty based on the sum of absolute coefficient values:

$$
\text{objective}
=
\text{sum of squared errors}
+
\lambda\sum_j |b_j|
$$

The L1 penalty has an important property: it can drive some coefficients exactly to zero.

This means Lasso can perform a form of **feature selection**.

For example:

```text
before regularization:
feature A -> 1.8
feature B -> 0.1
feature C -> -2.2
feature D -> 0.02

after stronger L1 regularization:
feature A -> 1.5
feature B -> 0.0
feature C -> -1.9
feature D -> 0.0
```

The model has effectively removed features B and D from the fitted equation.


### ElasticNet: combining L1 and L2

ElasticNet combines L1 and L2 penalties.

Conceptually:

$$
\text{objective}
=
\text{prediction error}
+
\lambda_1\sum_j|b_j|
+
\lambda_2\sum_jb_j^2
$$

This gives the model properties associated with both forms of regularization.

ElasticNet can be useful when there are many correlated features and we want both shrinkage and the possibility of sparse coefficients.


### Decision Tree Regression

A decision tree does not fit a single straight-line equation across the entire feature space.

Instead, it repeatedly asks questions such as:

```text
Is TRIP_MILES < 3?
        /       \
      yes        no
      /           \
Is time < 10?   Is time < 20?
```

Eventually, the observations reach terminal regions called **leaves**, and the tree produces a numerical prediction for each region.

This makes decision trees naturally capable of representing nonlinear relationships and interactions between features.

Advantages include:

- nonlinear relationships;
- interactions between variables;
- little need for feature scaling.

A major risk is excessive tree depth, which can cause overfitting.


### Random Forest Regression

A random forest is an ensemble of decision trees.

Instead of relying on one tree, the algorithm builds many trees and combines their predictions.

The basic intuition is:

```text
Tree 1  ---> prediction 1
Tree 2  ---> prediction 2
Tree 3  ---> prediction 3
   ...
Tree N  ---> prediction N
               |
               v
        combined prediction
```

Combining many trees can reduce the instability associated with a single decision tree and often produces stronger generalization.

Random forests are useful when relationships are nonlinear and there are interactions between variables.


### Gradient Boosting Regression

Gradient boosting is another ensemble approach, but it builds its models **sequentially** rather than independently.

A simplified intuition is:

```text
initial model
     |
     v
find errors
     |
     v
add another model that helps reduce those errors
     |
     v
find remaining errors
     |
     v
add another model
     |
     v
final combined prediction
```

The new models are trained to improve the current ensemble.

Gradient boosting can be highly effective for tabular regression problems, but model complexity and hyperparameters need to be controlled carefully to avoid overfitting.


### Linear models versus tree-based models

The main conceptual difference can be summarized as follows.

**Linear and polynomial approaches**

They define predictions through mathematical combinations of input features.

For example:

$$
\hat{y}=b_0+b_1x_1+b_2x_2
$$

or:

$$
\hat{y}=b_0+b_1x+b_2x^2
$$

These models are often easier to interpret because the coefficients have a direct mathematical meaning.

**Tree-based approaches**

Trees divide the feature space into regions using decision rules.

They can naturally model nonlinear relationships and feature interactions without explicitly creating polynomial terms.

This flexibility can improve predictive performance, but the resulting model is often less straightforward to interpret than a simple regression equation.

---

# 24. Why start with simple models?

It is tempting to immediately use a powerful algorithm and optimize its parameters. That is often not the best way to learn or to build a trustworthy modeling workflow.

A simple baseline gives us a reference point.

For example:

```text
Baseline
   |
   v
Linear regression
   |
   v
Multiple linear regression
   |
   v
Polynomial regression
   |
   v
Regularized models
   |
   v
Tree ensembles
```

A more complicated model should earn its complexity by providing a meaningful improvement in performance, robustness, interpretability, or some other objective.

---

# 25. A complete regression workflow

The ideas in the lab fit into a general workflow:

```text
1. Understand the problem
          |
          v
2. Inspect the data
          |
          v
3. Identify features and target
          |
          v
4. Explore relationships
          |
          v
5. Split training and test data
          |
          v
6. Choose a baseline model
          |
          v
7. Fit the model using training data
          |
          v
8. Predict on unseen data
          |
          v
9. Evaluate with appropriate metrics
          |
          v
10. Inspect plots and residuals
          |
          v
11. Try a more suitable model if needed
          |
          v
12. Compare models fairly
```

This workflow is more important than memorizing individual scikit-learn class names.

---

# 26. Key ideas students should remember

### Regression

Regression predicts a continuous numerical target.

### Linear regression

A simple linear model uses one feature:

$$
\hat{y}=b_0+b_1x
$$

### Multiple linear regression

Multiple linear regression uses several features:

$$
\hat{y}=b_0+b_1x_1+b_2x_2+\cdots+b_px_p
$$

It is still a linear model.

### Polynomial regression

Polynomial regression adds transformed terms such as `x²`, `x³`, and interactions, then uses a linear estimator on those transformed features.

### Pearson correlation

Pearson's `r` measures the strength and direction of a **linear** association between two numerical variables. It ranges from -1 to +1.

### Train/test split

The model learns from the training data. The test set is held back for evaluating generalization.

### Underfitting

The model is too simple and fails to capture important structure.

### Overfitting

The model is too flexible and fits training-specific noise or details that do not generalize.

### MAE

Average absolute prediction error.

### RMSE

Square root of the average squared prediction error; larger mistakes receive more weight.

### R²

Measures performance relative to predicting the mean target value.

### L2 regularization

Adds a squared-coefficient penalty and shrinks coefficients toward zero. Ridge regression uses L2 regularization.

### Other regression algorithms

Ridge, Lasso, ElasticNet, Decision Trees, Random Forests, and Gradient Boosting provide additional choices when ordinary linear or polynomial regression is not sufficient.

---

# 27. Connection to the lab

The lab turns these ideas into practice in the following order:

```text
Explore the taxi data
        |
        v
Pearson correlation and visualization
        |
        v
Simple Linear Regression
        |
        v
Multiple Linear Regression
        |
        v
Polynomial Regression
        |
        v
Evaluate MAE / RMSE / R²
        |
        v
Inspect actual vs. predicted values
        |
        v
Think about underfitting / overfitting
        |
        v
Optional Pipeline versions
```

The goal is not simply to obtain the smallest error. Students should learn to explain:

> What is the model learning, what assumptions is it making, how well does it generalize, and what evidence supports choosing one model over another?

---

# References and further reading

This theory chapter accompanies the scikit-learn taxi regression lab.

The lab activity was adapted from Google's original **Linear Regression Taxi** exercise. The original notebook contains the source references and citations used for the dataset and activity.

- Google Colab notebook: https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/linear_regression_taxi.ipynb
- Google Engineering Education repository: https://github.com/google/eng-edu/tree/main/ml/cc/exercises
