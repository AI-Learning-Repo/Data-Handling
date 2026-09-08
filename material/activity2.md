# Chicago Taxi Fare Regression with scikit-learn

## A hands-on regression lab with scikit-learn

In this lab, you will build and evaluate regression models for predicting Chicago taxi fares with **scikit-learn**. You will work through each step explicitly so that you understand what the data, transformations, models, and evaluation metrics are doing. After the manual implementations, you will revisit the same ideas using `Pipeline` as an optional final exercise.

The goal here is not to hide machine learning behind a convenient API. The goal is to understand the pieces.

> **Important:** In the main part of this exercise, **do not use `Pipeline`**. For each model, you will explicitly perform the necessary preprocessing, create the model, fit it, predict with it, and evaluate it. Only after you understand those steps will you repeat the same three models using `Pipeline` as an optional final exercise.

---

## What you will learn

By the end of this exercise, you should be able to:

1. Load and inspect a real regression dataset.
2. Identify a target variable and candidate input features.
3. Use correlation to build an initial hypothesis about useful numerical features.
4. Split data into training and test sets.
5. Train a **simple linear regression** model with one feature.
6. Train a **multiple linear regression** model with two features.
7. Understand the difference between one-feature and multi-feature regression.
8. Create polynomial features explicitly and use them with linear regression.
9. Evaluate regression models with **MAE, RMSE, and R²**.
10. Interpret regression coefficients.
11. Understand when feature scaling matters and when it does not.
12. Compare models using the same evaluation procedure.
13. Understand what a `Pipeline` does rather than treating it as magic.

### The three main algorithms

We will deliberately stop after these three model families:

- **Linear Regression** with one feature.
- **Multiple Linear Regression** with two features.
- **Polynomial Regression**, implemented explicitly as `PolynomialFeatures + LinearRegression`.

Polynomial regression is not a separate estimator in scikit-learn. It is a modeling technique in which we transform the original features into polynomial features and then fit a linear regression model to the transformed data.

After the manual implementations, we will repeat the same three approaches using `Pipeline`.

---

# Part 1: Setup

## 1.1 Install and import the libraries

### Code

```python
# Uncomment this line in Google Colab if a package is missing.
!pip install -q scikit-learn pandas numpy matplotlib
```


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

### What each import is for

#### `numpy`

NumPy is the basic numerical computing library used by pandas and scikit-learn. We will use it here mainly for numerical operations such as taking the square root of the mean squared error.

#### `pandas`

Pandas gives us the `DataFrame`, which we use as a table for the taxi data.

#### `matplotlib.pyplot`

Matplotlib is used for charts such as:

- actual vs. predicted fare plots;
- residual plots;
- polynomial curves.

#### `train_test_split`

This function divides the data into a training portion and a test portion.

The key idea is:

```text
training data -> used to learn model parameters
                    |
                    v
                  model
                    |
                    v
test data ------> prediction
                    |
                    v
                evaluation
```

The test set should not be used to fit the model. Otherwise, the evaluation becomes much less trustworthy because the model has already seen the examples.

#### `LinearRegression`

This is scikit-learn's ordinary least-squares linear regression estimator.

#### `PolynomialFeatures`

This transformer creates additional features such as squares, cubes, and interactions.

For two features `x1` and `x2`, degree 2 can create terms conceptually like:

```text
1
x1
x2
x1²
x1*x2
x2²
```

#### `Pipeline`

We intentionally import it now but will **not use it in the main exercises**. It is saved for the optional final section.

#### Regression metrics

We will use three complementary metrics:

- **MAE**: Mean Absolute Error.
- **RMSE**: Root Mean Squared Error.
- **R²**: coefficient of determination.

---

## 1.2 Load the dataset

The dataset contains Chicago taxi trip information from a subset of City of Chicago Taxi Trips data.

### Code

```python
chicago_taxi_dataset = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv"
)

print(f"Rows: {len(chicago_taxi_dataset):,}")
print(f"Columns: {len(chicago_taxi_dataset.columns)}")
```

### Why use the URL directly?

`pd.read_csv()` can read a CSV file from a URL. This means you do not need to manually download the dataset first.

The important part is that `read_csv()` returns a pandas `DataFrame`.

---

## 1.3 Keep the columns that are relevant to the exercise

```python
training_df = chicago_taxi_dataset.loc[:, [
    "TRIP_MILES",
    "TRIP_SECONDS",
    "FARE",
    "COMPANY",
    "PAYMENT_TYPE",
    "TIP_RATE",
]].copy()

training_df.head()
```

### Why use `.copy()`?

Using `.copy()` makes it explicit that we are creating a separate DataFrame rather than merely holding another view onto the original object. This also helps avoid confusing pandas warnings when new columns are added later.

### What are the important columns?

- `TRIP_MILES`: trip distance in miles.
- `TRIP_SECONDS`: trip duration in seconds.
- `FARE`: the target we want to predict.
- `COMPANY`: taxi company.
- `PAYMENT_TYPE`: payment method.
- `TIP_RATE`: tip rate.

For our three regression models, the most important variables are numerical:

```text
TRIP_MILES
TRIP_SECONDS
FARE
```

We will derive `TRIP_MINUTES` from `TRIP_SECONDS` so that trip duration is easier to interpret in our regression models.

---

## 1.4 Create trip duration in minutes

```python
training_df["TRIP_MINUTES"] = training_df["TRIP_SECONDS"] / 60

training_df[["TRIP_SECONDS", "TRIP_MINUTES"]].head()
```

### Explanation

Every value in `TRIP_SECONDS` is divided by 60.

For example:

```text
300 seconds / 60 = 5 minutes
600 seconds / 60 = 10 minutes
900 seconds / 60 = 15 minutes
```

This is a simple feature-engineering step.

The model does not care that one feature is a converted version of another. What matters is that the feature is expressed in a useful representation.

---

# Part 2: Explore the data before modeling

Before training a model, inspect the data carefully. This gives you a chance to understand the variables and form a hypothesis about which features may be useful.

Do the same here.

---

## 2.1 Inspect basic statistics

### Code

```python
training_df.describe(include="all")
```

### What are you looking for?

For numerical columns, pay attention to:

- `count`
- `mean`
- `std`
- `min`
- quartiles (`25%`, `50%`, `75%`)
- `max`

For categorical columns, `include="all"` allows pandas to show useful information such as the number of unique values and the most frequent value.

Use the statistics to answer basic questions about the data before modeling.

---

## Check your understanding

**Question 1:** What is the maximum fare?

**Question 2:** What is the average trip distance?

**Question 3:** How many unique taxi companies are present?

**Question 4:** What is the most common payment type?

**Question 5:** Are there missing values in the columns we selected?

### Prompt you can give to an LLM

```text
I am analyzing the Chicago taxi regression dataset in pandas.

I ran:

training_df.describe(include="all")

and I need help understanding the output.

Please explain how I can answer these questions using pandas, but do NOT give me the answers immediately:

1. How do I find the maximum value of FARE?
2. How do I calculate the mean of TRIP_MILES?
3. How do I count unique COMPANY values?
4. How do I find the most frequent PAYMENT_TYPE?
5. How do I check whether any values are missing?

Give me hints first. If I still cannot solve it, show the exact pandas expressions and explain each one.
```

<details>
<summary><strong>Solution: expand if you are stuck</strong></summary>

```python
max_fare = training_df["FARE"].max()
mean_distance = training_df["TRIP_MILES"].mean()
num_companies = training_df["COMPANY"].nunique()
most_common_payment = training_df["PAYMENT_TYPE"].value_counts().idxmax()
missing_values = training_df.isna().sum()

print("Maximum fare:", max_fare)
print("Mean distance:", mean_distance)
print("Unique companies:", num_companies)
print("Most common payment type:", most_common_payment)
print("Missing values:\n", missing_values)
```

Use your pandas output to check the maximum fare, average trip distance, number of companies, most common payment type, and missing-value counts.

### Why each method works

- `.max()` finds the largest numerical value.
- `.mean()` calculates the arithmetic mean.
- `.nunique()` counts different values rather than rows.
- `.value_counts()` counts the occurrences of each category.
- `.idxmax()` returns the label associated with the largest count.
- `.isna()` creates a Boolean mask indicating missing values.

</details>

---

## 2.2 Inspect correlations

Use a correlation matrix to examine which numerical features are strongly associated with `FARE`.

### Code

```python
correlation_matrix = training_df.corr(numeric_only=True)
correlation_matrix
```

You can also inspect the correlation specifically with `FARE`:

```python
fare_correlations = correlation_matrix["FARE"].sort_values(ascending=False)
fare_correlations
```

### How to interpret correlation

A correlation coefficient is between `-1` and `+1`.

```text
 +1  -> perfect positive linear relationship
  0  -> no linear relationship
 -1  -> perfect negative linear relationship
```

A strong positive correlation means that, in general, larger values of one variable tend to occur with larger values of the other variable.

A strong correlation can be useful for choosing an initial feature, but **correlation is not proof of causation**, and correlation alone does not tell you whether a model will generalize well.

Use the correlation values to decide which numerical feature is a sensible starting point for a first regression model.

---

## Check your understanding

**Question 1:** Which feature has the strongest correlation with `FARE`?

**Question 2:** Does a high correlation automatically mean a feature is always useful in a machine-learning model?

**Question 3:** Why might `TRIP_SECONDS` and `TRIP_MINUTES` be highly correlated with each other?

### Prompt for an LLM

```text
I have this pandas correlation matrix:

fare_correlations = training_df.corr(numeric_only=True)["FARE"].sort_values(ascending=False)
print(fare_correlations)

Help me interpret it like a machine-learning instructor.

Please explain:
1. What a positive correlation means here.
2. What a value close to zero means.
3. How I can identify the feature most strongly related to FARE.
4. Why correlation should not be treated as proof that a feature will improve a model.
5. Why TRIP_SECONDS and TRIP_MINUTES should be strongly correlated.

Do not just give the final answer. Teach me how to read the table.
```

<details>
<summary><strong>Solution: expand if you are stuck</strong></summary>

`TRIP_MILES` is the strongest numerical correlate of `FARE` in this dataset. `TRIP_SECONDS` is also strongly related to fare. `TRIP_MINUTES` is just `TRIP_SECONDS / 60`, so it contains the same information and therefore should have exactly the same correlation with another variable up to numerical rounding.

A high correlation does not automatically guarantee better predictive performance because:

- correlation measures linear association;
- two features can contain almost the same information;
- a relationship can change after other features are included;
- a feature can correlate with the target but fail to generalize;
- model performance must ultimately be evaluated on data the model did not train on.

</details>

---

## 2.3 Visualize the data

Before fitting models, make the data visible.

### Distance vs. fare

```python
plt.figure(figsize=(8, 5))
plt.scatter(training_df["TRIP_MILES"], training_df["FARE"], alpha=0.3)
plt.xlabel("Trip miles")
plt.ylabel("Fare ($)")
plt.title("Trip distance vs. taxi fare")
plt.show()
```

### Duration vs. fare

```python
plt.figure(figsize=(8, 5))
plt.scatter(training_df["TRIP_MINUTES"], training_df["FARE"], alpha=0.3)
plt.xlabel("Trip duration (minutes)")
plt.ylabel("Fare ($)")
plt.title("Trip duration vs. taxi fare")
plt.show()
```

### Why use a scatter plot?

We are looking for the **shape** of the relationship.

For example:

- approximately straight line → linear regression may be reasonable;
- clearly curved relationship → polynomial or another nonlinear model may be useful;
- fan-shaped residuals → the error may change with feature magnitude;
- strong outliers → a few trips may have disproportionate influence.

Do not decide which model is best only from the picture. Use the picture to form a hypothesis, then test it.

---

# Part 3: A common evaluation procedure

Before training our models, we need a consistent way to evaluate them.

This section is deliberately explicit because later we will repeat the same steps for each algorithm.

---

## 3.1 Define features and target

For the first model we use one feature:

```python
X = training_df[["TRIP_MILES"]]
y = training_df["FARE"]
```

Notice the double brackets around `TRIP_MILES`:

```python
training_df[["TRIP_MILES"]]
```

This returns a **DataFrame** with one column.

Using one pair of brackets:

```python
training_df["TRIP_MILES"]
```

returns a **Series**.

Scikit-learn expects `X` to normally be two-dimensional:

```text
number of rows x number of features
```

So for one feature we still want a shape like:

```text
(n_samples, 1)
```

rather than:

```text
(n_samples,)
```

---

## 3.2 Split the data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)
```

### What happens here?

Suppose the dataset has `N` examples.

With `test_size=0.20`:

- approximately 80% become training data;
- approximately 20% become test data.

`random_state=42` makes the random split reproducible.

That means that another person running the same code gets the same split, assuming the same data and library behavior.

### Why do we need four variables?

```text
X_train -> input features for training
X_test  -> input features for testing

y_train -> true target values for training
 y_test -> true target values for testing
```

---

## 3.3 Create a small evaluation helper

We will use the same metric calculations repeatedly. This is not a modeling pipeline; it is only a convenience function for evaluation.

```python
def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"MAE : ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R²  : {r2:.4f}")
```

### Why calculate three metrics?

#### MAE

Mean Absolute Error is:

```text
average(|actual - prediction|)
```

If MAE is `$2.50`, the average absolute prediction error is about $2.50.

MAE is easy to explain because it uses the original target units.

#### RMSE

Root Mean Squared Error is:

```text
sqrt(average((actual - prediction)²))
```

Because errors are squared before averaging, larger errors receive more weight.

This makes RMSE especially sensitive to large mistakes.

#### R²

R² measures how much of the variation in the target is explained by the model relative to a baseline that predicts the mean.

A value closer to `1` indicates stronger explanatory performance on the evaluated data.

R² can be negative on test data when a model performs worse than the mean-prediction baseline.

### Which metric should you trust?

Do not treat one metric as universally best.

For taxi fares, a useful combination is:

- **MAE** for easy business interpretation;
- **RMSE** for sensitivity to large mistakes;
- **R²** for a standardized view of explained variation.

---

# Part 4: Model 1: Simple Linear Regression

## 4.1 What are we trying to model?

Our first hypothesis is:

> Can trip distance alone predict taxi fare reasonably well?

In mathematical notation, simple linear regression uses:

$$
\hat{y} = b_0 + b_1 x
$$

For our taxi example:

$$
\widehat{FARE} = b_0 + b_1(TRIP\_MILES)
$$

Where:

- $b_0$ is the intercept;
- $b_1$ is the slope/coefficient;
- `TRIP_MILES` is the input feature;
- predicted `FARE` is the output.

Scikit-learn estimates the coefficients using ordinary least squares.

---

## 4.2 Step 1: Select the feature and label

```python
X = training_df[["TRIP_MILES"]]
y = training_df["FARE"]
```

### Explanation

We are making a clear separation between:

```text
X = inputs

y = target
```

This is one of the most important conventions in scikit-learn.

---

## 4.3 Step 2: Split the data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)
```

We fit on `X_train` and `y_train`.

We evaluate on `X_test` and `y_test`.

---

## 4.4 Step 3: Create the model

```python
model_lr_1 = LinearRegression()
```

At this moment the model has not learned anything useful yet.

We have only created an estimator object.

Think of it as:

```text
LinearRegression object
        |
        | fit(...)
        v
learned coefficients
```

---

## 4.5 Step 4: Train the model

```python
model_lr_1.fit(X_train, y_train)
```

### What does `fit()` do?

It looks at the training examples and estimates the coefficient and intercept that minimize the sum of squared residuals.

After fitting, we can inspect the learned values:

```python
print("Coefficient:", model_lr_1.coef_[0])
print("Intercept:", model_lr_1.intercept_)
```

### How to interpret the coefficient

Suppose the coefficient were approximately `b` dollars per mile.

Then, in a simplified interpretation:

> A one-mile increase in trip distance is associated with an estimated increase of about `b` dollars in predicted fare, holding the model structure fixed.

Do not automatically interpret the coefficient as a causal statement.

---

## 4.6 Step 5: Predict

```python
y_pred_lr_1 = model_lr_1.predict(X_test)
```

`predict()` takes the feature values and applies the learned regression equation.

The output is an array with one predicted fare for every test row.

---

## 4.7 Step 6: Evaluate

```python
evaluate_regression(y_test, y_pred_lr_1)
```

### Important distinction

We are evaluating:

```text
predictions made on X_test
```

against:

```text
true fares in y_test
```

This is much more informative than evaluating the training examples that the model already used to learn its coefficients.

---

## 4.8 Step 7: Visualize predictions

```python
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred_lr_1, alpha=0.3)

min_fare = min(y_test.min(), y_pred_lr_1.min())
max_fare = max(y_test.max(), y_pred_lr_1.max())
plt.plot([min_fare, max_fare], [min_fare, max_fare])

plt.xlabel("Actual fare ($)")
plt.ylabel("Predicted fare ($)")
plt.title("Simple Linear Regression: actual vs predicted")
plt.show()
```

### What does the diagonal line mean?

The line represents perfect predictions:

```text
predicted = actual
```

A point exactly on the line means the model predicted that fare perfectly.

Points far away from the line represent larger errors.

---

## Check your understanding

**Question 1:** What equation did the model learn?

**Question 2:** What does `model.coef_` represent?

**Question 3:** What does `model.intercept_` represent?

**Question 4:** Why do we evaluate using `X_test` rather than `X_train`?

**Question 5:** What does the ideal diagonal line mean in the actual-vs-predicted plot?

**Question 6:** Which metric is more strongly affected by a few very large errors: MAE or RMSE?

### Prompt for an LLM

```text
I am learning scikit-learn LinearRegression using the Chicago taxi dataset.

Here is my code:

X = training_df[["TRIP_MILES"]]
y = training_df["FARE"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

Please teach me what each line is doing.

Then help me answer these questions without giving the final answers immediately:
1. What does coef_ mean?
2. What does intercept_ mean?
3. Why do X and y have different roles?
4. Why is the test set not used by fit()?
5. Why is RMSE more sensitive to large errors than MAE?

Give me hints first. If I ask for the solution, give a concrete explanation and a small numerical example.
```

<details>
<summary><strong>Solution: expand if you are stuck</strong></summary>

### 1. The equation

The model learns:

$$
\hat{y} = b_0 + b_1x
$$

For this dataset:

```text
predicted fare = intercept + coefficient * TRIP_MILES
```

### 2. `coef_`

`model.coef_[0]` is the estimated slope for `TRIP_MILES`.

If it were, for example, `2.5`, then increasing distance by one mile would increase the model's predicted fare by approximately `$2.50`, all else in this one-feature model being equal.

### 3. `intercept_`

`model.intercept_` is the estimated predicted fare when `TRIP_MILES = 0`.

Whether that is a meaningful real-world taxi fare is a separate question. A mathematically useful intercept is not necessarily a realistic business interpretation.

### 4. Why use the test set?

The purpose of the test set is to simulate future unseen examples.

If we fit and evaluate on the same rows, the result can make the model look better than it will perform on new data.

### 5. Why RMSE emphasizes large errors

Suppose two prediction errors are:

```text
error A = $1
error B = $10
```

MAE treats them as:

```text
1 and 10
```

RMSE squares them before averaging:

```text
1²  = 1
10² = 100
```

The `$10` mistake therefore receives dramatically more influence.

</details>

---

## Extra exercise: inspect the residuals

A residual is:

$$
residual = actual - predicted
$$

Create residuals:

```python
residuals_lr_1 = y_test - y_pred_lr_1

plt.figure(figsize=(8, 5))
plt.scatter(y_pred_lr_1, residuals_lr_1, alpha=0.3)
plt.axhline(0)
plt.xlabel("Predicted fare ($)")
plt.ylabel("Residual ($)")
plt.title("Simple Linear Regression residuals")
plt.show()
```

### Question

What would you ideally like the residual plot to look like?

### LLM prompt

```text
Explain residual plots for a regression model as if I am learning them for the first time.

I have:

residuals = y_test - y_pred

and plotted predicted values on the x-axis and residuals on the y-axis.

Tell me what patterns I should look for and what these patterns can indicate:
- curvature
- increasing spread
- clusters
- extreme outliers

Do not assume my model is good or bad. Teach me how to diagnose it from the plot.
```

<details>
<summary><strong>Solution: expand if you are stuck</strong></summary>

Ideally, residuals are scattered around zero without a strong systematic pattern.

Patterns can be informative:

- **Curvature** can suggest that a straight-line model is missing nonlinear structure.
- **Increasing spread** can suggest that the error variance changes with the predicted value.
- **Clusters** can suggest groups of observations behaving differently.
- **Extreme points** may be outliers or legitimate unusual trips.

A residual plot is a diagnostic tool, not a proof by itself.

</details>

---

# Part 5: Model 2: Multiple Linear Regression

Now we ask a stronger question:

> Can we predict fare better when we use both distance and trip duration?

Next, extend the model by adding trip duration as a second input feature.

The model becomes:

$$
\widehat{FARE} = b_0 + b_1(TRIP\_MILES) + b_2(TRIP\_MINUTES)
$$

This is still **linear regression**.

The word "multiple" means that there is more than one input feature.

---

## 5.1 Step 1: Select two features

```python
X = training_df[["TRIP_MILES", "TRIP_MINUTES"]]
y = training_df["FARE"]
```

Now `X` has two columns.

Its conceptual structure is:

```text
row 1 -> [miles, minutes]
row 2 -> [miles, minutes]
row 3 -> [miles, minutes]
...
```

---

## 5.2 Step 2: Split the data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)
```

Use the same split settings so that our comparison is fair.

---

## 5.3 Step 3: Create the model

```python
model_lr_2 = LinearRegression()
```

Notice that the estimator is exactly the same class as before.

The difference is not the estimator.

The difference is the number of input features.

---

## 5.4 Step 4: Train

```python
model_lr_2.fit(X_train, y_train)
```

---

## 5.5 Step 5: Inspect the coefficients

```python
print("Intercept:", model_lr_2.intercept_)

for feature, coefficient in zip(X.columns, model_lr_2.coef_):
    print(f"{feature}: {coefficient}")
```

### Why are there two coefficients?

Because there are two input columns.

Conceptually:

```text
coef_[0] -> effect of TRIP_MILES
coef_[1] -> effect of TRIP_MINUTES
```

The model is now a plane rather than a line when viewed in three-dimensional feature/target space.

---

## 5.6 Step 6: Predict

```python
y_pred_lr_2 = model_lr_2.predict(X_test)
```

---

## 5.7 Step 7: Evaluate

```python
evaluate_regression(y_test, y_pred_lr_2)
```

Now compare the metrics with Model 1.

---

## 5.8 Compare Model 1 and Model 2

```python
results_manual = pd.DataFrame({
    "Model": [
        "Linear Regression — TRIP_MILES",
        "Multiple Linear Regression — TRIP_MILES + TRIP_MINUTES",
    ],
    "MAE": [
        mean_absolute_error(y_test, y_pred_lr_1),
        mean_absolute_error(y_test, y_pred_lr_2),
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y_test, y_pred_lr_1)),
        np.sqrt(mean_squared_error(y_test, y_pred_lr_2)),
    ],
    "R2": [
        r2_score(y_test, y_pred_lr_1),
        r2_score(y_test, y_pred_lr_2),
    ],
})

results_manual
```

### Important comparison rule

Use the **same test set** when comparing these models.

If Model A is tested on one random split and Model B on another random split, differences in performance could be caused partly by the split rather than the model.

---

## Check your understanding

**Question 1:** Why is Model 2 called multiple linear regression?

**Question 2:** Is Model 2 nonlinear just because it has two features?

**Question 3:** Why do we now have two coefficients?

**Question 4:** What would a change in `TRIP_MINUTES` coefficient mean while holding `TRIP_MILES` constant in the model?

**Question 5:** If Model 2 has lower RMSE and lower MAE and higher R² on the same test set, what does that suggest?

**Question 6:** Why can two strongly correlated features create interpretation problems even when the model still makes useful predictions?

### Prompt for an LLM

```text
I am learning multiple linear regression with scikit-learn.

My model is:

X = training_df[["TRIP_MILES", "TRIP_MINUTES"]]
y = training_df["FARE"]

model = LinearRegression()
model.fit(X_train, y_train)

Please explain the model mathematically and in plain language.

Then help me reason about these questions:
1. Why is this still a linear model?
2. Why are there two coefficients?
3. What does the TRIP_MINUTES coefficient mean when TRIP_MILES is held constant?
4. Why can adding a feature improve prediction but make coefficient interpretation harder?
5. Why is using the same test set important when comparing two models?

Give hints before answers.
```

<details>
<summary><strong>Solution: expand if you are stuck</strong></summary>

### Why is it still linear?

Because the prediction is a linear combination of the input features:

$$
\hat{y} = b_0 + b_1x_1 + b_2x_2
$$

There is no squared feature, exponential transformation, tree split, or other nonlinear transformation in this basic model.

Having multiple variables does not make the model nonlinear.

### Why two coefficients?

There is one coefficient per input feature:

```text
TRIP_MILES   -> coefficient 1
TRIP_MINUTES -> coefficient 2
```

### Meaning of a coefficient

The coefficient for `TRIP_MINUTES` tells us how the predicted fare changes when duration changes by one minute, **while the other feature is held constant according to the model**.

### Why interpretation gets harder

`TRIP_MILES` and `TRIP_MINUTES` may contain overlapping information. Longer trips often take longer. When predictors are strongly related, coefficients can become less stable or harder to interpret individually even though the combined predictions remain useful.

### Why compare on the same test set?

A fair experiment changes one major factor at a time. Using the same held-out examples makes the model comparison much cleaner.

</details>

---

# Part 6: Model 3: Polynomial Regression

Now we ask a different question:

> What if the relationship between the features and fare is curved rather than perfectly straight?

This is where polynomial regression becomes useful.

---

## 6.1 Polynomial regression is not a completely different regressor

A common misunderstanding is that scikit-learn has a `PolynomialRegression` estimator similar to `LinearRegression`.

Instead, we usually do two things:

1. transform the input features with `PolynomialFeatures`;
2. fit `LinearRegression` to the transformed features.

Without a pipeline, the process is explicit:

```text
original features
       |
       v
PolynomialFeatures
       |
       v
expanded feature matrix
       |
       v
LinearRegression
       |
       v
predictions
```

That explicit sequence is exactly what we want you to understand before using a pipeline.

---

## 6.2 Start with one feature for clarity

We will first use only `TRIP_MILES`.

```python
X = training_df[["TRIP_MILES"]]
y = training_df["FARE"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)
```

---

## 6.3 Create the polynomial transformer

```python
poly = PolynomialFeatures(degree=2, include_bias=False)
```

### What does `degree=2` mean?

For one feature `x`, degree 2 gives:

```text
x
x²
```

If we used `degree=3`, we would get:

```text
x
x²
x³
```

### Why `include_bias=False`?

By default, `PolynomialFeatures` can add a column of ones.

That column corresponds to the intercept term.

Because `LinearRegression` already handles the intercept by default, setting `include_bias=False` avoids adding an unnecessary duplicate constant feature.

---

## 6.4 Transform the training data

```python
X_train_poly = poly.fit_transform(X_train)
```

### Why `fit_transform()` here?

For this transformer, `fit_transform()` learns the structure needed for the transformation from the training data and applies the transformation.

The resulting columns are approximately:

```text
TRIP_MILES
TRIP_MILES²
```

You can inspect the shape:

```python
print("Original shape:", X_train.shape)
print("Polynomial shape:", X_train_poly.shape)
```

You should see the number of rows stay the same while the number of columns increases.

---

## 6.5 Transform the test data

```python
X_test_poly = poly.transform(X_test)
```

Notice that we use **`transform()`**, not `fit_transform()`.

This distinction matters.

For supervised machine learning, we generally fit preprocessing steps using training data only and then apply the already-learned transformation to test data.

Here the transformation is deterministic and simple, but the principle is important and will become critical for transformations such as scaling or imputation.

---

## 6.6 Inspect the transformed features

```python
print(X_train_poly[:5])
```

For an input such as:

```text
TRIP_MILES = 3
```

the transformed representation is conceptually:

```text
[3, 9]
```

because:

```text
3² = 9
```

---

## 6.7 Create the linear regression model

```python
model_poly_2 = LinearRegression()
```

This may look surprising.

We are doing polynomial regression with `LinearRegression`.

That is because the model is linear in its **parameters** even though it is nonlinear in the original input variable.

The resulting equation is:

$$
\hat{y} = b_0 + b_1x + b_2x^2
$$

The relationship with the original `x` can therefore be curved.

---

## 6.8 Train on the transformed features

```python
model_poly_2.fit(X_train_poly, y_train)
```

We are **not** fitting on the original `X_train`.

We are fitting on the expanded matrix `X_train_poly`.

---

## 6.9 Predict using transformed test data

```python
y_pred_poly_2 = model_poly_2.predict(X_test_poly)
```

---

## 6.10 Evaluate

```python
evaluate_regression(y_test, y_pred_poly_2)
```

Then compare with the simple linear model:

```python
print("Linear Regression")
evaluate_regression(y_test, y_pred_lr_1)

print("\nPolynomial Regression (degree=2)")
evaluate_regression(y_test, y_pred_poly_2)
```

---

## 6.11 Visualize the polynomial curve

For visualization, create an ordered range of distance values.

```python
x_curve = np.linspace(
    X["TRIP_MILES"].min(),
    X["TRIP_MILES"].max(),
    300,
).reshape(-1, 1)

x_curve_poly = poly.transform(x_curve)
y_curve = model_poly_2.predict(x_curve_poly)

plt.figure(figsize=(9, 6))
plt.scatter(X_test["TRIP_MILES"], y_test, alpha=0.25, label="Test observations")
plt.plot(x_curve, y_curve, linewidth=2, label="Polynomial regression")
plt.xlabel("Trip miles")
plt.ylabel("Fare ($)")
plt.title("Polynomial regression, degree 2")
plt.legend()
plt.show()
```

### Why use `np.linspace()`?

We want many evenly spaced x-values so that the predicted curve looks smooth.

We are not creating new taxi trips for the model. We are only asking:

> If the trip distance were this value, what fare would the model predict?

for many distances between the minimum and maximum observed values.

---

## Check your understanding

**Question 1:** Why can `LinearRegression` be used for polynomial regression?

**Question 2:** What new feature does degree 2 add when there is only one original feature?

**Question 3:** Why do we use `fit_transform()` on the training data but `transform()` on the test data?

**Question 4:** What is one risk of increasing the polynomial degree too much?

**Question 5:** Why might a polynomial model fit the training data better but perform worse on unseen data?

**Question 6:** What is the difference between being nonlinear in the input feature and being nonlinear in the model parameters?

### Prompt for an LLM

```text
I am learning polynomial regression in scikit-learn without using Pipeline.

My code is:

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)
y_pred = model.predict(X_test_poly)

Please teach me exactly what happens to the feature matrix.

Use a tiny example such as:

X = [[2], [3], [4]]

and show what degree=2 does to it.

Then explain:
1. Why LinearRegression still works.
2. Why this creates a curved relationship.
3. Why transform() is used for the test data.
4. Why a high degree can cause overfitting.

Give me hints first and solutions second.
```

<details>
<summary><strong>Solution: expand if you are stuck</strong></summary>

For:

```python
X = [[2], [3], [4]]
```

a degree-2 polynomial transformation with `include_bias=False` creates:

```text
[2, 4]
[3, 9]
[4, 16]
```

The first column is `x`.

The second is `x²`.

The linear regression model then learns:

$$
\hat{y} = b_0 + b_1x + b_2x^2
$$

This is still linear in the parameters `b0`, `b1`, and `b2`, but it can form a curved relationship as a function of `x`.

A higher degree adds more terms. More flexibility can reduce training error, but excessive flexibility can cause the model to learn noise rather than the general relationship.

That is overfitting.

</details>

---

# Part 7: Polynomial Regression with two original features

Now combine the ideas from Model 2 and Model 3.

Original features:

```text
TRIP_MILES
TRIP_MINUTES
```

Polynomial degree 2 creates terms conceptually such as:

```text
TRIP_MILES
TRIP_MINUTES
TRIP_MILES²
TRIP_MILES * TRIP_MINUTES
TRIP_MINUTES²
```

So the model can learn a richer relationship.

---

## 7.1 Prepare the data

```python
X = training_df[["TRIP_MILES", "TRIP_MINUTES"]]
y = training_df["FARE"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)
```

---

## 7.2 Create and fit the transformer

```python
poly_2_features = PolynomialFeatures(
    degree=2,
    include_bias=False,
)

X_train_poly_2_features = poly_2_features.fit_transform(X_train)
X_test_poly_2_features = poly_2_features.transform(X_test)
```

You can inspect the feature names:

```python
poly_feature_names = poly_2_features.get_feature_names_out(X.columns)
print(poly_feature_names)
```

This is extremely useful for learning because it shows exactly what scikit-learn created.

---

## 7.3 Train

```python
model_poly_2_features = LinearRegression()
model_poly_2_features.fit(X_train_poly_2_features, y_train)
```

---

## 7.4 Predict

```python
y_pred_poly_2_features = model_poly_2_features.predict(
    X_test_poly_2_features
)
```

---

## 7.5 Evaluate

```python
evaluate_regression(y_test, y_pred_poly_2_features)
```

---

## 7.6 Inspect the learned coefficients

```python
print("Intercept:", model_poly_2_features.intercept_)

for name, coefficient in zip(
    poly_feature_names,
    model_poly_2_features.coef_,
):
    print(f"{name:30s} {coefficient: .6f}")
```

### What are you looking at?

You may see terms corresponding to:

- `TRIP_MILES`
- `TRIP_MINUTES`
- `TRIP_MILES^2`
- `TRIP_MILES TRIP_MINUTES`
- `TRIP_MINUTES^2`

This is one of the best ways to understand what polynomial expansion actually does: it turns one compact model specification into a larger set of explicit numerical features.

---

## Check your understanding

**Question 1:** How many polynomial features do you get from two original features at degree 2 when `include_bias=False`?

**Question 2:** What is the purpose of the interaction term `TRIP_MILES * TRIP_MINUTES`?

**Question 3:** Why does increasing degree increase the number of columns?

**Question 4:** Why should we inspect `get_feature_names_out()` when learning polynomial regression?

### Prompt for an LLM

```text
I am using:

PolynomialFeatures(degree=2, include_bias=False)

with these original features:

TRIP_MILES
TRIP_MINUTES

Explain exactly what transformed features are generated.

Please show:
1. The names of the transformed features.
2. Why the interaction term exists.
3. The resulting regression equation in symbolic form.
4. How the number of features changes if degree becomes 3.

Teach me step by step and ask me one short question at the end to test whether I understand.
```

<details>
<summary><strong>Solution: expand if you are stuck</strong></summary>

For two original features and degree 2, without the bias column, you get:

```text
TRIP_MILES
TRIP_MINUTES
TRIP_MILES²
TRIP_MILES * TRIP_MINUTES
TRIP_MINUTES²
```

So there are **5** transformed features.

The interaction term allows the model to represent a situation where the relationship associated with distance changes depending on duration, or vice versa.

The resulting model can be written conceptually as:

$$
\hat{y} = b_0 + b_1x_1 + b_2x_2 + b_3x_1^2 + b_4x_1x_2 + b_5x_2^2
$$

where:

- $x_1 = TRIP\_MILES$
- $x_2 = TRIP\_MINUTES$

</details>

---

# Part 8: Compare the three approaches

Now create a proper comparison table.

```python
comparison = pd.DataFrame({
    "Model": [
        "Linear — 1 feature",
        "Linear — 2 features",
        "Polynomial degree 2 — 1 feature",
        "Polynomial degree 2 — 2 features",
    ],
    "MAE": [
        mean_absolute_error(y_test, y_pred_lr_1),
        mean_absolute_error(y_test, y_pred_lr_2),
        mean_absolute_error(y_test, y_pred_poly_2),
        mean_absolute_error(y_test, y_pred_poly_2_features),
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y_test, y_pred_lr_1)),
        np.sqrt(mean_squared_error(y_test, y_pred_lr_2)),
        np.sqrt(mean_squared_error(y_test, y_pred_poly_2)),
        np.sqrt(mean_squared_error(y_test, y_pred_poly_2_features)),
    ],
    "R2": [
        r2_score(y_test, y_pred_lr_1),
        r2_score(y_test, y_pred_lr_2),
        r2_score(y_test, y_pred_poly_2),
        r2_score(y_test, y_pred_poly_2_features),
    ],
}).sort_values("RMSE")

comparison
```

### Important interpretation rule

Do not say:

> "The model with the highest R² is automatically the best model."

A better statement is:

> "On the held-out test set and according to the selected evaluation metrics, this model performed best among the models tested."

Also consider model complexity.

A more complex model may improve a metric slightly while becoming harder to interpret and more vulnerable to overfitting.

---

## Final reasoning questions for the manual models

**Question 1:** Which model has the lowest RMSE?

**Question 2:** Which model has the lowest MAE?

**Question 3:** Which model has the highest R²?

**Question 4:** Do all three metrics necessarily rank models in exactly the same order?

**Question 5:** Does a more complex model automatically deserve to be chosen?

**Question 6:** What evidence would you want before declaring one model "best" in a real project?

### Prompt for an LLM

```text
I compared four regression models on the same Chicago taxi test set:

1. Linear regression with TRIP_MILES
2. Multiple linear regression with TRIP_MILES + TRIP_MINUTES
3. Polynomial degree 2 with TRIP_MILES
4. Polynomial degree 2 with TRIP_MILES + TRIP_MINUTES

My results table contains MAE, RMSE, and R².

Teach me how to compare them properly.

Please focus on:
- what each metric means;
- why lower MAE/RMSE is better;
- why higher R² is usually better;
- why a more complex model is not automatically the best choice;
- why evaluating on unseen data matters.

Do not invent performance numbers that I have not provided.
```

<details>
<summary><strong>Solution: expand if you are stuck</strong></summary>

For the metrics:

- lower **MAE** is better;
- lower **RMSE** is better;
- higher **R²** is generally better.

But the model choice should also consider:

- test performance;
- complexity;
- stability;
- interpretability;
- computational cost;
- whether the model captures meaningful structure rather than noise.

A model should not be called best simply because it is more complicated.

</details>

---

# Part 9: Hyperparameter experiment: polynomial degree

For `LinearRegression`, there is no training loop with epochs or batch size. A more useful experiment is to change the polynomial degree and observe how model complexity affects training and test performance.

For polynomial regression, a much more meaningful experiment is the **degree**.

We will compare degrees 1 through 4.

> Degree 1 is effectively ordinary linear regression.

---

## 9.1 Experiment code

```python
degree_results = []

X = training_df[["TRIP_MILES"]]
y = training_df["FARE"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

for degree in [1, 2, 3, 4]:
    poly = PolynomialFeatures(
        degree=degree,
        include_bias=False,
    )

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    train_pred = model.predict(X_train_poly)
    test_pred = model.predict(X_test_poly)

    degree_results.append({
        "Degree": degree,
        "Train_RMSE": np.sqrt(mean_squared_error(y_train, train_pred)),
        "Test_RMSE": np.sqrt(mean_squared_error(y_test, test_pred)),
        "Train_R2": r2_score(y_train, train_pred),
        "Test_R2": r2_score(y_test, test_pred),
    })

degree_results_df = pd.DataFrame(degree_results)
degree_results_df
```

### Why evaluate both training and test performance here?

Because the purpose of this experiment is to observe the tension between:

```text
fit the training data well
```

and:

```text
generalize to unseen data
```

A model that becomes increasingly flexible can continue improving on the training set while eventually getting worse on unseen data.

That is a classic sign of overfitting.

---

## Plot the degree experiment

```python
plt.figure(figsize=(8, 5))
plt.plot(
    degree_results_df["Degree"],
    degree_results_df["Train_RMSE"],
    marker="o",
    label="Train RMSE",
)
plt.plot(
    degree_results_df["Degree"],
    degree_results_df["Test_RMSE"],
    marker="o",
    label="Test RMSE",
)
plt.xlabel("Polynomial degree")
plt.ylabel("RMSE")
plt.title("Training vs test error by polynomial degree")
plt.legend()
plt.show()
```

---

## Check your understanding

**Question 1:** What happens to the number of features as degree increases?

**Question 2:** Why might training RMSE continue to decrease?

**Question 3:** What pattern would suggest overfitting?

**Question 4:** Why is degree a hyperparameter in this experiment?

### Prompt for an LLM

```text
I ran polynomial regression for degrees 1, 2, 3, and 4 and recorded train and test RMSE.

Teach me how to detect overfitting from the results.

Explain these possible situations:
A. Train RMSE decreases and test RMSE decreases.
B. Train RMSE decreases but test RMSE starts increasing.
C. Both are high.
D. Test RMSE is much lower than expected after changing the split.

Please teach the reasoning rather than just naming the concepts.
```

<details>
<summary><strong>Solution: expand if you are stuck</strong></summary>

### A. Both improve

This suggests the added flexibility is helping capture useful structure that generalizes to unseen examples.

### B. Training improves, test gets worse

This is a classic overfitting pattern. The model is becoming more specialized to the training data while losing generalization ability.

### C. Both are high

This suggests underfitting or that the available features are simply insufficient to explain the target well.

### D. Test performance changes unexpectedly after changing the split

A single train/test split contains randomness. For stronger conclusions, real projects often use cross-validation and repeated experiments rather than trusting one split alone.

</details>

---

# Part 10: Why we deliberately did not scale features here

You may have heard that machine-learning features should always be standardized.

That is not true.

For ordinary scikit-learn `LinearRegression`, scaling is generally **not required** for the estimator to fit successfully.

For example:

```text
TRIP_MILES    -> around a few miles
TRIP_MINUTES -> tens of minutes
```

The model can still learn coefficients for those features without us standardizing them first.

This does **not** mean scaling is never important.

Scaling becomes particularly relevant when:

- model optimization depends on feature magnitudes;
- regularization penalizes coefficients;
- distance-based methods are used;
- algorithms are sensitive to numerical scale.

We are postponing these topics because the immediate learning goal is understanding the mechanics of regression, not adding preprocessing complexity everywhere.

### Q&A

**Question:** Why not add `StandardScaler` automatically to every model?

<details>
<summary><strong>Solution</strong></summary>

Because preprocessing should be connected to the needs of the model and the data.

For ordinary `LinearRegression`, scaling is not necessary simply to make the estimator work. Introducing unnecessary preprocessing can make the learning process harder to understand, especially when the goal is to study the regression algorithm itself.

That said, scaling can be useful in many other settings and should be considered when the chosen algorithm or data distribution makes it relevant.

</details>

---

# Part 11: Optional: repeat the three models with Pipeline

Now we finally introduce `Pipeline`.

The purpose is not to replace your understanding of the manual process.

The purpose is to **package steps that you already understand**.

The most important mental model is:

```text
Manual approach:

transform -> model

Pipeline approach:

Pipeline([
    (transformer, transformer),
    (model, model)
])
```

The pipeline does not create a fundamentally new machine-learning algorithm.

It organizes preprocessing and modeling into one estimator-like object.

---

# Part 12: Pipeline version of Model 1

Simple linear regression has no preprocessing requirement here, so the pipeline is mostly educational.

```python
pipeline_linear_1 = Pipeline([
    ("model", LinearRegression()),
])

X = training_df[["TRIP_MILES"]]
y = training_df["FARE"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

pipeline_linear_1.fit(X_train, y_train)
y_pred_pipeline_linear_1 = pipeline_linear_1.predict(X_test)

evaluate_regression(y_test, y_pred_pipeline_linear_1)
```

### Why is this pipeline almost pointless?

Because there is only one step.

We are using it to demonstrate that a pipeline can wrap an estimator.

---

## Q&A

**Question:** What does the pipeline add here?

<details>
<summary><strong>Solution</strong></summary>

Very little. It wraps `LinearRegression` as a one-step pipeline.

The benefit becomes clearer when we have multiple sequential transformations, as with polynomial regression.

</details>

---

# Part 13: Pipeline version of Model 2

Multiple linear regression also requires no mandatory transformation for this exercise.

```python
pipeline_linear_2 = Pipeline([
    ("model", LinearRegression()),
])

X = training_df[["TRIP_MILES", "TRIP_MINUTES"]]
y = training_df["FARE"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

pipeline_linear_2.fit(X_train, y_train)
y_pred_pipeline_linear_2 = pipeline_linear_2.predict(X_test)

evaluate_regression(y_test, y_pred_pipeline_linear_2)
```

## Q&A

**Question:** Why does this look almost identical to the manual model?

<details>
<summary><strong>Solution</strong></summary>

Because there is no preprocessing step to package. The pipeline contains only the estimator.

The pipeline becomes much more useful when we need a sequence such as:

```text
PolynomialFeatures -> LinearRegression
```

</details>

---

# Part 14: Pipeline version of Model 3: Polynomial Regression

This is where the pipeline becomes genuinely useful.

## Manual version

We previously had to write:

```python
poly = PolynomialFeatures(degree=2, include_bias=False)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)
```

The pipeline packages exactly this sequence.

## Pipeline version

```python
pipeline_poly_2 = Pipeline([
    (
        "polynomial_features",
        PolynomialFeatures(degree=2, include_bias=False),
    ),
    (
        "model",
        LinearRegression(),
    ),
])

X = training_df[["TRIP_MILES"]]
y = training_df["FARE"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

pipeline_poly_2.fit(X_train, y_train)
y_pred_pipeline_poly_2 = pipeline_poly_2.predict(X_test)

evaluate_regression(y_test, y_pred_pipeline_poly_2)
```

### What happened inside the pipeline?

When you run:

```python
pipeline_poly_2.fit(X_train, y_train)
```

the pipeline performs the equivalent sequence:

```text
X_train
   |
   v
PolynomialFeatures.fit_transform(X_train)
   |
   v
expanded training matrix
   |
   v
LinearRegression.fit(..., y_train)
```

When you run:

```python
pipeline_poly_2.predict(X_test)
```

the pipeline performs the equivalent sequence:

```text
X_test
   |
   v
PolynomialFeatures.transform(X_test)
   |
   v
expanded test matrix
   |
   v
LinearRegression.predict(...)
   |
   v
predictions
```

The important concept is that the test data does not bypass the transformation step.

---

## Q&A

**Question 1:** Why is `Pipeline` particularly useful for polynomial regression?

**Question 2:** Which operations are hidden inside `fit()` and `predict()` when using the pipeline?

**Question 3:** Does the pipeline change the underlying regression algorithm?

**Question 4:** Why can pipelines reduce accidental preprocessing mistakes?

### Prompt for an LLM

```text
I understand polynomial regression manually with:

poly.fit_transform(X_train)
poly.transform(X_test)
model.fit(...)
model.predict(...)

Now I want to understand Pipeline.

Explain this code:

pipeline = Pipeline([
    ("polynomial_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("model", LinearRegression()),
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

Do NOT explain it as magic.
Map each pipeline operation to the manual code I wrote before.
Also explain why transform is applied to the test data without fitting the transformer again.
```

<details>
<summary><strong>Solution: expand if you are stuck</strong></summary>

A pipeline is a sequence of estimators and transformers.

In this case:

```text
step 1: PolynomialFeatures
step 2: LinearRegression
```

`fit()` roughly corresponds to:

```python
X_train_poly = poly.fit_transform(X_train)
model.fit(X_train_poly, y_train)
```

`predict()` roughly corresponds to:

```python
X_test_poly = poly.transform(X_test)
model.predict(X_test_poly)
```

The pipeline does not replace the underlying model. It coordinates the steps in the correct order.

One of the biggest practical advantages is reducing the chance of accidentally forgetting to transform new data in exactly the same way as training data.

</details>

---

# Part 15: Pipeline version with two-feature polynomial regression

Now combine multiple features and polynomial expansion inside one pipeline.

```python
pipeline_poly_2_features = Pipeline([
    (
        "polynomial_features",
        PolynomialFeatures(degree=2, include_bias=False),
    ),
    (
        "model",
        LinearRegression(),
    ),
])

X = training_df[["TRIP_MILES", "TRIP_MINUTES"]]
y = training_df["FARE"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

pipeline_poly_2_features.fit(X_train, y_train)
y_pred_pipeline_poly_2_features = pipeline_poly_2_features.predict(X_test)

evaluate_regression(y_test, y_pred_pipeline_poly_2_features)
```

### Compare manual and pipeline results

```python
manual_poly_rmse = np.sqrt(
    mean_squared_error(y_test, y_pred_poly_2_features)
)

pipeline_poly_rmse = np.sqrt(
    mean_squared_error(y_test, y_pred_pipeline_poly_2_features)
)

print("Manual RMSE   :", manual_poly_rmse)
print("Pipeline RMSE :", pipeline_poly_rmse)
```

They should be effectively the same because the pipeline is implementing the same transformation and estimator sequence.

---

# Part 16: Final comparison: manual vs pipeline

Create one final comparison.

```python
final_comparison = pd.DataFrame({
    "Implementation": [
        "Manual",
        "Pipeline",
    ],
    "Model": [
        "Polynomial degree 2, 2 features",
        "Polynomial degree 2, 2 features",
    ],
    "MAE": [
        mean_absolute_error(y_test, y_pred_poly_2_features),
        mean_absolute_error(y_test, y_pred_pipeline_poly_2_features),
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y_test, y_pred_poly_2_features)),
        np.sqrt(mean_squared_error(y_test, y_pred_pipeline_poly_2_features)),
    ],
    "R2": [
        r2_score(y_test, y_pred_poly_2_features),
        r2_score(y_test, y_pred_pipeline_poly_2_features),
    ],
})

final_comparison
```

## Final questions

**Question 1:** Should the manual and pipeline implementations produce the same predictions?

**Question 2:** If they differ substantially, what kinds of coding mistakes would you check first?

**Question 3:** What is the main advantage of the pipeline version?

**Question 4:** What is the main advantage of learning the manual version first?

**Question 5:** In your own words, what is a pipeline?

### Prompt for an LLM

```text
I implemented polynomial regression in two ways:

1. Manually with PolynomialFeatures and LinearRegression.
2. Using Pipeline([PolynomialFeatures, LinearRegression]).

My manual and pipeline results are not identical.

Give me a debugging checklist.

Check these possibilities:
- same train/test split;
- same polynomial degree;
- same include_bias setting;
- fit_transform used only on training data;
- transform used on test data;
- same feature columns and order;
- same target values;
- same evaluation metric.

Explain why each item matters.
```

<details>
<summary><strong>Solution: expand if you are stuck</strong></summary>

The manual and pipeline approaches should represent the same computation, so their predictions should normally be the same or numerically extremely close.

The first things to check are:

1. **Same split**: if the train/test rows differ, the comparison is not apples-to-apples.
2. **Same feature columns and order**: the model coefficients correspond to feature positions.
3. **Same degree**: degree 2 and degree 3 are different feature spaces.
4. **Same `include_bias`**: adding a constant column changes the transformed matrix.
5. **Same fitting procedure**: preprocessing must be learned from the training data.
6. **Same test transformation**: the test data must go through the same feature expansion.
7. **Same target vector**: comparing different `y` values is meaningless.
8. **Same metric**: MAE and RMSE are not interchangeable.

The biggest conceptual advantage of the pipeline is not that it makes a better model automatically. It makes the sequence of transformations and the estimator easier to manage consistently.

The biggest educational advantage of the manual version is that you can see every transformation and model-fitting step.

</details>

---

# Part 17: Suggested challenge questions

These are deliberately open-ended. Try them before asking an LLM.

## Challenge 1

Why might `TRIP_MILES` alone work reasonably well as a predictor of fare?

<details>
<summary><strong>Solution</strong></summary>

Trip distance is directly related to how much taxi service is provided and therefore contains substantial information about the target fare. The correlation analysis also suggests that `TRIP_MILES` is strongly associated with `FARE` in this dataset.

</details>

## Challenge 2

Why might adding trip duration improve the model?

<details>
<summary><strong>Solution</strong></summary>

Two trips with similar distance can take different amounts of time. Duration can capture information about traffic, route conditions, and other factors that pure distance does not capture. The two-feature model therefore has access to information not present in the distance-only model.

</details>

## Challenge 3

Why might adding degree-2 polynomial terms improve the model?

<details>
<summary><strong>Solution</strong></summary>

A straight line imposes a constant rate of change. Polynomial terms allow the slope to vary with the input, which can model curvature that a purely linear relationship cannot.

</details>

## Challenge 4

Why could polynomial degree 10 be a bad idea even if it achieves tiny training error?

<details>
<summary><strong>Solution</strong></summary>

Very high-degree models can become excessively flexible. They may fit noise and unusual examples in the training set rather than the general relationship. Test performance can therefore worsen even while training performance improves.

</details>

## Challenge 5

Why does `Pipeline` not automatically prevent overfitting?

<details>
<summary><strong>Solution</strong></summary>

A pipeline organizes transformations and modeling steps. It does not magically choose an appropriate model complexity. A pipeline containing degree-10 polynomial features can still overfit. The pipeline improves workflow consistency, not statistical judgment.

</details>

---

# Part 18: A compact mental model of the whole exercise

You should now be able to describe the workflow like this:

```text
1. Load data
      |
      v
2. Inspect data
      |
      v
3. Choose features + target
      |
      v
4. Split train/test
      |
      v
5. Transform training data if needed
      |
      v
6. Fit model
      |
      v
7. Transform test data in the same way
      |
      v
8. Predict
      |
      v
9. Evaluate
      |
      v
10. Compare models
```

And the model families become:

```text
Simple linear regression

FARE = b0 + b1 * TRIP_MILES
```

```text
Multiple linear regression

FARE = b0 + b1 * TRIP_MILES + b2 * TRIP_MINUTES
```

```text
Polynomial regression

FARE = b0 + b1*x + b2*x² + ...
```

And finally:

```text
Manual implementation

transform -> model
```

becomes:

```text
Pipeline implementation

Pipeline([
    ("transform", transformer),
    ("model", estimator),
])
```

---

# Part 19: Final assessment

Try answering these without looking back.

### Conceptual questions

1. What is the difference between a feature and a target?
2. Why do we split into train and test sets?
3. What does `fit()` mean in scikit-learn?
4. What does `predict()` do?
5. What is the intercept?
6. What is a regression coefficient?
7. Why is multiple linear regression still linear?
8. Why is polynomial regression able to model curves?
9. Why do we call `fit_transform()` on training data and `transform()` on test data?
10. What is overfitting?
11. Why is RMSE more sensitive to large errors than MAE?
12. What does R² attempt to measure?
13. Why should model comparisons use the same test set?
14. Why is a pipeline useful?
15. Why did we learn the manual workflow before using a pipeline?

### Coding questions

16. Write the code to train a linear model from `TRIP_MILES`.
17. Write the code to train a two-feature linear model.
18. Write the code to create degree-2 polynomial features.
19. Write the code to evaluate predictions with MAE, RMSE, and R².
20. Write a pipeline for degree-2 polynomial regression.

### Final LLM prompt

```text
I have completed a scikit-learn taxi fare regression exercise covering:

- train/test splitting
- simple linear regression
- multiple linear regression
- polynomial regression
- MAE, RMSE, R²
- residuals
- polynomial degree and overfitting
- Pipeline

Act as my instructor.

Do NOT simply summarize the topics.
Instead, give me a mini oral exam with 10 questions, one at a time.

Mix:
- conceptual questions;
- code-reading questions;
- debugging questions;
- interpretation questions.

Do not reveal the answer until I respond.
After each response, tell me whether my reasoning is correct, explain any mistake, and ask the next question.
At the end, give me a score out of 10 and identify the two topics I should review most.
```

<details>
<summary><strong>Suggested answer key: expand after attempting the assessment</strong></summary>

### 1. Feature vs target

A feature is an input variable used to make a prediction. The target is the value we are trying to predict.

### 2. Train/test split

Training data are used to learn model parameters. Test data provide an estimate of performance on unseen examples.

### 3. `fit()`

`fit()` estimates the model parameters from the training data.

### 4. `predict()`

`predict()` applies the learned model to input examples and returns predicted target values.

### 5. Intercept

The predicted value when all features are zero, according to the fitted model.

### 6. Coefficient

The model parameter associated with a feature. In a linear model it represents the predicted change in the target associated with a one-unit change in that feature while the other model inputs are held constant.

### 7. Multiple linear regression

It is still linear because the prediction is a linear combination of the model parameters and transformed feature values:

$$
\hat{y} = b_0 + b_1x_1 + b_2x_2
$$

### 8. Polynomial regression

Polynomial features introduce terms such as $x^2$, $x^3$, and interactions, allowing the fitted function to curve while the model remains linear in its coefficients.

### 9. `fit_transform()` vs `transform()`

The training set is used to fit the preprocessing transformation and transform the training data. The test data should then be transformed using the already-fitted transformation rather than fitting a new transformation on the test set.

### 10. Overfitting

Overfitting occurs when a model captures training-specific noise or details that do not generalize well to unseen data.

### 11. RMSE vs MAE

RMSE squares errors before averaging, so large mistakes receive disproportionately more influence. MAE uses absolute errors and therefore treats the magnitude more directly.

### 12. R²

R² measures how much variation in the target is explained by the model relative to a mean-prediction baseline.

### 13. Same test set

Using the same held-out examples makes model comparisons fairer by keeping the evaluation data fixed.

### 14. Pipeline

A pipeline packages sequential transformations and an estimator into one object so the workflow can be applied consistently.

### 15. Manual first

The manual workflow exposes every step, which makes the later pipeline abstraction understandable rather than mysterious.

</details>

---

# Recommended learning order

Do not rush to the pipeline section.

A strong study sequence is:

```text
Dataset exploration
      ↓
Simple LinearRegression
      ↓
Multiple LinearRegression
      ↓
PolynomialFeatures + LinearRegression
      ↓
Compare metrics
      ↓
Experiment with polynomial degree
      ↓
Understand overfitting
      ↓
Pipeline
```

The most important habit throughout the exercise is to ask:

> **What exactly is happening to my data at this line of code?**

For every transformation, model, and metric, you should be able to explain:

```text
What goes in?
What happens?
What comes out?
Why are we doing it?
```

That understanding will transfer to many other scikit-learn models later.

---

# Source note

This lab is adopted from Google's Chicago taxi regression activity. The original notebook contains the source references for the dataset and the introductory data exploration.

**Original Google Colab notebook:**  
https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/linear_regression_taxi.ipynb

**GitHub notebook source:**  
https://github.com/google/eng-edu/blob/main/ml/cc/exercises/linear_regression_taxi.ipynb

Reference links and citations used in the source activity can be found in the original notebook. This lab intentionally focuses on the scikit-learn workflow and keeps the original notebook as the reference source rather than discussing it throughout the lab.
