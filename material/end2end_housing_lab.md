# Lab 1: End-to-End Housing System with Regression and Clustering

<!--
This lab reviews weeks 2-5 through one realistic housing-data system. It follows the structure of the week 5 labs: read the explanation, run the code in order, answer the questions, and only then open the detailed solutions.

The matching Theory 1 file will explain the same ideas conceptually. This lab is the practical side.
-->

- [1. From a Business Question to Machine Learning Tasks](#1-from-a-business-question-to-machine-learning-tasks)
  - [1.1 Understand the business objective](#11-understand-the-business-objective)
  - [1.2 Separate the regression and clustering questions](#12-separate-the-regression-and-clustering-questions)
- [2. Load and Understand the Housing Dataset](#2-load-and-understand-the-housing-dataset)
  - [2.1 Import libraries](#21-import-libraries)
  - [2.2 Load the data](#22-load-the-data)
  - [2.3 Inspect rows, columns, and data types](#23-inspect-rows-columns-and-data-types)
  - [2.4 Identify features, target, and feature types](#24-identify-features-target-and-feature-types)
- [3. Initial EDA and Data Quality](#3-initial-eda-and-data-quality)
  - [3.1 Check missing values and duplicates](#31-check-missing-values-and-duplicates)
  - [3.2 Explore the target](#32-explore-the-target)
  - [3.3 Explore numerical relationships](#33-explore-numerical-relationships)
  - [3.4 Explore the categorical feature](#34-explore-the-categorical-feature)
  - [3.5 Investigate potential outliers](#35-investigate-potential-outliers)
- [4. Feature Engineering](#4-feature-engineering)
  - [4.1 Create ratio features](#41-create-ratio-features)
  - [4.2 Recheck the engineered features](#42-recheck-the-engineered-features)
- [5. Regression System: Predict Housing Value](#5-regression-system-predict-housing-value)
  - [5.1 Select features and target](#51-select-features-and-target)
  - [5.2 Split before learned preprocessing](#52-split-before-learned-preprocessing)
  - [5.3 Build a baseline](#53-build-a-baseline)
  - [5.4 Train simple linear regression](#54-train-simple-linear-regression)
  - [5.5 Prepare multiple-feature data](#55-prepare-multiple-feature-data)
  - [5.6 Train multiple linear regression](#56-train-multiple-linear-regression)
  - [5.7 Train polynomial regression](#57-train-polynomial-regression)
  - [5.8 Compare model performance and generalization](#58-compare-model-performance-and-generalization)
  - [5.9 Inspect predictions and residuals](#59-inspect-predictions-and-residuals)
  - [5.10 Interpret coefficients carefully](#510-interpret-coefficients-carefully)
- [6. Clustering System: Discover Similar District Profiles](#6-clustering-system-discover-similar-district-profiles)
  - [6.1 Change the question](#61-change-the-question)
  - [6.2 Select clustering features](#62-select-clustering-features)
  - [6.3 Standardize clustering features](#63-standardize-clustering-features)
  - [6.4 Compare candidate K values](#64-compare-candidate-k-values)
  - [6.5 Profile the selected K-Means clusters](#65-profile-the-selected-k-means-clusters)
  - [6.6 Visualize clusters geographically](#66-visualize-clusters-geographically)
  - [6.7 Compare with agglomerative clustering on a sample](#67-compare-with-agglomerative-clustering-on-a-sample)
  - [6.8 Read a dendrogram sample](#68-read-a-dendrogram-sample)
- [7. Connect the Lab to Real Project Practice](#7-connect-the-lab-to-real-project-practice)
- [8. Final Concept Review and Exam-Style Questions](#8-final-concept-review-and-exam-style-questions)

## Learning Objectives

By the end of this lab, you should be able to:

- translate a business objective into regression and clustering questions;
- inspect a real tabular dataset and identify features, target, and feature types;
- perform focused EDA before modeling;
- identify missing values, duplicates, unusual values, and potential outliers;
- create simple engineered features from domain reasoning;
- explain and avoid data leakage;
- create a train/test split for supervised learning;
- build and interpret a baseline regression model;
- train simple and multiple linear regression models;
- create polynomial features for a regression model;
- evaluate regression with MAE, RMSE, and R2;
- distinguish underfitting from overfitting using training and test error;
- interpret actual-vs-predicted and residual plots;
- distinguish regression from clustering;
- standardize features for distance-based clustering;
- use K-Means, inertia, silhouette scores, and cluster profiles;
- compare K-Means with agglomerative clustering at an introductory level;
- explain what the system can and cannot prove.

## 1. From a Business Question to Machine Learning Tasks

### 1.1 Understand the business objective

Imagine you have joined a housing analytics team. The team studies districts using census-style data. A district is a geographical area with measurements such as population, households, median income, and median house value.

The business team asks:

> Can we use district information to support housing-value analysis and district profiling?

This is not yet a machine-learning task. A real project starts by clarifying how the result will be used.

Possible uses include:

- estimating house values when direct estimates are unavailable;
- identifying districts with similar profiles;
- supporting human analysts who compare areas;
- summarizing patterns for decision makers.

In practice, the machine-learning model is not the whole business solution. It is one component in a larger decision process.

**Question 1: Business objective versus model objective**

Why is "build a model" not enough as a project objective?

<details>
<summary>Solution</summary>

"Build a model" does not explain what the model is for. We need to know how the output will be used, what kind of mistake matters, what data will be available, and what evidence would make the model useful.

For example, predicting exact house value is a regression task. But if the business only needs categories such as low, medium, and high value, the task might become classification instead.

</details>

### 1.2 Separate the regression and clustering questions

This lab has two connected but different systems.

| System | Question | Learning type | Target used during fitting? |
|---|---|---|---|
| Regression | Can we predict `median_house_value`? | Supervised learning | Yes |
| Clustering | Which districts look similar? | Unsupervised learning | No |

The same dataset can support both questions, but the meaning of the model changes.

For regression, `median_house_value` is the target. The model learns from examples where the answer is known.

For clustering, we deliberately do **not** use `median_house_value` as a fitting feature. We use selected district characteristics to discover groups, then we may inspect house value afterward as part of interpretation.

**Question 2: Regression or clustering?**

A housing analyst asks: "Given a district's income, population, location, and housing age, estimate its median house value." Is this regression or clustering?

<details>
<summary>Solution</summary>

This is regression because the goal is to predict a numerical target: `median_house_value`.

</details>

**Question 3: Regression or clustering?**

A housing analyst asks: "Without using house values, group districts that have similar income, household, population, and location characteristics." Is this regression or clustering?

<details>
<summary>Solution</summary>

This is clustering because the goal is to discover groups from features without using a known target during fitting.

</details>

## 2. Load and Understand the Housing Dataset

### 2.1 Import libraries

Run this setup cell first.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

sns.set_theme(style="whitegrid")
```

**Code explanation**

- `pandas` stores the dataset as a table.
- `numpy` helps with numerical arrays.
- `matplotlib` and `seaborn` create plots for EDA and interpretation.
- `train_test_split` separates data used for learning from data used for evaluation.
- pandas `median()` and `fillna()` handle missing values using values calculated from training data.
- `LinearRegression` fits a regression model.
- `PolynomialFeatures` creates squared and interaction terms for polynomial regression.
- MAE, RMSE, and R2 evaluate numerical predictions.
- `StandardScaler` standardizes features for distance-based clustering.
- `KMeans` and `AgglomerativeClustering` create clusters.
- `linkage` and `dendrogram` help visualize a hierarchy on a sample.

### 2.2 Load the data

You will run this lab in Google Colab. There are two simple ways to make the CSV available.

**Option A: download the CSV with `wget`**

```python
!wget -O housing.csv https://raw.githubusercontent.com/AI-Learning-Repo/Data-Handling/refs/heads/week6-7/datasets/housing.csv
housing = pd.read_csv("housing.csv")
housing.head()
```

**Option B: upload the CSV manually**

Upload `housing.csv` to the Colab session, then run:

```python
# housing = pd.read_csv("/content/housing.csv")
# housing.head()
```

The cell below loads `housing.csv` if it is in the current notebook folder. If you are running from the course repository, it also checks `datasets/housing.csv` and `../datasets/housing.csv`.

```python
from pathlib import Path

if Path("housing.csv").exists():
    housing = pd.read_csv("housing.csv")
elif Path("datasets/housing.csv").exists():
    housing = pd.read_csv("datasets/housing.csv")
elif Path("../datasets/housing.csv").exists():
    housing = pd.read_csv("../datasets/housing.csv")
else:
    raise FileNotFoundError(
        "Could not find housing.csv. Upload it to the notebook session "
        "or download it with the wget cell above."
    )

housing.head()
```

### 2.3 Inspect rows, columns, and data types

Before modeling, inspect the structure.

```python
print("Rows and columns:", housing.shape)
```

```python
housing.info()
```

```python
housing.describe()
```

```python
housing.head()
```

**Code explanation**

- `shape` gives the number of rows and columns.
- `info()` shows column names, non-missing counts, and data types.
- `describe()` summarizes numerical columns.
- `head()` shows example rows.

These are not modeling steps. They help us understand what the dataset contains.

### 2.4 Identify features, target, and feature types

In this dataset:

- one row represents one district;
- `median_house_value` is the regression target;
- most columns are numerical;
- `ocean_proximity` is categorical.

```python
target = "median_house_value"

numeric_columns = housing.select_dtypes(include="number").columns.tolist()
categorical_columns = housing.select_dtypes(exclude="number").columns.tolist()

print("Numerical columns:")
print(numeric_columns)
print()
print("Categorical columns:")
print(categorical_columns)
```

**Question 4: Features and target**

Why should `median_house_value` not be included as an input feature when training a model to predict `median_house_value`?

<details>
<summary>Solution</summary>

That would give the model the answer as part of the input. The model would no longer be learning to predict house value from other district information.

This is a form of target leakage. It would make evaluation unrealistically good and useless for real prediction.

</details>

## 3. Initial EDA and Data Quality

### 3.1 Check missing values and duplicates

Real-world data is often incomplete or messy. Check missingness and duplicate rows.

```python
missing_counts = housing.isna().sum().sort_values(ascending=False)
missing_counts
```

```python
duplicate_count = housing.duplicated().sum()
print("Duplicate rows:", duplicate_count)
```

**Code explanation**

- `isna()` checks each cell for missingness.
- `sum()` counts missing values per column.
- `duplicated()` identifies repeated rows.

**Question 5: Missing values**

If `total_bedrooms` has missing values, why should we not simply ignore the issue?

<details>
<summary>Solution</summary>

Many machine-learning algorithms cannot work directly with missing values. Missingness can also tell us something about data quality.

We need to decide how to handle missing values. In this lab, we fill missing numerical values with medians calculated from the training data only, to avoid leakage.

</details>

### 3.2 Explore the target

The target is the value we want to predict.

```python
plt.figure(figsize=(8, 5))
housing["median_house_value"].hist(bins=50)
plt.xlabel("Median house value")
plt.ylabel("Number of districts")
plt.title("Distribution of the regression target")
plt.show()
```

```python
housing["median_house_value"].describe()
```

**Question 6: Target distribution**

What should you notice about the maximum value of `median_house_value`? Why might that matter?

<details>
<summary>Solution</summary>

The maximum value is often visible as a strong upper edge in this dataset. This suggests that the target may be capped at an upper limit.

This matters because a model may have difficulty predicting values beyond that cap, and the cap can affect error patterns. We should mention this limitation when interpreting results.

</details>

### 3.3 Explore numerical relationships

Start with a relationship that is likely to matter: median income and house value.

```python
sample_for_plot = housing.sample(3000, random_state=42)

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=sample_for_plot,
    x="median_income",
    y="median_house_value",
    alpha=0.4,
)
plt.title("Median income and median house value")
plt.show()
```

Now inspect correlations among numerical variables.

```python
correlations = housing[numeric_columns].corr(numeric_only=True)
correlations[target].sort_values(ascending=False)
```

```python
plt.figure(figsize=(10, 7))
sns.heatmap(correlations, cmap="coolwarm", center=0)
plt.title("Correlation matrix for numerical columns")
plt.show()
```

**Code explanation**

- A scatter plot helps us see the shape of a relationship.
- `corr()` calculates Pearson correlation between numerical columns.
- A heatmap helps compare many pairwise correlations.

**Question 7: Correlation and causation**

If `median_income` is strongly correlated with `median_house_value`, can we conclude that changing income alone causes house value to change?

<details>
<summary>Solution</summary>

No. Correlation describes association, not causation.

Higher-income districts may also differ in location, housing supply, ocean proximity, age of buildings, population patterns, and many other factors. A predictive model can use correlations without proving causal relationships.

</details>

### 3.4 Explore the categorical feature

The column `ocean_proximity` is categorical.

```python
housing["ocean_proximity"].value_counts()
```

```python
plt.figure(figsize=(9, 5))
sns.boxplot(data=housing, x="ocean_proximity", y="median_house_value")
plt.xticks(rotation=30)
plt.title("House value by ocean proximity")
plt.show()
```

**Question 8: Categorical variables**

Why should we not encode categories such as `NEAR BAY = 1`, `<1H OCEAN = 2`, and `INLAND = 3` as ordinary numbers for a linear model?

<details>
<summary>Solution</summary>

Those numbers would imply an artificial order and distance between categories. For example, the model might treat category 3 as greater than category 1 in a numerical sense.

For nominal categories, one-hot encoding is more appropriate because it creates separate indicator columns without inventing a false order.

</details>

### 3.5 Investigate potential outliers

Outliers are not automatically errors. They are observations that deserve investigation.

```python
columns_to_boxplot = [
    "median_income",
    "total_rooms",
    "population",
    "households",
    "median_house_value",
]

for col in columns_to_boxplot:
    plt.figure(figsize=(7, 2.8))
    sns.boxplot(x=housing[col])
    plt.title(f"Boxplot of {col}")
    plt.show()
```

```python
housing[columns_to_boxplot].describe()
```

**Question 9: Outliers**

Should we automatically delete districts with very high `total_rooms` or `population`?

<details>
<summary>Solution</summary>

No. A district with many rooms or a large population may be unusual but valid.

The correct action depends on domain knowledge and the goal of the analysis. We should investigate unusual values before deciding to keep, transform, cap, or remove them.

</details>

## 4. Feature Engineering

### 4.1 Create ratio features

Raw totals can be hard to compare across districts. A district with many rooms may simply have many households.

Create ratios that describe districts more meaningfully.

```python
housing_fe = housing.copy()

housing_fe["rooms_per_household"] = (
    housing_fe["total_rooms"] / housing_fe["households"]
)
housing_fe["bedrooms_per_room"] = (
    housing_fe["total_bedrooms"] / housing_fe["total_rooms"]
)
housing_fe["population_per_household"] = (
    housing_fe["population"] / housing_fe["households"]
)

housing_fe[
    ["rooms_per_household", "bedrooms_per_room", "population_per_household"]
].head()
```

**Code explanation**

- `rooms_per_household` describes average rooms per household.
- `bedrooms_per_room` describes the share of rooms that are bedrooms.
- `population_per_household` describes average people per household.

These are examples of feature engineering: creating useful variables from existing ones.

### 4.2 Recheck the engineered features

After creating features, inspect them.

```python
engineered_features = [
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
]

housing_fe[engineered_features].describe()
```

```python
for col in engineered_features:
    plt.figure(figsize=(7, 3))
    sns.histplot(housing_fe[col], bins=50)
    plt.title(f"Distribution of {col}")
    plt.show()
```

**Question 10: Feature engineering**

Why might `rooms_per_household` be more useful than `total_rooms`?

<details>
<summary>Solution</summary>

`total_rooms` is strongly affected by district size. A larger district may naturally have more rooms.

`rooms_per_household` adjusts for the number of households, so it better describes a district characteristic rather than only its size.

</details>

## 5. Regression System: Predict Housing Value

### 5.1 Select features and target

For the regression task, define `X` and `y`.

```python
target = "median_house_value"

numeric_features = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
]

categorical_features = ["ocean_proximity"]

X = housing_fe[numeric_features + categorical_features]
y = housing_fe[target]

print("X shape:", X.shape)
print("y shape:", y.shape)
```

**Code explanation**

- `X` contains the input features.
- `y` contains the target we want to predict.
- `median_house_value` is not included in `X`.

### 5.2 Split before learned preprocessing

Split the data before calculating training replacement values or fitting models.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
```

**Question 11: Leakage**

Why should we split before calculating the median used to fill missing values?

<details>
<summary>Solution</summary>

If we calculate the median using the full dataset, the test set influences preprocessing. The test set is supposed to represent unseen data.

Fitting preprocessing on the full dataset causes data leakage and can make evaluation look better than it really is.

</details>

### 5.3 Build a baseline

A baseline is a simple reference model.

Here the baseline predicts the median house value from the training data for every test example.

```python
baseline_value = y_train.median()
baseline_predictions = np.full(shape=len(y_test), fill_value=baseline_value)

def regression_scores(model_name, y_true, y_pred):
    return {
        "Model": model_name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }

results = []
results.append(
    regression_scores("Training median baseline", y_test, baseline_predictions)
)

pd.DataFrame(results)
```

**Code explanation**

- `y_train.median()` uses only the training target values.
- `np.full()` creates one repeated prediction for every test row.
- The helper function calculates MAE, RMSE, and R2.

**Question 12: Why use a baseline?**

Why is it not enough to say that a model has an RMSE of 80,000?

<details>
<summary>Solution</summary>

A metric needs context. An RMSE of 80,000 may be good or poor depending on how difficult the task is and how simple alternatives perform.

A baseline gives us a simple reference. A useful model should improve meaningfully over the baseline.

</details>

### 5.4 Train simple linear regression

Start with one feature: `median_income`.

```python
simple_model = LinearRegression()
simple_model.fit(X_train[["median_income"]], y_train)

simple_predictions = simple_model.predict(X_test[["median_income"]])

results.append(
    regression_scores("Simple linear regression", y_test, simple_predictions)
)

pd.DataFrame(results)
```

```python
print("Intercept:", simple_model.intercept_)
print("Coefficient for median_income:", simple_model.coef_[0])
```

**Code explanation**

- `LinearRegression()` creates the model object.
- `fit()` learns the intercept and coefficient from training data.
- `predict()` generates predictions for test features.

**Question 13: Coefficient interpretation**

What does the coefficient for `median_income` mean in this simple model?

<details>
<summary>Solution</summary>

The coefficient describes how much the model's predicted house value changes when `median_income` increases by one unit, according to this fitted straight-line relationship.

It is an association inside the model, not proof of causation.

</details>

### 5.5 Prepare multiple-feature data

Now use several numerical features and the categorical feature.

First, fill missing numerical values. Calculate the medians from training data only.

```python
numeric_medians = X_train[numeric_features].median()

X_train_num = X_train[numeric_features].fillna(numeric_medians)
X_test_num = X_test[numeric_features].fillna(numeric_medians)

print("Missing values after filling training data:")
print(X_train_num.isna().sum().sum())
print("Missing values after filling test data:")
print(X_test_num.isna().sum().sum())
```

**Code explanation**

- `X_train[numeric_features].median()` calculates medians from the training data only.
- `fillna(numeric_medians)` fills missing values using those training medians.
- We do not calculate new medians from the test set.

Next, one-hot encode the categorical feature.

```python
X_train_cat = pd.get_dummies(X_train[categorical_features], drop_first=True)
X_test_cat = pd.get_dummies(X_test[categorical_features], drop_first=True)

X_test_cat = X_test_cat.reindex(columns=X_train_cat.columns, fill_value=0)

display(X_train_cat.head())
```

**Code explanation**

- `pd.get_dummies()` converts categories into indicator columns.
- `drop_first=True` removes one reference category.
- `reindex()` ensures the test encoded columns match the training encoded columns.

Combine numerical and categorical data.

```python
X_train_ready = pd.concat([X_train_num, X_train_cat], axis=1)
X_test_ready = pd.concat([X_test_num, X_test_cat], axis=1)

print("Prepared training shape:", X_train_ready.shape)
print("Prepared test shape:", X_test_ready.shape)
```

**Question 14: One-hot encoding**

Why must the prepared training and test feature tables have the same columns?

<details>
<summary>Solution</summary>

The model learns one coefficient for each training feature column. During prediction, the test data must contain the same feature columns in the same meaning.

If a test column is missing or a new column appears unexpectedly, the model input no longer matches what the model learned.

</details>

### 5.6 Train multiple linear regression

Fit a regression model using the prepared feature tables.

```python
multiple_model = LinearRegression()
multiple_model.fit(X_train_ready, y_train)

multiple_predictions = multiple_model.predict(X_test_ready)

results.append(
    regression_scores("Multiple linear regression", y_test, multiple_predictions)
)

results_df = pd.DataFrame(results)
results_df
```

### 5.7 Train polynomial regression

Linear regression fits a straight-line relationship in the features it receives.

Sometimes a relationship is curved. Polynomial regression handles this by creating extra features such as squared terms and interaction terms, then fitting a linear regression model on the expanded feature table.

To keep the example understandable, use only a small set of numerical features.

```python
poly_base_features = [
    "median_income",
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
]

X_train_poly_base = X_train_num[poly_base_features]
X_test_poly_base = X_test_num[poly_base_features]

poly = PolynomialFeatures(degree=2, include_bias=False)

X_train_poly = poly.fit_transform(X_train_poly_base)
X_test_poly = poly.transform(X_test_poly_base)

print("Original feature count:", X_train_poly_base.shape[1])
print("Polynomial feature count:", X_train_poly.shape[1])
```

```python
poly_feature_names = poly.get_feature_names_out(poly_base_features)
poly_feature_names
```

```python
polynomial_model = LinearRegression()
polynomial_model.fit(X_train_poly, y_train)

polynomial_predictions = polynomial_model.predict(X_test_poly)

results.append(
    regression_scores("Polynomial regression degree 2", y_test, polynomial_predictions)
)

results_df = pd.DataFrame(results)
results_df
```

**Code explanation**

- `PolynomialFeatures(degree=2)` creates original features, squared features, and pairwise interaction features.
- `fit_transform()` is used on the training data.
- `transform()` is used on the test data so the same transformation is applied.
- The model is still `LinearRegression`, but it receives an expanded feature table.

**Question 15: Polynomial regression**

Why can polynomial regression model curved relationships even though it uses `LinearRegression()`?

<details>
<summary>Solution</summary>

`LinearRegression()` is linear in the coefficients it learns.

Polynomial regression first creates transformed features such as:

```text
median_income^2
median_income × rooms_per_household
```

After those features exist, a linear model can combine them with coefficients. The relationship can be nonlinear in the original input features even though the fitted model is linear in its coefficients.

</details>

### 5.8 Compare model performance and generalization

```python
results_df.sort_values("RMSE")
```

Also compare training and test RMSE. This helps us discuss underfitting and overfitting.

```python
generalization_rows = []

model_predictions = [
    (
        "Simple linear regression",
        y_train,
        simple_model.predict(X_train[["median_income"]]),
        y_test,
        simple_predictions,
    ),
    (
        "Multiple linear regression",
        y_train,
        multiple_model.predict(X_train_ready),
        y_test,
        multiple_predictions,
    ),
    (
        "Polynomial regression degree 2",
        y_train,
        polynomial_model.predict(X_train_poly),
        y_test,
        polynomial_predictions,
    ),
]

for model_name, y_train_true, y_train_pred, y_test_true, y_test_pred in model_predictions:
    train_rmse = mean_squared_error(y_train_true, y_train_pred) ** 0.5
    test_rmse = mean_squared_error(y_test_true, y_test_pred) ** 0.5
    generalization_rows.append({
        "Model": model_name,
        "Training RMSE": train_rmse,
        "Test RMSE": test_rmse,
        "Gap": test_rmse - train_rmse,
    })

generalization_df = pd.DataFrame(generalization_rows)
generalization_df.sort_values("Test RMSE")
```

**Code explanation**

- Training RMSE measures error on data used to fit the model.
- Test RMSE measures error on unseen data.
- A model with high training error and high test error may be underfitting.
- A model with very low training error but much higher test error may be overfitting.
- The best practical model is usually chosen using test performance, not training performance alone.

**Question 16: Model comparison**

Which model performs best on the test set? Which metric supports your answer?

<details>
<summary>Solution</summary>

Use your output. In the local test run, the multiple-feature models performed better than the simple model and the baseline because they had lower MAE and RMSE and higher R2.

The important point is to compare models on the same test set using the same metrics.

</details>

**Question 17: Underfitting and overfitting**

Suppose a model has high training RMSE and high test RMSE. Is that more likely underfitting or overfitting?

<details>
<summary>Solution</summary>

That pattern is more likely underfitting.

The model is not performing well even on the data it was trained on, so it may be too simple, missing important features, or unable to represent the relationship well.

</details>

**Question 18: Training error versus test error**

Why should we not automatically choose the model with the lowest training RMSE?

<details>
<summary>Solution</summary>

A very flexible model can memorize patterns or noise in the training data. This can produce a low training RMSE but poor performance on new data.

The test set is more useful for estimating how the model may perform on unseen observations.

</details>

**Question 19: MAE and RMSE**

Why is RMSE often larger than MAE?

<details>
<summary>Solution</summary>

RMSE squares errors before averaging and then takes the square root. Squaring gives larger errors more influence.

MAE averages absolute errors more directly. If a model makes some very large mistakes, RMSE will increase more strongly than MAE.

</details>

**Question 20: R2**

Does `R2 = 0.65` mean that 65 percent of individual predictions are correct?

<details>
<summary>Solution</summary>

No. Regression predictions are numerical, not simply correct or incorrect.

R2 compares the model against a baseline that predicts the mean target value. It summarizes how much variation is explained relative to that baseline.

</details>

### 5.9 Inspect predictions and residuals

Metrics summarize performance, but plots help reveal patterns.

```python
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=multiple_predictions, alpha=0.3)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "--",
    color="black",
)
plt.xlabel("Actual median house value")
plt.ylabel("Predicted median house value")
plt.title("Actual vs predicted values")
plt.show()
```

```python
residuals = y_test - multiple_predictions

plt.figure(figsize=(8, 5))
sns.scatterplot(x=multiple_predictions, y=residuals, alpha=0.3)
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Predicted value")
plt.ylabel("Residual: actual - predicted")
plt.title("Residual plot")
plt.show()
```

```python
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=50)
plt.xlabel("Residual")
plt.title("Distribution of residuals")
plt.show()
```

**Code explanation**

- The actual-vs-predicted plot compares true values with model predictions.
- The diagonal line shows perfect prediction.
- Residuals are errors: `actual - predicted`.
- A residual plot helps us see whether errors show systematic patterns.

**Question 21: Residual interpretation**

What does a positive residual mean?

<details>
<summary>Solution</summary>

A positive residual means:

```text
actual value - predicted value > 0
```

So the actual value is higher than the model predicted. The model underpredicted that district.

</details>

### 5.10 Interpret coefficients carefully

Inspect the coefficients from the multiple regression model.

```python
coef_table = pd.DataFrame({
    "feature": X_train_ready.columns,
    "coefficient": multiple_model.coef_,
}).sort_values("coefficient", key=abs, ascending=False)

coef_table
```

**Question 22: Coefficients in multiple regression**

Why should coefficient interpretation be more careful in a multiple-feature model than in a simple one-feature model?

<details>
<summary>Solution</summary>

In multiple regression, a coefficient describes the model's expected prediction change for one feature while the other included features are held constant.

Features can be correlated with each other. Coefficients can also depend on units. Therefore, coefficients should be interpreted carefully and not treated as simple causal effects.

</details>

## 6. Clustering System: Discover Similar District Profiles

### 6.1 Change the question

Regression asked:

> Can we predict the known numerical target `median_house_value`?

Clustering asks:

> Which districts look similar based on selected characteristics?

For clustering, we do not supply `median_house_value` as a target. We also do not use it as a clustering feature. After fitting clusters, we can inspect house values to help describe the groups.

**Question 20: Why remove the target?**

Why should `median_house_value` not be included in the clustering features if our goal is to discover district profiles based on other characteristics?

<details>
<summary>Solution</summary>

If `median_house_value` is included, the clusters may be strongly driven by the value we previously treated as the prediction target.

Leaving it out lets us ask a different question: which districts are similar according to income, household, population, and location characteristics? We can inspect house value afterward to describe the groups.

</details>

### 6.2 Select clustering features

Choose features that describe district profiles.

```python
cluster_features = [
    "median_income",
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
    "latitude",
    "longitude",
]

cluster_df = housing_fe[
    cluster_features + ["median_house_value", "ocean_proximity"]
].dropna().copy()

print("Rows available for clustering:", len(cluster_df))
display(cluster_df[cluster_features].describe())
```

**Code explanation**

- `cluster_features` defines the coordinates used for similarity.
- `median_house_value` is kept only for later interpretation.
- `dropna()` removes rows missing selected clustering features for this introductory exercise.

### 6.3 Standardize clustering features

K-Means uses distances, so feature scale matters.

```python
cluster_scaler = StandardScaler()
X_cluster_scaled = cluster_scaler.fit_transform(cluster_df[cluster_features])

scaled_check = pd.DataFrame(X_cluster_scaled, columns=cluster_features)
display(scaled_check.mean().round(6))
display(scaled_check.std(ddof=0).round(6))
```

**Question 21: Scaling**

Why is scaling important for K-Means in this clustering task?

<details>
<summary>Solution</summary>

K-Means uses distances. If one feature has much larger numerical values than another, it can dominate the distance calculation.

Standardization puts features on comparable scales, so differences in income, ratios, latitude, and longitude can contribute more fairly under the chosen representation.

Scaling does not guarantee meaningful clusters; it is still a modeling choice.

</details>

### 6.4 Compare candidate K values

Fit K-Means for several values of K.

```python
cluster_rows = []
labels_by_k = {}

for k in range(2, 7):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_cluster_scaled)
    labels_by_k[k] = labels
    
    silhouette = silhouette_score(
        X_cluster_scaled,
        labels,
        sample_size=min(4000, len(cluster_df)),
        random_state=42,
    )
    
    cluster_rows.append({
        "K": k,
        "Inertia": model.inertia_,
        "Silhouette": silhouette,
    })

cluster_scores = pd.DataFrame(cluster_rows)
cluster_scores
```

The silhouette calculation uses a reproducible sample so the lab remains responsive in Colab.

```python
plt.figure(figsize=(7, 4))
plt.plot(cluster_scores["K"], cluster_scores["Inertia"], marker="o")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow plot for housing districts")
plt.show()
```

```python
plt.figure(figsize=(7, 4))
plt.plot(cluster_scores["K"], cluster_scores["Silhouette"], marker="o")
plt.xlabel("K")
plt.ylabel("Mean silhouette score")
plt.title("Silhouette score by K")
plt.show()
```

**Code explanation**

- Inertia measures total squared distance to assigned centroids.
- Lower inertia alone does not prove a better clustering because adding clusters usually reduces inertia.
- Silhouette compares how close observations are to their own cluster versus other clusters.
- A higher silhouette is useful evidence, not absolute proof of meaningful groups.

**Question 22: Choosing K**

Which K would you investigate further? Use both inertia and silhouette evidence.

<details>
<summary>Solution</summary>

Use your actual output. A defensible answer should mention both:

- the elbow plot: where inertia improvement begins to slow;
- silhouette: which K gives stronger separation under the selected representation.

If they disagree, that is not a failure. It means we need to inspect profiles, plots, group sizes, and the purpose of the clustering.

</details>

### 6.5 Profile the selected K-Means clusters

For this lab, choose the K with the highest silhouette score as a starting point for interpretation.

```python
chosen_k = int(
    cluster_scores.sort_values("Silhouette", ascending=False).iloc[0]["K"]
)

print("Chosen K for profiling:", chosen_k)

cluster_df["kmeans_cluster"] = labels_by_k[chosen_k]
```

Build cluster profiles in original units.

```python
cluster_profile = (
    cluster_df
    .groupby("kmeans_cluster")
    .agg(
        districts=("kmeans_cluster", "size"),
        median_income=("median_income", "mean"),
        rooms_per_household=("rooms_per_household", "mean"),
        bedrooms_per_room=("bedrooms_per_room", "mean"),
        population_per_household=("population_per_household", "mean"),
        median_house_value=("median_house_value", "mean"),
    )
    .reset_index()
)

cluster_profile
```

```python
pd.crosstab(
    cluster_df["kmeans_cluster"],
    cluster_df["ocean_proximity"],
    normalize="index",
).round(3)
```

**Question 23: Cluster profiles**

What can a cluster profile support? What can it not prove?

<details>
<summary>Solution</summary>

A profile can support descriptions such as:

- this cluster has higher average median income;
- this cluster has larger average households;
- this cluster contains more inland districts;
- this cluster has higher or lower average house value after clustering.

It cannot prove causes. It also cannot prove that the clusters are natural real-world categories. The groups depend on selected features, scaling, algorithm, and K.

</details>

### 6.6 Visualize clusters geographically

Plot a sample of district clusters by longitude and latitude.

```python
plot_sample = cluster_df.sample(
    min(5000, len(cluster_df)),
    random_state=42,
)

plt.figure(figsize=(8, 7))
sns.scatterplot(
    data=plot_sample,
    x="longitude",
    y="latitude",
    hue="kmeans_cluster",
    palette="tab10",
    alpha=0.6,
)
plt.title("K-Means district clusters shown geographically")
plt.legend(title="Cluster")
plt.show()
```

**Question 24: Map interpretation**

If clusters appear geographically separated, does that prove geography caused the clusters?

<details>
<summary>Solution</summary>

No. Latitude and longitude were included as clustering features, so geography directly influenced the distance calculation.

The plot can help us interpret the groups, but it does not prove a causal relationship.

</details>

### 6.7 Compare with agglomerative clustering on a sample

Agglomerative clustering can be expensive on large datasets, so use a reproducible sample.

```python
agg_sample = cluster_df.sample(2000, random_state=42).copy()
agg_sample_scaled = cluster_scaler.transform(agg_sample[cluster_features])

agg_model = AgglomerativeClustering(
    n_clusters=chosen_k,
    linkage="ward",
)
agg_sample["agg_cluster"] = agg_model.fit_predict(agg_sample_scaled)

print("Agglomerative cluster counts:")
print(agg_sample["agg_cluster"].value_counts().sort_index())
```

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

sns.scatterplot(
    data=agg_sample,
    x="longitude",
    y="latitude",
    hue="kmeans_cluster",
    palette="tab10",
    alpha=0.6,
    ax=axes[0],
    legend=False,
)
axes[0].set_title("K-Means labels on sample")

sns.scatterplot(
    data=agg_sample,
    x="longitude",
    y="latitude",
    hue="agg_cluster",
    palette="tab10",
    alpha=0.6,
    ax=axes[1],
    legend=False,
)
axes[1].set_title("Agglomerative labels on sample")

plt.tight_layout()
plt.show()
```

**Code explanation**

- K-Means repeatedly assigns observations to centroids and updates centroids.
- Agglomerative clustering starts with individual observations and repeatedly merges groups.
- The two methods can produce different groups even with the same features and number of clusters.

**Question 25: Different algorithms**

If K-Means and agglomerative clustering produce different memberships, does that automatically mean one is wrong?

<details>
<summary>Solution</summary>

No. They use different procedures.

K-Means uses centroids and reassignment. Agglomerative clustering builds a hierarchy through merges. Different algorithms can answer the same broad clustering question differently.

We compare results using evidence: profiles, plots, sizes, silhouette, interpretability, and the purpose of the analysis.

</details>

### 6.8 Read a dendrogram sample

A full dendrogram for thousands of districts would be unreadable. Use a small sample to practice reading the hierarchy.

```python
dendro_sample = cluster_df.sample(40, random_state=7).copy()
dendro_scaled = cluster_scaler.transform(dendro_sample[cluster_features])

linked = linkage(dendro_scaled, method="ward")

plt.figure(figsize=(12, 5))
dendrogram(linked, no_labels=True)
plt.title("Sample dendrogram for housing districts")
plt.xlabel("Sampled districts")
plt.ylabel("Ward linkage height")
plt.show()
```

**Question 26: Dendrogram**

What does a high merge in a dendrogram suggest?

<details>
<summary>Solution</summary>

A high merge suggests that the groups being joined were relatively far apart according to the linkage criterion.

For Ward linkage, the height relates to the increase in within-cluster variation caused by a merge. Large jumps can suggest possible cut levels, but they do not prove a uniquely correct number of clusters.

</details>

## 7. Connect the Lab to Real Project Practice

This lab used a simplified but realistic workflow:

```text
Business question
    ↓
Data loading and inspection
    ↓
EDA and data quality checks
    ↓
Feature engineering
    ↓
Regression model for prediction
    ↓
Evaluation on held-out test data
    ↓
Clustering model for district profiles
    ↓
Interpretation, limitations, and communication
```

In real work, engineers and analysts also ask:

- Where did the data come from?
- Is the data current and reliable?
- What does each row represent?
- Which columns are available before prediction time?
- Are there missing values or unusual values?
- Which model errors would be most costly?
- How should results be explained to non-technical users?
- What limitations must be stated clearly?

This lab does not implement advanced engineering infrastructure. The goal is to understand the reasoning and core workflow using concepts covered in class.

**Question 27: Real-world assumptions**

Why should we check assumptions before building a model?

<details>
<summary>Solution</summary>

If the assumptions are wrong, the model may solve the wrong problem.

For example, if the business needs categories but we build a numerical regression model, the output may not match the real need. If a feature would not be available for future districts, using it in training would make the model unrealistic.

</details>

## 8. Final Concept Review and Exam-Style Questions

Answer these before opening the solutions.

### Q1. Leakage Scenario

Someone fills missing `total_bedrooms` values using the median calculated from the full dataset, then performs the train/test split.

Explain the problem and give the corrected workflow.

<details>
<summary>Solution</summary>

This causes data leakage because information from the test set helps define the replacement value.

Correct workflow:

1. split into training and test sets;
2. calculate the median on the training data only;
3. fill missing training values using the training median;
4. fill missing test values using the same training median.

</details>

### Q2. Correlation Interpretation

Suppose `median_income` has the strongest Pearson correlation with `median_house_value`. Someone concludes: "Median income causes house value, so it is the only feature we need."

What is wrong with this conclusion?

<details>
<summary>Solution</summary>

Pearson correlation measures linear association, not causation.

A high correlation does not prove that one variable causes the other. It also does not prove that other features are useless. Location, ocean proximity, housing age, and engineered ratios may still add useful information in a multiple-feature model.

</details>

### Q3. Polynomial Regression and Generalization

You train three polynomial models:

| Model | Training RMSE | Test RMSE |
|---|---:|---:|
| Degree 1 | 82,000 | 84,000 |
| Degree 2 | 67,000 | 69,000 |
| Degree 8 | 21,000 | 110,000 |

Which model would you choose and which one is overfitting?

<details>
<summary>Solution</summary>

Degree 2 is the best choice because it has the lowest test RMSE.

Degree 8 is overfitting. Its training RMSE is very low, but its test RMSE is much higher. This means it fits the training data closely but generalizes poorly.

</details>

### Q4. Metric Interpretation

A model has:

```text
MAE  = 49,800
RMSE = 68,800
R2   = 0.66
```

Explain what each value says and why RMSE is larger than MAE.

<details>
<summary>Solution</summary>

MAE means the average absolute prediction error is about 49,800 house-value units.

RMSE is larger because it squares errors before averaging, so large mistakes receive more weight.

R2 compares the model with a mean-prediction baseline. An R2 of 0.66 means the model explains a substantial amount of variation relative to that baseline, but it does not mean 66 percent of predictions are exactly correct.

</details>

### Q5. Baseline Comparison

Two models have the following test results:

| Model | RMSE | R2 |
|---|---:|---:|
| Training median baseline | 121,600 | -0.07 |
| Multiple linear regression | 68,800 | 0.66 |

What conclusion is justified?

<details>
<summary>Solution</summary>

The multiple linear regression model clearly improves over the baseline because it has much lower RMSE and much higher R2 on the same test set.

The conclusion should still mention limitations: this does not prove causal relationships, and performance depends on whether future data are similar to the test data.

</details>

### Q6. One-Hot Encoding and Test Columns

After one-hot encoding `ocean_proximity`, the test set is missing one category column that appeared in training. Why is this a problem, and how did the lab handle it?

<details>
<summary>Solution</summary>

The model expects the same feature columns at prediction time as it saw during fitting.

The lab used `reindex(columns=X_train_cat.columns, fill_value=0)` so the test encoded table has the same columns as the training encoded table. Missing category columns are filled with 0.

</details>

### Q7. Residual Pattern

In an actual-vs-predicted plot, many expensive districts are predicted far below their actual value. What does this suggest?

<details>
<summary>Solution</summary>

It suggests the model may underpredict high-value districts.

Possible reasons include capped target values, missing important features, nonlinear relationships, or a model that is too simple for that part of the data.

</details>

### Q8. Target Use in Regression Versus Clustering

Why is `median_house_value` used as `y` in the regression system but left out of the features used to fit K-Means?

<details>
<summary>Solution</summary>

In regression, `median_house_value` is the value we want to predict, so it is the supervised target.

In clustering, the goal is to discover district profiles from other characteristics. If `median_house_value` is used as a clustering feature, the groups are partly defined by the target-like value instead of being an independent profile of district characteristics.

</details>

### Q9. K-Means Scaling Scenario

Suppose K-Means is fitted using `population`, `median_income`, and `bedrooms_per_room` without scaling. What problem may occur?

<details>
<summary>Solution</summary>

Features with larger numerical ranges can dominate Euclidean distance.

For example, `population` may influence the cluster assignments far more than `bedrooms_per_room` simply because it has larger values. Standardization makes the distance calculation more balanced across selected features.

</details>

### Q10. Choosing K

You compare K values:

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 87,400 | 0.407 |
| 3 | 72,500 | 0.380 |
| 4 | 55,700 | 0.378 |
| 5 | 48,200 | 0.368 |

Why might K=2 be chosen even though K=5 has lower inertia?

<details>
<summary>Solution</summary>

Inertia decreases as K increases, so lower inertia alone is not enough.

K=2 has the highest silhouette score in this table and is easier to profile. If the two clusters also produce interpretable district profiles, K=2 is a reasonable choice.

</details>

### Q11. Cluster Profile Interpretation

A cluster has higher average `median_income`, lower `bedrooms_per_room`, and higher average `median_house_value` after clustering. What can and cannot be concluded?

<details>
<summary>Solution</summary>

We can describe the cluster profile: districts in that cluster tend to have higher income, lower bedrooms-per-room, and higher house values.

We cannot conclude that the cluster label causes higher house values. Cluster profiles are descriptive and depend on selected features, scaling, and the clustering algorithm.

</details>

### Q12. Agglomerative Versus K-Means

K-Means and agglomerative clustering produce similar but not identical groups. Why is that expected?

<details>
<summary>Solution</summary>

They define clusters differently.

K-Means uses centroids and assigns observations to the nearest centroid. Agglomerative clustering starts with individual observations and merges groups according to a linkage rule. Different algorithms can produce different group boundaries.

</details>

### Q13. Dendrogram Interpretation

In a dendrogram, some merges happen at much larger heights than earlier merges. What does that suggest?

<details>
<summary>Solution</summary>

Large merge heights suggest that groups being joined at that stage are relatively far apart under the chosen distance and linkage method.

This can help decide where a reasonable cut might be, but it is still an interpretation aid, not proof of natural categories.

</details>

### Q14. Outlier Decision

A district has an unusually high `population_per_household`. Should it automatically be removed before modeling?

<details>
<summary>Solution</summary>

No. It should be investigated first.

It could be a data error, a rare but valid district, or a meaningful unusual case. Removing it without justification can hide important information and change model behavior.

</details>

### Q15. Final Project Communication

Write three limitations that should be communicated with the housing regression and clustering results.

<details>
<summary>Solution</summary>

Reasonable limitations include:

- regression performance is measured on a held-out test set but may change on future data;
- correlations and coefficients do not prove causation;
- cluster labels depend on selected features, scaling, K, and algorithm;
- capped or transformed values affect interpretation;
- clustering creates profiles, not guaranteed natural categories.

</details>
