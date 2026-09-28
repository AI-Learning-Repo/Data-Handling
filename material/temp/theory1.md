# Theory 1: End-to-End Housing Review with Regression and Clustering

This theory file supports **Lab 1: End-to-End Housing System with Regression and Clustering**.

The purpose is to refresh the main ideas from weeks 2-5 in one connected story. The lab contains the code. This file explains the reasoning behind the code.

## Table of Contents

- [1. What Makes This End-to-End?](#1-what-makes-this-end-to-end)
- [2. From Business Objective to ML Task](#2-from-business-objective-to-ml-task)
- [3. Understanding the Housing Dataset](#3-understanding-the-housing-dataset)
- [4. Data Inspection and EDA](#4-data-inspection-and-eda)
- [5. Data Quality and Cleaning Decisions](#5-data-quality-and-cleaning-decisions)
- [6. Feature Engineering](#6-feature-engineering)
- [7. Regression System](#7-regression-system)
- [8. Regression Evaluation](#8-regression-evaluation)
- [9. Underfitting, Overfitting, and Generalization](#9-underfitting-overfitting-and-generalization)
- [10. Clustering System](#10-clustering-system)
- [11. Clustering Evaluation and Interpretation](#11-clustering-evaluation-and-interpretation)
- [12. Real-World Project Notes](#12-real-world-project-notes)
- [13. Concepts to Remember](#13-concepts-to-remember)
- [14. Review Questions](#14-review-questions)

## 1. What Makes This End-to-End?

An end-to-end machine-learning project does not start with `model.fit()`.

It usually moves through a chain of decisions:

```text
Business objective
    ↓
Data source and schema
    ↓
Inspection and EDA
    ↓
Data-quality decisions
    ↓
Feature engineering
    ↓
Train/test split for supervised learning
    ↓
Model training
    ↓
Evaluation
    ↓
Interpretation and limitations
```

The housing lab follows this idea, but with beginner-friendly tools.

The housing lab follows this idea using beginner-friendly tools and visible steps.

## 2. From Business Objective to ML Task

A business question is not automatically a machine-learning task.

For the housing example, the broad business question is:

> Can district information support housing-value analysis and district profiling?

This can become two ML questions.

| Question | ML task | Output |
|---|---|---|
| Can we estimate median house value? | Regression | A number |
| Which districts look similar? | Clustering | Group assignments |

The distinction matters because the target is used differently.

In regression, `median_house_value` is the value we want to predict. The model learns from examples where this value is known.

In clustering, we do not give the model a target. We choose features that define similarity, then interpret the groups afterward.

## 3. Understanding the Housing Dataset

In the housing dataset, one row represents one district.

Important columns include:

| Column | Meaning |
|---|---|
| `longitude`, `latitude` | Geographic location |
| `housing_median_age` | Median age of housing in the district |
| `total_rooms` | Total rooms in the district |
| `total_bedrooms` | Total bedrooms in the district |
| `population` | Population of the district |
| `households` | Number of households |
| `median_income` | Median income, already scaled in the dataset |
| `median_house_value` | Regression target |
| `ocean_proximity` | Categorical location descriptor |

This dataset also highlights a practical point: real datasets often contain preprocessed or capped values. For example, `median_income` is not raw income in ordinary currency units, and `median_house_value` has an upper cap.

That does not make the dataset unusable. It means analysts must understand how the data was created before interpreting model outputs.

## 4. Data Inspection and EDA

Initial inspection answers basic questions:

- How many rows and columns are there?
- What does one row represent?
- Which columns are numerical?
- Which columns are categorical?
- Which values are missing?
- Are there duplicates?
- What are the target values like?

Common pandas tools include:

```python
df.head()
df.shape
df.info()
df.describe()
df.isna().sum()
```

EDA then explores patterns.

In the housing lab, important EDA includes:

- the distribution of `median_house_value`;
- the relationship between `median_income` and `median_house_value`;
- the correlation matrix for numerical columns;
- boxplots by `ocean_proximity`;
- geographical plots using `longitude` and `latitude`.

EDA does not prove causal relationships. It helps us understand the data and decide what to investigate next.

## 5. Data Quality and Cleaning Decisions

Real-world data is rarely perfect.

The housing dataset includes missing values in `total_bedrooms`. A model such as ordinary linear regression cannot directly use missing values, so we need a strategy.

In Lab 1, the strategy is:

```text
Fit median imputation on the training data.
Apply the learned medians to both training and test data.
```

This prevents data leakage.

### Data Leakage

Data leakage happens when information from the evaluation data influences training.

Example:

```text
Wrong:
Calculate the median using the full dataset.
Then split into train/test.

Better:
Split into train/test.
Calculate the median from training data only.
Apply it to train and test.
```

Leakage can make evaluation look better than it really is.

## 6. Feature Engineering

Feature engineering means creating useful variables from existing ones.

The housing dataset contains totals:

- `total_rooms`
- `total_bedrooms`
- `population`
- `households`

Raw totals are partly measures of district size. Ratios can describe district characteristics more directly.

Lab 1 creates:

```text
rooms_per_household = total_rooms / households
bedrooms_per_room = total_bedrooms / total_rooms
population_per_household = population / households
```

These features connect to a real modeling idea: good features often come from understanding what the raw columns mean.

Feature engineering should be motivated by the problem, not done randomly.

## 7. Regression System

Regression predicts a numerical target.

In Lab 1:

```text
Features X → model → predicted median_house_value
```

The lab uses three levels:

1. Baseline model.
2. Simple linear regression.
3. Multiple linear regression.

### Baseline

The baseline predicts the training median house value for every test district.

This is intentionally simple. It gives us a reference point.

Without a baseline, it is hard to know whether a model has learned anything useful.

### Simple Linear Regression

Simple linear regression uses one feature:

```text
median_income → median_house_value
```

The model has the form:

```text
predicted value = intercept + coefficient × median_income
```

This helps students see what `fit()` and `predict()` mean before adding many features.

### Multiple Linear Regression

Multiple regression uses several features:

```text
longitude
latitude
housing_median_age
median_income
engineered ratio features
encoded ocean_proximity categories
...
```

The model is still linear in its coefficients, but it uses several inputs.

### Polynomial Regression

Polynomial regression is useful when a straight-line relationship may be too simple.

The idea is to create extra features from the original numerical features:

```text
x
x^2
x1 x x2
```

Then we fit `LinearRegression()` on the expanded feature table.

This can model curved relationships in the original features, but it also makes the model more flexible. More flexibility can help if the original model is underfitting, but too much flexibility can cause overfitting.

## 8. Regression Evaluation

Regression predictions are numerical, so classification accuracy is not appropriate.

Lab 1 uses:

| Metric | Meaning |
|---|---|
| MAE | Average absolute error |
| RMSE | Square root of average squared error |
| R2 | Performance relative to predicting the mean |

### MAE

MAE is easy to interpret because it is in the same units as the target.

If MAE is 50,000, predictions are off by about 50,000 house-value units on average in absolute terms.

### RMSE

RMSE gives larger errors more influence because errors are squared before averaging.

If RMSE is much larger than MAE, the model may be making some large mistakes.

### R2

R2 does not mean "percent correct."

Regression predictions are not simply right or wrong. R2 compares the model to a simple mean-prediction baseline.

### Residuals

A residual is:

```text
actual value - predicted value
```

Residual plots help reveal systematic errors that a single metric can hide.

## 9. Underfitting, Overfitting, and Generalization

A model should perform well on new observations, not only on the data used during fitting.

This is why Lab 1 compares training error and test error.

### Underfitting

Underfitting means the model is too simple or does not capture enough of the relationship in the data.

A common pattern is:

```text
Training RMSE: high
Test RMSE:     high
```

The model performs poorly even on the training data.

### Overfitting

Overfitting means the model fits the training data too closely, including noise or accidental patterns.

A common pattern is:

```text
Training RMSE: very low
Test RMSE:     much higher
```

The model looks strong on training data but does not generalize well.

### Why Test Error Matters

Training error tells us how well the model fits known data.

Test error gives a better estimate of how the model may perform on unseen data.

This is why we do not automatically choose the model with the lowest training RMSE.

## 10. Clustering System

Clustering changes the question.

Instead of predicting `median_house_value`, we ask:

> Which districts are similar based on selected characteristics?

Lab 1 uses:

- `median_income`
- `rooms_per_household`
- `bedrooms_per_room`
- `population_per_household`
- `latitude`
- `longitude`

It keeps `median_house_value` only for interpretation after clustering.

### Why Scaling Matters

K-Means uses distances. Features with large numerical scales can dominate distances.

Standardization transforms features so that they have approximately:

```text
mean = 0
standard deviation = 1
```

This does not prove the clusters are meaningful. It makes the distance calculation more balanced under the chosen representation.

### K-Means

K-Means:

1. chooses K cluster centers;
2. assigns observations to the nearest center;
3. updates centers using assigned observations;
4. repeats until assignments stabilize or the algorithm stops.

The output cluster numbers are arbitrary labels. Cluster `0` is not automatically better, lower, or earlier than cluster `1`.

### Agglomerative Clustering

Agglomerative clustering starts with each observation as its own cluster, then repeatedly merges groups.

Lab 1 uses it on a sample because full hierarchical clustering can be expensive and hard to visualize on large datasets.

## 11. Clustering Evaluation and Interpretation

Clustering has no supplied correct answer in this lab.

We therefore use several kinds of evidence:

- inertia;
- silhouette score;
- group sizes;
- cluster profiles;
- plots;
- interpretability.

### Inertia

Inertia measures total squared distance from observations to their assigned centroids.

Lower inertia is expected as K increases, so the lowest inertia is not enough to choose K.

### Silhouette

Silhouette compares how close an observation is to its own cluster versus other clusters.

Values near 1 suggest better separation. Values near 0 suggest overlap. Negative values suggest another cluster may be closer under the selected distance.

### Profiles

A cluster profile summarizes each group using original units.

For example:

- group size;
- mean median income;
- mean rooms per household;
- mean population per household;
- mean median house value after clustering.

Profiles support descriptions. They do not prove causes or natural categories.

## 12. Real-World Project Notes

Lab 1 uses a simplified version of real project thinking.

**Frame the problem before modeling.**  
Ask what the business objective is before choosing the model.

**Understand the current solution.**  
A baseline or current human process gives context for evaluating whether a model is useful.

**Get and inspect the data.**  
Real projects require knowing where data comes from, what each row means, and what each column represents.

**Explore before modeling.**  
Use geographic plots, target distributions, and correlations to understand the housing data before training a model.

**Understand capped and transformed values.**  
The dataset includes capped or scaled values. This affects interpretation.

**Use clustering as a housing-data idea.**  
Location and district similarity are natural housing-data questions. Lab 1 uses clustering directly to create district profiles.

**Present limitations.**  
A useful project summary explains what worked, what did not, and what assumptions remain.

## 13. Concepts to Remember

### Feature

An input variable used by the model.

### Target

The value a supervised model learns to predict.

### EDA

Exploratory data analysis: inspecting distributions, relationships, missing values, and unusual observations.

### Imputation

Filling missing values with a chosen estimate, such as the median.

### One-Hot Encoding

Representing categories using indicator columns.

### Data Leakage

When evaluation information influences training or preprocessing.

### Regression

Predicting a numerical value.

### Baseline

A simple reference prediction.

### Polynomial Regression

Regression that creates squared or interaction features before fitting a model.

### MAE

Average absolute error.

### RMSE

Error metric that gives larger mistakes more influence.

### R2

Performance relative to predicting the mean target.

### Underfitting

When a model is too simple and performs poorly on both training and test data.

### Overfitting

When a model fits the training data too closely and performs much worse on test data.

### Clustering

Grouping observations without using a target label.

### Scaling

Transforming features to comparable numerical scales.

### Inertia

K-Means compactness measure based on squared distances to centroids.

### Silhouette

Measure comparing within-cluster closeness to nearest competing cluster.

## 14. Review Questions

### Question 1

Suppose a team asks for "a housing AI system" but does not say whether it needs price estimates, district groups, or anomaly flags. Why is this not enough information to choose a model?

<details>
<summary>Suggested answer</summary>

The requested output determines the ML task.

Price estimates suggest regression. District groups suggest clustering. Anomaly flags suggest a different kind of analysis. Without clarifying the objective, the model may produce an output that does not support the real decision.

</details>

### Question 2

The correlation between `median_income` and `median_house_value` is strong. Why should this be treated as evidence for modeling, not as proof of causation or final feature selection?

<details>
<summary>Suggested answer</summary>

Pearson correlation measures linear association.

It does not prove that income causes house value. It also does not prove that other features are useless. A multiple-feature model may use location, ocean proximity, housing age, and engineered ratios together.

</details>

### Question 3

You see this pattern:

```text
Training RMSE = 20,000
Test RMSE     = 95,000
```

What does it suggest, and why is the test result more important for model choice?

<details>
<summary>Suggested answer</summary>

This suggests overfitting.

The model fits the training data very well but generalizes poorly to unseen data. Test RMSE is more important because it estimates performance on observations not used during fitting.

</details>

### Question 4

A student chooses the model with the lowest training RMSE. Why can this be a poor decision?

<details>
<summary>Suggested answer</summary>

A low training RMSE can mean the model has fit noise or accidental patterns in the training data.

Model choice should be based on test performance, comparison with a baseline, and interpretation of errors. A model with slightly higher training error may generalize better.

</details>

### Question 5

K-Means produces two clusters. One cluster has higher income and higher average house value after profiling. What can you conclude, and what should you avoid claiming?

<details>
<summary>Suggested answer</summary>

You can describe the cluster profile: the districts assigned to that cluster tend to have higher income and higher house value.

You should avoid claiming that the cluster causes high values or that the two clusters are guaranteed natural categories. The result depends on selected features, scaling, K, and algorithm.

</details>
