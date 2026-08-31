# Lab 1: Titanic Binary Classification

**Predicting Survival on the Titanic with Supervised Classification**

---

## Lab Purpose

This lab introduces **binary classification** using the Titanic dataset.

Unlike the previous Titanic activity, this lab does not focus on extensive exploratory data analysis (EDA). The main data investigation was already completed in the earlier Titanic activity. Here, we move from understanding the data to building and evaluating a supervised classification model.

The main workflow is:

```text
Understand the classification task
              ↓
Briefly validate the dataset
              ↓
Define features and target
              ↓
Split the data
              ↓
Build a preprocessing pipeline
              ↓
Establish a baseline
              ↓
Train a real classifier
              ↓
Evaluate the classifier
              ↓
Use cross-validation
              ↓
Compare classifiers
              ↓
Tune the model
              ↓
Evaluate once on the held-out test set
```

The central question is:

> **Can we use information about a passenger to predict whether that passenger survived?**

---

# Learning Outcomes

By the end of this lab, you should be able to:

* explain why Titanic survival prediction is a **supervised-learning** problem
* explain why Titanic survival prediction is a **binary classification** problem
* distinguish between **features (`X`)** and the **target (`y`)**
* identify numerical and categorical features
* use a preprocessing pipeline for mixed numerical and categorical data
* explain why data should be split before model training
* establish and interpret a simple classification baseline
* train a logistic regression classifier
* evaluate a classifier using:

  * accuracy
  * confusion matrix
  * precision
  * recall
  * F1 score
* explain why cross-validation provides a stronger estimate of model performance
* compare two different classifiers
* understand how a classification threshold affects precision and recall
* perform simple hyperparameter tuning without using the test set
* evaluate a final model on previously unseen test data
* distinguish **binary classification** from **multiclass classification**

---

# Prior Connection

In the previous Titanic activity, you practiced:

* understanding the dataset
* inspecting columns and data types
* identifying missing values
* performing EDA
* cleaning and preparing data

We will **not repeat the full EDA process** in this lab.

Instead, we will use that previous work as a starting point and move to the next stage:

> **supervised classification**

This is an important transition in a machine-learning workflow.

Previously, the main question was:

> **What does the data look like?**

Now the question becomes:

> **Can we learn a model that predicts an outcome from the data?**

---

# Dataset

We will use the Titanic dataset available through OpenML.

* **Source:** OpenML
* **OpenML ID:** `40945`
* **Target:** `survived`

For this introductory classification task, we will use the following features:

```text
pclass
sex
age
sibsp
parch
fare
embarked
```

These features give us a mixture of numerical and categorical data while keeping the lab focused on classification.

---

# 1. Frame the Problem

## Supervised Learning

In supervised learning, a model learns from examples where the correct answer is already known.

For Titanic:

* each row represents a passenger
* the passenger information is the input
* `survived` is the known outcome
* the model learns a relationship between passenger information and survival

For example:

```text
Passenger information
        ↓
     Model
        ↓
Predicted survival
```

Because the historical dataset contains the correct survival outcome, we can train a supervised-learning model.

---

## Binary Classification

Classification means predicting a **category or class**.

Titanic survival has two possible outcomes:

```text
0 → Did not survive
1 → Survived
```

There are exactly **two classes**, so this is a:

> **binary classification problem**

Later, in Lab 2, we will use the Iris dataset, where there are **three possible classes**. That will be an example of:

> **multiclass classification**

### Task

Answer the following questions before writing code:

1. Why is this a supervised-learning problem?
2. Why is this specifically a binary classification problem?
3. What is the target variable?
4. What information will be used as features?

---

# 2. Environment Setup

Use the online notebook environment used in class.

Run:

```python
!pip install pandas scikit-learn matplotlib seaborn -q
```

Then import the required libraries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
```

---

# 3. Load the Dataset

Load the Titanic dataset directly from OpenML.

```python
titanic = fetch_openml(
    name="titanic",
    version=1,
    as_frame=True
)

df = titanic.frame.copy()

display(df.head())
print("Shape:", df.shape)
```

### Questions

1. How many rows and columns does the dataset contain?
2. Does the dataset contain more columns than we need for this lab?
3. Why might a smaller feature set be useful for an introductory classification task?

---

# 4. Briefly Validate the Dataset

We are not repeating the complete EDA from the previous activity.

However, before building a model, we should still perform a **brief validation check**.

Run:

```python
print("Columns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nTarget distribution:")
print(df["survived"].value_counts(dropna=False))

print("\nNumerical summary:")
print(df[["age", "fare"]].describe())

print("\nMissing values:")
print(df[["age", "embarked"]].isnull().sum())
```

### What should you notice?

Look for the following:

* the target column exists
* the selected features exist
* there are both numerical and categorical variables
* some variables contain missing values
* the target classes are not perfectly balanced

### Why does class balance matter?

Suppose a dataset contained:

```text
80% class 0
20% class 1
```

A model that always predicted class 0 would achieve:

```text
80% accuracy
```

without learning anything useful.

This is one reason why **accuracy alone is not always sufficient** for evaluating a classifier.

### Questions

1. Which selected features are numerical?
2. Which selected features are categorical?
3. Which selected variables contain missing values?
4. Why should we still check the data before modeling, even though EDA was completed previously?
5. Why can class imbalance make accuracy misleading?

---

# 5. Define the Feature Set

We will keep only the variables needed for this lab.

```python
selected_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
]

target_column = "survived"

df_model = df[selected_features + [target_column]].copy()

display(df_model.head())

print("\nMissing values:")
print(df_model.isnull().sum())
```

### Why use a smaller feature set?

For an introductory classification lab, we want students to focus on:

* classification
* preprocessing
* model training
* evaluation
* model comparison

rather than spending most of the lab dealing with complicated text or highly incomplete variables.

We therefore leave out variables such as:

* passenger name
* ticket
* cabin
* other identifiers or text-heavy fields

This does **not** mean those variables could never be useful. It simply keeps the current experiment focused and manageable.

### Questions

1. Why are passenger name, ticket, and cabin not necessary for this introductory lab?
2. Why can using fewer features make it easier to understand a machine-learning model?

---

# 6. Separate Features and Target

In supervised learning, we separate the input variables from the outcome we want to predict.

```python
X = df_model[selected_features]

y = df_model[target_column].astype(int)

print("X shape:", X.shape)
print("y shape:", y.shape)
```

Here:

* `X` contains the **features**
* `y` contains the **target**

Conceptually:

```text
X → information used to make the prediction

y → correct answer that the model is trying to learn
```

### Task

Write one sentence explaining the difference between `X` and `y`.

---

# 7. Split the Data

Before training a model, we need to separate the data into:

* a **training set**
* a **test set**

We will use 80% for training and 20% for testing.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True))

print("\nTest target distribution:")
print(y_test.value_counts(normalize=True))
```

---

## Why Split the Data?

The model should be evaluated on data that it did not use during training.

The basic idea is:

```text
Training data
     ↓
  Learn model
     ↓
Test data
     ↓
Evaluate model
```

If we train and evaluate on exactly the same data, the evaluation may make the model look better than it really is.

---

## Why Use Stratification?

The Titanic target contains two classes.

`stratify=y` attempts to preserve approximately the same class proportions in both datasets.

For example:

```text
Full dataset
Class 0 → 62%
Class 1 → 38%

Training set
Class 0 → approximately 62%
Class 1 → approximately 38%

Test set
Class 0 → approximately 62%
Class 1 → approximately 38%
```

This is useful when evaluating classification models.

### Questions

1. Why should we split the data before training?
2. Why is stratification useful?
3. Why should the test set remain untouched during model development?
4. What could happen if we repeatedly used the test set to choose the best model?

---

# 8. Build the Preprocessing Pipeline

Our dataset contains different types of variables.

### Numerical features

```text
age
sibsp
parch
fare
```

### Categorical features

```text
pclass
sex
embarked
```

We will preprocess these two groups differently.

```python
numeric_features = [
    "age",
    "sibsp",
    "parch",
    "fare",
]

categorical_features = [
    "pclass",
    "sex",
    "embarked",
]
```

## A note about `pclass`

Although `pclass` contains numbers such as 1, 2, and 3, those numbers represent **passenger-class categories**.

For this lab, we treat `pclass` as categorical rather than assuming that the difference between class 1 and class 2 has exactly the same meaning as the difference between class 2 and class 3.

---

## Numerical Preprocessing

For numerical variables, we will:

1. fill missing values using the median
2. standardize the variables

```python
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
```

### Why median imputation?

The median is relatively resistant to extreme values.

For example, if most fares are moderate but a few fares are extremely large, the median is less affected by those extreme observations than the mean.

---

## Categorical Preprocessing

For categorical variables, we will:

1. fill missing values using the most frequent category
2. convert categories into one-hot encoded columns

```python
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
```

### Why one-hot encoding?

A model should not interpret categories as if their numeric labels represent meaningful quantities.

For example, assigning:

```text
male   → 0
female → 1
```

can be problematic if the encoding is treated as a numerical scale.

One-hot encoding instead creates separate indicator variables.

---

## Combine the Preprocessing Steps

```python
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])
```

---

## Why Use a Pipeline?

A pipeline allows preprocessing and modeling to be treated as one workflow.

More importantly, preprocessing is **learned from the training data** and then applied to other data.

Conceptually:

```text
Training data
     ↓
Learn preprocessing
     ↓
Transform training data
     ↓
Train model
```

Then:

```text
New/test data
     ↓
Use the already-learned preprocessing
     ↓
Make predictions
```

We should **not** calculate imputation values, scaling parameters, or encoding information using the complete dataset before the train/test split.

Doing so can allow information from the test set to influence the model-development process. This is called **data leakage**.

### Questions

1. Why do we use median imputation for numerical features?
2. Why do we use most-frequent imputation for categorical features?
3. Why is one-hot encoding useful for categorical variables?
4. Why do we treat `pclass` as categorical in this lab?
5. What problem could occur if we fit preprocessing using the entire dataset before splitting?

---

# 9. Establish a Baseline

Before training a sophisticated model, we should establish a simple baseline.

A baseline gives us something to compare against.

We will use `DummyClassifier` with the `most_frequent` strategy.

```python
dummy_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DummyClassifier(strategy="most_frequent")),
])

dummy_pipeline.fit(X_train, y_train)

y_dummy_pred = dummy_pipeline.predict(X_test)

baseline_accuracy = accuracy_score(
    y_test,
    y_dummy_pred
)

print("Baseline accuracy:", baseline_accuracy)
```

## What is the baseline doing?

The classifier simply predicts the most common class for every passenger.

It does not learn a meaningful relationship between passenger characteristics and survival.

For example, if the majority class is:

```text
0 → Did not survive
```

the baseline predicts:

```text
0
0
0
0
0
...
```

for every passenger.

### Why is this useful?

Suppose:

```text
Baseline accuracy = 0.62
Model accuracy    = 0.80
```

The real model has clearly improved over the simple strategy.

But if:

```text
Baseline accuracy = 0.62
Model accuracy    = 0.63
```

then the model may not be providing much useful improvement.

### Questions

1. What is the baseline classifier doing?
2. Why can a baseline have surprisingly good accuracy?
3. Why should a real model be compared with a baseline?
4. Why is accuracy alone still not enough?

---

# 10. Train the First Real Classifier

We will start with **logistic regression**.

Despite its name, logistic regression is commonly used for classification.

For this lab, it predicts one of two classes:

```text
0 → Did not survive
1 → Survived
```

```python
logreg_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

logreg_pipeline.fit(X_train, y_train)

y_train_pred = logreg_pipeline.predict(X_train)
y_test_pred = logreg_pipeline.predict(X_test)
```

## Why Logistic Regression?

Logistic regression is a useful first classification model because:

* it is widely used
* it works well for many binary classification problems
* it produces class probabilities
* it provides a relatively simple model to interpret
* it gives us a useful reference point for comparing other classifiers

In this lab, we are not trying to find the most sophisticated possible Titanic model.

The goal is to understand the **classification workflow**.

---

# 11. Evaluate Training Performance

First, look at performance on the training data.

```python
training_accuracy = accuracy_score(
    y_train,
    y_train_pred
)

print("Training accuracy:", training_accuracy)
```

### Important

A high training accuracy does **not** automatically mean that the model is good.

The model has already seen the training examples.

What matters more is how well it performs on **unseen data**.

### Questions

1. If training accuracy is high, does that prove the model is good?
2. Why can a model perform better on training data than on unseen data?
3. What does a large difference between training and test performance potentially indicate?

---

# 12. Evaluate Test Performance

Now evaluate the model on the held-out test set.

```python
test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

test_precision = precision_score(
    y_test,
    y_test_pred
)

test_recall = recall_score(
    y_test,
    y_test_pred
)

test_f1 = f1_score(
    y_test,
    y_test_pred
)

print("Test accuracy:", test_accuracy)
print("Precision:", test_precision)
print("Recall:", test_recall)
print("F1 score:", test_f1)

print("\nClassification report:")
print(classification_report(y_test, y_test_pred))
```

---

# 13. Understand the Classification Metrics

For this lab:

```text
Positive class = survived
Negative class = did not survive
```

## Accuracy

Accuracy answers:

> What proportion of all predictions were correct?

```text
Accuracy =
correct predictions / all predictions
```

---

## Precision

Precision answers:

> Of the passengers the model predicted would survive, how many actually survived?

High precision means that positive predictions are usually correct.

---

## Recall

Recall answers:

> Of the passengers who actually survived, how many did the model correctly identify?

High recall means that the model successfully identifies a large proportion of the actual survivors.

---

## F1 Score

F1 combines precision and recall into one measure.

It is useful when we want a balance between:

* avoiding incorrect positive predictions
* finding as many positive cases as possible

### Task

Write a short interpretation of the model's:

* precision
* recall
* F1 score

Use the Titanic context.

For example:

> A precision of ___ means that among the passengers predicted to have survived, approximately ___% actually survived.

---

# 14. Build and Interpret the Confusion Matrix

A confusion matrix shows the types of predictions the classifier made.

```python
cm = confusion_matrix(
    y_test,
    y_test_pred
)

print(cm)
```

Visualize it:

```python
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix: Logistic Regression on Titanic")
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.show()
```

For binary classification, the confusion matrix contains:

```text
                    Predicted
                  0          1

Actual  0       TN         FP

        1       FN         TP
```

Where:

* **TN** = true negative
* **FP** = false positive
* **FN** = false negative
* **TP** = true positive

For Titanic:

* TP = predicted survived and actually survived
* TN = predicted did not survive and actually did not survive
* FP = predicted survived but actually did not survive
* FN = predicted did not survive but actually survived

### Questions

1. How many true positives are there?
2. How many true negatives are there?
3. How many false positives are there?
4. How many false negatives are there?
5. Which type of mistake would you consider more concerning in this classroom example? Explain your reasoning.

There is not necessarily one universally correct answer. The importance of an error depends on the purpose of the model.

---

# 15. Cross-Validation

So far, we have used one training/test split.

A single split can sometimes give an unstable estimate of performance because the result depends partly on which observations happen to be placed in the training and test sets.

Cross-validation gives us another way to evaluate the model during development.

We will use **5-fold cross-validation**.

```python
cv_scores = cross_val_score(
    logreg_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="f1",
)

print("Cross-validation F1 scores:")
print(cv_scores)

print("\nMean CV F1:")
print(cv_scores.mean())

print("\nStandard deviation:")
print(cv_scores.std())
```

## How 5-fold cross-validation works

Conceptually:

```text
Training data

Fold 1 → validation
Folds 2–5 → training

Fold 2 → validation
Folds 1,3–5 → training

Fold 3 → validation
Folds 1,2,4,5 → training

Fold 4 → validation
Folds 1–3,5 → training

Fold 5 → validation
Folds 1–4 → training
```

Each observation gets a chance to be part of the validation set.

The five scores are then summarized using their mean.

### Questions

1. Why can cross-validation be more informative than relying on one split?
2. Why might the mean cross-validation F1 differ from the final test F1?
3. What does the standard deviation of the cross-validation scores tell us?
4. Why is it useful to look at both the mean and the variation?

---

# 16. Precision and Recall Trade-Off

Logistic regression can produce probabilities.

For example:

```text
Passenger A → 0.91 probability of survival
Passenger B → 0.73 probability of survival
Passenger C → 0.42 probability of survival
Passenger D → 0.18 probability of survival
```

The model then needs a rule for converting probabilities into class predictions.

A common threshold is:

```text
probability ≥ 0.50 → predict survived
probability <  0.50 → predict did not survive
```

We can change that threshold.

First, obtain the predicted probabilities:

```python
y_test_proba = logreg_pipeline.predict_proba(X_test)[:, 1]
```

Now compare thresholds of 0.5 and 0.7.

```python
y_pred_05 = (
    y_test_proba >= 0.5
).astype(int)

y_pred_07 = (
    y_test_proba >= 0.7
).astype(int)
```

Evaluate them:

```python
print("Threshold 0.5")

print(
    "Precision:",
    precision_score(y_test, y_pred_05)
)

print(
    "Recall:",
    recall_score(y_test, y_pred_05)
)

print("\nThreshold 0.7")

print(
    "Precision:",
    precision_score(y_test, y_pred_07)
)

print(
    "Recall:",
    recall_score(y_test, y_pred_07)
)
```

## What does changing the threshold do?

The threshold does **not** change the probabilities produced by the model.

It changes how those probabilities are converted into class predictions.

When we increase the threshold from 0.5 to 0.7, the model becomes more demanding before predicting the positive class.

In many situations, this leads to:

```text
Higher threshold
      ↓
Fewer positive predictions
      ↓
Usually higher precision
      ↓
Usually lower recall
```

The exact results depend on the dataset and model.

### Questions

1. What happened to precision when the threshold increased?
2. What happened to recall?
3. Why might a higher threshold be useful in some applications?
4. Why might a lower threshold be useful in other applications?
5. Why does threshold choice depend on the goal of the classification task?

---

# 17. Compare with a Decision Tree

Different classification algorithms can learn different types of relationships.

Now we will compare logistic regression with a decision tree.

```python
tree_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        DecisionTreeClassifier(
            random_state=42,
            max_depth=4
        )
    ),
])

tree_pipeline.fit(X_train, y_train)

y_tree_pred = tree_pipeline.predict(X_test)
```

Evaluate the tree:

```python
tree_accuracy = accuracy_score(
    y_test,
    y_tree_pred
)

tree_precision = precision_score(
    y_test,
    y_tree_pred
)

tree_recall = recall_score(
    y_test,
    y_tree_pred
)

tree_f1 = f1_score(
    y_test,
    y_tree_pred
)

print("Decision tree accuracy:", tree_accuracy)
print("Decision tree precision:", tree_precision)
print("Decision tree recall:", tree_recall)
print("Decision tree F1:", tree_f1)
```

---

## Compare the Models

Create a simple comparison table:

```python
comparison = pd.DataFrame({
    "Model": [
        "Baseline",
        "Logistic Regression",
        "Decision Tree"
    ],
    "Accuracy": [
        baseline_accuracy,
        test_accuracy,
        tree_accuracy
    ],
    "Precision": [
        np.nan,
        test_precision,
        tree_precision
    ],
    "Recall": [
        np.nan,
        test_recall,
        tree_recall
    ],
    "F1": [
        np.nan,
        test_f1,
        tree_f1
    ]
})

display(comparison)
```

### Questions

1. Which model has the highest test accuracy?
2. Which model has the highest F1 score?
3. Which model has the highest precision?
4. Which model has the highest recall?
5. Did the two models make exactly the same mistakes?
6. Why might a decision tree behave differently from logistic regression?
7. Is the model with the highest accuracy automatically the best model?

---

# 18. Hyperparameter Tuning

A model often has settings called **hyperparameters**.

These are values that we choose before or during the model-development process rather than values learned directly from the training examples.

For logistic regression, one important hyperparameter is `C`.

We will try a small set of values:

```python
param_grid = {
    "classifier__C": [
        0.1,
        1.0,
        10.0
    ]
}
```

We will use `GridSearchCV` to compare these settings using 5-fold cross-validation.

```python
grid_search = GridSearchCV(
    Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000)
        ),
    ]),
    param_grid=param_grid,
    cv=5,
    scoring="f1",
)
```

Fit the grid search using **training data only**:

```python
grid_search.fit(X_train, y_train)
```

Inspect the results:

```python
print("Best parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation F1:")
print(grid_search.best_score_)
```

---

# 19. Why Should We Not Tune Using the Test Set?

The test set is supposed to represent **unseen data**.

If we repeatedly check test performance and choose the model or hyperparameters that perform best on that test set, the test set is no longer truly independent.

The workflow should instead be:

```text
Training data
     ↓
Cross-validation
     ↓
Choose model/hyperparameters
     ↓
Final model
     ↓
Test set used once for final evaluation
```

The test set should be treated as a final exam.

### Questions

1. Why would it be a mistake to choose hyperparameters by repeatedly checking test-set performance?
2. Why is cross-validation useful for model selection?
3. What role does the final test set play?

---

# 20. Final Evaluation

Now retrieve the best model selected through cross-validation.

```python
best_model = grid_search.best_estimator_
```

Use it on the held-out test set.

```python
y_final_pred = best_model.predict(X_test)
```

Calculate the final metrics:

```python
final_accuracy = accuracy_score(
    y_test,
    y_final_pred
)

final_precision = precision_score(
    y_test,
    y_final_pred
)

final_recall = recall_score(
    y_test,
    y_final_pred
)

final_f1 = f1_score(
    y_test,
    y_final_pred
)

print("Final accuracy:", final_accuracy)
print("Final precision:", final_precision)
print("Final recall:", final_recall)
print("Final F1:", final_f1)
```

And the confusion matrix:

```python
final_cm = confusion_matrix(
    y_test,
    y_final_pred
)

print("Final confusion matrix:")
print(final_cm)
```

Visualize it:

```python
sns.heatmap(
    final_cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Final Confusion Matrix: Tuned Logistic Regression")
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.show()
```

---

# 21. Final Reflection

Answer the following questions.

### 1. Baseline

Is the final model meaningfully better than the baseline?

Explain using the results rather than simply saying "yes" or "no."

---

### 2. Untuned vs Tuned Model

Did hyperparameter tuning meaningfully improve the logistic regression model?

Compare:

* the original test performance
* the cross-validation performance
* the final tuned test performance

---

### 3. Model Choice

Which model would you choose for this problem?

Consider:

* accuracy
* precision
* recall
* F1
* consistency across validation
* simplicity and interpretability

The model with the highest single metric does not necessarily have to be your choice.

---

### 4. Error Types

Look at the final confusion matrix.

Which type of error is more common?

```text
False positives
or
False negatives
```

What does this mean in the Titanic context?

---

### 5. Threshold

How could changing the classification threshold affect the practical behavior of the model?

---

### 6. Binary Classification

Why is Titanic survival prediction called **binary classification**?

---

# 22. Looking Ahead: Multiclass Classification

In this lab, the target had two possible classes:

```text
0 → Did not survive
1 → Survived
```

Therefore:

> Titanic survival prediction is a **binary classification** problem.

In the next lab, we will use the **Iris dataset**.

Iris contains three possible species:

```text
Iris setosa
Iris versicolor
Iris virginica
```

Therefore:

> Iris species prediction is a **multiclass classification** problem.

The overall machine-learning workflow will remain familiar:

```text
Define the task
      ↓
Prepare features and target
      ↓
Split data
      ↓
Preprocess
      ↓
Train model
      ↓
Evaluate
      ↓
Compare models
```

However, multiclass classification introduces new questions.

For example:

* How does a classifier choose among three or more classes?
* How does the confusion matrix change?
* How should precision, recall, and F1 be calculated?
* What does "positive class" mean when there are three classes?
* How can we establish a useful multiclass baseline?

These questions will be explored in **Lab 2**.

---

# 23. Lab Summary

The main workflow from this lab is:

```text
1. Define the classification problem
        ↓
2. Validate the dataset briefly
        ↓
3. Choose features and target
        ↓
4. Split training and test data
        ↓
5. Build preprocessing pipelines
        ↓
6. Establish a baseline
        ↓
7. Train a real classifier
        ↓
8. Evaluate with multiple metrics
        ↓
9. Interpret the confusion matrix
        ↓
10. Use cross-validation
        ↓
11. Examine threshold effects
        ↓
12. Compare classifiers
        ↓
13. Tune hyperparameters
        ↓
14. Evaluate once on the held-out test set
        ↓
15. Reflect on model choice
```

The key idea is:

> **A classification model is not judged by accuracy alone. We need to understand what the model predicts correctly, what it gets wrong, how stable its performance is, and whether its behavior matches the purpose of the task.**

---

# Deliverables

Submit:

1. **A completed notebook or Python script**
2. **Short written answers** to the interpretation questions
3. **The model comparison table**
4. **The final confusion matrix**
5. **One paragraph explaining which model you would choose and why**

Your final paragraph should refer to evidence from the evaluation rather than simply stating which model had the highest accuracy.

---

# Optional Self-Study: ROC Curve and ROC AUC

ROC curves and ROC AUC are useful classification concepts, but they are **not required for the main lab workflow**.

If you want to explore them independently, logistic regression provides predicted probabilities that can be used to construct an ROC curve.

For example:

```python
from sklearn.metrics import roc_curve, roc_auc_score

y_test_proba = best_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_test_proba
)

roc_auc = roc_auc_score(
    y_test,
    y_test_proba
)

plt.plot(
    fpr,
    tpr,
    label=f"ROC AUC = {roc_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Titanic Logistic Regression")
plt.legend()
plt.show()

print("ROC AUC:", roc_auc)
```

### Self-study question

Why might ROC AUC be useful when we want to understand classifier performance across **different classification thresholds**, rather than evaluating the model at only one threshold?

You do not need to master ROC curves for the main lab. The important concepts for this lab are:

**binary classification → preprocessing → baseline → classification metrics → confusion matrix → cross-validation → threshold → model comparison → tuning.**
