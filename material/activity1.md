# Lab 1: Titanic Binary Classification

## Predicting Survival on the Titanic with Classification

---

# Lab Purpose

In the previous Titanic activity, you explored the dataset and performed EDA.

In this lab, we move from **understanding the data** to using the data to **make predictions**.

The question we want to answer is:

> **Can we use information about a passenger to predict whether that passenger survived?**

This is a **supervised classification** problem.

We will focus on the main ideas behind classification rather than the details of machine-learning software.

---

# Learning Outcomes

By the end of this lab, you should be able to:

* explain why Titanic survival prediction is a supervised-learning problem
* explain why it is a binary classification problem
* distinguish between **features (`X`)** and the **target (`y`)**
* split data into training and test sets
* prepare numerical and categorical data for a classifier
* train a logistic regression classifier
* make predictions
* evaluate a classifier using:

  * accuracy
  * precision
  * recall
  * F1 score
* interpret a confusion matrix
* understand why different classification models can produce different results
* compare logistic regression with a decision tree

---

# 1. From EDA to Classification

In the previous Titanic activity, the main question was:

> **What does the data look like?**

We looked at variables, distributions, missing values, relationships between variables, and other characteristics of the dataset.

Now we ask a different question:

> **Can we use the information in the dataset to predict an outcome?**

For Titanic:

```text
Passenger information
        ↓
   Classification model
        ↓
Predicted survival
```

The dataset contains the actual outcome for each passenger, so the model can learn from examples where the correct answer is already known.

---

## What is the target?

The variable we want to predict is:

```text
survived
```

It has two possible values:

```text
0 → Did not survive
1 → Survived
```

Because there are only two possible classes, this is called:

> **binary classification**

### Questions

1. Why is Titanic survival prediction a supervised-learning problem?
2. Why is it a binary classification problem?

<details>
<summary><strong>Sample answer</strong></summary>

Titanic survival prediction is supervised learning because the dataset contains the known survival outcome for each passenger. The model can learn a relationship between the passenger information and the known outcome.

It is binary classification because the target has two possible classes: 0 (did not survive) and 1 (survived).

</details>

---

# 2. Set Up the Environment

Run the following in your notebook:

```python
!pip install pandas scikit-learn matplotlib seaborn -q
```

Import the libraries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
```

---

# 3. Load the Titanic Dataset

We will use the Titanic dataset from OpenML.

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

You have already explored this dataset in the previous activity, so we will not repeat the EDA.

---

# 4. Select the Variables

For this introductory classification task, we will use:

```text
pclass
sex
age
sibsp
parch
fare
embarked
```

The target is:

```text
survived
```

Create the feature and target datasets:

```python
features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X = df[features].copy()
y = df["survived"].astype(int)
```

Remember:

```text
X → information used to make predictions

y → the outcome we want to predict
```

### Question

In your own words, what is the difference between `X` and `y`?

<details>
<summary><strong>Sample answer</strong></summary>

`X` contains the features or input information used by the model to make predictions. `y` contains the target, which is the outcome the model is trying to predict.

For Titanic, `X` contains passenger information such as age, sex, class, and fare, while `y` contains whether the passenger survived.

</details>

---

# 5. Split the Data

We need to separate the data into:

* **training data** — used to learn the model
* **test data** — used to evaluate the model on unseen examples

We will use 80% for training and 20% for testing.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Check the sizes:

```python
print("Training set:", X_train.shape)
print("Test set:", X_test.shape)
```

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

We do not want to evaluate a model only on the examples it has already seen.

### Question

Why do we need a separate test set?

<details>
<summary><strong>Sample answer</strong></summary>

The test set allows us to evaluate the model on data that was not used to train it. This gives us a better idea of how the model may perform on new, unseen passengers.

</details>

---

# 6. Prepare the Data

Before a classifier can use the data, we need to convert the different types of variables into a suitable numerical form.

We have two types of features.

### Numerical features

```python
numeric_features = [
    "age",
    "sibsp",
    "parch",
    "fare"
]
```

### Categorical features

```python
categorical_features = [
    "pclass",
    "sex",
    "embarked"
]
```

Notice that `pclass` contains numbers, but in this activity we treat it as a category because it represents passenger classes rather than a continuous numerical measurement.

---

## 6.1 Prepare Numerical Features

For numerical variables:

1. fill missing values using the median
2. standardize the values

```python
imputer_num = SimpleImputer(strategy="median")

X_train_num = imputer_num.fit_transform(
    X_train[numeric_features]
)

X_test_num = imputer_num.transform(
    X_test[numeric_features]
)
```

Notice the difference:

```text
Training data → fit_transform()

Test data     → transform()
```

The median is calculated using the training data only.

Now standardize:

```python
scaler = StandardScaler()

X_train_num = scaler.fit_transform(X_train_num)
X_test_num = scaler.transform(X_test_num)
```

---

## 6.2 Prepare Categorical Features

For categorical variables, we will:

1. fill missing values using the most frequent category
2. convert categories into one-hot encoded columns

```python
imputer_cat = SimpleImputer(strategy="most_frequent")

X_train_cat = imputer_cat.fit_transform(
    X_train[categorical_features]
)

X_test_cat = imputer_cat.transform(
    X_test[categorical_features]
)
```

Now encode the categories:

```python
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

X_train_cat = encoder.fit_transform(X_train_cat)
X_test_cat = encoder.transform(X_test_cat)
```

One-hot encoding converts categories into numerical indicator variables that the classifier can use.

For example:

```text
sex

male
female
```

can become something like:

```text
sex_female
sex_male
```

---

## 6.3 Combine the Features

Now combine the numerical and categorical data:

```python
X_train_processed = np.hstack([
    X_train_num,
    X_train_cat
])

X_test_processed = np.hstack([
    X_test_num,
    X_test_cat
])
```

Check the result:

```python
print("Training data shape:", X_train_processed.shape)
print("Test data shape:", X_test_processed.shape)
```

At this point, our original data has been converted into numerical data that can be used by the classifier.

### Important idea

The preprocessing is learned from the **training data** and then applied to the test data.

We should not calculate values such as the median or scaling parameters using the complete dataset before the split.

---

# 7. Establish a Simple Baseline

Before training a real classifier, it is useful to have something simple to compare against.

Suppose we simply predict the most common class for every passenger.

For example:

```text
0
0
0
0
0
...
```

This is a very simple strategy. It does not learn a meaningful relationship between passenger characteristics and survival.

Calculate the majority-class baseline:

```python
majority_class = y_train.mode()[0]

y_baseline = np.full(
    len(y_test),
    majority_class
)

baseline_accuracy = accuracy_score(
    y_test,
    y_baseline
)

print("Baseline accuracy:", baseline_accuracy)
```

### Question

Why is a baseline useful?

<details>
<summary><strong>Sample answer</strong></summary>

A baseline gives us a simple reference point. A real classifier should perform better than a strategy that simply predicts the most common class. If the model performs only slightly better than the baseline, it may not be providing much useful predictive information.

</details>

---

# 8. Train a Logistic Regression Classifier

We will start with **logistic regression**.

Despite its name, logistic regression is commonly used for classification.

```python
model = LogisticRegression(max_iter=1000)

model.fit(
    X_train_processed,
    y_train
)
```

The model has now learned from the training data.

---

# 9. Make Predictions

Use the trained model to predict the test data:

```python
y_pred = model.predict(X_test_processed)
```

Look at some predictions:

```python
print("Predictions:")
print(y_pred[:20])

print("\nActual values:")
print(y_test.iloc[:20].values)
```

The model produces a predicted class:

```text
0 → predicted not survived
1 → predicted survived
```

We can now compare the predictions with the actual outcomes.

---

# 10. Evaluate the Classifier

Classification models can be evaluated using several metrics.

We will use:

* accuracy
* precision
* recall
* F1 score

Calculate them:

```python
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 score :", f1)
```

---

## Accuracy

Accuracy answers:

> **What proportion of all predictions were correct?**

```text
Accuracy =
correct predictions / all predictions
```

---

## Precision

Precision answers:

> **Of the passengers predicted to have survived, how many actually survived?**

For example, if precision is 0.80:

> Among the passengers predicted to survive, approximately 80% actually survived.

---

## Recall

Recall answers:

> **Of the passengers who actually survived, how many did the model correctly identify?**

For example, if recall is 0.70:

> The model correctly identified approximately 70% of the passengers who actually survived.

---

## F1 Score

F1 combines precision and recall into a single measure.

It is useful when we want a balance between precision and recall.

### Task

Write 2–3 sentences interpreting the model's precision and recall in the Titanic context.

<details>
<summary><strong>Sample answer structure</strong></summary>

My model has a precision of **[value]**. This means that among the passengers predicted to survive, approximately **[value × 100]%** actually survived.

The recall is **[value]**. This means that the model correctly identified approximately **[value × 100]%** of the passengers who actually survived.

</details>

---

# 11. Understand the Confusion Matrix

A confusion matrix gives us more detail about the predictions.

```python
cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)
```

We can visualize it:

```python
sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

A binary confusion matrix contains four types of results:

```text
                    Predicted
                  0          1

Actual  0       TN         FP

        1       FN         TP
```

For Titanic:

* **TN** = predicted did not survive, and actually did not survive
* **FP** = predicted survived, but actually did not survive
* **FN** = predicted did not survive, but actually survived
* **TP** = predicted survived, and actually survived

### Question

Explain what a **false positive** and a **false negative** mean in the Titanic context.

<details>
<summary><strong>Sample answer</strong></summary>

A false positive occurs when the model predicts that a passenger survived, but the passenger actually did not survive.

A false negative occurs when the model predicts that a passenger did not survive, but the passenger actually survived.

</details>

---

# 12. Look at the Classification Report

Scikit-learn can summarize the main classification metrics:

```python
print(
    classification_report(
        y_test,
        y_pred
    )
)
```

The report includes precision, recall, F1 score, and the number of observations in each class.

Remember that the two classes have different meanings:

```text
0 → Did not survive
1 → Survived
```

---

# 13. Try a Second Classifier: Decision Tree

Different classifiers can learn different types of relationships.

Now we will compare logistic regression with a **decision tree**.

```python
tree = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

tree.fit(
    X_train_processed,
    y_train
)
```

Make predictions:

```python
y_tree_pred = tree.predict(
    X_test_processed
)
```

Calculate the metrics:

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

print("Decision Tree")
print("Accuracy :", tree_accuracy)
print("Precision:", tree_precision)
print("Recall   :", tree_recall)
print("F1 score :", tree_f1)
```

---

# 14. Compare the Classifiers

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
        accuracy,
        tree_accuracy
    ],

    "Precision": [
        np.nan,
        precision,
        tree_precision
    ],

    "Recall": [
        np.nan,
        recall,
        tree_recall
    ],

    "F1": [
        np.nan,
        f1,
        tree_f1
    ]
})

display(comparison)
```

### Questions

1. Which model has the highest F1 score?
2. Which model would you choose for this problem, and why?

<details>
<summary><strong>Sample answer</strong></summary>

The model with the highest F1 score is **[model]**.

I would choose **[model]** because it provides the best balance between precision and recall according to the results. I would also consider whether its accuracy and other metrics are reasonable compared with the baseline and the other classifier.

There is not necessarily one universally correct choice. The important part is to justify the choice using the results.

</details>

---

# 15. What Have We Learned?

The classification workflow in this lab can be summarized as:

```text
Define the classification problem
            ↓
       Select X and y
            ↓
      Split the data
            ↓
     Prepare the data
            ↓
     Train a classifier
            ↓
       Make predictions
            ↓
     Evaluate predictions
            ↓
   Understand errors
            ↓
     Compare classifiers
```

The important concepts are:

### Supervised learning

The model learns from examples where the correct outcome is known.

### Binary classification

The target contains two possible classes.

### Features and target

```text
X → input information

y → outcome we want to predict
```

### Training and test data

The model learns from the training data and is evaluated on unseen test data.

### Classification metrics

Different metrics answer different questions.

* **Accuracy** → How many predictions were correct?
* **Precision** → When the model predicts survival, how often is it correct?
* **Recall** → How many actual survivors did the model identify?
* **F1** → How well does the model balance precision and recall?

### Confusion matrix

The confusion matrix helps us understand **what kinds of mistakes** the classifier makes.

### Different classifiers

Logistic regression and decision trees can produce different predictions because they learn relationships in different ways.

---

# 16. Final Reflection

Answer the following questions.

### 1. Classification

Why is Titanic survival prediction a binary classification problem?

<details>
<summary><strong>Sample answer</strong></summary>

It is binary classification because the target variable `survived` has two possible classes: 0 (did not survive) and 1 (survived).

</details>

### 2. Model performance

Is your classifier meaningfully better than the baseline?

Use the accuracy and/or F1 score to support your answer.

<details>
<summary><strong>Sample answer structure</strong></summary>

The baseline accuracy was **[value]**, while the **[chosen model]** achieved an accuracy of **[value]**. The model therefore performed **[better/slightly better/not clearly better]** than the baseline.

I would also consider the F1 score because accuracy alone does not describe the types of errors made by the classifier.

</details>

### 3. Model choice

Which model would you choose: logistic regression or decision tree?

Explain your choice using at least two metrics.

<details>
<summary><strong>Sample answer structure</strong></summary>

I would choose **[model]** because it achieved an F1 score of **[value]** and an accuracy of **[value]**. Compared with the other model, it provided **[better precision / better recall / better balance between precision and recall]**.

</details>

---

# Deliverables

Submit:

1. **Your completed notebook**
2. **Your answers to the short questions**
3. **The model comparison table**
4. **The confusion matrix**
5. **Your final model-choice explanation**

Your final explanation should use evidence from the evaluation results rather than simply saying that one model is "better."

---

# Looking Ahead: Multiclass Classification

In this lab, the target had two possible classes:

```text
0 → Did not survive
1 → Survived
```

Therefore, Titanic survival prediction is **binary classification**.

In the next lab, we will use the Iris dataset.

Iris contains three species:

```text
Iris setosa
Iris versicolor
Iris virginica
```

This is an example of:

> **multiclass classification**

The basic workflow will remain familiar:

```text
Features and target
        ↓
   Split the data
        ↓
     Prepare data
        ↓
   Train classifier
        ↓
     Predict
        ↓
    Evaluate
```

The difference is that the model will have to choose between **more than two classes**.
