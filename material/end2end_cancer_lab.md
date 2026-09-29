# Lab 2: End-to-End Cancer Diagnosis Support with Classification and Clustering

<!--
This lab reviews weeks 2-5 through a second realistic end-to-end system.
Lab 1 focused on housing regression and clustering.
Lab 2 focuses on cancer classification and clustering.
-->

- [1. From a Business Question to Machine Learning Tasks](#1-from-a-business-question-to-machine-learning-tasks)
  - [1.1 Understand the business objective](#11-understand-the-business-objective)
  - [1.2 Separate the classification and clustering questions](#12-separate-the-classification-and-clustering-questions)
- [2. Load and Understand the Cancer Dataset](#2-load-and-understand-the-cancer-dataset)
  - [2.1 Import libraries](#21-import-libraries)
  - [2.2 Load the data](#22-load-the-data)
  - [2.3 Inspect rows, columns, and data types](#23-inspect-rows-columns-and-data-types)
  - [2.4 Identify features, target, and classes](#24-identify-features-target-and-classes)
- [3. Initial EDA and Data Quality](#3-initial-eda-and-data-quality)
  - [3.1 Check missing values and duplicates](#31-check-missing-values-and-duplicates)
  - [3.2 Explore the target distribution](#32-explore-the-target-distribution)
  - [3.3 Explore feature distributions](#33-explore-feature-distributions)
  - [3.4 Compare features by diagnosis](#34-compare-features-by-diagnosis)
  - [3.5 Inspect correlations](#35-inspect-correlations)
- [4. Classification System: Predict Diagnosis](#4-classification-system-predict-diagnosis)
  - [4.1 Select features and target](#41-select-features-and-target)
  - [4.2 Split before learned preprocessing](#42-split-before-learned-preprocessing)
  - [4.3 Build a baseline](#43-build-a-baseline)
  - [4.4 Standardize features for distance-based and coefficient-based models](#44-standardize-features-for-distance-based-and-coefficient-based-models)
  - [4.5 Train logistic regression](#45-train-logistic-regression)
  - [4.6 Train K-nearest neighbors](#46-train-k-nearest-neighbors)
  - [4.7 Train a decision tree](#47-train-a-decision-tree)
  - [4.8 Compare classification metrics](#48-compare-classification-metrics)
  - [4.9 Inspect the confusion matrix](#49-inspect-the-confusion-matrix)
  - [4.10 Interpret prediction probabilities](#410-interpret-prediction-probabilities)
- [5. Clustering System: Discover Tumor Measurement Profiles](#5-clustering-system-discover-tumor-measurement-profiles)
  - [5.1 Change the question](#51-change-the-question)
  - [5.2 Select clustering features](#52-select-clustering-features)
  - [5.3 Standardize clustering features](#53-standardize-clustering-features)
  - [5.4 Compare candidate K values](#54-compare-candidate-k-values)
  - [5.5 Profile the selected K-Means clusters](#55-profile-the-selected-k-means-clusters)
  - [5.6 Compare clusters with diagnosis after fitting](#56-compare-clusters-with-diagnosis-after-fitting)
  - [5.7 Compare with agglomerative clustering](#57-compare-with-agglomerative-clustering)
- [6. Connect the Lab to Real Project Practice](#6-connect-the-lab-to-real-project-practice)
- [7. Final Concept Review and Exam-Style Questions](#7-final-concept-review-and-exam-style-questions)

## Learning Objectives

By the end of this lab, you should be able to:

- translate a business objective into classification and clustering questions;
- distinguish classification from regression;
- distinguish binary classification from multiclass classification;
- identify features, target, and class labels;
- perform focused EDA before modeling;
- check missing values, duplicates, class balance, distributions, and correlations;
- create a train/test split for supervised classification;
- explain and avoid data leakage;
- build a baseline classifier;
- standardize numerical features when appropriate;
- train logistic regression, K-nearest neighbors, and a decision tree;
- evaluate classification using accuracy, precision, recall, F1, and a confusion matrix;
- explain false positives and false negatives in a medical-support context;
- use K-Means and agglomerative clustering to explore tumor measurement profiles;
- explain what this system can and cannot prove.

## 1. From a Business Question to Machine Learning Tasks

### 1.1 Understand the business objective

Imagine you have joined a health analytics team. The team studies measurements taken from breast mass images. Each row describes one case using numerical measurements such as radius, texture, smoothness, compactness, concavity, and symmetry.

The business team asks:

> Can we use tumor measurements to support diagnosis review and discover groups of similar cases?

This is not yet a machine-learning task. In practice, we must clarify what the model is for.

Possible uses include:

- supporting a clinical review workflow;
- flagging cases that need careful human attention;
- comparing simple classification models;
- exploring whether cases form measurement-based profiles.

This lab is for learning machine learning. It is not a medical device, and it must not be treated as medical advice.

**Question 1 — Business objective versus model objective**

Why is the business objective not the same thing as the model objective?

<details>
<summary>Solution</summary>

The business objective describes the real-world purpose: supporting diagnosis review and understanding similar cases.

The model objective is more specific. For classification, the model predicts a class label from input features. For clustering, the model groups observations using feature similarity.

A useful real-world system needs both: a clear practical purpose and a clear modeling task.

</details>

### 1.2 Separate the classification and clustering questions

This lab builds two connected systems from the same dataset.

**System A: Classification**

```text
Tumor measurements → predict diagnosis class
```

This is supervised learning because the training data contains known diagnosis labels.

**System B: Clustering**

```text
Tumor measurements → discover similar measurement profiles
```

This is unsupervised learning because the clustering algorithm does not use the diagnosis label during fitting.

**Question 2 — Diagnosis-support framing**

A stakeholder says: "I do not only want a label. I want to know which mistakes would be most serious if this model supported review." What should you clarify before comparing classifiers?

<details>
<summary>Solution</summary>

You should clarify:

- which class should receive special attention;
- whether false negatives or false positives are more costly;
- which metrics should be emphasized, such as recall, precision, F1, and the confusion matrix;
- that the model is decision support, not a replacement for expert judgement.

</details>

**Question 3 — Clustering workflow**

An analyst wants to group cases by measurement similarity, then compare the groups with diagnosis afterward. Why should diagnosis be left out during clustering and used only after the clusters are created?

<details>
<summary>Solution</summary>

If diagnosis is included during clustering, the groups are partly based on the answer label.

Leaving diagnosis out keeps the clustering task focused on measurement similarity. Comparing clusters with diagnosis afterward is an interpretation step, not supervised training.

</details>

## 2. Load and Understand the Cancer Dataset

### 2.1 Import libraries

Run this setup cell first.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

sns.set_theme(style="whitegrid")
```

**Code explanation**

- `load_breast_cancer` loads the dataset directly from scikit-learn.
- `train_test_split` separates data used for learning from data used for evaluation.
- `StandardScaler` standardizes features for models affected by scale.
- `DummyClassifier` creates a simple baseline.
- `LogisticRegression`, `KNeighborsClassifier`, and `DecisionTreeClassifier` are classification models.
- Accuracy, precision, recall, F1, confusion matrix, and classification report evaluate classification.
- `KMeans` and `AgglomerativeClustering` create clusters.
- `silhouette_score` helps compare clustering results.

### 2.2 Load the data

This dataset is built into scikit-learn, so you can run it directly in Google Colab.

```python
cancer = load_breast_cancer()

df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df["diagnosis"] = cancer.target
df["diagnosis_name"] = df["diagnosis"].map({
    0: cancer.target_names[0],
    1: cancer.target_names[1],
})

df.head()
```

```python
print("Target names:", cancer.target_names)
print("Feature count:", len(cancer.feature_names))
print("Rows and columns:", df.shape)
```

**Code explanation**

- `cancer.data` contains the numerical input features.
- `cancer.target` contains the known diagnosis class.
- `diagnosis_name` gives readable class labels.
- In this dataset, scikit-learn uses `0 = malignant` and `1 = benign`.

### 2.3 Inspect rows, columns, and data types

Before modeling, inspect the structure.

```python
df.info()
```

```python
df.describe()
```

```python
df.head()
```

**Code explanation**

- `info()` shows columns, non-missing counts, and data types.
- `describe()` summarizes numerical columns.
- `head()` shows example rows.

**Question 4 — Row meaning**

What does one row represent in this dataset?

<details>
<summary>Solution</summary>

One row represents one breast mass case described by numerical measurements and a known diagnosis label.

</details>

### 2.4 Identify features, target, and classes

In this dataset:

- the input features are numerical measurements;
- the target is `diagnosis`;
- the readable target is `diagnosis_name`;
- this is binary classification because there are two classes.

```python
target = "diagnosis"
readable_target = "diagnosis_name"
feature_columns = cancer.feature_names.tolist()

print("Number of features:", len(feature_columns))
print("Target:", target)
print("Classes:")
print(df[[target, readable_target]].drop_duplicates().sort_values(target))
```

**Question 5 — Features and target**

Why should `diagnosis` and `diagnosis_name` not be included as input features?

<details>
<summary>Solution</summary>

They contain the answer we want the model to predict.

Including the target as an input feature would cause data leakage. The model would appear to perform extremely well because it was given the answer.

</details>

## 3. Initial EDA and Data Quality

### 3.1 Check missing values and duplicates

```python
missing_counts = df.isna().sum().sort_values(ascending=False)
missing_counts.head(10)
```

```python
duplicate_count = df.duplicated().sum()
print("Duplicate rows:", duplicate_count)
```

**Code explanation**

- Missing values may require imputation.
- Duplicates may distort model evaluation if repeated cases appear in both training and test sets.
- This dataset is already clean, but we still check.

**Question 6 — Data quality**

If this dataset has no missing values, why do we still check for them?

<details>
<summary>Solution</summary>

Checking data quality is part of the workflow.

In real projects, datasets often change. A clean dataset today does not guarantee that future data will always be clean.

</details>

### 3.2 Explore the target distribution

```python
class_counts = df[readable_target].value_counts()
class_counts
```

```python
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x=readable_target, order=class_counts.index)
plt.xlabel("Diagnosis")
plt.ylabel("Number of cases")
plt.title("Target class distribution")
plt.show()
```

```python
class_percentages = df[readable_target].value_counts(normalize=True).mul(100).round(1)
class_percentages
```

**Code explanation**

- `value_counts()` shows how many observations are in each class.
- Class balance matters because accuracy can be misleading when one class is much more common.
- The most frequent class also gives a useful baseline.

**Question 7 — Baseline thinking**

If most cases are benign, why might a classifier with high accuracy still be weak?

<details>
<summary>Solution</summary>

A classifier can get many cases correct by mostly predicting the majority class.

In medical-support problems, we also care about the types of mistakes. Missing malignant cases can be much more serious than the overall accuracy suggests.

</details>

### 3.3 Explore feature distributions

Start with a few representative features.

```python
selected_eda_features = [
    "mean radius",
    "mean texture",
    "mean smoothness",
    "mean concavity",
    "worst radius",
    "worst concavity",
]

df[selected_eda_features].hist(figsize=(12, 8), bins=25)
plt.suptitle("Distributions of selected tumor measurements", y=1.02)
plt.show()
```

**Code explanation**

- Histograms show the shape and range of each feature.
- Some features have larger numerical ranges than others.
- Large differences in scale matter for distance-based models such as KNN and K-Means.

### 3.4 Compare features by diagnosis

```python
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x=readable_target, y="mean radius")
plt.xlabel("Diagnosis")
plt.ylabel("Mean radius")
plt.title("Mean radius by diagnosis")
plt.show()
```

```python
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x=readable_target, y="worst concavity")
plt.xlabel("Diagnosis")
plt.ylabel("Worst concavity")
plt.title("Worst concavity by diagnosis")
plt.show()
```

```python
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="mean radius",
    y="mean texture",
    hue=readable_target,
    alpha=0.8,
)
plt.title("Two measurements colored by diagnosis")
plt.show()
```

**Code explanation**

- Boxplots compare distributions between classes.
- Scatterplots help us see whether classes are separated in two selected features.
- We do not expect two features to explain everything.

**Question 8 — EDA and modeling**

Why is it useful to compare feature distributions by diagnosis before training a classifier?

<details>
<summary>Solution</summary>

It helps us understand whether some features may contain useful class information.

It also helps us notice overlap. If the classes overlap strongly, the task may be difficult and mistakes should be expected.

</details>

### 3.5 Inspect correlations

Correlation is not the main evaluation tool for classification, but it still helps us understand numerical relationships.

```python
correlation_with_target = (
    df[feature_columns + [target]]
    .corr(numeric_only=True)[target]
    .drop(target)
    .sort_values(key=abs, ascending=False)
)

correlation_with_target.head(10)
```

```python
top_corr_features = correlation_with_target.head(8).index.tolist()

plt.figure(figsize=(9, 7))
sns.heatmap(
    df[top_corr_features + [target]].corr(numeric_only=True),
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
)
plt.title("Correlation heatmap for selected features")
plt.show()
```

**Code explanation**

- `corr()` calculates Pearson correlation between numerical columns.
- Because `diagnosis` is coded as `0` and `1`, correlation with the target gives an initial view of linear association with the class code.
- This does not replace classifier evaluation.

**Question 9 — Correlation limitation**

Why should we not choose the classifier only from Pearson correlations?

<details>
<summary>Solution</summary>

Pearson correlation measures linear association between two numerical columns.

Classification models may use several features together, and some relationships may not be captured by a single correlation value.

Correlation is useful for investigation, but it is not the final model-selection method.

</details>

## 4. Classification System: Predict Diagnosis

### 4.1 Select features and target

```python
X = df[feature_columns]
y = df[target]

print("X shape:", X.shape)
print("y shape:", y.shape)
```

**Code explanation**

- `X` contains the input measurements.
- `y` contains the diagnosis class.
- The readable diagnosis label is not included in `X`.

### 4.2 Split before learned preprocessing

Split the data before fitting scalers or models.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print("Training class counts:")
print(y_train.value_counts().sort_index())
print("Test class counts:")
print(y_test.value_counts().sort_index())
```

**Code explanation**

- `test_size=0.2` keeps 20 percent of rows for testing.
- `random_state=42` makes the split reproducible.
- `stratify=y` keeps class proportions similar in train and test sets.

**Question 10 — Stratification**

Why is `stratify=y` useful in classification?

<details>
<summary>Solution</summary>

It helps keep the class distribution similar in the training and test sets.

This is useful because evaluation can be misleading if the test set has very different class proportions from the data used for training.

</details>

### 4.3 Build a baseline

A baseline is a simple reference model.

Here the baseline always predicts the most frequent training class.

```python
baseline_model = DummyClassifier(strategy="most_frequent")
baseline_model.fit(X_train, y_train)

baseline_predictions = baseline_model.predict(X_test)
baseline_predictions[:10]
```

```python
def classification_scores(model_name, y_true, y_pred):
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "F1": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
    }

results = []
results.append(
    classification_scores("Most frequent baseline", y_test, baseline_predictions)
)

pd.DataFrame(results)
```

**Code explanation**

- `DummyClassifier` creates a simple baseline.
- `strategy="most_frequent"` always predicts the majority class from the training data.
- We use `pos_label=0` so precision, recall, and F1 focus on the malignant class.

**Question 11 — Positive class**

Why might we focus precision and recall on the malignant class?

<details>
<summary>Solution</summary>

In this dataset, `0` means malignant.

For a diagnosis-support task, missing malignant cases is especially important. Focusing metrics on the malignant class helps us inspect how well the model identifies that class.

</details>

### 4.4 Standardize features for distance-based and coefficient-based models

KNN uses distances, so feature scale matters. Logistic regression can also behave better when numerical features are on comparable scales.

Fit the scaler on the training data only.

```python
scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=feature_columns,
    index=X_train.index,
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=feature_columns,
    index=X_test.index,
)

X_train_scaled.head()
```

**Code explanation**

- `fit_transform()` learns means and standard deviations from training data, then applies them to training data.
- `transform()` applies the same learned scaling to test data.
- We do not fit a new scaler on the test set.

**Question 12 — Leakage**

Why should the scaler be fitted on the training data only?

<details>
<summary>Solution</summary>

The test set should represent unseen data.

If we use the test set to learn scaling values, information from the test set influences preprocessing. That is data leakage.

</details>

### 4.5 Train logistic regression

Logistic regression is a classification algorithm. It estimates class membership using learned coefficients.

```python
logistic_model = LogisticRegression(max_iter=5000)
logistic_model.fit(X_train_scaled, y_train)

logistic_predictions = logistic_model.predict(X_test_scaled)
logistic_predictions[:10]
```

```python
results.append(
    classification_scores("Logistic regression", y_test, logistic_predictions)
)

pd.DataFrame(results)
```

**Code explanation**

- `fit()` learns from the scaled training features and training target.
- `predict()` returns class labels for the test data.
- Logistic regression is used here as a classifier, not as a regression model.

**Question 13 — Logistic regression**

Why is logistic regression used for classification even though its name contains "regression"?

<details>
<summary>Solution</summary>

Logistic regression models the probability of belonging to a class, then converts that probability into a class prediction.

In this course context, it is a classification algorithm.

</details>

### 4.6 Train K-nearest neighbors

KNN classifies a new observation by looking at nearby training observations.

```python
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

knn_predictions = knn_model.predict(X_test_scaled)
knn_predictions[:10]
```

```python
results.append(
    classification_scores("KNN, k=5", y_test, knn_predictions)
)

pd.DataFrame(results)
```

**Code explanation**

- `n_neighbors=5` means the classifier uses the five nearest training observations.
- KNN depends on distances, so scaling matters.
- The model needs the training observations when making predictions because it compares new cases to stored cases.

**Question 14 — KNN and scaling**

Why can KNN perform poorly when features are not scaled?

<details>
<summary>Solution</summary>

KNN uses distances.

If one feature has much larger numerical values than another, it can dominate the distance calculation even if it is not more important.

</details>

### 4.7 Train a decision tree

A decision tree learns a sequence of feature-based questions.

```python
tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_model.fit(X_train, y_train)

tree_predictions = tree_model.predict(X_test)
tree_predictions[:10]
```

```python
results.append(
    classification_scores("Decision tree, depth 4", y_test, tree_predictions)
)

results_df = pd.DataFrame(results)
results_df
```

```python
plt.figure(figsize=(18, 8))
plot_tree(
    tree_model,
    feature_names=feature_columns,
    class_names=cancer.target_names,
    filled=True,
    max_depth=2,
    fontsize=8,
)
plt.title("First levels of the decision tree")
plt.show()
```

**Code explanation**

- A decision tree repeatedly splits the data using feature rules.
- `max_depth=4` limits tree complexity.
- The tree is trained on unscaled features because decision trees do not depend on distance calculations in the same way as KNN.

**Question 15 — Decision tree**

How does a decision tree make a classification?

<details>
<summary>Solution</summary>

It follows a sequence of feature-based decisions.

For example, it may ask whether a measurement is below or above a threshold. After several splits, the case reaches a leaf node with a predicted class.

</details>

### 4.8 Compare classification metrics

```python
results_df.sort_values("F1", ascending=False)
```

```python
print("Logistic regression report:")
print(classification_report(
    y_test,
    logistic_predictions,
    target_names=cancer.target_names,
))
```

**Code explanation**

- Accuracy is the proportion of correct predictions.
- Precision answers: of the cases predicted as a class, how many were actually that class?
- Recall answers: of the actual cases in a class, how many did the model find?
- F1 combines precision and recall.
- The classification report gives metrics for each class.

**Question 16 — Accuracy**

Why is accuracy alone not always enough?

<details>
<summary>Solution</summary>

Accuracy gives one overall number.

It can hide which class is being misclassified. In medical-support problems, the type of mistake matters, not only the total number of correct predictions.

</details>

**Question 17 — Precision and recall**

In this lab, what is the difference between precision for malignant cases and recall for malignant cases?

<details>
<summary>Solution</summary>

Precision for malignant cases asks:

```text
Of the cases predicted malignant, how many were actually malignant?
```

Recall for malignant cases asks:

```text
Of the truly malignant cases, how many did the model identify as malignant?
```

</details>

### 4.9 Inspect the confusion matrix

Use the logistic regression model for detailed inspection.

```python
cm = confusion_matrix(y_test, logistic_predictions)
cm
```

```python
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=cancer.target_names,
)
disp.plot(cmap="Blues")
plt.title("Confusion matrix: logistic regression")
plt.show()
```

```python
cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {name}" for name in cancer.target_names],
    columns=[f"Predicted {name}" for name in cancer.target_names],
)
cm_df
```

**Code explanation**

- Rows represent actual classes.
- Columns represent predicted classes.
- Diagonal values are correct predictions.
- Off-diagonal values are mistakes.

**Question 18 — False negative in context**

For the malignant class, what would it mean if an actual malignant case is predicted as benign?

<details>
<summary>Solution</summary>

That is a false negative for the malignant class.

In a diagnosis-support context, this is a serious type of error because a malignant case was not flagged as malignant by the model.

</details>

### 4.10 Interpret prediction probabilities

Some classifiers can provide class probabilities.

```python
probabilities = logistic_model.predict_proba(X_test_scaled)

probability_df = pd.DataFrame(
    probabilities,
    columns=[f"probability_{name}" for name in cancer.target_names],
    index=X_test.index,
)

probability_df["actual"] = y_test.map({0: "malignant", 1: "benign"})
probability_df["predicted"] = pd.Series(
    logistic_predictions,
    index=X_test.index,
).map({0: "malignant", 1: "benign"})

probability_df.head(10)
```

**Code explanation**

- `predict()` returns class labels.
- `predict_proba()` returns estimated probabilities for each class.
- A probability is not a guarantee. It is the model's estimate under its learned patterns.

**Question 19 — Probability interpretation**

If the model gives `probability_malignant = 0.90`, does that prove the case is malignant?

<details>
<summary>Solution</summary>

No.

It means the model estimates a high probability for the malignant class based on its learned patterns. It is not proof, and medical decisions require appropriate clinical processes.

</details>

## 5. Clustering System: Discover Tumor Measurement Profiles

### 5.1 Change the question

Classification used the diagnosis label.

Clustering asks a different question:

> Without using the diagnosis label, can we group cases that have similar tumor measurements?

This may help analysts inspect measurement profiles. It does not replace diagnosis.

**Question 20 — Target in clustering**

Should `diagnosis` be used as a clustering feature?

<details>
<summary>Solution</summary>

No.

If we use `diagnosis` as a clustering feature, the clusters are partly based on the answer label. For unsupervised exploration, we should cluster using measurement features only.

</details>

### 5.2 Select clustering features

Use a small set of interpretable measurements.

```python
cluster_features = [
    "mean radius",
    "mean texture",
    "mean smoothness",
    "mean concavity",
    "worst radius",
    "worst concavity",
]

cluster_df = df[cluster_features + [target, readable_target]].copy()
cluster_df.head()
```

**Code explanation**

- The clustering features are all numerical measurements.
- `diagnosis` is kept only for later interpretation after clustering.
- The clustering model will not use `diagnosis` during fitting.

### 5.3 Standardize clustering features

```python
cluster_scaler = StandardScaler()
X_cluster_scaled = cluster_scaler.fit_transform(cluster_df[cluster_features])

X_cluster_scaled[:5]
```

**Code explanation**

- K-Means and agglomerative clustering use distances.
- Standardization prevents large-scale features from dominating distance calculations.
- Here the full clustering dataset is being explored. If this were a future-data scoring system, the scaling process would need to be fitted on a reference dataset.

### 5.4 Compare candidate K values

```python
cluster_selection_rows = []
labels_by_k = {}

for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_cluster_scaled)
    labels_by_k[k] = labels
    
    cluster_selection_rows.append({
        "K": k,
        "Inertia": kmeans.inertia_,
        "Silhouette": silhouette_score(X_cluster_scaled, labels),
    })

cluster_selection = pd.DataFrame(cluster_selection_rows)
cluster_selection
```

```python
plt.figure(figsize=(7, 4))
sns.lineplot(data=cluster_selection, x="K", y="Inertia", marker="o")
plt.title("Elbow check for K-Means")
plt.show()
```

```python
plt.figure(figsize=(7, 4))
sns.lineplot(data=cluster_selection, x="K", y="Silhouette", marker="o")
plt.title("Silhouette score by K")
plt.show()
```

**Code explanation**

- Inertia measures compactness inside clusters.
- Inertia normally decreases as K increases, so it is not enough by itself.
- Silhouette compares separation and cohesion.
- These metrics guide interpretation; they do not prove that the clusters are clinically meaningful.

**Question 21 — K selection**

Why should we not simply choose the largest K because it has the lowest inertia?

<details>
<summary>Solution</summary>

Inertia almost always decreases when K increases because more clusters can fit the data more tightly.

Choosing K requires balancing compactness, separation, interpretability, and the purpose of the analysis.

</details>

### 5.5 Profile the selected K-Means clusters

Choose the K with the highest silhouette score for this lab.

```python
best_k = int(cluster_selection.sort_values("Silhouette", ascending=False).iloc[0]["K"])
print("Selected K:", best_k)

cluster_df["cluster"] = labels_by_k[best_k]
cluster_df.head()
```

```python
cluster_profile = (
    cluster_df
    .groupby("cluster")
    .agg(
        cases=("cluster", "size"),
        mean_radius=("mean radius", "mean"),
        mean_texture=("mean texture", "mean"),
        mean_smoothness=("mean smoothness", "mean"),
        mean_concavity=("mean concavity", "mean"),
        worst_radius=("worst radius", "mean"),
        worst_concavity=("worst concavity", "mean"),
    )
    .round(3)
)

cluster_profile
```

```python
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=cluster_df,
    x="mean radius",
    y="worst concavity",
    hue="cluster",
    palette="tab10",
)
plt.title("K-Means clusters using two selected measurements")
plt.show()
```

**Code explanation**

- Cluster numbers are labels created by the algorithm.
- Cluster `0` is not automatically better or worse than cluster `1`.
- A profile table helps us describe how clusters differ.

**Question 22 — Cluster labels**

Does cluster `0` mean "benign" and cluster `1` mean "malignant"?

<details>
<summary>Solution</summary>

No.

Cluster labels are arbitrary group names. We can compare clusters with diagnosis after fitting, but the clustering algorithm did not know the diagnosis labels.

</details>

### 5.6 Compare clusters with diagnosis after fitting

After clustering, we can compare cluster membership with diagnosis to interpret the groups.

```python
cluster_diagnosis_table = pd.crosstab(
    cluster_df["cluster"],
    cluster_df[readable_target],
    normalize="index",
).round(3)

cluster_diagnosis_table
```

```python
pd.crosstab(cluster_df["cluster"], cluster_df[readable_target])
```

**Code explanation**

- The crosstab compares clusters with known diagnosis labels after clustering.
- This is interpretation, not supervised training.
- If one cluster has many malignant cases, we can describe that pattern, but we should not claim the cluster is a diagnosis rule.

**Question 23 — Interpretation limit**

Why is it useful but risky to compare clusters with diagnosis labels?

<details>
<summary>Solution</summary>

It is useful because it helps us understand whether measurement-based clusters relate to known diagnosis.

It is risky because clusters are not supervised predictions. A cluster can contain a mix of diagnoses, and cluster labels should not be treated as medical decisions.

</details>

### 5.7 Compare with agglomerative clustering

Agglomerative clustering starts with individual observations and merges them into larger groups.

```python
agg_model = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
cluster_df["agg_cluster"] = agg_model.fit_predict(X_cluster_scaled)

cluster_df["agg_cluster"].value_counts().sort_index()
```

```python
pd.crosstab(cluster_df["agg_cluster"], cluster_df[readable_target])
```

```python
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=cluster_df,
    x="mean radius",
    y="worst concavity",
    hue="agg_cluster",
    palette="tab10",
)
plt.title("Agglomerative clusters using two selected measurements")
plt.show()
```

**Code explanation**

- K-Means and agglomerative clustering can produce different groupings.
- Both depend on selected features and scaling.
- Agreement between methods can support an interpretation, but it still does not prove that clusters are clinically meaningful.

**Question 24 — Comparing clustering methods**

Why might K-Means and agglomerative clustering create different clusters?

<details>
<summary>Solution</summary>

They use different algorithms.

K-Means represents clusters using centroids. Agglomerative clustering repeatedly merges observations or groups based on linkage rules. Because the definitions differ, the results can differ.

</details>

## 6. Connect the Lab to Real Project Practice

This lab follows a simplified version of real project practice.

In a real project, engineers and analysts usually:

1. clarify the business objective;
2. understand where the data came from;
3. inspect the schema and row meaning;
4. check missing values, duplicates, distributions, and unusual values;
5. decide which columns are features and which column is the target;
6. split data before learned preprocessing;
7. compare against a baseline;
8. evaluate using metrics that match the problem;
9. inspect errors, not only overall scores;
10. explain limitations clearly.

This lab keeps the implementation simple and visible. The goal is to review the concepts covered in class and understand the reasoning behind each step.

**Question 25 — Real-world limitation**

Why is a high test score not enough to declare this system ready for medical use?

<details>
<summary>Solution</summary>

Medical use requires much more than a classroom test score.

The system would need careful validation, expert review, ethical review, monitoring, documentation, and safety processes. This lab teaches machine-learning workflow; it does not create a deployable medical system.

</details>

## 7. Final Concept Review and Exam-Style Questions

Answer these before opening the solutions.

### Q1. Baseline Interpretation

The most-frequent baseline has accuracy `0.632` and malignant-class recall `0.000`.

Why can the accuracy look reasonable while the model is useless for identifying malignant cases?

<details>
<summary>Answer</summary>

The baseline always predicts the majority class. If benign cases are more common, it can get many benign cases correct and still never identify malignant cases.

The malignant-class recall is `0.000`, which means it found none of the actual malignant cases. This is why accuracy alone is not enough.

</details>

### Q2. Metric Choice in a Medical-Support Context

For malignant cases, which metric is especially important if the goal is to avoid missing malignant cases, and why?

<details>
<summary>Answer</summary>

Recall for the malignant class is especially important.

It asks: of all truly malignant cases, how many did the model identify as malignant? A low malignant recall means many malignant cases were predicted as benign.

</details>

### Q3. Confusion Matrix Calculation

A classifier produces this malignant-focused confusion summary:

```text
Actual malignant predicted malignant: 40
Actual malignant predicted benign:     3
Actual benign predicted malignant:     2
Actual benign predicted benign:       69
```

Calculate malignant precision and malignant recall.

<details>
<summary>Answer</summary>

Malignant precision:

```text
40 / (40 + 2) = 40 / 42 ≈ 0.952
```

Malignant recall:

```text
40 / (40 + 3) = 40 / 43 ≈ 0.930
```

</details>

### Q4. Precision-Recall Tradeoff

Two models have the following malignant-class scores:

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| A | 0.99 | 0.82 | 0.90 |
| B | 0.94 | 0.96 | 0.95 |

Which model is more appropriate if missing malignant cases is the larger concern?

<details>
<summary>Answer</summary>

Model B is more appropriate because it has higher malignant recall.

Model A has higher precision, but it misses more actual malignant cases. If false negatives are the larger concern, recall should receive special attention.

</details>

### Q5. Data Leakage Scenario

Someone standardizes all rows first, then performs the train/test split. Explain why this is a problem and how to fix it.

<details>
<summary>Answer</summary>

This leaks test-set information into preprocessing because the scaler learned means and standard deviations from all rows.

Correct workflow:

1. split into training and test sets;
2. fit the scaler on the training features only;
3. transform the training features;
4. transform the test features using the same fitted scaler.

</details>

### Q6. KNN Scaling Scenario

Suppose KNN is trained on unscaled cancer measurements. Features such as radius and area have much larger numerical ranges than smoothness. What problem may occur?

<details>
<summary>Answer</summary>

KNN uses distances.

Large-scale features can dominate the distance calculation, so the nearest neighbors may be chosen mostly because of those features. Scaling makes the selected features more comparable in the distance calculation.

</details>

### Q7. Decision Tree Versus KNN

Why can a decision tree be trained on the original feature scales while KNN usually needs scaling?

<details>
<summary>Answer</summary>

A decision tree uses threshold splits on individual features. It does not calculate nearest-neighbor distances.

KNN calculates distances between observations, so differences in feature scale strongly affect which observations count as nearest.

</details>

### Q8. Training Accuracy Warning

A decision tree has training accuracy `1.00` and test accuracy `0.82`. What pattern does this suggest?

<details>
<summary>Answer</summary>

This suggests possible overfitting.

The model fits the training data perfectly but performs much worse on unseen test data. The training score alone is not enough to judge generalization.

</details>

### Q9. Probability Interpretation

Logistic regression gives one case `probability_malignant = 0.91`. What can and cannot be concluded?

<details>
<summary>Answer</summary>

We can conclude that the model estimates a high probability for the malignant class based on the learned feature patterns.

We cannot conclude that the case is definitely malignant. A probability is a model estimate, not proof or a medical decision.

</details>

### Q10. Macro Versus Weighted Average

In a classification report, macro F1 is much lower than weighted F1. What can this suggest about class performance?

<details>
<summary>Answer</summary>

It can suggest that the model performs poorly on one or more smaller classes.

The weighted average is influenced more by larger classes, while the macro average gives each class equal weight. A gap between them can reveal uneven class performance.

</details>

### Q11. Clustering Without Diagnosis

Why should `diagnosis` be excluded when fitting K-Means, even though it is useful to compare clusters with diagnosis afterward?

<details>
<summary>Answer</summary>

Including `diagnosis` would make the clusters partly based on the known answer label.

For unsupervised profile discovery, clusters should be fitted using measurement features only. Diagnosis can then be used afterward to interpret whether the measurement-based groups relate to known labels.

</details>

### Q12. Cluster-Diagnosis Crosstab

A cluster contains 90 percent malignant cases. Can we rename that cluster "malignant" and use it as a diagnosis rule?

<details>
<summary>Answer</summary>

No.

The cluster may be associated with malignant cases, but it is not a supervised classifier and may still contain benign cases. Cluster membership is descriptive and depends on selected features, scaling, K, and algorithm.

</details>

### Q13. Choosing K

K=2 has the highest silhouette score, but K=4 gives more detailed groups. What should be considered before choosing K?

<details>
<summary>Answer</summary>

Consider silhouette score, inertia/elbow pattern, group sizes, cluster profiles, visual separation, and whether the groups are interpretable for the project purpose.

The best K is not chosen by one number alone.

</details>

### Q14. Comparing K-Means and Agglomerative Clustering

K-Means and agglomerative clustering produce different cluster assignments for some cases. Why is this not automatically an error?

<details>
<summary>Answer</summary>

They use different clustering logic.

K-Means uses centroids. Agglomerative clustering merges observations or groups according to linkage rules. Different algorithms can reasonably produce different boundaries.

</details>

### Q15. Final System Limitation

Write three limitations that should be communicated with the cancer classification and clustering results.

<details>
<summary>Answer</summary>

Reasonable limitations include:

- high test performance in a classroom dataset does not make a medical device;
- false negatives and false positives have different practical consequences;
- probabilities are model estimates, not proof;
- cluster labels are descriptive, not diagnosis rules;
- performance may change on future data from a different source;
- expert validation and governance would be required in real use.

</details>
