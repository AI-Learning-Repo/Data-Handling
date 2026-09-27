# Theory 2: End-to-End Cancer Review with Classification and Clustering

This theory file supports **Lab 2: End-to-End Cancer Diagnosis Support with Classification and Clustering**.

The purpose is to refresh the main classification ideas from week 3 and connect them to the clustering ideas reviewed in week 5. The lab contains the code. This file explains the reasoning behind the code.

This is a machine-learning education example. It is not medical advice and it is not a deployable medical diagnosis system.

## Table of Contents

- [1. What Makes This End-to-End?](#1-what-makes-this-end-to-end)
- [2. From Business Objective to ML Task](#2-from-business-objective-to-ml-task)
- [3. Understanding the Cancer Dataset](#3-understanding-the-cancer-dataset)
- [4. Classification Review](#4-classification-review)
- [5. Data Inspection and EDA](#5-data-inspection-and-eda)
- [6. Data Quality and Leakage](#6-data-quality-and-leakage)
- [7. Classification System](#7-classification-system)
- [8. Classification Evaluation](#8-classification-evaluation)
- [9. Model Comparison and Generalization](#9-model-comparison-and-generalization)
- [10. Clustering System](#10-clustering-system)
- [11. Clustering Evaluation and Interpretation](#11-clustering-evaluation-and-interpretation)
- [12. Real-World Notes](#12-real-world-notes)
- [13. Concepts to Remember](#13-concepts-to-remember)
- [14. Review Questions](#14-review-questions)

## 1. What Makes This End-to-End?

An end-to-end machine-learning project is more than training a model.

It usually follows a chain of decisions:

```text
Business objective
    ↓
Data source and row meaning
    ↓
Feature and target definition
    ↓
Inspection and EDA
    ↓
Train/test split
    ↓
Preprocessing learned from training data
    ↓
Baseline model
    ↓
Classification models
    ↓
Evaluation and error analysis
    ↓
Interpretation and limitations
```

Lab 2 follows this workflow with beginner-friendly tools.

The real-world framing matters because machine learning is not only about code. Before selecting an algorithm, engineers and analysts need to understand what the system is supposed to support, what data is available, what errors matter, and what the model is not allowed to claim.

## 2. From Business Objective to ML Task

A business question is not automatically a machine-learning task.

For the cancer example, the broad business question is:

> Can tumor measurements support diagnosis review and help discover groups of similar cases?

This becomes two ML questions.

| Question | ML task | Output |
|---|---|---|
| Can we predict the diagnosis class from measurements? | Classification | A class label |
| Which cases have similar measurement profiles? | Clustering | Group assignments |

The distinction matters because the target is used differently.

In classification, the diagnosis is the target. The model learns from examples where the correct diagnosis label is known.

In clustering, the diagnosis is not used during fitting. We choose measurement features that define similarity, then interpret the groups afterward.

## 3. Understanding the Cancer Dataset

Lab 2 uses the scikit-learn breast cancer dataset.

In this dataset, one row represents one case. Each case has numerical measurements computed from a breast mass image.

Examples of features include:

| Feature | General meaning |
|---|---|
| `mean radius` | Average radius measurement |
| `mean texture` | Average texture measurement |
| `mean smoothness` | Average smoothness measurement |
| `mean concavity` | Average concavity measurement |
| `worst radius` | Larger or more extreme radius-related measurement |
| `worst concavity` | Larger or more extreme concavity-related measurement |

The target is:

```text
diagnosis
```

The class labels are:

```text
0 → malignant
1 → benign
```

This coding is important. In many examples, people assume `1` is the positive class. In this lab, we often focus precision, recall, and F1 on class `0` because class `0` means malignant.

## 4. Classification Review

Classification is supervised learning where the target is a category.

Examples:

```text
spam / not spam
disease / no disease
survived / did not survive
malignant / benign
setosa / versicolor / virginica
```

Classification is different from regression.

| Task | Target type | Example output |
|---|---|---|
| Regression | Numerical value | `250000` |
| Classification | Category/class | `malignant` |

### Binary Classification

Binary classification has exactly two classes.

Examples:

```text
malignant / benign
survived / did not survive
spam / not spam
```

The breast cancer lab is binary classification because the diagnosis target has two possible classes.

### Multiclass Classification

Multiclass classification has more than two classes.

The Iris lab from week 3 is multiclass classification:

```text
setosa
versicolor
virginica
```

The overall workflow is similar:

```text
Define X and y
Split the data
Prepare the data
Train a classifier
Predict classes
Evaluate predictions
```

The difference is that a multiclass classifier chooses among more than two class labels, and the confusion matrix has more rows and columns.

## 5. Data Inspection and EDA

Initial inspection answers basic questions:

- How many rows and columns are there?
- What does one row represent?
- Which columns are features?
- Which column is the target?
- Are the values numerical or categorical?
- Are there missing values?
- Are there duplicate rows?
- Are the target classes balanced?

Common pandas tools include:

```python
df.head()
df.shape
df.info()
df.describe()
df.isna().sum()
df.duplicated().sum()
```

EDA then explores patterns.

For Lab 2, useful EDA includes:

- target class counts;
- feature distributions;
- boxplots comparing feature values by diagnosis;
- scatterplots colored by diagnosis;
- correlations among numerical features.

### Class Balance

Class balance means how many observations each class has.

If one class is much more common, accuracy can be misleading.

For example, if 90 percent of cases are benign, a model that always predicts benign could have 90 percent accuracy while failing to identify malignant cases.

This is why we compare against a baseline and inspect precision, recall, F1, and the confusion matrix.

### Correlation in a Classification Lab

Pearson correlation is not the main evaluation tool for classification.

However, it can still help with EDA when the target is encoded numerically.

In Lab 2, `diagnosis` is coded as `0` and `1`, so correlation with `diagnosis` gives an initial view of linear association with the class code.

This should not be treated as final feature selection. A classifier may use combinations of features, and a low single-feature correlation does not prove a feature is useless.

## 6. Data Quality and Leakage

Data quality checks help prevent hidden problems.

Examples:

- missing values;
- duplicate rows;
- unexpected data types;
- target leakage;
- train/test contamination.

### Data Leakage

Data leakage occurs when information that should not be available during training influences the model.

Examples:

- using the target as a feature;
- fitting a scaler on the full dataset before train/test split;
- letting test-set information influence preprocessing;
- evaluating on data that the model effectively saw during training.

In Lab 2, we split the data before fitting `StandardScaler`.

The correct pattern is:

```text
Fit scaler on training data
Transform training data
Transform test data using the same scaler
```

We do not fit a separate scaler on the test set.

## 7. Classification System

Lab 2 compares several classifiers.

The goal is not to prove that one algorithm is always best. The goal is to understand how different classifiers work and how to evaluate them.

### Baseline Classifier

A baseline classifier is a simple reference strategy.

In Lab 2, the baseline predicts the most frequent training class.

The baseline does not learn useful feature relationships. It answers:

> How well could we do with a trivial strategy?

A real classifier should be evaluated relative to this simple reference.

### Logistic Regression

Despite its name, logistic regression is a classification algorithm.

For binary classification, it estimates class probabilities.

Conceptually:

```text
measurements → estimated probability → class prediction
```

For example:

```text
probability_malignant = 0.90
```

This means the model estimates a high probability for the malignant class. It is not proof. It is a model estimate.

### K-Nearest Neighbors

K-nearest neighbors, or KNN, classifies a new observation by looking at nearby training observations.

If `k = 5`, the model looks at the five nearest training cases and predicts based on their classes.

KNN depends on distance.

This is why scaling matters. If one feature has much larger numerical values than another feature, it can dominate the distance calculation.

### Decision Tree

A decision tree makes predictions by asking a sequence of feature-based questions.

Conceptually:

```text
Is worst radius <= some threshold?
    ↓
Is worst concavity <= some threshold?
    ↓
Predict class
```

Decision trees are useful for learning because their decision process is easier to visualize than many other models.

In Lab 2, `max_depth=4` limits tree complexity. A very deep tree may fit training data too closely.

## 8. Classification Evaluation

Classification predictions are class labels, so regression metrics such as MAE and RMSE are not appropriate.

Lab 2 uses:

| Metric/tool | Meaning |
|---|---|
| Accuracy | Overall proportion correct |
| Precision | Correctness among predicted positives |
| Recall | Coverage among actual positives |
| F1 | Balance between precision and recall |
| Confusion matrix | Counts of actual versus predicted classes |
| Classification report | Per-class metrics |

### Accuracy

Accuracy is:

```text
number of correct predictions / total number of predictions
```

Accuracy is useful, but it can hide important mistakes.

If classes are imbalanced, a high accuracy may mostly reflect performance on the majority class.

### Confusion Matrix

A confusion matrix compares actual classes with predicted classes.

For binary classification:

```text
                  Predicted class
                class 0   class 1
Actual class 0    ...       ...
Actual class 1    ...       ...
```

The diagonal contains correct predictions.

Off-diagonal cells contain mistakes.

In Lab 2:

```text
0 → malignant
1 → benign
```

So an actual malignant case predicted as benign is a serious error to notice.

### Precision

Precision asks:

```text
Of the cases predicted as a class, how many were actually that class?
```

For malignant precision:

```text
Of the cases predicted malignant, how many were actually malignant?
```

High precision means the model's positive predictions are often correct.

### Recall

Recall asks:

```text
Of the actual cases in a class, how many did the model find?
```

For malignant recall:

```text
Of the truly malignant cases, how many did the model identify as malignant?
```

High recall means the model misses fewer actual cases of that class.

### F1

F1 combines precision and recall.

It is useful when we want a single metric that considers both false positives and false negatives.

A high F1 requires both precision and recall to be reasonably high.

### Positive Class

Many metric functions require us to identify the positive class.

In Lab 2, we focus some metrics on:

```text
pos_label=0
```

because class `0` is malignant.

The positive class is not always `1`. It depends on the problem and what error we are studying.

### Macro and Weighted Averages

Lab 2 is binary, but week 3 also covered multiclass classification.

For multiclass classification, the classification report may show:

```text
macro avg
weighted avg
```

Macro average calculates the metric separately for each class and gives each class equal importance.

Weighted average also calculates class-level metrics, but gives more weight to classes with more observations.

If classes are balanced, macro and weighted averages may be similar. If one class is much larger, the weighted average is strongly influenced by the larger class.

## 9. Model Comparison and Generalization

A model should perform well on new observations, not only on training data.

This is why Lab 2 uses a train/test split.

Training data is used to fit the model.

Test data is held back for evaluation.

### Why Compare Models?

Lab 2 compares:

- a most-frequent baseline;
- logistic regression;
- KNN;
- decision tree.

We compare models using the same test set and the same metrics.

Do not automatically choose the model with the highest accuracy. Consider:

- Does it improve meaningfully over the baseline?
- What is its recall for the important class?
- What is its precision?
- What is its F1 score?
- What mistakes appear in the confusion matrix?
- Is the model understandable enough for the purpose?

### Training Accuracy Is Not Enough

A classifier with high training accuracy is not automatically good.

It may be overfitting: learning patterns that do not generalize to new observations.

Test performance is more useful for estimating how the model may behave on unseen data.

## 10. Clustering System

Clustering changes the question.

Instead of predicting diagnosis, we ask:

> Which cases have similar tumor measurement profiles?

Lab 2 uses selected numerical measurements such as:

- `mean radius`;
- `mean texture`;
- `mean smoothness`;
- `mean concavity`;
- `worst radius`;
- `worst concavity`.

It keeps `diagnosis` only for interpretation after clustering.

### Why Diagnosis Is Not a Clustering Feature

If we use `diagnosis` as a clustering feature, the cluster algorithm receives the answer label.

That would no longer be unsupervised profile discovery.

The correct idea is:

```text
Fit clusters using measurements only
Compare clusters with diagnosis afterward
```

### Why Scaling Matters

K-Means and agglomerative clustering use distances.

Features with larger numerical ranges can dominate the distance calculation.

Standardization puts features on comparable scales:

```text
mean ≈ 0
standard deviation ≈ 1
```

This does not prove the clusters are meaningful. It only makes the distance calculation more balanced for the selected features.

### K-Means

K-Means:

1. chooses K cluster centers;
2. assigns observations to the nearest center;
3. updates centers;
4. repeats until the algorithm stops.

Cluster labels are arbitrary.

Cluster `0` is not automatically malignant or benign. We need to inspect the cluster profile.

### Agglomerative Clustering

Agglomerative clustering starts with individual observations, then repeatedly merges observations or groups.

Lab 2 uses it as a second introductory clustering method so students can compare whether a different algorithm gives a similar grouping pattern.

## 11. Clustering Evaluation and Interpretation

Clustering has no supplied correct answer in this lab.

We therefore use several kinds of evidence:

- inertia;
- silhouette score;
- cluster sizes;
- cluster profiles;
- scatterplots;
- comparison with diagnosis after fitting;
- interpretability.

### Inertia

Inertia measures total squared distance from observations to their assigned cluster centers.

Lower inertia usually occurs when K increases.

Therefore, we should not simply choose the largest K because it has the lowest inertia.

### Silhouette

Silhouette compares how close observations are to their own cluster versus the nearest other cluster.

Values closer to 1 suggest better separation.

Values near 0 suggest overlapping clusters.

Negative values suggest some observations may be closer to another cluster.

### Cluster Profiles

A cluster profile summarizes the average feature values inside each cluster.

For example, one cluster may have larger average radius and concavity values than another cluster.

Profiles help us describe clusters in plain language.

### Comparing Clusters with Diagnosis

After clustering, Lab 2 compares clusters with known diagnosis labels.

This is useful for interpretation:

```text
Do the measurement-based clusters relate to diagnosis?
```

But it has limits.

Clusters are not supervised predictions. A cluster can contain a mix of diagnoses, and cluster membership should not be treated as a medical decision.

## 12. Real-World Notes

Lab 2 keeps the implementation inside the concepts covered in class.

In real projects, engineers would also think about:

- where the data came from;
- whether the data represents the population where the model may be used;
- whether labels are reliable;
- what false positives and false negatives cost;
- whether the model should be used only as decision support;
- how results would be reviewed by domain experts;
- whether performance changes over time.

These are important real-world ideas, but they are not required lab mechanics here.

The lab deliberately excludes required implementation of:

- scikit-learn `Pipeline`;
- `ColumnTransformer`;
- custom transformers;
- grid search;
- random search;
- cross-validation as a required step;
- deployment monitoring code.

Those topics can be discussed as context later, but the required student work stays within the week 2-5 concepts.

## 13. Concepts to Remember

### Classification

Supervised learning where the target is a class label.

### Binary Classification

Classification with exactly two classes.

### Multiclass Classification

Classification with more than two classes.

### Feature

An input variable used by the model.

### Target

The value or class the supervised model learns to predict.

### Baseline

A simple reference prediction strategy.

### Train/Test Split

Separating data used for fitting from data used for evaluation.

### Data Leakage

When evaluation information influences training or preprocessing.

### Standardization

Transforming features to comparable scales, often with mean 0 and standard deviation 1.

### Logistic Regression

A classification algorithm that estimates class probabilities.

### KNN

A classifier that predicts using nearby training observations.

### Decision Tree

A classifier that makes predictions using a sequence of feature-based splits.

### Accuracy

Overall proportion of correct predictions.

### Precision

Among predicted positives, the proportion that are truly positive.

### Recall

Among actual positives, the proportion found by the model.

### F1

A metric that combines precision and recall.

### Confusion Matrix

A table comparing actual classes with predicted classes.

### Macro Average

Average of class-level metrics giving each class equal importance.

### Weighted Average

Average of class-level metrics weighted by class size.

### Clustering

Unsupervised learning that groups observations using feature similarity.

### Inertia

K-Means compactness measure based on squared distances to centroids.

### Silhouette

Measure comparing within-cluster closeness to nearest competing cluster.

## 14. Review Questions

### Question 1

Why is the cancer lab a classification problem?

<details>
<summary>Suggested answer</summary>

The target is a category: malignant or benign. The model predicts a class label, not a numerical value.

</details>

### Question 2

Why is it binary classification?

<details>
<summary>Suggested answer</summary>

There are exactly two target classes: malignant and benign.

</details>

### Question 3

What is the difference between classification and regression?

<details>
<summary>Suggested answer</summary>

Classification predicts a category or class label. Regression predicts a numerical value.

</details>

### Question 4

Why do we need a baseline classifier?

<details>
<summary>Suggested answer</summary>

A baseline gives a simple reference point. It helps us decide whether a real classifier has learned useful patterns beyond a trivial strategy.

</details>

### Question 5

Why can accuracy be misleading?

<details>
<summary>Suggested answer</summary>

Accuracy can hide which class is being misclassified. If one class is much more common, a model can have high accuracy while performing poorly on the less common or more important class.

</details>

### Question 6

What does precision for malignant cases mean?

<details>
<summary>Suggested answer</summary>

It means: among cases predicted as malignant, what proportion were actually malignant?

</details>

### Question 7

What does recall for malignant cases mean?

<details>
<summary>Suggested answer</summary>

It means: among cases that were actually malignant, what proportion did the model identify as malignant?

</details>

### Question 8

What does F1 measure?

<details>
<summary>Suggested answer</summary>

F1 combines precision and recall into one metric. It is useful when we care about balancing both types of performance.

</details>

### Question 9

Why is a confusion matrix useful?

<details>
<summary>Suggested answer</summary>

It shows which classes were predicted correctly and which classes were confused with each other. It reveals error types that accuracy alone can hide.

</details>

### Question 10

In this lab, why do we sometimes use `pos_label=0`?

<details>
<summary>Suggested answer</summary>

Because class `0` is malignant in this dataset. We use `pos_label=0` when we want precision, recall, and F1 to focus on malignant cases.

</details>

### Question 11

Why should scaling be fitted on the training data only?

<details>
<summary>Suggested answer</summary>

The scaler learns means and standard deviations. If it learns them from the full dataset, test-set information influences preprocessing. That is data leakage.

</details>

### Question 12

Why does KNN need scaling?

<details>
<summary>Suggested answer</summary>

KNN uses distances. If features have very different numerical scales, a large-scale feature can dominate the distance calculation.

</details>

### Question 13

How does a decision tree make predictions?

<details>
<summary>Suggested answer</summary>

A decision tree follows a sequence of feature-based splits. Each split sends an observation down a branch until it reaches a leaf with a predicted class.

</details>

### Question 14

Why is logistic regression a classifier?

<details>
<summary>Suggested answer</summary>

Despite its name, logistic regression estimates class probabilities and converts them into class predictions.

</details>

### Question 15

What is the difference between macro and weighted averages?

<details>
<summary>Suggested answer</summary>

Macro average gives each class equal importance. Weighted average gives more influence to classes with more observations.

</details>

### Question 16

Why should `diagnosis` not be used as a clustering feature?

<details>
<summary>Suggested answer</summary>

`diagnosis` is the known label. If we include it in clustering, the clusters are partly based on the answer. For unsupervised profile discovery, we use measurement features only.

</details>

### Question 17

What does inertia measure in K-Means?

<details>
<summary>Suggested answer</summary>

Inertia measures the total squared distance between observations and their assigned cluster centers.

</details>

### Question 18

Why is the lowest inertia not enough to choose K?

<details>
<summary>Suggested answer</summary>

Inertia usually decreases as K increases. A larger K can always make clusters more compact, so we also need interpretability and other evidence such as silhouette score.

</details>

### Question 19

What does silhouette score tell us?

<details>
<summary>Suggested answer</summary>

It compares how close observations are to their own cluster versus the nearest other cluster. Higher values suggest better separated clusters under the chosen features and distance measure.

</details>

### Question 20

Why are cluster labels arbitrary?

<details>
<summary>Suggested answer</summary>

Cluster numbers are names assigned by the algorithm. Cluster `0` is not automatically malignant, benign, better, or worse. We must inspect the cluster profiles.

</details>

### Question 21

Why is comparing clusters with diagnosis useful but limited?

<details>
<summary>Suggested answer</summary>

It is useful because it helps us interpret whether measurement-based groups relate to known diagnosis. It is limited because clustering is not supervised prediction, and clusters can contain mixed diagnoses.

</details>

### Question 22

Why is Lab 2 not a medical diagnosis system?

<details>
<summary>Suggested answer</summary>

It is a classroom learning system. A real medical system would require expert validation, safety review, documentation, monitoring, and appropriate clinical governance.

</details>
