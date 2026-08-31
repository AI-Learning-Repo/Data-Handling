# Lab 2: Iris Multiclass Classification

**Classifying Iris Species with Multiclass Classification**

---

# Lab Purpose

In Lab 1, we used the Titanic dataset to introduce **binary classification**.

The Titanic task had two possible outcomes:

```text
0 → Did not survive
1 → Survived
```

In this lab, we move to a classification problem with **more than two possible classes**.

We will use the Iris dataset to predict the species of an iris flower:

```text
setosa
versicolor
virginica
```

Because there are three possible classes, this is a:

> **multiclass classification problem**

The important idea is that the overall machine-learning workflow has not changed.

```text
Lab 1: Titanic                  Lab 2: Iris

Binary classification           Multiclass classification
        ↓                                ↓
Define X and y                  Define X and y
        ↓                                ↓
Split the data                  Split the data
        ↓                                ↓
Preprocess                       Preprocess
        ↓                                ↓
Train classifier                 Train classifier
        ↓                                ↓
Evaluate                         Evaluate
        ↓                                ↓
Compare models                   Compare models
        ↓                                ↓
Tune                             Tune
```

The main difference is the **number of possible classes**.

---

# Learning Outcomes

By the end of this lab, you should be able to:

* explain the difference between binary and multiclass classification
* identify the target and features in the Iris dataset
* explain why Iris is a multiclass classification problem
* establish a simple multiclass baseline
* understand the basic idea behind K-nearest neighbors (KNN)
* explain why KNN can be affected by feature scale
* train a KNN classifier
* evaluate a multiclass classifier using:

  * accuracy
  * confusion matrix
  * precision
  * recall
  * F1 score
* interpret a multiclass confusion matrix
* understand per-class evaluation
* explain macro and weighted averages
* use cross-validation for model evaluation
* use logistic regression for multiclass classification
* compare two different classifiers
* investigate classification errors
* tune KNN hyperparameters
* evaluate a final model on a held-out test set
* explain how multiclass classification differs from binary classification

---

# 1. Connect Lab 1 and Lab 2

Before working with the Iris dataset, compare the two classification problems.

|                       | Lab 1: Titanic                      | Lab 2: Iris                                                    |
| --------------------- | ----------------------------------- | -------------------------------------------------------------- |
| Task                  | Survival prediction                 | Species prediction                                             |
| Type                  | Binary classification               | Multiclass classification                                      |
| Number of classes     | 2                                   | 3                                                              |
| Features              | Numerical + categorical             | Numerical                                                      |
| Missing values        | Present                             | Essentially none                                               |
| Main preprocessing    | Imputation + encoding               | Minimal preprocessing                                          |
| Baseline              | Most frequent class                 | Most frequent class                                            |
| First classifier      | Logistic regression                 | KNN                                                            |
| Model comparison      | Logistic regression + decision tree | KNN + logistic regression                                      |
| Main evaluation focus | Precision, recall, F1               | Per-class metrics + confusion matrix + macro/weighted averages |

The purpose of this lab is **not** to learn a completely different machine-learning workflow.

Instead, we are extending what you learned in Lab 1.

### The key question

> What changes when a classification problem has **three classes instead of two**?

---

# 2. What Is Multiclass Classification?

In binary classification, there are two possible classes.

For Titanic:

```text
0 → Did not survive
1 → Survived
```

In multiclass classification, there are more than two possible classes.

For Iris:

```text
0 → setosa
1 → versicolor
2 → virginica
```

The model must choose one of three possible classes.

Conceptually:

```text
Flower measurements
        ↓
    Classifier
        ↓
 ┌──────┼────────┐
 ↓      ↓        ↓
setosa  versicolor  virginica
```

This is different from binary classification because there is no single positive and negative class in the same sense as Titanic.

In Titanic we could say:

```text
positive = survived
negative = did not survive
```

With Iris, every species is a class that can be predicted.

---

# 3. The Iris Dataset

The Iris dataset contains measurements of iris flowers.

Each observation contains four measurements:

* sepal length
* sepal width
* petal length
* petal width

The target is the flower species.

There are three species:

```text
setosa
versicolor
virginica
```

The classic Iris dataset contains 150 observations:

```text
50 setosa
50 versicolor
50 virginica
```

The classes are therefore balanced.

This is useful when learning classification because accuracy is less likely to be misleading due to a large difference in class sizes.

---

# 4. Environment Setup

If necessary, install the required libraries:

```python
!pip install pandas scikit-learn matplotlib seaborn -q
```

Import the libraries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
```

---

# 5. Load the Dataset

Load Iris from scikit-learn.

```python
iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = pd.Series(
    iris.target,
    name="species"
)

display(X.head())
display(y.head())

print("X shape:", X.shape)
print("y shape:", y.shape)
```

We can also display the class names:

```python
print("Target classes:")
print(iris.target_names)
```

The numerical target values correspond to:

```text
0 → setosa
1 → versicolor
2 → virginica
```

---

# 6. Briefly Validate the Dataset

As in Lab 1, we should perform a small validation check before modeling.

Run:

```python
print("Feature columns:")
print(X.columns.tolist())

print("\nData types:")
print(X.dtypes)

print("\nMissing values:")
print(X.isnull().sum())

print("\nClass distribution:")
print(y.value_counts().sort_index())

print("\nFeature summary:")
display(X.describe())
```

### What should you notice?

You should find that:

* there are four features
* all four features are numerical
* there are 150 observations
* there are three target classes
* each class has 50 observations
* there are no important missing values

This is considerably simpler than the Titanic dataset.

---

# 7. Understand the Features

The four measurements are:

```text
sepal length
sepal width
petal length
petal width
```

Unlike Titanic, there are no categorical variables that need one-hot encoding.

We therefore do not need the more complicated `ColumnTransformer` used in Lab 1.

This is an important lesson:

> **Preprocessing should depend on the data and the model. We do not automatically apply every preprocessing technique to every dataset.**

### Questions

1. How many observations are in the dataset?
2. How many features are there?
3. How many classes are there?
4. Are the features numerical or categorical?
5. Are there important missing values?
6. Is the target balanced?

---

# 8. Understand the Classification Task

Our features are:

```text
X → flower measurements
```

Our target is:

```text
y → flower species
```

For example:

```text
sepal length = 5.1
sepal width  = 3.5
petal length = 1.4
petal width  = 0.2
        ↓
   Classifier
        ↓
     setosa
```

### Task

Write one sentence explaining what `X` and `y` represent in this problem.

Then answer:

> Why is Iris a multiclass classification problem rather than a binary classification problem?

---

# 9. Split the Data

As in Lab 1, we need to separate training data from test data.

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

print("\nTraining class distribution:")
print(y_train.value_counts(normalize=True).sort_index())

print("\nTest class distribution:")
print(y_test.value_counts(normalize=True).sort_index())
```

We use:

```python
stratify=y
```

so that the three classes remain approximately equally represented in the training and test sets.

### Questions

1. Why do we split the data before training?
2. Why is stratification useful here?
3. Why should the test set remain untouched during model development?

---

# 10. Establish a Multiclass Baseline

Before training a real classifier, establish a simple baseline.

```python
dummy_model = DummyClassifier(
    strategy="most_frequent"
)

dummy_model.fit(
    X_train,
    y_train
)

y_dummy_pred = dummy_model.predict(X_test)

baseline_accuracy = accuracy_score(
    y_test,
    y_dummy_pred
)

print("Baseline accuracy:", baseline_accuracy)
```

Because the Iris classes are balanced, a classifier that always predicts one class will achieve approximately:

```text
1 / 3 ≈ 33%
```

accuracy.

The exact value can vary slightly depending on the split.

### Why is the baseline useful?

Suppose:

```text
Baseline accuracy = 0.33
Model accuracy    = 0.96
```

The real classifier has made a substantial improvement over the simple strategy.

The baseline gives us a reference point.

### Important

A real classifier does **not** have to outperform the baseline on every individual metric or every random test split.

Instead, we ask:

> Does the model provide meaningful improvement over a simple strategy, and is that improvement supported by appropriate evaluation?

### Questions

1. What is the baseline classifier doing?
2. Why is its accuracy approximately one-third?
3. Why is a baseline useful?
4. Why should we not simply say that the real model "must" beat the baseline in every situation?

---

# 11. First Classifier: K-Nearest Neighbors

Our first real classifier will be **K-nearest neighbors**, or KNN.

The basic idea is simple:

> A new observation is classified according to the classes of nearby observations.

Suppose we have a new flower.

KNN looks for the nearest flowers in the training data and examines their species.

For example, with:

```text
k = 5
```

the algorithm looks at the five nearest training observations.

If the nearest five flowers are:

```text
setosa
setosa
setosa
versicolor
setosa
```

then the predicted class would be:

```text
setosa
```

because it is the majority class among the nearest neighbors.

---

# 12. Why KNN Is Useful for Iris

KNN is a useful first model for Iris because:

* the dataset is small
* the features are numerical
* the observations can be compared using distances
* the species have measurable differences in their flower dimensions
* the algorithm is relatively easy to understand

KNN also provides a useful contrast with logistic regression.

---

# 13. A Note About Feature Scaling

KNN is a **distance-based** algorithm.

This means that the numerical scale of the features can affect its predictions.

For example, imagine two features:

```text
Feature A → values between 0 and 5
Feature B → values between 0 and 500
```

Feature B could have a much larger influence on the distance simply because its numerical values are larger.

Standardization is one common solution.

However, in this introductory lab, we will **not use `StandardScaler()` in the main workflow**.

The Iris measurements are all reasonably interpretable numerical measurements, and we want to keep the focus on multiclass classification.

This is a deliberate simplification, not a statement that scaling is never useful.

### Optional experiment

If you want to investigate preprocessing, try adding `StandardScaler()` later and compare the results.

The question to investigate is:

> Does scaling change KNN's performance on Iris?

---

# 14. Train the KNN Classifier

We will begin with:

```text
k = 5
```

```python
knn_model = KNeighborsClassifier(
    n_neighbors=5
)

knn_model.fit(
    X_train,
    y_train
)

y_knn_pred = knn_model.predict(
    X_test
)
```

---

# 15. Evaluate KNN Accuracy

Start with accuracy.

```python
knn_accuracy = accuracy_score(
    y_test,
    y_knn_pred
)

print("KNN accuracy:", knn_accuracy)
```

Accuracy is reasonable here because the three Iris classes are balanced.

However, we should not stop with accuracy.

A multiclass classifier can have high overall accuracy while performing differently for different classes.

---

# 16. Evaluate Precision, Recall, and F1

Calculate the metrics for each class.

```python
print(
    classification_report(
        y_test,
        y_knn_pred,
        target_names=iris.target_names
    )
)
```

The classification report provides metrics for:

```text
setosa
versicolor
virginica
```

as well as summary averages.

---

# 17. Understand Per-Class Metrics

In Lab 1, we talked about:

```text
positive class = survived
negative class = did not survive
```

With Iris, we do not have just one positive class.

Instead, we can evaluate each species separately.

For example:

> How well does the classifier identify **setosa**?

Then:

> How well does it identify **versicolor**?

And:

> How well does it identify **virginica**?

For each class, precision, recall, and F1 can be calculated by treating that class as the class of interest and the other classes as the alternatives.

For example, when evaluating `setosa`:

```text
setosa       → class of interest
versicolor   → other class
virginica    → other class
```

### Questions

1. Which species has the highest precision?
2. Which species has the highest recall?
3. Which species has the highest F1 score?
4. Are the three classes performing equally well?
5. Why might one species be easier to classify than another?

---

# 18. Multiclass Confusion Matrix

In Lab 1, the confusion matrix was:

```text
2 × 2
```

because there were two classes.

For Iris, there are three classes, so the confusion matrix is:

```text
3 × 3
```

Create it:

```python
cm = confusion_matrix(
    y_test,
    y_knn_pred
)

print(cm)
```

Visualize it with class names:

```python
plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.title("Confusion Matrix: KNN on Iris")
plt.xlabel("Predicted species")
plt.ylabel("Actual species")
plt.show()
```

The structure is:

```text
                    Predicted
              setosa  versicolor  virginica

Actual
setosa
versicolor
virginica
```

The diagonal represents correct predictions.

The off-diagonal cells represent mistakes.

---

# 19. Interpret the Confusion Matrix

Look carefully at the confusion matrix.

Answer:

1. How many `setosa` flowers were correctly classified?
2. How many `versicolor` flowers were correctly classified?
3. How many `virginica` flowers were correctly classified?
4. Which species is most frequently confused with another?
5. Is `setosa` easier to distinguish than the other species?
6. Which two species appear to overlap more?

### Important idea

A confusion matrix tells us **where the model makes mistakes**.

Accuracy tells us:

> How many predictions were correct overall?

The confusion matrix tells us:

> **Which classes are being confused?**

This becomes particularly important in multiclass classification.

---

# 20. Macro Average vs Weighted Average

The classification report includes summary averages.

Two important ones are:

* macro average
* weighted average

## Macro Average

The macro average gives every class equal importance.

Conceptually:

```text
Macro F1 =
(F1 setosa + F1 versicolor + F1 virginica) / 3
```

Each class receives equal weight.

---

## Weighted Average

The weighted average takes the number of observations in each class into account.

A class with more observations has more influence on the final score.

Because Iris has:

```text
50 setosa
50 versicolor
50 virginica
```

the classes are balanced.

Therefore, macro and weighted averages should be relatively similar.

### Why does this matter?

On an imbalanced dataset, the two averages can be noticeably different.

For example:

```text
Class A → 90 observations
Class B → 8 observations
Class C → 2 observations
```

A weighted average will be strongly influenced by Class A.

A macro average gives all three classes equal importance.

### Questions

1. What does the macro average represent?
2. What does the weighted average represent?
3. Why are the two averages relatively similar for Iris?
4. On an imbalanced dataset, why might macro average be useful?

---

# 21. Cross-Validation

As in Lab 1, we should not rely only on one train/test split.

We can use cross-validation on the training data.

Because the Iris classes are balanced, we will use accuracy as the scoring metric.

```python
cv_scores = cross_val_score(
    knn_model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("Cross-validation accuracy scores:")
print(cv_scores)

print("\nMean CV accuracy:")
print(cv_scores.mean())

print("\nStandard deviation:")
print(cv_scores.std())
```

### Connection to Lab 1

In Lab 1, we used F1 for cross-validation.

Here we use accuracy because:

* Iris has balanced classes
* accuracy is easy to interpret
* we want to focus on the multiclass workflow

The choice of evaluation metric should depend on the problem.

### Questions

1. Why is cross-validation useful?
2. Why might the mean CV accuracy differ from the test accuracy?
3. What does the standard deviation tell us?
4. Why are we using accuracy here instead of F1?

---

# 22. Logistic Regression: The Name Can Be Confusing

Now we will use another classifier:

**Logistic Regression**

The name can be confusing.

You might see the word:

> regression

and assume that the algorithm is used to predict a continuous numerical value such as:

```text
house price
temperature
salary
```

That is **not** what logistic regression is doing here.

### Important

> **Logistic regression is a classification algorithm.**

It is commonly used to estimate the probability that an observation belongs to a class and then use those probabilities to make a classification decision.

In Lab 1, we used logistic regression for:

```text
2 classes
↓
binary classification
```

In this lab, logistic regression can be used for:

```text
3 classes
↓
multiclass classification
```

So the important lesson is:

> The name "logistic regression" contains the word regression, but the algorithm is widely used for classification.

---

# 23. Train Logistic Regression on Iris

We can train logistic regression directly on the Iris data.

```python
logreg_model = LogisticRegression(
    max_iter=1000
)

logreg_model.fit(
    X_train,
    y_train
)

y_logreg_pred = logreg_model.predict(
    X_test
)
```

Notice that we did not have to change the target into a binary problem.

The classifier can work with all three Iris classes.

---

# 24. Evaluate Logistic Regression

Calculate accuracy:

```python
logreg_accuracy = accuracy_score(
    y_test,
    y_logreg_pred
)

print("Logistic regression accuracy:", logreg_accuracy)
```

Now examine the classification report:

```python
print(
    classification_report(
        y_test,
        y_logreg_pred,
        target_names=iris.target_names
    )
)
```

### Questions

1. Which species does logistic regression classify best?
2. Which species is most difficult?
3. How does logistic regression compare with KNN?
4. Are the errors made by the two models identical?

---

# 25. Compare KNN and Logistic Regression

Create a simple comparison table.

```python
comparison = pd.DataFrame({
    "Model": [
        "Baseline",
        "KNN",
        "Logistic Regression"
    ],
    "Accuracy": [
        baseline_accuracy,
        knn_accuracy,
        logreg_accuracy
    ]
})

display(comparison)
```

### Questions

1. Which model has the highest accuracy?
2. How much better is each real model than the baseline?
3. Do KNN and logistic regression perform similarly?
4. Is accuracy alone enough to explain the difference between the models?
5. Which model would you choose at this stage, and why?

---

# 26. Compare the Confusion Matrices

Create a confusion matrix for logistic regression.

```python
cm_logreg = confusion_matrix(
    y_test,
    y_logreg_pred
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_logreg,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.title("Confusion Matrix: Logistic Regression on Iris")
plt.xlabel("Predicted species")
plt.ylabel("Actual species")
plt.show()
```

Compare this with the KNN confusion matrix.

### Questions

1. Do the two models make the same errors?
2. Which species is most frequently confused?
3. Does one model appear to separate any particular pair of species better?
4. What does the confusion matrix tell you that accuracy does not?

---

# 27. Error Analysis

A useful machine-learning habit is to examine the observations that the model classified incorrectly.

Let's find KNN's errors.

```python
incorrect = X_test[
    y_knn_pred != y_test
].copy()

incorrect["actual"] = y_test[
    y_knn_pred != y_test
].map(
    dict(enumerate(iris.target_names))
)

incorrect["predicted"] = pd.Series(
    y_knn_pred[
        y_knn_pred != y_test
    ],
    index=incorrect.index
).map(
    dict(enumerate(iris.target_names))
)

display(incorrect)
```

Now examine how many errors occurred for each actual/predicted combination:

```python
errors = pd.crosstab(
    incorrect["actual"],
    incorrect["predicted"]
)

display(errors)
```

### Questions

1. How many flowers were misclassified?
2. Which species was most frequently misclassified?
3. Which species was it confused with?
4. Look at the measurements of some incorrectly classified flowers. Why might they be difficult to classify?
5. Do the measurements of `versicolor` and `virginica` appear more similar than those of `setosa`?

### Important idea

Model evaluation should not stop at:

> "The accuracy is 96%."

We can also ask:

> **Where does the model struggle, and why?**

Errors can help us understand both the dataset and the model.

---

# 28. KNN Hyperparameters

KNN has several hyperparameters.

One important parameter is:

```python
n_neighbors
```

This determines how many nearby observations are considered when making a prediction.

For example:

```text
n_neighbors = 3
```

means that KNN looks at the three nearest observations.

```text
n_neighbors = 9
```

means that it looks at the nine nearest observations.

A smaller value can make the model more sensitive to local patterns.

A larger value can produce smoother decisions.

---

## Uniform vs Distance Weighting

KNN can also decide how much influence each neighbor has.

### Uniform weighting

```text
weights="uniform"
```

Every neighbor contributes equally.

### Distance weighting

```text
weights="distance"
```

Closer neighbors have more influence than farther neighbors.

---

# 29. Tune KNN

We will test a small number of combinations.

```python
param_grid = {
    "n_neighbors": [
        3,
        5,
        7,
        9
    ],
    "weights": [
        "uniform",
        "distance"
    ]
}
```

Use cross-validation on the training data:

```python
grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)

grid_search.fit(
    X_train,
    y_train
)
```

Inspect the best result:

```python
print("Best parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation accuracy:")
print(grid_search.best_score_)
```

### Questions

1. What value of `n_neighbors` was selected?
2. Was `uniform` or `distance` weighting selected?
3. Did tuning improve the cross-validation result?
4. Why should the test set not be used to choose these parameters?

---

# 30. Examine the Tuning Results

We can inspect all tested combinations.

```python
results = pd.DataFrame(
    grid_search.cv_results_
)

display(
    results[
        [
            "param_n_neighbors",
            "param_weights",
            "mean_test_score",
            "std_test_score",
            "rank_test_score"
        ]
    ].sort_values("rank_test_score")
)
```

This allows us to see not only the best combination but also how the other combinations performed.

### Question

Do several hyperparameter settings perform similarly?

If so, what does that suggest about how sensitive KNN is to these settings on this dataset?

---

# 31. Final Evaluation

Now use the best KNN model selected using cross-validation.

```python
best_knn = grid_search.best_estimator_

y_final_pred = best_knn.predict(
    X_test
)
```

Calculate the final accuracy:

```python
final_accuracy = accuracy_score(
    y_test,
    y_final_pred
)

print("Final KNN accuracy:", final_accuracy)
```

Now examine the full classification report:

```python
print(
    classification_report(
        y_test,
        y_final_pred,
        target_names=iris.target_names
    )
)
```

Finally, create the confusion matrix:

```python
final_cm = confusion_matrix(
    y_test,
    y_final_pred
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    final_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.title("Final Confusion Matrix: Tuned KNN on Iris")
plt.xlabel("Predicted species")
plt.ylabel("Actual species")
plt.show()
```

---

# 32. Final Model Comparison

Create a final comparison.

```python
final_comparison = pd.DataFrame({
    "Model": [
        "Baseline",
        "Initial KNN",
        "Logistic Regression",
        "Tuned KNN"
    ],
    "Accuracy": [
        baseline_accuracy,
        knn_accuracy,
        logreg_accuracy,
        final_accuracy
    ]
})

display(final_comparison)
```

For a fuller comparison, you can also calculate the macro F1 score for the real classifiers:

```python
knn_macro_f1 = f1_score(
    y_test,
    y_knn_pred,
    average="macro"
)

logreg_macro_f1 = f1_score(
    y_test,
    y_logreg_pred,
    average="macro"
)

final_macro_f1 = f1_score(
    y_test,
    y_final_pred,
    average="macro"
)

macro_comparison = pd.DataFrame({
    "Model": [
        "KNN",
        "Logistic Regression",
        "Tuned KNN"
    ],
    "Accuracy": [
        knn_accuracy,
        logreg_accuracy,
        final_accuracy
    ],
    "Macro F1": [
        knn_macro_f1,
        logreg_macro_f1,
        final_macro_f1
    ]
})

display(macro_comparison)
```

### Questions

1. Which model has the highest accuracy?
2. Which model has the highest macro F1?
3. Did tuning meaningfully improve KNN?
4. Are the model differences large or small?
5. Would you choose the model with the highest accuracy automatically?
6. What other factors could influence your choice?

---

# 33. Binary vs Multiclass Classification

Complete the following comparison.

| Concept                    | Titanic | Iris |
| -------------------------- | ------- | ---- |
| Number of classes          | ___     | ___  |
| Classification type        | ___     | ___  |
| Target                     | ___     | ___  |
| Positive class             | ___     | ___  |
| Main confusion matrix size | ___     | ___  |
| Main evaluation challenge  | ___     | ___  |

### Questions

1. What is the main difference between binary and multiclass classification?
2. Why does the confusion matrix become larger in a multiclass problem?
3. Why is there no single positive class in the Iris problem?
4. Why do per-class metrics become particularly useful in multiclass classification?
5. How can a classifier perform well overall but still perform poorly for one class?

---

# 34. Logistic Regression Across the Two Labs

Think back to Lab 1.

In Lab 1:

```text
Titanic
   ↓
2 classes
   ↓
Logistic Regression
   ↓
Binary classification
```

In Lab 2:

```text
Iris
   ↓
3 classes
   ↓
Logistic Regression
   ↓
Multiclass classification
```

### Reflection

Complete this statement:

> Logistic regression is called "regression," but in these labs it is being used for classification because ____________________________.

Then answer:

> What does this example show about the relationship between an algorithm and the type of classification problem?

---

# 35. Final Reflection

Answer the following questions in complete sentences.

### 1. Binary vs multiclass

Explain the difference between binary and multiclass classification using Titanic and Iris as examples.

### 2. Baseline

Why is a baseline important even when the real classifier performs very well?

### 3. KNN

Explain KNN in your own words.

### 4. Confusion matrix

What does the Iris confusion matrix tell you about the model's errors?

### 5. Per-class performance

Why is it useful to examine precision, recall, and F1 for each Iris species rather than looking only at overall accuracy?

### 6. Macro vs weighted

Why are macro and weighted averages relatively similar for Iris?

### 7. Model comparison

Which model would you choose:

* KNN
* logistic regression
* tuned KNN

Explain your choice using evidence from the results.

### 8. Error analysis

Which species was most difficult for the model to classify, and why might that be?

### 9. Preprocessing

Why did we not use `StandardScaler()` in the main workflow?

What potential benefit could scaling have for KNN?

---

# 36. Key Takeaways

The most important ideas from this lab are:

### 1. Multiclass classification

Iris has three possible classes:

```text
setosa
versicolor
virginica
```

Therefore, it is a multiclass classification problem.

---

### 2. The workflow remains the same

The move from binary to multiclass classification does not require an entirely new machine-learning workflow.

We still:

```text
Define X and y
      ↓
Split the data
      ↓
Train the model
      ↓
Evaluate
      ↓
Compare
      ↓
Tune
      ↓
Final evaluation
```

---

### 3. Evaluation becomes more detailed

With multiple classes, we should ask:

> How well does the model perform for **each class**?

This makes:

* per-class precision
* per-class recall
* per-class F1
* confusion matrices
* macro averages

particularly useful.

---

### 4. Logistic regression is a classification algorithm

The name can be misleading.

> **Logistic regression is commonly used for classification, not just regression.**

It can be used in both:

```text
Binary classification
```

and:

```text
Multiclass classification
```

---

### 5. KNN is based on similarity

KNN predicts a class using nearby observations.

Because it relies on distances, feature scale can matter.

In this introductory lab, we deliberately avoided adding `StandardScaler()` so that the focus remains on classification.

---

### 6. Model errors are informative

A good evaluation does more than report:

```text
Accuracy = 96%
```

We should also ask:

> Which classes does the model confuse?

and:

> Why might those observations be difficult to classify?

---

# 37. Optional Self-Study

The following topics are **not required** for the main lab, but they are useful if you want to explore further.

## A. StandardScaler and KNN

Try adding `StandardScaler()` before KNN.

Compare:

```text
KNN without scaling
```

with:

```text
StandardScaler + KNN
```

Questions:

* Does accuracy change?
* Do the predictions change?
* Does the confusion matrix change?
* Why might scaling affect a distance-based model?

---

## B. Different Values of K

Try:

```text
k = 1
k = 3
k = 5
k = 10
k = 20
```

Observe how the model changes.

Consider:

> What happens when K is very small?

and:

> What happens when K becomes very large?

---

## C. One-vs-Rest and One-vs-One

Some classification algorithms handle multiclass classification directly.

Other approaches can transform a multiclass problem into several binary classification problems.

Two common strategies are:

* **one-vs-rest (OvR)**
* **one-vs-one (OvO)**

At a high level:

### One-vs-rest

For three classes:

```text
setosa vs everything else
versicolor vs everything else
virginica vs everything else
```

### One-vs-one

Create pairwise classifiers:

```text
setosa vs versicolor
setosa vs virginica
versicolor vs virginica
```

You do not need to implement these strategies for this lab.

The important point is simply that **multiclass classification can be handled in different ways depending on the algorithm**.

---

## D. ROC and AUC

ROC curves and ROC AUC were introduced as optional self-study in Lab 1.

You can explore them further if you are interested in how classifier performance can be examined across different thresholds.

For multiclass classification, ROC analysis becomes more complicated because there are multiple classes.

This is therefore an **optional advanced topic**, not part of the required Lab 2 workflow.

---

# Deliverables

Submit:

1. **A completed notebook or Python script**
2. **Short written answers** to the interpretation questions
3. **The baseline, KNN, logistic regression, and tuned KNN results**
4. **At least one multiclass confusion matrix**
5. **The final model comparison**
6. **One paragraph explaining which model you would choose and why**
7. **A short explanation of the difference between binary and multiclass classification**

Your final model choice should be supported by evidence from:

* accuracy
* per-class performance
* macro F1
* confusion matrix
* cross-validation
* model simplicity

---

# Final Message

Lab 1 introduced:

> **Binary classification**

Lab 2 extends the same workflow to:

> **Multiclass classification**

The most important progression is:

```text
                 CLASSIFICATION
                       │
          ┌────────────┴────────────┐
          ↓                         ↓
       Binary                   Multiclass
          │                         │
       Titanic                     Iris
          │                         │
       2 classes                 3 classes
          │                         │
   Precision/Recall/F1       Per-class metrics
          │                         │
    2 × 2 confusion matrix    3 × 3 confusion matrix
          │                         │
     Thresholds             Macro / weighted averages
```

The goal is not simply to learn another classifier.

The goal is to understand how the **same supervised-learning workflow extends from two classes to multiple classes**, and how evaluation must become more detailed as the number of classes increases.
