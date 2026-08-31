# Lab 2: Multiclass Classification with the Iris Dataset

## Predicting Iris Flower Species

---

# Lab Purpose

In Lab 1, we used the Titanic dataset to learn about **binary classification**.

The Titanic model had two possible outcomes:

```text
0 → Did not survive
1 → Survived
```

In this lab, we move to a classification problem with **more than two possible classes**.

We will use the famous **Iris dataset** to predict the species of an iris flower.

There are three possible species:

```text
Iris setosa
Iris versicolor
Iris virginica
```

Because there are three possible classes, this is called:

> **multiclass classification**

The main goal of this lab is to understand what changes when we move from **two classes to three classes**.

The overall workflow is still familiar:

```text
Define X and y
      ↓
Split the data
      ↓
Prepare the data
      ↓
Establish a baseline
      ↓
Train a classifier
      ↓
Make predictions
      ↓
Evaluate the model
      ↓
Compare models
```

---

# Learning Outcomes

By the end of this lab, you should be able to:

* explain the difference between binary and multiclass classification
* identify features and target variables in the Iris dataset
* explain what a multiclass classifier does
* establish a simple baseline
* explain the basic idea of K-nearest neighbors (KNN)
* train a KNN classifier
* evaluate a multiclass classifier
* interpret a multiclass confusion matrix
* understand precision, recall, and F1 for individual classes
* explain macro and weighted averages
* use logistic regression for multiclass classification
* compare two classifiers

---

# 1. From Binary to Multiclass Classification

In Lab 1, we had two possible classes:

```text
Titanic

0 → Did not survive
1 → Survived
```

This was **binary classification**.

In this lab, we have three possible classes:

```text
Iris

0 → setosa
1 → versicolor
2 → virginica
```

This is **multiclass classification**.

The important point is:

> The basic classification workflow does not change. The main difference is that the model now has more than two classes to choose from.

### Binary classification

```text
        Features
           ↓
      Classifier
           ↓
      ┌────┴────┐
      ↓         ↓
   Class 0    Class 1
```

### Multiclass classification

```text
        Features
           ↓
      Classifier
           ↓
   ┌────┼────┐
   ↓    ↓    ↓
Class 0 Class 1 Class 2
```

### Question

What is the main difference between binary and multiclass classification?

<details>
<summary><strong>Sample answer</strong></summary>

Binary classification has two possible classes, while multiclass classification has more than two possible classes. Titanic survival prediction is binary because there are two outcomes, while Iris species prediction is multiclass because there are three species.

</details>

---

# 2. The Iris Dataset

The Iris dataset contains measurements of iris flowers.

For each flower, we have four measurements:

```text
sepal length
sepal width
petal length
petal width
```

We want to use these measurements to predict the species.

Conceptually:

```text
Flower measurements
        ↓
    Classifier
        ↓
Iris species
```

The dataset contains three species:

```text
setosa
versicolor
virginica
```

Each species has 50 observations, so the dataset contains 150 flowers in total.

Because the three classes have the same number of observations, the classes are **balanced**.

---

# 3. Load the Dataset

We can load Iris directly from scikit-learn.

```python id="2w4y9s"
from sklearn.datasets import load_iris

iris = load_iris()
```

The `load_iris()` function gives us the dataset together with information about the features and target.

Let's look at the feature names:

```python id="k3f7q2"
print(iris.feature_names)
```

And the class names:

```python id="5z8m1p"
print(iris.target_names)
```

We can also check the shape:

```python id="j9x4sc"
print("Features:", iris.data.shape)
print("Target:", iris.target.shape)
```

We should get:

```text
Features: (150, 4)
Target: (150,)
```

This means:

* 150 flowers
* 4 features for each flower

---

# 4. Features and Target

As in Lab 1, we separate the **features** from the **target**.

The features are the flower measurements:

```python id="9m2r8k"
X = iris.data
```

The target is the species:

```python id="4t7n6v"
y = iris.target
```

So:

```text
X → flower measurements

y → flower species
```

The target values are:

```text
0 → setosa
1 → versicolor
2 → virginica
```

We can check the first few observations:

```python id="6p4x2d"
print(X[:5])
print(y[:5])
```

### Question

What does `X` represent, and what does `y` represent in the Iris problem?

<details>
<summary><strong>Sample answer</strong></summary>

`X` contains the four measurements of each flower and is used as input to the classifier. `y` contains the species of each flower and is the target that the model is trying to predict.

</details>

---

# 5. What Does the Classification Problem Look Like?

For one flower, we might have:

```text
Sepal length = 5.1
Sepal width  = 3.5
Petal length = 1.4
Petal width  = 0.2
```

The correct species is:

```text
setosa
```

The classifier learns from many examples like this.

Eventually, we want to give it the measurements of a flower whose species is unknown:

```text
Flower measurements
        ↓
      Model
        ↓
Predicted species
```

For example:

```text
5.1, 3.5, 1.4, 0.2
        ↓
     setosa
```

---

# 6. Split the Data

As in Lab 1, we should not train and evaluate our model using exactly the same observations.

We split the dataset into:

* training data
* test data

```python id="q8m3nv"
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### What does this code do?

```python id="n7v2la"
test_size=0.2
```

means that 20% of the flowers are placed in the test set.

```python id="x5b8cd"
random_state=42
```

makes the split reproducible.

```python id="j6q9te"
stratify=y
```

helps keep the three species represented in similar proportions in both the training and test sets.

Check the sizes:

```python id="r1z7qp"
print("Training set:", X_train.shape)
print("Test set:", X_test.shape)
```

---

# 7. Do We Need Much Preprocessing?

This is much simpler than the Titanic dataset.

The Iris features are already numerical:

```text
sepal length
sepal width
petal length
petal width
```

There are also essentially no missing values in this dataset.

Therefore, we do **not** need the imputation and categorical encoding that we used in Lab 1.

We can work directly with the numerical features.

However, there is one important issue we need to understand when using KNN:

> **Feature scale can affect distance-based models.**

We will return to this later.

---

# 8. What Is a Baseline?

Before training a real classifier, it is useful to establish a simple reference point.

This is called a **baseline**.

A baseline is a simple strategy that does not try to learn useful relationships between the features and target.

For example, we could always predict the most common species.

Because Iris has three equally represented species:

```text
setosa       → 50
versicolor   → 50
virginica    → 50
```

there is no single majority class.

A simple baseline can therefore randomly predict one of the three classes.

For this lab, we will use a simple **stratified random baseline**.

```python id="t8y5km"
from sklearn.dummy import DummyClassifier

baseline = DummyClassifier(
    strategy="stratified",
    random_state=42
)

baseline.fit(
    X_train,
    y_train
)
```

The baseline learns only the class distribution.

It does not learn relationships such as:

```text
large petals → possibly virginica
small petals → possibly setosa
```

Make predictions:

```python id="w4p8ns"
y_baseline = baseline.predict(X_test)
```

Calculate accuracy:

```python id="q6k2mf"
from sklearn.metrics import accuracy_score

baseline_accuracy = accuracy_score(
    y_test,
    y_baseline
)

print("Baseline accuracy:", baseline_accuracy)
```

### Why use a baseline?

Suppose a real classifier achieves 95% accuracy.

That sounds good.

But we still want to know:

> **How much better is it than a simple strategy?**

The baseline gives us a reference point.

### Question

What is the purpose of a baseline?

<details>
<summary><strong>Sample answer</strong></summary>

A baseline provides a simple reference point against which we can compare a real classifier. It helps us determine whether the model has learned useful patterns rather than simply producing predictions that could be achieved by a simple strategy.

</details>

---

# 9. K-Nearest Neighbors (KNN)

Our first real classifier will be **K-nearest neighbors**, usually called **KNN**.

## The basic idea

KNN makes a prediction by looking at observations that are **similar** to the new observation.

Imagine that we have a new flower.

KNN asks:

> Which flowers in the training data are closest to this flower?

Suppose we use:

```text
k = 5
```

The model looks at the five nearest flowers.

Imagine their species are:

```text
setosa
setosa
setosa
versicolor
setosa
```

The majority is:

```text
setosa
```

So KNN predicts:

```text
setosa
```

Conceptually:

```text
New flower
     ↓
Find nearest flowers
     ↓
Look at their classes
     ↓
Majority vote
     ↓
Predicted class
```

---

# 10. Why Is KNN Called "K-Nearest Neighbors"?

The name tells us how the algorithm works.

### K

`K` is the number of neighbors we look at.

For example:

```text
K = 3
```

means:

> Look at the three nearest training observations.

### Nearest

We need some way to measure how close two flowers are.

Because Iris contains numerical measurements, we can use distance.

### Neighbors

The observations closest to our new observation are its neighbors.

---

# 11. An Important Issue: Feature Scale

KNN uses distances.

Suppose one feature has values between:

```text
1 and 5
```

while another has values between:

```text
1 and 100
```

The larger-scale feature can have a much stronger effect on the distance.

For example:

```text
Feature A difference = 2

Feature B difference = 40
```

The second feature contributes much more to the distance.

This is why **scaling can be important for KNN**.

In the Iris dataset, the four features are measured in the same unit and have relatively similar ranges, so we will first keep the workflow simple and use the original values.

---

# 12. Train a KNN Classifier

Import KNN:

```python id="d6m2qa"
from sklearn.neighbors import KNeighborsClassifier
```

Create the classifier:

```python id="v9r4kc"
knn = KNeighborsClassifier(
    n_neighbors=5
)
```

The parameter:

```python id="q2s8jd"
n_neighbors=5
```

means that the classifier will look at the five nearest training observations.

Now train the classifier:

```python id="b7w3fp"
knn.fit(
    X_train,
    y_train
)
```

### What does `fit()` mean?

As in Lab 1:

> `fit()` means that we are giving the model the training data so it can prepare itself for making predictions.

For KNN, there is an important difference from some other models.

KNN does not learn a complicated mathematical equation in the same way logistic regression does.

Instead, it keeps the training observations and uses them when a new observation needs to be classified.

---

# 13. Make Predictions

Use the trained KNN classifier to predict the test flowers:

```python id="z5n2kx"
y_pred = knn.predict(
    X_test
)
```

Look at some predictions:

```python id="w1c8sd"
print("Predicted:")
print(y_pred[:15])

print("\nActual:")
print(y_test[:15])
```

Remember:

```text
0 → setosa
1 → versicolor
2 → virginica
```

The model has now produced one of three possible classes for each test flower.

---

# 14. Accuracy in Multiclass Classification

Accuracy works in the same basic way as in binary classification.

It answers:

> **What proportion of all predictions were correct?**

```python id="e7k4ps"
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Accuracy:", accuracy)
```

For example:

```text
Accuracy = 0.93
```

means that approximately 93% of the test flowers were classified correctly.

However, accuracy does not tell us:

> Which species did the model get wrong?

For that, we need a confusion matrix.

---

# 15. The Multiclass Confusion Matrix

In Lab 1, the confusion matrix had two classes:

```text
0
1
```

Therefore, it was a 2 × 2 matrix.

Here we have three classes:

```text
0 → setosa
1 → versicolor
2 → virginica
```

So the confusion matrix becomes a 3 × 3 matrix.

```python id="p9h5xv"
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)
```

Visualize it:

```python id="a3q7nf"
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Iris Confusion Matrix")
plt.show()
```

---

# 16. How to Read the Confusion Matrix

The rows represent the **actual** species.

The columns represent the **predicted** species.

For example:

```text
                    Predicted

                 setosa  versicolor  virginica

Actual setosa
Actual versicolor
Actual virginica
```

The diagonal contains correct predictions.

```text
setosa     → setosa
versicolor → versicolor
virginica  → virginica
```

Values away from the diagonal represent errors.

For example:

```text
Actual: versicolor
Predicted: virginica
```

means:

> The flower was actually versicolor, but the model predicted virginica.

---

# 17. Why Is the Multiclass Confusion Matrix Useful?

Imagine the model has:

```text
Accuracy = 95%
```

That sounds excellent.

But suppose almost all errors involve:

```text
versicolor ↔ virginica
```

while setosa is almost always classified correctly.

The confusion matrix shows us this.

This is important because:

> A model can perform very well overall while still having difficulty distinguishing particular classes.

### Question

What does a value away from the diagonal of the confusion matrix represent?

<details>
<summary><strong>Sample answer</strong></summary>

A value away from the diagonal represents an incorrect prediction. For example, if the actual class is versicolor but the predicted class is virginica, the observation appears outside the diagonal.

</details>

---

# 18. Precision, Recall, and F1 in Multiclass Classification

In Lab 1, we talked about:

* precision
* recall
* F1

The same ideas can be used for multiclass classification.

The difference is that we can calculate them **for each class**.

For example, we can ask:

> How well does the model identify setosa?

Then:

> How well does it identify versicolor?

And:

> How well does it identify virginica?

Scikit-learn can calculate these values for us.

```python id="k6x9pt"
from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)
```

You should see something similar to:

```text
              precision    recall    f1-score    support

setosa           ...
versicolor       ...
virginica        ...

accuracy         ...

macro avg        ...
weighted avg     ...
```

---

# 19. What Do the Per-Class Metrics Mean?

Consider **setosa**.

### Precision for setosa

> Of the flowers predicted to be setosa, how many actually were setosa?

### Recall for setosa

> Of all the flowers that actually were setosa, how many did the model correctly identify?

### F1 for setosa

> How well does the model balance precision and recall for setosa?

The same questions can be asked separately for:

```text
setosa
versicolor
virginica
```

This is one of the important differences between binary and multiclass evaluation.

### Question

Why is it useful to look at metrics for each species rather than only overall accuracy?

<details>
<summary><strong>Sample answer</strong></summary>

Overall accuracy tells us how well the model performs across all flowers, but it does not show whether one particular species is harder to classify. Per-class precision, recall, and F1 allow us to see how well the model performs for each species.

</details>

---

# 20. Macro Average vs Weighted Average

The classification report also gives us:

```text
macro avg
weighted avg
```

These are ways of combining the metrics from the three classes into a single value.

## Macro average

The macro average gives each class equal importance.

For example:

```text
setosa F1       = 0.98
versicolor F1   = 0.92
virginica F1    = 0.94
```

The macro F1 is simply the average of these three class-level F1 scores.

Conceptually:

```text
Macro F1 =
(F1 setosa + F1 versicolor + F1 virginica) / 3
```

Every species has equal importance.

---

## Weighted average

The weighted average also combines the class-level metrics, but gives more weight to classes that contain more observations.

For the Iris dataset, each class has 50 observations.

Therefore, macro and weighted averages should be quite similar.

This is because the classes are balanced.

### Question

Why are macro and weighted averages similar for the Iris dataset?

<details>
<summary><strong>Sample answer</strong></summary>

The Iris dataset contains the same number of observations for each of the three species. Because the classes are balanced, giving every class equal weight or weighting them according to their size produces very similar results.

</details>

---

# 21. What Happens When Classes Are Imbalanced?

Imagine a different dataset:

```text
Class A → 90 observations
Class B → 8 observations
Class C → 2 observations
```

A weighted average would be strongly influenced by Class A because it contains most of the observations.

A macro average gives all three classes equal importance.

Therefore:

> **Macro averages are particularly useful when we want to treat every class as equally important, especially when the classes are imbalanced.**

For Iris, the difference is small because the classes are balanced.

---

# 22. Logistic Regression Can Also Do Multiclass Classification

In Lab 1, we used logistic regression for binary classification.

Here we can use logistic regression again, but now there are three classes.

This is an important idea:

> **The same classification algorithm can often be used for both binary and multiclass classification.**

In Lab 1:

```text
Titanic
   ↓
2 classes
   ↓
Logistic regression
```

In Lab 2:

```text
Iris
   ↓
3 classes
   ↓
Logistic regression
```

The name can be confusing because it contains the word "regression."

But remember:

> **Logistic regression is a classification algorithm.**

---

# 23. Train Logistic Regression

Import the model:

```python id="u2m8qc"
from sklearn.linear_model import LogisticRegression
```

Create the classifier:

```python id="f7w3kj"
logreg = LogisticRegression(
    max_iter=1000
)
```

The `max_iter=1000` setting gives the algorithm enough iterations to find a solution.

Train the model:

```python id="s9c5bn"
logreg.fit(
    X_train,
    y_train
)
```

Again:

```text
fit()
```

means:

> Learn from the training data.

---

# 24. Make Logistic Regression Predictions

```python id="r4v8qd"
y_logreg_pred = logreg.predict(
    X_test
)
```

Calculate accuracy:

```python id="p6n2kx"
logreg_accuracy = accuracy_score(
    y_test,
    y_logreg_pred
)

print("Logistic Regression accuracy:",
      logreg_accuracy)
```

Now calculate the classification report:

```python id="e8w1qm"
print(
    classification_report(
        y_test,
        y_logreg_pred,
        target_names=iris.target_names
    )
)
```

---

# 25. Compare KNN and Logistic Regression

We now have two classifiers:

```text
KNN
↓
Uses nearby observations

Logistic Regression
↓
Learns relationships between features and class
```

Let's compare their accuracy:

```python id="n4x7ps"
print("KNN accuracy:",
      accuracy)

print("Logistic Regression accuracy:",
      logreg_accuracy)
```

We can also create a small comparison table:

```python id="h5k9rc"
comparison = pd.DataFrame({
    "Model": [
        "Baseline",
        "KNN",
        "Logistic Regression"
    ],

    "Accuracy": [
        baseline_accuracy,
        accuracy,
        logreg_accuracy
    ]
})

display(comparison)
```

---

# 26. Which Model Is Better?

Do not assume that one algorithm is always better.

A model should be judged using the results of the evaluation.

For this dataset, we can consider:

* accuracy
* per-class precision
* per-class recall
* per-class F1
* confusion matrix

Because the Iris dataset is balanced, accuracy is a particularly easy metric to interpret.

However, the confusion matrix and per-class results tell us more about **where the model makes mistakes**.

### Questions

1. Which classifier achieved the higher accuracy?
2. Which species appears to be the most difficult to classify?
3. What evidence from the confusion matrix or classification report supports your answer?

<details>
<summary><strong>Sample answer</strong></summary>

The classifier with the higher accuracy was **[model]**.

The most difficult species appears to be **[species]**, because it has the lowest recall/F1 score or because it is involved in more incorrect predictions in the confusion matrix.

The confusion matrix shows that the main errors occur between **[species A]** and **[species B]**.

</details>

---

# 27. Connecting Iris to Titanic

We can now compare the two labs.

|                | Titanic                 | Iris                      |
| -------------- | ----------------------- | ------------------------- |
| Task           | Predict survival        | Predict species           |
| Type           | Binary classification   | Multiclass classification |
| Classes        | 2                       | 3                         |
| Features       | Numerical + categorical | Numerical                 |
| Missing values | Present                 | Essentially none          |
| First model    | Logistic regression     | KNN                       |
| Other model    | Decision tree           | Logistic regression       |

The most important point is that the overall workflow is still the same.

```text
Titanic                         Iris

X and y                         X and y
   ↓                               ↓
Train/test split                Train/test split
   ↓                               ↓
Preprocessing                   Minimal preprocessing
   ↓                               ↓
Classifier                      Classifier
   ↓                               ↓
Predictions                     Predictions
   ↓                               ↓
Evaluation                      Evaluation
```

The major change is:

```text
Titanic
2 classes

        ↓

Iris
3 classes
```

---

# 28. Binary vs Multiclass Evaluation

In Lab 1, we could talk about:

```text
positive class
negative class
```

For example:

```text
Survived
Did not survive
```

With three Iris species, there is no single natural positive class.

Instead, we can evaluate each class separately.

For example, when evaluating setosa:

```text
setosa
vs
not setosa
```

Then when evaluating versicolor:

```text
versicolor
vs
not versicolor
```

And similarly for virginica.

This is why the classification report gives us separate precision, recall, and F1 values for each class.

---

# 29. Why Can Iris Be Easier to Classify Than Titanic?

Compare the two datasets.

Titanic contains:

* missing values
* categorical variables
* numerical variables
* noisy historical information
* a complex real-world outcome

Iris is much simpler.

It contains:

* four numerical measurements
* three well-defined species
* balanced classes
* very little missing data

This means that a classifier can often achieve very high accuracy on Iris.

That does **not** mean that classification is always easy.

Iris is a useful teaching dataset because it allows us to focus on the classification concepts without dealing with a large amount of preprocessing.

---

# 30. Final Reflection

Answer the following four questions.

## Question 1 — Binary vs Multiclass

Explain the difference between binary and multiclass classification using Titanic and Iris as examples.

<details>
<summary><strong>Sample answer</strong></summary>

Binary classification has two possible classes. Titanic survival prediction is binary because the passenger either survived or did not survive.

Multiclass classification has more than two possible classes. Iris species prediction is multiclass because the model chooses between setosa, versicolor, and virginica.

</details>

---

## Question 2 — Baseline

Why is it useful to compare a classifier with a baseline?

<details>
<summary><strong>Sample answer</strong></summary>

A baseline provides a simple reference point. It helps us determine whether the classifier has learned useful patterns and performs better than a simple prediction strategy.

</details>

---

## Question 3 — Confusion Matrix

Look at your Iris confusion matrix.

Which species was most often confused with another species?

<details>
<summary><strong>Sample answer structure</strong></summary>

The species most often confused with another species was **[species]**. The confusion matrix shows **[number]** cases where the actual species was **[species]** but the model predicted **[other species]**.

</details>

---

## Question 4 — Model Comparison

Which model would you choose for this dataset: KNN or logistic regression?

Use at least two pieces of evidence from your results.

<details>
<summary><strong>Sample answer structure</strong></summary>

I would choose **[model]**.

Its accuracy was **[value]**, compared with **[value]** for the other model. Its per-class results also show **[better F1 / better recall / fewer errors for a particular species]**.

Therefore, based on the results, I would prefer **[model]** for this dataset.

</details>

---

# 31. Key Takeaways

The most important ideas from this lab are:

### 1. Multiclass classification

Iris has three possible classes:

```text
setosa
versicolor
virginica
```

Therefore, Iris species prediction is a **multiclass classification** problem.

---

### 2. The workflow is familiar

We still:

```text
Define X and y
      ↓
Split the data
      ↓
Prepare the data
      ↓
Establish a baseline
      ↓
Train a classifier
      ↓
Make predictions
      ↓
Evaluate
      ↓
Compare
```

The workflow did not fundamentally change from Lab 1.

---

### 3. The confusion matrix gets larger

Binary classification:

```text
2 classes
→ 2 × 2 confusion matrix
```

Multiclass classification:

```text
3 classes
→ 3 × 3 confusion matrix
```

With more classes, the confusion matrix shows which classes are being confused with each other.

---

### 4. Evaluation can be done per class

Instead of evaluating only overall performance, we can ask:

```text
How well does the model classify setosa?

How well does it classify versicolor?

How well does it classify virginica?
```

This gives us a more detailed picture of model performance.

---

### 5. Macro and weighted averages

Macro average:

> Gives every class equal importance.

Weighted average:

> Gives more influence to classes with more observations.

Because Iris has balanced classes, the two averages are similar.

---

### 6. Different classifiers can solve the same problem

KNN and logistic regression are different algorithms, but both can be used to classify Iris flowers.

The important question is not:

> Which algorithm is always best?

Instead, ask:

> **Which model performs well for this particular problem, and what evidence supports that conclusion?**

---

# 32. Final Classification Workflow

You have now seen both binary and multiclass classification.

```text
                 CLASSIFICATION
                       │
             ┌─────────┴─────────┐
             ↓                   ↓
          Binary              Multiclass
             ↓                   ↓
         Titanic                Iris
             ↓                   ↓
        2 classes             3 classes
             ↓                   ↓
       Train classifier     Train classifier
             ↓                   ↓
          Predict              Predict
             ↓                   ↓
         Evaluate            Evaluate
             ↓                   ↓
      Confusion matrix    Confusion matrix
             ↓                   ↓
       Compare models     Compare models
```

The central idea is:

> **Classification means using features to predict which class an observation belongs to.**

In Lab 1, there were two possible classes.

In Lab 2, there are three.

The underlying machine-learning workflow remains largely the same.
