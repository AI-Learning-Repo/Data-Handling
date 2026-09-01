# Classification with Machine Learning

## From Titanic to Iris

---

# 1. What Are We Trying to Do?

In exploratory data analysis (EDA), we use data to understand patterns.

For example, in the Titanic dataset we might discover that:

* passenger class is related to survival
* sex is related to survival
* age is related to survival
* some variables contain missing values

EDA helps us answer:

> **What does the data look like?**

Machine learning takes us one step further.

We now ask:

> **Can we use the information in the data to make predictions?**

This chapter introduces **classification**, a type of supervised machine learning.

We will use two examples:

* **Titanic** — binary classification
* **Iris** — multiclass classification

The two problems are different, but the basic machine-learning workflow is very similar.

---

# 2. What Is Machine Learning?

Machine learning is a way of building models that learn patterns from data and use those patterns to make predictions.

Instead of explicitly writing every rule ourselves, we provide examples to an algorithm.

For example, imagine we want to predict whether a Titanic passenger survived.

We have historical examples:

```text
Passenger information        Known outcome

female, 1st class, 30        survived
male, 3rd class, 25          did not survive
female, 2nd class, 40        survived
...
```

A machine-learning algorithm can use these examples to learn patterns associated with survival.

We can then give the model information about a new passenger and ask it to predict the outcome.

Conceptually:

```text
Historical data
      ↓
Learning algorithm
      ↓
Trained model
      ↓
New passenger
      ↓
Prediction
```

---

# 3. What Is Supervised Learning?

**Supervised learning** means that the training data contains the correct answer, called the **target**.

The model learns a relationship between:

* the information we use as input
* the known answer we want to predict

For Titanic:

```text
Input:
passenger information

Target:
survived
```

For Iris:

```text
Input:
flower measurements

Target:
species
```

The word *supervised* comes from the fact that the model can compare its predictions with known answers during training.

---

# 4. Features and Target

Two terms are especially important:

### Features

**Features** are the variables used by the model to make a prediction.

They are often represented by:

```python
X
```

### Target

The **target** is the variable we want the model to predict.

It is often represented by:

```python
y
```

For Titanic:

```text
X → passenger information
y → survival
```

For Iris:

```text
X → flower measurements
y → species
```

We can think of the problem as:

```text
             Features (X)
                  ↓
                Model
                  ↓
             Target (y)
```

During training, the model sees both `X` and `y`.

Later, when making predictions, we give it only `X`.

---

# 5. What Is Classification?

**Classification** is a supervised-learning task in which the target is a **category or class**.

For example:

```text
spam / not spam
disease / no disease
survived / did not survive
cat / dog / bird
```

The model's job is to decide which class an observation belongs to.

This is different from **regression**.

In regression, the target is a numerical quantity.

For example:

```text
Predict house price
Predict temperature
Predict sales
```

So:

```text
Classification → predict a class
Regression     → predict a numerical value
```

The two tasks use different types of models and evaluation measures.

---

# 6. Binary Classification

A classification problem with exactly **two classes** is called **binary classification**.

The Titanic problem is binary classification.

The target can have two values:

```text
0 → did not survive
1 → survived
```

The model therefore has two possible answers.

Conceptually:

```text
Passenger information
        ↓
     Classifier
        ↓
   ┌────┴────┐
   ↓         ↓
No survival  Survival
```

Other examples of binary classification include:

* fraud / not fraud
* spam / not spam
* defective / not defective
* disease / no disease

---

# 7. Multiclass Classification

A classification problem with more than two classes is called **multiclass classification**.

The Iris dataset provides a simple example.

There are three species:

```text
0 → setosa
1 → versicolor
2 → virginica
```

The model must choose one of these three classes.

```text
Flower measurements
        ↓
     Classifier
        ↓
 ┌──────┼──────┐
 ↓      ↓      ↓
setosa  versicolor  virginica
```

The important point is that **the overall classification workflow does not fundamentally change**.

We still:

```text
Choose features and target
        ↓
Split the data
        ↓
Prepare the data
        ↓
Train a classifier
        ↓
Make predictions
        ↓
Evaluate the predictions
```

The main difference is the number of possible classes.

---

# 8. Binary vs Multiclass Classification

The two labs can therefore be summarized as follows:

|                   | Titanic                 | Iris                      |
| ----------------- | ----------------------- | ------------------------- |
| Task              | Predict survival        | Predict species           |
| Type              | Binary classification   | Multiclass classification |
| Number of classes | 2                       | 3                         |
| Features          | Numerical + categorical | Numerical                 |
| Target            | `survived`              | `species`                 |

This connection is important.

**Lab 2 is not introducing an entirely new machine-learning workflow.**

It extends the ideas from Lab 1 to a problem with more than two classes.

---

# 9. Why Do We Split the Data?

Suppose we train a model using 100 observations.

After training, we ask the model to predict those same 100 observations.

The model may perform very well.

But this does not tell us whether it can make good predictions for **new observations**.

We therefore divide the data into at least two parts:

```text
Dataset
   │
   ├──────────────┐
   ↓              ↓
Training data   Test data
   ↓              ↓
Train model    Evaluate model
```

### Training data

The training data is used to fit the model.

### Test data

The test data is kept separate and is used to evaluate how well the trained model performs on observations it did not use during training.

This gives us a better idea of how the model may perform on new data.

---

# 10. Why Not Use the Test Data for Training?

The test set should represent **unseen data**.

If we repeatedly use the test set to make decisions about the model, it is no longer a truly independent evaluation.

For example, imagine that we try five different models and choose the one that performs best on the test set.

We have now used the test results to make a modelling decision.

The test set is no longer completely "unseen" from our decision-making perspective.

For these introductory labs, the important rule is simple:

> **Train the model on the training data and use the test data to evaluate it.**

More advanced methods for model selection, such as **cross-validation**, will be introduced later.

---

# 11. Preparing the Data

Machine-learning algorithms generally require data in a suitable numerical form.

The type of preparation depends on the dataset.

## Titanic

Titanic contains both numerical and categorical variables.

For example:

```text
Numerical:
age
fare
sibsp

Categorical:
sex
embarked
```

Some variables also contain missing values.

Therefore, the Titanic lab needs to deal with:

* missing values
* categorical variables
* numerical variables

---

## Iris

The Iris dataset is simpler.

Its four input variables are numerical measurements:

```text
sepal length
sepal width
petal length
petal width
```

Therefore, Iris requires much less preprocessing.

This is useful because it lets us focus on **classification itself** rather than spending most of the lab cleaning the data.

---

# 12. Missing Values

A **missing value** means that a value is not available for an observation.

For example:

```text
age = missing
```

Many machine-learning algorithms cannot work directly with missing values.

One common solution is **imputation**.

Imputation means replacing missing values with a reasonable value calculated from the available data.

For example, we might replace missing ages with the median age.

```text
Before:
22
35
missing
28

After:
22
35
28
28
```

The exact strategy depends on the problem.

The important concept is:

> **We need to handle missing values before giving the data to many machine-learning algorithms.**

---

# 13. Categorical Variables

A categorical variable contains categories rather than numerical measurements.

For example:

```text
sex

male
female
```

Many machine-learning algorithms work with numerical inputs.

We therefore need a way to represent categories numerically.

One common method is **one-hot encoding**.

For example:

```text
sex

male
female
male
```

can become:

```text
sex_male    sex_female

1           0
0           1
1           0
```

The model can then use these numerical columns as features.

---

# 14. Scaling

Some algorithms are affected by the scale of numerical variables.

Suppose we have:

```text
age:    18–80
income: 20,000–100,000
```

A difference in income is numerically much larger than a difference in age.

For algorithms that rely on distances, this can matter.

**Scaling** transforms numerical variables so that they are on comparable scales.

A common method is standardization:

```text
new value =
(value - mean) / standard deviation
```

After standardization, a feature is approximately centered around 0 with a standard deviation of 1.

We will see why this is especially relevant to **K-nearest neighbors** in Lab 2.

---

# 15. A Note About Preprocessing and the Test Set

There is an important rule when preprocessing data.

Suppose we calculate the average age to replace missing values.

We should calculate that average using the **training data**, not the entire dataset.

Similarly, if we calculate scaling parameters, they should be learned from the training data.

Then those same transformations are applied to the test data.

Conceptually:

```text
Training data
      ↓
Learn preprocessing
      ↓
Transform training data

Same preprocessing
      ↓
Transform test data
```

Why?

Because the test data is supposed to represent unseen data.

If information from the test set is used while preparing the training data, we allow information to leak from the test set into the training process.

This is called **data leakage**.

For the simplified labs, the important idea is simply:

> **Do not use information from the test set to prepare or train the model.**

---

# 16. What Is a Baseline?

Before training a machine-learning model, it is useful to have a simple reference point.

This is called a **baseline**.

A baseline is a simple prediction strategy that does not attempt to learn useful relationships between the features and the target.

For Titanic, one simple baseline is:

> Always predict the most common survival class.

For example, if 62% of passengers in the training data did not survive, a baseline that always predicts "did not survive" would have an accuracy of approximately 62%.

A real classifier should ideally do better than this.

---

# 17. Why Do We Need a Baseline?

Suppose a classifier achieves:

```text
Accuracy = 65%
```

Is that good?

We cannot answer without context.

If the baseline accuracy is:

```text
Accuracy = 50%
```

then 65% may represent useful improvement.

But if the baseline is:

```text
Accuracy = 64%
```

then the model has added very little.

Therefore:

> **A baseline gives us something simple to compare the model against.**

The baseline does not have to be good.

Its purpose is to provide a reference point.

---

# 18. Baselines and Balanced Classes

Iris is slightly different.

There are 50 observations of each species:

```text
setosa       50
versicolor   50
virginica    50
```

The classes are therefore **balanced**.

There is no single majority class.

A simple baseline can instead use the class distribution to produce predictions.

The important lesson is not the exact baseline strategy.

It is:

> **Before deciding that a model performs well, compare it with a simple reference.**

---

# 19. What Is a Classifier?

A **classifier** is a machine-learning model that predicts a class.

We will use three different classifiers across the two labs:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)

They solve the same general problem:

```text
Features
   ↓
Classifier
   ↓
Predicted class
```

But they learn patterns in different ways.

---

# 20. Logistic Regression

Despite its name, **logistic regression is a classification algorithm**.

It is called regression because it models a mathematical relationship between the input variables and the probability of belonging to a class.

For binary classification, the model estimates the probability of one of the two classes.

For example:

```text
Passenger information
        ↓
Logistic regression
        ↓
Probability of survival
        ↓
Class prediction
```

Imagine the model estimates:

```text
Probability of survival = 0.82
```

A classification rule can then use that probability to determine the predicted class.

For example:

```text
0.82 → predicted survived
```

The important idea for this course is:

> **Logistic regression learns a relationship between the features and the probability of belonging to a class.**

---

# 21. Logistic Regression in Titanic

Suppose our features include:

```text
pclass
sex
age
fare
```

Logistic regression learns how these variables are associated with the probability of survival.

Conceptually:

```text
pclass ─┐
sex ────┤
age ────┤
fare ───┤
        ↓
Logistic Regression
        ↓
Probability
        ↓
Survived / Did not survive
```

The model does not simply memorize the training examples.

It learns a mathematical relationship that can be applied to new passengers.

---

# 22. Logistic Regression in Iris

Logistic regression can also be used when there are more than two classes.

For Iris, the possible classes are:

```text
setosa
versicolor
virginica
```

The model estimates which class is most likely for a flower based on its measurements.

Conceptually:

```text
Flower measurements
        ↓
Logistic Regression
        ↓
Class probabilities
        ↓
Most likely species
```

This gives us an important lesson:

> **An algorithm such as logistic regression can be used for both binary and multiclass classification.**

The classification problem changes, but the algorithm can still be used.

---

# 23. Decision Trees

A **decision tree** makes predictions by asking a sequence of questions about the features.

Imagine a simplified Titanic tree:

```text
Is the passenger female?
       /       \
     Yes        No
     /           \
Survived?      Is class 1?
                /     \
              Yes      No
              /         \
          Survived   Did not survive
```

The actual tree learned by the algorithm is more complicated, but the basic idea is the same.

The model divides the observations into groups using questions about the features.

---

# 24. How Does a Decision Tree Learn?

During training, the algorithm searches for useful ways to split the training observations.

For example, it might discover that:

```text
sex = female
```

is useful for separating passengers with different survival outcomes.

It may then find another useful split, such as:

```text
pclass <= 1
```

The result is a tree of decisions.

Conceptually:

```text
                 Feature
                    ↓
              ┌─────┴─────┐
              ↓           ↓
            Group 1      Group 2
              ↓           ↓
           Feature      Feature
              ↓           ↓
             ...         ...
```

---

# 25. Why Are Decision Trees Useful?

Decision trees have an important advantage:

> **They are relatively easy to interpret.**

We can inspect the tree and see the decisions it makes.

For example:

```text
If sex = female
    then ...
else
    if pclass <= 1
        then ...
```

This makes decision trees useful when we want a model whose decisions can be explained.

However, trees can also become too complex and memorize the training data.

This is called **overfitting**.

We will keep the discussion of controlling tree complexity simple in these labs. More detailed tree tuning can be studied later.

---

# 26. K-Nearest Neighbors (KNN)

The third classifier used in our labs is **K-nearest neighbors**, usually abbreviated as **KNN**.

KNN uses a very different idea.

Instead of learning a set of rules like a decision tree, KNN looks at the training observations that are **closest to a new observation**.

The basic idea is:

> **Similar observations are likely to belong to the same class.**

---

# 27. How Does KNN Work?

Imagine we have a new Iris flower.

KNN looks for flowers in the training data that are most similar to it.

Suppose we choose:

```text
K = 5
```

The model finds the five nearest flowers.

Imagine their species are:

```text
setosa
setosa
setosa
versicolor
setosa
```

Four out of five are setosa.

KNN therefore predicts:

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
Predicted species
```

---

# 28. What Does K Mean?

`K` is the number of neighbors used to make the prediction.

For example:

```text
K = 3
```

means:

> Look at the three closest training observations.

```text
K = 5
```

means:

> Look at the five closest training observations.

A small value of `K` means that the model pays close attention to nearby observations.

A larger value means that the prediction is based on a larger group.

Choosing the value of `K` is a **hyperparameter**.

A hyperparameter is a setting chosen by the person building the model rather than learned directly from the training data.

For the introductory lab, we will simply use a reasonable value such as:

```python
n_neighbors=5
```

More systematic hyperparameter tuning will be introduced later.

---

# 29. Why Does Scaling Matter for KNN?

KNN depends on **distance**.

Imagine comparing two flowers using two features:

```text
petal length
petal width
```

The algorithm calculates how close observations are.

If one feature has a much larger numerical scale than another, it can dominate the distance calculation.

For example:

```text
Feature A difference = 1
Feature B difference = 100
```

The second feature can have a much larger influence on the distance.

Scaling puts features on more comparable scales.

This is why:

> **Scaling is particularly important for distance-based algorithms such as KNN.**

In our Iris lab, the features are measured in the same unit and have reasonably similar ranges, so we can keep the main workflow simple.

Scaling remains an important concept to remember for future datasets.

---

# 30. Comparing the Three Algorithms

The three algorithms can be understood through their main ideas:

| Algorithm           | Main idea                                                                  |
| ------------------- | -------------------------------------------------------------------------- |
| Logistic Regression | Learn a mathematical relationship between features and class probabilities |
| Decision Tree       | Ask a sequence of feature-based questions                                  |
| KNN                 | Look at nearby observations and use their classes                          |

They all perform classification, but they do it differently.

This is an important machine-learning idea:

> **There is usually more than one algorithm that can be used to solve a classification problem.**

---

# 31. Training a Model: `fit()`

In scikit-learn, models are usually trained using:

```python
model.fit(X_train, y_train)
```

For example:

```python
logreg.fit(X_train, y_train)
```

or:

```python
tree.fit(X_train, y_train)
```

or:

```python
knn.fit(X_train, y_train)
```

The `fit()` method means:

> **Use the training data to learn the model.**

The exact learning process depends on the algorithm.

For example:

* logistic regression estimates model parameters
* a decision tree finds useful splits
* KNN stores the training observations that it will use when making predictions

---

# 32. Making Predictions: `predict()`

After training, we can use:

```python
model.predict(X_test)
```

For example:

```python
y_pred = logreg.predict(X_test)
```

This means:

> Use the trained model to predict the class for each observation in `X_test`.

The result is a set of predicted classes.

For Titanic:

```text
0
1
0
0
1
...
```

For Iris:

```text
0
2
1
0
2
...
```

The model's predictions can then be compared with the true target values.

---

# 33. Predictions vs Actual Values

Suppose the test data contains:

```text
Actual:
1  0  1  1  0
```

and the model predicts:

```text
Predicted:
1  0  0  1  0
```

We compare the two:

```text
Actual     Predicted

1          1       ✓
0          0       ✓
1          0       ✗
1          1       ✓
0          0       ✓
```

The model made four correct predictions and one incorrect prediction.

This comparison forms the basis of classification evaluation.

---

# 34. Accuracy

**Accuracy** is the proportion of predictions that are correct.

The formula is:

```text
Accuracy =
Number of correct predictions
-----------------------------
Total number of predictions
```

For example, if the model makes 90 correct predictions out of 100:

```text
Accuracy = 90 / 100 = 0.90
```

or:

```text
90%
```

Accuracy is easy to understand and is useful when the classes are reasonably balanced.

However, accuracy does not tell us **what kinds of mistakes** the model makes.

For that, we need other evaluation tools.

---

# 35. The Confusion Matrix

A **confusion matrix** shows how predicted classes compare with actual classes.

For binary classification, we can have:

```text
                  Predicted
                  0      1

Actual 0         TN     FP
Actual 1         FN     TP
```

The four terms are:

### True Positive (TP)

The model predicted positive and the actual class was positive.

### True Negative (TN)

The model predicted negative and the actual class was negative.

### False Positive (FP)

The model predicted positive, but the actual class was negative.

### False Negative (FN)

The model predicted negative, but the actual class was positive.

---

# 36. Titanic Example

For Titanic, we can think of:

```text
Positive = survived
Negative = did not survive
```

A false positive would mean:

> The model predicted that the passenger survived, but the passenger actually did not survive.

A false negative would mean:

> The model predicted that the passenger did not survive, but the passenger actually survived.

This is why the confusion matrix is useful:

> **It shows us the types of errors the model makes.**

---

# 37. Multiclass Confusion Matrices

For Iris, there are three classes:

```text
setosa
versicolor
virginica
```

The confusion matrix therefore becomes a 3 × 3 matrix.

Conceptually:

```text
                    Predicted

                 setosa  versicolor  virginica

Actual setosa

Actual versicolor

Actual virginica
```

The diagonal contains correct predictions.

For example:

```text
Actual setosa
Predicted setosa
```

is a correct prediction.

An off-diagonal entry is an error.

For example:

```text
Actual versicolor
Predicted virginica
```

means the model confused those two species.

---

# 38. Precision

**Precision** asks:

> Of the observations the model predicted as a particular class, how many actually belonged to that class?

For a binary positive class:

```text
Precision =
TP
--------
TP + FP
```

Imagine the model predicted 20 passengers would survive.

If only 15 actually survived:

```text
Precision = 15 / 20 = 0.75
```

So 75% of the model's positive predictions were correct.

Precision is particularly important when **false positives are costly**.

---

# 39. Recall

**Recall** asks:

> Of all observations that actually belong to a particular class, how many did the model correctly identify?

The formula is:

```text
Recall =
TP
--------
TP + FN
```

Suppose 30 passengers actually survived, and the model correctly identified 24 of them.

Then:

```text
Recall = 24 / 30 = 0.80
```

The model found 80% of the actual survivors.

Recall is particularly important when **missing positive cases is costly**.

---

# 40. F1 Score

Precision and recall measure different aspects of performance.

The **F1 score** combines them into one measure.

It is the harmonic mean of precision and recall:

```text
F1 = 2 × (precision × recall)
          --------------------
          precision + recall
```

A high F1 score requires both precision and recall to be reasonably high.

This makes F1 useful when we want a balance between the two.

---

# 41. Why Not Just Use Accuracy?

Imagine a dataset where:

```text
95% → class A
5%  → class B
```

A model that always predicts class A would have:

```text
95% accuracy
```

That sounds excellent.

But it would never correctly identify class B.

This illustrates why:

> **Accuracy alone can hide poor performance for an important class.**

This is why we also examine:

* confusion matrices
* precision
* recall
* F1

---

# 42. Precision, Recall, and F1 for Multiclass Problems

With three Iris species, there is no single obvious "positive" class.

Instead, we can evaluate each class separately.

For example, for **setosa**, we can ask:

> How well does the model identify setosa?

Then we can do the same for:

```text
versicolor
virginica
```

Conceptually, each class can be considered against all the other classes:

```text
setosa vs not-setosa

versicolor vs not-versicolor

virginica vs not-virginica
```

This allows us to calculate precision, recall, and F1 for each species.

---

# 43. Per-Class Metrics

A classification report might look like:

```text
              precision    recall    f1-score

setosa           ...
versicolor       ...
virginica        ...
```

Each row describes the model's performance for one class.

This is useful because the model may perform differently for different classes.

For example:

```text
setosa       F1 = 0.99
versicolor   F1 = 0.91
virginica    F1 = 0.93
```

This suggests that setosa is easier for the model to distinguish than the other species.

---

# 44. Macro Average

When we have several classes, we may want one overall summary of the class-level metrics.

The **macro average** calculates the metric separately for each class and then gives every class equal importance.

For example:

```text
F1 setosa       = 0.99
F1 versicolor   = 0.91
F1 virginica    = 0.93
```

The macro F1 is:

```text
(0.99 + 0.91 + 0.93) / 3
```

Every class receives equal weight.

This makes macro averaging useful when we care about performance across all classes equally.

---

# 45. Weighted Average

The **weighted average** also combines the class-level metrics, but gives each class weight according to the number of observations it contains.

Suppose we had:

```text
Class A → 90 observations
Class B → 8 observations
Class C → 2 observations
```

Class A would have much more influence on the weighted average.

For Iris, the three species each have 50 observations.

Therefore, the classes are balanced, and:

```text
macro average ≈ weighted average
```

This is an important interpretation point.

---

# 46. What Does "Better Model" Mean?

Suppose we compare two classifiers:

```text
Model A → accuracy = 0.94
Model B → accuracy = 0.92
```

It is tempting to immediately say that Model A is better.

But model comparison should consider more than one number.

We can examine:

* accuracy
* confusion matrix
* precision
* recall
* F1
* performance for individual classes

We should also consider whether the model is appropriate and understandable for the problem.

For example, a slightly less accurate decision tree might be easier to explain than a more complex model.

The best model is not always the one with the highest single metric.

---

# 47. Overfitting

A model can perform extremely well on the training data but poorly on new data.

This is called **overfitting**.

The model has learned the training examples too closely rather than learning patterns that generalize.

Conceptually:

```text
Training performance
        ↑
       very high

Test performance
        ↓
       much lower
```

This is why we evaluate models on data that was not used for training.

A simple model can sometimes generalize better than a very complicated model.

---

# 48. Underfitting

The opposite problem is **underfitting**.

An underfitted model is too simple to capture useful patterns in the data.

Conceptually:

```text
Training performance → poor
Test performance     → poor
```

So we want a model that learns useful patterns without simply memorizing the training data.

This is part of the broader idea of **generalization**:

> **A good machine-learning model should perform well on new data, not just on the examples it was trained on.**

---

# 49. A Simple Classification Workflow

Across both labs, the main workflow is:

```text
1. Understand the problem
          ↓
2. Identify features and target
          ↓
3. Split into training and test data
          ↓
4. Prepare the features
          ↓
5. Establish a baseline
          ↓
6. Train a classifier
          ↓
7. Make predictions
          ↓
8. Evaluate the predictions
          ↓
9. Compare models
          ↓
10. Interpret the results
```

This is the workflow students should remember.

---

# 50. The Workflow in Lab 1

In the Titanic lab:

```text
Titanic
   ↓
Binary classification
   ↓
X = passenger information
y = survival
   ↓
Train/test split
   ↓
Handle missing values
Encode categorical variables
   ↓
Baseline
   ↓
Logistic Regression
   ↓
Decision Tree
   ↓
Predictions
   ↓
Accuracy
Confusion matrix
Precision
Recall
F1
   ↓
Compare models
```

The main challenge is that Titanic contains mixed types of data and missing values.

---

# 51. The Workflow in Lab 2

In the Iris lab:

```text
Iris
   ↓
Multiclass classification
   ↓
X = flower measurements
y = species
   ↓
Train/test split
   ↓
Minimal preprocessing
   ↓
Baseline
   ↓
KNN
   ↓
Logistic Regression
   ↓
Predictions
   ↓
Accuracy
Confusion matrix
Per-class metrics
Macro / weighted averages
   ↓
Compare models
```

The main new idea is that there are **three classes rather than two**.

---

# 52. What Changes from Binary to Multiclass?

The fundamental workflow stays the same.

What changes is mainly the interpretation.

### Binary classification

There are two classes:

```text
0 / 1
```

We can naturally talk about:

```text
positive
negative
```

and:

```text
TP
TN
FP
FN
```

### Multiclass classification

There are three or more classes:

```text
0 / 1 / 2 / ...
```

There is no single natural positive class.

Instead, we evaluate each class separately.

The confusion matrix also becomes larger.

```text
2 classes → 2 × 2

3 classes → 3 × 3

4 classes → 4 × 4
```

---

# 53. A Useful Mental Model

Students should be able to think about classification in four stages.

## Stage 1 — The problem

> What are we trying to predict?

Example:

```text
Titanic → survival
Iris → species
```

## Stage 2 — The information

> What information can we use to make the prediction?

Example:

```text
Titanic → age, sex, class, fare, ...
Iris → sepal and petal measurements
```

## Stage 3 — The model

> What algorithm will learn from these features?

Examples:

```text
Logistic Regression
Decision Tree
KNN
```

## Stage 4 — The evaluation

> How well did the model predict unseen data?

Examples:

```text
Accuracy
Confusion matrix
Precision
Recall
F1
```

---

# 54. What We Are Not Covering Yet

Machine learning contains many additional concepts that are important, but they are not necessary for understanding these first two classification labs.

Later, you may encounter:

### Cross-validation

A technique for obtaining a more reliable estimate of model performance by repeatedly training and validating models on different subsets of the training data.

### Hyperparameter tuning

Systematically trying different model settings to find a good configuration.

### Classification thresholds

Changing the probability cutoff used to turn predicted probabilities into class predictions.

### ROC curves and ROC-AUC

Tools for evaluating binary classifiers across different classification thresholds.

### Precision-recall curves

Another way to examine the trade-off between precision and recall.

### Pipelines

A scikit-learn tool for combining preprocessing and modelling steps into a single object.

Pipelines are especially useful in larger or more complex projects because they help prevent mistakes such as inconsistent preprocessing and data leakage.

### More advanced classification

Later courses may also introduce:

* multiclass strategies in more detail
* multilabel classification
* multioutput classification
* ensemble methods
* support vector machines
* neural networks

These are useful topics, but they are **not required to understand the two introductory labs**.

---

# 55. Key Concepts to Remember

Before starting the labs, make sure you understand these ideas.

### Supervised learning

Learning from examples where the correct target is known.

### Classification

Predicting a category or class.

### Binary classification

Classification with two possible classes.

Example:

```text
Titanic → survived / did not survive
```

### Multiclass classification

Classification with more than two possible classes.

Example:

```text
Iris → setosa / versicolor / virginica
```

### Features

The input variables used to make predictions.

```text
X
```

### Target

The variable we want to predict.

```text
y
```

### Training data

Data used to fit the model.

### Test data

Previously unseen data used to evaluate the model.

### Baseline

A simple reference prediction against which a real model can be compared.

### Classifier

A model that predicts a class.

### Logistic Regression

A classifier that models the relationship between features and class probabilities.

### Decision Tree

A classifier that makes predictions through a sequence of feature-based decisions.

### KNN

A classifier that predicts a class based on the classes of nearby training observations.

### Accuracy

The proportion of predictions that are correct.

### Confusion matrix

A table showing actual classes versus predicted classes.

### Precision

Of the observations predicted as a class, how many actually belong to that class?

### Recall

Of the observations that actually belong to a class, how many did the model identify?

### F1 score

A measure that combines precision and recall.

### Macro average

Gives every class equal importance when combining class-level metrics.

### Weighted average

Weights each class according to its number of observations.

### Overfitting

When a model fits the training data too closely and does not generalize well to new data.

---

# 56. Final Picture

The two labs can now be viewed as two examples of the same fundamental idea.

```text
                    CLASSIFICATION
                          │
             ┌────────────┴────────────┐
             ↓                         ↓
          Titanic                     Iris
             ↓                         ↓
          Binary                   Multiclass
          2 classes                3 classes
             ↓                         ↓
       ┌─────┴─────┐             ┌─────┴─────┐
       ↓           ↓             ↓           ↓
 Logistic      Decision         KNN       Logistic
 Regression      Tree                     Regression
       │           │             │           │
       └─────┬─────┘             └─────┬─────┘
             ↓                         ↓
             └──────────┬──────────────┘
                        ↓
                    Predictions
                        ↓
                  Evaluation
                        ↓
             ┌──────────┼──────────┐
             ↓          ↓          ↓
          Accuracy  Confusion    Precision
                     Matrix       Recall
                                  F1
                        ↓
                  Model comparison
```

The most important idea is not to memorize individual algorithms.

It is to understand the overall reasoning:

> **We have examples with known answers. We use features to train a classifier, use that classifier to make predictions for unseen observations, and evaluate how well those predictions match the actual outcomes.**

Titanic teaches this idea with **two classes**.

Iris extends the same idea to **three classes**.

The algorithms are different ways of learning the relationship between the features and the target.

----

# Links

- scikit-learn
  - [Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
  - [Supervised learning](https://scikit-learn.org/stable/supervised_learning.html)
- [Introduction to Supervised Learning](https://developers.google.com/machine-learning/intro-to-ml/supervised)
- [Classification](https://developers.google.com/machine-learning/crash-course/classification)