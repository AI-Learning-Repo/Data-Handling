# Lab 1: Introduction to Binary Classification with the Titanic Dataset

## Predicting Passenger Survival

---

# Lab Purpose

In the previous Titanic activity, you explored the dataset using **exploratory data analysis (EDA)**.

You investigated questions such as:

* What variables are in the dataset?
* Are there missing values?
* What are the distributions of the variables?
* Are some variables related to survival?

In this lab, we take the next step.

Instead of only asking:

> **What does the data tell us?**

we ask:

> **Can we use the data to make predictions?**

Our goal is to build a model that predicts whether a Titanic passenger **survived** or **did not survive**.

---

# Learning Outcomes

By the end of this lab, you should be able to:

1. explain what classification is
2. distinguish classification from regression
3. identify features and target variables
4. explain the difference between training and test data
5. prepare numerical and categorical data for a classifier
6. explain the idea of a baseline
7. train a logistic regression classifier
8. make predictions with a classifier
9. evaluate predictions using:

   * accuracy
   * precision
   * recall
   * F1 score
10. interpret a confusion matrix
11. explain the basic idea of a decision tree
12. compare two classification models

---

# 1. What Is Classification?

## From Prediction to Classification

Machine learning can be used for different types of prediction problems.

Suppose we want to predict the **price of a house**.

The answer might be:

```text
€350,000
```

This is a numerical value.

This type of problem is called **regression**.

Now suppose we want to predict whether a Titanic passenger survived.

The answer is one of two categories:

```text
Did not survive
Survived
```

This is a **classification** problem.

### Regression vs Classification

| Problem        | What is predicted? | Example                     |
| -------------- | ------------------ | --------------------------- |
| Regression     | A numerical value  | House price                 |
| Classification | A category/class   | Spam or not spam            |
| Classification | A category/class   | Disease or no disease       |
| Classification | A category/class   | Survived or did not survive |

The key difference is:

> **Regression predicts a number. Classification predicts a class.**

---

# 2. What Is Binary Classification?

Classification problems can have different numbers of classes.

For example:

```text
Spam detection

0 → Not spam
1 → Spam
```

There are two classes.

This is **binary classification**.

The Titanic problem is also binary classification:

```text
0 → Did not survive
1 → Survived
```

There are exactly two possible outcomes.

Therefore:

> **Titanic survival prediction is a binary classification problem.**

### Question

Why is the Titanic problem a classification problem rather than a regression problem?

<details>
<summary><strong>Sample answer</strong></summary>

The Titanic problem is a classification problem because the model predicts one of two categories: survived or did not survive. It is not predicting a continuous numerical value.

</details>

---

# 3. What Is Supervised Learning?

Classification is often performed using **supervised learning**.

In supervised learning, we train a model using examples where we already know the correct answer.

For the Titanic dataset, we know:

```text
Passenger information        Actual outcome

Age = 22                     Did not survive
Sex = male                  

Age = 38                     Survived
Sex = female
```

The model uses many such examples to learn patterns.

Conceptually:

```text
Passenger information
        +
Known survival outcome
        ↓
      Model
        ↓
Learns patterns
```

Once the model has learned from the training data, we can give it a new passenger and ask:

```text
Passenger information
        ↓
      Model
        ↓
Predicted survival
```

### Question

Why is Titanic survival prediction a supervised-learning problem?

<details>
<summary><strong>Sample answer</strong></summary>

It is supervised learning because the training data contains the known survival outcome for each passenger. The model can use these examples to learn a relationship between passenger characteristics and survival.

</details>

---

# 4. Features and Target

A machine-learning problem usually has two important parts:

### Features

Features are the information we give to the model.

For Titanic, examples include:

```text
pclass
sex
age
sibsp
parch
fare
embarked
```

We usually call the features:

```python
X
```

### Target

The target is what we want the model to predict.

For Titanic:

```text
survived
```

We usually call the target:

```python
y
```

So:

```text
X → passenger information

y → survival outcome
```

### Code

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

### What does this code do?

The first part creates a list containing the columns we want to use.

```python
features = [...]
```

Then:

```python
X = df[features].copy()
```

creates a new DataFrame containing only those columns.

Finally:

```python
y = df["survived"].astype(int)
```

selects the target column and converts it to integers.

### Question

What do `X` and `y` represent in this problem?

<details>
<summary><strong>Sample answer</strong></summary>

`X` contains the passenger features that are used to make predictions. `y` contains the survival outcome that the model is trying to predict.

</details>

---

# 5. Load the Dataset

We will use the Titanic dataset from OpenML.

```python
from sklearn.datasets import fetch_openml

titanic = fetch_openml(
    name="titanic",
    version=1,
    as_frame=True
)

df = titanic.frame.copy()

display(df.head())
```

### What does this code do?

```python
fetch_openml(...)
```

downloads the dataset from OpenML.

```python
as_frame=True
```

asks scikit-learn to return the data as pandas DataFrames.

```python
df = titanic.frame.copy()
```

stores a copy of the dataset in `df`.

We can check the data:

```python
print(df.shape)
print(df.columns)
```

You have already performed detailed EDA on this dataset, so we will not repeat that work here.

---

# 6. Select Features and Target

For this lab, we will use:

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

Check the result:

```python
print("X shape:", X.shape)
print("y shape:", y.shape)
```

You should think of the data as:

```text
X
↓
Information about passengers

y
↓
Correct survival outcome
```

---

# 7. Why Do We Split the Data?

We want to know whether our model can make predictions for passengers it has **not seen before**.

If we trained and tested the model on exactly the same passengers, the evaluation would not tell us much about its ability to generalize.

Instead, we divide the data into two parts:

```text
Training data
    ↓
Used to learn the model

Test data
    ↓
Used to evaluate the model
```

We will use:

* 80% training data
* 20% test data

### Code

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### What does the code mean?

```python
test_size=0.2
```

means that 20% of the observations are placed in the test set.

```python
random_state=42
```

makes the split reproducible. If we run the code again, we get the same split.

```python
stratify=y
```

helps maintain approximately the same proportion of the two survival classes in the training and test sets.

We can check the sizes:

```python
print("Training:", X_train.shape)
print("Test:", X_test.shape)
```

### Question

Why should the test data not be used to train the model?

<details>
<summary><strong>Sample answer</strong></summary>

The test data should represent unseen data. If we use it to train the model, we can no longer use it as an independent test of how well the model generalizes to new observations.

</details>

---

# 8. Preparing the Data for a Classifier

Our features are not all the same type.

We have numerical variables:

```text
age
sibsp
parch
fare
```

and categorical variables:

```text
pclass
sex
embarked
```

A classifier needs numerical input, so we need to prepare these variables.

We will do this manually so that we can see what is happening.

---

## 8.1 Numerical Features

First, select the numerical variables:

```python
numeric_features = [
    "age",
    "sibsp",
    "parch",
    "fare"
]
```

### Missing values

`age` contains missing values.

We will replace missing numerical values with the **median** calculated from the training data.

```python
from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(
    strategy="median"
)

X_train_num = num_imputer.fit_transform(
    X_train[numeric_features]
)

X_test_num = num_imputer.transform(
    X_test[numeric_features]
)
```

### Why are there two different methods?

For the training data:

```python
fit_transform()
```

means:

1. calculate the median from the training data
2. use that median to fill missing values

For the test data:

```python
transform()
```

means:

> use the median that was already calculated from the training data.

We do **not** calculate a new median from the test data.

This keeps the test data independent.

---

## 8.2 Standardizing Numerical Features

The numerical variables have very different scales.

For example:

```text
age  → approximately 0–80

fare → can be much larger
```

Standardization puts numerical variables onto a comparable scale.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_num = scaler.fit_transform(
    X_train_num
)

X_test_num = scaler.transform(
    X_test_num
)
```

Again, the scaler is fitted using the training data and then applied to the test data.

---

# 9. Preparing Categorical Features

Now select the categorical variables:

```python
categorical_features = [
    "pclass",
    "sex",
    "embarked"
]
```

A classifier cannot directly work with text such as:

```text
male
female
```

We therefore convert the categories into numerical columns.

First, deal with missing values:

```python
cat_imputer = SimpleImputer(
    strategy="most_frequent"
)

X_train_cat = cat_imputer.fit_transform(
    X_train[categorical_features]
)

X_test_cat = cat_imputer.transform(
    X_test[categorical_features]
)
```

The most frequent category is used to replace missing categorical values.

---

## One-Hot Encoding

Now convert the categories into numerical indicator columns.

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

X_train_cat = encoder.fit_transform(
    X_train_cat
)

X_test_cat = encoder.transform(
    X_test_cat
)
```

For example, a variable such as:

```text
sex

male
female
```

can become:

```text
sex_female    sex_male

     1            0
     0            1
```

The model can now work with these numerical values.

---

# 10. Combine the Prepared Features

We now have:

```text
X_train_num → prepared numerical features

X_train_cat → prepared categorical features
```

We combine them:

```python
import numpy as np

X_train_processed = np.hstack([
    X_train_num,
    X_train_cat
])

X_test_processed = np.hstack([
    X_test_num,
    X_test_cat
])
```

`np.hstack()` places the columns next to each other.

Conceptually:

```text
Numerical features
        +
Categorical features
        ↓
Prepared feature matrix
        ↓
Classifier
```

Check the shape:

```python
print("Training data:", X_train_processed.shape)
print("Test data:", X_test_processed.shape)
```

At this point, the data is ready for a classification model.

---

# 11. What Is a Classification Model?

A classification model learns a relationship between:

```text
Features → Class
```

For Titanic:

```text
Passenger information
        ↓
Classification model
        ↓
0 = Did not survive
1 = Survived
```

The model does not simply memorize one answer for every passenger.

It attempts to learn patterns in the training data.

For example, it may learn that variables such as passenger class, sex, age, and fare are useful for predicting survival.

The model can then apply what it learned to passengers in the test set.

---

# 12. What Is a Baseline?

Before using a machine-learning model, we need a simple point of comparison.

This is called a **baseline**.

A baseline is a simple strategy that gives us a minimum level of performance to compare against.

For Titanic, a very simple baseline is:

> **Always predict the most common class.**

Suppose the training data contains:

```text
Class 0 → 62%
Class 1 → 38%
```

The baseline simply predicts:

```text
0
0
0
0
0
...
```

for every passenger.

It does not learn a relationship between passenger characteristics and survival.

It simply says:

> "I'll always guess the class that occurs most often."

---

## Create the Baseline

Find the most common class:

```python
majority_class = y_train.mode()[0]

print("Majority class:", majority_class)
```

Create predictions:

```python
y_baseline = np.full(
    len(y_test),
    majority_class
)
```

This creates an array containing the same prediction for every test passenger.

Calculate its accuracy:

```python
from sklearn.metrics import accuracy_score

baseline_accuracy = accuracy_score(
    y_test,
    y_baseline
)

print("Baseline accuracy:", baseline_accuracy)
```

### Why do we need a baseline?

Imagine:

```text
Baseline accuracy = 0.62

Model accuracy = 0.80
```

The model has clearly improved over the simple strategy.

But if:

```text
Baseline accuracy = 0.62

Model accuracy = 0.63
```

the model has provided very little improvement.

Therefore:

> **A model should be evaluated relative to a simple baseline, not only by looking at its accuracy in isolation.**

### Question

Why might a simple baseline have reasonably high accuracy?

<details>
<summary><strong>Sample answer</strong></summary>

If one class is more common than the other, always predicting the majority class will already produce a relatively high accuracy. For example, if 62% of passengers belong to class 0, always predicting class 0 gives 62% accuracy without learning any useful relationships.

</details>

---

# 13. Logistic Regression

Our first real classifier will be **logistic regression**.

## What is logistic regression?

Logistic regression is a classification algorithm commonly used when the target has two classes.

For this problem, the two classes are:

```text
0 → Did not survive
1 → Survived
```

The model estimates the probability that an observation belongs to the positive class.

For example:

```text
Passenger A → probability of survival = 0.82

Passenger B → probability of survival = 0.31
```

A probability close to 1 means the model considers survival more likely.

A probability close to 0 means the model considers survival less likely.

The model then converts the probability into a class prediction.

A common rule is:

```text
probability ≥ 0.50 → class 1
probability <  0.50 → class 0
```

For example:

```text
0.82 → 1 → survived
0.31 → 0 → did not survive
```

---

# 14. Train the Logistic Regression Model

Import the model:

```python
from sklearn.linear_model import LogisticRegression
```

Create the model:

```python
logreg = LogisticRegression(
    max_iter=1000
)
```

The `max_iter=1000` setting allows the algorithm enough iterations to find a solution.

Now train the model:

```python
logreg.fit(
    X_train_processed,
    y_train
)
```

### What does `fit()` mean?

`fit()` means:

> **Learn from the training data.**

The model receives:

```text
X_train_processed
        +
y_train
```

and learns patterns that connect the features to the target.

---

# 15. Make Predictions

Now ask the trained model to predict the test passengers:

```python
y_pred = logreg.predict(
    X_test_processed
)
```

The result contains predictions such as:

```text
0
1
1
0
0
1
...
```

Each value represents the model's predicted class.

We can compare some predictions with the actual values:

```python
print("Predicted:")
print(y_pred[:20])

print("\nActual:")
print(y_test.iloc[:20].values)
```

Remember:

```text
0 → Did not survive

1 → Survived
```

---

# 16. How Do We Know Whether the Model Is Good?

We compare:

```text
Predicted class
       vs
Actual class
```

This allows us to calculate classification metrics.

We will use four:

1. Accuracy
2. Precision
3. Recall
4. F1 score

Different metrics tell us different things about the classifier.

---

# 17. Accuracy

Accuracy is the proportion of predictions that were correct.

```text
Accuracy =
number of correct predictions
------------------------------
number of all predictions
```

Calculate it:

```python
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Accuracy:", accuracy)
```

For example, an accuracy of:

```text
0.80
```

means that approximately 80% of the predictions were correct.

### Limitation of accuracy

Accuracy does not tell us **what kinds of mistakes** the model made.

For that, we need additional metrics and the confusion matrix.

---

# 18. Precision

Precision asks:

> **When the model predicts that a passenger survived, how often is that prediction correct?**

For Titanic:

```text
Predicted survived
        ↓
How many actually survived?
```

Calculate it:

```python
from sklearn.metrics import precision_score

precision = precision_score(
    y_test,
    y_pred
)

print("Precision:", precision)
```

For example, if:

```text
Precision = 0.82
```

we can say:

> Among the passengers predicted to have survived, approximately 82% actually survived.

---

# 19. Recall

Recall asks:

> **Of all the passengers who actually survived, how many did the model identify?**

Calculate it:

```python
from sklearn.metrics import recall_score

recall = recall_score(
    y_test,
    y_pred
)

print("Recall:", recall)
```

For example:

```text
Recall = 0.74
```

means:

> The model correctly identified approximately 74% of the passengers who actually survived.

---

# 20. F1 Score

Precision and recall measure different aspects of performance.

The F1 score combines them into one measure.

Calculate it:

```python
from sklearn.metrics import f1_score

f1 = f1_score(
    y_test,
    y_pred
)

print("F1 score:", f1)
```

The F1 score is useful when we want a balance between precision and recall.

We do not need to calculate the formula by hand for this lab.

The important idea is:

```text
Precision → How reliable are positive predictions?

Recall    → How many positive cases did we find?

F1        → Balance between precision and recall
```

---

# 21. Calculate All Metrics Together

Let's calculate all four metrics:

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

### Task

Write a short interpretation of your model's precision and recall.

<details>
<summary><strong>Sample answer</strong></summary>

My model has a precision of **[your value]**. This means that among the passengers predicted to have survived, approximately **[your percentage]%** actually survived.

The model has a recall of **[your value]**. This means that it correctly identified approximately **[your percentage]%** of the passengers who actually survived.

</details>

---

# 22. The Confusion Matrix

Metrics such as accuracy give us a summary.

A **confusion matrix** lets us look more closely at the predictions.

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)
```

Visualize it:

```python
sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

---

## Understanding the Four Outcomes

A binary classifier can make four types of predictions.

```text
                    Predicted
                  0          1

Actual  0       TN         FP

        1       FN         TP
```

### True Negative — TN

The model predicted:

```text
0 → did not survive
```

and the passenger actually did not survive.

### True Positive — TP

The model predicted:

```text
1 → survived
```

and the passenger actually survived.

### False Positive — FP

The model predicted:

```text
1 → survived
```

but the passenger actually did not survive.

### False Negative — FN

The model predicted:

```text
0 → did not survive
```

but the passenger actually survived.

---

# 23. Why Does the Confusion Matrix Matter?

Consider two models:

```text
Model A
Accuracy = 80%

Model B
Accuracy = 80%
```

They have the same accuracy.

But they might make different mistakes.

For example:

```text
Model A
Many false positives
Few false negatives

Model B
Few false positives
Many false negatives
```

The confusion matrix helps us see this difference.

### Question

In the Titanic context, what is a false positive and what is a false negative?

<details>
<summary><strong>Sample answer</strong></summary>

A false positive occurs when the model predicts that a passenger survived, but the passenger actually did not survive.

A false negative occurs when the model predicts that a passenger did not survive, but the passenger actually survived.

</details>

---

# 24. A Second Classifier: Decision Trees

So far, we have used logistic regression.

Now we will try a different type of classifier: a **decision tree**.

## What Is a Decision Tree?

A decision tree makes predictions by asking a sequence of questions about the data.

You can think of it like a flowchart.

For example, a simplified Titanic tree might look like:

```text
             Sex?
           /     \
       female     male
         /          \
      Survive?     Pclass?
                   /     \
                  1      3
                 /        \
             Survive?    Did not survive
```

The actual tree will learn its own questions from the training data.

A decision tree tries to divide the observations into groups that have increasingly similar target values.

For example, it may discover that certain combinations of:

* sex
* passenger class
* age
* fare

are useful for predicting survival.

---

# 25. Why Try Another Classifier?

Different algorithms make different assumptions and learn patterns differently.

### Logistic regression

Logistic regression learns a relationship between the features and the probability of belonging to a class.

### Decision tree

A decision tree creates a sequence of rules that divide the observations into groups.

Neither method is automatically "better."

We need to compare their performance.

---

# 26. Train a Decision Tree

Import the classifier:

```python
from sklearn.tree import DecisionTreeClassifier
```

Create the model:

```python
tree = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)
```

The important parameter here is:

```python
max_depth=4
```

This limits how deep the tree can become.

A very deep tree can become extremely complicated and may learn the training data too closely.

For this introductory activity, we keep the tree relatively small.

Train it:

```python
tree.fit(
    X_train_processed,
    y_train
)
```

Again:

```python
fit()
```

means:

> Learn from the training data.

---

# 27. Make Decision Tree Predictions

```python
y_tree_pred = tree.predict(
    X_test_processed
)
```

Now calculate its metrics:

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
print("F1       :", tree_f1)
```

---

# 28. Compare the Models

Let's put the results into a table:

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

Your table should look something like:

```text
Model                 Accuracy    Precision    Recall    F1

Baseline                ...          -          -       -

Logistic Regression     ...         ...         ...     ...

Decision Tree           ...         ...         ...     ...
```

The exact values depend on the data and model.

---

# 29. Interpreting the Comparison

Do not immediately choose the model with the highest accuracy.

Ask:

* Is it substantially better than the baseline?
* Which model has better precision?
* Which has better recall?
* Which has better F1?
* Are the differences large or small?

A model with slightly higher accuracy is not automatically the best model.

The appropriate choice depends on what we care about.

For this introductory exercise, we will focus particularly on **F1, precision, recall, and accuracy**.

### Questions

1. Which model performed best according to F1?
2. Which model would you choose? Give two reasons based on the results.

<details>
<summary><strong>Sample answer</strong></summary>

The model with the highest F1 score was **[model]**.

I would choose **[model]** because it achieved **[higher F1 / higher recall / higher precision / higher accuracy]** than the other classifier. It also performed better than the baseline, suggesting that it learned useful patterns from the passenger data.

Another reasonable answer may be possible if it is supported by the results.

</details>

---

# 30. What Happens When We Give the Model a Passenger?

The ultimate purpose of classification is prediction.

Suppose we have information about a new passenger:

```text
pclass = 3
sex = male
age = 25
sibsp = 0
parch = 0
fare = 8
embarked = S
```

We could prepare this passenger using the **same preprocessing steps learned from the training data** and then give the resulting data to the trained classifier.

The classifier would return:

```text
0 → predicted not to survive

or

1 → predicted to survive
```

The important idea is:

> **A trained classifier takes feature information as input and produces a predicted class as output.**

---

# 31. The Complete Classification Workflow

We can now summarize the entire lab:

```text
Titanic dataset
      ↓
Choose features and target
      ↓
Split into training and test data
      ↓
Prepare the features
      ↓
Establish a baseline
      ↓
Train a classifier
      ↓
Make predictions
      ↓
Evaluate predictions
      ↓
Examine the confusion matrix
      ↓
Try another classifier
      ↓
Compare the results
```

The key concepts are:

### 1. Classification

Predicting a category rather than a numerical value.

### 2. Binary classification

There are two possible classes.

### 3. Features

The information used to make the prediction.

```text
X
```

### 4. Target

The outcome we want to predict.

```text
y
```

### 5. Training data

Used to learn the model.

### 6. Test data

Used to evaluate the model on unseen examples.

### 7. Baseline

A simple strategy used as a reference point.

### 8. Logistic regression

A classification algorithm that estimates the probability of belonging to a class.

### 9. Decision tree

A classification algorithm that makes predictions using a sequence of learned rules.

### 10. Classification metrics

Different ways of measuring model performance.

### 11. Confusion matrix

Shows the types of correct and incorrect predictions.

---

# 32. Final Reflection

Answer the following three questions.

## Question 1 — Classification

Explain why Titanic survival prediction is a binary classification problem.

<details>
<summary><strong>Sample answer</strong></summary>

Titanic survival prediction is binary classification because the target variable has two possible classes: 0, meaning the passenger did not survive, and 1, meaning the passenger survived.

</details>

---

## Question 2 — Model Performance

Is your best classifier meaningfully better than the baseline?

Use at least two metrics in your answer.

<details>
<summary><strong>Sample answer structure</strong></summary>

The baseline accuracy was **[value]**, while the best classifier achieved an accuracy of **[value]**.

The classifier also achieved an F1 score of **[value]**, showing that it provided a better balance between precision and recall.

Therefore, I would say that the classifier **[clearly improved / somewhat improved / did not substantially improve]** over the baseline.

</details>

---

## Question 3 — Model Choice

Which classifier would you choose: logistic regression or decision tree?

Explain your choice using evidence from the comparison table.

<details>
<summary><strong>Sample answer structure</strong></summary>

I would choose **[logistic regression / decision tree]**.

It achieved an accuracy of **[value]**, a precision of **[value]**, a recall of **[value]**, and an F1 score of **[value]**.

Compared with the other model, I consider this model preferable because **[explanation]**.

</details>

---

# Deliverables

Submit:

1. Your completed notebook
2. Your answers to the questions
3. The classification metrics for logistic regression
4. The logistic regression confusion matrix
5. The model comparison table
6. Your final explanation of which classifier you would choose

---

# Looking Ahead: Multiclass Classification

In this lab, we predicted between two classes:

```text
0 → Did not survive
1 → Survived
```

This is **binary classification**.

In the next lab, we will use the Iris dataset.

Iris contains three possible species:

```text
Iris setosa
Iris versicolor
Iris virginica
```

Now the model has to choose between **three classes** rather than two.

This is called:

> **multiclass classification**

The basic idea remains the same:

```text
Features
   ↓
Classifier
   ↓
Predicted class
```

But the evaluation and prediction process becomes slightly more interesting when there are more than two possible classes.
