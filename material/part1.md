# Theory Draft: Classification with Titanic First, Then Iris

## Purpose of This Draft

This draft adapts the classification theory so that it follows the same teaching order as the labs:

1. Titanic first for binary classification
2. Iris second for multiclass classification

It also reduces the time spent on Titanic EDA in this week's classification material.

That earlier EDA work was already done in last week's Titanic activity. In this week's theory and labs, Titanic is used mainly as a bridge from data preparation to supervised classification.

## 1. Classification in Supervised Learning

In supervised learning, a model learns from labeled examples.

Two of the most common supervised-learning tasks are:

- regression, which predicts a numerical value
- classification, which predicts a class or category

This week's topic is classification.

The teaching sequence should be:

- start with Titanic as a binary classification example
- then move to Iris as a multiclass classification example

This ordering is useful because Titanic connects directly to last week's work on understanding data, EDA, and cleaning, while Iris provides a cleaner setting for multiclass ideas.

## 2. Why Start with Titanic?

Titanic is the better first example for this week because it creates continuity.

Students already know the dataset structure from the earlier activity. That means the theory can move more quickly into classification concepts without repeating a long EDA discussion.

### Teaching decision

This week should not spend much time on Titanic EDA.

Instead:

- briefly remind students that EDA and data cleaning were covered last week
- use a focused OpenML Titanic dataset for classification
- perform only a short validation check of columns, data types, missing values, and class balance
- move quickly into target definition, preprocessing, modeling, and evaluation

## 3. What Is Classification?

Classification is the task of predicting a discrete label.

Examples for this week:

- Titanic: predict whether a passenger survived
- Iris: predict which flower species is observed

Classification problems can take several forms:

- binary classification: two classes
- multiclass classification: more than two classes
- multilabel classification: one instance can receive several labels
- multioutput classification: multiple outputs are predicted at once

For this stage of the course, the focus should remain on binary and multiclass classification.

## 4. Titanic as the Binary Classification Entry Point

Titanic should introduce the first core classification ideas.

### Why Titanic works well here

- the target is easy to explain: `survived`
- the dataset includes both numerical and categorical features
- students already encountered its data-quality issues earlier
- it allows the course to connect preparation decisions with later modeling decisions

### Positive and negative classes

- positive class: survived
- negative class: did not survive

This makes Titanic a straightforward binary classification example.

## 5. Minimal Data Preparation for Titanic This Week

The theory should explicitly state that this week is not a repeat of the full Titanic cleaning activity.

A practical classification-oriented workflow is:

1. load the Titanic dataset from OpenML
2. keep a focused subset of useful features such as `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, and `embarked`
3. define the target `survived`
4. do a brief validation check
5. split into training and test sets
6. build preprocessing into a pipeline
7. train classifiers and evaluate them

### Brief validation check instead of full EDA

The validation check should be limited to:

- confirming that the target and selected features exist
- checking numerical versus categorical variables
- checking whether `age` or `embarked` contain missing values
- checking whether the target classes are balanced

That is enough for this week.

## 6. Preparing Data for Machine Learning Algorithms

Before training a classifier, the data must be prepared in a form the model can use.

This connects directly to last week's work, but the focus now is operational rather than exploratory.

### Titanic preparation ideas

- separate features from the target
- impute missing numeric values such as `age`
- impute missing categorical values such as `embarked`
- encode categorical variables such as `sex` and `embarked`
- scale numeric variables when appropriate for the model
- ensure preprocessing is fit only on the training data

This is a useful place to introduce data leakage at a high level.

## 7. Select and Train a Model

Once the data is prepared, the next step is to choose a classifier and train it.

At a conceptual level:

- `fit()` trains the model on labeled data
- `predict()` returns class predictions
- some models also return predicted probabilities or decision scores

Good classifier examples for this week include:

- `LogisticRegression`
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `KNeighborsClassifier`
- `SVC`

For Titanic, a strong teaching sequence is:

1. `DummyClassifier` baseline
2. `LogisticRegression` as the first real model
3. one comparison model such as `DecisionTreeClassifier`

## 8. Why a Baseline Matters

Before students trust a real classifier, they should compare it with a trivial strategy.

Titanic is a good example because the majority class can already produce a deceptively reasonable accuracy score.

This makes an important theoretical point:

- accuracy alone can be misleading
- a classifier must be better than a naive baseline in a meaningful way

## 9. Performance Measures Beyond Accuracy

Classification evaluation is richer than regression evaluation because the kinds of mistakes matter.

### Accuracy

Accuracy is the proportion of correct predictions.

It is useful, but it should not be the only metric.

### Why Titanic is useful here

With Titanic, students can see that a model may seem strong in terms of accuracy while still making poor decisions about the positive class.

This creates a natural reason to move to better metrics.

## 10. Cross-Validation

Cross-validation gives a stronger estimate of model performance than a single split.

Main idea:

- divide the training data into folds
- train on some folds and validate on the remaining fold
- repeat the process
- average the results

For classification, stratified cross-validation is especially useful because it preserves class proportions more consistently across folds.

This idea should be introduced during Titanic before moving to multiclass classification.

## 11. Confusion Matrix

The confusion matrix is one of the most important concepts in classification.

For binary classification, it introduces:

- true positives
- true negatives
- false positives
- false negatives

### Titanic interpretation

- true positive: the passenger survived and the model predicted survived
- true negative: the passenger did not survive and the model predicted did not survive
- false positive: the model predicted survived but the passenger did not survive
- false negative: the model predicted did not survive but the passenger survived

The confusion matrix makes classifier behavior more visible than accuracy alone.

## 12. Precision, Recall, and F1 Score

These metrics should be introduced through Titanic.

### Precision

Precision answers:

Of all the passengers predicted to survive, how many actually survived?

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

### Recall

Recall answers:

Of all the passengers who actually survived, how many did the model detect?

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

### F1 score

The F1 score combines precision and recall into one measure.

$$
F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

### Teaching point

Students should see that two models with similar accuracy may still differ substantially in precision, recall, and F1.

## 13. The Precision/Recall Trade-off

Titanic is also a good dataset for introducing threshold-based decisions.

Many classifiers output predicted probabilities.

If the positive-class probability is above a chosen threshold, the model predicts the positive class.

Changing that threshold changes model behavior:

- a higher threshold often increases precision and decreases recall
- a lower threshold often increases recall and decreases precision

### Titanic interpretation

If the model predicts survival only when it is very confident, it may reduce false positives but miss more actual survivors.

This section should stay practical and interpretive rather than mathematical.

## 14. ROC Curve and ROC AUC

After precision and recall, the theory can introduce the ROC curve briefly.

The ROC curve compares:

- true positive rate
- false positive rate

The ROC AUC summarizes classifier separation across thresholds.

At this course stage, ROC should be presented as:

- a threshold-aware comparison tool
- a high-level summary of binary classifier quality
- a complement to, not a replacement for, precision and recall

## 15. Transition from Titanic to Iris

Once the core ideas of binary classification are established, the theory should move to Iris.

This transition matters because students have now seen:

- how to define a target
- how to build a baseline
- how to evaluate a binary classifier
- why metrics beyond accuracy matter

Now they are ready to generalize these ideas to more than two classes.

## 16. Iris as the Multiclass Example

Iris is ideal for introducing multiclass classification because:

- it has three target classes
- its features are numerical and easy to handle
- it has little or no missing-data complexity
- it allows students to focus on model behavior rather than preprocessing difficulty

### Iris task framing

- input features: four flower measurements
- target: flower species
- number of classes: three

This makes Iris a clean follow-up to Titanic.

## 17. Data Preparation for Iris

Iris requires much less preparation.

A minimal workflow is:

1. load the dataset
2. define features and target
3. split the data
4. optionally scale the features depending on the model
5. train and evaluate classifiers

This contrast helps reinforce that preprocessing needs depend on the dataset.

## 18. Multiclass Classification Concepts

Binary classification distinguishes between two classes.

Multiclass classification distinguishes among more than two classes.

Some classifiers support multiclass classification directly.

Others use strategies such as:

- one-versus-rest
- one-versus-one

At this stage, students mainly need a conceptual understanding rather than implementation detail.

## 19. Evaluating Multiclass Classification

Many of the same evaluation ideas still apply, but they must now be interpreted class by class.

### Accuracy

Accuracy is often more acceptable on Iris than on Titanic because Iris is smaller and more balanced.

### Multiclass confusion matrix

The multiclass confusion matrix shows which species are being confused with each other.

### Precision, recall, and F1 in multiclass settings

These can be extended by reporting:

- per-class scores
- macro average
- weighted average

This is enough for this stage of the course.

## 20. Error Analysis

After evaluation, students should examine model mistakes.

Error analysis helps answer questions such as:

- which classes are most often confused?
- which examples are difficult for the model?
- do the errors suggest feature overlap?
- would another model behave differently?

Iris is especially good for this because misclassifications can often be connected to overlapping flower measurements.

## 21. Fine-Tuning and Model Comparison

Once students have a working model, the next step is to improve it.

This usually means tuning hyperparameters with validation or cross-validation.

Examples that fit the labs well include:

- `C` for logistic regression
- `max_depth` for decision trees
- `n_neighbors` and `weights` for k-nearest neighbors

Two key theoretical points should be emphasized:

- tuning should be systematic rather than guesswork
- the test set should not guide tuning decisions

## 22. Multilabel and Multioutput Classification

These topics can be kept brief.

### Multilabel classification

A single instance can receive several labels at once.

Examples:

- document tagging
- movie genre tagging
- assigning multiple attributes to one observation

### Multioutput classification

The model predicts multiple outputs, and each output may itself be binary or multiclass.

These topics should be presented as advanced extensions rather than core lab material.

## 23. Summary of the Theory Flow

The classification theory should now follow this sequence:

1. introduce supervised learning and distinguish classification from regression
2. explain classification as prediction of categories
3. start with Titanic because it connects to last week's EDA and cleaning work
4. avoid long Titanic EDA this week and use only a brief validation check
5. define the binary classification task on Titanic
6. prepare data with pipelines and discuss data leakage
7. introduce baseline models, logistic regression, and model comparison
8. evaluate with accuracy, cross-validation, confusion matrix, precision, recall, and F1
9. explain threshold trade-offs and introduce ROC at a high level
10. transition to Iris for multiclass classification
11. evaluate multiclass models with confusion matrices and class-based metrics
12. include error analysis and simple tuning
13. close with brief mention of multilabel and multioutput classification

## 24. Alignment with the Labs

This revised theory aligns with the lab sequence:

- Lab 1: Titanic binary classification with limited repeated EDA
- Lab 2: Iris multiclass classification with light preprocessing

That gives the week a coherent progression:

- last week: EDA and cleaning
- this week: classification starting from Titanic, then moving to Iris
