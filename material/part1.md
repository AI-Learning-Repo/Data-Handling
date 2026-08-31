# Theory: Supervised Classification

## Part A — Classification Foundations

### 1. Supervised Learning: Regression vs Classification
In supervised learning, a machine learning model learns from historical data where the correct answers (the labels) are already known. By finding patterns in this data, the model can make predictions on new, unseen data.

The two most common supervised-learning tasks are regression and classification:
* **Classification predicts categories.** (e.g., predicting whether an email is "Spam" or "Not Spam").
* **Regression predicts numerical values.** (e.g., predicting the price of a house in dollars).

This document focuses entirely on classification.

### 2. What is Classification?
Classification is the task of assigning an observation to a specific group, category, or class. 

To train a classification model, the data is divided into two parts:
* **Features (`X`)**: The input variables or information used to make the prediction.
* **Target (`y`)**: The specific category or outcome the model is trying to predict.

### 3. Binary vs Multiclass Classification
Classification problems are defined by the number of possible outcomes:
* **Binary classification**: The target has exactly two possible classes (e.g., a patient tests positive or negative for a disease).
* **Multiclass classification**: The target has three or more possible classes (e.g., predicting whether a flower is a rose, a tulip, or a sunflower). 

### 4. Classification Workflow
Whether a problem is binary or multiclass, the fundamental machine learning workflow remains consistent:
1. Define features (`X`) and target (`y`).
2. Split the data into training and test sets.
3. Preprocess the data (e.g., handle missing values, encode text).
4. Establish a simple baseline.
5. Train a machine learning classifier.
6. Evaluate model performance using appropriate metrics.
7. Tune and compare models.
8. Perform a final evaluation on the test set.

### 5. Features, Target, Training, and Test Data
Before any model training begins, the dataset must be split into a **training set** and a **test set**. 

The training set is used to teach the model, while the test set acts as unseen data to evaluate how well the model generalizes. 

A critical concept in this phase is avoiding **data leakage**. Data leakage occurs when information from the test set accidentally influences the model training process. To prevent this, data preparation should be organized into **pipelines**:

```text
Training data
     ↓
fit preprocessing
     ↓
fit model
     ↓
validation
     ↓
final test
```
Transformations such as calculating the median for missing values must be learned exclusively from the training data and then applied to the test data.

---

## Part B — Titanic: Binary Classification

### 6. Why Titanic?
The Titanic dataset is a classic example of **binary classification**. The target variable represents a single question with two possible outcomes: *Did the passenger survive?* 
* **Positive class (1)**: Survived
* **Negative class (0)**: Did not survive

The dataset contains a mix of numerical features (like age and fare) and categorical features (like gender and passenger class), making it ideal for learning how to process mixed data types.

### 7. Brief Data Validation
Before modeling, data must be validated. Even if extensive Exploratory Data Analysis (EDA) was performed previously, a brief check ensures the data is ready for modeling. This includes:
* Confirming the target and features exist.
* Checking data types.
* Identifying missing values.
* Checking the class distribution (whether there are significantly more negative classes than positive classes).

### 8. Preprocessing and Pipelines
Machine learning algorithms require clean, numerical inputs. Typical preprocessing steps include:
* **Imputation**: Filling missing numerical values (e.g., using the median) and categorical values (e.g., using the most frequent category).
* **Encoding**: Converting categorical text into numerical format using techniques like one-hot encoding.

*Note on feature scaling:* While scaling numerical features is generally a good practice, algorithms like Decision Trees and Logistic Regression do not strictly require scaled features to make valid predictions. Therefore, scaling is not an absolute requirement for all models.

### 9. Baseline
Before trusting a complex machine learning model, its performance must be compared to a **baseline**. A baseline is a trivial strategy, such as always predicting the most frequent class. 

If 62% of passengers on the Titanic did not survive, a baseline model that always predicts "Did not survive" will achieve 62% accuracy. A real classifier must prove it can learn meaningful patterns by significantly outperforming this baseline.

### 10. Logistic Regression
One of the most common algorithms for binary classification is **Logistic Regression**.

> **Important Distinction:** Despite its name, **logistic regression is a classification algorithm**. The word "regression" refers to the underlying mathematical formulation, not to the type of prediction task being performed.

Logistic regression calculates the probability that an observation belongs to the positive class, making it highly interpretable.

### 11. Confusion Matrix
A single accuracy score can hide how a model is failing. The **confusion matrix** breaks down the exact predictions into four categories. For binary classification, it is a $2 \times 2$ table:

* **True Positive (TP)**: Model predicted Survived, and the passenger actually Survived.
* **True Negative (TN)**: Model predicted Did Not Survive, and the passenger actually Did Not Survive.
* **False Positive (FP)**: Model predicted Survived, but the passenger Did Not Survive.
* **False Negative (FN)**: Model predicted Did Not Survive, but the passenger actually Survived.

### 12. Accuracy
**Accuracy** is the proportion of total correct predictions out of all predictions made. While useful, it can be misleading in imbalanced datasets where one class dominates the other.

### 13. Precision, Recall, and F1
To get a more detailed view of classification performance, three core metrics are used:

* **Precision**: Of all the observations the model *predicted* as positive, how many were actually positive? 
  $$Precision = \frac{TP}{TP + FP}$$
* **Recall**: Of all the *actual* positive observations, how many did the model successfully find? 
  $$Recall = \frac{TP}{TP + FN}$$
* **F1 Score**: The harmonic mean of precision and recall. It provides a single score that balances both metrics, which is especially useful when false positives and false negatives are both concerning.

### 14. Decision Thresholds
Classifiers like Logistic Regression do not immediately predict a hard class (0 or 1). Instead, they output an estimated probability. 

A **decision threshold** is applied to convert this probability into a final class prediction:

```text
Model
  ↓
Estimated probability (e.g., 0.72)
  ↓
Decision threshold (e.g., 0.50)
  ↓
Predicted class (e.g., 1)
```

By default, the threshold is usually 0.50. Raising the threshold requires the model to be more confident before predicting the positive class. This generally increases Precision (fewer false positives) but decreases Recall (more missed positive cases).

### 15. Cross-Validation
Evaluating a model on a single train/test split can result in unstable metrics—a model might just get a "lucky" split. 

**Cross-validation** divides the training data into multiple sections (folds). The model is trained on some folds and evaluated on the remaining fold, repeating this process until every fold has been used for evaluation. This provides a much more reliable estimate of how the model will perform on unseen data.

### 16. Model Comparison
Different algorithms approach problems differently. For example, a **Decision Tree** learns classification rules by splitting the data based on feature thresholds, while Logistic Regression applies a mathematical equation. Comparing different models side-by-side reveals which algorithm's approach best fits the specific dataset.

### 17. Parameters vs. Hyperparameters
When tuning a model, it is crucial to understand the difference between two types of values:
* **Parameters**: Values learned automatically by the model from the training data (e.g., the importance of the 'age' feature in a logistic regression equation).
* **Hyperparameters**: Settings chosen by the data scientist *before* training begins (e.g., the maximum depth of a decision tree, or the `C` penalty term in logistic regression).

Hyperparameter tuning searches for the best settings using cross-validation on the training set. The test set must remain entirely untouched during tuning to serve as a valid final exam for the model.

---

## Part C — Iris: Multiclass Classification

### 18. Why Iris? Transitioning to Multiclass
While the Titanic dataset asks a binary question ("Survived or not?"), the Iris dataset introduces a new dynamic. The model must answer:
> **"Setosa, versicolor, or virginica?"**

The fundamental machine learning workflow (split, preprocess, train, evaluate) remains identical. What changes is the mathematical representation of the problem, the structure of the confusion matrix, and how evaluation metrics are calculated.

### 19. Three-Class Prediction
In a multiclass problem, **there is no single, universal positive class**. 
When evaluating the model, we cannot simply say "positive = true." Instead, evaluation metrics must be calculated for each specific class by temporarily treating that class as the "class of interest" (positive) and all other classes as the alternative (negative).

### 20. K-Nearest Neighbors (KNN)
**K-Nearest Neighbors (KNN)** is an algorithm well-suited for numerical datasets like Iris. It classifies a new observation by looking at the known classes of the data points closest to it in the training set. If the majority of a point's nearest neighbors are 'setosa', the model predicts 'setosa'.

### 21. Feature Scaling
Because KNN makes decisions based on geometric distances, the scale of the features matters. If one measurement is very large and another is very small, the larger measurement will mathematically dominate the distance calculation. 

*Note: In introductory contexts, scaling may be temporarily omitted if the features share a similar natural scale (like petal measurements in centimeters), but standardizing features is a best practice for distance-based models.*

### 22. Multiclass Confusion Matrix
For three classes, the confusion matrix expands from a $2 \times 2$ grid to a $3 \times 3$ grid. 
The diagonal still represents correct predictions. The off-diagonal cells are now highly informative because they reveal exactly *which* classes are being confused with one another (e.g., identifying that the model frequently mistakes versicolor for virginica).

### 23. Per-Class Precision, Recall, and F1
In multiclass classification, the `classification_report` provides Precision, Recall, and F1 scores for *each* individual class. A model might be perfect at identifying 'setosa' (F1 = 1.0) but struggle significantly with 'versicolor' (F1 = 0.85).

### 24. Macro vs Weighted Averages
To summarize multiclass performance into a single number, two types of averages are used:
* **Macro Average**: Calculates the metric for each class and averages them equally. It treats a class with 5 examples as equally important as a class with 500 examples.
* **Weighted Average**: Calculates the metric for each class, but weights the final average based on how many observations belong to each class. 

For balanced datasets (like Iris, where every class has 50 flowers), the macro and weighted averages will be virtually identical.

### 25. Logistic Regression for Multiclass Classification
Logistic regression is not restricted to binary problems. Modern implementations can handle multiclass classification directly, outputting a probability for every possible class, and predicting the class with the highest probability.

### 26. Model Comparison and Error Analysis
When comparing multiclass models (like KNN vs. Logistic Regression), accuracy is evaluated alongside the confusion matrix. 

**Error analysis** goes a step further by actively investigating the misclassified rows. By looking at the raw data of the mistakes, it is often possible to see why the model struggled—for instance, noting that certain flowers have overlapping physical measurements.

### 27. Hyperparameter Tuning
Multiclass models are tuned the same way as binary models. For KNN, this might involve tuning hyperparameters such as `n_neighbors` (how many nearby points to consider) or `weights` (whether closer neighbors should have more voting power than distant ones).

---

## Part D — Extensions

### 28. Binary vs Multiclass Recap
* **Binary Classification**: 2 possible classes, 1 positive class, $2 \times 2$ confusion matrix, utilizes single summary metrics for Precision/Recall, and allows for manual threshold adjustments.
* **Multiclass Classification**: 3 or more classes, no universal positive class, $N \times N$ confusion matrix, requires per-class metrics, and relies on Macro or Weighted averages for summarization.

### 29. Multilabel Classification
In standard classification, an observation belongs to exactly one class. In **multilabel classification**, a single instance can receive multiple labels simultaneously (e.g., a single movie can be classified as both "Action" and "Comedy").

### 30. Multioutput Classification
In **multioutput classification**, a model is tasked with predicting multiple different targets at the same time (e.g., analyzing a plant to predict both its species AND its health status simultaneously).

### 31. Optional Self-Study: ROC Curve and ROC AUC
The **Receiver Operating Characteristic (ROC)** curve is an advanced evaluation tool used in binary classification. Instead of calculating metrics based on a single threshold (like 0.50), the ROC curve plots the True Positive Rate against the False Positive Rate across *all possible decision thresholds*.

The **ROC AUC** (Area Under the Curve) summarizes this plot into a single score between 0 and 1. It represents the model's overall ability to distinguish between the positive and negative classes, independent of whatever specific threshold is eventually chosen.