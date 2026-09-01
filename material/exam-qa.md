# Session Review & Sample Exam Questions: Classification

Use these questions to test your understanding of supervised classification, binary and multiclass problems, preprocessing, classification algorithms, baselines, and model evaluation.

---

## Section A: Multiple Choice Questions (MCQ)

**1. Which statement best describes supervised learning?**

* A) A model discovers patterns without having any known target values.
* B) A model learns from examples where the correct target values are already known.
* C) A model only works with numerical variables.
* D) A model automatically cleans all missing values before training.

---

**2. In a machine-learning problem, what do `X` and `y` normally represent?**

* A) `X` = predictions, `y` = errors
* B) `X` = features, `y` = target
* C) `X` = training data, `y` = test data
* D) `X` = numerical variables, `y` = categorical variables

---

**3. You are predicting whether a Titanic passenger survived or did not survive. What type of machine-learning problem is this?**

* A) Regression
* B) Unsupervised learning
* C) Binary classification
* D) Multiclass regression

---

**4. You are predicting whether an Iris flower is `setosa`, `versicolor`, or `virginica`. What type of problem is this?**

* A) Binary classification
* B) Multiclass classification
* C) Regression
* D) Clustering

---

**5. Why do we separate data into training and test sets?**

* A) To make the dataset smaller.
* B) To remove missing values from the test set.
* C) To evaluate how well the trained model performs on unseen data.
* D) To guarantee that the model will have 100% accuracy.

---

**6. What is the purpose of a baseline classifier?**

* A) To provide a simple reference against which the real model can be compared.
* B) To guarantee better accuracy than all other models.
* C) To remove outliers from the dataset.
* D) To replace the need for a test set.

---

**7. A Titanic baseline always predicts the most frequent class. If 62% of the training passengers did not survive, approximately what accuracy would you expect from this baseline?**

* A) 38%
* B) 50%
* C) 62%
* D) 100%

---

**8. Why is the Iris baseline accuracy approximately one-third when using the most frequent class strategy?**

* A) Iris contains three equally represented classes.
* B) Iris contains three features.
* C) Iris has approximately 33 observations.
* D) The model randomly removes two-thirds of the observations.

---

**9. Which statement best describes logistic regression in these labs?**

* A) It is a regression algorithm that predicts continuous numerical values only.
* B) It is a classification algorithm that models the probability of belonging to a class.
* C) It is a tree-based algorithm.
* D) It classifies an observation only by finding its nearest neighbors.

---

**10. Which statement best describes a decision tree?**

* A) It predicts classes by calculating distances to nearby observations.
* B) It learns a sequence of feature-based decisions that divide the data into groups.
* C) It always predicts the most frequent class.
* D) It can only be used for regression.

---

**11. Which statement best describes K-nearest neighbors (KNN)?**

* A) It classifies a new observation according to the classes of nearby training observations.
* B) It builds a sequence of if/else rules using tree branches.
* C) It calculates only the average value of the target.
* D) It ignores the numerical values of the features.

---

**12. Why can feature scaling be particularly important for KNN?**

* A) KNN uses distances, so variables with larger numerical scales can have a greater influence.
* B) Scaling converts categorical variables into classes.
* C) Scaling removes all missing values.
* D) KNN can only work with standardized data.

---

**13. Which metric answers the question: "Of all observations predicted as positive, how many were actually positive?"**

* A) Recall
* B) Accuracy
* C) Precision
* D) F1 score

---

**14. Which metric answers the question: "Of all actual positive observations, how many did the model correctly identify?"**

* A) Recall
* B) Precision
* C) Accuracy
* D) Macro average

---

**15. Which statement about a multiclass confusion matrix is correct?**

* A) It must always be 2 × 2.
* B) It has one row and one column for each class.
* C) It can only be used with binary classification.
* D) It contains only correct predictions.

---

**16. In a three-class Iris problem, what does an off-diagonal value in the confusion matrix represent?**

* A) A correctly classified flower.
* B) A missing value.
* C) A flower whose predicted species differs from its actual species.
* D) The overall accuracy of the model.

---

**17. What is the main difference between macro and weighted averages in multiclass classification?**

* A) Macro averages give equal importance to each class, while weighted averages account for class size.
* B) Macro averages are only used for binary classification.
* C) Weighted averages always produce a higher score than macro averages.
* D) There is no difference between them.

---

## Section B: True or False (with 1-Sentence Justification)

**18. Statement:** *"A classifier that achieves 90% accuracy on the training data must be a good model."*

* **True / False?** Justify: __________________________________________________________________

---

**19. Statement:** *"The test set should be kept separate while the model is being trained."*

* **True / False?** Justify: __________________________________________________________________

---

**20. Statement:** *"A baseline is a machine-learning model that is expected to outperform sophisticated classifiers."*

* **True / False?** Justify: __________________________________________________________________

---

**21. Statement:** *"Iris is a multiclass classification problem because the target contains three possible species."*

* **True / False?** Justify: __________________________________________________________________

---

**22. Statement:** *"KNN is affected by feature scale because it uses distances between observations."*

* **True / False?** Justify: __________________________________________________________________

---

**23. Statement:** *"Accuracy is always the best metric for evaluating a classifier."*

* **True / False?** Justify: __________________________________________________________________

---

## Section C: Short-Answer & Scenario Questions

**24. Binary vs Multiclass Classification**

Consider the following two problems:

```text
Problem A:
Predict whether a passenger survived the Titanic.

Problem B:
Predict whether an Iris flower is setosa, versicolor, or virginica.
```

* **(a)** Identify the type of classification problem in each case.
* **(b)** How many possible classes does each problem have?
* **(c)** What is the main conceptual difference between the two problems?

---

**25. Training and Test Data**

You have 1,000 observations. You use 800 for training and 200 for testing.

After training, your classifier achieves:

```text
Training accuracy = 98%
Test accuracy     = 79%
```

* **(a)** What does the large difference between training and test accuracy suggest?
* **(b)** What machine-learning concept does this illustrate?
* **(c)** Why is the test accuracy more useful for estimating performance on new observations?

---

**26. Understanding a Baseline**

A Titanic dataset contains:

```text
Did not survive = 550 passengers
Survived        = 341 passengers
```

A baseline classifier always predicts `Did not survive`.

* **(a)** Approximately what accuracy would this baseline achieve?
* **(b)** Why is this baseline useful even though it does not learn relationships between the features and survival?
* **(c)** If a real classifier achieves 65% accuracy, would you consider this a meaningful improvement over the baseline? Explain.

---

**27. Choosing an Algorithm**

You are given a new Iris flower and want to predict its species.

Explain how each of the following algorithms could make the prediction:

* **(a)** Logistic Regression
* **(b)** Decision Tree
* **(c)** KNN

Your answer should focus on the **basic idea** of how each algorithm makes a classification.

---

**28. Interpreting a Confusion Matrix**

A binary classifier produces the following confusion matrix:

```text
                 Predicted
                 No     Yes

Actual No        80      10
Actual Yes       15      45
```

* **(a)** How many predictions were correct?
* **(b)** How many false positives are there?
* **(c)** How many false negatives are there?
* **(d)** Calculate the accuracy.
* **(e)** Calculate the precision for the positive (`Yes`) class.
* **(f)** Calculate the recall for the positive (`Yes`) class.

---

**29. Interpreting an Iris Confusion Matrix**

A classifier produces the following confusion matrix:

```text
                 Predicted

                 setosa  versicolor  virginica

Actual setosa       10       0          0

Actual versicolor   0        8          2

Actual virginica    0        1          9
```

* **(a)** How many flowers were classified correctly?
* **(b)** Which species was classified perfectly?
* **(c)** Which two species are being confused?
* **(d)** What does the value `2` in the versicolor row and virginica column mean?
* **(e)** Why is the confusion matrix more informative than accuracy alone?

---

**30. Precision, Recall, and F1**

A binary classifier produces:

```text
TP = 80
FP = 20
FN = 10
TN = 90
```

* **(a)** Calculate precision.
* **(b)** Calculate recall.
* **(c)** Explain in words what the precision value means.
* **(d)** Explain in words what the recall value means.
* **(e)** Why might it be useful to consider both precision and recall rather than accuracy alone?

---

## Section D: Code & Concept Interpretation

**31. Train/Test Split**

Examine the following code:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

* **(a)** What do `X_train` and `X_test` contain?
* **(b)** What do `y_train` and `y_test` contain?
* **(c)** What does `test_size=0.2` mean?
* **(d)** Why do we use `random_state=42`?

---

**32. Training and Prediction**

Examine:

```python
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

Explain the purpose of each line.

---

**33. KNN**

Examine:

```python
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
```

* **(a)** What does `n_neighbors=5` mean?
* **(b)** What happens when `fit()` is called?
* **(c)** What happens when `predict()` is called?
* **(d)** Why does KNN need the training observations when making predictions?

---

**34. Classification Report**

Suppose the following code is used:

```python
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))
```

* **(a)** What are `y_test` and `y_pred` being compared?
* **(b)** Why does the report contain separate rows for `setosa`, `versicolor`, and `virginica`?
* **(c)** What do precision, recall, and F1 describe for each species?
* **(d)** What is the purpose of the macro average?

---

## Section E: Integrated Exam Question

**35. Complete Classification Scenario**

You are given a dataset of flowers. Each observation contains:

```text
sepal_length
sepal_width
petal_length
petal_width
species
```

The target `species` contains:

```text
setosa
versicolor
virginica
```

You want to build a machine-learning classifier.

Answer the following:

* **(a)** Is this a classification or regression problem? Why?
* **(b)** Is it binary or multiclass classification? Why?
* **(c)** Which variable is the target?
* **(d)** Which variables are the features?
* **(e)** Why should you create a training and test set?
* **(f)** Why would a baseline be useful?
* **(g)** Explain, in one or two sentences, how KNN would classify a new flower.
* **(h)** Why can feature scaling matter for KNN?
* **(i)** Name two appropriate metrics for evaluating this classifier.
* **(j)** Why is a confusion matrix useful?
* **(k)** If the confusion matrix shows that many `versicolor` flowers are predicted as `virginica`, what does this tell you?
* **(l)** If the model has high accuracy but a low recall for `virginica`, what does this tell you about its performance?

---

# Solutions

## Section A: Multiple Choice

**1. B** — Supervised learning uses examples for which the correct target values are known. The model learns from these examples and can then make predictions for new observations.

**2. B** — `X` represents the features/input variables, while `y` represents the target that the model is trying to predict.

**3. C** — Titanic survival has two possible outcomes: survived and did not survive. Therefore, it is binary classification.

**4. B** — Iris has three possible target classes: setosa, versicolor, and virginica, so it is multiclass classification.

**5. C** — The test set provides previously unseen observations that allow us to estimate how well the trained model generalizes to new data.

**6. A** — A baseline provides a simple reference point. A real classifier should be evaluated in relation to this reference.

**7. C** — If 62% of passengers belong to the most frequent class, always predicting that class gives approximately 62% accuracy.

**8. A** — Iris has three equally represented classes. Always predicting one class therefore gives approximately $1/3 \approx 33%$ accuracy.

**9. B** — Logistic regression is a classification algorithm that models the probability of belonging to a class.

**10. B** — A decision tree learns a sequence of decisions based on feature values and uses those decisions to assign observations to classes.

**11. A** — KNN finds nearby training observations and uses their known classes, typically through a majority vote.

**12. A** — KNN uses distances. A feature with a much larger numerical scale can therefore have a disproportionately large effect on the distance.

**13. C** — Precision asks: of everything predicted as positive, how much was actually positive?

**14. A** — Recall asks: of everything that was actually positive, how much did the model correctly identify?

**15. B** — A multiclass confusion matrix has one row and one column for each class.

**16. C** — An off-diagonal entry represents an incorrect prediction: the predicted class differs from the actual class.

**17. A** — Macro averaging gives each class equal importance; weighted averaging gives more influence to classes containing more observations.

---

## Section B: True or False

**18. False** — High training accuracy alone does not guarantee good performance on unseen data; the model may be overfitting.

**19. True** — The test set should represent unseen data and therefore should not be used to train the model.

**20. False** — A baseline is intentionally simple and exists as a reference point, not as a model that should outperform more informative classifiers.

**21. True** — Iris has three possible target classes, making it a multiclass classification problem.

**22. True** — KNN uses distances between observations, so differences in feature scales can affect which observations are considered nearest.

**23. False** — Accuracy can be useful, but it may hide poor performance for particular classes, especially when classes are imbalanced. Precision, recall, F1, and the confusion matrix provide additional information.

---

## Section C: Short Answer

**24. Binary vs Multiclass Classification**

**(a)**

* Problem A: **Binary classification**
* Problem B: **Multiclass classification**

**(b)**

* Titanic: **2 classes**
* Iris: **3 classes**

**(c)** The main difference is the number of possible target classes. Titanic has two possible outcomes, while Iris has three possible species.

---

**25. Training and Test Data**

**(a)** The large difference suggests that the model performs much better on the data it learned from than on unseen data.

**(b)** This is a possible sign of **overfitting**.

**(c)** Test accuracy is more useful for estimating performance on new observations because the test observations were not used to fit the model.

---

**26. Understanding a Baseline**

**(a)**

There are 550 passengers in the majority class out of 891:

$$
Accuracy = \frac{550}{891} \approx 0.617
$$

So the baseline accuracy is approximately **61.7%**.

**(b)** The baseline gives us a simple reference point. It tells us how well we could perform using a trivial strategy without learning relationships between the features and target.

**(c)** Yes. A classifier achieving 65% accuracy would outperform the baseline, although only by a small amount:

$$
65\% - 61.7\% \approx 3.3
$$

percentage points.

We would need additional metrics before concluding that the model is genuinely useful.

---

**27. Choosing an Algorithm**

**(a) Logistic Regression:** It learns a mathematical relationship between the features and the probability of belonging to each class, then uses those probabilities to make a class prediction.

**(b) Decision Tree:** It learns a sequence of feature-based decisions. For example, it may split flowers according to petal length and then make additional decisions until it reaches a predicted class.

**(c) KNN:** It finds the training flowers that are closest to the new flower and predicts the class based on the classes of those neighbors.

---

**28. Interpreting a Confusion Matrix**

The matrix is:

```text
                 Predicted
                 No     Yes

Actual No        80      10
Actual Yes       15      45
```

**(a) Correct predictions:**

$$
80 + 45 = 125
$$

**125 predictions** were correct.

**(b) False positives:**

A false positive is an actual `No` predicted as `Yes`.

$$
FP = 10
$$

**(c) False negatives:**

A false negative is an actual `Yes` predicted as `No`.

$$
FN = 15
$$

**(d) Accuracy:**

There are:

$$
80+10+15+45=150
$$

total observations.

Therefore:

$$
Accuracy = \frac{125}{150} \approx 0.833
$$

**Accuracy ≈ 83.3%**.

**(e) Precision:**

$$
Precision = \frac{TP}{TP+FP}
$$

$$
Precision = \frac{45}{45+10}
= \frac{45}{55}
\approx 0.818
$$

**Precision ≈ 81.8%**.

**(f) Recall:**

$$
Recall = \frac{TP}{TP+FN}
$$

$$
Recall = \frac{45}{45+15}
= \frac{45}{60}
= 0.75
$$

**Recall = 75%**.

---

**29. Interpreting an Iris Confusion Matrix**

```text
                 Predicted

                 setosa  versicolor  virginica

Actual setosa       10       0          0

Actual versicolor   0        8          2

Actual virginica    0        1          9
```

**(a)** Correct predictions are on the diagonal:

$$
10 + 8 + 9 = 27
$$

So **27 flowers** were classified correctly.

**(b)** `setosa` was classified perfectly: all 10 setosa flowers were correctly identified.

**(c)** `versicolor` and `virginica` are being confused.

**(d)** The `2` means:

> Two flowers whose actual species was `versicolor` were incorrectly predicted as `virginica`.

**(e)** Accuracy tells us only the total proportion of correct predictions. The confusion matrix additionally tells us **which classes are being confused**.

---

**30. Precision, Recall, and F1**

Given:

```text
TP = 80
FP = 20
FN = 10
TN = 90
```

**(a) Precision**

$$
Precision = \frac{80}{80+20}
= 0.80
$$

**Precision = 80%**.

**(b) Recall**

$$
Recall = \frac{80}{80+10}
\approx 0.889
$$

**Recall ≈ 88.9%**.

**(c)** Precision of 80% means:

> Of all observations predicted as positive, 80% were actually positive.

**(d)** Recall of approximately 88.9% means:

> Of all observations that were actually positive, the model correctly identified approximately 88.9%.

**(e)** Accuracy alone does not show the types of errors the model makes. Precision and recall provide more detailed information about false positives and false negatives.

---

## Section D: Code & Concept Interpretation

**31. Train/Test Split**

**(a)** `X_train` contains the feature values used to train the model. `X_test` contains feature values reserved for testing.

**(b)** `y_train` contains the known target values corresponding to `X_train`. `y_test` contains the true target values corresponding to `X_test`.

**(c)** `test_size=0.2` means that approximately 20% of the observations are placed in the test set and 80% in the training set.

**(d)** `random_state=42` makes the random split reproducible. Running the same code again produces the same split.

---

**32. Training and Prediction**

```python
model.fit(X_train, y_train)
```

This trains the model using the training features and their known target values.

```python
y_pred = model.predict(X_test)
```

This uses the trained model to predict the target class for the previously unseen observations in `X_test`.

---

**33. KNN**

**(a)** `n_neighbors=5` tells KNN to consider the five nearest training observations when making a prediction.

**(b)** `fit()` prepares the KNN model using the training data. KNN needs the training observations and their known classes so that it can find neighbors later.

**(c)** `predict()` finds the nearest training observations for each test observation and uses their classes to determine the prediction.

**(d)** KNN is based on the idea that nearby observations are likely to have similar classes. Therefore, it needs the training observations to determine which observations are closest.

---

**34. Classification Report**

**(a)** `y_test` contains the actual classes, while `y_pred` contains the classes predicted by the model. The report compares them.

**(b)** There are separate rows because Iris has three classes, and we want to evaluate the model's performance for each species.

**(c)** For each species:

* **Precision** tells us how reliable the predictions of that species are.
* **Recall** tells us how many actual flowers of that species the model successfully identified.
* **F1** combines precision and recall.

**(d)** The macro average gives each of the three species equal importance when calculating the overall average metric.

---

## Section E: Integrated Exam Question

**35. Complete Classification Scenario**

**(a)** This is a **classification** problem because the target is a category rather than a numerical quantity.

**(b)** It is **multiclass classification** because there are three possible classes: setosa, versicolor, and virginica.

**(c)** The target is:

```text
species
```

**(d)** The features are:

```text
sepal_length
sepal_width
petal_length
petal_width
```

**(e)** We create training and test sets so that we can evaluate the trained model on observations it did not use during training.

**(f)** A baseline provides a simple reference point. It allows us to determine whether the classifier provides meaningful improvement over a simple prediction strategy.

**(g)** KNN finds the flowers in the training set that are closest to the new flower and predicts the species based on the classes of those neighbors.

**(h)** KNN uses distances. If features have very different scales, a feature with larger numerical values can have too much influence on the distance calculation.

**(i)** Appropriate metrics include:

* accuracy
* precision
* recall
* F1 score

A confusion matrix is also an appropriate evaluation tool.

**(j)** A confusion matrix shows which classes were correctly predicted and which classes were confused with one another.

**(k)** If many `versicolor` flowers are predicted as `virginica`, the model has difficulty distinguishing these two species.

**(l)** High accuracy but low recall for `virginica` means that the model performs well overall but fails to identify a relatively large proportion of the flowers that are actually `virginica`. This is an example of why overall accuracy should not always be considered sufficient.
