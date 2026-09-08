# Session Review & Sample Exam Questions: Regression

Use these questions to test your understanding of supervised regression, features and targets, Pearson correlation, linear and multiple linear regression, polynomial regression, preprocessing, train/test splitting, model evaluation, underfitting, overfitting, and pipelines.

---

## Section A: Multiple Choice Questions (MCQ)

**1. Which statement best describes supervised regression?**

* A) A model discovers groups without using target values.
* B) A model learns from examples where the correct numerical target values are already known.
* C) A model can only work with categorical target values.
* D) A model automatically removes all outliers before training.

---

**2. In a regression problem, what do `X` and `y` normally represent?**

* A) `X` = predictions, `y` = errors
* B) `X` = features, `y` = numerical target
* C) `X` = training data, `y` = test data
* D) `X` = categorical variables, `y` = numerical features

---

**3. You want to predict the fare of a taxi trip in euros. What type of machine-learning problem is this?**

* A) Binary classification
* B) Multiclass classification
* C) Regression
* D) Clustering

---

**4. Which statement best describes linear regression?**

* A) It predicts a continuous numerical target using a linear relationship between features and the target.
* B) It predicts only two possible classes.
* C) It always creates a decision tree.
* D) It can only use one feature.

---

**5. Which equation represents simple linear regression?**

* A) \(y = b_0 + b_1x\)
* B) \(y = x^2 + b_0\)
* C) \(y = b_0 + b_1/x\)
* D) \(y = \log(x)\)

---

**6. What is the main difference between simple linear regression and multiple linear regression?**

* A) Simple linear regression predicts categories, while multiple linear regression predicts numbers.
* B) Simple linear regression uses one feature, while multiple linear regression can use several features.
* C) Multiple linear regression cannot use numerical features.
* D) Simple linear regression always uses polynomial features.

---

**7. Which of the following is an example of multiple linear regression?**

* A) `FARE = b0 + b1 * TRIP_MILES`
* B) `FARE = b0 + b1 * TRIP_MILES + b2 * TRIP_MINUTES`
* C) `FARE = most_frequent_class`
* D) `FARE = TRIP_MILES ** 2`

---

**8. What is the purpose of the intercept in a linear regression model?**

* A) It represents the predicted target when all features are zero.
* B) It represents the number of training observations.
* C) It is always equal to zero.
* D) It measures the test-set error.

---

**9. What does a positive regression coefficient generally indicate?**

* A) Increasing that feature is associated with an increase in the model's prediction, holding other features constant.
* B) Increasing that feature always causes the target to increase.
* C) The feature is categorical.
* D) The model is overfitting.

---

**10. What is the main purpose of Pearson Correlation Coefficient (`r`)?**

* A) To measure the strength and direction of a linear relationship between two numerical variables.
* B) To measure classification accuracy.
* C) To calculate the number of features in a dataset.
* D) To determine whether a model is a decision tree.

---

**11. What is the possible range of Pearson correlation `r`?**

* A) 0 to 100
* B) -100 to 100
* C) -1 to +1
* D) 1 to infinity

---

**12. What does `r = 0.92` generally indicate?**

* A) A strong positive linear relationship
* B) A strong negative linear relationship
* C) No relationship of any kind
* D) Perfect negative correlation

---

**13. What does `r = -0.85` generally indicate?**

* A) A strong positive linear relationship
* B) A strong negative linear relationship
* C) Perfect positive correlation
* D) No possible relationship

---

**14. Which statement about Pearson correlation is correct?**

* A) A value close to zero proves that two variables are completely unrelated.
* B) Pearson correlation is specifically designed to measure linear association.
* C) Pearson correlation can only be used for categorical variables.
* D) Pearson correlation is the same as RMSE.

---

**15. Consider the relationship \(y = x^2\). Which statement is most accurate?**

* A) It is perfectly linear, so Pearson correlation must be +1.
* B) It is nonlinear, and Pearson correlation can be close to zero even though a strong relationship exists.
* C) There is no relationship between `x` and `y`.
* D) Pearson correlation cannot be calculated.

---

**16. Why might we calculate the correlation between each feature and the target?**

* A) To get an initial understanding of linear relationships in the data.
* B) To guarantee that the feature should be included in the final model.
* C) To replace model evaluation completely.
* D) To remove the need for a train/test split.

---

**17. Why do we split data into training and test sets?**

* A) To make the model train faster.
* B) To evaluate how the trained model performs on unseen data.
* C) To guarantee that the model will have zero error.
* D) To remove all outliers.

---

**18. Which function is commonly used to split data into training and test sets in scikit-learn?**

* A) `train_test_split()`
* B) `split_data()`
* C) `data_partition()`
* D) `test_train()`

---

**19. What does `test_size=0.2` mean in the following code?**

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

* A) 20% of the observations are approximately assigned to the test set.
* B) 20% of the features are removed.
* C) The model trains for 20 epochs.
* D) The target is reduced by 20%.

---

**20. What is the main purpose of `random_state=42`?**

* A) It guarantees the best model.
* B) It controls the random number generation so the split can be reproduced.
* C) It removes randomness from the dataset permanently.
* D) It changes the target values.

---

**21. What does `model.fit(X_train, y_train)` do in regression?**

* A) It evaluates the model using the test data.
* B) It learns the model parameters from the training data.
* C) It creates the test set.
* D) It calculates Pearson correlation.

---

**22. What does `model.predict(X_test)` do?**

* A) It trains the model again.
* B) It predicts target values for observations in `X_test`.
* C) It changes the test data.
* D) It calculates the training correlation.

---

**23. Which metric measures the average absolute prediction error?**

* A) MAE
* B) MSE
* C) R²
* D) Pearson `r`

---

**24. Which regression metric penalizes large errors more strongly because the errors are squared?**

* A) MAE
* B) MSE
* C) Accuracy
* D) Precision

---

**25. Which metric is expressed in the same units as the target and is calculated as the square root of MSE?**

* A) MAE
* B) RMSE
* C) R²
* D) Pearson `r`

---

**26. What does an R² value of `0.85` generally indicate?**

* A) The model explains a large proportion of the variation in the target relative to the chosen reference.
* B) The model has an accuracy of exactly 85%.
* C) The model makes an average error of 0.85 units.
* D) The Pearson correlation is exactly 0.85.

---

**27. Which statement best describes Polynomial Regression?**

* A) It creates polynomial features and can use a linear regression estimator to model nonlinear relationships.
* B) It is always a decision-tree algorithm.
* C) It can only model straight lines.
* D) It can only use categorical targets.

---

**28. What happens when we use `PolynomialFeatures(degree=2)`?**

* A) The model automatically becomes a classification model.
* B) Additional polynomial terms such as squared features and interaction terms can be created.
* C) All features are removed.
* D) The target is squared.

---

**29. Which statement is correct about increasing polynomial degree?**

* A) Increasing the degree always improves test performance.
* B) Increasing the degree makes the model less flexible.
* C) Increasing the degree can improve the fit but may eventually cause overfitting.
* D) Polynomial degree has no effect on model complexity.

---

**30. Which situation is a typical sign of overfitting?**

* A) High training error and high test error
* B) Very low training error and much higher test error
* C) Low training error and equally low test error
* D) No training data

---

**31. Which situation is a typical sign of underfitting?**

* A) Very low training error and very high test error
* B) High training error and high test error
* C) Zero training error and zero test error
* D) A high Pearson correlation

---

**32. Why can Polynomial Regression overfit when the degree is too high?**

* A) The model becomes less flexible.
* B) The model can become flexible enough to fit noise and unusual details in the training data.
* C) The target automatically becomes categorical.
* D) The train/test split disappears.

---

**33. What is the main purpose of feature scaling?**

* A) To put numerical features on comparable scales when the algorithm benefits from this.
* B) To convert the target into a categorical variable.
* C) To guarantee a higher R².
* D) To remove all missing values.

---

**34. Which statement about `Pipeline` is correct?**

* A) A pipeline can combine preprocessing and a machine-learning model into one workflow.
* B) A pipeline is required for every scikit-learn model.
* C) A pipeline eliminates the need for a test set.
* D) A pipeline automatically guarantees no overfitting.

---

**35. Why is it useful to learn preprocessing separately before introducing `Pipeline`?**

* A) It helps students understand each operation before those operations are combined into a single workflow.
* B) Pipelines cannot perform preprocessing.
* C) Preprocessing is unrelated to machine learning.
* D) Separate preprocessing always gives better models.

---

## Section B: True or False (with 1-Sentence Justification)

**36. Statement:** *"Regression is used when the target is a continuous numerical value."*

* **True / False?** Justify: __________________________________________________________________

---

**37. Statement:** *"A Pearson correlation of `r = 0` proves that two variables have no relationship at all."*

* **True / False?** Justify: __________________________________________________________________

---

**38. Statement:** *"The relationship \(y=2x\) is linear."*

* **True / False?** Justify: __________________________________________________________________

---

**39. Statement:** *"The relationship \(y=x^2\) is nonlinear."*

* **True / False?** Justify: __________________________________________________________________

---

**40. Statement:** *"If a feature has a high Pearson correlation with the target, it must always be included in the final regression model."*

* **True / False?** Justify: __________________________________________________________________

---

**41. Statement:** *"The test set should be used to fit the regression model."*

* **True / False?** Justify: __________________________________________________________________

---

**42. Statement:** *"A model with very low training error can still perform poorly on unseen test data."*

* **True / False?** Justify: __________________________________________________________________

---

**43. Statement:** *"Underfitting occurs when a model is too simple to capture important patterns in the data."*

* **True / False?** Justify: __________________________________________________________________

---

**44. Statement:** *"Overfitting occurs when the model performs much better on unseen test data than on training data."*

* **True / False?** Justify: __________________________________________________________________

---

**45. Statement:** *"Increasing polynomial degree always improves generalization."*

* **True / False?** Justify: __________________________________________________________________

---

**46. Statement:** *"RMSE is expressed in the same units as the target variable."*

* **True / False?** Justify: __________________________________________________________________

---

**47. Statement:** *"MAE gives larger errors more influence than smaller errors by squaring them."*

* **True / False?** Justify: __________________________________________________________________

---

**48. Statement:** *"A pipeline can help organize preprocessing and model training into a single workflow."*

* **True / False?** Justify: __________________________________________________________________

---

## Section C: Short-Answer & Scenario Questions

**49. Regression vs Classification**

Consider the following two problems:

```text
Problem A:
Predict the fare of a taxi trip.

Problem B:
Predict whether a passenger survived or did not survive.
```

* **(a)** Identify whether each problem is regression or classification.
* **(b)** What type of output does each model produce?
* **(c)** What is the key difference between the target variables?

---

**50. Features and Target**

A taxi dataset contains:

```text
TRIP_MILES
TRIP_MINUTES
PASSENGER_COUNT
FARE
```

You want to predict `FARE`.

* **(a)** Which variable is the target?
* **(b)** Which variables could be used as features?
* **(c)** What do `X` and `y` represent in the machine-learning code?
* **(d)** Write a possible definition of `X` and `y` in Python.

---

**51. Pearson Correlation by Hand**

Consider:

```text
x = [1, 2, 3, 4]
y = [2, 4, 6, 8]
```

* **(a)** Calculate the mean of `x`.
* **(b)** Calculate the mean of `y`.
* **(c)** Calculate the deviations from the means.
* **(d)** Calculate the products of the deviations.
* **(e)** Calculate the squared deviations.
* **(f)** Use the Pearson formula to calculate `r`.
* **(g)** Interpret the result.

---

**52. Pearson Correlation and Nonlinear Relationships**

Consider:

```text
x = [-3, -2, -1, 0, 1, 2, 3]
y = [ 9,  4,  1, 0, 1, 4, 9]
```

The relationship is:

$$
y=x^2
$$

* **(a)** Is there a relationship between `x` and `y`?
* **(b)** Is the relationship linear or nonlinear?
* **(c)** What can happen to Pearson `r` in this example?
* **(d)** Why can Pearson correlation be misleading here?
* **(e)** What lesson should you remember when interpreting a correlation close to zero?

---

**53. Correlation with Multiple Features**

A dataset contains:

```text
TRIP_MILES
TRIP_MINUTES
PASSENGER_COUNT
HOUR
FARE
```

* **(a)** Why might we calculate the correlation between each feature and `FARE`?
* **(b)** Write four correlation questions that could be investigated.
* **(c)** If `TRIP_MILES` has `r = 0.88`, what does this suggest?
* **(d)** Does `r = 0.88` prove that distance causes fare to increase? Explain.
* **(e)** Can a feature with low Pearson correlation still be useful in a regression model? Explain.

---

**54. Training and Test Data**

You have 1,000 taxi trips.

You use:

```text
800 observations for training
200 observations for testing
```

After training a regression model, you obtain:

```text
Training RMSE = 3.2
Test RMSE     = 9.8
```

* **(a)** What does the difference between the two RMSE values suggest?
* **(b)** What machine-learning concept might this illustrate?
* **(c)** Why is test RMSE important?
* **(d)** Why should the test data not be used to fit the model?

---

**55. Interpreting a Linear Regression Equation**

Suppose a model is:

$$
\widehat{FARE}=4.50+2.10(TRIP\_MILES)
$$

* **(a)** What is the intercept?
* **(b)** What is the coefficient of `TRIP_MILES`?
* **(c)** What does the coefficient mean?
* **(d)** What fare does the model predict for a 5-mile trip?
* **(e)** Is the coefficient positive or negative?
* **(f)** What does its sign tell us?

---

**56. Multiple Linear Regression**

Suppose a model is:

$$
\widehat{FARE}
=
3.0
+
1.8(TRIP\_MILES)
+
0.5(TRIP\_MINUTES)
$$

* **(a)** How many features does this model use?
* **(b)** What does the coefficient `1.8` represent?
* **(c)** What does the coefficient `0.5` represent?
* **(d)** What is the predicted fare for a trip with 5 miles and 20 minutes?
* **(e)** Why might using both distance and time be more informative than using only distance?

---

**57. Polynomial Regression**

Consider the model:

$$
\hat{y}=b_0+b_1x+b_2x^2
$$

* **(a)** What is the polynomial degree?
* **(b)** Why can this model represent a curved relationship?
* **(c)** How is this different from simple linear regression?
* **(d)** Why is this still commonly implemented using `LinearRegression()` after creating polynomial features?

---

**58. Polynomial Degree and Overfitting**

You test the following models:

| Degree | Training RMSE | Test RMSE |
| -----: | ------------: | --------: |
|      1 |          15.0 |      15.5 |
|      2 |          10.0 |      10.8 |
|      3 |           8.5 |       9.2 |
|      5 |           5.0 |       8.0 |
|     10 |           1.5 |      14.0 |

* **(a)** Which degree has the best test RMSE?
* **(b)** Which degree has the lowest training RMSE?
* **(c)** Why are those two answers different?
* **(d)** Which degree appears to be overfitting?
* **(e)** Which degree appears to be underfitting?
* **(f)** What general lesson does this table demonstrate?

---

**59. Regression Metrics**

A regression model produces:

```text
MAE  = 4.2
RMSE = 6.8
R²   = 0.76
```

* **(a)** What does MAE measure?
* **(b)** What does RMSE measure?
* **(c)** Why is RMSE larger than MAE in this example?
* **(d)** What does an R² of 0.76 suggest?
* **(e)** Which of these metrics is expressed in the same units as the target?

---

**60. Choosing a Model**

You are comparing the following models:

```text
Linear Regression
Multiple Linear Regression
Polynomial Regression
```

Explain when each might be useful.

* **(a)** When might Linear Regression be appropriate?
* **(b)** When might Multiple Linear Regression be better?
* **(c)** When might Polynomial Regression be useful?
* **(d)** What is one risk of Polynomial Regression?

---

## Section D: Code & Concept Interpretation

**61. Selecting Features and Target**

Examine:

```python
X = df[["TRIP_MILES", "TRIP_MINUTES"]]
y = df["FARE"]
```

* **(a)** What does `X` contain?
* **(b)** What does `y` contain?
* **(c)** How many features are being used?
* **(d)** What is the target?

---

**62. Train/Test Split**

Examine:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

* **(a)** What does `X_train` contain?
* **(b)** What does `X_test` contain?
* **(c)** What does `y_train` contain?
* **(d)** What does `y_test` contain?
* **(e)** What does `test_size=0.2` mean?
* **(f)** Why do we use `random_state=42`?

---

**63. Training and Prediction**

Examine:

```python
model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

Explain the purpose of each line.

---

**64. Examining Model Coefficients**

Examine:

```python
print(model.intercept_)
print(model.coef_)
```

* **(a)** What does `intercept_` represent?
* **(b)** What does `coef_` contain?
* **(c)** If the model has two features, how many coefficients should you expect?
* **(d)** Why might the sign of a coefficient be useful to interpret?

---

**65. Evaluating a Regression Model**

Examine:

```python
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)
```

* **(a)** What does `mae` measure?
* **(b)** Why do we take the square root of MSE to calculate RMSE?
* **(c)** What does `r2` measure?
* **(d)** Why is it useful to calculate several metrics instead of only one?

---

**66. Polynomial Features**

Examine:

```python
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
```

* **(a)** What is the purpose of `PolynomialFeatures`?
* **(b)** What does `degree=2` mean?
* **(c)** Why do we use `fit_transform()` on the training data?
* **(d)** Why do we use `transform()` on the test data?
* **(e)** Why should we not independently fit the transformation on the test set?

---

**67. Polynomial Regression**

Examine:

```python
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()

model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)
```

Explain how these steps work together.

---

**68. Training and Test RMSE**

Examine:

```python
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_rmse = np.sqrt(
    mean_squared_error(y_train, train_predictions)
)

test_rmse = np.sqrt(
    mean_squared_error(y_test, test_predictions)
)
```

* **(a)** What is `train_rmse` measuring?
* **(b)** What is `test_rmse` measuring?
* **(c)** What pattern might suggest overfitting?
* **(d)** What pattern might suggest underfitting?

---

**69. Pipeline**

Examine:

```python
from sklearn.pipeline import Pipeline

model = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("regression", LinearRegression())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

* **(a)** What are the two stages in this pipeline?
* **(b)** What does the first stage do?
* **(c)** What does the second stage do?
* **(d)** What advantage does this have compared with writing the steps separately?
* **(e)** Why might students still learn the separate steps before using a pipeline?

---

**70. Scaling and Leakage**

Examine:

```python
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

* **(a)** Why is `fit()` performed using only `X_train`?
* **(b)** Why is `X_test` transformed rather than fitted separately?
* **(c)** What machine-learning problem are we trying to avoid?
* **(d)** Why is this idea important when evaluating a model?

---

## Section E: Integrated Exam Questions

**71. Complete Regression Scenario**

You are given a taxi dataset containing:

```text
TRIP_MILES
TRIP_MINUTES
PASSENGER_COUNT
FARE
```

You want to predict taxi fare.

Answer the following:

* **(a)** Is this a classification or regression problem? Why?
* **(b)** Which variable is the target?
* **(c)** Which variables could be used as features?
* **(d)** Why would you inspect Pearson correlations before training?
* **(e)** Why should you create a training and test set?
* **(f)** Explain the difference between simple and multiple linear regression.
* **(g)** When might Polynomial Regression be useful?
* **(h)** What is one danger of using a very high polynomial degree?
* **(i)** Name two regression evaluation metrics.
* **(j)** What does RMSE tell you?
* **(k)** What does a large difference between training RMSE and test RMSE suggest?
* **(l)** Why is generalization important?

---

**72. Model Comparison Scenario**

Three models produce these results:

| Model                | Training RMSE | Test RMSE |
| -------------------- | ------------: | --------: |
| Linear Regression    |          12.0 |      12.5 |
| Polynomial Degree 2  |           8.0 |       8.7 |
| Polynomial Degree 10 |           1.2 |      18.4 |

Answer:

* **(a)** Which model performs best on the training data?
* **(b)** Which model performs best on the test data?
* **(c)** Which model appears to be overfitting?
* **(d)** Which model may be underfitting?
* **(e)** Why should you not automatically choose the model with the lowest training RMSE?
* **(f)** Which model would you choose if your main goal is good performance on unseen data? Explain.

---

**73. Correlation and Regression Scenario**

Suppose you obtain the following correlations with `FARE`:

```text
TRIP_MILES       r = 0.88
TRIP_MINUTES     r = 0.81
PASSENGER_COUNT  r = 0.05
HOUR             r = -0.02
```

Answer:

* **(a)** Which feature has the strongest positive linear relationship with fare?
* **(b)** Which feature has the weakest linear relationship with fare?
* **(c)** Does the low correlation of `PASSENGER_COUNT` prove that it is useless?
* **(d)** Does the high correlation of `TRIP_MILES` prove that it must be the only feature used?
* **(e)** Why should correlation be considered a starting point rather than the final model-selection decision?

---

**74. Underfitting and Overfitting Scenario**

A student trains three models:

```text
Model A:
Training RMSE = 18
Test RMSE     = 19

Model B:
Training RMSE = 9
Test RMSE     = 10

Model C:
Training RMSE = 1
Test RMSE     = 25
```

Answer:

* **(a)** Which model most likely underfits?
* **(b)** Which model appears to generalize best?
* **(c)** Which model most clearly overfits?
* **(d)** Explain the reasoning using training and test performance.
* **(e)** What could the student try to reduce the overfitting in Model C?

---

**75. Complete Scikit-Learn Regression Workflow**

Write a complete basic regression workflow for a dataset called `df`.

Your solution should include:

* **(a)** Selecting the features and target.
* **(b)** Splitting the data into training and test sets.
* **(c)** Creating a `LinearRegression` model.
* **(d)** Training the model.
* **(e)** Making predictions.
* **(f)** Calculating MAE.
* **(g)** Calculating RMSE.
* **(h)** Calculating R².
* **(i)** Briefly explaining why the test set is important.

---

# Solutions

## Section A: Multiple Choice

**1. B** — Supervised regression learns from examples where the correct numerical target values are known.

**2. B** — `X` normally represents the input features and `y` represents the numerical target.

**3. C** — Taxi fare is a numerical quantity, so predicting it is a regression problem.

**4. A** — Linear regression models a continuous numerical target using a linear relationship between features and the prediction.

**5. A** — Simple linear regression can be represented as:

$$
y=b_0+b_1x
$$

**6. B** — Simple linear regression uses one feature, while multiple linear regression uses two or more features.

**7. B** — The model uses both `TRIP_MILES` and `TRIP_MINUTES`, so it is multiple linear regression.

**8. A** — The intercept is the model's predicted value when all features are zero.

**9. A** — In a multiple regression model, a positive coefficient means that increasing that feature is associated with an increase in the prediction while the other model features are held constant.

**10. A** — Pearson correlation measures the strength and direction of a linear relationship between two numerical variables.

**11. C** — Pearson `r` ranges from `-1` to `+1`.

**12. A** — `r = 0.92` indicates a strong positive linear relationship.

**13. B** — `r = -0.85` indicates a strong negative linear relationship.

**14. B** — Pearson correlation is specifically a measure of linear association.

**15. B** — \(y=x^2\) is nonlinear, and Pearson `r` can be close to zero even though the variables have a strong mathematical relationship.

**16. A** — Correlations provide an initial understanding of how individual features are linearly associated with the target.

**17. B** — The test set lets us estimate how well the trained model performs on unseen observations.

**18. A** — `train_test_split()` is the scikit-learn function commonly used for this task.

**19. A** — Approximately 20% of the observations are assigned to the test set.

**20. B** — `random_state` makes the randomized split reproducible.

**21. B** — `fit()` learns model parameters from the training data.

**22. B** — `predict()` generates predicted target values for the supplied observations.

**23. A** — MAE is the mean absolute error.

**24. B** — MSE squares the errors, giving large errors more influence.

**25. B** — RMSE is the square root of MSE and is expressed in the same units as the target.

**26. A** — R² of `0.85` indicates that the model explains a large proportion of the variation in the target relative to the baseline/reference represented by the metric.

**27. A** — Polynomial regression creates polynomial terms and then can use a linear regression estimator to fit them.

**28. B** — Degree 2 creates terms such as squared features and, depending on the input, interaction terms.

**29. C** — Increasing the degree makes the model more flexible, which can help initially but can eventually cause overfitting.

**30. B** — Very low training error combined with much higher test error is a classic sign of overfitting.

**31. B** — High training and high test error can indicate that the model is too simple or otherwise unable to capture the important structure.

**32. B** — A very flexible polynomial can fit training noise as well as the useful underlying relationship.

**33. A** — Scaling puts numerical variables onto comparable scales and is particularly important for algorithms that depend on numerical scale or distance.

**34. A** — A pipeline can combine preprocessing and a model into one reproducible workflow.

**35. A** — Learning each step separately helps students understand what the pipeline is doing internally.

---

## Section B: True or False

**36. True** — Regression is generally used when the target is a numerical quantity rather than a category.

**37. False** — A Pearson correlation near zero means little or no linear association; a nonlinear relationship may still exist.

**38. True** — \(y=2x\) is a straight-line relationship and therefore linear.

**39. True** — \(y=x^2\) is a curved, nonlinear relationship.

**40. False** — Correlation is useful for exploration but does not by itself determine whether a feature belongs in the final model.

**41. False** — The model should be fitted using training data, while the test set is reserved for evaluation.

**42. True** — A model can fit the training observations extremely well but generalize poorly, which is overfitting.

**43. True** — Underfitting occurs when the model is too simple or otherwise unable to capture important patterns.

**44. False** — Overfitting usually means training performance is much better than test performance.

**45. False** — Increasing degree can improve training performance but can eventually reduce test performance because of overfitting.

**46. True** — RMSE has the same units as the target variable.

**47. False** — MSE squares errors; MAE uses absolute values and does not square the errors.

**48. True** — A pipeline can combine preprocessing and modeling steps into one workflow.

---

## Section C: Short Answer

**49. Regression vs Classification**

**(a)**

* Problem A: **Regression**
* Problem B: **Binary classification**

**(b)**

* Problem A produces a numerical prediction such as `€12.50`.
* Problem B produces a class prediction such as `Survived` or `Did not survive`.

**(c)** The regression target is numerical and can take a range of values, while the classification target represents categories.

---

**50. Features and Target**

**(a)** The target is:

```text
FARE
```

**(b)** Possible features are:

```text
TRIP_MILES
TRIP_MINUTES
PASSENGER_COUNT
```

**(c)** `X` represents the input features and `y` represents the target.

**(d)** For example:

```python
X = df[[
    "TRIP_MILES",
    "TRIP_MINUTES",
    "PASSENGER_COUNT"
]]

y = df["FARE"]
```

---

**51. Pearson Correlation by Hand**

Given:

```text
x = [1, 2, 3, 4]
y = [2, 4, 6, 8]
```

**(a)**

$$
\bar{x}=2.5
$$

**(b)**

$$
\bar{y}=5
$$

**(c)** Deviations:

| `x` | `x - x̄` | `y` | `y - ȳ` |
| --: | -------: | --: | ------: |
|   1 |     -1.5 |   2 |      -3 |
|   2 |     -0.5 |   4 |      -1 |
|   3 |     +0.5 |   6 |      +1 |
|   4 |     +1.5 |   8 |      +3 |

**(d)** Products:

$$
4.5,\;0.5,\;0.5,\;4.5
$$

Their sum is:

$$
10
$$

**(e)**

$$
\sum(x-\bar{x})^2=5
$$

and:

$$
\sum(y-\bar{y})^2=20
$$

**(f)**

$$
r=
\frac{10}{\sqrt{5\times20}}
=
\frac{10}{10}
=1
$$

**(g)** There is a perfect positive linear relationship between `x` and `y`.

---

**52. Pearson Correlation and Nonlinear Relationships**

**(a)** Yes. The relationship is:

$$
y=x^2
$$

**(b)** It is nonlinear.

**(c)** Pearson correlation can be:

$$
r=0
$$

for this symmetric example.

**(d)** Pearson measures linear association. The data have a strong curved relationship but do not follow a straight-line trend.

**(e)** A correlation close to zero means little or no **linear** association; it does not prove that no relationship exists.

---

**53. Correlation with Multiple Features**

**(a)** We calculate correlations to get an initial understanding of how individual features are linearly associated with the target.

**(b)** For example:

$$
corr(TRIP\_MILES,FARE)
$$

$$
corr(TRIP\_MINUTES,FARE)
$$

$$
corr(PASSENGER\_COUNT,FARE)
$$

$$
corr(HOUR,FARE)
$$

**(c)** `TRIP_MILES` has a strong positive linear relationship with fare.

**(d)** No. Correlation measures association, not causation.

**(e)** Yes. A feature can have a nonlinear relationship with the target or provide useful information in combination with other features.

---

**54. Training and Test Data**

**(a)** The test error is much larger than the training error.

**(b)** This suggests possible **overfitting**.

**(c)** Test RMSE provides information about how well the model generalizes to unseen observations.

**(d)** If the test set were used during training, it would no longer represent genuinely unseen data.

---

**55. Interpreting a Linear Regression Equation**

Given:

$$
\widehat{FARE}=4.50+2.10(TRIP\_MILES)
$$

**(a)** Intercept:

$$
4.50
$$

**(b)** Coefficient:

$$
2.10
$$

**(c)** A one-unit increase in `TRIP_MILES` is associated with an increase of 2.10 units in predicted fare.

**(d)** For 5 miles:

$$
4.50+2.10(5)
$$

$$
=4.50+10.50
$$

$$
=15.00
$$

Predicted fare:

$$
\boxed{15.00}
$$

**(e)** Positive.

**(f)** It means that, according to this model, larger trip distance is associated with larger predicted fare.

---

**56. Multiple Linear Regression**

Given:

$$
\widehat{FARE}
=
3.0
+
1.8(TRIP\_MILES)
+
0.5(TRIP\_MINUTES)
$$

**(a)** Two features.

**(b)** The coefficient `1.8` represents the expected change in predicted fare associated with a one-unit increase in trip miles, holding trip minutes constant.

**(c)** The coefficient `0.5` represents the expected change in predicted fare associated with a one-unit increase in trip minutes, holding trip miles constant.

**(d)** For 5 miles and 20 minutes:

$$
3.0+1.8(5)+0.5(20)
$$

$$
=3+9+10
$$

$$
=22
$$

Predicted fare:

$$
\boxed{22}
$$

**(e)** Distance and time provide different information. Two trips with the same distance can take different amounts of time, so using both may give the model more predictive information.

---

**57. Polynomial Regression**

Given:

$$
\hat{y}=b_0+b_1x+b_2x^2
$$

**(a)** Degree 2.

**(b)** The \(x^2\) term allows the relationship between `x` and `y` to curve rather than remain a straight line.

**(c)** Simple linear regression uses:

$$
y=b_0+b_1x
$$

while polynomial regression can include:

$$
x^2,x^3,\ldots
$$

**(d)** Once the polynomial features are created, linear regression can estimate coefficients for those features. The relationship is nonlinear in the original input `x`, but linear in the coefficients being estimated.

---

**58. Polynomial Degree and Overfitting**

**(a)** Degree 5 has the best test RMSE:

$$
8.0
$$

**(b)** Degree 10 has the lowest training RMSE:

$$
1.5
$$

**(c)** A more complex model can fit training observations extremely closely without necessarily generalizing well.

**(d)** Degree 10 appears to be overfitting because training RMSE is extremely low while test RMSE increases substantially.

**(e)** Degree 1 may be underfitting because both training and test error are relatively high.

**(f)** Increasing model complexity can initially improve performance, but beyond some point it can hurt generalization.

---

**59. Regression Metrics**

Given:

```text
MAE  = 4.2
RMSE = 6.8
R²   = 0.76
```

**(a)** MAE is the average absolute prediction error.

**(b)** RMSE is based on the square root of the average squared prediction error.

**(c)** RMSE gives larger errors more influence because the errors are squared before averaging.

**(d)** An R² of 0.76 indicates that the model explains a substantial proportion of the variation in the target relative to the metric's reference baseline.

**(e)** MAE and RMSE are expressed in the same units as the target. In particular, RMSE is explicitly obtained by taking the square root of squared target-unit errors.

---

**60. Choosing a Model**

**(a)** Linear Regression is appropriate when a roughly linear relationship provides a good representation of the data.

**(b)** Multiple Linear Regression is useful when several features contribute to predicting the target.

**(c)** Polynomial Regression is useful when the relationship between features and target appears nonlinear and a curved relationship can improve predictions.

**(d)** A major risk is overfitting, especially with unnecessarily high polynomial degrees.

---

## Section D: Code & Concept Interpretation

**61. Selecting Features and Target**

```python
X = df[["TRIP_MILES", "TRIP_MINUTES"]]
y = df["FARE"]
```

**(a)** `X` contains the two feature columns.

**(b)** `y` contains the target values.

**(c)** Two features.

**(d)** `FARE` is the target.

---

**62. Train/Test Split**

**(a)** `X_train` contains the training feature observations.

**(b)** `X_test` contains the test feature observations.

**(c)** `y_train` contains the target values corresponding to `X_train`.

**(d)** `y_test` contains the target values corresponding to `X_test`.

**(e)** Approximately 20% of the observations are placed in the test set.

**(f)** `random_state=42` makes the random split reproducible.

---

**63. Training and Prediction**

```python
model = LinearRegression()
```

Creates a linear regression model object.

```python
model.fit(X_train, y_train)
```

Trains the model by learning its parameters from the training data.

```python
y_pred = model.predict(X_test)
```

Uses the trained model to predict target values for the test observations.

---

**64. Examining Model Coefficients**

**(a)** `intercept_` is the learned intercept.

**(b)** `coef_` contains the learned coefficient for each feature.

**(c)** Two coefficients.

**(d)** The sign gives information about the direction of the model's relationship with that feature, holding other model features constant.

---

**65. Evaluating a Regression Model**

**(a)** `mae` measures the average absolute prediction error.

**(b)** We take the square root of MSE to return the metric to the target's original units.

**(c)** `r2` measures how much variation in the target the model explains relative to the reference baseline used by R².

**(d)** Different metrics emphasize different aspects of model performance. Looking at several metrics provides a more complete picture.

---

**66. Polynomial Features**

**(a)** `PolynomialFeatures` creates additional polynomial and interaction terms from the input features.

**(b)** `degree=2` means terms up to degree two are created.

**(c)** `fit_transform()` learns whatever transformation information is needed from the training features and then transforms them.

**(d)** `transform()` applies the already learned transformation to the test features.

**(e)** We do not fit separately on the test set because the test set should remain unseen during model development. Fitting preprocessing on it can cause information leakage.

---

**67. Polynomial Regression**

The process works as follows:

```python
poly = PolynomialFeatures(degree=2)
```

Creates a polynomial-feature transformation.

```python
X_train_poly = poly.fit_transform(X_train)
```

Learns the transformation using the training data and creates the polynomial features.

```python
X_test_poly = poly.transform(X_test)
```

Applies the same learned transformation to the test data.

```python
model = LinearRegression()
```

Creates the regression estimator.

```python
model.fit(X_train_poly, y_train)
```

Learns coefficients using the expanded polynomial features.

```python
y_pred = model.predict(X_test_poly)
```

Produces predictions for the test observations.

---

**68. Training and Test RMSE**

**(a)** `train_rmse` measures prediction error on observations used to train the model.

**(b)** `test_rmse` measures prediction error on unseen test observations.

**(c)** Very low training RMSE combined with much higher test RMSE suggests overfitting.

**(d)** High training RMSE together with high test RMSE can indicate underfitting.

---

**69. Pipeline**

**(a)** The two stages are:

1. `PolynomialFeatures`
2. `LinearRegression`

**(b)** The first stage creates polynomial features.

**(c)** The second stage fits the linear regression model using those features.

**(d)** The pipeline organizes the steps into one object, making it easier to train and predict without manually calling each step.

**(e)** Learning the steps separately helps students understand exactly what happens before those operations are combined.

---

**70. Scaling and Leakage**

**(a)** The scaler is fitted only on `X_train` because the training data is what the model-development process is allowed to learn from.

**(b)** `X_test` is transformed using the scaling information learned from the training data.

**(c)** We are trying to avoid **data leakage**.

**(d)** Data leakage can make test performance look better than it should because information from the test set has indirectly influenced the training process.

---

## Section E: Integrated Exam Questions

**71. Complete Regression Scenario**

**(a)** Regression, because the target `FARE` is a numerical value.

**(b)** `FARE`.

**(c)** Possible features include `TRIP_MILES`, `TRIP_MINUTES`, and `PASSENGER_COUNT`.

**(d)** Pearson correlations provide an initial view of the linear relationships between individual features and the target.

**(e)** We want to estimate performance on data that the model did not use during training.

**(f)** Simple linear regression uses one feature; multiple linear regression can use several features.

**(g)** Polynomial Regression is useful when a nonlinear relationship may be represented better by adding polynomial terms.

**(h)** A very high polynomial degree can cause overfitting.

**(i)** Examples include MAE, RMSE, MSE, and R².

**(j)** RMSE summarizes prediction error while giving larger errors greater influence and expressing the result in the same units as the target.

**(k)** A large gap can indicate overfitting.

**(l)** The model ultimately needs to make useful predictions on observations it has not seen before.

---

**72. Model Comparison Scenario**

**(a)** Model C has the lowest training RMSE:

$$
1.2
$$

**(b)** Model B has the lowest test RMSE:

$$
10.0
$$

**(c)** Model C appears to be overfitting because its training performance is excellent but its test performance is poor.

**(d)** Model A may be underfitting because both training and test RMSE are relatively high.

**(e)** A very low training error can come from memorizing training patterns rather than learning generalizable relationships.

**(f)** Model B would be the preferred choice among these models if the main objective is good performance on unseen data.

---

**73. Correlation and Regression Scenario**

**(a)** `TRIP_MILES` has the strongest positive linear relationship with fare:

$$
r=0.88
$$

**(b)** `HOUR` has the weakest linear relationship:

$$
r=-0.02
$$

**(c)** No. Low Pearson correlation means weak linear association, but the feature could still have nonlinear or interaction effects.

**(d)** No. Correlation does not establish causation and does not imply that other features are unnecessary.

**(e)** Correlation is exploratory. Final model decisions should consider model performance, domain knowledge, data quality, and possible nonlinear relationships.

---

**74. Underfitting and Overfitting Scenario**

**(a)** Model A most likely underfits.

**(b)** Model B generalizes best because it has the lowest test RMSE.

**(c)** Model C clearly overfits.

**(d)** Model A performs poorly on both training and test data. Model B performs well on both. Model C performs extremely well on training data but poorly on test data.

**(e)** Possible strategies include:

* reducing model complexity;
* reducing polynomial degree;
* selecting more appropriate features;
* using regularization;
* collecting more data;
* using better validation procedures.

---

**75. Complete Scikit-Learn Regression Workflow**

One possible solution is:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

# Select features and target
X = df[["TRIP_MILES", "TRIP_MINUTES"]]
y = df["FARE"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)

# Calculate RMSE
rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

# Calculate R²
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)
```

The test set is important because it gives us an estimate of how well the trained model can predict observations that were not used during model fitting.
