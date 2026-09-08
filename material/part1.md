# Theory — Introduction to Regression with Auto MPG

This theory section introduces the main concepts needed for **Activity 1: Introduction to Regression with Auto MPG**.

The focus is on understanding what regression is, how the data are represented, how we inspect relationships between variables, how a linear regression model works, and how we evaluate its predictions.

The goal is not to study every regression technique in this section. More advanced topics will be introduced later.

---

# 1. What Is Regression?

## 1.1 Supervised Learning

Regression is a type of **supervised machine learning**.

In supervised learning, the model learns from examples where the correct answer is already known.

For example, suppose we have information about many cars:

| Weight | Horsepower | MPG |
| -----: | ---------: | --: |
|   3500 |        130 |  20 |
|   2800 |         90 |  28 |
|   2200 |         70 |  34 |

The model can use the known `MPG` values to learn relationships between the car characteristics and fuel efficiency.

Once the model has learned from the training examples, it can be used to predict MPG for a car that it has not seen before.

The general supervised-learning structure is:

```text
Known examples
      ↓
Features + target
      ↓
Learning algorithm
      ↓
Trained model
      ↓
Prediction for new data
```

---

# 2. Regression vs Classification

One of the first questions in a machine-learning problem is:

> What type of target are we trying to predict?

The distinction between regression and classification is important.

## Classification

Classification predicts a **category**.

For example:

```text
spam / not spam
survived / did not survive
setosa / versicolor / virginica
```

The output is a class.

## Regression

Regression predicts a **numerical value**.

Examples include:

```text
house price
temperature
salary
taxi fare
fuel efficiency
```

For Auto MPG, the target is:

```text
mpg
```

This is a numerical quantity, so the problem is a regression problem.

---

# 3. Features and Target

A supervised machine-learning dataset normally contains:

* **features** — information used to make a prediction;
* **target** — the value we want to predict.

For Auto MPG, possible features include:

```text
cylinders
displacement
horsepower
weight
acceleration
model_year
origin
```

The target is:

```text
mpg
```

We often represent the features by:

$$
X
$$

and the target by:

$$
y
$$

So conceptually:

$$
X \rightarrow y
$$

means:

> Use the features in `X` to predict the target `y`.

For a simple example:

$$
X=\text{weight}
$$

and:

$$
y=\text{mpg}
$$

For a multiple-feature example:

$$
X=
\{
\text{weight},
\text{horsepower},
\text{displacement}
\}
$$

and:

$$
y=\text{mpg}
$$

---

# 4. Understanding the Auto MPG Dataset

The Auto MPG dataset contains information about automobiles and their fuel efficiency. The UCI Machine Learning Repository describes 398 observations and automobile variables including cylinders, displacement, horsepower, weight, acceleration, model year, and origin. `mpg` is the continuous target, and `horsepower` contains missing values. ([uci-ics-mlr-prod.aws.uci.edu](https://uci-ics-mlr-prod.aws.uci.edu/dataset/9/auto%2Bmpg?utm_source=chatgpt.com))

This dataset is useful for introductory regression because there is an intuitive question:

> Can characteristics of a car be used to predict its fuel efficiency?

For example:

```text
vehicle weight
horsepower
engine characteristics
       ↓
Regression model
       ↓
predicted MPG
```

The dataset also illustrates an important feature of real-world machine-learning data:

> A dataset may contain several variables with different meanings, scales, and data-quality issues.

For example, `horsepower` contains missing values. In a complete machine-learning project, we would have to decide how to handle those missing values. In the introductory activity, we avoid making missing-value handling the main focus by selecting appropriate complete features for the main examples.

---

# 5. What Does MPG Mean?

## 5.1 Miles Per Gallon

`MPG` stands for:

> **miles per gallon**

It is a measure of fuel efficiency.

For example:

```text
20 MPG
```

means that a vehicle can travel approximately 20 miles using one US gallon of fuel.

A higher MPG value generally means greater fuel efficiency.

For example:

```text
20 MPG
30 MPG
40 MPG
```

The 40 MPG car is more fuel-efficient than the 20 MPG car.

---

# 6. MPG and European Units

Students who normally use the metric system may be more familiar with:

* kilometres per litre (`km/L`), or
* litres per 100 kilometres (`L/100 km`).

These measures should not be confused.

## MPG and km/L

Both are "distance per amount of fuel" measures.

For US MPG used in the Auto MPG dataset:

$$
1\ MPG \approx 0.425\ km/L
$$

So:

$$
km/L\approx MPG\times0.425
$$

For example:

$$
30\times0.425=12.75
$$

Therefore:

$$
30\ MPG\approx12.75\ km/L
$$

---

## MPG and L/100 km

`L/100 km` works in the opposite direction.

For `L/100 km`:

> Lower is better.

For MPG:

> Higher is better.

This can initially feel confusing.

For example:

```text
MPG:
40 is better than 20

L/100 km:
5 is better than 10
```

The lab keeps the original target:

```text
mpg
```

There is no need to convert it for the machine-learning model.

The conversion is only provided to make the values easier to interpret.

---

# 7. Exploring Data Before Building a Model

A machine-learning model should not be treated as a black box.

Before training a model, it is useful to ask:

* What variables are available?
* What is the target?
* Are there missing values?
* What are the typical values?
* Are some observations unusual?
* Do the features appear to have relationships with the target?

This is called **exploratory data analysis**, or EDA.

Typical first steps include:

```python
df.head()
```

to inspect the first rows,

```python
df.info()
```

to inspect data types and missing values,

and:

```python
df.describe()
```

to examine numerical summaries.

These steps do not build a predictive model.

They help us understand what we are about to model.

---

# 8. Why Use a Scatter Plot?

Suppose we want to predict MPG from vehicle weight.

We can draw:

```text
x-axis → weight
y-axis → mpg
```

using a scatter plot.

Each point represents one car.

A scatter plot helps us see whether the observations seem to follow a pattern.

For example, we may observe that:

```text
weight increases
        ↓
mpg generally decreases
```

This suggests a negative relationship.

---

# 9. Positive and Negative Relationships

Suppose increasing `x` tends to increase `y`.

This is a **positive relationship**.

```text
x ↑
y ↑
```

Suppose increasing `x` tends to decrease `y`.

This is a **negative relationship**.

```text
x ↑
y ↓
```

For Auto MPG, weight and MPG generally show a negative relationship.

This makes sense because heavier vehicles tend to require more energy to move and therefore often have lower fuel efficiency.

The relationship is not perfect because fuel efficiency depends on many characteristics of the vehicle.

---

# 10. Linear Relationships

A linear relationship can be approximated by a straight line.

The general equation is:

$$
y=b_0+b_1x
$$

where:

* \(y\) is the predicted target;
* \(x\) is the feature;
* \(b_0\) is the intercept;
* \(b_1\) is the coefficient or slope.

For Auto MPG:

$$
\widehat{MPG}=b_0+b_1(Weight)
$$

The `hat` over \(y\):

$$
\hat{y}
$$

means that this is a **prediction**, not necessarily the actual observed value.

---

# 11. Pearson Correlation Coefficient

Before fitting a regression model, we can summarize a linear relationship using **Pearson Correlation Coefficient**.

It is represented by:

$$
r
$$

Its value is between:

$$
-1\le r\le1
$$

## Interpretation

### Positive correlation

$$
r>0
$$

suggests a positive linear association.

### Negative correlation

$$
r<0
$$

suggests a negative linear association.

### Near-zero correlation

$$
r\approx0
$$

suggests little or no linear association.

The important word is:

> **linear**

---

# 12. What Does the Magnitude of `r` Mean?

Consider:

$$
r=0.90
$$

This indicates a strong positive linear relationship.

Now:

$$
r=-0.90
$$

This indicates a strong negative linear relationship.

The sign tells us the direction.

The magnitude tells us the strength of the linear association.

For example:

```text
r = +0.90
```

and:

```text
r = -0.90
```

have similar strengths but opposite directions.

---

# 13. Correlation Does Not Mean Causation

Suppose:

$$
r=-0.85
$$

between weight and MPG.

This tells us that the two variables have a strong negative linear association.

It does **not** prove:

> Weight alone causes MPG to decrease.

Other variables may also affect the relationship.

Correlation is therefore useful for:

> **describing association**

but it is not proof of:

> **causation**.

---

# 14. A Major Limitation of Pearson Correlation

Pearson correlation measures linear association.

It does not detect every possible mathematical relationship.

Consider:

$$
y=x^2
$$

with:

| `x` | `y` |
| --: | --: |
|  -3 |   9 |
|  -2 |   4 |
|  -1 |   1 |
|   0 |   0 |
|   1 |   1 |
|   2 |   4 |
|   3 |   9 |

There is clearly a relationship:

$$
y=x^2
$$

but it is curved rather than linear.

In this symmetric example:

$$
r=0
$$

even though `x` completely determines `y`.

Therefore:

> A Pearson correlation close to zero does not prove that there is no relationship.

It means that there is little or no **linear** relationship.

---

# 15. Important Clarification: `y = 2x` Is Linear

Multiplication by a constant does not make a relationship nonlinear.

For example:

$$
y=2x
$$

is linear.

Likewise:

$$
y=2x+5
$$

is linear.

These relationships can have:

$$
r=1
$$

when the relationship is exact and increasing.

The important contrast is:

$$
y=2x
$$

versus:

$$
y=x^2
$$

The first is a straight-line relationship.

The second is curved.

---

# 16. Why Examine Every Feature?

In machine learning, we often have several features.

For example:

```text
weight
horsepower
displacement
cylinders
model_year
```

and want to predict:

```text
mpg
```

We can investigate:

$$
corr(weight,mpg)
$$

$$
corr(horsepower,mpg)
$$

$$
corr(displacement,mpg)
$$

and so on.

This provides an initial understanding of the data.

For example, suppose we found:

| Feature      | Correlation with MPG |
| ------------ | -------------------: |
| Weight       |                -0.83 |
| Horsepower   |                -0.78 |
| Displacement |                -0.81 |
| Model year   |                +0.58 |

We could say that, in this dataset, these features have different strengths and directions of linear association with MPG.

However, we should not conclude that:

> "The highest correlation feature must be the only feature used."

A feature can provide useful information even if its individual correlation is relatively weak.

---

# 17. Correlation Between Features

It is also possible to compare features with one another.

For example:

$$
corr(weight,displacement)
$$

might be relatively high.

That would indicate that heavier vehicles in this dataset tend to have larger engine displacement.

This is important because machine-learning features can contain overlapping information.

When predictor variables are strongly related to each other, the situation is commonly described as **multicollinearity**.

Multicollinearity becomes especially important when interpreting regression coefficients.

For the introductory activity, the main point is simply:

> Correlation can describe relationships between features and the target, and also relationships among features.

More detailed treatment of multicollinearity can be introduced later.

---

# 18. From Correlation to Regression

Correlation and regression are related, but they answer different questions.

### Correlation asks:

> How strongly are two numerical variables linearly associated?

### Regression asks:

> Can we use one or more variables to predict a numerical target?

For example:

Correlation:

$$
corr(weight,mpg)
$$

Regression:

$$
\widehat{MPG}
=
b_0+b_1(Weight)
$$

Correlation summarizes a relationship.

Regression creates a model that can be used for prediction.

---

# 19. Simple Linear Regression

When one feature is used to predict a numerical target, we can use **simple linear regression**.

The mathematical form is:

$$
\hat{y}=b_0+b_1x
$$

For Auto MPG:

$$
\widehat{MPG}
=
b_0+b_1(Weight)
$$

This means that the model uses a straight line to describe the relationship between weight and MPG.

---

# 20. What Does the Model Learn?

The regression model needs to determine:

$$
b_0
$$

and:

$$
b_1
$$

The intercept:

$$
b_0
$$

controls the vertical position of the line.

The coefficient:

$$
b_1
$$

controls its slope.

For example, if:

$$
b_1<0
$$

the line slopes downward.

If:

$$
b_1>0
$$

the line slopes upward.

---

# 21. Interpreting the Coefficient

Suppose a model has:

$$
\widehat{MPG}
=
50-0.005(Weight)
$$

The coefficient is:

$$
-0.005
$$

This means that increasing weight by one unit is associated with a decrease of `0.005` MPG in the model's prediction.

The exact interpretation depends on the unit of the feature.

This is important because changing the unit of measurement changes the numerical coefficient.

For example, a coefficient calculated using pounds will have a different numerical value from one calculated using kilograms.

---

# 22. Interpreting Multiple Regression Coefficients

Later, several features may be included:

$$
\widehat{MPG}
=
b_0+
b_1(Weight)+
b_2(Displacement)+
b_3(Cylinders)
$$

In this situation, the coefficient for one feature is interpreted while the other model features are held constant.

For example:

> The coefficient for weight represents the expected change in predicted MPG associated with a one-unit increase in weight, holding the other included features constant.

This is an important distinction from simple correlation.

---

# 23. The Intercept

The intercept is:

$$
b_0
$$

It represents the model's prediction when all features are equal to zero.

For:

$$
\widehat{MPG}=b_0+b_1(Weight)
$$

the intercept is the predicted MPG when:

$$
Weight=0
$$

However, zero weight is not a realistic automobile.

Therefore, the intercept may not have a meaningful physical interpretation.

That does not mean it is unnecessary.

The intercept is part of the fitted mathematical model.

---

# 24. What Does `fit()` Mean?

In scikit-learn:

```python
model.fit(X_train, y_train)
```

means:

> Learn the model parameters from the training examples.

For linear regression, the algorithm estimates values for:

* the intercept;
* one coefficient for each feature.

The model therefore changes from an untrained object into a fitted predictive model.

---

# 25. What Does `predict()` Mean?

After training:

```python
y_pred = model.predict(X_test)
```

means:

> Use the fitted model to estimate the target values for the observations in `X_test`.

The predictions are numerical values.

For example:

```text
Actual MPG     Predicted MPG

28             27.4
22             23.1
35             31.8
```

Predictions rarely match real observations exactly.

A useful model should instead produce predictions that are reasonably close to the true values.

---

# 26. Why Do Predictions Have Errors?

A simple model might use only:

```text
weight
```

to predict:

```text
mpg
```

But fuel efficiency is influenced by many other factors.

For example:

* horsepower;
* engine displacement;
* number of cylinders;
* vehicle design;
* model year;
* aerodynamics;
* transmission;
* other characteristics.

Therefore:

$$
Weight \rightarrow MPG
$$

is an incomplete description of reality.

The regression line summarizes an overall relationship rather than describing every vehicle perfectly.

---

# 27. Training and Test Data

A fundamental machine-learning principle is that a model should be evaluated on data that it did not use for training.

Suppose we have 398 observations.

We could divide them approximately into:

```text
80% → training
20% → test
```

The training data are used to learn the model.

The test data are reserved for evaluation.

The basic idea is:

```text
All data
   |
   +-------- Training data
   |            ↓
   |         Learn model
   |
   +-------- Test data
                ↓
          Evaluate model
```

---

# 28. Why Not Train and Test on the Same Data?

Suppose a flexible model has already seen every observation.

If we then evaluate it on exactly those same observations, we are asking:

> How well does the model reproduce data it has already seen?

That is not the same as asking:

> How well will it predict new observations?

Machine learning is usually interested in the second question.

Therefore, the test set should represent unseen observations.

---

# 29. `train_test_split()`

Scikit-learn provides:

```python
train_test_split()
```

For example:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Here:

* `X_train` contains training features;
* `y_train` contains training targets;
* `X_test` contains test features;
* `y_test` contains the true test targets.

---

# 30. The Meaning of `test_size`

If:

```python
test_size=0.2
```

then approximately 20% of the observations are assigned to the test set.

The remaining approximately 80% are used for training.

The exact counts depend on the dataset size and splitting behavior.

---

# 31. The Meaning of `random_state`

The data split is randomized.

Setting:

```python
random_state=42
```

makes the random operation reproducible.

The value `42` is not special.

Another fixed integer could be used.

The important idea is:

> Using the same random state allows the same code to produce the same split again.

This is useful for:

* teaching;
* debugging;
* comparing experiments;
* reproducing results.

---

# 32. Evaluating Regression Models

Once predictions have been generated, we need to measure their quality.

Regression uses metrics that compare:

$$
y
$$

the actual values, with:

$$
\hat{y}
$$

the predicted values.

Important introductory regression metrics are:

* MAE;
* MSE;
* RMSE;
* R².

---

# 33. Mean Absolute Error — MAE

MAE stands for:

> **Mean Absolute Error**

It is calculated as:

$$
MAE=
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat{y}_i|
$$

The steps are:

1. calculate the prediction error;
2. take its absolute value;
3. calculate the average.

For example:

```text
Actual:     30
Predicted:  27
```

The error is:

$$
30-27=3
$$

The absolute error is:

$$
|3|=3
$$

MAE summarizes these absolute errors across all observations.

---

# 34. Interpreting MAE

Suppose:

$$
MAE=2.5
$$

and the target is MPG.

We can describe this approximately as:

> The model's predictions are, on average, about 2.5 MPG away from the actual values in absolute terms.

MAE is easy to understand because it uses the same units as the target.

---

# 35. Mean Squared Error — MSE

MSE stands for:

> **Mean Squared Error**

It is calculated as:

$$
MSE=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
$$

The important difference is that the errors are squared.

For example:

```text
error = 2
```

becomes:

$$
2^2=4
$$

while:

```text
error = 5
```

becomes:

$$
5^2=25
$$

Therefore, large errors have a much greater influence on MSE.

---

# 36. Root Mean Squared Error — RMSE

RMSE is:

$$
RMSE=\sqrt{MSE}
$$

Taking the square root returns the result to the same units as the original target.

For example, if the target is MPG:

$$
RMSE
$$

is also expressed in MPG.

Like MAE, lower values generally indicate smaller prediction errors.

But RMSE gives greater influence to large errors.

---

# 37. MAE vs RMSE

Consider two models.

### Model A

```text
MAE = 2.0
RMSE = 2.5
```

### Model B

```text
MAE = 2.0
RMSE = 5.0
```

The models have the same MAE but different RMSE.

This can happen when Model B makes some much larger errors.

Therefore:

> RMSE is especially sensitive to large prediction errors.

This is one reason it is useful to examine both MAE and RMSE.

---

# 38. R² — Coefficient of Determination

R² is another common regression metric.

It compares the model with a reference prediction based on the mean target value.

A useful conceptual interpretation is:

> How much variation in the target is explained by the model relative to that baseline?

R² is not classification accuracy.

For example:

$$
R^2=0.75
$$

does **not** mean:

> "75% of predictions are correct."

Regression predictions are numerical and can differ from the actual values by different amounts.

---

# 39. Why Use More Than One Regression Metric?

Different metrics show different aspects of model behavior.

### MAE

Easy to interpret as average absolute error.

### RMSE

Emphasizes larger errors.

### R²

Provides a relative measure of how well the model explains target variation compared with the mean-based reference.

Therefore, examining several metrics gives a fuller picture than relying on one number alone.

---

# 40. Actual vs Predicted Plot

Another useful evaluation method is a scatter plot of:

```text
Actual values
```

against:

```text
Predicted values
```

A perfect prediction would satisfy:

$$
\hat{y}=y
$$

So the ideal points would lie on a diagonal line.

For example:

```text
Predicted
   ^
   |       *
   |     *
   |   *
   | *
   +---------------> Actual
```

The closer the observations are to the diagonal, the closer the predictions are to the actual values.

---

# 41. Multiple Linear Regression

Simple linear regression uses one feature:

$$
\hat{y}=b_0+b_1x
$$

Multiple linear regression uses several features:

$$
\hat{y}
=
b_0+
b_1x_1+
b_2x_2+
b_3x_3+\cdots
$$

For Auto MPG, an example is:

$$
\widehat{MPG}
=
b_0+
b_1(Weight)+
b_2(Displacement)+
b_3(Cylinders)
$$

The model can therefore use several aspects of the vehicle simultaneously.

---

# 42. Why Use Multiple Features?

A car's fuel efficiency is not determined by one characteristic.

For example:

```text
weight
horsepower
displacement
cylinders
model year
```

may each provide information.

A multiple regression model can combine that information.

This is one reason machine learning is different from examining one pair of variables at a time.

---

# 43. Does Adding More Features Always Improve the Model?

No.

Adding a feature does not automatically make a model better on unseen data.

A feature may:

* contain useful information;
* contain little useful information;
* contain noise;
* duplicate information already present in another feature;
* create interpretation difficulties.

Therefore:

> More features do not automatically mean better predictions.

The model should be evaluated on unseen data.

---

# 44. Simple vs Multiple Linear Regression

The difference can be summarized as:

| Type                       | Features  | Example                                   |
| -------------------------- | --------- | ----------------------------------------- |
| Simple Linear Regression   | 1         | `weight → mpg`                            |
| Multiple Linear Regression | 2 or more | `weight + displacement + cylinders → mpg` |

Both use a linear regression model.

The main difference is the number of predictor variables.

---

# 45. From Linear to More Flexible Models

A straight line is useful, but real relationships are not always perfectly linear.

Suppose the relationship between a feature and a target is curved.

A simple linear model may not represent the pattern well.

One approach is **Polynomial Regression**.

A polynomial model might be:

$$
\hat{y}=b_0+b_1x+b_2x^2
$$

The \(x^2\) term allows the model to represent curvature.

Polynomial Regression is therefore a useful next step after understanding ordinary linear regression.

It is explored more extensively in Activity 2.

---

# 46. Model Complexity

The idea of model complexity is important.

A simple linear model is relatively limited:

$$
\hat{y}=b_0+b_1x
$$

A polynomial model with many terms can be much more flexible:

$$
\hat{y}
=
b_0+b_1x+b_2x^2+\cdots+b_kx^k
$$

Greater flexibility can help the model capture complicated patterns.

But too much flexibility can create a problem:

> The model may begin fitting noise rather than the underlying relationship.

This leads to the concepts of underfitting and overfitting.

---

# 47. Underfitting

Underfitting occurs when a model is too simple to capture important patterns in the data.

A typical pattern is:

```text
Training error: high
Test error:     high
```

The model performs poorly even on the training data.

For example, if the true relationship is strongly curved but we force it into a straight line, the model may underfit.

---

# 48. Overfitting

Overfitting occurs when the model learns the training data too closely.

A typical pattern is:

```text
Training error: very low
Test error:     much higher
```

The model may have learned:

```text
useful pattern + noise
```

instead of:

```text
useful pattern
```

It performs well on examples it has already seen but poorly on new observations.

---

# 49. Generalization

The real goal of machine learning is **generalization**.

A model generalizes well when it performs reasonably well on new data.

The goal is therefore not:

> Minimize training error at any cost.

The goal is:

> **Learn useful patterns that also work on unseen observations.**

This is why the test set is so important.

---

# 50. Data Leakage

When preparing machine-learning data, we must also be careful not to let information from the test set influence training.

This is called **data leakage**.

For example, if preprocessing requires learning parameters from the data, we should learn those parameters from the training data only.

The basic principle is:

```text
Training data
    ↓
fit preprocessing
    ↓
transform training data
    ↓
transform test data
```

The test data should not be used to fit the preprocessing operation.

This becomes particularly important when scaling features or using other learned transformations.

---

# 51. Why Learn the Individual Steps Before `Pipeline`?

A scikit-learn `Pipeline` can combine several operations.

For example:

```python
Pipeline([
    ("preprocessing", ...),
    ("model", ...)
])
```

This is useful because it helps keep preprocessing and prediction steps together.

However, it can hide some of the individual operations.

For learning purposes, it is often better to first understand:

```text
1. Split data
2. Fit preprocessing
3. Transform data
4. Create model
5. Fit model
6. Predict
7. Evaluate
```

Then a pipeline becomes a way of organizing steps that are already understood.

---

# 52. The Complete Introductory Regression Workflow

The concepts in Activity 1 can be summarized as:

```text
Dataset
   ↓
Identify features and target
   ↓
Explore data
   ↓
Visualize relationships
   ↓
Calculate Pearson correlation
   ↓
Select features
   ↓
Split training/test data
   ↓
Create regression model
   ↓
Fit model
   ↓
Make predictions
   ↓
Calculate evaluation metrics
   ↓
Interpret results
```

This workflow is more important than memorizing individual commands.

The code is implementing these conceptual steps.

---

# 53. Connecting the Workflow to Python

For example:

### Select features and target

```python
X = df[["weight"]]
y = df["mpg"]
```

Concept:

> Define what information the model will use and what it will predict.

---

### Split the data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Concept:

> Separate data used for learning from data used for evaluation.

---

### Create the model

```python
model = LinearRegression()
```

Concept:

> Choose the regression algorithm.

---

### Train the model

```python
model.fit(X_train, y_train)
```

Concept:

> Learn the model parameters from training examples.

---

### Predict

```python
y_pred = model.predict(X_test)
```

Concept:

> Generate predictions for new observations.

---

### Evaluate

```python
mae = mean_absolute_error(y_test, y_pred)
```

Concept:

> Measure prediction error on unseen data.

---

# 54. Important Distinction: Correlation vs Regression

These concepts should not be confused.

| Correlation                               | Regression                            |
| ----------------------------------------- | ------------------------------------- |
| Describes association                     | Builds a predictive model             |
| Usually considers two variables at a time | Can use one or many features          |
| Measures linear association               | Estimates model parameters            |
| Produces `r`                              | Produces coefficients and predictions |
| Does not by itself create predictions     | Designed to make predictions          |

For example:

```text
Pearson:
weight ↔ mpg

Regression:
weight → predicted mpg
```

Correlation helps us understand the data.

Regression uses the data to create a model.

---

# 55. Important Distinction: Correlation vs Causation

Another important distinction is:

$$
Correlation \neq Causation
$$

A strong correlation tells us that variables are associated.

It does not establish that changing one variable will necessarily cause a change in the other.

Machine learning is primarily concerned with prediction.

A predictive model can be useful even when it does not establish causal relationships.

---

# 56. Important Distinction: Training Performance vs Generalization

A model can fit training data very well.

That is not enough.

Consider:

```text
Model A
Training RMSE = 3
Test RMSE     = 4
```

versus:

```text
Model B
Training RMSE = 1
Test RMSE     = 15
```

Model B has better training performance.

But Model A has much better test performance.

This illustrates why a model must be evaluated on unseen data.

---

# 57. What Should Be Remembered from Activity 1?

The main concepts are:

### Regression

Regression predicts numerical values.

### Features and target

`X` contains input features.

`y` contains the target.

### Pearson correlation

Measures the strength and direction of a **linear** relationship.

### Linear regression

Models a numerical target using a linear combination of features.

### `fit()`

Learns model parameters from training data.

### `predict()`

Produces predictions for new observations.

### Train/test split

Allows evaluation on unseen data.

### MAE

Measures average absolute prediction error.

### RMSE

Measures prediction error and gives larger errors greater influence.

### R²

Provides a relative measure of how much target variation is explained by the model.

### Multiple regression

Uses several features to predict a target.

### Generalization

A good model should perform well on new data, not only on the data used for training.

---

# 58. Looking Ahead

Activity 1 introduces the fundamental regression workflow.

The next stages of the course can build on these concepts.

Important topics include:

* Polynomial Regression;
* model complexity;
* underfitting;
* overfitting;
* feature scaling;
* regularization;
* other regression algorithms;
* model comparison;
* pipelines;
* more advanced regression diagnostics.

The key idea is to build these concepts gradually.

The fundamental question remains:

> **Can the model learn a useful relationship from the available features and use that relationship to make good predictions for observations it has not seen before?**
