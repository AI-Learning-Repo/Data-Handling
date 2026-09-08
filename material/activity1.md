# Activity 1 — Introduction to Regression with Auto MPG

## Learning Objectives

By the end of this activity, you should be able to:

* explain what a regression problem is;
* identify features and a target variable;
* describe the Auto MPG dataset;
* explain the meaning of MPG;
* use a scatter plot to investigate a relationship between two variables;
* calculate and interpret Pearson Correlation Coefficient (`r`);
* build a simple linear regression model;
* interpret a regression coefficient and intercept;
* make predictions using a regression model;
* understand the difference between training data and test data;
* evaluate a regression model using MAE, RMSE, and R²;
* explain the basic idea of multiple linear regression.

---

# 1. What is Regression?

Machine learning problems can involve different types of predictions.

For example:

```text
Will this customer cancel the subscription?
        ↓
Classification
        ↓
Yes / No
```

Another problem might be:

```text
What will this house sell for?
        ↓
Regression
        ↓
Numerical value
```

Regression is used when the target is a **numerical quantity**.

Examples include:

* predicting house prices;
* predicting temperature;
* predicting electricity consumption;
* predicting salary;
* predicting taxi fare;
* predicting fuel efficiency.

In this activity, the goal is to predict the fuel efficiency of a car.

We will use the **Auto MPG dataset**.

---

# 2. The Auto MPG Dataset

The Auto MPG dataset is a well-known regression dataset originally collected for studying automobile fuel consumption.

The UCI Machine Learning Repository describes the dataset as containing **398 observations** and **7 features**, with `mpg` as the continuous target. The dataset contains automobile characteristics such as cylinders, displacement, horsepower, weight, acceleration, model year, and origin. `horsepower` contains missing values in the dataset.

The important variables for this activity are:

| Variable       | Meaning                            |
| -------------- | ---------------------------------- |
| `mpg`          | Miles per gallon — fuel efficiency |
| `cylinders`    | Number of engine cylinders         |
| `displacement` | Engine displacement                |
| `horsepower`   | Engine power                       |
| `weight`       | Vehicle weight                     |
| `acceleration` | Vehicle acceleration               |
| `model_year`   | Model year                         |
| `origin`       | Country/region of origin           |

The variable we want to predict is:

```text
mpg
```

Therefore:

```text
Target:
mpg
```

Possible input features include:

```text
weight
horsepower
displacement
cylinders
...
```

The dataset also contains a `car_name` identifier. We will not use it as a numerical predictor in this introductory activity.

---

# 3. What Does MPG Mean?

Students in countries that normally use the metric system may be less familiar with MPG.

**MPG** means:

> **miles per gallon**

It measures how far a vehicle can travel using one gallon of fuel.

For example:

```text
30 MPG
```

means approximately:

> The vehicle can travel 30 miles using one US gallon of fuel.

Higher MPG generally means **better fuel efficiency**.

For example:

```text
20 MPG
30 MPG
40 MPG
```

A 40 MPG vehicle is more fuel-efficient than a 20 MPG vehicle.

---

## 3.1 MPG versus km/L

Many European countries use:

> **litres per 100 km (L/100 km)**

and some educational examples use:

> **kilometres per litre (km/L)**.

These measures work differently.

### MPG

Higher is generally better:

```text
20 MPG < 40 MPG
```

### km/L

Higher is also better:

```text
8 km/L < 16 km/L
```

### L/100 km

Lower is better:

```text
8 L/100 km < 16 L/100 km
```

---

## 3.2 Converting MPG to km/L

The Auto MPG dataset uses **miles per US gallon**.

The conversion is approximately:

$$
1\ MPG \approx 0.425\ km/L
$$

Therefore:

$$
km/L \approx MPG \times 0.425
$$

For example:

$$
30\ MPG \times 0.425 \approx 12.75\ km/L
$$

So:

```text
30 MPG ≈ 12.75 km/L
```

The conversion works in the other direction as well:

$$
MPG \approx \frac{km/L}{0.425}
$$

For example:

$$
12.75\ km/L \approx 30\ MPG
$$

### Important

The lab will use the original **`mpg`** variable.

There is no need to convert the target to km/L.

The conversion is provided only so that the numerical values are easier to interpret.

---

# 4. Question 1 — Identify the Machine-Learning Problem

Consider the problem:

> We know characteristics of a car and want to predict its fuel efficiency.

Answer:

1. Is this supervised or unsupervised learning?
2. Is it classification or regression?
3. What is the target?
4. Give two possible features.

### LLM Hint Prompt

You can ask an LLM:

> I am studying supervised machine learning. I have a dataset containing car characteristics and a numerical variable called `mpg`. I want to predict `mpg`. Without giving me the final answer immediately, help me determine whether this is classification or regression, what the target is, and what the features could be. Explain your reasoning step by step.

<details>
<summary>Solution</summary>

1. It is **supervised learning** because the dataset contains known target values (`mpg`) that the model can learn from.
2. It is a **regression** problem because `mpg` is a numerical quantity.
3. The target is:

```text
mpg
```

4. Possible features include:

```text
weight
horsepower
displacement
cylinders
```

</details>

---

# 5. Install the Required Libraries

In Google Colab, many common machine-learning libraries are already available. Nevertheless, the installation step is shown separately so that the environment and dependencies are explicit.

Run this cell first.

### Code Cell

```python
!pip install -q ucimlrepo scikit-learn pandas matplotlib
```

The `ucimlrepo` package allows the dataset to be obtained directly from the UCI Machine Learning Repository.

---

# 6. Import the Libraries

### Code Cell

```python
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import numpy as np
```

The main libraries have different purposes.

### pandas

Used for working with tables and datasets.

### matplotlib

Used for visualizations.

### scikit-learn

Used for machine-learning tasks such as:

* splitting data;
* creating models;
* training models;
* making predictions;
* evaluating models.

### ucimlrepo

Used here to retrieve the Auto MPG dataset from UCI.

---

# 7. Load the Auto MPG Dataset

The UCI repository provides Auto MPG as dataset ID `9`.

### Code Cell

```python
auto_mpg = fetch_ucirepo(id=9)

X_full = auto_mpg.data.features
y_full = auto_mpg.data.targets
```

Let's inspect the data.

### Code Cell

```python
X_full.head()
```

And the target:

### Code Cell

```python
y_full.head()
```

You can also inspect the dataset information:

### Code Cell

```python
print(auto_mpg.metadata)
```

And the variables:

### Code Cell

```python
print(auto_mpg.variables)
```

---

# 8. Question 2 — Explore the Dataset

Before building a model, inspect the data.

Use:

```python
X_full.info()
```

and:

```python
X_full.describe()
```

Answer:

1. Approximately how many observations are there?
2. Which variable is the target?
3. Which variables are numerical?
4. Does any feature contain missing values?
5. Which feature would be a reasonable candidate for predicting `mpg`?

### LLM Hint Prompt

> I am exploring the UCI Auto MPG dataset. Help me interpret the output of `info()` and `describe()`. Explain what I should look for when deciding which variables can be used as regression features, and how I can identify missing values. Do not solve the exercise for me; guide me through the reasoning.

<details>
<summary>Solution</summary>

The dataset contains **398 observations** and the target is `mpg`. The dataset includes numerical variables such as cylinders, displacement, horsepower, weight, acceleration, model year, and origin. `horsepower` contains missing values in the original dataset.

For the introductory activity, `weight` is a useful feature because it is numerical and has a natural relationship with fuel efficiency.

</details>

---

# 9. Select a Feature and Target

For the first regression example, we will use:

```text
weight → mpg
```

That means:

```text
X = weight
y = mpg
```

We can create the two variables as follows.

### Code Cell

```python
X = X_full[["weight"]]
y = y_full["mpg"]
```

Notice the double brackets:

```python
X_full[["weight"]]
```

This produces a **DataFrame** containing one feature.

We use:

```python
y_full["mpg"]
```

for the target.

---

# 10. Question 3 — Why Use `weight`?

Before fitting a model, consider the likely relationship.

Answer:

> Do you expect heavier cars to have higher or lower MPG?

Explain your reasoning.

### LLM Hint Prompt

> I am using vehicle weight to predict MPG. Help me reason about whether I should expect a positive or negative relationship between these variables. Explain what "positive relationship" and "negative relationship" mean in this context, but do not give me a final answer until I have attempted it.

<details>
<summary>Solution</summary>

A negative relationship is expected.

As vehicle weight increases, fuel efficiency generally decreases. Therefore, heavier cars tend to have lower MPG.

In statistical language, we would expect a negative association between `weight` and `mpg`.

</details>

---

# 11. Visualise the Relationship

Before calculating a correlation or fitting a model, it is useful to look at the data.

### Code Cell

```python
plt.scatter(X["weight"], y)

plt.xlabel("Vehicle Weight")
plt.ylabel("MPG")
plt.title("Vehicle Weight vs MPG")

plt.show()
```

Look carefully at the graph.

You should see that the points generally move downward from left to right.

This suggests:

```text
higher weight
      ↓
lower MPG
```

The points will not form a perfect straight line.

There will be variation because fuel efficiency depends on more than just vehicle weight.

---

# 12. Question 4 — Read the Scatter Plot

Look at the scatter plot and answer:

1. Does the relationship appear positive or negative?
2. Does it appear approximately linear?
3. Are all points exactly on one line?
4. Why might there be variation around the general trend?

### LLM Hint Prompt

> I have a scatter plot of vehicle weight versus MPG. Help me interpret the pattern. Teach me how to decide whether a relationship looks positive, negative, linear, or nonlinear. Also explain why real-world data usually do not fall exactly on one line.

<details>
<summary>Solution</summary>

1. The relationship appears **negative**.
2. It appears **approximately linear**, although there is substantial variation.
3. No. Real-world observations do not normally lie exactly on one line.
4. Other factors affect fuel efficiency, such as engine characteristics, aerodynamics, transmission, model year, and other vehicle properties.

</details>

---

# 13. Pearson Correlation

The scatter plot gives us a visual impression.

Pearson Correlation Coefficient gives us a numerical summary of the **strength and direction of a linear relationship**.

It is represented by:

$$
r
$$

and its possible values are:

$$
-1\le r\le1
$$

A positive value means a positive linear relationship.

A negative value means a negative linear relationship.

A value near zero means little or no **linear** association.

---

# 14. Calculate the Correlation

We can calculate Pearson correlation directly with pandas.

### Code Cell

```python
correlation = X["weight"].corr(y)

print("Pearson correlation:", correlation)
```

You can also use:

```python
df_temp = pd.DataFrame({
    "weight": X["weight"],
    "mpg": y
})

print(df_temp.corr())
```

The exact value depends on the data representation being used, but the important feature is that the correlation is **strongly negative**.

---

# 15. Question 5 — Interpret Pearson `r`

Suppose the correlation is:

```text
r = -0.83
```

Answer:

1. What does the negative sign mean?
2. What does the magnitude tell you?
3. Does `r = -0.83` prove that weight causes MPG to decrease?
4. Does it mean that every heavy car has lower MPG than every light car?

### LLM Hint Prompt

> Explain how to interpret a Pearson correlation of -0.83. I want to distinguish the meaning of the sign from the meaning of the magnitude, and I also want to understand why correlation does not prove causation. Give me hints rather than simply writing the complete answer.

<details>
<summary>Solution</summary>

1. The negative sign indicates a **negative linear relationship**. Larger vehicle weight tends to be associated with lower MPG.
2. The magnitude, `0.83`, indicates a **strong linear association**.
3. No. Correlation measures association and does not prove causation.
4. No. Correlation describes the overall pattern. Individual observations can still differ substantially.

</details>

---

# 16. A Limitation of Pearson Correlation

Pearson correlation is useful, but it has an important limitation.

It measures **linear association**.

Consider:

$$
y=x^2
$$

For example:

| `x` | `y` |
| --: | --: |
|  -3 |   9 |
|  -2 |   4 |
|  -1 |   1 |
|   0 |   0 |
|   1 |   1 |
|   2 |   4 |
|   3 |   9 |

There is an exact relationship:

$$
y=x^2
$$

but it is curved rather than linear.

For this symmetric example:

$$
r=0
$$

even though `x` completely determines `y`.

This is why:

> A Pearson correlation close to zero does not necessarily mean that there is no relationship.

It may mean:

> There is little or no **linear** relationship.

---

# 17. Question 6 — Pearson Limitation

Consider:

```text
x = [-3, -2, -1, 0, 1, 2, 3]

y = [9, 4, 1, 0, 1, 4, 9]
```

Answer:

1. Is there a relationship?
2. Is it linear?
3. Would Pearson correlation necessarily identify it as a strong relationship?
4. Why?

### LLM Hint Prompt

> I have x values from -3 to 3 and y = x². Explain why a strong mathematical relationship can exist even when Pearson correlation is close to zero. Focus on the difference between linear and nonlinear relationships.

<details>
<summary>Solution</summary>

1. Yes. There is a perfect mathematical relationship:

$$
y=x^2
$$

2. It is nonlinear.
3. No. Pearson correlation measures linear association, and the correlation for this symmetric example is zero.
4. The positive and negative values of `x` produce the same positive `y` values. Their linear effects cancel out.

</details>

---

# 18. From Correlation to Regression

Correlation answers a question such as:

> How strongly are weight and MPG linearly associated?

Regression asks a different question:

> Can we use weight to predict MPG?

A simple linear regression model can be written as:

$$
\hat{y}=b_0+b_1x
$$

For our example:

$$
\widehat{MPG}
=
b_0+b_1(Weight)
$$

where:

* \(b_0\) is the intercept;
* \(b_1\) is the coefficient for weight.

The model learns these values from the data.

---

# 19. Create the Linear Regression Model

Importantly, scikit-learn uses:

```python
LinearRegression()
```

to create the regression estimator.

### Code Cell

```python
model = LinearRegression()
```

At this point, the model has been created, but it has not learned anything from the data yet.

---

# 20. Train the Model

The `fit()` method trains the model.

### Code Cell

```python
model.fit(X, y)
```

The model examines:

```text
X = vehicle weight
y = MPG
```

and learns the regression parameters.

---

# 21. Examine the Model

Now inspect the intercept and coefficient.

### Code Cell

```python
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])
```

The model can now be written conceptually as:

$$
\widehat{MPG}
=
b_0+b_1(Weight)
$$

using the values learned by the algorithm.

---

# 22. Question 7 — Interpret the Coefficient

Suppose the coefficient is negative.

Answer:

1. What does the negative sign mean?
2. What does the magnitude of the coefficient mean?
3. Why should the coefficient be interpreted in terms of the units of the feature?
4. Does a negative coefficient mean the model is wrong?

### LLM Hint Prompt

> I have a linear regression model predicting MPG from vehicle weight, and the weight coefficient is negative. Help me understand how to interpret the sign and magnitude of the coefficient in the original units of weight. Also explain why a negative coefficient is not automatically a problem.

<details>
<summary>Solution</summary>

1. The negative sign means that increasing vehicle weight is associated with decreasing predicted MPG.
2. The magnitude describes how much the model's predicted MPG changes for a one-unit increase in weight.
3. Regression coefficients depend on the units used by the features. Changing from pounds to kilograms, for example, would change the numerical value of the coefficient.
4. No. A negative coefficient makes sense because heavier vehicles generally have lower fuel efficiency in this dataset.

</details>

---

# 23. Understanding the Intercept

The intercept is:

$$
b_0
$$

It represents the model's predicted target when all features are zero.

In our simple model:

$$
\widehat{MPG}
=
b_0+b_1(Weight)
$$

the intercept is the predicted MPG when:

$$
Weight=0
$$

However, a vehicle with zero weight is not realistic.

Therefore, the intercept may be mathematically useful without having a meaningful real-world interpretation.

This is an important distinction:

> A model parameter can be mathematically necessary even when its value does not correspond to a realistic observation.

---

# 24. Draw the Regression Line

We can use the trained model to generate predictions for the data.

### Code Cell

```python
predictions = model.predict(X)
```

Now plot the original observations and the fitted line.

### Code Cell

```python
plt.scatter(X["weight"], y, label="Observed data")
plt.plot(X["weight"], predictions, label="Regression line")

plt.xlabel("Vehicle Weight")
plt.ylabel("MPG")
plt.title("Linear Regression: Weight vs MPG")

plt.legend()
plt.show()
```

The points represent the observed cars.

The line represents the model's predictions.

The model is trying to capture the general trend using a straight line.

---

# 25. Question 8 — Understand the Regression Line

Look at the plot.

Answer:

1. What do the individual points represent?
2. What does the line represent?
3. Why does the line not pass through every point?
4. Why can the line still be useful even though it does not perfectly describe every car?

### LLM Hint Prompt

> Explain how to interpret a scatter plot with a fitted regression line. I want to understand why the line does not need to pass through every observation and how a regression model can still be useful when individual predictions are not perfect.

<details>
<summary>Solution</summary>

1. Each point represents one car in the dataset.
2. The line represents the MPG predicted by the linear regression model for different vehicle weights.
3. Real-world data contain variation. Vehicle weight is not the only factor affecting fuel efficiency.
4. The regression line summarizes the overall relationship and can be used to make predictions for new observations.

</details>

---

# 26. Make a Prediction for a New Car

Suppose we want to estimate MPG for a car weighing:

```text
3000
```

in the dataset's weight units.

Create a new input:

### Code Cell

```python
new_car = [[3000]]

prediction = model.predict(new_car)

print("Predicted MPG:", prediction[0])
```

The result is the model's estimated fuel efficiency.

---

# 27. Convert the Prediction to km/L

The model predicts MPG because the target is `mpg`.

For European interpretation, we can convert the predicted value.

### Code Cell

```python
predicted_mpg = prediction[0]

predicted_km_per_liter = predicted_mpg * 0.425

print("Predicted MPG:", predicted_mpg)
print("Approximate km/L:", predicted_km_per_liter)
```

Remember:

$$
km/L\approx MPG\times0.425
$$

The machine-learning model itself is still predicting the original `mpg` target.

The conversion is only for interpretation.

---

# 28. Question 9 — Prediction

Suppose the model predicts:

```text
30 MPG
```

Answer:

1. What does this prediction mean?
2. Approximately how many km/L is this?
3. Does this mean that every 3000-unit-weight car gets exactly 30 MPG?
4. Is the prediction guaranteed to be correct?

### LLM Hint Prompt

> A regression model predicts 30 MPG for a car. Help me interpret this prediction and convert it approximately to km/L. Also explain why a machine-learning prediction is an estimate rather than a guarantee.

<details>
<summary>Solution</summary>

1. The model estimates that a car with the given weight has fuel efficiency of approximately 30 miles per US gallon.
2. Approximately:

$$
30\times0.425\approx12.75\ km/L
$$

3. No. The model provides an estimate based on the learned relationship.
4. No. Real observations contain variation and the model does not know every factor affecting the actual fuel efficiency.

</details>

---

# 29. Why Do We Need Training and Test Data?

So far, we trained the model using all available observations.

That is useful for demonstrating how regression works, but it creates a problem if we want to evaluate the model.

The model has already seen the data.

We therefore split the data into two parts:

```text
                     Dataset
                        |
              --------------------
              |                  |
          Training             Test
             80%               20%
              |                  |
          Learn model       Evaluate model
```

The purpose of the test set is to estimate how well the trained model works on **unseen observations**.

---

# 30. Create the Train/Test Split

### Code Cell

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Here:

```text
X_train
```

contains the training feature values.

```text
y_train
```

contains their known MPG values.

```text
X_test
```

contains the test feature values.

```text
y_test
```

contains the true MPG values for those test observations.

---

# 31. Question 10 — Understand the Split

Answer:

1. What percentage of the observations is approximately used for testing?
2. Which data should be used to train the model?
3. Which data should be used for final evaluation?
4. Why should the test data not be used for training?
5. What is the purpose of `random_state=42`?

### LLM Hint Prompt

> Explain this scikit-learn code:
>
> `train_test_split(X, y, test_size=0.2, random_state=42)`
>
> Help me understand the roles of X_train, X_test, y_train, and y_test and why the test data should remain unseen during training.

<details>
<summary>Solution</summary>

1. Approximately 20% is used for testing.
2. `X_train` and `y_train`.
3. `X_test` and `y_test`.
4. The test data should represent unseen observations. If it is used during training, it is no longer an independent test of generalization.
5. `random_state=42` makes the randomized split reproducible.

</details>

---

# 32. Train a Model Using Only the Training Data

Create a new model.

### Code Cell

```python
model = LinearRegression()

model.fit(X_train, y_train)
```

Notice that the model is now fitted using:

```text
X_train
y_train
```

not:

```text
X_test
y_test
```

---

# 33. Predict the Test Data

Now make predictions for observations the model did not use during training.

### Code Cell

```python
y_pred = model.predict(X_test)
```

We now have:

```text
y_test
```

The actual MPG values.

and:

```text
y_pred
```

The predicted MPG values.

These can be compared.

---

# 34. Evaluate the Model with MAE

Mean Absolute Error is:

$$
MAE=
\frac{1}{n}\sum|y_i-\hat{y}_i|
$$

It tells us the average absolute distance between the predicted values and the actual values.

### Code Cell

```python
mae = mean_absolute_error(y_test, y_pred)

print("MAE:", mae)
```

For example, if:

```text
MAE = 2.5
```

the typical absolute prediction error is about 2.5 MPG.

---

# 35. Evaluate with RMSE

RMSE is:

$$
RMSE=
\sqrt{
\frac{1}{n}
\sum(y_i-\hat{y}_i)^2
}
$$

It gives larger errors more influence because the errors are squared before averaging.

### Code Cell

```python
rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

print("RMSE:", rmse)
```

RMSE is expressed in the same units as the target.

In this activity, that means MPG.

---

# 36. Evaluate with R²

R² provides another perspective on model performance.

### Code Cell

```python
r2 = r2_score(y_test, y_pred)

print("R²:", r2)
```

R² can be interpreted as a measure of how much variation in the target is explained by the model relative to the reference baseline represented by the metric.

It is important not to interpret R² as "accuracy".

Regression does not use classification accuracy.

---

# 37. Question 11 — Interpret the Metrics

Suppose the model produces:

```text
MAE  = 2.4
RMSE = 3.3
R²   = 0.65
```

Answer:

1. What does MAE tell us?
2. What does RMSE tell us?
3. Why is RMSE greater than MAE in this example?
4. What does R² tell us?
5. Why should we not call R² "65% accuracy"?

### LLM Hint Prompt

> I have three regression metrics: MAE = 2.4, RMSE = 3.3, and R² = 0.65. Help me interpret each metric in plain language. Also explain why R² should not be described as classification accuracy.

<details>
<summary>Solution</summary>

1. MAE indicates that the average absolute prediction error is about 2.4 MPG.
2. RMSE summarizes prediction error while giving larger errors greater influence.
3. RMSE is often larger than MAE because squaring the errors gives more weight to large errors.
4. R² indicates how well the model explains variation in the target relative to the reference baseline.
5. Accuracy is a classification metric. R² is a regression metric and does not mean that 65% of predictions are "correct."

</details>

---

# 38. Actual vs Predicted Values

A useful visualization compares:

```text
actual MPG
```

with:

```text
predicted MPG
```

### Code Cell

```python
plt.scatter(y_test, y_pred)

plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.title("Actual vs Predicted MPG")

plt.show()
```

A model that predicts perfectly would have:

$$
Predicted=Actual
$$

for every observation.

The points would therefore lie along a diagonal line.

---

# 39. Add a Reference Diagonal

We can add a reference line representing perfect predictions.

### Code Cell

```python
plt.scatter(y_test, y_pred)

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.title("Actual vs Predicted MPG")

plt.show()
```

Points closer to the diagonal generally represent more accurate predictions.

---

# 40. Question 12 — Interpret the Prediction Plot

Answer:

1. What would a point exactly on the diagonal mean?
2. What does a point far from the diagonal mean?
3. Would you expect all points to lie exactly on the diagonal?
4. What does a cluster of points around the diagonal suggest?

### LLM Hint Prompt

> Explain how to interpret an actual-versus-predicted regression plot. Focus on what the diagonal line represents, what points far from it mean, and what a good model would generally look like.

<details>
<summary>Solution</summary>

1. A point on the diagonal means the prediction equals the actual value.
2. A point far from the diagonal represents a larger prediction error.
3. No. Real-world predictions are rarely perfect.
4. A cluster of points close to the diagonal suggests that the predictions are generally close to the actual values.

</details>

---

# 41. Multiple Linear Regression

So far, the model has used one feature:

```text
weight
```

This is called **simple linear regression**.

However, real-world problems usually involve more than one feature.

For example, car fuel efficiency may depend on:

```text
weight
displacement
cylinders
horsepower
model_year
```

We can therefore use several features.

A multiple linear regression model can be written as:

$$
\hat{y}
=
b_0+
b_1x_1+
b_2x_2+
b_3x_3+\cdots
$$

For example:

$$
\widehat{MPG}
=
b_0+
b_1(Weight)+
b_2(Displacement)+
b_3(Cylinders)
$$

---

# 42. Select Multiple Features

To avoid introducing missing-value handling at this stage, use features that are complete in the UCI Auto MPG dataset.

For example:

```python
cylinders
displacement
weight
```

The UCI documentation indicates that these features do not have missing values, while `horsepower` does.

### Code Cell

```python
X = X_full[
    [
        "cylinders",
        "displacement",
        "weight"
    ]
]

y = y_full["mpg"]
```

---

# 43. Split the Multiple-Feature Data

### Code Cell

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

# 44. Train Multiple Linear Regression

### Code Cell

```python
model = LinearRegression()

model.fit(X_train, y_train)
```

Now make predictions:

### Code Cell

```python
y_pred = model.predict(X_test)
```

And evaluate:

### Code Cell

```python
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)
```

---

# 45. Examine the Multiple Regression Coefficients

We can inspect the coefficients:

### Code Cell

```python
print("Intercept:", model.intercept_)

for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, ":", coefficient)
```

This tells us how the model uses the different features.

For example, the model conceptually has the form:

$$
\widehat{MPG}
=
b_0+
b_1(Cylinders)+
b_2(Displacement)+
b_3(Weight)
$$

---

# 46. Important Idea: Interpreting Multiple Regression Coefficients

Suppose a coefficient for `weight` is negative.

A simplified interpretation is:

> Holding the other model features constant, an increase in vehicle weight is associated with a decrease in predicted MPG.

The phrase:

> **holding the other features constant**

is important.

In multiple regression, each coefficient represents the feature's contribution while accounting for the other features included in the model.

This is different from simply looking at the correlation between one feature and the target.

---

# 47. Question 13 — Simple vs Multiple Linear Regression

Complete the following table:

| Model                      | Number of Features | Example |
| -------------------------- | -----------------: | ------- |
| Simple Linear Regression   |                  ? | ?       |
| Multiple Linear Regression |                  ? | ?       |

Then answer:

1. Why might multiple regression perform better than simple regression?
2. Does adding more features always improve the model?
3. Why might some features be unhelpful?

### LLM Hint Prompt

> Explain the difference between simple and multiple linear regression. Help me understand why multiple features can provide more information but also why adding every available feature does not automatically produce a better model.

<details>
<summary>Solution</summary>

| Model                      | Number of Features | Example                                   |
| -------------------------- | -----------------: | ----------------------------------------- |
| Simple Linear Regression   |                One | `weight → mpg`                            |
| Multiple Linear Regression |        Two or more | `weight + displacement + cylinders → mpg` |

1. Multiple regression can use information from several characteristics of the car.
2. No. Additional features can be irrelevant, noisy, redundant, or introduce other modeling problems.
3. Some features may contain little useful information for the target or may overlap strongly with other features.

</details>

---

# 48. Compare the One-Feature and Multiple-Feature Models

The previous experiments allow us to compare:

```text
Model 1:
weight → mpg
```

with:

```text
Model 2:
weight + displacement + cylinders → mpg
```

Create a small comparison table.

### Code Cell

```python
results = pd.DataFrame({
    "Model": [
        "Simple Linear Regression",
        "Multiple Linear Regression"
    ],
    "MAE": [
        mae_simple,
        mae
    ],
    "RMSE": [
        rmse_simple,
        rmse
    ],
    "R2": [
        r2_simple,
        r2
    ]
})

results
```

Before running this, we need to save the results from the simple model.

For example, after evaluating the simple model:

```python
mae_simple = mean_absolute_error(y_test, y_pred)
```

However, because we later replaced `X` and `model` with the multiple-feature version, it is cleaner to repeat the simple regression experiment in one block.

### Code Cell

```python
# Simple regression
X_simple = X_full[["weight"]]
y_simple = y_full["mpg"]

X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
    X_simple,
    y_simple,
    test_size=0.2,
    random_state=42
)

simple_model = LinearRegression()
simple_model.fit(X_train_simple, y_train_simple)

y_pred_simple = simple_model.predict(X_test_simple)

mae_simple = mean_absolute_error(y_test_simple, y_pred_simple)

rmse_simple = np.sqrt(
    mean_squared_error(y_test_simple, y_pred_simple)
)

r2_simple = r2_score(y_test_simple, y_pred_simple)

print("Simple Regression")
print("MAE:", mae_simple)
print("RMSE:", rmse_simple)
print("R²:", r2_simple)
```

Now train the multiple regression model again.

### Code Cell

```python
X_multiple = X_full[
    [
        "cylinders",
        "displacement",
        "weight"
    ]
]

y_multiple = y_full["mpg"]

X_train_multiple, X_test_multiple, y_train_multiple, y_test_multiple = train_test_split(
    X_multiple,
    y_multiple,
    test_size=0.2,
    random_state=42
)

multiple_model = LinearRegression()

multiple_model.fit(
    X_train_multiple,
    y_train_multiple
)

y_pred_multiple = multiple_model.predict(X_test_multiple)

mae_multiple = mean_absolute_error(
    y_test_multiple,
    y_pred_multiple
)

rmse_multiple = np.sqrt(
    mean_squared_error(
        y_test_multiple,
        y_pred_multiple
    )
)

r2_multiple = r2_score(
    y_test_multiple,
    y_pred_multiple
)

print("Multiple Regression")
print("MAE:", mae_multiple)
print("RMSE:", rmse_multiple)
print("R²:", r2_multiple)
```

Now create the comparison.

### Code Cell

```python
results = pd.DataFrame({
    "Model": [
        "Simple Linear Regression",
        "Multiple Linear Regression"
    ],
    "MAE": [
        mae_simple,
        mae_multiple
    ],
    "RMSE": [
        rmse_simple,
        rmse_multiple
    ],
    "R²": [
        r2_simple,
        r2_multiple
    ]
})

results
```

---

# 49. Question 14 — Compare the Models

Use the results table.

Answer:

1. Which model has the lower MAE?
2. Which model has the lower RMSE?
3. Which model has the higher R²?
4. Does the multiple-feature model perform better?
5. Is a model automatically better simply because it has more features?

### LLM Hint Prompt

> I have results for a simple regression model and a multiple regression model. Help me compare them using MAE, RMSE, and R². Explain why lower MAE/RMSE is generally better and why a higher R² is generally better, while reminding me that model comparison should be based on unseen test data.

<details>
<summary>Solution</summary>

1. The model with the lower MAE has smaller average absolute prediction error.
2. The model with the lower RMSE has lower prediction error while giving greater influence to large errors.
3. The model with the higher R² explains more variation relative to the metric's reference baseline.
4. The test results determine whether the additional features improved generalization.
5. No. Adding features does not automatically guarantee better performance on unseen data.

</details>

---

# 50. A First Look at Overfitting and Underfitting

A model should not simply memorize the training observations.

The goal is to perform well on data it has not seen.

This gives us two important concepts.

## Underfitting

The model is too simple to capture important patterns.

Typical pattern:

```text
Training error: high
Test error:     high
```

## Overfitting

The model fits the training data extremely well but performs poorly on unseen data.

Typical pattern:

```text
Training error: very low
Test error:     much higher
```

We will study these ideas in more detail after this activity.

---

# 51. Compare Training and Test Performance

For the multiple regression model, calculate training predictions.

### Code Cell

```python
train_predictions = multiple_model.predict(
    X_train_multiple
)

test_predictions = multiple_model.predict(
    X_test_multiple
)
```

Calculate both RMSE values.

### Code Cell

```python
train_rmse = np.sqrt(
    mean_squared_error(
        y_train_multiple,
        train_predictions
    )
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test_multiple,
        test_predictions
    )
)

print("Training RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
```

---

# 52. Question 15 — Interpret Training vs Test Error

Suppose a model produces:

```text
Training RMSE = 2.0
Test RMSE     = 8.0
```

Answer:

1. Is the model performing better on training or test data?
2. Is the difference concerning?
3. What machine-learning concept might this indicate?
4. Why is test performance important?

### LLM Hint Prompt

> Explain what it means when a regression model has a much lower training RMSE than test RMSE. Help me connect this pattern to generalization and overfitting without assuming that every difference automatically proves overfitting.

<details>
<summary>Solution</summary>

1. The model performs better on the training data.
2. The large difference is a warning sign that the model may not generalize well.
3. It may indicate **overfitting**.
4. Test performance gives information about how the model performs on observations it did not use during training.

</details>

---

# 53. Why This Activity Uses Only Linear Regression

The purpose of this first activity is to establish the basic regression workflow.

The central process is:

```text
Dataset
   ↓
Features and target
   ↓
Data exploration
   ↓
Scatter plot
   ↓
Correlation
   ↓
Regression model
   ↓
Fit
   ↓
Prediction
   ↓
Evaluation
   ↓
Interpretation
```

At this stage, it is not necessary to introduce every regression algorithm.

Later activities can introduce:

* Polynomial Regression;
* model complexity;
* underfitting and overfitting in more detail;
* regularization;
* other regression algorithms;
* pipelines.

---

# 54. Activity Reflection

Answer the following in your own words.

### Question 16

What is regression?

### LLM Hint Prompt

> Ask me questions that help me explain supervised regression in my own words. Do not give me a definition immediately. Help me improve my explanation if I am missing important ideas.

<details>
<summary>Solution</summary>

Regression is a supervised machine-learning approach used to predict a numerical target from one or more input features.

</details>

---

### Question 17

What is the difference between a feature and a target?

### LLM Hint Prompt

> I am learning the difference between features and target variables in supervised learning. Give me a simple car example and ask me questions that help me identify which variables are inputs and which variable is the prediction target.

<details>
<summary>Solution</summary>

A **feature** is an input variable used by the model to make a prediction.

The **target** is the variable the model is trying to predict.

For example:

```text
weight      → feature
displacement → feature
cylinders    → feature
mpg          → target
```

</details>

---

### Question 18

What does Pearson correlation tell us?

### LLM Hint Prompt

> Explain Pearson correlation through a simple example involving car weight and MPG. Help me distinguish the direction of the correlation from its strength and remind me of its main limitation.

<details>
<summary>Solution</summary>

Pearson correlation describes the **strength and direction of a linear relationship** between two numerical variables.

Its value ranges from:

$$
-1\le r\le1
$$

The sign indicates the direction, while the magnitude indicates the strength of the linear association.

Its important limitation is that it measures linear association. A strong nonlinear relationship can have a Pearson correlation close to zero.

</details>

---

### Question 19

Why is `r = 0` not proof that two variables have no relationship?

### LLM Hint Prompt

> Give me a nonlinear example where Pearson correlation is zero even though one variable depends completely on the other. Explain why the linear contributions cancel.

<details>
<summary>Solution</summary>

For example:

$$
y=x^2
$$

can produce:

$$
r=0
$$

for a symmetric dataset such as:

```text
x = [-3,-2,-1,0,1,2,3]
y = [ 9, 4, 1,0,1,4,9]
```

There is a perfect nonlinear relationship, but there is no linear trend.

</details>

---

### Question 20

What does `fit()` do?

### LLM Hint Prompt

> Explain what `model.fit(X_train, y_train)` means in scikit-learn. Focus on what information the model learns from the training data.

<details>
<summary>Solution</summary>

`fit()` trains the model using the training features and their known target values. For linear regression, it estimates the coefficients and intercept that define the fitted regression model.

</details>

---

### Question 21

What does `predict()` do?

### LLM Hint Prompt

> Explain the difference between `fit()` and `predict()` using a car MPG example. Help me understand what happens before prediction and what happens when a new car is supplied.

<details>
<summary>Solution</summary>

`fit()` learns the model from known training examples.

`predict()` uses the trained model to estimate the target for new observations.

For example:

```python
model.fit(X_train, y_train)
```

learns the relationship.

Then:

```python
model.predict(X_test)
```

produces predicted MPG values for the test cars.

</details>

---

### Question 22

Why do we use a test set?

### LLM Hint Prompt

> Explain why evaluating a regression model on the same observations used for training can be misleading. Use a simple analogy involving a student studying for an exam.

<details>
<summary>Solution</summary>

The test set gives us observations that were not used to fit the model. This helps estimate how well the model generalizes to new data rather than simply measuring how well it fits observations it has already seen.

</details>

---

### Question 23

What is the difference between simple and multiple linear regression?

### LLM Hint Prompt

> Explain simple and multiple linear regression using the Auto MPG dataset. Use `weight` as an example of one feature and `weight + displacement + cylinders` as an example of multiple features.

<details>
<summary>Solution</summary>

Simple linear regression uses one feature:

$$
\widehat{MPG}
=
b_0+b_1(Weight)
$$

Multiple linear regression uses multiple features:

$$
\widehat{MPG}
=
b_0+
b_1(Weight)+
b_2(Displacement)+
b_3(Cylinders)
$$

</details>

---

### Question 24

Why should a model not automatically be judged by its training error?

### LLM Hint Prompt

> I have two regression models. One has a lower training RMSE but a much higher test RMSE. Help me decide which model is more useful for predicting new observations and explain why.

<details>
<summary>Solution</summary>

A low training error alone may indicate that the model fits the training data very well, but it does not tell us whether the model generalizes.

A model with slightly higher training error but much lower test error may be more useful for predicting new observations.

This is related to overfitting.

</details>

---

# 55. Final Knowledge Check

Before finishing Activity 1, answer these questions without looking at the solutions.

### Q1

What type of machine-learning problem is Auto MPG?

### Q2

What is the target variable?

### Q3

What does MPG mean?

### Q4

Approximately how can MPG be converted to km/L?

### Q5

What does Pearson `r` measure?

### Q6

What does a negative Pearson correlation mean?

### Q7

Why can `r` be close to zero even when a strong relationship exists?

### Q8

What is the purpose of `LinearRegression()`?

### Q9

What does `fit()` do?

### Q10

What does `predict()` do?

### Q11

Why do we use a train/test split?

### Q12

What does MAE measure?

### Q13

What does RMSE measure?

### Q14

What does R² tell us?

### Q15

What is the difference between simple and multiple linear regression?

### Q16

What is underfitting?

### Q17

What is overfitting?

### Q18

Why is test performance important?

---

# Final Answers

<details>
<summary>Q1 — What type of machine-learning problem is Auto MPG?</summary>

It is a **supervised regression problem** because the target `mpg` is a numerical value.

</details>

<details>
<summary>Q2 — What is the target variable?</summary>

The target is:

```text
mpg
```

</details>

<details>
<summary>Q3 — What does MPG mean?</summary>

MPG means **miles per gallon** and represents how many miles a vehicle can travel using one US gallon of fuel.

</details>

<details>
<summary>Q4 — How can MPG be converted to km/L?</summary>

Approximately:

$$
km/L \approx MPG\times0.425
$$

For example:

$$
30 MPG\approx12.75\ km/L
$$

</details>

<details>
<summary>Q5 — What does Pearson r measure?</summary>

Pearson `r` measures the **strength and direction of a linear relationship** between two numerical variables.

</details>

<details>
<summary>Q6 — What does a negative Pearson correlation mean?</summary>

It means that larger values of one variable tend to be associated with smaller values of the other variable in a linear relationship.

</details>

<details>
<summary>Q7 — Why can r be close to zero when a relationship exists?</summary>

Because Pearson measures linear association. A strong nonlinear relationship such as:

$$
y=x^2
$$

can have a Pearson correlation close to zero.

</details>

<details>
<summary>Q8 — What is the purpose of LinearRegression()?</summary>

It creates a scikit-learn linear regression model that can be trained to predict a numerical target.

</details>

<details>
<summary>Q9 — What does fit() do?</summary>

It learns the model parameters from the training data.

</details>

<details>
<summary>Q10 — What does predict() do?</summary>

It uses the trained model to produce predictions for new observations.

</details>

<details>
<summary>Q11 — Why use a train/test split?</summary>

To evaluate the model on observations that were not used during training and therefore obtain an estimate of generalization performance.

</details>

<details>
<summary>Q12 — What does MAE measure?</summary>

MAE measures the average absolute prediction error.

</details>

<details>
<summary>Q13 — What does RMSE measure?</summary>

RMSE summarizes prediction error while giving larger errors more influence. It is expressed in the same units as the target.

</details>

<details>
<summary>Q14 — What does R² tell us?</summary>

R² describes how much variation in the target is explained by the model relative to the metric's reference baseline.

</details>

<details>
<summary>Q15 — Simple vs multiple linear regression?</summary>

Simple linear regression uses one feature.

Multiple linear regression uses two or more features.

</details>

<details>
<summary>Q16 — What is underfitting?</summary>

Underfitting occurs when the model is too simple to capture important patterns in the data. Training and test performance are often both poor.

</details>

<details>
<summary>Q17 — What is overfitting?</summary>

Overfitting occurs when the model fits the training data too closely and performs substantially worse on unseen data.

</details>

<details>
<summary>Q18 — Why is test performance important?</summary>

Because the ultimate goal of a machine-learning model is to make useful predictions on new observations, not merely to fit the training data.

</details>

---

# Activity 1 Summary

The main workflow introduced in this activity is:

```text
Auto MPG Dataset
       ↓
Identify Features and Target
       ↓
Explore the Data
       ↓
Scatter Plot
       ↓
Pearson Correlation
       ↓
Simple Linear Regression
       ↓
Fit the Model
       ↓
Make Predictions
       ↓
Evaluate the Model
       ↓
Train/Test Split
       ↓
Multiple Linear Regression
       ↓
Compare Models
```

The most important concepts to take from Activity 1 are:

> **Regression predicts numerical values.**

> **Pearson correlation measures linear association.**

> **A regression model learns parameters from training data.**

> **Predictions are made using the fitted model.**

> **MAE and RMSE measure prediction error.**

> **R² provides another way to assess regression performance.**

> **Multiple regression can use several features.**

> **Test performance is important because the model should generalize to unseen data.**

These concepts will be used throughout Activity 2.
