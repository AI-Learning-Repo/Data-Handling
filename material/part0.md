# Regression Theory: Pearson Correlation, Underfitting, and Overfitting

This theory chapter supports the regression lab and explains two important ideas that appear throughout the exercises:

1. **Pearson Correlation Coefficient (`r`)**
2. **Underfitting and Overfitting**

The explanations begin with small examples that can be calculated by hand and then connect those ideas to practical machine-learning work with Python and scikit-learn.

---

# Part A — Pearson Correlation Coefficient (`r`)

## A.1 What is correlation?

In machine learning, we usually work with variables that describe our observations.

For example, in a taxi-fare prediction problem, we might have:

* `TRIP_MILES`
* `TRIP_MINUTES`
* `PASSENGER_COUNT`

and we may want to predict:

* `FARE`

The variables used to make predictions are called **features**, while the value we want to predict is called the **target**.

Before training a regression model, it is useful to explore the data and ask:

> Is there a relationship between a feature and the target?

One common way to investigate this is the **Pearson Correlation Coefficient**, written as:

$$
r
$$

Pearson correlation measures the **strength and direction of a linear relationship between two numerical variables**.

The value of `r` is always between:

$$
-1 \leq r \leq 1
$$

---

## A.2 Understanding the value of `r`

The sign tells us the **direction** of the relationship.

The magnitude tells us the **strength of the linear relationship**.

|   Pearson `r` | Interpretation                       |
| ------------: | ------------------------------------ |
|          `+1` | Perfect positive linear relationship |
| Close to `+1` | Strong positive linear relationship  |
|    Around `0` | Little or no linear relationship     |
| Close to `-1` | Strong negative linear relationship  |
|          `-1` | Perfect negative linear relationship |

### Positive correlation

Suppose:

```text
x = trip distance
y = taxi fare
```

As trip distance increases, fare often increases.

That would produce a positive correlation.

For example:

$$
r=0.85
$$

would indicate a strong positive linear association.

### Negative correlation

Suppose:

```text
x = age of a car
y = resale value
```

As the car gets older, its resale value may decrease.

That would result in a negative correlation.

For example:

$$
r=-0.90
$$

would indicate a strong negative linear association.

### Correlation near zero

If:

$$
r\approx0
$$

then the variables do not show much **linear association**.

The word **linear** is extremely important.

---

# A.3 What does "linear" mean?

A linear relationship can be represented by a straight line.

For example:

$$
y = 2x+1
$$

This means that when `x` increases by one unit, `y` increases by two units.

For example:

| `x` | `y = 2x + 1` |
| --: | -----------: |
|   1 |            3 |
|   2 |            5 |
|   3 |            7 |
|   4 |            9 |

The points lie exactly on a straight line.

Pearson correlation is very useful for detecting this kind of relationship.

---

# A.4 A small dataset

Let's calculate Pearson correlation manually.

Consider this dataset:

| Observation | `x` | `y` |
| ----------- | --: | --: |
| 1           |   1 |   2 |
| 2           |   2 |   4 |
| 3           |   3 |   6 |
| 4           |   4 |   8 |

We can see that:

$$
y=2x
$$

There is an exact linear relationship.

We expect:

$$
r=1
$$

But let's calculate it ourselves.

---

# A.5 Step 1 — Calculate the mean of `x`

The `x` values are:

$$
1,2,3,4
$$

The mean is:

$$
\bar{x}=
\frac{1+2+3+4}{4}
$$

Therefore:

$$
\bar{x}=
\frac{10}{4}
=2.5
$$

So:

$$
\boxed{\bar{x}=2.5}
$$

---

# A.6 Step 2 — Calculate the mean of `y`

The `y` values are:

$$
2,4,6,8
$$

The mean is:

$$
\bar{y}=
\frac{2+4+6+8}{4}
$$

Therefore:

$$
\bar{y}=
\frac{20}{4}
=5
$$

So:

$$
\boxed{\bar{y}=5}
$$

---

# A.7 Step 3 — Calculate deviations from the mean

For each observation, calculate:

$$
x_i-\bar{x}
$$

and:

$$
y_i-\bar{y}
$$

The resulting table is:

| `x` | `y` | `x - x̄` | `y - ȳ` |
| --: | --: | -------: | ------: |
|   1 |   2 |     -1.5 |      -3 |
|   2 |   4 |     -0.5 |      -1 |
|   3 |   6 |     +0.5 |      +1 |
|   4 |   8 |     +1.5 |      +3 |

For example, for the first row:

$$
x-\bar{x}=1-2.5=-1.5
$$

and:

$$
y-\bar{y}=2-5=-3
$$

---

# A.8 Step 4 — Multiply the deviations

Now calculate:

$$
(x_i-\bar{x})(y_i-\bar{y})
$$

| `x` | `y` | `x - x̄` | `y - ȳ` | Product |
| --: | --: | -------: | ------: | ------: |
|   1 |   2 |     -1.5 |      -3 |     4.5 |
|   2 |   4 |     -0.5 |      -1 |     0.5 |
|   3 |   6 |     +0.5 |      +1 |     0.5 |
|   4 |   8 |     +1.5 |      +3 |     4.5 |

Now add the products:

$$
4.5+0.5+0.5+4.5=10
$$

Thus:

$$
\sum(x_i-\bar{x})(y_i-\bar{y})=10
$$

---

# A.9 Step 5 — Square the deviations of `x`

Calculate:

$$
(x_i-\bar{x})^2
$$

| `x` | `x - x̄` | `(x - x̄)²` |
| --: | -------: | ----------: |
|   1 |     -1.5 |        2.25 |
|   2 |     -0.5 |        0.25 |
|   3 |     +0.5 |        0.25 |
|   4 |     +1.5 |        2.25 |

Add them:

$$
2.25+0.25+0.25+2.25=5
$$

Therefore:

$$
\sum(x_i-\bar{x})^2=5
$$

---

# A.10 Step 6 — Square the deviations of `y`

Calculate:

$$
(y_i-\bar{y})^2
$$

| `y` | `y - ȳ` | `(y - ȳ)²` |
| --: | ------: | ---------: |
|   2 |      -3 |          9 |
|   4 |      -1 |          1 |
|   6 |      +1 |          1 |
|   8 |      +3 |          9 |

Add them:

$$
9+1+1+9=20
$$

Therefore:

$$
\sum(y_i-\bar{y})^2=20
$$

---

# A.11 Step 7 — Apply the Pearson formula

The Pearson correlation coefficient is:

$$
r=
\frac{
\sum(x_i-\bar{x})(y_i-\bar{y})
}{
\sqrt{
\sum(x_i-\bar{x})^2
\sum(y_i-\bar{y})^2
}
}
$$

We calculated:

$$
\sum(x_i-\bar{x})(y_i-\bar{y})=10
$$

$$
\sum(x_i-\bar{x})^2=5
$$

$$
\sum(y_i-\bar{y})^2=20
$$

Substitute:

$$
r=
\frac{10}
{\sqrt{5\times20}}
$$

$$
r=
\frac{10}
{\sqrt{100}}
$$

$$
r=
\frac{10}{10}
$$

Therefore:

$$
\boxed{r=1}
$$

This is a perfect positive linear relationship.

That makes sense because:

$$
y=2x
$$

and every point lies exactly on a straight line.

---

# A.12 Another way to understand the calculation

The calculation becomes more intuitive when we think about the deviations.

For observations below the mean:

* `x - x̄` is negative
* `y - ȳ` is negative

Multiplying them gives a positive result:

$$
(-)(-)=(+)
$$

For observations above the mean:

* `x - x̄` is positive
* `y - ȳ` is positive

Again:

$$
(+)(+)=(+)
$$

Therefore the products are consistently positive.

This indicates that `x` and `y` tend to move in the same direction.

---

# A.13 What would negative correlation look like?

Consider:

| `x` | `y` |
| --: | --: |
|   1 |   8 |
|   2 |   6 |
|   3 |   4 |
|   4 |   2 |

Here:

$$
y=10-2x
$$

As `x` increases, `y` decreases.

All the points lie on a straight line with a negative slope.

Therefore:

$$
\boxed{r=-1}
$$

This is a perfect negative linear relationship.

---

# A.14 Important limitation of Pearson correlation

Pearson correlation measures **linear** association.

This means that a value of `r` close to zero does not necessarily mean that the two variables are unrelated.

They may have a strong **nonlinear** relationship.

This is one of the most important limitations of Pearson correlation.

---

# A.15 Example of a nonlinear relationship

Consider:

$$
y=x^2
$$

Use this dataset:

| `x` | `y = x²` |
| --: | -------: |
|  -3 |        9 |
|  -2 |        4 |
|  -1 |        1 |
|   0 |        0 |
|   1 |        1 |
|   2 |        4 |
|   3 |        9 |

There is obviously a relationship:

$$
y=x^2
$$

The value of `y` is completely determined by `x`.

But the relationship is curved.

It forms a U-shaped pattern rather than a straight line.

---

# A.16 Pearson correlation for `y = x²`

The mean of `x` is:

$$
\bar{x}=0
$$

The mean of `y` is:

$$
\bar{y}=
\frac{9+4+1+0+1+4+9}{7}
=
\frac{28}{7}
=4
$$

Now consider:

$$
\sum(x_i-\bar{x})(y_i-\bar{y})
$$

Because `x̄ = 0`, this becomes:

$$
\sum x_i(y_i-4)
$$

The terms are:

$$
(-3)(5)=-15
$$

$$
(-2)(0)=0
$$

$$
(-1)(-3)=3
$$

$$
(0)(-4)=0
$$

$$
(1)(-3)=-3
$$

$$
(2)(0)=0
$$

$$
(3)(5)=15
$$

Adding them:

$$
-15+0+3+0-3+0+15=0
$$

Therefore the numerator is zero:

$$
r=0
$$

So:

$$
\boxed{r=0}
$$

---

# A.17 Why is this surprising?

The variables are clearly related.

In fact:

$$
y=x^2
$$

is a perfect mathematical relationship.

But Pearson gives:

$$
r=0
$$

Why?

Because Pearson is looking for a **linear trend**.

The positive and negative values of `x` cancel each other.

For example:

$$
(-3)^2=9
$$

and:

$$
(3)^2=9
$$

Both sides of the curve produce the same `y`.

There is a strong relationship, but it is not linear.

Therefore:

> **Pearson correlation near zero does not necessarily mean "no relationship." It can mean "no linear relationship."**

---

# A.18 Important clarification: `y = 2x` is linear

It is important not to confuse multiplication by a constant with nonlinearity.

The relationship:

$$
y=2x
$$

is linear.

So Pearson correlation is appropriate.

For example:

```text
x:  1   2   3   4
y:  2   4   6   8
```

gives:

$$
r=1
$$

Similarly:

$$
y=2x+5
$$

is also linear.

The important difference is:

```text
y = 2x       → linear
y = 2x + 5   → linear
y = x²       → nonlinear
y = x³       → nonlinear relationship, although monotonic over all real x
```

The problem with Pearson is not the number `2`.

The issue is whether the relationship between the variables is approximately linear.

---

# A.19 Why do we need to check correlation for several features?

A machine-learning dataset often has more than one feature.

For example, suppose we want to predict taxi fare.

Our data might contain:

| Trip | Miles | Minutes | Passengers | Fare |
| ---- | ----: | ------: | ---------: | ---: |
| 1    |     1 |      10 |          1 |    8 |
| 2    |     2 |      15 |          1 |   12 |
| 3    |     3 |      20 |          2 |   16 |
| 4    |     4 |      25 |          2 |   20 |

The target is:

$$
y=\text{Fare}
$$

The features are:

$$
X_1=\text{Miles}
$$

$$
X_2=\text{Minutes}
$$

$$
X_3=\text{Passengers}
$$

We can investigate each feature separately.

---

# A.20 Correlation between Miles and Fare

Take:

```text
Miles = 1, 2, 3, 4
Fare  = 8, 12, 16, 20
```

These values satisfy:

$$
Fare=4(Miles)+4
$$

Therefore, this is a perfect positive linear relationship.

So:

$$
\boxed{
r_{\text{Miles,Fare}}=1
}
$$

---

# A.21 Correlation between Minutes and Fare

Now examine:

```text
Minutes = 10, 15, 20, 25
Fare     = 8, 12, 16, 20
```

These values satisfy:

$$
Fare=0.8(Minutes)
$$

Therefore:

$$
\boxed{
r_{\text{Minutes,Fare}}=1
}
$$

Again, perfect positive linear correlation in this small example.

---

# A.22 Correlation between Passengers and Fare

Now consider:

```text
Passengers = 1, 1, 2, 2
Fare       = 8, 12, 16, 20
```

The mean number of passengers is:

$$
\bar{x}=
\frac{1+1+2+2}{4}
=
1.5
$$

The mean fare is:

$$
\bar{y}=
\frac{8+12+16+20}{4}
=
14
$$

Calculate the deviations:

| Passengers | Fare | `x - x̄` | `y - ȳ` |
| ---------: | ---: | -------: | ------: |
|          1 |    8 |     -0.5 |      -6 |
|          1 |   12 |     -0.5 |      -2 |
|          2 |   16 |     +0.5 |      +2 |
|          2 |   20 |     +0.5 |      +6 |

Multiply:

| Passengers | Fare | `x - x̄` | `y - ȳ` | Product |
| ---------: | ---: | -------: | ------: | ------: |
|          1 |    8 |     -0.5 |      -6 |       3 |
|          1 |   12 |     -0.5 |      -2 |       1 |
|          2 |   16 |     +0.5 |      +2 |       1 |
|          2 |   20 |     +0.5 |      +6 |       3 |

The numerator is:

$$
3+1+1+3=8
$$

The squared deviations of `x` sum to:

$$
0.25+0.25+0.25+0.25=1
$$

The squared deviations of `y` sum to:

$$
36+4+4+36=80
$$

Therefore:

$$
r=
\frac{8}{\sqrt{1\times80}}
$$

$$
r\approx0.894
$$

So:

$$
\boxed{r\approx0.894}
$$

This is a strong positive linear relationship, although it is not perfect.

---

# A.23 What have we learned from multiple features?

We now have:

| Feature    | Correlation with Fare |
| ---------- | --------------------: |
| Miles      |               `1.000` |
| Minutes    |               `1.000` |
| Passengers |               `0.894` |

This gives us an initial picture of the data.

We can repeat the same idea with a larger dataset.

For example, if we have:

```text
TRIP_MILES
TRIP_MINUTES
PASSENGER_COUNT
HOUR
FARE
```

we can calculate:

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

and so on.

---

# A.24 Does this mean we should always choose the highest-correlated features?

No.

Correlation is useful for **exploration**, but it is not a complete feature-selection strategy.

There are several reasons.

### A feature can have a nonlinear relationship

For example:

$$
y=x^2
$$

can have Pearson correlation close to zero even though the relationship is very strong.

### A feature may be useful together with other features

A feature might have weak individual correlation but still provide useful information once combined with other features.

### Correlation does not imply causation

A high correlation does not prove that one variable causes another.

### Correlation can be affected by outliers

A few extreme observations can substantially affect the correlation.

Therefore:

> Pearson correlation is a useful first step in understanding the data, but it should not be the only criterion used to select features.

---

# A.25 Correlation between features

We can also calculate correlation between features themselves.

For example:

$$
corr(TRIP\_MILES,TRIP\_MINUTES)
$$

might be high.

This makes intuitive sense:

> Longer trips often take longer.

When features are strongly correlated with each other, we may have **multicollinearity**.

This can matter for some regression models, particularly when interpreting coefficients.

Therefore, correlation analysis can be useful for investigating both:

1. feature-target relationships, and
2. feature-feature relationships.

---

# A.26 Calculating correlation with Python

We do not need to calculate Pearson's formula by hand for every dataset.

In practice, we normally use a library.

For example, with pandas:

```python
import pandas as pd

data = pd.DataFrame({
    "miles": [1, 2, 3, 4],
    "minutes": [10, 15, 20, 25],
    "passengers": [1, 1, 2, 2],
    "fare": [8, 12, 16, 20]
})

print(data.corr())
```

`data.corr()` calculates correlations between the numeric columns.

To calculate the correlation between one specific feature and the target:

```python
print(data["miles"].corr(data["fare"]))
```

We can also calculate:

```python
print(data["minutes"].corr(data["fare"]))
```

and:

```python
print(data["passengers"].corr(data["fare"]))
```

---

# A.27 Where does scikit-learn fit?

For a typical machine-learning workflow, different Python libraries are often used for different jobs.

For example:

### pandas

Useful for:

* loading data,
* inspecting data,
* selecting columns,
* calculating correlations,
* cleaning data.

### scikit-learn

Useful for:

* splitting data,
* preprocessing,
* regression models,
* predictions,
* evaluation metrics,
* pipelines.

For example:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = data[["miles", "minutes"]]
y = data["fare"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

Here:

```python
train_test_split(...)
```

creates separate training and testing data.

```python
LinearRegression()
```

creates the regression model.

```python
model.fit(...)
```

learns the model parameters from the training data.

```python
model.predict(...)
```

produces predictions for new observations.

---

# A.28 Summary of Pearson correlation

Pearson correlation:

* measures the **strength and direction of a linear relationship**;
* ranges from `-1` to `+1`;
* is positive when two variables tend to increase together;
* is negative when one tends to increase while the other decreases;
* can be close to zero when there is little linear association;
* can also be close to zero when there is a strong **nonlinear** relationship;
* is useful for exploring numerical features;
* can be calculated for every feature against the target;
* does not prove causation;
* should not be treated as the only method for feature selection.

The most important sentence to remember is:

> **Pearson correlation measures linear association, not every possible type of relationship.**

---

# Part B — Underfitting and Overfitting

## B.1 What are underfitting and overfitting?

After exploring the data, we train a machine-learning model.

The model's goal is not simply to memorize the training examples.

The real goal is:

> **Learn useful patterns that also work on new, unseen data.**

This ability is called **generalization**.

Two common problems can prevent good generalization:

* **underfitting**
* **overfitting**

---

# B.2 Underfitting

A model **underfits** when it is too simple to capture the important structure in the data.

In other words:

> The model has not learned enough.

Suppose the true relationship looks like a curve:

```text
       *
     *   *
   *       *
  *         *
```

but we use only a straight line:

```text
*
  *
    *
      *
```

The line cannot represent the curved relationship well.

That is underfitting.

---

# B.3 Example of underfitting

Suppose a model produces:

| Model        | Training RMSE | Test RMSE |
| ------------ | ------------: | --------: |
| Simple model |            20 |        21 |

The training error is:

$$
20
$$

and the test error is:

$$
21
$$

Both are relatively high.

The model performs poorly even on the training data.

That suggests the model may be too simple.

---

# B.4 Signs of underfitting

A typical underfitting pattern is:

```text
Training error: high
Test error:     high
```

There is often no dramatic difference between the two.

The important observation is that the model is simply not performing well enough.

---

# B.5 What causes underfitting?

Possible causes include:

* model is too simple;
* important features are missing;
* incorrect model form;
* too much regularization;
* insufficient training;
* poor-quality data;
* useful nonlinear relationships have not been represented.

For example, suppose taxi fare depends on both distance and trip time, but we build a model using only one weak feature.

The model may not have enough information to make good predictions.

---

# B.6 How can we reduce underfitting?

Possible approaches include:

* add useful features;
* choose a more flexible model;
* add nonlinear features;
* reduce excessive regularization;
* improve data quality;
* provide the model with better information.

In our regression lab, moving from:

```text
Linear Regression
```

to:

```text
Polynomial Regression
```

can increase model flexibility.

---

# B.7 Overfitting

Overfitting is almost the opposite problem.

A model **overfits** when it learns the training data too closely.

It learns:

* the useful underlying pattern,
* random noise,
* unusual observations,
* and accidental characteristics of the training dataset.

The model may therefore perform extremely well on training data but poorly on new data.

---

# B.8 A useful analogy

Imagine a student preparing for an exam.

A student who understands the concepts can answer:

> new questions that test the same concepts in a different way.

A student who memorizes the exact answers to practice questions may get:

> very high scores on those exact questions.

But when the questions are changed, performance may drop significantly.

A machine-learning model that memorizes training data has a similar problem.

The model needs to **generalize**, not memorize.

---

# B.9 Example of overfitting

Suppose we train a very flexible model:

| Model              | Training RMSE | Test RMSE |
| ------------------ | ------------: | --------: |
| Very complex model |             1 |        18 |

The training error is:

$$
1
$$

which is excellent.

But test error is:

$$
18
$$

which is much worse.

This large difference is a warning sign for overfitting.

---

# B.10 Signs of overfitting

A typical pattern is:

```text
Training error: very low
Test error:     much higher
```

The important clue is the **gap** between training performance and test performance.

The model performs very well on data it has already seen, but much worse on data it has not seen.

---

# B.11 Why does overfitting happen?

A model with a lot of flexibility has many possible ways to fit the training data.

That flexibility is useful when the underlying relationship is genuinely complex.

But if the model is too flexible, it can start learning noise.

For example, suppose a dataset contains random measurement errors.

A simple model may mostly ignore them and learn the overall trend.

A highly flexible model might bend itself specifically around those errors.

It begins learning:

```text
signal + noise
```

instead of:

```text
signal
```

---

# B.12 Polynomial regression is an excellent example

Suppose the true relationship is approximately:

$$
y=x^2
$$

We could start with a degree-1 polynomial:

$$
y=b_0+b_1x
$$

This is just linear regression.

Then we can use degree 2:

$$
y=b_0+b_1x+b_2x^2
$$

Then degree 3:

$$
y=b_0+b_1x+b_2x^2+b_3x^3
$$

Then degree 5:

$$
y=b_0+b_1x+b_2x^2+\cdots+b_5x^5
$$

and eventually perhaps a very high degree.

As degree increases, the model becomes more flexible.

But more flexible does not automatically mean better.

---

# B.13 Model complexity

A useful concept is **model complexity**.

Very simple model:

```text
low complexity
```

Moderately flexible model:

```text
appropriate complexity
```

Very flexible model:

```text
high complexity
```

A common pattern is:

```text
Too simple
     ↓
Underfitting

Appropriate complexity
     ↓
Good generalization

Too complex
     ↓
Overfitting
```

The challenge is to find a model that is complex enough to learn the underlying structure but not so complex that it memorizes noise.

---

# B.14 Worked polynomial example

Suppose we try several polynomial degrees.

We obtain:

| Degree | Training RMSE | Test RMSE |
| -----: | ------------: | --------: |
|      1 |          15.0 |      15.5 |
|      2 |          10.0 |      10.8 |
|      3 |           8.5 |       9.2 |
|      5 |           5.0 |       8.0 |
|     10 |           1.5 |      14.0 |

Let's analyze them.

### Degree 1

Training:

$$
15.0
$$

Test:

$$
15.5
$$

Both errors are relatively high.

The model may be too simple.

Possible underfitting.

### Degree 2

Training:

$$
10.0
$$

Test:

$$
10.8
$$

Performance improves.

### Degree 3

Training:

$$
8.5
$$

Test:

$$
9.2
$$

Performance improves further.

### Degree 5

Training:

$$
5.0
$$

Test:

$$
8.0
$$

The model fits the training data much better, and test performance is still good.

### Degree 10

Training:

$$
1.5
$$

This is extremely good training performance.

However:

$$
Test\ RMSE=14.0
$$

The test performance has become much worse.

This is a strong sign that the model is overfitting.

---

# B.15 The important pattern

Look at what happens as model complexity increases.

Training error tends to decrease:

```text
15 → 10 → 8.5 → 5 → 1.5
```

That looks wonderful.

But test error behaves differently:

```text
15.5 → 10.8 → 9.2 → 8.0 → 14.0
```

Test performance improves initially, reaches a useful region, and then becomes worse.

This gives us an important lesson:

> **A model can continue improving on the training data while becoming worse on unseen data.**

That is the core idea behind overfitting.

---

# B.16 Why do we need separate training and test data?

Suppose we train and evaluate on exactly the same observations.

The model has already seen those observations.

A flexible model can adapt itself extremely well to those known examples.

This means:

```text
training performance
```

can be misleading.

We therefore divide the data.

For example:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

We train using:

```python
X_train
y_train
```

and evaluate using:

```python
X_test
y_test
```

The test data should represent observations the model did not see during training.

---

# B.17 Training performance versus test performance

We can think of the two datasets this way:

### Training data

Question:

> How well did the model learn from the examples it was given?

### Test data

Question:

> How well does the trained model perform on new examples?

The second question is much closer to the real purpose of machine learning.

---

# B.18 Generalization

**Generalization** means that the model can perform well on data that was not used to train it.

A useful machine-learning model should learn:

```text
underlying structure
```

rather than:

```text
specific training examples
```

Therefore, a good model is not necessarily the model with the smallest training error.

A better question is:

> Which model gives good performance on data it has not seen?

---

# B.19 The three basic situations

We can summarize model behavior using three cases.

## Underfitting

```text
Training error: high
Test error:     high
```

The model is too simple or lacks useful information.

## Good fit

```text
Training error: low
Test error:     low
```

The model has learned useful structure and generalizes reasonably well.

## Overfitting

```text
Training error: very low
Test error:     much higher
```

The model is fitting the training data too closely.

---

# B.20 A table to remember

| Situation           | Training Performance | Test Performance |
| ------------------- | -------------------- | ---------------- |
| Underfitting        | Poor                 | Poor             |
| Good generalization | Good                 | Good             |
| Overfitting         | Very good            | Poor             |

The exact numerical values depend on the problem.

There is no universal threshold such as:

> "RMSE above 10 means underfitting."

The interpretation depends on:

* the target scale,
* the dataset,
* the application,
* and the model.

The relationship between training and test performance is often more informative than one isolated number.

---

# B.21 How can we measure the errors?

For regression, common metrics include:

### Mean Absolute Error

$$
MAE=
\frac{1}{n}
\sum |y_i-\hat{y}_i|
$$

It tells us the average absolute prediction error.

### Mean Squared Error

$$
MSE=
\frac{1}{n}
\sum(y_i-\hat{y}_i)^2
$$

It gives larger errors more influence because the errors are squared.

### Root Mean Squared Error

$$
RMSE=
\sqrt{MSE}
$$

RMSE is expressed in the same units as the target.

These metrics allow us to compare training and test performance.

---

# B.22 Measuring training and test RMSE

For example:

```python
from sklearn.metrics import mean_squared_error
import numpy as np

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_rmse = np.sqrt(
    mean_squared_error(y_train, train_predictions)
)

test_rmse = np.sqrt(
    mean_squared_error(y_test, test_predictions)
)

print("Training RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
```

The important idea is that we calculate the metric separately.

We want to know:

```text
How well does the model fit training data?
```

and:

```text
How well does it generalize to test data?
```

---

# B.23 Why isn't the lowest training error always the best model?

Suppose we compare two models.

### Model A

```text
Training RMSE = 5
Test RMSE     = 8
```

### Model B

```text
Training RMSE = 1
Test RMSE     = 20
```

Model B is better on the training data.

But Model A is much better on unseen data.

If our goal is prediction, Model A is usually the more useful model.

This is one of the most important lessons in machine learning:

> **Do not select a model only because it has the lowest training error.**

---

# B.24 Bias and variance intuition

Underfitting and overfitting are often related to two concepts:

* **bias**
* **variance**

These concepts can be introduced intuitively.

### High bias

The model makes strong simplifying assumptions.

It is unable to represent the true pattern well.

This is associated with underfitting.

### High variance

The model is very sensitive to the particular training sample.

Small changes in the training data may lead to substantially different models.

This is associated with overfitting.

A simplified picture is:

```text
Too simple
     ↓
High bias
     ↓
Underfitting

Balanced
     ↓
Good generalization

Too flexible
     ↓
High variance
     ↓
Overfitting
```

This idea is known as the **bias-variance trade-off**.

---

# B.25 Is a complex model always bad?

No.

Complexity itself is not the problem.

Suppose the true relationship really is complicated.

A simple model may underfit.

A more flexible model could then perform better on new data.

The problem occurs when the model becomes more flexible than the available evidence can support.

The correct goal is therefore not:

> Always choose the simplest model.

Instead:

> **Choose a model with an appropriate level of complexity for the data and the problem.**

---

# B.26 What can cause overfitting?

Overfitting can be encouraged by:

* overly complex models;
* very high polynomial degree;
* too many irrelevant features;
* small training datasets;
* noisy data;
* excessive model flexibility;
* poor validation procedures.

For example:

```python
PolynomialFeatures(degree=20)
```

may create a very flexible representation.

That does not guarantee overfitting, but it increases the model's capacity to fit complicated patterns.

---

# B.27 How can we reduce overfitting?

There are several strategies.

## Use a simpler model

For polynomial regression, try a lower degree.

For example:

```python
PolynomialFeatures(degree=2)
```

instead of:

```python
PolynomialFeatures(degree=15)
```

## Use more data

More representative observations can help the model distinguish signal from noise.

## Use useful features

Remove features that add noise or irrelevant information where appropriate.

## Use regularization

Regularization discourages overly large model parameters.

Examples include:

* Ridge Regression,
* Lasso Regression,
* ElasticNet.

These are useful techniques that we can study after the basic regression models.

## Use validation techniques

Cross-validation can provide a more reliable estimate of model performance across different subsets of the data.

---

# B.28 Training, validation, and test sets

For a simple introductory lab, we often use:

```text
training data
+
test data
```

The model learns from the training data and is evaluated on the test data.

For more advanced machine learning, we may use:

```text
training data
validation data
test data
```

The roles are:

### Training set

Used to fit the model.

### Validation set

Used to compare choices such as:

* polynomial degree,
* hyperparameters,
* model types.

### Test set

Used for the final unbiased evaluation after model decisions have been made.

This helps prevent us from repeatedly using the test set to make decisions.

---

# B.29 Cross-validation

A more robust method is **cross-validation**.

Instead of relying on a single training/validation split, the data is divided into several folds.

For example, with 5-fold cross-validation:

```text
Fold 1 → validation
Fold 2 → validation
Fold 3 → validation
Fold 4 → validation
Fold 5 → validation
```

Each fold gets a turn as the validation set.

The scores can then be averaged.

This provides a more stable estimate of how the model behaves on different samples.

Cross-validation becomes especially useful when comparing model configurations.

---

# B.30 Data leakage and overfitting

There is another important issue related to model evaluation:

**data leakage**.

Data leakage occurs when information that should not be available during training influences the model.

For example, consider scaling.

Incorrect:

```python
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)
```

Here the scaler was fitted using all of `X`, including the future test data.

Instead:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Now the scaler learns from the training data only.

The test data is transformed using the information learned from training data.

The rule is:

> **Fit preprocessing on the training data, then transform the training and test data using that fitted preprocessing.**

This is one reason `Pipeline` becomes useful later in the course.

---

# B.31 Underfitting and overfitting in our three regression approaches

The three main regression methods in the lab provide a useful progression.

## Linear Regression

The model assumes:

$$
\hat{y}=b_0+b_1x
$$

For multiple features:

$$
\hat{y}=b_0+b_1x_1+b_2x_2+\cdots+b_px_p
$$

This is relatively simple.

It can underfit if the true relationship is strongly nonlinear.

---

## Multiple Linear Regression

Multiple linear regression uses several features.

For example:

$$
\hat{y}
=
b_0+
b_1(\text{Miles})+
b_2(\text{Minutes})+
b_3(\text{Passengers})
$$

This gives the model more information.

However:

> More features do not automatically mean better performance.

Some features can be irrelevant, noisy, or redundant.

---

## Polynomial Regression

Polynomial regression adds terms such as:

$$
x^2
$$

$$
x^3
$$

and interactions such as:

$$
x_1x_2
$$

This allows a linear regression estimator to represent nonlinear patterns.

For example:

$$
\hat{y}
=
b_0+
b_1x+
b_2x^2
$$

can represent a curved relationship.

But increasing polynomial degree indefinitely can lead to overfitting.

---

# B.32 A useful experiment

Suppose students run several models and record:

| Model                      | Training RMSE | Test RMSE | Interpretation |
| -------------------------- | ------------: | --------: | -------------- |
| Linear Regression          |           ... |       ... | ...            |
| Multiple Linear Regression |           ... |       ... | ...            |
| Polynomial Degree 2        |           ... |       ... | ...            |
| Polynomial Degree 3        |           ... |       ... | ...            |
| Polynomial Degree 5        |           ... |       ... | ...            |

Then ask:

1. Which model has the lowest training error?
2. Which model has the lowest test error?
3. Are they the same model?
4. Does increasing polynomial degree improve test performance?
5. At what point does test performance stop improving?
6. Is there evidence of underfitting?
7. Is there evidence of overfitting?

This is more meaningful than simply asking:

> Which model has the smallest training error?

---

# B.33 The relationship between complexity and performance

A useful conceptual diagram is:

```text
Error
 ^
 |\
 | \
 |  \        Test error
 |   \      /\
 |    \____/  \
 |         \
 |          \
 |           \ Training error
 +------------------------------> Model complexity
       low      optimal     high
```

The exact shape will vary from problem to problem.

But the general idea is:

### Low complexity

The model cannot capture enough of the structure.

Possible underfitting.

### Medium complexity

The model captures useful structure and generalizes well.

### Very high complexity

The model begins fitting noise.

Possible overfitting.

---

# B.34 The goal of machine learning

The overall objective can be summarized as:

$$
\boxed{\text{Learn useful patterns that generalize to unseen data}}
$$

Not:

$$
\boxed{\text{Minimize training error at all costs}}
$$

And not:

$$
\boxed{\text{Use the most complicated model possible}}
$$

Instead, we want an appropriate balance between:

* model flexibility,
* data quality,
* available information,
* and generalization performance.

---

# B.35 Final summary

## Pearson correlation

Remember:

> **Pearson `r` measures the strength and direction of a linear relationship between two numerical variables.**

Important points:

* `r` ranges from `-1` to `+1`.
* Positive values indicate positive linear association.
* Negative values indicate negative linear association.
* Values close to zero indicate little linear association.
* A value close to zero does not necessarily mean no relationship.
* Nonlinear relationships, such as:

$$
y=x^2
$$

may have low Pearson correlation.

Correlation is useful for exploring relationships between:

* features and the target,
* and features themselves.

---

## Underfitting

Remember:

> **Underfitting means that the model is too simple to capture the important structure in the data.**

Typical pattern:

```text
Training error: high
Test error:     high
```

---

## Overfitting

Remember:

> **Overfitting means that the model fits the training data too closely and does not generalize well to unseen data.**

Typical pattern:

```text
Training error: very low
Test error:     much higher
```

---

## Generalization

The ultimate objective is:

$$
\boxed{\text{Good performance on unseen data}}
$$

This is why we:

* split data into training and test sets,
* compare training and test performance,
* examine model complexity,
* use validation techniques,
* and eventually use methods such as regularization and cross-validation.

---

# B.36 Connection to the regression lab

The lab applies these concepts in sequence:

1. Explore the dataset.
2. Examine relationships between variables.
3. Calculate and interpret correlations.
4. Select features and a target.
5. Split the data into training and test sets.
6. Train a Linear Regression model.
7. Extend the model to Multiple Linear Regression.
8. Create polynomial features.
9. Train Polynomial Regression models with different degrees.
10. Compare training and test performance.
11. Look for evidence of underfitting or overfitting.
12. Evaluate models using regression metrics.
13. Finally, use `Pipeline` as an optional exercise to organize preprocessing and modeling.

The most important habit to develop is:

> **Always look beyond how well a model performs on training data. Ask whether it can generalize to data that it has not seen.**


<!-- # Part 1: Supervised Machine Learning: A Beginner's Guide


Machine Learning is behind many exciting technologies today, like self-driving cars, personalized recommendations, and even medical diagnoses. One of the key approaches to ML is **Supervised Learning**, where the computer learns from examples that include both the input data and the correct outputs. In this reading, we'll explore how this technique works and how it powers real-world applications.


-----
## Intro

**1. What is Machine Learning, Briefly?**

At its core, Machine Learning is about teaching computers to learn patterns from data *without* being explicitly programmed for every single scenario. Instead of writing exact rules, we give the computer data and let it figure out the rules itself.

> [!NOTE]  
> There is an example [here.](./img/ml_vs_traditional_paradigm.png)


**2. What Makes Learning "Supervised"?**

Imagine you're learning to identify different types of fruit. Someone (the "supervisor") shows you an apple and says "This is an apple." Then shows you a banana and says "This is a banana." You learn by connecting the fruit you see (the **input features**, like shape, color, size) with the correct name (the **output label** or **target**).

Supervised Learning works the same way:
*   We feed the computer **labeled data**.
*   This data consists of **input features** (characteristics of something) and the corresponding **correct output label** (the "answer").
*   The computer's goal is to learn a mapping, or a set of rules, to predict the output label for *new, unseen* inputs based on the patterns it learned from the examples.

**3. The Big Picture: A Standard Recipe (The ML Pipeline)**

Just like following a recipe when cooking, there's a standard process, or **pipeline**, that we follow in most supervised learning projects. This helps keep things organized and ensures we cover all the important steps:

-  **Get the Data:** Find and load the dataset you want to learn from.
-  **Clean and Prepare the Data (Preprocessing):** This is super important! Real-world data is often messy. We need to handle missing pieces, make sure everything is in a format the computer understands (like numbers instead of text), and sometimes adjust the scale of numbers. (We saw this clearly in Part 5 of the lab!).
-  **Split the Data:** Divide your data into two parts:
    *   **Training Set:** Used to *teach* the model. (Usually the larger part, like 80%).
    *   **Testing Set:** Used to *evaluate* how well the model learned, using data it hasn't seen before. (Usually the smaller part, like 20%). This prevents the model from just "memorizing" the answers.
-  **Choose Your Tool (The Model):** Select an appropriate algorithm or "model" based on your task.
-  **Teach the Tool (Train the Model):** Feed the *training data* to the model. This is where the learning happens! (`model.fit()`).
-  **Test the Tool (Make Predictions):** Ask the trained model to make predictions on the *testing data's* input features. (`model.predict()`).
-  **Check the Score (Evaluate the Model):** Compare the model's predictions on the test set with the actual known answers (the test set labels). This tells us how well the model performs on unseen data.

-----
##  The Two Main Flavors of Supervised Learning

Based on the *type* of "answer" or label we want to predict, supervised learning splits into two main categories:

*   **Classification (Predicting Categories):**
    *   **Goal:** Assign data points to distinct groups or classes. The output is a category name.
    *   **Examples:**
        *   Will a passenger survive on the Titanic? (*Categories: Survived, Did Not Survive*) - Lab Part 1
        *   What species is this Iris flower? (*Categories: Setosa, Versicolor, Virginica*) - Lab Part 2
        *   Is this email Spam or Not Spam?
    *   **A Classic Tool: Decision Tree:** Imagine a flowchart of yes/no questions. A Decision Tree asks a series of questions about the input features to arrive at a final category prediction.

*   **Regression (Predicting Numbers):**
    *   **Goal:** Predict a continuous numerical value. The output is a number on a scale.
    *   **Examples:**
        *   What is the predicted fuel efficiency (MPG) of this car? (*Value: e.g., 25.5 MPG*) - Lab Part 3
        *   What is the predicted diabetes progression score? (*Value: e.g., 150.0*) - Lab Part 4
        *   What will the temperature be tomorrow?
    *   **A Classic Tool: Linear Regression:** Think of drawing the "best-fit" straight line through data points on a graph. Linear Regression tries to find a linear relationship between the input features and the numerical output value.

<details>
<summary>About the dataset</summary>

### The MPG dataset

The **MPG dataset** (Miles Per Gallon dataset) is a popular dataset used in statistics and machine learning to analyze fuel efficiency of various cars. It originates from the **1974 Motor Trend US magazine** and contains specifications of various automobile models from the **1970s to early 1980s**.

**Dataset Overview**
The dataset includes **398 rows (observations)** and **9 columns (features)**, with some missing values. It is often used to practice **regression analysis**, particularly predicting **fuel efficiency (MPG)** based on car features.

**Features**
1. **mpg** (Miles per gallon) – *Target variable (continuous)*
2. **cylinders** – Number of cylinders (categorical: 4, 6, 8)
3. **displacement** – Engine displacement (cubic inches) (continuous)
4. **horsepower** – Engine horsepower (continuous, some missing values)
5. **weight** – Vehicle weight (in pounds) (continuous)
6. **acceleration** – Time to accelerate from 0 to 60 mph (continuous)
7. **model year** – Year of manufacture (1970-1982) (categorical)
8. **origin** – Country of origin (1 = USA, 2 = Europe, 3 = Japan) (categorical)
9. **car name** – Name of the car (string)

**Common Use Cases**
- **Predicting fuel efficiency** (MPG) using regression models.
- **Feature analysis** to determine which variables influence fuel efficiency the most.
- **Exploratory Data Analysis (EDA)** to identify trends in car performance over time.
- **Machine Learning** applications, including linear regression, decision trees, and neural networks.

### The Diabetes Progression Score

The **Diabetes Progression Score** is a key target variable in a well-known **diabetes dataset** often used for regression analysis in machine learning. This dataset is commonly referred to as the **Diabetes Dataset** from the **Scikit-learn library**, which originates from medical research.

**Overview of the Diabetes Dataset**
- The dataset is based on a study of **diabetes progression** in patients.
- It includes **442 observations** (patients) and **10 features** related to health metrics.
- The **target variable** is a **continuous numeric score** representing **disease progression one year after baseline**.

**Features (Predictors)**
The dataset contains 10 baseline variables (all numeric and standardized), which are:
1. **Age** – Patient’s age in years.
2. **Sex** – A binary variable representing sex.
3. **BMI** – Body Mass Index (a measure of body fat based on height and weight).
4. **Blood Pressure** – Average blood pressure.
5. **S1-S6 (Six blood serum measurements)** – Various biochemical markers in the blood.

**Target Variable**
- The **diabetes progression score** is a **quantitative measure** of how diabetes has progressed **one year after the baseline measurement**.
- Higher scores indicate **worse** progression of the disease.

**Use Cases**
- **Regression Analysis**: Predicting the progression score using features like BMI, blood pressure, and serum measurements.
- **Feature Importance**: Identifying which factors most strongly influence diabetes progression.
- **Medical Research**: Understanding diabetes trends and patient health over time.
- **Machine Learning Models**: Used for training and evaluating regression models.


</details>


-----
##  How Do We Know if the Model is Any Good? (Evaluation)

Training a model isn't enough; we need to know if it actually learned well! That's why we use the **Testing Set** – data the model never saw during training.

*   For **Classification**, we often look at **Accuracy** (what percentage did it get right overall?). We also use metrics like **Precision** and **Recall** to understand specific types of errors (covered in the lab summary).
*   For **Regression**, accuracy doesn't make sense. Instead, we measure the *error*, how far off were the predictions on average? Metrics like **Mean Absolute Error (MAE)** or **Mean Squared Error (MSE)** tell us this. We also use **R-squared (R²)** to see how much of the variation in the real answers our model could explain.

**6. The Reality Check: Why Cleaning Data Matters (Preprocessing)**

Real datasets aren't perfect (as illustrated in activity1). They might have:

*   **Missing Values:** Blank spots where data should be.
*   **Categorical Data:** Text descriptions (like 'USA', 'Europe', 'Japan') that models can't directly use.
*   **Different Scales:** Some numbers might be huge (like car weight) while others are small (like number of cylinders), which can confuse some models.

**Preprocessing** is the step where we fix these issues:
*   We **impute** (fill in) missing values reasonably.
*   We **encode** categorical text into numerical representations (like One-Hot Encoding).
*   We **scale** numerical features so they are on a more level playing field (like Standardization).

*Without good preprocessing, even the best model might perform poorly!*

-----
## Putting it All Together

Supervised Learning is a powerful technique where we teach computers by showing them examples with answers (labeled data). We follow a standard pipeline: get data, clean it (preprocess!), split it, choose a model (like a Decision Tree for categories or Linear Regression for numbers), train it, test it, and evaluate how well it learned. Understanding this process is your first big step into the world of building intelligent systems!

-----
## FAQ


**1. Why is Accuracy Not Enough in Classification?**
Accuracy alone can be misleading, especially when dealing with **imbalanced datasets**. This is because accuracy simply measures how many predictions are correct, but it doesn't distinguish between **false positives** and **false negatives**, which can be critical in certain applications.

**Example: Medical Diagnosis**
Imagine a classification model predicting whether a patient has a **rare disease**:
- 990 patients **don’t** have the disease.
- 10 patients **do** have the disease.

If the model **always predicts "No disease"**, it will be correct for 990 out of 1000 patients, giving an **accuracy of 99%**. However, it completely **fails to detect any actual cases**, which is dangerous.

This is why we need **Precision and Recall**:

- **Precision** (Positive Predictive Value) – Measures how many of the predicted positive cases were actually correct. High precision means fewer false positives.
- **Recall** (Sensitivity) – Measures how many of the actual positive cases were correctly identified. High recall means fewer false negatives.

For a disease prediction model, **high recall** is crucial because missing a real case (false negative) could be life-threatening.



**Why Precision, Recall, and F1 Score Matter in Medical Diagnosis**
In medical diagnoses, why are precision and recall not always enough, and how does the F1 score provide a better evaluation metric? Explain with an example.

<details>  
<summary><strong>View Answer</strong></summary>

In medical diagnosis, precision and recall alone may not provide a balanced view of a model’s performance, especially when there’s a need to strike a balance between false positives and false negatives. The **F1 score**, which is the harmonic mean of precision and recall, is particularly useful when both false positives and false negatives are undesirable and need to be balanced.

**Example:** Consider a medical test for detecting a rare disease that affects only 1 in 1,000 people. Suppose a diagnostic model is trained to identify whether a person has the disease (positive) or not (negative). If the model predicts "no disease" for every patient, it will have a **high accuracy** (99.9%) but will miss all actual cases of the disease. This is a clear failure of the model in a medical context, even though accuracy is high.

* **Precision** would tell us that when the model predicts someone has the disease, how likely it is that they actually do. A high precision would be important to avoid unnecessary treatments for false positives.
* **Recall** would measure how well the model identifies actual patients with the disease, minimizing the risk of false negatives (i.e., missed diagnoses).

In this case, both **Precision** and **Recall** are important, but **F1 Score** combines them into a single metric. It allows us to optimize for both detecting as many true positives (sick patients) as possible while minimizing the chance of false positives (unnecessary treatment or worry).

For instance, a model with:

* **Precision = 0.95** (95% of the positive predictions are correct)
* **Recall = 0.50** (50% of actual cases are detected)

Might have a **F1 Score** of 0.67, which is a better representation of the model's overall effectiveness than looking at either precision or recall alone.

This is why **F1 score** is crucial: it balances the importance of both **false negatives** and **false positives**, giving a more holistic view of a model’s real-world performance in scenarios like medical diagnosis.

</details> 

<br>

---

**2. What is a Confusion Matrix?**
A **Confusion Matrix** is a table used to evaluate the performance of a classification model. It shows the number of:
- **True Positives (TP)** – Correctly predicted positive cases.
- **False Positives (FP)** – Incorrectly predicted as positive (Type I error).
- **False Negatives (FN)** – Incorrectly predicted as negative (Type II error).
- **True Negatives (TN)** – Correctly predicted negative cases.

#### **Example of a Confusion Matrix for a Binary Classifier:**
| Actual \ Predicted | Positive (1) | Negative (0) |
|--------------------|-------------|-------------|
| **Positive (1)**   | TP (Correct) | FN (Missed case) |
| **Negative (0)**   | FP (False alarm) | TN (Correct) |

From this matrix, we can calculate:
- **Accuracy** = (TP + TN) / (TP + FP + TN + FN)
- **Precision** = TP / (TP + FP)
- **Recall (Sensitivity)** = TP / (TP + FN)
- **F1-score** = 2 × (Precision × Recall) / (Precision + Recall) (harmonic mean of precision & recall)


**3. MAE, MSE, and RMSE in Regression**  

When evaluating a **regression model**, we often use error metrics to measure how far the predictions are from the actual values. The three most common metrics are:  

- **Mean Absolute Error (MAE)**  
   - Measures the average absolute difference between predicted and actual values.  
   - Formula:  
     $$
     MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
     $$
   - **Pros:** Easy to interpret, gives equal weight to all errors.  
   - **Cons:** Does not emphasize large errors.  

- **Mean Squared Error (MSE)**  
   - Measures the average squared difference between predicted and actual values.  
   - Formula:  
     $$
     MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
     $$
   - **Pros:** Penalizes larger errors more than smaller ones, making it useful when large errors need to be avoided.  
   - **Cons:** Squaring the errors makes the metric harder to interpret in the original unit.  

- **Root Mean Squared Error (RMSE)**  
   - Square root of MSE, which brings the error back to the original unit.  
   - Formula:  
     $$
     RMSE = \sqrt{MSE}
     $$
   - **Pros:** More interpretable than MSE because it’s in the same unit as the target variable.  
   - **Cons:** Sensitive to outliers, since squaring the errors gives more weight to larger deviations.

**Which One is Better?**  
- **MAE** is better when you want an interpretable, balanced metric that treats all errors equally.  
- **MSE and RMSE** are better when larger errors are more concerning, as they penalize big mistakes more heavily.  
- **RMSE** is often preferred in practical applications because it maintains the units of the target variable while still emphasizing larger errors.  


---
## Further Exploration

For more information, please refer to this reading:
- [Classic Machine Learning Algorithms](https://www.ncbi.nlm.nih.gov/books/NBK597496/)
- [Introduction to Supervised Learning](https://developers.google.com/machine-learning/intro-to-ml/supervised)
- [Linear regression](https://developers.google.com/machine-learning/crash-course/linear-regression)
- [Classification](https://developers.google.com/machine-learning/crash-course/classification)
- [Algorithms for Supervised machine learning](https://scikit-learn.org/stable/supervised_learning.html)
- Video Course: Intro to Machine Learning with Python
  - [Part 4: Train a Classification Model](https://www.youtube.com/watch?v=f3kSEebz8QA)
  - [Part 5: Cross-Validation and Identifying Misclassified Points](https://www.youtube.com/watch?v=cEs7UfimlEk)
  - [Part 6: Model Tuning and Test Set Accuracy](https://www.youtube.com/watch?v=eU6Jr9DpcrQ)
 - [Part 7: Wrap-Up](https://www.youtube.com/watch?v=zI58IQpE2uE)  
 -->
