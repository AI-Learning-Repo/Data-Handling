# Activity 1: Understanding, Exploring, and Preparing Data

In this activity, you will investigate two classic datasets:

* **Titanic** — a relatively messy, real-world dataset containing missing values, categorical variables, and unusual observations.
* **Iris** — a small, clean academic dataset that allows us to see what happens when many data-quality problems are already absent.

The activity follows the workflow from the lecture:

```text
Understand the data
        ↓
Initial inspection / EDA
        ↓
Identify data-quality problems
        ↓
Clean / transform the data
        ↓
EDA again and validate
        ↓
Feature engineering and preparation
        ↓
Next stage: supervised learning
```

> [!IMPORTANT]
> **EDA and data cleaning are iterative.**
>
> Do not think of EDA and cleaning as two completely separate steps.
>
> During the activity, you will first explore the data, discover problems, clean or transform the data, and then explore it again to determine whether the changes improved the dataset or revealed additional issues.

> [!IMPORTANT]
> **If you use LLM as a learning assistant.**
>
> Do not simply copy code without understanding what it does. You are encouraged to question every line of code. 



---

# Setup: Environment Preparation

## 1. Open Google Colab

Open [Google Colab](https://colab.research.google.com/) and create a **New Notebook**.

## 2. Install the Required Libraries

Create a new code cell and run:

```bash
!pip install pandas seaborn matplotlib -q
```

### Libraries used

* `pandas`: A Python library for working with structured data. It can be thought of as Python's version of a spreadsheet or database table.
* `seaborn`: A visualization library built on Matplotlib that is particularly useful for statistical data visualization.
* `matplotlib`: A general-purpose Python visualization library.

---

# Part 1: The Titanic Dataset

The Titanic dataset contains historical information about passengers aboard the Titanic.

Unlike a perfectly prepared academic dataset, it contains several characteristics that make it useful for learning about real-world data:

* missing values;
* numerical variables;
* categorical variables;
* ordinal variables;
* unusual observations;
* variables that may contain redundant or less useful information.

The goal of this activity is **not yet to build a predictive model**.

Instead, we will use the Titanic dataset to practice:

1. understanding data;
2. inspecting data;
3. performing EDA;
4. identifying data-quality problems;
5. cleaning the data;
6. performing EDA again;
7. creating useful features;
8. preparing data for the next stage.

---

# 1. Understanding the Dataset

Before analyzing or modifying data, we need to understand what the dataset represents.

## Your Task is to write Python code that:

1. imports Pandas, Seaborn, and Matplotlib;
2. loads the Titanic dataset from Seaborn;
3. displays the first few rows;
4. displays the dataset dimensions;
5. displays the dataset information;
6. displays summary statistics.

### Suggested prompt for Gemini

> "Write Python code for Google Colab. Import pandas, seaborn, and matplotlib. Load the 'titanic' dataset using seaborn. Display the first five rows, the shape of the dataset, the dataset information using `.info()`, and numerical summary statistics using `.describe()`. Explain what each command tells me about the dataset."

---

## Working Code

<details>
<summary><strong>Click here for the working code and explanations</strong></summary>

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
df_titanic = sns.load_dataset("titanic")

# First few observations
display(df_titanic.head())

# Number of rows and columns
print("Dataset shape:")
print(df_titanic.shape)

# Dataset information
print("\n--- Dataset Info ---")
df_titanic.info()

# Summary statistics
print("\n--- Summary Statistics ---")
display(df_titanic.describe())
```

### What should you notice?

The dataset contains **891 rows**.

Some columns contain fewer than 891 non-null observations.

For example:

* `age` contains missing values.
* `deck` contains a very large number of missing values.

The `.info()` output therefore provides an early indication that the dataset contains data-quality issues.

However, **we should not immediately start cleaning**.

We first want to explore the data and understand these problems more carefully.

</details>

---

## Understanding the Schema

Look at the columns and classify them.

| Variable      | Type                 | Description                       |
| ------------- | -------------------- | --------------------------------- |
| `survived`    | Categorical / binary | Whether the passenger survived    |
| `pclass`      | Ordinal              | Passenger class                   |
| `sex`         | Categorical          | Passenger sex                     |
| `age`         | Numerical            | Passenger age                     |
| `sibsp`       | Numerical            | Number of siblings/spouses aboard |
| `parch`       | Numerical            | Number of parents/children aboard |
| `fare`        | Numerical            | Ticket price                      |
| `embarked`    | Categorical          | Port of embarkation               |
| `class`       | Ordinal/categorical  | Passenger class as a label        |
| `who`         | Categorical          | Passenger category                |
| `deck`        | Categorical          | Deck information                  |
| `embark_town` | Categorical          | Embarkation town                  |
| `alive`       | Categorical          | Survival represented as text      |

### Think About It

Before continuing, answer:

1. Which variable could be the **target** in a supervised-learning problem?
2. Which variables are numerical?
3. Which variables are categorical?
4. Which variable is ordinal?
5. Which variables contain missing values?

> **Important:** We are identifying the target because it helps us understand the dataset. We are not building a supervised-learning model yet.

---

# 2. Initial Data Inspection and EDA

Now we begin the first EDA pass.

The purpose of this first pass is **not to produce a perfectly clean dataset**.

Instead, we want to answer:

> **What does this dataset look like, and what problems or interesting patterns can we discover?**

Remember the three main categories of EDA:

* **Univariate:** one variable
* **Bivariate:** two variables
* **Multivariate:** three or more variables

We will practice each.

---

# 2.1 Univariate EDA: One Variable at a Time

## A. Numerical Summary

Start with:

```python
df_titanic.describe()
```

### Your Task

Investigate:

* `age`
* `fare`
* `sibsp`
* `parch`

For each variable, consider:

* What is the typical value?
* How widely are values spread?
* What are the minimum and maximum values?
* Does the distribution appear symmetric or skewed?
* Are there potentially unusual observations?

---

## B. Age Distribution

Create a histogram of `age`.

### Suggested Gemini prompt

> "Using my Titanic dataframe, create a Seaborn histogram of the `age` column. Add an appropriate title and axis labels. Explain what a histogram tells me about a numerical variable."

### Working Code

<details>
<summary><strong>Click here for the working code</strong></summary>

```python
plt.figure(figsize=(10, 5))

sns.histplot(data=df_titanic, x="age", kde=True)

plt.title("Distribution of Passenger Age")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.show()
```

### Questions

1. What does the shape of the distribution tell you?
2. Are there missing ages?
3. Which age ranges appear most common?
4. Does the distribution appear perfectly symmetric?

</details>

---

# 2.2 Univariate EDA: Categorical Variables

Numerical variables are not the only things we need to explore.

Use:

```python
df_titanic["sex"].value_counts()
```

and:

```python
df_titanic["class"].value_counts()
```

### Your Task

Determine:

* How many male and female passengers are there?
* How many passengers belong to each passenger class?
* Are the categories evenly distributed?

### Suggested Gemini prompt

> "Show me how to use Pandas `value_counts()` to examine the categorical variables `sex` and `class` in my Titanic dataframe. Explain what the results tell me."

---

# 2.3 Bivariate EDA: Two Variables

Now we investigate relationships between variables.

A particularly interesting question is:

> **How is survival related to passenger characteristics?**

---

## A. Survival by Sex

Create a bar plot showing survival rate by sex.

### Suggested Gemini prompt

> "Using the Titanic dataframe, create a Seaborn barplot showing the average `survived` value for each `sex` category. Add an appropriate title and labels. Explain why the average of a binary variable represents a survival rate."

### Working Code

<details>
<summary><strong>Click here for the working code</strong></summary>

```python
plt.figure(figsize=(8, 5))

sns.barplot(
    data=df_titanic,
    x="sex",
    y="survived"
)

plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")

plt.show()
```

### Python Concept

Because `survived` is coded as:

```text
0 = Did not survive
1 = Survived
```

the mean represents the proportion of passengers who survived.

For example:

```text
mean = 0.74
```

corresponds to approximately:

```text
74% survival
```

### Questions

1. Is survival rate similar for males and females?
2. What pattern do you observe?
3. Does `sex` appear potentially informative for understanding survival?

> **Important:** We are describing a relationship in the data. We are not claiming that `sex` causes survival, and we are not yet training a model.

</details>

---

# 2.4 Bivariate EDA: Two Variables with a Third Dimension

EDA can become more informative when we use a third variable to divide the observations into groups.

Create a plot showing survival rate by passenger class and sex.

### Suggested Gemini prompt

> "Using the Titanic dataframe, create a Seaborn barplot with `class` on the x-axis, `survived` on the y-axis, and `sex` as the `hue`. Add a title and axis labels. Explain what the hue parameter does."

### Working Code

<details>
<summary><strong>Click here for the working code</strong></summary>

```python
plt.figure(figsize=(10, 5))

sns.barplot(
    data=df_titanic,
    x="class",
    y="survived",
    hue="sex"
)

plt.title("Survival Rate by Passenger Class and Sex")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.show()
```

### Questions

Look carefully at the graph.

1. How does survival differ between passenger classes?
2. How does survival differ between males and females?
3. Does the combination of class and sex reveal a more detailed pattern than either variable alone?

</details>

---

# 2.5 Multivariate EDA

Multivariate EDA examines relationships involving several variables.

One useful tool is a **correlation matrix**.

### Your Task

Calculate the correlation matrix for numerical variables.

### Suggested Gemini prompt

> "Write Pandas and Seaborn code to create a correlation matrix and heatmap for the numerical variables in my Titanic dataframe. Explain what correlation measures and why correlation does not imply causation."

### Working Code

<details>
<summary><strong>Click here for the working code</strong></summary>

```python
# Select numerical columns
numeric_columns = df_titanic.select_dtypes(
    include="number"
)

# Calculate correlations
correlation_matrix = numeric_columns.corr()

# Display the matrix
display(correlation_matrix)

# Visualize the matrix
plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix of Numerical Variables")

plt.show()
```

### Questions

1. Which variables appear to have relatively strong relationships?
2. Are any variables highly correlated?
3. Does correlation prove that one variable causes another?

> **Remember:** Correlation describes an association, not causation.

---

# 2.6 Investigating Potential Outliers

Our initial EDA may reveal unusual values.

The Titanic `fare` variable is a useful example.

### Your Task

Create a boxplot of `fare`.

### Suggested Gemini prompt

> "Create a Seaborn boxplot of the Titanic `fare` variable. Explain how a boxplot can help identify potential outliers."

### Working Code

<details>
<summary><strong>Click here for the working code</strong></summary>

```python
plt.figure(figsize=(10, 4))

sns.boxplot(data=df_titanic, x="fare")

plt.title("Distribution of Ticket Fare")
plt.xlabel("Fare")

plt.show()
```

### Think Before Cleaning

You may see very large fares.

Do **not** immediately delete or cap them.

Ask:

* Is this value impossible?
* Could it represent a genuine passenger?
* Could several passengers have shared a ticket?
* Is the observation a data-entry error?
* What does the domain context tell us?

> **Key principle:** An outlier is not automatically an error.

---

# 3. Identify Data-Quality Problems

We have now performed an initial inspection and EDA.

We can identify several potential data-quality issues.

## Your Task

Investigate:

### Missing values

```python
df_titanic.isnull().sum()
```

### Data types

```python
df_titanic.info()
```

### Duplicate rows

```python
df_titanic.duplicated().sum()
```

### Unique categorical values

For example:

```python
df_titanic["sex"].unique()
```

and:

```python
df_titanic["class"].unique()
```

### Numerical ranges

Inspect variables such as:

```python
df_titanic[["age", "fare"]].describe()
```

---

## Create a Data-Quality Summary

Write down at least one observation for each category:

| Data-quality question                      | What did you discover? |
| ------------------------------------------ | ---------------------- |
| Missing values                             |                        |
| Incorrect or unexpected data types         |                        |
| Duplicate observations                     |                        |
| Inconsistent categories                    |                        |
| Potential outliers                         |                        |
| Potentially irrelevant/redundant variables |                        |

This is an important part of EDA:

> **EDA does not only reveal interesting patterns. It can also reveal problems in the data.**

---

# 4. Data Cleaning

We now have evidence about the quality of the dataset.

Only now do we make cleaning decisions.

---

# 4.1 Handling Missing Values

Start by checking the number of missing values.

```python
print(df_titanic.isnull().sum())
```

You should see missing values in several columns.

The original activity focused on:

* `age`
* `deck`

We will keep that exercise, but now we understand **why** we are doing it.

---

## A. Investigating Age

The `age` variable contains missing values.

One possible strategy is to replace missing ages with the median.

### Suggested Gemini prompt

> "In Pandas, how do I calculate the median of the Titanic `age` column and use it to fill missing values? Explain why median may be preferred to mean when a numerical variable contains extreme values."

### Working Code

```python
median_age = df_titanic["age"].median()

df_titanic["age"] = df_titanic["age"].fillna(median_age)
```

### Why Median?

The mean can be strongly influenced by extreme values.

The median is more robust to outliers.

For example, if most ages are between 20 and 40 but a few values are extremely large, the mean may be pulled upward.

The median is therefore often a reasonable choice for skewed data, although the correct method depends on the context.

---

# 4.2 Handling a Variable with Extensive Missingness

The `deck` variable contains a very large proportion of missing values.

Inspect it:

```python
print(df_titanic["deck"].isnull().sum())
print(df_titanic["deck"].notnull().sum())
```

The original activity drops this column.

We will keep that decision for the exercise.

### Working Code

```python
df_titanic = df_titanic.drop(columns=["deck"])
```

### Important Reasoning

We are **not** dropping `deck` simply because it contains missing values.

We are considering:

* how much information is missing;
* whether meaningful imputation is possible;
* whether the remaining information is sufficient;
* whether retaining the variable is useful for our purpose.

> **Cleaning is a decision-making process, not just a collection of commands.**

---

# 4.3 Verify the Cleaning

Always verify that the intended change actually occurred.

```python
print("Missing values after cleaning:")
print(df_titanic.isnull().sum())
```

Specifically check:

```python
print("Missing age values:", df_titanic["age"].isnull().sum())
```

You should now have:

```text
Missing age values: 0
```

---

# 4.4 Revisit Data Types

Run:

```python
df_titanic.info()
```

Look for variables that are represented appropriately.

For example:

* numerical variables should be numerical;
* categorical variables should not accidentally be stored as numerical values;
* dates should be represented appropriately when working with time-based data.

If a numerical value were stored as:

```python
"20"
```

instead of:

```python
20
```

we would need to correct its representation.

---

# 4.5 Revisit Potential Outliers

Return to the `fare` variable.

Create the boxplot again:

```python
plt.figure(figsize=(10, 4))

sns.boxplot(data=df_titanic, x="fare")

plt.title("Ticket Fare After Initial Cleaning")

plt.show()
```

### Question

Do we need to remove or transform the extreme fare values?

There is no automatic answer.

For this activity, **do not remove the fare outliers**.

The purpose is to practice the reasoning:

> Detect → Investigate → Decide → Treat if necessary

We have detected unusual values, but we do not have sufficient evidence that they are errors.

---

# 5. EDA Again: Validate the Cleaning

This is one of the most important parts of the activity.

We have changed the data.

Therefore, we should **explore it again**.

This demonstrates the iterative relationship between EDA and cleaning.

---

# 5.1 Check the Age Distribution Again

Create the age histogram again.

```python
plt.figure(figsize=(10, 5))

sns.histplot(
    data=df_titanic,
    x="age",
    kde=True
)

plt.title("Age Distribution After Cleaning")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.show()
```

### Questions

1. Are there still missing values?
2. Did the distribution change?
3. What effect did median imputation have?
4. Do you notice a concentration around the median value?

> **Important:** Imputation can change the distribution. This is one reason why we should inspect the data again after cleaning.

---

# 5.2 Check Survival Patterns Again

Recreate the survival-by-sex plot:

```python
plt.figure(figsize=(8, 5))

sns.barplot(
    data=df_titanic,
    x="sex",
    y="survived"
)

plt.title("Survival Rate by Sex After Cleaning")
plt.ylabel("Survival Rate")

plt.show()
```

Then recreate the class/sex visualization:

```python
plt.figure(figsize=(10, 5))

sns.barplot(
    data=df_titanic,
    x="class",
    y="survived",
    hue="sex"
)

plt.title("Survival Rate by Passenger Class and Sex")
plt.ylabel("Survival Rate")

plt.show()
```

### Question

Did cleaning the missing ages fundamentally change these relationships?

Why might it or might it not?

---

# 5.3 Check the Data Quality Again

Run:

```python
print(df_titanic.isnull().sum())
```

and:

```python
print(df_titanic.info())
```

Then compare the dataset **before and after cleaning**.

### Complete the following:

| Check                     | Before Cleaning | After Cleaning |
| ------------------------- | --------------: | -------------: |
| Number of rows            |                 |                |
| Number of columns         |                 |                |
| Missing `age` values      |                 |                |
| `deck` present?           |                 |                |
| Potential `fare` outliers |                 |                |

---

# 6. Feature Engineering — High-Level Preview

The next part of the lecture introduces feature engineering.

Feature engineering means transforming existing information or creating new variables that may represent useful information.

We will demonstrate this with the Titanic dataset, but the detailed role of engineered features in supervised-learning models will be covered in the **next lecture**.

---

# 6.1 Creating Family Size

The Titanic dataset contains:

* `sibsp`: siblings/spouses
* `parch`: parents/children

We can combine them to create:

```text
family_size
```

### Suggested Gemini prompt

> "Write Pandas code to create a `family_size` column by adding `sibsp` and `parch`. Explain what information this new feature represents."

### Working Code

```python
df_titanic["family_size"] = (
    df_titanic["sibsp"] + df_titanic["parch"]
)

display(
    df_titanic[
        ["sibsp", "parch", "family_size"]
    ].head()
)
```

### Think About It

Why might `family_size` be useful?

What information does it combine?

Could it also hide information by combining two variables that may have different meanings?

These are examples of questions we will revisit when discussing supervised learning.

---

# 6.2 Encoding Categorical Variables

Many machine-learning algorithms require numerical input.

For demonstration, we can convert `sex` into a binary variable.

### Suggested Gemini prompt

> "Show me how to use Pandas `get_dummies()` to convert the Titanic `sex` column into a numerical representation. Explain what one-hot encoding does and why nominal categories should not be given an artificial numerical order."

### Working Code

```python
df_titanic_encoded = pd.get_dummies(
    df_titanic,
    columns=["sex"],
    drop_first=True
)

display(
    df_titanic_encoded[
        ["age", "family_size", "sex_male"]
    ].head()
)
```

### What happened?

The original:

```text
sex
---
male
female
```

is represented numerically.

The resulting `sex_male` column contains Boolean/numerical information indicating whether the passenger is male.

### Important Concept

We should not encode:

```text
male = 1
female = 2
```

and then interpret `2 > 1` as a meaningful relationship.

That would create **false ordinality**.

One-hot encoding avoids imposing such an artificial order.

The details of encoding strategies and their relationship to particular supervised-learning algorithms will be covered next week.

---

# 7. Preparing Data for the Next Stage

At this point, we have:

* understood the dataset;
* inspected its structure;
* performed initial EDA;
* identified data-quality problems;
* cleaned missing values;
* investigated potential outliers;
* performed EDA again;
* created an example engineered feature;
* demonstrated categorical encoding.

There may still be columns that are unnecessary or redundant.

For example:

```text
alive
embark_town
```

may not be needed for a particular analysis.

However, **feature selection is task-dependent**.

Therefore, instead of claiming that we have produced a universally "final" dataset, we will create an example prepared dataset.

### Working Code

```python
columns_to_drop = [
    "alive",
    "embark_town"
]

df_prepared = df_titanic_encoded.drop(
    columns=columns_to_drop
)

print("Prepared dataset:")
df_prepared.info()
```

> **Important:** This dataset is prepared for the next stage of the workflow. It is not necessarily the final input to a machine-learning algorithm.
>
> In the next lecture, we will learn how data is divided into training and test sets and how preprocessing must be handled to avoid data leakage.

---

# 8. Exporting the Data

Pandas provides a simple method for exporting a DataFrame.

For example:

```python
df_prepared.to_csv(
    "titanic_prepared.csv",
    index=False
)
```

Here:

* `"titanic_prepared.csv"` is the filename.
* `index=False` prevents Pandas from writing row numbers as an additional column.

You can check the current directory in Google Colab using:

```bash
!ls
```

---

# 9. Part 2: The Iris Dataset

The Iris dataset contains measurements of three species of Iris flowers.

It is commonly used as an academic or "toy" dataset.

This provides a useful contrast with Titanic.

The Titanic dataset contains several data-quality issues.

The Iris dataset is already relatively clean.

> [!NOTE]
> **Domain Knowledge**
>
> * **Sepal:** The outer part of the flower, measured using `sepal_length` and `sepal_width`.
> * **Petal:** The inner, colorful part of the flower, measured using `petal_length` and `petal_width`.

<img src="./iris.png" width="50%">

---

# 10. Understanding and Inspecting Iris

## Your Task

Load the Iris dataset and inspect:

* the first few rows;
* its shape;
* its data types;
* missing values;
* summary statistics.

### Suggested Gemini prompt

> "Load the Iris dataset from Seaborn and use Pandas to display the first five rows, shape, dataset information, missing-value counts, and summary statistics."

### Working Code

<details>
<summary><strong>Click here for the working code and explanations</strong></summary>

```python
# Load the Iris dataset
df_iris = sns.load_dataset("iris")

# Inspect the dataset
display(df_iris.head())

print("Shape:")
print(df_iris.shape)

print("\n--- Dataset Info ---")
df_iris.info()

print("\n--- Missing Values ---")
print(df_iris.isnull().sum())

print("\n--- Summary Statistics ---")
display(df_iris.describe())
```

### Expected Results

You should find:

* no missing values;
* numerical measurement columns;
* a categorical `species` column;
* a small number of observations.

This demonstrates an important point:

> **Not every dataset requires extensive cleaning.**

The correct workflow is to **inspect the data first** and then decide whether cleaning is necessary.

</details>

---

# 11. Iris: Univariate EDA

Choose one numerical variable, such as `sepal_length`.

Create a histogram.

### Suggested Gemini prompt

> "Create a Seaborn histogram of `sepal_length` from the Iris dataset. Add a title and axis labels. Explain the distribution."

### Working Code

```python
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df_iris,
    x="sepal_length",
    kde=True
)

plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Number of Observations")

plt.show()
```

### Questions

1. What does the distribution look like?
2. Are there obvious extreme values?
3. Does the variable appear symmetric?
4. Are there multiple concentrations of observations?

---

# 12. Iris: Bivariate EDA

Now compare two numerical variables.

For example:

* `sepal_length`
* `sepal_width`

### Working Code

```python
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df_iris,
    x="sepal_length",
    y="sepal_width"
)

plt.title("Sepal Length vs. Sepal Width")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")

plt.show()
```

### Questions

1. Do the observations form an obvious pattern?
2. Is there a strong relationship?
3. Do you see different groups?

---

# 13. Iris: Multivariate EDA

Now investigate several variables simultaneously.

Use a pairplot.

### Suggested Gemini prompt

> "Write Python code using Seaborn to create a pairplot for the Iris dataset. Set `hue='species'` so that the observations are visually separated by species. Explain what information the pairplot provides."

### Working Code

<details>
<summary><strong>Click here for the working code and explanations</strong></summary>

```python
sns.pairplot(
    data=df_iris,
    hue="species"
)

plt.suptitle(
    "Pairwise Relationships Between Iris Features",
    y=1.02
)

plt.show()
```

### Python Concept

`sns.pairplot()` automatically creates a grid showing pairwise relationships between numerical variables.

Using:

```python
hue="species"
```

allows us to visually distinguish the three species.

### Expected Results

You should see that the species form different groups in several of the feature combinations.

Some variables appear particularly useful for distinguishing the species.

> **Important:** This is an EDA finding.
>
> We are not yet training a model or claiming a particular model accuracy.

---

# 14. Comparing Titanic and Iris

We have now worked with two very different datasets.

Complete the following table:

| Characteristic                | Titanic | Iris |
| ----------------------------- | ------- | ---- |
| Real-world historical data    |         |      |
| Missing values                |         |      |
| Numerical variables           |         |      |
| Categorical variables         |         |      |
| Potential outliers            |         |      |
| Need for substantial cleaning |         |      |
| Need for EDA                  |         |      |

### Reflection

Why is it important to inspect a dataset before deciding what cleaning operations to perform?

The answer should be:

> Because different datasets have different data-quality problems. We should identify the problems first rather than applying the same cleaning operations to every dataset.

---

# 15. Knowledge Check

Test your understanding before revealing the answers.

---

## Section A: Data Understanding and EDA

### 1. Why don't we simply clean the Titanic dataset before doing any EDA?

<details>
<summary><strong>Reveal Answer</strong></summary>

Because EDA helps us understand the dataset and identify problems that require attention.

For example, EDA can reveal:

* missing values;
* unusual distributions;
* potential outliers;
* inconsistent categories;
* relationships between variables.

After cleaning, we should perform EDA again to verify the effect of our changes.

Therefore, EDA and cleaning are **iterative rather than strictly sequential**.

</details>

---

### 2. What is the difference between univariate, bivariate, and multivariate analysis?

<details>
<summary><strong>Reveal Answer</strong></summary>

* **Univariate analysis** examines one variable.
* **Bivariate analysis** examines two variables and their relationship.
* **Multivariate analysis** examines three or more variables simultaneously.

For example:

* Histogram of `age` → univariate.
* `age` vs. `fare` scatter plot → bivariate.
* Pairplot containing several Iris variables → multivariate.

</details>

---

### 3. Why did we use the median rather than the mean when filling missing Titanic ages?

<details>
<summary><strong>Reveal Answer</strong></summary>

The median is more robust to extreme values.

A few unusually high or low ages can significantly affect the mean, while the median is much less sensitive to such observations.

However, median imputation is not automatically the correct choice for every dataset. The appropriate method depends on the data and the context.

</details>

---

### 4. We found a very large Titanic fare. Should we automatically delete it?

<details>
<summary><strong>Reveal Answer</strong></summary>

No.

An outlier is not automatically an error.

We should:

1. detect the unusual observation;
2. investigate it;
3. use domain knowledge where possible;
4. determine whether it is valid;
5. decide whether to retain, correct, remove, cap, or transform it.

The correct workflow is:

**Detect → Investigate → Decide → Treat if necessary**

</details>

---

### 5. Why did we perform EDA again after cleaning the Titanic data?

<details>
<summary><strong>Reveal Answer</strong></summary>

Because cleaning can change the data.

For example, replacing missing ages with the median changes the age distribution.

Repeating EDA allows us to:

* verify that the intended cleaning occurred;
* examine the effect of the changes;
* identify remaining problems;
* discover new patterns.

This demonstrates that EDA and cleaning are iterative.

</details>

---

### 6. What does a correlation of approximately +1 indicate?

<details>
<summary><strong>Reveal Answer</strong></summary>

It indicates a very strong positive **linear relationship** between two numerical variables.

As one variable increases, the other tends to increase as well.

However, correlation does not prove causation.

</details>

---

### 7. Why should correlation not be interpreted as causation?

<details>
<summary><strong>Reveal Answer</strong></summary>

Two variables can be strongly associated without one causing the other.

The relationship may be caused by:

* another variable;
* a common underlying factor;
* coincidence;
* the structure of the dataset.

EDA can reveal relationships, but additional analysis is required to make causal claims.

</details>

---

## Section B: Data Cleaning and Preparation

### 8. Why might we drop the Titanic `deck` variable?

<details>
<summary><strong>Reveal Answer</strong></summary>

The `deck` variable contains a very large proportion of missing values.

Rather than guessing values for most observations, we may decide that the available information is insufficient to retain the variable for the current analysis.

The decision is based on the amount and nature of missingness and the purpose of the analysis.

</details>

---

### 9. What is imputation?

<details>
<summary><strong>Reveal Answer</strong></summary>

Imputation is the process of replacing missing values with estimated or substituted values.

For example:

```python
df["age"] = df["age"].fillna(df["age"].median())
```

replaces missing ages with the median age.

</details>

---

### 10. Why is checking data types part of data inspection?

<details>
<summary><strong>Reveal Answer</strong></summary>

Because the way data is stored affects how Python and analytical tools interpret it.

For example:

```text
"20"
```

is text, while:

```text
20
```

is numerical.

If a numerical variable is incorrectly stored as text, calculations and visualizations may not behave as expected.

</details>

---

## Section C: Feature Engineering

### 11. What is feature engineering?

<details>
<summary><strong>Reveal Answer</strong></summary>

Feature engineering is the process of creating or transforming variables to represent useful information in a form suitable for analysis or later modeling.

For example, we created:

```python
family_size = sibsp + parch
```

This combines two existing variables into a new representation.

</details>

---

### 12. Why should we not encode `male = 1` and `female = 2` and then treat the values as ordinary numerical measurements?

<details>
<summary><strong>Reveal Answer</strong></summary>

Because the numbers would create an artificial order.

There is no meaningful mathematical statement that:

```text
female > male
```

The numerical labels would therefore create **false ordinality**.

One-hot encoding avoids imposing this artificial ordering.

</details>

---

## Section D: Python and Google Colab

### 13. At the beginning of the activity, we ran `!pip install pandas`. Why did we use the exclamation mark?

<details>
<summary><strong>Reveal Answer</strong></summary>

Google Colab is a Python environment running on a computer.

The `!` tells Colab to execute the following command through the underlying system shell rather than as ordinary Python code.

For example:

```bash
!pip install pandas
```

runs the system command `pip install pandas`.

The same approach can be used for other system-level commands.

</details>

---

### 14. In Google Colab, you sometimes see commands starting with `%` or `%%`. What are these?

<details>
<summary><strong>Reveal Answer</strong></summary>

These are called **Magic Commands**.

A single `%` is generally a **line magic** and applies to one line.

For example:

```python
%time x = 5 + 5
```

A `%%` command is a **cell magic** and applies to the entire cell.

For example:

```python
%%time
x = 5 + 5
y = x * 10
```

Magic commands provide additional functionality for notebooks and Colab.

</details>

---

### 15. Why is the `hue` argument useful in Seaborn EDA?

<details>
<summary><strong>Reveal Answer</strong></summary>

The `hue` argument allows us to visually divide observations according to a categorical variable.

For example:

```python
sns.histplot(
    data=df_titanic,
    x="age",
    hue="sex"
)
```

allows us to compare the age distributions of different sex categories.

Similarly:

```python
sns.barplot(
    data=df_titanic,
    x="class",
    y="survived",
    hue="sex"
)
```

allows us to examine survival patterns across both passenger class and sex.

This is useful because EDA is about discovering patterns and relationships in data.

</details>

---

# 16. Final Reflection

Before finishing the activity, answer the following questions in your own words.

### Question 1

Why is it dangerous to assume that every dataset needs the same cleaning procedure?

### Question 2

Why is an outlier not automatically an error?

### Question 3

Why do we perform EDA both before and after important cleaning operations?

### Question 4

What is the difference between:

**finding a relationship in EDA**

and

**building a predictive model**?

### Question 5

What information did the Titanic dataset reveal that would not have been obvious from simply looking at the column names?

---

# 17. Summary: What You Practiced

In this activity, you practiced the complete **data-focused workflow** introduced in the lecture.

## 1. Understanding the Data

You learned to:

* inspect rows and columns;
* identify features and potential targets;
* distinguish numerical, categorical, and ordinal variables;
* understand the meaning of variables.

## 2. Initial Inspection and EDA

You practiced:

* summary statistics;
* value counts;
* histograms;
* boxplots;
* bar plots;
* scatter plots;
* correlation matrices;
* heatmaps;
* pairplots.

You also practiced the three main EDA perspectives:

* **Univariate**
* **Bivariate**
* **Multivariate**

## 3. Identifying Data-Quality Problems

You investigated:

* missing values;
* data types;
* duplicates;
* categorical values;
* unusual numerical values;
* potential outliers.

## 4. Data Cleaning

You practiced:

* median imputation;
* dropping a variable with extensive missingness;
* checking data types;
* verifying cleaning operations.

## 5. EDA Again

You repeated your analysis after cleaning to see:

* whether the changes worked;
* how cleaning affected distributions;
* whether important relationships remained;
* whether additional issues were present.

This demonstrated the key principle:

> **EDA and cleaning are iterative.**

## 6. Feature Engineering

You practiced:

* creating `family_size`;
* one-hot encoding a categorical variable;
* recognizing false ordinality.

## 7. Preparing for the Next Stage

You learned that data may eventually need:

* feature selection;
* encoding;
* scaling;
* train/test separation;
* additional preprocessing.

These topics become particularly important when building supervised-learning models.

> [!NOTE]
> **Next Lecture: Supervised Learning**
>
> In the next lecture, we will move from understanding and preparing data to using data to train predictive models.
>
> We will examine:
>
> * what supervised learning is;
> * training and test data;
> * classification and regression;
> * model training;
> * evaluation;
> * overfitting;
> * cross-validation;
> * data leakage;
> * and different supervised-learning algorithms.
