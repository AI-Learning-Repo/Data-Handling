# Part 1: Data and Datasets for Classical Machine Learning

## Introduction

### The Role of Data in Machine Learning

In classical Machine Learning (ML), **data is the foundational building block of model performance**. Unlike traditional programming—where a human explicitly writes `if/then` rules—ML models rely on mathematical algorithms to learn patterns from data and use those patterns to make predictions.

If the data is messy, biased, incomplete, or incorrectly represented, the quality of the resulting analysis or model will suffer.

> [!WARNING]
> **The GIGO Principle**
>
> In data science, a fundamental rule is **Garbage In, Garbage Out (GIGO)**. Regardless of how sophisticated an algorithm is, poor-quality input data can lead to poor-quality results.

### Classical ML vs. Modern AI

Both classical ML and modern AI systems rely heavily on data, but the type and structure of their data can differ significantly:

* **Classical ML** typically works with **structured datasets**, such as databases or spreadsheets organized into rows and columns with predefined variables.
* **Modern AI**, including Large Language Models (LLMs), can work with massive **unstructured datasets**, such as text documents, images, audio, and video.

This lecture focuses primarily on **structured, tabular data**, because understanding how to inspect, explore, clean, and prepare such data is fundamental to classical machine learning.

> [!NOTE]
> **The Data Workflow**
>
> Before data can be used effectively, we need to understand it, inspect it, explore its patterns and quality, and clean or transform it when necessary.
>
> Importantly, **EDA and data cleaning are iterative**. We may discover a problem during EDA, clean or transform the data, and then perform EDA again to verify the result.
>
> Detailed supervised-learning algorithms, model training, evaluation, overfitting, and cross-validation will be covered in the **next lecture on Supervised Learning**.

---

# 1. Understanding the Dataset

Before manipulating or analyzing data, it is necessary to understand what the dataset contains, what each variable represents, and what question the data is intended to answer.

### Dataset Schema & Terminology

A structured dataset operates like a spreadsheet.

* **Features (Independent Variables):** The input variables that describe an observation. Examples include `Age`, `Income`, and `Credit_Score`.
* **Target (Dependent Variable):** The outcome or label that we may ultimately want to predict in a supervised-learning problem.

At this stage, the important goal is not to build a model. The goal is to **understand the data and its meaning**.

### Types of Features

Different types of variables require different forms of analysis and, later, different forms of preparation.

* **Numerical Features:** Measurable values such as income, age, temperature, or ticket price.
* **Categorical Features:** Categories without an inherent numerical meaning, such as country, job title, or gender.
* **Ordinal Features:** Categorical variables with a meaningful order, such as:

  `Poor < Fair < Good < Excellent`

Recognizing these types is an important first step in data analysis.

### Contextual Example: The Titanic Survival Dataset

Throughout this material, we use the **Titanic dataset** as a running example.

The objective is to understand the data and, later, use it in a supervised-learning problem to predict passenger survival.

* **Target:** `Survived`

  * `1` = survived
  * `0` = did not survive

* **Features:**

  * `Pclass`: Passenger class — ordinal
  * `Age`: Passenger age — numerical
  * `Sex`: Gender — categorical
  * `Fare`: Ticket price — numerical
  * `Cabin`: Cabin/room information — categorical
  * `SibSp`: Number of siblings/spouses aboard — numerical
  * `Parch`: Number of parents/children aboard — numerical

At this point, we are primarily interested in **what these variables mean and what kinds of data they contain**.

---

# 2. Initial Data Inspection

Before performing detailed EDA (Exploratory Data Analysis), we should inspect the dataset and establish a basic understanding of its structure and quality.

Typical questions include:

* How many rows and columns are there?
* What does each row represent?
* What does each column represent?
* What are the data types?
* Are there missing values?
* Are there duplicate records?
* Are there impossible or suspicious values?
* Are categorical values consistently represented?
* Are numerical variables within reasonable ranges?

In Python, Pandas provides functions that make this initial inspection straightforward.

For example:

```python
df.head()
df.shape
df.info()
df.describe()
df.isnull().sum()
```

These commands do not replace EDA, but they provide the initial information needed to begin understanding the dataset.

---

# 3. Exploratory Data Analysis (EDA)

## What Is EDA?

**Exploratory Data Analysis (EDA)** is the process of investigating, summarizing, and visualizing data to understand its structure, distributions, relationships, patterns, and anomalies.

EDA helps us:

* understand the characteristics of the dataset;
* identify missing or suspicious values;
* examine distributions;
* detect potential outliers;
* discover relationships between variables;
* formulate questions or hypotheses;
* identify potentially useful variables;
* communicate important patterns in the data.

EDA was strongly promoted by statistician **John Tukey**, who emphasized exploring data before relying on formal statistical models or assumptions.

The important point is that EDA is **exploratory**. We are not simply calculating numbers; we are trying to understand what the data is telling us.

[GeeksforGeeks: Exploratory Data Analysis](https://www.geeksforgeeks.org/data-analysis/what-is-exploratory-data-analysis/)

[GeeksforGeeks: Exploratory Data Analysis — Types and Tools](https://www.geeksforgeeks.org/machine-learning/exploratory-data-analysis-eda-types-and-tools/)

---

## EDA and Data Cleaning Are Iterative

It is tempting to think of the workflow as:

**Clean → EDA**

or:

**EDA → Clean**

In practice, neither description is completely accurate.

A more useful view is:

```text
Understand the data
        ↓
Initial inspection / EDA
        ↓
Identify problems and questions
        ↓
Clean or transform data
        ↓
EDA again
        ↓
Identify additional problems or patterns
        ↓
Clean / transform again if necessary
        ↓
Continue analysis
```

For example:

1. We inspect the `Age` variable.
2. We discover many missing values.
3. We investigate the missingness.
4. We decide how the missing values should be handled.
5. We examine the resulting distribution again.
6. We may discover additional issues or patterns.

Therefore:

> **EDA and data cleaning should be viewed as an iterative process rather than two completely separate steps.**

This is particularly important because cleaning can change the distribution and relationships in the data.

---

# 4. EDA Taxonomy

EDA can be classified according to the **number of variables** being examined and according to whether the analysis is **graphical or non-graphical**.

The three main categories are:

1. **Univariate Analysis**: one variable
2. **Bivariate Analysis**: two variables
3. **Multivariate Analysis**: three or more variables

We can also distinguish between:

* **Non-graphical EDA:** numerical summaries and tables
* **Graphical EDA:** charts and visualizations

---

## 4.1 Univariate Analysis

**Univariate analysis** examines one variable at a time.

The goal is to understand characteristics such as:

* central tendency;
* spread;
* distribution;
* skewness;
* possible outliers;
* frequency of categories.

### Non-Graphical Analysis

Functions such as:

```python
df.describe()
```

provide a mathematical summary of numerical variables.

Important measures include:

* **Mean:** Average value.
* **Median:** Middle value when observations are ordered.
* **Minimum and maximum:** Range of observed values.
* **Standard deviation:** A measure of spread.
* **Quartiles:** Values dividing the data into sections.
* **Skewness:** A measure of asymmetry in a distribution.
* **Kurtosis:** A measure related to the shape or tail behavior of a distribution.

For skewed data or data containing extreme values, the **median** can sometimes provide a more representative measure of central tendency than the mean.

### Graphical Analysis

Common univariate visualizations include:

* **Histograms:** Show the distribution and frequency of numerical values.
* **Boxplots:** Show central tendency and spread and can help identify potential outliers.
* **Bar charts:** Useful for categorical variables.

### Titanic Example

A histogram of `Age` can show whether passengers were concentrated in particular age ranges.

A boxplot of `Fare` can reveal that most passengers paid relatively moderate fares while a small number of observations had much larger values.

---

## 4.2 Bivariate Analysis

**Bivariate analysis** examines the relationship between two variables.

Typical questions include:

* Does one numerical variable increase when another increases?
* Does a categorical variable differ across groups?
* How does a feature relate to the target?
* Are two variables strongly associated?

### Common Techniques

* **Scatter plots:** Useful for two numerical variables.
* **Bar charts:** Useful for comparing categories or group-level statistics.
* **Cross-tabulation:** Useful for examining relationships between categorical variables.
* **Correlation:** Measures the strength and direction of certain relationships between numerical variables.
* **Line graphs:** Useful when one variable represents time.

### Titanic Example

We can compare survival rates by `Sex`.

A visualization may reveal that survival rates differed substantially between female and male passengers.

This is an important EDA finding because it suggests that `Sex` may contain useful information for a future predictive model.

At this stage, however, our goal is **to understand and describe the data**, not yet to build the model.

---

## 4.3 Multivariate Analysis

**Multivariate analysis** examines three or more variables simultaneously.

This allows us to investigate more complex relationships.

Common techniques include:

* Correlation matrices
* Heatmaps
* Pair plots
* Grouped visualizations
* Multivariate charts

### Correlation Matrix

A **correlation matrix** summarizes pairwise correlations between numerical variables.

Correlation coefficients are commonly represented on a scale from approximately:

`-1.0 → 0 → +1.0`

A positive value indicates that two variables tend to increase together, while a negative value indicates that one tends to decrease as the other increases.

A value close to zero indicates little linear relationship.

> **Important:** Correlation does not imply causation.

### Titanic Example

`Fare` and `Pclass` may show a strong relationship because passenger class is associated with ticket price.

EDA can therefore reveal that some variables contain overlapping information.

Detailed decisions about feature selection and model design will be covered in the **Supervised Learning lecture**.

---

# 5. Data Quality and Cleaning

Real-world data is rarely perfect.

During initial inspection and EDA, we may discover:

* missing values;
* incorrect data types;
* duplicate records;
* inconsistent categories;
* impossible values;
* unusual observations;
* outliers;
* measurement or recording errors.

Cleaning is the process of identifying and appropriately handling these data-quality problems.

> **Important:** Cleaning does not mean automatically deleting everything unusual. The correct action depends on what the data represents.

---

## 5.1 Handling Missing Data

Missing data is common in real-world datasets.

### Imputation

**Imputation** means replacing a missing value with an estimated value.

Possible approaches include:

* mean;
* median;
* mode;
* more advanced statistical methods.

For example, if `Age` contains missing values, we may use the median age as an estimate.

### Dropping Variables

If a variable contains an overwhelming amount of missing information and provides insufficient useful information, it may be appropriate to remove the variable.

### Titanic Example

In the Titanic dataset:

* `Age` contains missing values.
* `Cabin` contains a very large proportion of missing values.

These observations require investigation before deciding what to do.

The decision should consider:

* how much data is missing;
* why it is missing;
* whether the missingness is systematic;
* how important the variable is;
* whether removing observations or variables could introduce bias.

---

## 5.2 Incorrect Data Types

A value such as:

```text
20
```

may be stored incorrectly as the string:

```text
"20"
```

The value looks numerical to a human, but the computer may treat it as text.

Therefore, data types should be checked and corrected when necessary.

Examples include:

* converting strings to numerical values;
* converting dates into appropriate date/time types;
* converting categorical variables into appropriate categorical representations.

---

## 5.3 Duplicates and Inconsistent Values

Data can contain duplicate observations or inconsistent representations of the same category.

For example:

```text
Finland
finland
FINLAND
```

may represent the same category.

Before analyzing categorical variables, we should determine whether such differences are meaningful or simply data-entry inconsistencies.

---

## 5.4 Invalid or Impossible Values

Some observations may be technically present but logically impossible.

Examples might include:

* a negative age;
* an impossible date;
* a temperature outside the physically possible range for the context;
* a category that does not belong to the defined set of categories.

These should be investigated rather than automatically deleted.

---

# 6. Outliers and Anomalies

An **outlier** is an observation that differs substantially from other observations.

Outliers can occur because of:

* measurement errors;
* data-entry errors;
* unusual but genuine events;
* rare populations;
* natural variability.

### Titanic Example

Most Titanic passengers paid relatively moderate fares, but one passenger paid approximately `$512`.

This observation is unusual.

However:

> **An unusual observation is not automatically an incorrect observation.**

We should investigate it before deciding what to do.

### Detecting Outliers

Possible techniques include:

* Boxplots
* Interquartile Range (IQR)
* Z-scores
* Distribution plots
* Domain knowledge

### Possible Actions

Depending on the context, we may:

* keep the observation;
* correct it if it is clearly an error;
* remove it if it is demonstrably invalid;
* cap the value;
* transform the variable.

### Logarithmic Transformation

A logarithmic transformation can compress a highly skewed numerical distribution.

For example:

```text
10   → log10(10)   = 1
100  → log10(100)  = 2
1000 → log10(1000) = 3
```

This reduces the numerical distance between very large values and smaller values.

> **Important:** A transformation should be motivated by the characteristics of the data and the purpose of the analysis. It should not be applied simply because an observation looks unusual.

---

# 7. Repeating EDA After Cleaning

Cleaning is not necessarily the end of the process.

After modifying the data, we should examine it again.

For example:

* Did the missing values get handled correctly?
* Did the distribution change substantially?
* Are there still suspicious values?
* Did the transformation reduce skewness?
* Did the relationships between variables change?
* Did cleaning accidentally introduce a new problem?

This is why EDA and cleaning should be viewed as an **iterative cycle**.

---

# 8. Feature Engineering and Transformation

Feature engineering means creating or transforming variables so that they represent useful information in a more suitable form.

This topic becomes particularly important when building predictive models, so here we introduce it only at a high level.

Detailed feature engineering and preprocessing techniques will be discussed further in the **Supervised Learning lecture**.

---

## 8.1 Handling Categorical Variables

Many machine-learning algorithms require numerical inputs.

Categorical variables therefore often need to be represented numerically.

### Label Encoding

Label encoding assigns numerical values to categories.

For example:

```text
1st Class → 1
2nd Class → 2
3rd Class → 3
```

This can be appropriate when the categories have a meaningful order.

### One-Hot Encoding

One-hot encoding creates separate binary variables for categories.

For example:

```text
Sex
----
Male
Female
```

can become:

```text
Is_Male
Is_Female
```

This avoids imposing an artificial numerical order on nominal categories.

### False Ordinality

Suppose we encode:

```text
Male   = 1
Female = 2
```

If `Male` and `Female` are simply categories, the numbers do not mean that Female is mathematically "greater" than Male.

This artificial ordering is called **false ordinality**.

One-hot encoding can avoid this problem.

### Dummy Variable Trap

In some statistical models, keeping every one-hot encoded category together with an intercept can create perfect multicollinearity.

This is commonly called the **dummy variable trap**.

The detailed implications of encoding and multicollinearity will be discussed later when supervised-learning models are introduced.

---

## 8.2 Creating New Features

Sometimes existing variables can be combined to create a more meaningful variable.

### Titanic Example

The Titanic dataset contains:

* `SibSp`: siblings/spouses
* `Parch`: parents/children

We can combine these variables to create a new variable such as:

```text
Family_Size
```

This is an example of **feature engineering**.

The purpose is to represent potentially useful information in a form that may be easier to analyze or use later.

---

## 8.3 Scaling Numerical Features

Different numerical variables may have very different ranges.

For example:

```text
Age  → 0–80
Fare → 0–500
```

Two common transformations are:

* **Min-Max Scaling (Normalization):** Transforms values into a specified range, commonly `[0, 1]`.
* **Standardization:** Transforms values so that the resulting variable has a mean of approximately `0` and standard deviation of approximately `1`.

Scaling is particularly important for some machine-learning algorithms.

The reasons for scaling, which algorithms require it, and how it affects supervised-learning models will be covered in the **next lecture**.

---

# 9. Data Leakage: An Important Preview

When preparing data for predictive modeling, we must be careful not to allow information from the data used for final evaluation to influence the training process.

This problem is called **data leakage**.

For example, suppose we calculate the median age using the entire dataset and then use that median to fill missing values.

If the dataset contains a future test set, information from that test set has influenced the preprocessing.

This can lead to overly optimistic evaluation results.

Therefore, when we reach supervised learning, preprocessing operations such as:

* imputation;
* scaling;
* encoding;
* feature selection;

must be handled carefully so that information from evaluation data does not leak into the training process.

> [!NOTE]
> **This will be covered in more detail next week.**
>
> The purpose here is simply to understand that data preparation is not independent of the modeling process. The exact train/test workflow and prevention of data leakage will be introduced with supervised learning.

---

# 10. Preparing Data for Supervised Learning — High-Level Overview

Once the data has been understood, explored, cleaned, and appropriately transformed, it may eventually be used for machine learning.

At a high level, a supervised-learning workflow involves:

1. Understanding the problem and data.
2. Exploring and cleaning the data.
3. Separating the target from the features.
4. Preparing the features appropriately.
5. Splitting data appropriately for model development and evaluation.
6. Training a model.
7. Evaluating its performance.

The details of steps 4–7 depend on the particular supervised-learning algorithm.

For example, different algorithms may have different requirements regarding:

* numerical scaling;
* categorical encoding;
* missing values;
* feature selection;
* class imbalance.

These topics will be developed in the **next lecture on Supervised Learning**.

---

# 11. Feature Selection and Class Imbalance — High-Level Preview

Two additional issues may become important when preparing data for predictive modeling.

## Feature Selection

A dataset may contain variables that provide little useful information.

For example, in the Titanic dataset, variables such as:

* `Passenger_Name`
* `Ticket_Number`

may not provide useful generalizable information in their raw form.

Feature selection is the process of deciding which variables should be retained for a particular analysis or model.

However, a variable should not be removed simply because it appears unimportant at first glance. Its usefulness depends on the analytical question and the modeling approach.

Detailed feature selection will be discussed later.

## Class Imbalance

Suppose a classification dataset contains:

```text
99% → Class A
1%  → Class B
```

A model could become heavily biased toward the majority class.

This situation is called **class imbalance**.

Possible approaches include:

* oversampling the minority class;
* undersampling the majority class;
* using algorithms or evaluation measures designed to handle imbalance.

These are supervised-learning topics and will be discussed in detail next week.

---

# 12. Train/Test Split, Overfitting and Cross-Validation — High-Level Preview

A predictive model should not simply be evaluated on the same observations used to train it.

If a model memorizes the training data but performs poorly on new observations, it is said to suffer from **overfitting**.

A common approach is to separate data into:

* **Training data:** Used to learn from the data.
* **Test data:** Reserved for evaluating performance on previously unseen observations.

A common illustrative split is:

```text
80% → Training
20% → Test
```

However, the exact split depends on the problem and dataset.

### Cross-Validation

**Cross-validation** provides another way to evaluate model performance by repeatedly dividing the available training data into different subsets.

For example, with 5-fold cross-validation, the training data is divided into five parts and the model is trained and evaluated across multiple configurations.

The goal is to obtain a more reliable estimate of model performance.

> **These concepts are introduced here only to show where data preparation fits into the larger ML workflow.**
>
> **Train/test splitting, overfitting, cross-validation, evaluation metrics, and supervised-learning algorithms will be covered in the next lecture.**

---

# 13. The Overall Data Workflow

For this lecture, the most important workflow to remember is:

```text
1. Understand the data
          ↓
2. Inspect the data
          ↓
3. Explore the data (EDA)
          ↓
4. Identify data-quality problems
          ↓
5. Clean / transform the data
          ↓
6. Explore again and validate
          ↓
7. Engineer useful representations
          ↓
8. Prepare the data for the next stage
```

The relationship between EDA and cleaning is **iterative**:

```text
        ┌─────────────────────┐
        │      Initial EDA    │
        └──────────┬──────────┘
                   ↓
        ┌─────────────────────┐
        │ Identify problems   │
        └──────────┬──────────┘
                   ↓
        ┌─────────────────────┐
        │ Clean / transform   │
        └──────────┬──────────┘
                   ↓
        ┌─────────────────────┐
        │     EDA again       │
        └──────────┬──────────┘
                   │
                   └──────→ repeat if necessary
```

The goal is not to perform a fixed sequence of mechanical operations.

The goal is to develop a **reliable understanding of the data**.

---

# 14. Key Takeaways

### Data

* Data is the foundation of machine learning and data analysis.
* Poor-quality data can lead to poor-quality results: **Garbage In, Garbage Out**.
* Before analyzing data, understand what each observation and variable represents.
* Distinguish between numerical, categorical, and ordinal variables.

### EDA

* **Exploratory Data Analysis (EDA)** is used to understand data before further analysis or modeling.
* EDA uses both numerical summaries and visualizations.
* **Univariate analysis** examines one variable.
* **Bivariate analysis** examines two variables.
* **Multivariate analysis** examines several variables.
* EDA can reveal distributions, relationships, anomalies, missing values, and potential outliers.

### Data Cleaning

* Missing values should be investigated rather than automatically deleted.
* Incorrect data types can lead to incorrect analysis.
* Duplicate, inconsistent, and impossible values should be identified and investigated.
* Outliers are not automatically errors.
* Domain knowledge is often necessary when deciding how to handle unusual observations.

### EDA and Cleaning

* EDA and data cleaning are **iterative**.
* Initial EDA can reveal data-quality problems.
* Cleaning can change the data.
* EDA should therefore be repeated after important cleaning or transformation steps.

### Machine Learning Preview

* Feature engineering transforms existing information into useful representations.
* Encoding and scaling are important for some machine-learning algorithms.
* Data leakage must be avoided when preparing data for predictive modeling.
* Train/test splitting, class imbalance, overfitting, cross-validation, and model evaluation are important—but they will be covered in detail in the **next lecture on Supervised Learning**.

---

# Further Exploration

### Data Analysis and EDA

[GeeksforGeeks — What is Exploratory Data Analysis?](https://www.geeksforgeeks.org/data-analysis/what-is-exploratory-data-analysis/)

[GeeksforGeeks — Exploratory Data Analysis: Types and Tools](https://www.geeksforgeeks.org/machine-learning/exploratory-data-analysis-eda-types-and-tools/)

### Pandas

- [Pandas: Crash Course — Kaggle Learn](https://www.kaggle.com/learn/pandas)
- https://colab.research.google.com/github/ageron/handson-ml3/blob/main/tools_pandas.ipynb 

### Data Visualization

[Data Visualization: Crash Course — Kaggle Learn](https://www.kaggle.com/learn/data-visualization)

### Videos

[Pandas for Data Science — Video (~50 min)](https://www.youtube.com/watch?v=Yp3fccNNfjQ)

[NumPy for Data Science — Video (~50 min)](https://www.youtube.com/watch?v=EmA_TuC2Vdk)

### Intro to Machine Learning with Python

* Part 1: Welcome and Project Setup
* Part 2: [Part 2: Exploratory Data Analysis](https://youtu.be/6BagRiSY1ds)
* Part 3: [Part 3: Train Test Split and Baseline Modeling](https://youtu.be/MufPx3L7nXM)


### Practical Tips

[Choosing Plot Types — Kaggle](https://www.kaggle.com/code/alexisbcook/choosing-plot-types-and-custom-styles)

[How to Handle Missing Values — Kaggle](https://www.kaggle.com/code/alexisbcook/missing-values)

> **Next lecture:** Supervised Learning — how cleaned and prepared data is used to train predictive models, including the train/test workflow, model types, evaluation, and overfitting.


### Introduction to Machine Learning


<!-- [Google — Introduction to Machine Learning: Supervised Learning](https://developers.google.com/machine-learning/intro-to-ml/supervised#data) -->
