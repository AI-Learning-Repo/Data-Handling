# Activity: Understanding, Exploring, and Preparing Data (NYC Taxi Dataset)

In this activity, you will investigate two contrasting datasets:

* **Taxis** - a real-world dataset containing missing values, categorical variables, extreme outliers, date/time features, and redundant columns.
* **Iris** - a small, clean academic dataset that allows us to see what happens when data-quality problems are already absent.

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
> During the activity, you will first explore the data, discover problems, clean or transform the data, and then explore it again to determine whether the changes improved the dataset or revealed additional issues.

> [!IMPORTANT]
> **If you use an LLM as a learning assistant:**
>
> Do not simply copy code without understanding what it does. Question every line of code.

---

# Setup: Environment Preparation

## 1. Open Google Colab
Open [Google Colab](https://colab.research.google.com/) and create a **New Notebook**.

## 2. Install / Import Required Libraries
Create a new code cell and run:

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set plotting visual theme
sns.set_theme(style="whitegrid")
```

### Libraries used
* `pandas`: For manipulating tabular, structured data.
* `seaborn`: For high-level statistical data visualization.
* `matplotlib.pyplot`: For plot adjustments and customization.
* `numpy`: For numerical operations.

---

# Part 1: The NYC Taxis Dataset

The `taxis` dataset contains trip records from New York City yellow and green taxis in March 2019.

Unlike a clean academic dataset, it exhibits real-world tabular characteristics:
* missing values in categorical and location fields;
* right-skewed numerical variables (`fare`, `distance`, `tip`);
* nominal categorical features (`payment`, `color`, boroughs);
* extreme values / outliers;
* redundant or high-cardinality text columns.

The goal of this activity is **not yet to build a predictive model**, but to practice the end-to-end data preparation workflow.

---

# 1. Understanding the Dataset

## Your Task is to write Python code that:
1. Loads the `taxis` dataset from Seaborn.
2. Displays the first 5 rows.
3. Displays the dataset dimensions (`shape`).
4. Displays the dataset structure and data types (`info()`).
5. Displays numerical summary statistics (`describe()`).

### Suggested prompt for Gemini / LLM:
> "Write Python code for Google Colab. Load the 'taxis' dataset using Seaborn. Display the first five rows, the shape of the dataset, the column information using `.info()`, and numerical summary statistics using `.describe()`. Explain what each command reveals."

---

## Working Code

<details>
<summary><strong>Click here for the working code and explanations</strong></summary>

```python
# Load the NYC Taxis dataset
df_taxis = sns.load_dataset("taxis")

# 1. First few observations
print("--- First 5 Rows ---")
display(df_taxis.head())

# 2. Dimensions of the dataset
print(f"\nDataset Shape: {df_taxis.shape[0]} rows, {df_taxis.shape[1]} columns")

# 3. Column information and data types
print("\n--- Column Info & Non-Null Counts ---")
df_taxis.info()

# 4. Numerical summary statistics
print("\n--- Summary Statistics ---")
display(df_taxis.describe())
```

### What should you notice?
* The dataset contains **6,433 rows** and **14 columns**.
* Several columns contain missing values (e.g., `payment`, `pickup_borough`, `dropoff_borough`).
* Minimum `distance` and `fare` values are zero (or near zero), while maximum values are substantially higher than the mean.

</details>

---

## Understanding the Schema

Review the columns and classify them into their analytical feature types:

| Variable | Type | Description |
| :--- | :--- | :--- |
| `pickup` | Datetime / Object | Timestamp when the trip started |
| `dropoff` | Datetime / Object | Timestamp when the trip ended |
| `passengers` | Numerical (Discrete) | Number of passengers in the vehicle |
| `distance` | Numerical (Continuous) | Trip distance in miles |
| `fare` | Numerical (Continuous) | Base meter fare in dollars |
| `tip` | Numerical (Continuous) | Tip amount in dollars |
| `tolls` | Numerical (Continuous) | Toll fees paid |
| `total` | Numerical (Continuous) | Total charged amount (fare + tip + tolls + extra) |
| `color` | Categorical (Nominal) | Taxi fleet color (`yellow` or `green`) |
| `payment` | Categorical (Nominal) | Payment method (`credit card` or `cash`) |
| `pickup_zone` | Categorical / Text | Specific pickup neighborhood (High cardinality) |
| `dropoff_zone` | Categorical / Text | Specific dropoff neighborhood (High cardinality) |
| `pickup_borough`| Categorical (Nominal) | Broader NYC borough of pickup |
| `dropoff_borough`| Categorical (Nominal) | Broader NYC borough of dropoff |

### Think About It
1. If we wanted to predict how much a passenger will tip, which column is the **target**? (`tip`)
2. What if we wanted to predict whether a ride was paid by card or cash? What would the **target** be? (`payment`)
3. Which columns are continuous numerical measurements?
4. Which columns are unordered (nominal) categories?

---

# 2. Initial Data Inspection and EDA

Now we begin our first EDA pass across three dimensions:
* **Univariate:** one variable at a time.
* **Bivariate:** relationships between two variables.
* **Multivariate:** interactions among three or more variables.

---

## 2.1 Univariate EDA: Numerical Variables

### A. Numerical Summary
Examine the `.describe()` output specifically for `distance`, `fare`, and `tip`.

```python
display(df_taxis[['distance', 'fare', 'tip', 'passengers']].describe())
```

### B. Distance Distribution
Create a histogram with KDE for `distance`.

<details>
<summary><strong>Click here for working code</strong></summary>

```python
plt.figure(figsize=(9, 4))
sns.histplot(df_taxis['distance'], kde=True, color='teal', bins=40)
plt.title('Univariate Distribution: Trip Distance (Miles)')
plt.xlabel('Distance (Miles)')
plt.ylabel('Count')
plt.show()
```

### Questions:
1. Is the distribution of `distance` symmetric, or is it right-skewed?
2. Are most trips short or long?
</details>

---

## 2.2 Univariate EDA: Categorical Variables

Use `.value_counts()` to inspect categorical distributions.

```python
print("--- Taxi Color Distribution ---")
display(df_taxis['color'].value_counts())

print("\n--- Payment Method Distribution ---")
display(df_taxis['payment'].value_counts(dropna=False))

print("\n--- Pickup Borough Distribution ---")
display(df_taxis['pickup_borough'].value_counts(dropna=False))
```

### Questions:
1. Are yellow taxis or green taxis more common in this sample?
2. How many trips have a missing (`NaN`) payment method?
3. Which NYC borough generates the most pickups?

---

## 2.3 Bivariate EDA: Two Variables

### A. Distance vs. Fare (Numerical vs. Numerical)
Explore how fare scales with trip distance using a scatter plot.

<details>
<summary><strong>Click here for working code</strong></summary>

```python
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_taxis, x='distance', y='fare', alpha=0.5, color='darkblue')
plt.title('Bivariate Relationship: Trip Distance vs. Base Fare')
plt.xlabel('Distance (Miles)')
plt.ylabel('Fare ($)')
plt.show()
```

### Questions:
1. Does fare generally increase as distance increases?
2. Do you notice flat horizontal lines or vertical clusters (such as fixed airport flat-rate fares)?
</details>

---

### B. Average Tip by Payment Method (Categorical vs. Numerical)
Create a bar plot comparing recorded tips across payment methods.

<details>
<summary><strong>Click here for working code</strong></summary>

```python
plt.figure(figsize=(6, 4))
sns.barplot(data=df_taxis, x='payment', y='tip', errorbar=None, palette='Set2')
plt.title('Mean Tip Recorded by Payment Method')
plt.xlabel('Payment Method')
plt.ylabel('Average Tip ($)')
plt.show()
```

### Critical Thinking (Domain Context):
Why is the recorded cash tip approximately `$0.00`? In taxi payment systems, cash tips are almost never recorded in the meter, whereas credit card tips are logged electronically. This is an essential data collection nuance that EDA uncovers.

</details>

---

## 2.4 Bivariate EDA with a Third Dimension (`hue`)

Compare fare amounts across pickup boroughs separated by taxi fleet `color`.

<details>
<summary><strong>Click here for working code</strong></summary>

```python
plt.figure(figsize=(10, 5))
sns.barplot(
    data=df_taxis, 
    x='pickup_borough', 
    y='fare', 
    hue='color', 
    errorbar=None, 
    palette=['gold', 'limegreen']
)
plt.title('Average Fare across Boroughs by Taxi Fleet Color')
plt.xlabel('Pickup Borough')
plt.ylabel('Average Fare ($)')
plt.show()
```

### Questions:
1. In which boroughs do green cabs operate most frequently compared to yellow cabs?
2. Does the average fare differ noticeably by fleet type?
</details>

---

## 2.5 Multivariate EDA: Correlation Matrix

Calculate and visualize the pairwise Pearson correlation across all numerical variables.

<details>
<summary><strong>Click here for working code</strong></summary>

```python
# 1. Select numeric columns
numeric_taxis = df_taxis.select_dtypes(include='number')

# 2. Compute correlation matrix
corr_matrix = numeric_taxis.corr()

# 3. Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Numerical Taxi Features')
plt.show()
```

### Questions:
1. Which pairs of features exhibit strong positive linear correlation (close to $+1.0$)?
2. Does `passengers` strongly correlate with `fare` or `distance`?
3. Does high correlation between `distance` and `fare` mean distance causes higher fare? (Remember: *Correlation does not prove causation*).
</details>

---

## 2.6 Investigating Potential Outliers

Create a boxplot of the `fare` column to inspect extreme observations.

<details>
<summary><strong>Click here for working code</strong></summary>

```python
plt.figure(figsize=(8, 3))
sns.boxplot(x=df_taxis['fare'], color='coral')
plt.title('Boxplot of Taxi Fare (Outlier Detection)')
plt.xlabel('Fare ($)')
plt.show()
```

### Think Before Cleaning:
You will see fares exceeding `$100`. 
* Is an expensive trip to an outer airport or long-distance suburb impossible? **No.**
* Do not delete an observation simply because it is unusual. 
* Rule: **Detect $\rightarrow$ Investigate $\rightarrow$ Decide $\rightarrow$ Treat if necessary**.

</details>

---

# 3. Identify Data-Quality Problems

Consolidate the issues discovered in the first EDA pass:

```python
# 1. Missing values
print("--- Missing Values ---")
display(df_taxis.isnull().sum()[df_taxis.isnull().sum() > 0])

# 2. Duplicate rows
print(f"\nDuplicate Rows: {df_taxis.duplicated().sum()}")

# 3. Check minimum values for potential invalid entries (e.g. 0 distance or 0 passengers)
print("\n--- Minimum Values Check ---")
display(df_taxis[['distance', 'fare', 'passengers']].min())
```

### Complete the Data-Quality Summary:

| Data-Quality Question | What did you discover in `taxis`? |
| :--- | :--- |
| **Missing values** | `payment` (44 missing), boroughs/zones (~26–31 missing). |
| **Duplicate records** | Check if any fully duplicate trip records exist (`.duplicated().sum()`). |
| **Unusual/Suspicious values** | Zero-distance trips with positive fares; zero-passenger trips. |
| **High cardinality features** | `pickup_zone` and `dropoff_zone` contain hundreds of unique strings. |
| **Skewed distributions** | `distance`, `fare`, and `tip` are heavily right-skewed. |

---

# 4. Data Cleaning

Now we make informed cleaning decisions based on our EDA findings.

---

## 4.1 Handling Missing Values

1. **`payment`:** Impute missing categorical values with the **mode** (`credit card`).
2. **Missing Boroughs:** For records where `pickup_borough` or `dropoff_borough` is missing, drop those specific rows (~30 rows out of 6,433 is < 0.5%).

<details>
<summary><strong>Click here for working code</strong></summary>

```python
# 1. Impute missing 'payment' with mode
mode_payment = df_taxis['payment'].mode()[0]
df_taxis['payment'] = df_taxis['payment'].fillna(mode_payment)

# 2. Drop rows with missing borough values
df_taxis.dropna(subset=['pickup_borough', 'dropoff_borough'], inplace=True)

# 3. Verify missing counts
print("Missing values after cleaning:")
display(df_taxis[['payment', 'pickup_borough', 'dropoff_borough']].isnull().sum())
```
</details>

---

## 4.2 Handling Zero-Distance / Zero-Passenger Anomalies

Investigate trips where `distance == 0` but `fare > 0`. These often represent cancelled rides after pickup or meter initialization errors.

```python
# Filter and inspect zero-distance trips with fare
zero_dist_count = (df_taxis['distance'] == 0).sum()
print(f"Number of zero-distance trips: {zero_dist_count}")

# For this demo, remove zero-distance anomalies to maintain physical relationship integrity
df_taxis = df_taxis[df_taxis['distance'] > 0].copy()
```

---

# 5. EDA Again: Validate the Cleaning

We modified the dataset. Therefore, we **must explore it again** to verify our transformations did not corrupt distributions or relationships.

---

## 5.1 Check the Payment Distribution Again

<details>
<summary><strong>Click here for working code</strong></summary>

```python
plt.figure(figsize=(6, 4))
sns.countplot(data=df_taxis, x='payment', palette='Set2')
plt.title('Payment Distribution After Mode Imputation (No Missing Values)')
plt.show()
```
</details>

---

## 5.2 Check Distance vs. Fare Again

<details>
<summary><strong>Click here for working code</strong></summary>

```python
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_taxis, x='distance', y='fare', alpha=0.5, color='teal')
plt.title('Distance vs. Fare After Removing Zero-Distance Entries')
plt.xlabel('Distance (Miles)')
plt.ylabel('Fare ($)')
plt.show()
```

### Questions:
* Did removing `distance == 0` records remove the unnatural vertical stack on the y-axis at $x=0$?
* Did the overall positive trend between distance and fare become clearer?

</details>

---

# 6. Feature Engineering & Preprocessing

Feature engineering transforms raw columns into more informative numerical inputs for machine learning.

---

## 6.1 Creating New Features: Tip Percentage and Trip Duration

1. **`tip_pct`:** Ratio of tip to base fare ($\frac{\text{tip}}{\text{fare}} \times 100$).
2. **`duration_min`:** Trip duration calculated from timestamps (`dropoff - pickup`) in minutes.

<details>
<summary><strong>Click here for working code</strong></summary>

```python
# 1. Calculate tip percentage
df_taxis['tip_pct'] = (df_taxis['tip'] / df_taxis['fare']) * 100

# 2. Convert timestamps to datetime and calculate duration in minutes
df_taxis['pickup'] = pd.to_datetime(df_taxis['pickup'])
df_taxis['dropoff'] = pd.to_datetime(df_taxis['dropoff'])
df_taxis['duration_min'] = (df_taxis['dropoff'] - df_taxis['pickup']).dt.total_seconds() / 60.0

# Preview new engineered features
display(df_taxis[['distance', 'fare', 'tip', 'tip_pct', 'duration_min']].head())
```
</details>

---

## 6.2 Encoding Categorical Variables (One-Hot Encoding)

Machine learning models require numeric inputs. 
* We apply **One-Hot Encoding** to nominal features (`color`, `payment`, `pickup_borough`).
* We set `drop_first=True` to avoid the **dummy variable trap** (multicollinearity).
* We set `dtype=int` to ensure columns are encoded as `0` and `1` instead of `True` and `False`.

<details>
<summary><strong>Click here for working code</strong></summary>

```python
# One-hot encode categorical predictors
df_taxis_encoded = pd.get_dummies(
    df_taxis, 
    columns=['color', 'payment', 'pickup_borough'], 
    drop_first=True, 
    dtype=int
)

# Preview encoded columns
display(df_taxis_encoded.head())
```

### Why not encode `yellow = 1` and `green = 2`?
Because that would imply $2 > 1$ (green is greater than yellow). This artificial numerical hierarchy is called **false ordinality**. One-hot encoding creates independent binary indicators instead.

</details>

---

# 7. Dropping Redundant Columns & Final Preparation

Drop high-cardinality neighborhood names and raw timestamp columns that have already been converted into features.

```python
# Define redundant/high-cardinality columns
cols_to_drop = ['pickup', 'dropoff', 'pickup_zone', 'dropoff_zone', 'dropoff_borough']

# Create final prepared DataFrame
df_prepared = df_taxis_encoded.drop(columns=cols_to_drop)

print("--- Final Prepared Dataset ---")
df_prepared.info()
```

---

# 8. Exporting the Data

Export the clean dataset to CSV for future modeling:

```python
# Export clean dataset to CSV
df_prepared.to_csv("taxis_prepared.csv", index=False)
print("Successfully exported 'taxis_prepared.csv'")
```

---

# Part 2: The Iris Dataset (Clean Contrast)

> This is the same dataset used in activity 1

The Iris dataset contains flower petal and sepal measurements. It is an academic dataset that contrasts with real-world data because it is already clean.

---

# 9. Understanding and Inspecting Iris

```python
# Load Iris dataset
df_iris = sns.load_dataset("iris")

# Display structure and missing counts
display(df_iris.head())
print("Shape:", df_iris.shape)
print("\nMissing values in Iris:")
display(df_iris.isnull().sum())
```

### What should you notice?
* **Zero missing values.**
* No high-cardinality text columns or missing payment labels.
* Demonstrates the rule: **Inspect the dataset first; do not blindly apply cleaning steps if the data does not require it.**

---

# 10. Iris: Multivariate EDA with Pairplot

Explore feature interactions across species using `sns.pairplot()`.

<details>
<summary><strong>Click here for working code</strong></summary>

```python
sns.pairplot(df_iris, hue="species", palette="Set1")
plt.suptitle("Pairwise Feature Relationships in Iris", y=1.02)
plt.show()
```

### Key Takeaway:
Petal measurements (`petal_length`, `petal_width`) cleanly separate the three iris species. This is an EDA finding showing which features carry strong predictive signals.

</details>

---

# 11. Comparing Taxis and Iris

Complete the comparison table:

| Characteristic | NYC Taxis | Iris |
| :--- | :--- | :--- |
| **Real-world observational data** | Yes (messy, operational) | Academic benchmark |
| **Missing values present?** | Yes (`payment`, boroughs) | No (0 missing) |
| **Extreme outliers present?** | Yes (high fares, long distances) | Minimal |
| **Requires feature engineering?** | Yes (`duration_min`, `tip_pct`) | No |
| **Requires categorical encoding?** | Yes (`color`, `payment`, boroughs) | Only if modeling species |
| **Iterative EDA necessary?** | Yes (validate post-cleaning) | Minimal |

---

# 12. Knowledge Check

Test your understanding before revealing the answers.

---

### 1. Why didn't we delete all taxi fares greater than $100?
<details>
<summary><strong>Reveal Answer</strong></summary>

An outlier is not automatically an error. Fares over $100 can represent valid trips to distant destinations or out-of-city airports. We only remove observations when there is clear evidence of recording or physical impossibility. The correct workflow is: **Detect $\rightarrow$ Investigate $\rightarrow$ Decide $\rightarrow$ Treat if necessary**.
</details>

---

### 2. Why did we use `drop_first=True` when running `pd.get_dummies` on `color` and `payment`?
<details>
<summary><strong>Reveal Answer</strong></summary>

To prevent the **Dummy Variable Trap** (perfect multicollinearity). For a binary variable like `color` (`yellow` vs. `green`), knowing that `color_yellow == 0` mathematically guarantees the taxi is green. Keeping both columns introduces redundant information.
</details>

---

### 3. What is false ordinality?
<details>
<summary><strong>Reveal Answer</strong></summary>

False ordinality occurs when distinct, unordered nominal categories are assigned numbers (e.g., `Manhattan = 1, Queens = 2, Brooklyn = 3`) that lead an algorithm to treat them as an ordered mathematical scale ($3 > 2 > 1$). One-hot encoding prevents this.
</details>

---

### 4. Why is EDA considered iterative?
<details>
<summary><strong>Reveal Answer</strong></summary>

Because cleaning operations (like imputing missing values or filtering zero-distance records) alter the underlying distributions and relationships. We must perform EDA again after cleaning to verify that no artificial distortion was introduced.
</details>

---

### 5. What does the GIGO principle mean in Machine Learning?
<details>
<summary><strong>Reveal Answer</strong></summary>

**Garbage In, Garbage Out**. If an algorithm is trained on incomplete, biased, duplicate, or miscoded data, the resulting model will make unreliable predictions, regardless of algorithm sophistication.
</details>

---

# 13. Final Reflection Questions

1. Why was it necessary to convert `pickup` and `dropoff` strings into `datetime` before calculating trip duration?
2. Why is cash tip recorded as $0.00$ in the dataset, and what does this teach us about understanding how data is collected?
3. How does comparing `taxis` with `iris` illustrate why we should never apply a single, generic cleaning script to all datasets?