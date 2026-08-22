# Activity 1: The Machine Learning Data Pipeline

In this activity, you will put the 5-step Classical Machine Learning pipeline into practice using two classic datasets: **Titanic** (messy, real-world data) and **Iris** (perfect, academic data). 

> [!IMPORTANT]  
> During this activity, you are encouraged to question every line of code. Use Gemini to generate code, explain errors, or clarify functions you do not understand.

### Setup: Environment Preparation
1. Open [Google Colab](https://colab.research.google.com/) and create a **New Notebook**.
2. Create a new code cell, paste the following command, and run it to install the required libraries.

```bash
!pip install pandas seaborn matplotlib -q
```

> [!NOTE]  
> **Libraries used:**
> *   `pandas`: The industry standard for data manipulation (acts like Python's version of Excel).
> *   `seaborn` & `matplotlib`: Used for creating graphs and visual EDA (Exploratory Data Analysis).

---

## Part 1: The Titanic Dataset (Real-World Data)

The Titanic dataset contains historical passenger information. Because it is based on real-world events, the data is incomplete and requires us to go through every step of the ML pipeline before it can be fed to an algorithm.

### Step 1: Understanding the Dataset
**Your Task:**
Use Gemini to write Python code to load the data and understand its schema. Import your libraries, load the `titanic` dataset from seaborn, print the `.info()`, and print the summary `.describe()`.

**Suggested prompt for Gemini:**
> *"Write Python code for Google Colab. Import pandas, seaborn, and matplotlib. Load the 'titanic' dataset using seaborn. Print the dataset info, and describe the summary statistics."*

<details>
<summary><strong>Click here for the working code and explanations</strong></summary>

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
df_titanic = sns.load_dataset("titanic")

# Check Dataset Info (Data types and nulls)
print("--- Dataset Info ---")
df_titanic.info()

# Summary Statistics (Math snapshot)
print("\n--- Summary Statistics ---")
display(df_titanic.describe())
```
**Expected Results:**
Look at the output of `.info()`. You will see 891 total rows, but columns like `age` only have 714 non-null values. The `deck` column only has 203! This tells us Step 2 (Cleaning) is mandatory.

> Note that Pandas provides a simple method to export a DataFrame.

```python
df_titanic.to_csv("titanic.csv", index=False)
```

- `"titanic.csv"` is the filename created in the current working directory.
- `index=False` prevents Pandas from writing row numbers as a separate column.


</details>

### Step 2: Data Cleaning (Imputation & Dropping)
**Your Task:**
An ML model will crash if it sees missing data. Ask Gemini to help you clean the dataset by filling in the missing ages and dropping the mostly empty `deck` column.

**Suggested prompt for Gemini:**
> *"In pandas, how do I check the sum of missing values in my titanic dataframe? Then, write code to drop the 'deck' column entirely, and fill the missing values in the 'age' column with the median age."*

<details>
<summary><strong>Click here for the working code and explanations</strong></summary>

```python
# 1. Check missing values before cleaning
print("Missing values BEFORE cleaning:\n", df_titanic.isnull().sum())

# 2. Drop the 'deck' column (Too much missing data to save)
df_titanic = df_titanic.drop(columns=['deck'])

# 3. Impute (fill) missing ages with the median
median_age = df_titanic['age'].median()
df_titanic['age'] = df_titanic['age'].fillna(median_age)

# 4. Verify cleaning
print("\nMissing values AFTER cleaning:\n", df_titanic.isnull().sum()[['age']])
```
**Python Concepts:**
*   **Imputation (`.fillna()`)**: We replace the blank ages with the median (middle) age. We use median instead of mean because extreme outliers (like a few 80-year-olds) can artificially drag the mean upward.
*   **Dropping (`.drop()`)**: Since ~77% of the `deck` column is missing, any guesses we make to fill it would be garbage. (Remember GIGO: Garbage In, Garbage Out).
</details>

### Step 3: Exploratory Data Analysis (EDA)
**Your Task:**
Now that the data is clean, we can explore it visually to find patterns. Create visualizations to see how age, gender, and passenger class affected survival.

**Suggested prompt for Gemini:**
> *"Using the clean titanic dataframe, create two seaborn plots. Plot 1: A histplot for the 'age' column with kde=True and hue='sex'. Plot 2: A barplot with 'class' on the x-axis, 'survived' on the y-axis, and hue='sex'. Add titles to both."*

<details>
<summary><strong>Click here for the working code and explanations</strong></summary>

```python
# Plot 1: Age Distribution by Gender
plt.figure(figsize=(10, 5))
sns.histplot(data=df_titanic, x='age', kde=True, hue='sex')
plt.title('Age Distribution by Gender')
plt.show()

# Plot 2: Survival Rate by Passenger Class and Gender
plt.figure(figsize=(10, 5))
sns.barplot(data=df_titanic, x='class', y='survived', hue='sex')
plt.title('Survival Rate by Passenger Class and Gender')
plt.ylabel('Survival Rate (0 to 1)')
plt.show()
```
**Python Concepts:**
*   `sns.histplot()`: Groups continuous numbers (age) into visual bins. `kde=True` adds a smooth trendline.
*   `sns.barplot()`: When the target is binary (0 for died, 1 for survived), a barplot automatically calculates the mathematical average, which gives us the exact survival *rate percentage*.

**Expected Results:**
EDA is working! The bar chart instantly reveals a massive pattern: Females in 1st class had a nearly 100% survival rate, while males in 3rd class had less than 20%. The algorithm will heavily rely on these features.
</details>

### Step 4: Feature Engineering
**Your Task:**
Let's create new data from existing data. Create a `family_size` column by adding Siblings/Spouses and Parents/Children together. Then, we need to mathematically encode the text-based `sex` column so the ML model can read it.

**Suggested prompt for Gemini:**
> *"Write pandas code to create a 'family_size' column by adding 'sibsp' and 'parch'. Then, use pandas get_dummies to One-Hot Encode the 'sex' column into 1s and 0s."*

<details>
<summary><strong>Click here for the working code and explanations</strong></summary>

```python
# 1. Create a new Feature (Family Size)
df_titanic['family_size'] = df_titanic['sibsp'] + df_titanic['parch']

# 2. One-Hot Encoding (Converting Text to Numbers)
# drop_first=True prevents the Dummy Variable Trap (if you aren't male, you must be female)
df_titanic = pd.get_dummies(df_titanic, columns=['sex'], drop_first=True)

display(df_titanic[['age', 'family_size', 'sex_male']].head())
```
**Python Concepts:**
*   **`pd.get_dummies()`**: This is pandas' built-in tool for **One-Hot Encoding**. It turns the text "male" and "female" into a binary math column called `sex_male` (1 for True, 0 for False). ML models can now process this data!
</details>

### Step 5: Preparing for Modeling
**Your Task:**
The final step is to drop "noisy" or redundant columns that provide no mathematical value, leaving behind a clean matrix of numbers ready for a Train/Test split.

<details>
<summary><strong>Click here for the working code and explanations</strong></summary>

```python
# Drop redundant columns created by Seaborn or text columns we can't use
columns_to_drop =[ 'alive', 'embark_town']
df_final = df_titanic.drop(columns=columns_to_drop)

print("Final Dataset ready for an ML Algorithm:")
df_final.info()
```
**Conclusion of Part 1:**
Notice how we transformed a messy dataset with missing values and text categories into a clean, numerical matrix. **This is what Data Scientists spend 80% of their time doing!**
</details>

---

## Part 2: The Iris Dataset (Academic Data)

The Iris dataset contains measurements of three species of Iris flowers. 
**Why is this part shorter?** Unlike Titanic, Iris is a famous "toy" dataset used in academia. As you will see, it is perfectly clean. In the real world, you *must* do all 5 steps. With academic datasets, Steps 2, 4, and 5 are often skipped.

> [!NOTE]  
> **Domain Knowledge:**
> *   **Sepal**: The outermost part of the flower (Measured as `sepal_length` and `sepal_width`).
> *   **Petal**: The colorful inner part (Measured as `petal_length` and `petal_width`).


<img src="./iris.png" width="50%">

### Step 1 & 2: Load, Understand, and (Attempt to) Clean
**Your Task:**
Load the `iris` dataset from Seaborn. Check for null values and print the summary statistics. 

<details>
<summary><strong>Click here for the working code and explanations</strong></summary>

```python
# 1. Load the Iris dataset
df_iris = sns.load_dataset("iris")

# 2. Check for missing values (Step 2: Cleaning)
print("--- Missing Values ---")
print(df_iris.isnull().sum())
```
**Expected Results:**
The output shows **exactly 0 missing values**. Because there are no missing values, no extreme outliers, and all feature columns are already numerical, **we get to completely skip Step 2 (Cleaning) and Step 4 (Feature Engineering)!**
</details>

### Step 3: EDA (Finding the Best Predictors)
**Your Task:**
Since the data is clean, we jump straight to Exploratory Data Analysis. Use Gemini to generate a pairplot to compare every numerical feature against every other feature simultaneously. Color the data points by species.

**Suggested prompt for Gemini:**
> *"Write Python code using seaborn to create a pairplot for the iris dataset. Set the hue to 'species' to color the dots based on the flower type."*

<details>
<summary><strong>Click here for the working code and explanations</strong></summary>

```python
# Create a pairplot (Visual Correlation check)
sns.pairplot(data=df_iris, hue='species')
plt.suptitle('Pairwise Relationships Between Features', y=1.02)
plt.show()
```

**Python Concepts:**
*   `sns.pairplot()`: Automatically creates a grid of scatterplots mapping every column against every other column. 

**Expected Results:**
Look at the scatterplots. You will see that the *Setosa* species (usually marked in blue) forms an entirely separate cluster from the other two species in almost every graph. This tells us that if an ML model looks at these numbers, it will have a 100% accuracy rate identifying *Setosa* flowers, simply by drawing a mathematical line between the blue dots and the rest!
</details>



***

## Part 3: Knowledge Check (Concepts & Code)

Test your understanding of the ML Data Pipeline and the Google Colab environment. Try to answer these questions before clicking to reveal the explanations!

### Section A: General Concepts & The 5 Steps

**1. Why did we have to perform all 5 steps on the Titanic dataset, but skip Steps 2 and 4 for the Iris dataset?**
<details>
<summary><strong>Reveal Answer</strong></summary>
The Titanic dataset represents <b>real-world data</b>. It is messy, has missing values (requiring Step 2: Cleaning), and contains text-based categories like gender (requiring Step 4: Feature Engineering). <br><br>
The Iris dataset is a pre-cleaned, academic "toy" dataset. Because it perfectly represents continuous numbers without a single missing value, we could skip the cleaning and engineering steps and jump straight to EDA. In the real world, you almost never get to skip steps!
</details>

**2. During Step 2 (Cleaning) of the Titanic data, why did we replace the missing ages with the <i>median</i> rather than the <i>mean</i> (average)?**
<details>
<summary><strong>Reveal Answer</strong></summary>
The <i>mean</i> is highly sensitive to extreme outliers. If a few passengers on the Titanic were 80 years old, it would artificially drag the average age up. The <i>median</i> (the exact middle number when all ages are lined up) is <b>robust against outliers</b>, making it a much safer guess when imputing (filling) missing data.
</details>

**3. In Step 4 (Feature Engineering), we used One-Hot Encoding to change the text "male" and "female" into 1s and 0s. Why is this strictly mandatory?**
<details>
<summary><strong>Reveal Answer</strong></summary>
Classical Machine Learning algorithms are purely mathematical—they use equations to draw lines and calculate distances. They cannot multiply or subtract text strings like "male" or "female". If you do not encode your text categories into numbers, the algorithm will instantly crash.
</details>

### Section B: Python & Google Colab Tools

**4. At the beginning of the activity, we ran `!pip install pandas`. Why did we put an exclamation mark (`!`) at the front instead of writing normal Python code?**
<details>
<summary><strong>Reveal Answer</strong></summary>
Google Colab is essentially a Python environment running on a Linux computer in the cloud. <br><br>
The <code>!</code> tells Colab: <i>"Do not run this as Python code. Instead, pass this directly to the underlying computer's terminal (shell)."</i> We use it for system-level tasks, like installing software packages, downloading files, or checking the current directory (<code>!ls</code>).
</details>

**5. In Google Colab, you will sometimes see commands starting with `%` or `%%` (e.g., `%time` or `%%bash`). What are these, and when should you use them?**
<details>
<summary><strong>Reveal Answer</strong></summary>
These are called <b>Magic Commands</b>. They are special built-in shortcuts unique to Jupyter Notebooks and Google Colab that give you extra control over how the code runs.
<ul>
  <li><b>Line Magic (<code>%</code>)</b>: Applies <i>only to a single line</i> of code. 
    <ul><li><i>Example:</i> <code>%time x = 5 + 5</code> will tell you exactly how many milliseconds it took the computer to run that one specific line.</li></ul>
  </li>
  <li><b>Cell Magic (<code>%%</code>)</b>: Must be placed at the very top of a cell, and it applies to the <i>entire block of code</i> inside that cell.
    <ul>
      <li><i>Example:</i> <code>%%time</code> will time how long the entire cell takes to run.</li>
      <li><i>Example:</i> <code>%%writefile data.txt</code> will take everything you typed in that cell and save it as a text file on the Colab server instead of running it as code.</li>
    </ul>
  </li>
</ul>
<b>When to use them:</b> Use them when you need to debug your environment, measure performance (timing code is very common in ML!), or interact with files without writing complex Python scripts to do it.
</details>

**6. In Step 3 (EDA), we heavily used the `hue` argument in our Seaborn Python code (e.g., `sns.histplot(hue='sex')`). Why is this argument so important for Data Scientists?**
<details>
<summary><strong>Reveal Answer</strong></summary>
The <code>hue</code> argument color-codes the data points based on a specific category. In EDA, our primary goal is to find out which features strongly influence the Target. By using <code>hue</code> to split the data by gender or survival, the visual differences become instantly obvious to the human eye, telling us exactly which features the ML model will find most valuable.
</details>