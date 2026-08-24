# Session Review & Sample Exam Questions: EDA & Data Preparation

Use these questions to test your understanding of the foundational principles of data handling, exploratory analysis, and feature preparation.

---

### Section A: Multiple Choice Questions (MCQ)

**1. What does the acronym GIGO stand for in data science, and what does it imply?**
- A) Good Input, Great Output – means complex algorithms can automatically fix bad data.
- B) Garbage In, Garbage Out – means poor-quality data leads to poor results regardless of algorithm sophistication.
- C) General Input, General Output – means any structured tabular dataset works for any model without preprocessing.
- D) Graph In, Graph Out – means visual charts are strictly required before running statistical tests.

**2. A dataset contains the variable `Education_Level` with values: `"High School"`, `"Bachelor"`, `"Master"`, `"PhD"`. This variable is best classified as:**
- A) Continuous Numerical
- B) Nominal Categorical
- C) Ordinal Categorical
- D) Binary

**3. You are exploring a dataset and create a scatter plot of `Age` vs. `Fare`. Which type of EDA are you performing?**
- A) Univariate graphical
- B) Bivariate graphical
- C) Multivariate non-graphical
- D) Univariate non-graphical

**4. Why is the median often preferred over the mean for imputing missing values in a skewed numerical variable like `Income`?**
- A) The median is always an integer.
- B) The median is robust and far less sensitive to extreme outliers than the mean.
- C) The median executes faster in Pandas.
- D) The median guarantees that the original variance remains unchanged.

**5. You find a strong positive correlation of $+0.90$ between `Ice_Cream_Sales` and `Drowning_Incidents`. What is the correct statistical interpretation?**
- A) Buying ice cream directly causes drowning accidents.
- B) High drowning rates cause people to purchase more ice cream.
- C) Correlation does not imply causation; a lurking confounding variable (e.g., hot summer weather) likely drives both.
- D) Any correlation above $+0.80$ is mathematically invalid and represents recording error.

**6. In One-Hot Encoding, we create binary columns for categories. Why is this preferred over simply assigning `"Male" = 1` and `"Female" = 2`?**
- A) It drastically reduces total memory usage.
- B) It prevents the algorithm from assuming an artificial mathematical hierarchy ($2 > 1$, i.e., false ordinality).
- C) It converts categorical variables directly into continuous floating-point variables.
- D) It automatically removes rows with missing values.

---

### Section B: True or False (with 1-Sentence Justification)

**7. Statement:** *"Before performing any data cleaning or imputation, we should first inspect the dataset and perform initial EDA."*
- **True / False?** Justify: __________________________________________________________________

**8. Statement:** *"If a boxplot reveals extreme high-value outliers in the `Fare` variable, we should immediately delete those observations."*
- **True / False?** Justify: __________________________________________________________________

**9. Statement:** *"Exploratory Data Analysis (EDA) and Data Cleaning are strictly sequential steps (you clean once, and then you are finished)."*
- **True / False?** Justify: __________________________________________________________________

**10. Statement:** *"Running Pandas `df.info()` displays column data types and the count of non-null values for each feature."*
- **True / False?** Justify: __________________________________________________________________

---

### Section C: Short-Answer & Scenario Questions

**11. The Iterative Workflow**
You inspect a passenger dataset and identify 177 missing values in `Age`. You impute these missing values using the median age ($28$).
- **(a)** Why is it essential to plot the histogram of `Age` *again* after completing the imputation?
- **(b)** What visual artifact or feature change will you likely observe in the post-cleaning histogram?

**12. Outlier Reasoning (Detect $\rightarrow$ Investigate $\rightarrow$ Decide)**
A boxplot of ticket fares shows that a few passengers paid over $\$500$, while 75% of passengers paid under $\$31$.
- **(a)** According to lecture principles, what should you do first rather than immediately deleting these rows?
- **(b)** Give two realistic domain explanations why a passenger legitimately paid $\$500$ without it being a data-entry error.

**13. Feature Engineering**
A dataset includes `SibSp` (number of siblings/spouses aboard) and `Parch` (number of parents/children aboard).
- **(a)** Creating a combined variable `Family_Size = SibSp + Parch` is an example of what process?
- **(b)** Why might `Family_Size` provide a stronger signal to a predictive model than keeping the two raw counts separate?

**14. Preventing Data Leakage**
You have 1,000 rows of housing data. Before dividing your data into an 80% training set and a 20% test set, you calculate the median `Lot_Area` across all 1,000 rows and use it to fill missing values.
- Why is this practice flawed, and what specific machine learning problem does it introduce?

**15. Interpreting `df.describe()`**
A dataset has 891 total passenger records. Running `df['age'].describe()` returns:
`count: 714 | mean: 29.70 | std: 14.53 | min: 0.42 | 25%: 20.12 | 50%: 28.00 | 75%: 38.00 | max: 80.00`
- **(a)** Why does `count` equal 714 instead of 891?
- **(b)** Comparing the `mean` ($29.70$) and the `median` / `50%` ($28.00$), what can you infer about the skewness of the age distribution?

---

### Section D: Code & Concept Interpretation

**16.** Examine the following Pandas code:
```python
df_encoded = pd.get_dummies(df, columns=["sex", "embarked"], drop_first=True, dtype=int)
```
- **(a)** What transformation does `pd.get_dummies()` perform on the categorical columns?
- **(b)** Why is `drop_first=True` applied? (Mention multicollinearity or the dummy variable trap).

---

## Solutions & Marking Guide

1. **B** — GIGO means algorithm sophistication cannot overcome flawed input data.
2. **C** — Ordinal, because there is an intrinsic educational hierarchy ($\text{High School} < \text{Bachelor} < \text{Master} < \text{PhD}$).
3. **B** — Bivariate (two variables: `Age` and `Fare`) and graphical (scatter plot).
4. **B** — The median represents the 50th percentile and is resistant to extreme values in skewed distributions.
5. **C** — Correlation measures linear association, not causality; warm weather is a classic confounding factor.
6. **B** — Avoids false ordinality (preventing models from interpreting categories as mathematically greater or smaller).
7. **True** — You must inspect and explore the data first to identify specific quality issues before deciding how to clean them.
8. **False** — Outliers are not automatically errors; they must be investigated using domain knowledge before taking action.
9. **False** — EDA and cleaning are iterative (Inspect $\rightarrow$ Clean $\rightarrow$ Validate with EDA again).
10. **True** — `df.info()` summarizes non-null counts, data types, and memory usage.
11. **(a)** To verify the cleaning operation succeeded and to check how imputation distorted the distribution.  
    **(b)** A prominent artificial spike/peak will appear at the median value ($28$).
12. **(a)** Investigate the observation using domain context to determine if it is a genuine record.  
    **(b)** 1) Luxury first-class suite accommodations; 2) A group/family ticket where multiple passengers travelled under one shared fare.
13. **(a)** Feature Engineering.  
    **(b)** It directly captures whether a passenger travelled alone or in a family unit, which often correlates more cleanly with survival dynamics.
14. It causes **Data Leakage**. Using statistics calculated across the whole dataset allows information from the future test set to influence the preprocessing of the training set, leading to overly optimistic evaluation metrics.
15. **(a)** There are $177$ missing (`NaN`) values ($891 - 714 = 177$).  
    **(b)** Because $\text{mean} (29.7) > \text{median} (28.0)$, the distribution has a slight right skew (pulled upward by older passengers).
16. **(a)** Converts nominal categorical columns into binary $0/1$ indicator variables for model compatibility.  
    **(b)** Drops the reference category column to prevent perfect multicollinearity (the **Dummy Variable Trap**).
