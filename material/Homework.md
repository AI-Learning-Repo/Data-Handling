
# Homework

## Before Next Session 

- [Introduction to classification](https://github.com/microsoft/ML-For-Beginners/blob/main/4-Classification/1-Introduction/README.md)
- [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- Underfitting and Overfitting: 
  - [IBM](https://www.youtube.com/watch?v=0RT2Q0qwXSA)
  - Kaggle: [Short Video](https://youtu.be/MDiZg88mg9c?si=GbAj6Bp-eeG2dFZk&t=26), [Intro](https://www.kaggle.com/code/dansbecker/underfitting-and-overfitting)


 <!-- [2025 Spring Introduction to Classical Machine Learning: Supervised ML: Classification Algorithms](https://www.youtube.com/watch?v=n4rl8yjdMlM) -->

--------
## Links

* [Seaborn Crash Course - Data Visualization in Python (16min)](https://www.youtube.com/watch?v=rLVCSmtoA7U)
* [Exploratory Data Analysis (45 minutes only)](https://youtu.be/1MGG75oUK7o)
* [Python for Data Analysis, 3rd Edition - Wes McKinney](https://metropolia.finna.fi/Record/nelli15.5590000000474271)
* [AI Engineering - Chip Huyen](https://metropolia.finna.fi/Record/nelli15.39388935400041)
* [Data Science for Beginners - Microsoft](https://github.com/microsoft/Data-Science-For-Beginners)
* [ML for Beginners - Microsoft](https://github.com/microsoft/ML-For-Beginners)

<details>
<summary>Suggested Tasks</summary>

Here are three hands-on tasks using real-world datasets from Kaggle. Each task reinforces the lecture workflow: **Inspect → Explore → Clean → Engineer → Prepare**.

### Task 1: Chocolate Bar Ratings

**Dataset Link:** [Kaggle: Chocolate Bar Ratings](https://www.kaggle.com/datasets/rtatman/chocolate-bar-ratings)  
**Context:** Over 1,700 expert ratings of dark chocolate bars from around the world.

1. **Inspect (§2):**
   - Check dataset dimensions (`shape`), column data types (`info()`), and missing values (`isnull().sum()`).
   - Notice that `Cocoa Percent` is stored as text (`object`) with a `%` sign, and `Rating` is numerical.
2. **Clean & Parse (§5.2, §5.1):**
   - Clean `Cocoa Percent` by removing the `%` character and converting the column to `float` (e.g., `"70%"` $\rightarrow$ `70.0`).
   - Identify missing or blank strings in `Broad Bean Origin` and `Bean Type` and decide whether to impute (mode / `"Unknown"`) or drop.
3. **Univariate & Bivariate EDA (§4.1, §4.2):**
   - Plot the distribution of `Rating` using a histogram with KDE.
   - Create a boxplot of `Rating` across the top 10 most frequent `Company Location` countries.
   - Use a scatter plot to examine the relationship between `Cocoa Percent` and `Rating`.
4. **Feature Engineering (§8.2):**
   - Create `cocoa_category` by binning `Cocoa Percent` into ordinal ranges (e.g., `"< 70%"`, `"70-79%"`, `"80-89%"`, `"90%+"`).
   - Create a binary feature `is_domestic` indicating whether the manufacturer's location matches the bean origin.
5. **Categorical Encoding (§8.1):**
   - One-hot encode the top categories of `Company Location` and `Broad Bean Origin` using `pd.get_dummies(..., drop_first=True, dtype=int)`.

### Task 2: Mushroom Classification (Categorical & Quality Inspection)

**Dataset Link:** [Kaggle: Mushroom Classification](https://www.kaggle.com/datasets/uciml/mushroom-classification)  
**Context:** Classifying whether a mushroom is edible (`e`) or poisonous (`p`) based on 22 categorical physical characteristics.

1. **Inspect & Quality Check (§2, §5.3):**
   - Use `.nunique()` and `value_counts()` on all columns.
   - **Quality Discovery 1:** Identify that `veil-type` has only 1 unique value across all rows (zero variance) and drop it.
   - **Quality Discovery 2:** Look closely at `stalk-root` and detect missing values disguised as the string `'?'`. Replace `'?'` with `np.nan` and decide on an imputation strategy (e.g., mode or `"missing"`).
2. **Bivariate EDA (§4.2):**
   - Create grouped bar plots showing the proportion of edible vs. poisonous mushrooms across key features like `odor`, `cap-color`, and `habitat`.
   - Identify which feature (`odor`) provides almost perfect separation between classes.
3. **Encoding & Avoiding Traps (§8.1):**
   - One-hot encode the nominal categorical features using `pd.get_dummies(..., drop_first=True, dtype=int)` to avoid false ordinality and the dummy variable trap.
4. **Final Check (§10):**
   - Confirm that the resulting dataset is 100% numerical and ready for classification modeling next week.


### Task 3: Student Stress & Lifestyle Analysis

**Dataset Link:** [Kaggle: Student Stress Monitoring Datasets](https://www.kaggle.com/datasets/mdsultanulislamovi/student-stress-monitoring-datasets)  
**Context:** Survey data measuring student stress levels, daily habits, study routines, and physiological metrics.

1. **Inspect & Outlier Detection (§2, §6):**
   - Inspect summary statistics with `.describe()`.
   - Create boxplots for daily `study_hours` (or study load) and `sleep_hours` to detect potential outliers or impossible data (e.g., >24 hours/day).
2. **Data Cleaning (§5.1):**
   - Handle missing values using **median imputation** for numerical columns and **mode imputation** for categorical survey questions.
3. **EDA (Univariate, Bivariate & Multivariate) (§4):**
   - Examine the distribution of the target variable (`stress_level`).
   - Create a scatter plot with regression line or grouped bar plot showing `sleep_hours` vs. `stress_level`.
   - Compute a Pearson correlation matrix and visualize it using a Seaborn heatmap.
4. **Feature Engineering (§8.2):**
   - Create a lifestyle ratio feature (e.g., `study_to_sleep_ratio = study_hours / sleep_hours`).
   - Create an ordinal sleep category: `sleep_quality_category` (`"short (<6h)"`, `"normal (6-8h)"`, `"long (>8h)"`).
5. **Iterative EDA & Validation (§7):**
   - Re-plot distributions before and after imputation/cleaning to confirm that no artificial distortion was introduced.

</details>



<!-- 
- Exploratory Data Analysis (EDA): **Objective** To explore and understand the fundamentals of EDA using Python in a hands-on mini-project. 

   1. **Video to Watch**:  
      - Watch the first **45 minutes only** of the following video: [Exploratory Data Analysis Video](https://youtu.be/1MGG75oUK7o).  
      - Pay close attention to the explanation of the **concepts and code walkthroughs**.  

   2. **Source Code**:  
      - Download the source code from [the following repository](https://github.com/codebasics/project-da-online-retail-pandas).  

   3. **Task**:  
      - Use **Google Colab** to run the source code provided in the repository.  
      - Ensure that all the code executes without errors.  
      - Identify and document any challenges faced (e.g., errors, unclear code snippets).  

   4. **Clarifications and Help**:  
      - If certain parts of the code are unclear or confusing, use tools your favorite **Large Language Model (LLM)** to get explanations and insights.  

   5. **Reflection**:  
      - Highlight any issues encountered during the code execution process.  
-->
