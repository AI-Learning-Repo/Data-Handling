# **Part 1: Data and Datasets for Classical Machine Learning**  

### **Introduction**  

**The Role of Data in Machine Learning**  
In classical Machine Learning (ML), **data is the foundational building block** of model performance. Unlike traditional programming—where a human explicitly writes `if/then` rules—ML models rely on mathematical algorithms to ingest data, discover hidden patterns, and generate their own rules to make predictions. **If the data is messy, biased, or incomplete, the model’s performance will inevitably suffer.**

>[!WARNING]  
> **The GIGO Principle**
> In data science, there is a golden rule: **Garbage In, Garbage Out (GIGO)**. No matter how advanced your ML algorithm is, if you feed it low-quality data, it will produce low-quality predictions. 

**Classical ML vs. Modern AI (LLMs)**  
While both classical ML and modern Large Language Models (LLMs) rely heavily on data, the *type* of data they use differs significantly:
*   **Classical ML** typically relies on **structured datasets** (e.g., databases or Excel spreadsheets where data is neatly organized into rows and columns with predefined features).  
*   **Modern AI** (like ChatGPT) deals with massive **unstructured datasets** (e.g., raw text documents, images, or audio files).  

> [!NOTE]  
> **Objective of This Section: The ML Data Pipeline**
> To build an accurate model, data must pass through a strict, interconnected pipeline. You must first **Understand** what you have, **Clean** the errors, **Explore** the patterns (EDA), **Engineer** the data into mathematical formats, and finally **Prepare** it for the algorithm. We will explore this sequence using a famous beginner dataset: **Predicting passenger survival on the Titanic.**

---

### **1. Understanding the Dataset**  

**The Flow:** *Before you can manipulate or fix data, you must understand what each column represents and what your ultimate goal is.*

**Dataset Schema & Terminology**  
A structured dataset operates like a giant spreadsheet. 
*   **Features (Independent Variables)**: The inputs or clues the model uses to make a decision (e.g., *Age, Income, Credit Score*).  
*   **Target (Dependent Variable)**: The final outcome or label the model is trying to predict.  

**Types of Features**  
Algorithms treat different types of data differently. Recognizing these types is your first step in ML:
*   **Numerical Features**: Continuous, measurable values (e.g., income, age, temperature).  
*   **Categorical Features**: Discrete groups or categories with no mathematical value (e.g., gender, country, job title).  
*   **Ordinal Features**: Categorical variables that have a strict, logical order (e.g., *Customer Satisfaction*: Poor < Fair < Good < Excellent).  

<details>
<summary><b> Continuous Example: The Titanic Survival Dataset</b></summary>
Throughout this summary, we will use the Titanic dataset. Our goal is to predict who survived the sinking.
<ul>
  <li><b>Target (Output)</b>: <code>Survived</code> (Categorical: 1 for Yes, 0 for No).</li>
  <li><b>Features (Inputs)</b>: 
    <ul>
      <li><code>Pclass</code>: Passenger Class (Ordinal: 1st, 2nd, 3rd)</li>
      <li><code>Age</code>: (Numerical)</li>
      <li><code>Sex</code>: Gender (Categorical)</li>
      <li><code>Fare</code>: Ticket Price (Numerical)</li>
      <li><code>Cabin</code>: Room number (Categorical)</li>
    </ul>
  </li>
</ul>
</details>


<img src="./labeled_example.png" width="50%">

<!-- <img src="./unlabeled_example.png" width="50%"> -->

---

### **2. Data Cleaning and Preprocessing**  

**The Flow:** *Now that we know what our dataset looks like, we quickly realize real-world data is chaotic. Before we can analyze patterns, we must fix incomplete, noisy, and broken data. (Note: In practice, Cleaning and EDA are highly iterative—you often jump back and forth between them).*

**Common Data Issues & Solutions**  

**1. Handling Missing Data (Imputation)**  
If an algorithm encounters a blank cell, it will crash. 
*   **Imputation** is the process of replacing missing data with substituted values.
*   **Titanic Example**: In the Titanic dataset, the `Age` column is missing 177 values. We can't delete these passengers, so we fill the blanks using the **median** age (e.g., 28). Conversely, the `Cabin` column is missing 77% of its data. Because the missing data is so overwhelming, we simply **drop the column entirely**.

**2. Handling Incorrect Data Types**  
If a system saves the number `20` as text `"20"`, an ML model will misinterpret it as a category. You must explicitly cast data types (e.g., convert `string` to `integer`).  

**3. Handling Outliers**  
Outliers heavily distort distance-based ML models.  
*   **Titanic Example**: Most passengers paid a `Fare` between $8 and $30, but one passenger paid $512! This extreme outlier will confuse the model.
*   **Solutions**:  
    *   **Capping**: Set a maximum threshold (treat any fare over $100 as exactly $100).  
    *   **Transformations**: Use **logarithmic scaling**. 

<details>
<summary><b> Explain Term: Logarithmic Scaling</b></summary>
Logarithmic scaling compresses a wide range of numbers into a smaller, more manageable range. If you have fares of $10, $100, and $1,000, calculating the Log10 of these values transforms them into 1, 2, and 3. This brings the $512 Titanic fare closer to the rest of the data, preventing it from ruining the model's math.
</details>

> [!TIP]
> **Beware of Data Leakage**
> When filling missing values with the mean or median, **only calculate that average from your training data**, not the entire dataset! Calculating the mean using your final test data gives the model "hints" about the answers, leading to falsely high accuracy scores.

---

### **3. Exploratory Data Analysis (EDA)**  

**The Flow:** *With a clean dataset, we can now safely "look" at the data. EDA acts as a health check to uncover hidden relationships and decide which features actually matter for predicting our Target.*

**EDA Techniques**:  

**1. Descriptive Statistics**  
Functions like `df.describe()` in Pandas provide a mathematical snapshot: mean, min/max values, and **Standard Deviation** (how spread out the data is). This helps identify **Skewness** (lopsided data).

**2. Visualizing Data Distributions**  
*   **Histograms**: Show frequency. *Titanic Example:* A histogram of `Age` shows us most passengers were in their 20s and 30s.
*   **Scatter/Bar Plots**: Great for spotting trends against the target. *Titanic Example:* Plotting survival rates by `Sex` instantly reveals that females had a ~74% survival rate compared to males at ~19%. EDA has just told us `Sex` is a highly predictive feature!

**3. Checking Correlations**  
A **Correlation Matrix** measures how strongly variables are linked, scoring them from `-1.0` to `+1.0`.  
*   *Titanic Example:* We might find `Fare` and `Pclass` (Class) are highly correlated (1st class tickets cost more). Feeding the model perfectly correlated, redundant features adds unnecessary noise.

---

### **4. Feature Engineering**  

**The Flow:** *EDA showed us which relationships are important. Feature Engineering is the step where we actually transform the raw data to highlight those relationships, making it mathematically digestible for the algorithm.*

**Techniques**:  

**1. Handling Categorical Variables**  
ML models *only* understand numbers. You must mathematically encode text categories.
*   **Label Encoding**: Assigning a number to a category. Best for *Ordinal* data. *(Titanic Example: 1st Class = 1, 2nd Class = 2, 3rd Class = 3).*
*   **One-Hot Encoding**: Creating a new binary (0 or 1) column for every single category. Best for non-ordered categories.

<details>
<summary><b> Explain Term: The Dummy Variable Trap</b></summary>
If you use Label Encoding on the Titanic <code>Sex</code> column (Male=1, Female=2), the algorithm might accidentally assume that Female is "greater" or "worth more" mathematically because 2 > 1. <br><br>
To fix this, we use <b>One-Hot Encoding</b>, creating separate columns: <code>Is_Male</code> and <code>Is_Female</code>. If a passenger is male, they get a 1 in <code>Is_Male</code> and 0 in <code>Is_Female</code>. This removes any false mathematical hierarchy.
</details>

**2. Creating New Features**  
Sometimes human logic beats raw data. *Titanic Example:* Instead of keeping the `SibSp` (Siblings/Spouses) and `Parch` (Parents/Children) columns separate, we can add them together mathematically to create a brand new, highly predictive feature called `Family_Size`. 

**3. Scaling Numerical Features**  
If `Age` ranges from 0–80 and `Fare` ranges from 0–500, the algorithm might assume `Fare` is more important just because the numbers are bigger. **Scaling** forces all numbers onto a level playing field.
*   **Min-max scaling (Normalization)**: Squeezes all values into a range exactly between `[0, 1]`.  
*   **Standardization**: Centers the data around a mean of `0` with a standard deviation of `1`.

---

### **5. Preparing the Dataset for Modeling**  

**The Flow:** *The data is now clean, explored, and engineered into perfectly scaled numbers. The final step is to organize it structurally so the algorithm can train on it without cheating or developing biases.*

**Final Steps Before Training**  

**1. Feature Selection**  
Drop columns that add no predictive value to reduce "noise." *(Titanic Example: We drop `Passenger_Name` and `Ticket_Number` because they don't logically influence survival).*

**2. Balance the Dataset**  
If you have a severe distortion in class distribution (e.g., 99% of data is Class A, 1% is Class B), the model will develop a **Bias** toward the majority class. 
*   *Solution:* Use **Oversampling** (duplicating the minority class) or **Undersampling** (reducing the majority class) to ensure the model learns fairly.

**3. Train/Test Split & Avoiding Overfitting**  
You cannot test an ML model on the same data it learned from—that is like giving a student the answers to a test beforehand. If a model memorizes the training data perfectly but fails in the real world, it is called **Overfitting**. 
*   **Training Set (80%)**: Used to teach the model. *(Titanic Example: We show the model 80% of the passengers and let it learn the survival rules).*
*   **Test Set (20%)**: Hidden away and used only at the very end to evaluate true accuracy. *(Titanic Example: We ask the model to predict survival for the remaining 20% to see if its rules actually work).*

<details>
<summary><b> Advanced Solution to Overfitting: Cross-Validation</b></summary>
Instead of splitting the data into Train/Test just once, <b>Cross-Validation</b> splits it into 5 or 10 different chunks, training and testing multiple times. This ensures the model's high accuracy isn't just a lucky fluke based on how you split the data.
</details>

> [!NOTE] 
> **The Final Outcome**
> By following these 5 steps, your dataset has transformed from a messy, real-world spreadsheet into a pristine, mathematically scaled matrix. **Quality dictates performance**, and your data is now fully optimized and ready to be fed into a Machine Learning algorithm!

---
## Further Exploration: 

- [Pandas: Crash course](https://www.kaggle.com/learn/pandas)
- [Video ~50min: Pandas for Data Science](https://www.youtube.com/watch?v=Yp3fccNNfjQ)
- [Data Visualization: Crash course](https://www.kaggle.com/learn/data-visualization)
- [Video ~50min: NumPy for Data Science](https://www.youtube.com/watch?v=EmA_TuC2Vdk)
- Video Course: Intro to Machine Learning with Python
  - [Part 1: Welcome and Project Setup](https://youtu.be/rdaG53khzv0)
  - [Part 2: Exploratory Data Analysis](https://youtu.be/6BagRiSY1ds)
  - [Part 3: Train Test Split and Baseline Modeling](https://youtu.be/MufPx3L7nXM)
- Tips
  - [Choosing Plot Types](https://www.kaggle.com/code/alexisbcook/choosing-plot-types-and-custom-styles)
  - [How to Handle Missing Values](https://www.kaggle.com/code/alexisbcook/missing-values)
- [Intro to ML Course](https://developers.google.com/machine-learning/intro-to-ml/supervised#data)
