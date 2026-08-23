# Mapping of Theory, Demos, and Activities to Core Topics

This document maps the theoretical concepts introduced in the lecture to their practical implementation in the code demos and associated hands-on activities. It covers the three main pillars of the data preparation workflow: **Data Handling**, **Data Processing (EDA & Preprocessing)**, and **Feature Engineering**.

---

## 1. Data Handling
*(Inspection, Cleaning, Missing Values, Outliers, Data Types)*

| Theory Section | Concept | Implementation in Demos & Activities |
| :--- | :--- | :--- |
| **§1** | GIGO Principle & Schema Understanding | The concept of data quality and the importance of understanding feature semantics are covered conceptually. The demos (Step 3) and activities explicitly classify features into numerical, categorical, and ordinal types, linking directly to schema definition. |
| **§2** | Initial Inspection (`head`, `shape`, `info`, `isnull()`, `describe()`) | The theoretical inspection workflow is applied immediately in the demos (Step 2) and activities. Standard Pandas functions establish a baseline understanding of dataset dimensions, schema, and initial missingness. |
| **§5.1** | Handling Missing Data (Imputation) | Imputation strategies (median for skewed numerical data, mode for categorical data) are practiced directly. For example, `age` is imputed with the median, and `embarked` / `sex` / `payment` are imputed with the mode. |
| **§5.1** | Dropping Variables with High Missingness | In line with the rule to drop columns with overwhelming missing data, the Titanic materials remove the `deck` column (~77% missing) rather than introducing severe imputation bias. |
| **§5.2** | Data Types & Parsing | `df.info()` is used to inspect stored data types. Where appropriate (e.g., taxi trip timestamps), string objects are parsed into true `datetime` types to enable temporal computation. |
| **§5.3** | Duplicates & Inconsistent Values | Demos and activities check for duplicate rows (`df.duplicated().sum()`) and inspect unique category values (`.unique()`), reinforcing the need to detect redundancy and inconsistent representations. |
| **§5.4** | Invalid / Impossible Values | Range inspections via `describe()` check for impossible negative measurements or zero-distance anomalies, ensuring values remain within logical domain boundaries. |
| **§6** | Outliers & Anomalies (Detect → Investigate → Decide) | Outliers are detected via boxplots (e.g., `fare`). In line with the theory, unusual observations are **not** automatically deleted; they are investigated and retained when they represent valid domain phenomena. |

---

## 2. Data Processing (EDA & Preprocessing)
*(Visualization, Distribution Analysis, Correlation, Iterative Workflow)*

| Theory Section | Concept | Implementation in Demos & Activities |
| :--- | :--- | :--- |
| **§3 & §7** | Iterative EDA Cycle (EDA → Clean → EDA again) | The iterative nature of EDA is a core theme. Both demos and activities feature a dedicated **validation step** (Step 10 / Step 5) where distributions and relationships are re-plotted after cleaning to verify the impact of imputation. |
| **§4.1** | Univariate Analysis (Histograms, Boxplots, Value Counts) | Univariate techniques are practiced extensively: histograms with KDEs for distribution shape and skewness (`age`, `flipper_length_mm`, `distance`), boxplots for spread, and `value_counts()` for category frequencies. |
| **§4.2** | Bivariate Analysis (Scatter plots, Bar plots, `hue`) | Relationships between two variables (and a third grouping variable) are explored through grouped bar charts (e.g., survival by class and sex) and scatter plots (e.g., morphological measurements by species). |
| **§4.3** | Multivariate Analysis (Correlation Matrices, Heatmaps, Pairplots) | Multivariate interactions are examined using Pearson correlation heatmaps (`sns.heatmap`) for numerical variables and multi-variable pair plots (`sns.pairplot`) in the Iris activity. |
| **§9** | Data Leakage (Preview) | The theory introduces data leakage as an essential boundary condition. Demos and activities intentionally stop **before** train/test splitting and modeling, noting that preprocessing pipelines will be scoped strictly to training splits next week. |

---

## 3. Feature Engineering
*(Creating new features, Encoding categoricals, Avoiding traps)*

| Theory Section | Concept | Implementation in Demos & Activities |
| :--- | :--- | :--- |
| **§8.2** | Creating New Features | Meaningful features are derived by combining raw variables: `family_size = sibsp + parch` (Titanic), `bill_ratio = bill_length / bill_depth` (Penguins), and `duration_min` / `tip_pct` (Taxis). |
| **§8.1** | Encoding Categoricals (One-Hot Encoding) | Categorical features are converted into machine-readable numeric formats using `pd.get_dummies()`. |
| **§8.1** | Avoiding False Ordinality | The materials explicitly emphasize that nominal categories (e.g., `sex`, `island`, `color`) must not be assigned arbitrary integers ($1, 2, 3$), which would falsely imply a mathematical hierarchy ($3 > 2 > 1$). |
| **§8.1** | Dummy Variable Trap | The `drop_first=True` parameter is consistently applied and explained as the standard technique to prevent perfect multicollinearity (the dummy variable trap). |
| **§8.3** | Scaling Numerical Features (Min-Max / Standardization) | Introduced conceptually in theory, but **intentionally deferred** in practice because feature scaling must be fit strictly on training splits during the Supervised Learning lecture. |
| **§6** | Log Transformations | Mentioned in theory as a mathematical tool to compress right-skewed distributions. Practical application is deferred to model tuning. |

---

## Summary: What is Intentionally Deferred to Next Week

To respect the modeling boundary and avoid **data leakage**, the following concepts are introduced theoretically but reserved for hands-on execution in the Supervised Learning lecture:

1. **Train/Test Splitting & Cross-Validation:** Dividing data prior to model training and performance evaluation.
2. **Feature Scaling (Normalization / Standardization):** Deriving scaling parameters (mean/std or min/max) exclusively from the training partition.
3. **Supervised Model Training & Evaluation:** Fitting classifiers/regressors and measuring generalization metrics.

---

## Conclusion

The current materials provide foundation for the complete exploratory data lifecycle:
* **Data Handling:** Structural inspection, missingness treatment, and outlier evaluation.
* **Data Processing:** Iterative visualization, distribution assessment, and post-cleaning validation.
* **Feature Engineering:** Domain-driven feature creation, one-hot encoding, and multicollinearity prevention.