# Session Review & Sample Exam Questions: Unsupervised Learning

These questions focus on unsupervised learning, customer segmentation, K-Means, centroids, Euclidean distance, scaling, inertia, the elbow method, silhouette score, `make_blobs`, cluster spread, and interpreting clusters.

---

## Section A: Multiple Choice

**1. Which statement best describes unsupervised learning?**

A) A model learns from known target values.  
B) A model discovers patterns without a supplied target variable.  
C) A model only predicts continuous values.  
D) A model requires class labels.

**2. What is the main goal of clustering?**

A) Predict a known target.  
B) Group observations according to similarity.  
C) Calculate regression error.  
D) Remove every outlier.

**3. Why is `CustomerID` usually excluded from the Mall Customers clustering features?**

A) It is an identifier, not a meaningful customer characteristic.  
B) It is always the target.  
C) It contains missing values by definition.  
D) K-Means cannot use numerical values.

**4. Which features are used in the basic customer-segmentation example?**

A) `CustomerID` and `Gender`  
B) `Age` and `CustomerID`  
C) `Annual Income (k$)` and `Spending Score (1-100)`  
D) `Gender` and `Age`

**5. What does `n_clusters=4` mean?**

A) Use four features.  
B) Create four clusters.  
C) Use four training epochs.  
D) Remove four rows.

**6. What is a centroid?**

A) A target label.  
B) The mean position of observations assigned to a cluster.  
C) The observation with the highest ID.  
D) A test-set error.

**7. Why can feature scaling be important for K-Means?**

A) K-Means uses distances, so large-scale features can dominate.  
B) Scaling guarantees the correct number of clusters.  
C) Scaling removes missing values automatically.  
D) Scaling creates target labels.

**8. What does inertia measure?**

A) Within-cluster squared distances from observations to their centroids.  
B) Classification accuracy.  
C) Pearson correlation.  
D) The number of features.

**9. What is the purpose of the elbow method?**

A) To inspect how inertia changes as `k` increases.  
B) To prove that there are exactly two clusters.  
C) To calculate MAE.  
D) To remove outliers.

**10. What does silhouette score generally measure?**

A) Cluster cohesion and separation from other clusters.  
B) The number of rows.  
C) The number of missing values.  
D) Causation between features.

**11. What generally happens when `cluster_std` increases in `make_blobs()`?**

A) Clusters become more dispersed and may overlap more.  
B) Clusters always become tighter.  
C) The number of observations becomes zero.  
D) The data become supervised.

**12. Why are `make_blobs()` labels not supplied to K-Means during fitting?**

A) K-Means should discover structure without supervised labels.  
B) K-Means only accepts text labels.  
C) The labels are centroids.  
D) The labels are always missing.

**13. Which statement about cluster labels is correct?**

A) Label 0 is better than label 1.  
B) Labels are identifiers and have no inherent ranking.  
C) Label 2 always means high income.  
D) Labels are supplied target values.

**14. Why does inertia usually decrease as `k` increases?**

A) More centroids can place observations closer to their assigned centroids.  
B) The dataset becomes smaller.  
C) The algorithm stops using distance.  
D) Missing values are removed.

**15. Which statement is most accurate about customer clusters?**

A) They are objectively correct categories.  
B) They are analytical groupings based on selected features and algorithm settings.  
C) They prove causal customer types.  
D) They always generalize to every business purpose.

---

## Section B: True or False

**16.** Unsupervised learning normally uses a supplied target variable.
**17.** K-Means repeatedly assigns observations to centroids and updates the centroids.
**18.** A feature with a larger numerical scale can have greater influence on Euclidean distance.
**19.** Scaling guarantees that K-Means finds the correct clusters.
**20.** Inertia generally decreases as the number of clusters increases.
**21.** The elbow method guarantees one objectively correct value of `k`.
**22.** A silhouette score closer to 1 generally indicates compact and separated clusters.
**23.** A larger `cluster_std` generally creates tighter synthetic clusters.
**24.** The numerical value of a cluster label has an inherent ranking.
**25.** Clustering proves that the discovered groups are meaningful for a business decision.

---

## Section C: Short-Answer Questions

**26. Unsupervised vs. supervised learning**

Consider:

```text
Problem A:
Predict annual income from age and spending score.

Problem B:
Group customers using annual income and spending score without supplied labels.
```

(a) Classify each problem as supervised or unsupervised.  
(b) What is the target in Problem A?  
(c) Why is there no target in Problem B?  
(d) What does Problem B produce?

**27. Features and identifiers**

The dataset contains:

```text
CustomerID
Gender
Age
Annual Income (k$)
Spending Score (1-100)
```

(a) Which two features are used in the basic example?  
(b) Why should `CustomerID` usually be excluded?  
(c) Why might `Gender` be left out of the first numerical example?  
(d) Does excluding a variable prove that it can never be useful?

**28. Euclidean distance**

Let:

```text
A = (2, 3)
B = (5, 7)
```

(a) Write the Euclidean distance formula.  
(b) Calculate the distance.  
(c) Why is distance important in K-Means?  
(d) What can happen if one feature has a much larger scale?

**29. Centroid update**

A cluster contains:

```text
(2, 4)
(4, 6)
(6, 8)
```

(a) Calculate the mean x-coordinate.  
(b) Calculate the mean y-coordinate.  
(c) Give the updated centroid.  
(d) Why does K-Means update centroids?

**30. Choosing `k`**

| k | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 900 | 0.41 |
| 3 | 600 | 0.56 |
| 4 | 470 | 0.49 |
| 5 | 390 | 0.38 |

(a) Which solution has the lowest inertia?  
(b) Which has the highest silhouette score?  
(c) Why is lowest inertia alone insufficient?  
(d) What other evidence should be considered?

**31. Synthetic data**

```python
from sklearn.datasets import make_blobs

X, generated_labels = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=2.5,
    random_state=42
)
```

(a) What does `n_samples=300` mean?  
(b) What does `centers=3` mean?  
(c) What does `cluster_std=2.5` control?  
(d) What are `generated_labels`?  
(e) Why should this not be treated as proof that real customer data form Gaussian blobs?

**32. Limitations**

Explain four limitations or considerations when using K-Means. Discuss any four of: choice of `k`, scaling, outliers, initialization, cluster shape, cluster density, and interpretation.

---

## Section D: Code Interpretation

**33. Selecting features**

```python
X = customers[
    ["Annual Income (k$)", "Spending Score (1-100)"]
]
```

(a) What does `X` contain?  
(b) How many features are used?  
(c) Why is `CustomerID` excluded?  
(d) What might change if age were added?

**34. Fitting K-Means**

```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

kmeans.fit(X)
```

(a) Explain `n_clusters=5`.  
(b) Explain `random_state=42`.  
(c) Explain `n_init=10`.  
(d) What does `fit(X)` learn?  
(e) Is a target variable supplied?

**35. Labels and centroids**

```python
customers["Cluster"] = kmeans.labels_
print(kmeans.cluster_centers_)
```

(a) What is stored in `kmeans.labels_`?  
(b) Why add a `Cluster` column?  
(c) What does `cluster_centers_` contain?  
(d) How many centroids exist when `n_clusters=5` and there are two features?

**36. Inertia loop**

```python
inertia_values = []

for k in range(1, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    model.fit(X)
    inertia_values.append(model.inertia_)
```

(a) Which values of `k` are tested?  
(b) What is stored in `inertia_values`?  
(c) Why does the curve generally decrease?  
(d) Why should the lowest inertia not automatically determine `k`?

**37. Scaling**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
labels = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
).fit_predict(X_scaled)
```

(a) What does `StandardScaler` do?  
(b) Why is scaling relevant to K-Means?  
(c) Why might these labels differ from labels based on unscaled data?  
(d) Does scaling guarantee better clustering?

---

## Section E: Integrated Questions

**38. Complete customer-segmentation scenario**

Answer the following:

(a) Is K-Means customer segmentation supervised or unsupervised?  
(b) What are the two basic features?  
(c) Why exclude `CustomerID`?  
(d) What is a centroid?  
(e) What does K-Means minimize?  
(f) What is the elbow method used for?  
(g) What does silhouette score add?  
(h) Why are clusters not automatically objectively correct?  
(i) Give two limitations of K-Means.

**39. Cluster-spread experiment**

Compare:

```text
Experiment A: cluster_std = 0.5
Experiment B: cluster_std = 2.5
```

(a) Which has tighter clusters?  
(b) Which is more likely to overlap?  
(c) How might overlap affect silhouette score?  
(d) Why is `make_blobs()` useful for learning clustering?  
(e) Why should synthetic results not automatically be generalized to customer data?

<!-- 
**40. Complete workflow**

Write a basic K-Means workflow for `customers` that:

(a) selects annual income and spending score;  
(b) creates a K-Means model;  
(c) fits the model and saves labels;  
(d) prints the centroids and inertia;  
(e) plots the clusters; and  
(f) explains why the clusters are analytical groupings rather than guaranteed objective categories. 
-->

---

# Solutions

## Section A: Multiple Choice

**1. B** — Unsupervised learning discovers patterns without a supplied target.
**2. B** — Clustering groups observations according to similarity.
**3. A** — `CustomerID` is an identifier, not a meaningful customer characteristic.
**4. C** — The basic example uses annual income and spending score.
**5. B** — `n_clusters=4` requests four clusters.
**6. B** — A centroid is the mean position of observations assigned to a cluster.
**7. A** — K-Means uses distances, so feature scale matters.
**8. A** — Inertia is the within-cluster sum of squared distances to centroids.
**9. A** — The elbow method examines the change in inertia as `k` increases.
**10. A** — Silhouette score considers cohesion and separation.
**11. A** — Larger spread produces more dispersed clusters and potentially more overlap.
**12. A** — K-Means should discover structure without using the generated labels.
**13. B** — Cluster labels are identifiers and have no inherent ranking.
**14. A** — More centroids generally reduce within-cluster distances.
**15. B** — Clusters depend on selected features, scaling, and algorithm settings.

## Section B: True or False

**16. False** — Unsupervised learning generally does not use supplied target labels.
**17. True** — K-Means alternates between assignment and centroid-update steps.
**18. True** — A larger numerical scale can dominate Euclidean distance.
**19. False** — Scaling helps comparability but does not guarantee correct clusters.
**20. True** — Inertia generally decreases as more clusters are allowed.
**21. False** — The elbow method is a heuristic, not a guarantee.
**22. True** — A score closer to 1 generally indicates compact and separated clusters.
**23. False** — Larger `cluster_std` generally creates more dispersed clusters.
**24. False** — Cluster labels do not have an inherent ranking.
**25. False** — Clustering does not automatically establish business meaning or causation.

## Section C: Short Answers

**26.**

(a) Problem A is supervised regression; Problem B is unsupervised clustering.  
(b) The target is annual income.  
(c) Problem B has no target because the goal is to discover groups.  
(d) It produces a cluster assignment for each observation.

**27.**

(a) Annual income and spending score.  
(b) `CustomerID` is an identifier and can introduce meaningless numerical differences.  
(c) The first example focuses on two numerical features that can be plotted directly.  
(d) No. A variable may be useful in another analysis or representation.

**28.**

(a) $d=\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}$
(b)$d=\sqrt{(5-2)^2+(7-3)^2}=\sqrt{9+16}=5$
(c) K-Means assigns observations to the closest centroid.  
(d) The larger-scale feature may dominate the distance calculation.

**29.**

(a) Mean x-coordinate: $(2+4+6)/3=4$
(b) Mean y-coordinate: $(4+6+8)/3=6$
(c) Updated centroid: `(4, 6)`.  
(d) The centroid is updated to represent the mean position of the currently assigned observations.

**30.**

(a) Lowest inertia: `k=5`.  
(b) Highest silhouette score: `k=3`, with `0.56`.  
(c) Inertia normally decreases as `k` increases, so the minimum may favor too many clusters.  
(d) Consider silhouette score, the elbow plot, visualization, cluster sizes, stability, interpretability, and domain knowledge.

**31.**

(a) 300 generated observations.  
(b) Three generated centers.  
(c) The spread of the generated clusters.  
(d) Labels from the synthetic data-generation process; they are not supplied to K-Means during unsupervised fitting.  
(e) Real customer data may not be spherical, Gaussian-shaped, or cleanly separated.

**32.** Possible limitations include the need to choose `k`, sensitivity to feature scaling, sensitivity to outliers and initialization, assumptions about cluster shape, difficulty with different densities or sizes, and the fact that mathematical clusters are not automatically meaningful business categories.

## Section D: Code Interpretation

**33.**

(a) `X` contains annual income and spending score.  
(b) Two features.  
(c) `CustomerID` is an identifier rather than a meaningful similarity feature.  
(d) Adding age changes the distance calculation and may change the clusters.

**34.**

(a) Five clusters are requested.  
(b) The random initialization becomes reproducible.  
(c) Ten initialization attempts are used.  
(d) The model learns centroids and assignments from `X`.  
(e) No target variable is supplied.

**35.**

(a) Cluster assignment for each fitted observation.  
(b) It allows the assignments to be inspected and visualized with the original data.  
(c) The learned centroid coordinates.  
(d) Five centroids, each with two coordinates.

**36.**

(a) `k=1` through `k=10`.  
(b) The inertia for each value of `k`.  
(c) More centroids usually make observations closer to their assigned centroid.  
(d) The lowest inertia often occurs at a large `k`; interpretability and other measures are also needed.

**37.**

(a) `StandardScaler` standardizes numerical features, commonly using each feature's mean and standard deviation.  
(b) K-Means uses distance, so scale affects the result.  
(c) Scaling changes the relative contribution of the features to distance.  
(d) No. Scaling does not guarantee better or more meaningful clusters.

## Section E: Integrated Questions

**38.**

(a) Unsupervised.  
(b) Annual income and spending score.  
(c) It is an identifier, not a meaningful behavioral feature.  
(d) The mean position of observations in a cluster.  
(e) Within-cluster squared distance to assigned centroids.  
(f) To inspect how inertia changes as `k` increases.  
(g) It considers both within-cluster cohesion and separation from other clusters.  
(h) Clusters depend on features, scaling, distance, algorithm settings, and context.  
(i) Examples: choosing `k`, sensitivity to scaling and outliers, initialization, cluster-shape assumptions, and interpretability.

**39.**

(a) Experiment A, with `cluster_std=0.5`.  
(b) Experiment B, with `cluster_std=2.5`.  
(c) Overlap can make assignments less distinct and reduce silhouette score.  
(d) It allows controlled experiments with the number of centers and cluster spread.  
(e) Synthetic data are simplified and may not reflect the structure of real customer data.

<!-- 
**40.**

One possible solution:

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Select features
X = customers[
    ["Annual Income (k$)", "Spending Score (1-100)"]
]

# Create and fit the model
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

customers["Cluster"] = kmeans.fit_predict(X)

# Inspect model information
print("Centroids:")
print(kmeans.cluster_centers_)

print("Inertia:", kmeans.inertia_)

# Plot the clusters
plt.scatter(
    X["Annual Income (k$)"],
    X["Spending Score (1-100)"],
    c=customers["Cluster"]
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments")
plt.show()
```

The resulting clusters are analytical groupings based on the selected features and K-Means settings. They are not guaranteed to be objectively correct, causal, or universally useful customer categories. 
-->
