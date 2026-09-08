# Activity 2 — Unsupervised Learning in More Detail

## Overview

This activity extends the clustering concepts introduced in Activity 1.

The main focus is on:

1. **K-Means in more detail**
2. **Choosing the number of clusters**
3. **Feature scaling**
4. **Cluster evaluation**
5. **Hierarchical clustering and dendrograms**
6. **Anomaly detection**
7. **Association rule learning**
8. **Comparing different unsupervised-learning tasks**

The activity uses several small datasets because different unsupervised-learning tasks require different types of data.

The main principle remains:

> Unsupervised-learning algorithms attempt to discover structure in the available data without using a target variable to guide the learning process.

---

# Learning Objectives

By the end of this activity, the following should be understood:

* why the choice of features matters in clustering;
* why feature scaling can affect distance-based algorithms;
* how K-Means uses centroids;
* how inertia is used to evaluate K-Means;
* how the elbow method can help investigate the number of clusters;
* how silhouette scores can help compare clusterings;
* how hierarchical clustering works;
* how to interpret a dendrogram;
* how anomaly detection differs from clustering;
* how Isolation Forest can identify unusual observations;
* the meaning of contamination in anomaly detection;
* the difference between anomaly detection and clustering;
* the basic ideas of association rule learning;
* support, confidence, and lift;
* why different unsupervised-learning techniques answer different questions.

---

# Part A — K-Means in More Detail

## 1. Start with a Dataset

The first activity introduced K-Means using a simple two-dimensional dataset.

This activity will use the same general idea, but the analysis will be more systematic.

Create a dataset with four clusters.

### Code Cell

```python id="0z7jmf"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import silhouette_score

from sklearn.ensemble import IsolationForest
```

Generate the data:

### Code Cell

```python id="3lqkmq"
X, _ = make_blobs(
    n_samples=500,
    centers=4,
    cluster_std=1.2,
    random_state=42
)

print(X.shape)
```

The dataset contains:

```text
500 observations
2 features
```

---

# 2. Question 1 — Inspect the Data Before Clustering

Before running K-Means:

1. How many observations are there?
2. How many features are there?
3. Does the data appear to contain groups?
4. Approximately how many groups appear visible?

### LLM Hint Prompt

> I have a two-dimensional synthetic dataset generated with `make_blobs`. Help me inspect the shape of the data and interpret the scatter plot before applying a clustering algorithm. Focus on what can be observed visually without assuming that the visible groups are automatically the correct number of clusters.

<details>
<summary>Solution</summary>

There are 500 observations and two features.

The scatter plot should show approximately four visible groups because the dataset was generated around four centers.

The important point is that the visual structure provides an initial hypothesis. It does not by itself establish that four is always the best clustering solution.

</details>

---

# 3. Visualize the Dataset

### Code Cell

```python id="5y9fcl"
plt.figure(figsize=(8, 6))

plt.scatter(
    X[:, 0],
    X[:, 1]
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Clustering Dataset")

plt.show()
```

The visualization is particularly useful because there are only two dimensions.

When there are many features, direct visualization becomes difficult. Later, dimensionality-reduction techniques can help create lower-dimensional representations for visualization.

---

# 4. Run K-Means with the Expected Number of Clusters

Suppose four groups appear visually reasonable.

Run:

### Code Cell

```python id="y8z1b2"
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init="auto"
)

labels = kmeans.fit_predict(X)
```

Current scikit-learn uses `n_clusters` to specify the number of clusters and `n_init` to control how many initializations are considered. With `n_init="auto"`, the number of runs depends on the initialization method.

---

# 5. Inspect the Cluster Sizes

The labels identify the cluster assigned to each observation.

Count them:

### Code Cell

```python id="k2hqnw"
cluster_counts = pd.Series(
    labels
).value_counts().sort_index()

print(cluster_counts)
```

The result might look approximately like:

```text
0    125
1    124
2    126
3    125
```

The exact numbers depend on the generated data.

---

# 6. Question 2 — Interpret Cluster Sizes

Why might it be useful to examine how many observations belong to each cluster?

### LLM Hint Prompt

> Explain why examining the number of observations assigned to each cluster can be useful after K-Means. Include both balanced clusters and the possibility of one very small cluster.

<details>
<summary>Solution</summary>

Cluster sizes provide information about the structure discovered by the algorithm.

If clusters have similar sizes, the grouping may be reasonably balanced.

If one cluster is extremely small, it may represent:

* a genuinely small group;
* an unusual region of the data;
* or potentially an issue with the chosen number of clusters or algorithm.

Cluster size alone does not determine whether a clustering is good.

</details>

---

# 7. Visualize the Clusters

### Code Cell

```python id="a0y2c6"
plt.figure(figsize=(8, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    s=200
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means with K = 4")

plt.show()
```

The colored points represent cluster assignments.

The large `X` markers represent the centroids.

---

# 8. Inertia

K-Means attempts to create compact clusters.

One measure associated with the K-Means objective is **inertia**.

In scikit-learn, it is available through:

```python
kmeans.inertia_
```

### Code Cell

```python id="l2gg4m"
print("Inertia:", kmeans.inertia_)
```

Conceptually, inertia measures the total within-cluster squared distance from observations to their assigned cluster centers.

A simplified expression is:

$$
Inertia
=
\sum_{i=1}^{n}
\left\|
x_i-\mu_{c_i}
\right\|^2
$$

where:

* \(x_i\) is an observation;
* \(\mu_{c_i}\) is the centroid of its assigned cluster.

Lower inertia means that observations are, overall, closer to their assigned centroids.

---

# 9. Why Inertia Alone Is Not Enough

Suppose we increase the number of clusters.

With:

```text
K = 1
```

almost all variation must be represented by one centroid.

With:

```text
K = 10
```

the algorithm has many more centroids and can make clusters much smaller.

Therefore, inertia generally decreases as `K` increases.

This creates a problem.

If lower inertia were the only criterion, we could keep increasing K.

Eventually:

$$
K=n
$$

would give every observation its own cluster and an extremely small inertia.

That would not necessarily be a useful clustering.

---

# 10. Question 3 — Why Not Choose the Largest K?

Suppose:

```text
K = 2 → inertia = 1500
K = 3 → inertia = 900
K = 4 → inertia = 500
K = 5 → inertia = 400
K = 6 → inertia = 330
```

Why not always choose `K=6`?

### LLM Hint Prompt

> Explain why K-Means inertia always tends to improve when more clusters are allowed and why this means that the lowest inertia alone cannot determine the best number of clusters.

<details>
<summary>Solution</summary>

More clusters give K-Means greater flexibility to place centroids closer to individual observations.

Therefore inertia normally decreases as K increases.

The smallest inertia would occur when there are enough clusters to represent the observations extremely closely, but that does not mean the resulting grouping is meaningful.

The number of clusters should therefore be considered using additional evidence.

</details>

---

# 11. The Elbow Method

One common technique is the **elbow method**.

Run K-Means for several values of `K`.

### Code Cell

```python id="j7g1zz"
inertias = []

k_values = range(1, 9)

for k in k_values:
    
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init="auto"
    )
    
    model.fit(X)
    
    inertias.append(
        model.inertia_
    )
```

Plot:

### Code Cell

```python id="khdnqs"
plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    inertias,
    marker="o"
)

plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.show()
```

---

# 12. Interpreting the Elbow

The idea is to look for a point where increasing K produces a much smaller improvement.

For example:

```text id="t0zkko"
K=1 → large improvement
K=2 → large improvement
K=3 → large improvement
K=4 → useful improvement
K=5 → small improvement
K=6 → very small improvement
```

The curve may appear to "bend" around:

$$
K=4
$$

This is called the elbow.

The elbow is not always obvious.

Therefore:

> The elbow method is a useful heuristic, not a guarantee of the correct number of clusters.

---

# 13. Question 4 — Find the Elbow

Look at the elbow plot.

1. Where does the largest reduction in inertia occur?
2. Around which K does the curve begin to flatten?
3. What K would be a reasonable candidate?
4. Why is this not an automatic answer?

### LLM Hint Prompt

> Help me interpret an elbow plot for K-Means. Explain how to identify a point where the reduction in inertia becomes smaller and why the elbow method should be treated as a heuristic rather than proof of the correct number of clusters.

<details>
<summary>Solution</summary>

The candidate K is the point where increasing the number of clusters begins to produce substantially smaller improvements in inertia.

For the generated data, the elbow should be near the number of underlying groups.

However, the elbow can be ambiguous, and the best K may require additional evaluation and domain knowledge.

</details>

---

# 14. Silhouette Score

Another useful measure is the **silhouette score**.

The silhouette coefficient considers:

1. how close an observation is to points in its own cluster;
2. how far it is from points in neighboring clusters.

The score ranges from:

$$
-1
$$

to:

$$
1
$$

Values closer to `1` generally indicate that observations are well matched to their own cluster and separated from neighboring clusters. Values around zero indicate observations near cluster boundaries, while negative values can indicate potentially inappropriate assignments.

---

# 15. Calculate Silhouette Scores

Run K-Means for several K values.

### Code Cell

```python id="63h7cq"
silhouette_results = []

for k in range(2, 9):
    
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init="auto"
    )
    
    labels_k = model.fit_predict(X)
    
    score = silhouette_score(
        X,
        labels_k
    )
    
    silhouette_results.append({
        "K": k,
        "Silhouette": score
    })

silhouette_results = pd.DataFrame(
    silhouette_results
)

silhouette_results
```

Plot:

### Code Cell

```python id="b8n0gl"
plt.figure(figsize=(8, 5))

plt.plot(
    silhouette_results["K"],
    silhouette_results["Silhouette"],
    marker="o"
)

plt.xlabel("Number of clusters (K)")
plt.ylabel("Average silhouette score")
plt.title("Silhouette Score by K")

plt.show()
```

---

# 16. Question 5 — Compare Elbow and Silhouette

Use both analyses.

Answer:

1. Which K is suggested by the elbow?
2. Which K has the highest silhouette score?
3. Are they the same?
4. What should happen if they disagree?

### LLM Hint Prompt

> Help me compare the elbow method and silhouette score when choosing K for K-Means. Explain what each method measures and why two methods might suggest different values.

<details>
<summary>Solution</summary>

The elbow method looks at the reduction in within-cluster inertia as K increases.

The silhouette score examines both cluster cohesion and separation.

They may suggest different values because they measure different characteristics of the clustering.

When they disagree, additional analysis is required, including visualization and domain knowledge.

</details>

---

# 17. Why Is K Not Always Objective?

Consider two possible interpretations.

### Business interpretation

A company may want:

```text
3 customer segments
```

because three groups are easy to use operationally.

### Statistical interpretation

The data might appear to contain:

```text
4 groups
```

according to a clustering metric.

Both can be reasonable depending on the objective.

Therefore:

> Selecting K is partly a data-analysis decision, not merely a mathematical calculation.

---

# 18. Feature Scaling and Clustering

Consider a dataset containing:

```text
Age:       18–80
Income:    20,000–200,000
```

K-Means uses distances.

The numerical scale of income is much larger than age.

Without scaling, income can dominate the distance calculation.

This does not mean income is inherently more important.

It means that its numerical units contribute more strongly to the distance.

---

# 19. Create a Dataset with Different Feature Scales

### Code Cell

```python id="pr1i82"
rng = np.random.default_rng(42)

age = rng.normal(
    40,
    10,
    300
)

income = rng.normal(
    50000,
    15000,
    300
)

X_scale_demo = pd.DataFrame({
    "Age": age,
    "Income": income
})
```

Inspect:

### Code Cell

```python id="ob2p1o"
X_scale_demo.describe()
```

---

# 20. Standardize the Features

Use:

### Code Cell

```python id="6g0muj"
scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X_scale_demo
)
```

`StandardScaler` transforms each feature using its mean and standard deviation.

Conceptually:

$$
z=
\frac{x-\mu}{\sigma}
$$

where:

* \(x\) is the original value;
* \(\mu\) is the feature mean;
* \(\sigma\) is the feature standard deviation.

---

# 21. Question 6 — Why Scale Features?

Explain:

> Why might K-Means produce a different result before and after scaling?

### LLM Hint Prompt

> Explain why StandardScaler can change K-Means clustering results when features have very different numerical scales. Focus on Euclidean distance and not on the machine-learning target.

<details>
<summary>Solution</summary>

K-Means relies on distances.

Before scaling, a feature with very large numerical values can dominate those distances.

After scaling, the features are placed on a more comparable scale.

Therefore, observations can have different nearest centroids and the resulting clusters can change.

</details>

---

# 22. Scale a Clustering Dataset

Return to the earlier blob dataset and compare.

### Code Cell

```python id="vvdn43"
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Fit K-Means:

### Code Cell

```python id="fve06n"
kmeans_scaled = KMeans(
    n_clusters=4,
    random_state=42,
    n_init="auto"
)

labels_scaled = kmeans_scaled.fit_predict(
    X_scaled
)
```

Compare labels:

### Code Cell

```python id="f0u8un"
print(labels[:20])
print(labels_scaled[:20])
```

Because this particular synthetic dataset has similarly scaled features, the overall structure may remain similar.

The purpose of the experiment is to understand the principle for datasets where feature scales differ substantially.

---

# 23. A Critical Preprocessing Rule

When scaling is used with data that will later be evaluated on unseen observations, the scaling parameters should be learned from the training data.

Conceptually:

```text id="2jxh0y"
Training data
     ↓
fit scaler
     ↓
transform training data

Test data
     ↓
transform using the same scaler
```

The scaler should not be fitted separately to the test data.

This is the same data-leakage principle encountered in supervised learning.

For the unsupervised examples where clustering is being demonstrated on one complete dataset, there may be no separate test set. When a separate validation or test set is used, the same principle applies.

---

# Part B — Hierarchical Clustering in More Detail

# 24. Agglomerative Hierarchical Clustering

The first activity introduced the basic idea.

The more detailed process is:

```text id="7kuh4m"
Start:
Every observation = its own cluster

        ↓

Find clusters that should be merged

        ↓

Merge them

        ↓

Repeat

        ↓

Hierarchy
```

Scikit-learn's agglomerative clustering recursively merges pairs of clusters according to the chosen linkage criterion.

---

# 25. Linkage

A hierarchical algorithm needs a way to decide how distance between clusters is measured.

Common linkage methods include:

* `ward`
* `complete`
* `average`
* `single`

At a high level:

### Ward

Attempts to minimize the increase in within-cluster variance.

### Complete

Uses the maximum pairwise distance between observations in two clusters.

### Average

Uses the average pairwise distance.

### Single

Uses the minimum pairwise distance.

The exact mathematical differences are not required for the basic workflow, but the choice can affect the clustering result.

---

# 26. Run Agglomerative Clustering

### Code Cell

```python id="z9s55j"
hierarchical = AgglomerativeClustering(
    n_clusters=4,
    linkage="ward"
)

hierarchical_labels = hierarchical.fit_predict(
    X_scaled
)
```

The `ward` linkage is a common starting point.

---

# 27. Visualize the Result

### Code Cell

```python id="8kld4m"
plt.figure(figsize=(8, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=hierarchical_labels
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Agglomerative Hierarchical Clustering")

plt.show()
```

---

# 28. Question 7 — Compare the Results

Compare the K-Means and hierarchical clustering plots.

Answer:

1. Do they produce the same number of clusters?
2. Are the individual assignments identical?
3. Why might the results differ?
4. Does a difference mean that one algorithm is automatically wrong?

### LLM Hint Prompt

> Compare K-Means and agglomerative hierarchical clustering applied to the same data. Explain why different algorithms can produce different clusters and why different results do not automatically mean that one algorithm is incorrect.

<details>
<summary>Solution</summary>

Both models were requested to produce four clusters.

Their assignments may differ because they use different procedures for creating groups.

K-Means is based on centroids and iterative assignment.

Agglomerative clustering progressively merges observations or clusters according to a linkage criterion.

Different results do not automatically mean that one method is wrong.

</details>

---

# 29. Dendrogram

A major advantage of hierarchical clustering is that the merging process can be visualized as a **dendrogram**.

Install SciPy if needed.

### Code Cell

```python id="u9w0hm"
!pip install -q scipy
```

Import:

### Code Cell

```python id="k4u4j1"
from scipy.cluster.hierarchy import (
    linkage,
    dendrogram
)
```

Calculate the hierarchy:

### Code Cell

```python id="ce7zpl"
Z = linkage(
    X_scaled,
    method="ward"
)
```

Visualize:

### Code Cell

```python id="w74dsv"
plt.figure(figsize=(12, 6))

dendrogram(Z)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Observations")
plt.ylabel("Distance")

plt.show()
```

---

# 30. Understanding the Dendrogram

The bottom of the dendrogram represents individual observations.

As we move upward:

```text id="y1f0y8"
individual observations
       ↓
small groups
       ↓
larger groups
       ↓
one large group
```

The height at which two groups merge represents the distance or linkage value associated with that merge.

A large vertical jump can suggest that groups being merged are relatively far apart.

---

# 31. Question 8 — Where Could the Hierarchy Be Cut?

Consider a dendrogram with a large vertical gap between two sets of merges.

Why might a horizontal cut through that gap provide a reasonable clustering?

### LLM Hint Prompt

> Explain how a horizontal cut through a dendrogram can create clusters. Focus on why a large gap in merge distance may indicate that several groups were relatively separate before being merged.

<details>
<summary>Solution</summary>

A horizontal cut determines which merges are accepted.

If there is a large increase in merge distance, groups below that level were relatively separate.

Cutting before the large merge can preserve those separate groups and produce a reasonable clustering.

</details>

---

# 32. Try Different Linkage Methods

Run:

### Code Cell

```python id="j0h81t"
linkage_methods = [
    "ward",
    "complete",
    "average",
    "single"
]

for method in linkage_methods:
    
    if method == "ward":
        model = AgglomerativeClustering(
            n_clusters=4,
            linkage=method
        )
    else:
        model = AgglomerativeClustering(
            n_clusters=4,
            linkage=method
        )
    
    labels_method = model.fit_predict(
        X_scaled
    )
    
    print(
        method,
        np.bincount(labels_method)
    )
```

The exact cluster sizes can change with linkage.

---

# 33. Question 9 — Why Does Linkage Matter?

Why might changing:

```text
ward
complete
average
single
```

change the clustering result?

### LLM Hint Prompt

> Explain why different linkage methods in hierarchical clustering can produce different groups. Focus on the fact that each method defines the distance between clusters differently.

<details>
<summary>Solution</summary>

Each linkage method defines the distance between groups differently.

Therefore, the sequence of merges can change.

Because the hierarchy is built from those merges, the final clusters can also change.

</details>

---

# Part C — Anomaly Detection

# 34. What Is Anomaly Detection?

Clustering asks:

> Which observations form groups?

Anomaly detection asks:

> Which observations appear unusually different from the rest?

Examples include:

* unusual credit-card transactions;
* abnormal sensor measurements;
* unusual network activity;
* potentially fraudulent behavior.

The two tasks are related but different.

```text id="jct4ut"
Clustering
     ↓
Find groups

Anomaly Detection
     ↓
Find unusual observations
```

---

# 35. Create Data with Anomalies

Create a normal group:

### Code Cell

```python id="7l2jwh"
X_normal, _ = make_blobs(
    n_samples=300,
    centers=1,
    cluster_std=1.0,
    random_state=42
)
```

Create several unusual points manually:

### Code Cell

```python id="u9ri0u"
X_anomalies = np.array([
    [8, 8],
    [-8, 7],
    [7, -7],
    [-7, -8],
    [9, -6]
])
```

Combine them:

### Code Cell

```python id="iy74wq"
X_anomaly_demo = np.vstack([
    X_normal,
    X_anomalies
])
```

Visualize:

### Code Cell

```python id="n9s7dm"
plt.figure(figsize=(8, 6))

plt.scatter(
    X_anomaly_demo[:, 0],
    X_anomaly_demo[:, 1]
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Dataset with Potential Anomalies")

plt.show()
```

---

# 36. Question 10 — Identify the Potential Anomalies

Looking at the plot:

1. Which points appear unusual?
2. Why?
3. Are they far from the main group?
4. Is visual inspection always practical?

### LLM Hint Prompt

> Help me identify unusual observations in a two-dimensional scatter plot. Explain what makes an observation appear anomalous and why visual inspection becomes difficult with many dimensions or many observations.

<details>
<summary>Solution</summary>

The manually added points far from the main group appear unusual because they are spatially separated from most observations.

Visual inspection works for simple two-dimensional datasets but becomes difficult with many observations or many features.

This motivates automated anomaly-detection methods.

</details>

---

# 37. Isolation Forest

One anomaly-detection method available in scikit-learn is **Isolation Forest**.

The basic idea is different from K-Means.

Isolation Forest uses random partitions of the feature space.

Anomalies tend to be isolated with fewer random splits because they are unusual and separated from many other observations.

---

# 38. Create the Isolation Forest Model

### Code Cell

```python id="flx5cb"
isolation_model = IsolationForest(
    contamination=0.02,
    random_state=42
)
```

The parameter:

```text
contamination=0.02
```

indicates that approximately 2% of the observations are expected to be anomalies for thresholding purposes.

The contamination parameter should therefore be chosen carefully; it is not simply a statement that the detected observations are definitely fraudulent or incorrect.

---

# 39. Fit and Predict

### Code Cell

```python id="xj8h2d"
anomaly_labels = isolation_model.fit_predict(
    X_anomaly_demo
)
```

Isolation Forest returns:

```text
1  → normal observation
-1 → anomaly
```

These labels are documented by scikit-learn's examples and implementation behavior.

Count them:

### Code Cell

```python id="7my4v9"
print(
    pd.Series(anomaly_labels).value_counts()
)
```

---

# 40. Visualize the Anomalies

### Code Cell

```python id="2vrw5u"
normal = anomaly_labels == 1
anomalies = anomaly_labels == -1

plt.figure(figsize=(8, 6))

plt.scatter(
    X_anomaly_demo[normal, 0],
    X_anomaly_demo[normal, 1],
    label="Normal"
)

plt.scatter(
    X_anomaly_demo[anomalies, 0],
    X_anomaly_demo[anomalies, 1],
    marker="x",
    s=100,
    label="Anomaly"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Isolation Forest")

plt.legend()
plt.show()
```

---

# 41. Question 11 — Anomaly vs Cluster

Suppose an observation is far away from all clusters.

Would K-Means and Isolation Forest answer the same question?

### LLM Hint Prompt

> Explain the conceptual difference between asking K-Means to group observations and asking Isolation Forest to identify unusual observations. Use an isolated point as an example.

<details>
<summary>Solution</summary>

No.

K-Means asks which cluster an observation belongs to.

Isolation Forest asks whether an observation appears unusual relative to the rest of the data.

An isolated point may still be assigned to a K-Means cluster, while an anomaly-detection algorithm may identify it as unusual.

</details>

---

# 42. Important Caution About Anomalies

An anomaly is not automatically:

```text
wrong
```

or:

```text
fraud
```

or:

```text
bad data
```

It means that the observation appears unusual according to the model.

For example:

```text id="6bn9w2"
Normal sensor readings:
20, 21, 22, 21, 20

Unusual reading:
75
```

The value 75 may be:

* a sensor error;
* a legitimate unusual event;
* an equipment failure;
* or something else.

The model identifies an observation for investigation.

It does not automatically provide the explanation.

---

# Part D — Association Rule Learning

# 43. A Different Type of Unsupervised Learning

Clustering and anomaly detection work with observations represented by features.

Association rule learning works differently.

It often deals with **transactions containing items**.

For example:

```text id="wunnhk"
Transaction 1:
Bread, Butter, Milk

Transaction 2:
Bread, Butter

Transaction 3:
Bread, Milk

Transaction 4:
Butter, Milk

Transaction 5:
Bread, Butter, Milk
```

The goal is to discover patterns such as:

$$
Bread \Rightarrow Butter
$$

This means:

> Transactions that contain bread frequently also contain butter.

It does **not** automatically mean:

> Buying bread causes someone to buy butter.

Again, association is not causation.

---

# 44. Install `mlxtend`

A convenient Python library for association-rule learning is `mlxtend`.

### Code Cell

```python id="5yab76"
!pip install -q mlxtend
```

Import:

### Code Cell

```python id="5iy2gy"
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import (
    apriori,
    association_rules
)
```

---

# 45. Create a Transaction Dataset

### Code Cell

```python id="5kz8r1"
transactions = [
    ["Bread", "Butter", "Milk"],
    ["Bread", "Butter"],
    ["Bread", "Milk"],
    ["Butter", "Milk"],
    ["Bread", "Butter", "Milk"],
    ["Bread", "Butter"],
    ["Milk"],
    ["Bread", "Milk"]
]
```

Each list represents one transaction.

---

# 46. Convert Transactions into a Matrix

Machine-learning algorithms need a numerical representation.

Use `TransactionEncoder`.

### Code Cell

```python id="t7f3vb"
encoder = TransactionEncoder()

encoded = encoder.fit(
    transactions
).transform(
    transactions
)
```

Create a DataFrame:

### Code Cell

```python id="k7ff4h"
transaction_df = pd.DataFrame(
    encoded,
    columns=encoder.columns_
)

transaction_df
```

The result will look approximately like:

```text
   Bread  Butter  Milk
0   True    True   True
1   True    True  False
2   True   False   True
...
```

`True` means the item occurs in that transaction.

---

# 47. Frequent Itemsets

The first step is to find **frequent itemsets**.

An itemset is a combination of items.

Examples:

```text
{Bread}
{Butter}
{Milk}
{Bread, Butter}
{Bread, Milk}
```

Run Apriori:

### Code Cell

```python id="0c7qq0"
frequent_itemsets = apriori(
    transaction_df,
    min_support=0.3,
    use_colnames=True
)

frequent_itemsets
```

---

# 48. Support

**Support** measures how frequently an itemset occurs in the transactions.

The formula is:

$$
Support(A)=
\frac{
\text{Number of transactions containing A}
}{
\text{Total number of transactions}
}
$$

Suppose:

```text
8 total transactions
```

and:

```text
Bread appears in 6
```

Then:

$$
Support(Bread)=
\frac{6}{8}
=
0.75
$$

So:

$$
\boxed{Support(Bread)=0.75}
$$

---

# 49. Question 12 — Calculate Support

Suppose there are 10 transactions.

The item `Bread` appears in 7.

Calculate:

$$
Support(Bread)
$$

### LLM Hint Prompt

> Help me calculate the support of an item that appears in 7 out of 10 transactions. Explain what support means before giving the final calculation.

<details>
<summary>Solution</summary>

Support is:

$$
Support=
\frac{
\text{transactions containing the item}
}{
\text{total transactions}
}
$$

Therefore:

$$
Support(Bread)
=
\frac{7}{10}
=
0.7
$$

So:

$$
\boxed{Support=0.70}
$$

</details>

---

# 50. Generate Association Rules

Now generate rules.

### Code Cell

```python id="qy1x2q"
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.5
)

rules
```

This may produce rules such as:

```text
Bread → Butter
Butter → Bread
Bread → Milk
```

The actual output depends on the transaction data and selected thresholds.

---

# 51. Confidence

Confidence measures how often the consequent appears when the antecedent appears.

For:

$$
A\Rightarrow B
$$

the formula is:

$$
Confidence(A\Rightarrow B)
=
\frac{Support(A\cap B)}
{Support(A)}
$$

Suppose:

```text
Bread appears in 7 transactions.
Bread and Butter appear together in 5 transactions.
```

Then:

$$
Confidence(Bread\Rightarrow Butter)
=
\frac{5}{7}
$$

$$
\approx0.714
$$

So:

$$
\boxed{Confidence\approx71.4\%}
$$

Interpretation:

> Among transactions containing bread, about 71.4% also contain butter.

---

# 52. Question 13 — Calculate Confidence

There are:

```text
10 transactions
```

Bread appears in:

```text
8
```

Bread and Butter appear together in:

```text
6
```

Calculate:

$$
Confidence(Bread\Rightarrow Butter)
$$

### LLM Hint Prompt

> Help me calculate the confidence of the rule Bread → Butter when Bread appears in 8 transactions and Bread and Butter occur together in 6. Explain why the denominator is the number of transactions containing Bread.

<details>
<summary>Solution</summary>

Use:

$$
Confidence(Bread\Rightarrow Butter)
=
\frac{
Support(Bread\cap Butter)
}{
Support(Bread)
}
$$

Using counts:

$$
Confidence=
\frac{6}{8}
=
0.75
$$

Therefore:

$$
\boxed{Confidence=75\%}
$$

This means that among transactions containing bread, 75% also contain butter.

</details>

---

# 53. Lift

Confidence alone can sometimes be misleading.

Suppose butter is already extremely common.

We therefore want to know whether butter appears with bread **more often than would be expected based on butter's overall popularity**.

This is where **lift** is useful.

For:

$$
A\Rightarrow B
$$

lift is:

$$
Lift(A\Rightarrow B)
=
\frac{
Confidence(A\Rightarrow B)
}{
Support(B)
}
$$

Interpretation:

### Lift > 1

The items occur together more often than expected under independence.

### Lift = 1

The relationship is approximately what would be expected if the items were independent.

### Lift < 1

The items occur together less often than expected.

---

# 54. Worked Lift Example

Suppose:

$$
Confidence(Bread\Rightarrow Butter)=0.75
$$

and:

$$
Support(Butter)=0.50
$$

Then:

$$
Lift=
\frac{0.75}{0.50}
$$

$$
=1.5
$$

Therefore:

$$
\boxed{Lift=1.5}
$$

A lift of 1.5 indicates that bread and butter co-occur more often than would be expected under independence, according to this measure.

---

# 55. Question 14 — Interpret Lift

Suppose the following rules are produced:

| Rule           | Confidence | Lift |
| -------------- | ---------: | ---: |
| Bread → Butter |       0.75 | 1.50 |
| Milk → Butter  |       0.70 | 1.05 |
| Butter → Bread |       0.80 | 1.60 |

Which rule has the strongest association according to lift?

### LLM Hint Prompt

> Help me compare the three association rules using lift. Explain why lift rather than confidence alone can be useful for comparing associations.

<details>
<summary>Solution</summary>

The rule:

$$
Butter\Rightarrow Bread
$$

has the highest lift:

$$
1.60
$$

Therefore, according to lift, it has the strongest association among the three rules.

Confidence alone would not necessarily produce the same ranking because it does not account for how common the consequent is overall.

</details>

---

# 56. Association Is Not Causation

Suppose:

$$
Bread\Rightarrow Butter
$$

has:

$$
Lift=1.8
$$

This does not mean:

> Buying bread causes people to buy butter.

It means that bread and butter occur together more frequently than expected according to the lift calculation.

Possible explanations could include:

* customers commonly purchase them together;
* they belong to the same shopping occasion;
* another factor influences both purchases.

Association rule learning identifies patterns.

It does not establish causal relationships.

---

# 57. Question 15 — Association or Causation?

Suppose a supermarket discovers:

$$
Bread\Rightarrow Butter
$$

with high confidence and lift.

Can the supermarket conclude:

> "Buying bread causes customers to buy butter"?

### LLM Hint Prompt

> Explain why an association rule with high confidence and lift does not establish a causal relationship. Use the distinction between co-occurrence and causation.

<details>
<summary>Solution</summary>

No.

The rule describes an association or co-occurrence pattern.

It does not prove that buying bread causes customers to buy butter.

Other variables or purchasing habits could explain the relationship.

</details>

---

# Part E — Comparing the Unsupervised-Learning Techniques

# 58. Different Questions, Different Methods

The techniques covered in this session solve different problems.

| Technique                | Main Question                                                  | Typical Output                  |
| ------------------------ | -------------------------------------------------------------- | ------------------------------- |
| K-Means                  | Which observations belong to groups?                           | Cluster labels + centroids      |
| Hierarchical Clustering  | How are observations/groups related at different levels?       | Hierarchy + cluster labels      |
| Anomaly Detection        | Which observations are unusual?                                | Normal/anomaly                  |
| Association Rules        | Which items tend to occur together?                            | Rules + support/confidence/lift |
| Dimensionality Reduction | Can the data be represented using fewer dimensions?            | Reduced representation          |
| Density Estimation       | What distribution describes the data?                          | Estimated density/model         |
| Generative Modeling      | Can new observations be generated from a learned distribution? | Generated observations          |

These methods should not be treated as interchangeable.

---

# 59. Clustering vs Anomaly Detection

Consider a dataset of customer spending.

### Clustering

The question is:

> What groups of customers exist?

Possible output:

```text
Cluster 0
Cluster 1
Cluster 2
```

### Anomaly Detection

The question is:

> Which customers have unusually different behavior?

Possible output:

```text
Normal
Normal
Anomaly
Normal
...
```

A customer can belong to a cluster and still be unusual relative to that cluster or the overall dataset.

---

# 60. Clustering vs Association Rules

These methods use data differently.

### Clustering

Works with observations represented by features.

Example:

```text
Customer
Age
Income
Spending
```

The goal is to group similar observations.

### Association Rules

Works with transactions containing items.

Example:

```text
Transaction 1:
Bread, Milk, Butter
```

The goal is to discover co-occurrence relationships between items.

---

# 61. Question 16 — Choose the Technique

Choose an appropriate unsupervised-learning technique.

### Scenario A

A company wants to divide customers into groups based on age, income, and spending.

### Scenario B

A bank wants to identify unusual credit-card transactions.

### Scenario C

A supermarket wants to discover products that are frequently purchased together.

### Scenario D

A dataset contains 200 numerical features and needs to be visualized in two dimensions.

### LLM Hint Prompt

> For each scenario, choose between clustering, anomaly detection, association rule learning, and dimensionality reduction. Explain the question being asked in each scenario before selecting the technique.

<details>
<summary>Solution</summary>

**Scenario A → Clustering**

The objective is to discover groups of similar customers.

**Scenario B → Anomaly Detection**

The objective is to identify unusual transactions.

**Scenario C → Association Rule Learning**

The objective is to find items that frequently occur together.

**Scenario D → Dimensionality Reduction**

The objective is to represent high-dimensional data using fewer dimensions, such as two dimensions for visualization.

</details>

---

# 62. Choosing Between K-Means and Hierarchical Clustering

Consider:

### Dataset A

```text
1,000,000 observations
20 numerical features
```

### Dataset B

```text
30 observations
4 numerical features
```

There is no universal rule, but K-Means may be attractive for very large datasets because it is designed to form clusters around centroids efficiently.

Hierarchical clustering may be attractive for small datasets where examining relationships between observations and the resulting hierarchy is useful.

The choice also depends on:

* cluster structure;
* feature scaling;
* desired interpretation;
* computational requirements;
* distance/linkage assumptions.

---

# 63. Question 17 — Algorithm Selection

A dataset contains 50 observations and the main interest is understanding which observations are closely related to one another at several levels.

Would hierarchical clustering be a reasonable candidate?

Explain.

### LLM Hint Prompt

> Explain whether hierarchical clustering is suitable for a small dataset when the relationships among observations at multiple levels are important. Compare this briefly with K-Means.

<details>
<summary>Solution</summary>

Yes.

Hierarchical clustering is a reasonable candidate because it creates a hierarchy of relationships and can be visualized with a dendrogram.

K-Means instead directly creates a selected number of centroid-based clusters.

</details>

---

# 64. A More Detailed K-Means Experiment

Return to the blob data.

Try several K values:

```text
2
3
4
5
6
```

Create a results table containing:

```text
K
Inertia
Silhouette
```

### Code Cell

```python id="2sh4c6"
results = []

for k in range(2, 7):
    
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init="auto"
    )
    
    labels_k = model.fit_predict(X_scaled)
    
    inertia = model.inertia_
    
    silhouette = silhouette_score(
        X_scaled,
        labels_k
    )
    
    results.append({
        "K": k,
        "Inertia": inertia,
        "Silhouette": silhouette
    })

results_df = pd.DataFrame(results)

results_df
```

---

# 65. Question 18 — Model Selection

Use the results table.

Answer:

1. Which K has the lowest inertia?
2. Which K has the highest silhouette score?
3. Why might these answers not be identical?
4. Which other information should be considered?

### LLM Hint Prompt

> Help me interpret K-Means model-selection results containing inertia and silhouette score. Explain why lower inertia alone favors larger K and why silhouette score provides a different perspective.

<details>
<summary>Solution</summary>

The largest K tested will generally have the lowest inertia because more clusters allow the data to be represented more closely.

The highest silhouette score identifies the K that provides the strongest average balance between within-cluster similarity and separation from neighboring clusters.

Other useful information includes:

* visualization;
* cluster sizes;
* domain knowledge;
* the purpose of the clustering.

</details>

---

# 66. Cluster Profiling

Once clusters are created, the numerical cluster labels are often not enough.

Suppose a customer dataset contains:

```text
Age
Income
Spending
```

and K-Means creates:

```text
Cluster 0
Cluster 1
Cluster 2
```

It is useful to calculate the average characteristics of each group.

Example:

```python id="v7lh75"
customer_df["Cluster"] = labels

profile = customer_df.groupby(
    "Cluster"
)[
    [
        "Age",
        "Income",
        "Spending"
    ]
].mean()

profile
```

The resulting summary can provide meaning to the clusters.

---

# 67. Question 19 — Why Profile Clusters?

Why is a cluster label such as `Cluster 2` not sufficient for interpreting a real-world clustering result?

### LLM Hint Prompt

> Explain why cluster labels are only identifiers and why summary statistics of each cluster are necessary to interpret what the groups actually represent.

<details>
<summary>Solution</summary>

Cluster numbers have no inherent meaning.

Cluster profiling provides information about the characteristics of observations in each group.

For example, it may reveal that one cluster contains younger customers with lower income, while another contains older customers with higher income.

</details>

---

# Part F — Integrated Analysis

# 68. Complete Unsupervised-Learning Workflow

The workflow can now be summarized as:

```text id="y7q8h2"
Understand the problem
        ↓
Identify the available features
        ↓
Explore the data
        ↓
Visualize when possible
        ↓
Choose an appropriate unsupervised task
        ↓
Prepare / scale data when necessary
        ↓
Fit the algorithm
        ↓
Inspect the discovered structure
        ↓
Evaluate the result
        ↓
Interpret the output
        ↓
Check limitations
```

This workflow is different from supervised learning because there is no target variable guiding the model.

---

# 69. Integrated Scenario

Consider the following dataset:

```text
Customer_ID
Age
Income
Spending
Number_of_Purchases
```

Suppose the objective is to understand customer groups.

Answer:

1. Which variable should not normally be used as a clustering feature?
2. Which variables could be used?
3. Would clustering be appropriate?
4. Would K-Means be a possible starting point?
5. Why might scaling be important?
6. How could the resulting clusters be interpreted?

### LLM Hint Prompt

> Analyze a customer dataset containing Customer_ID, Age, Income, Spending, and Number_of_Purchases. Explain which variables should be used for clustering, why the identifier should generally be excluded, why scaling may matter, and how cluster profiles could be interpreted.

<details>
<summary>Solution</summary>

1. `Customer_ID` should generally not be used because it is an identifier rather than a meaningful measurement of customer similarity.
2. Possible features include:

   * Age
   * Income
   * Spending
   * Number_of_Purchases
3. Yes. Clustering is appropriate if the objective is to discover groups of similar customers.
4. K-Means can be a reasonable starting point for numerical features.
5. The variables may have very different scales, and K-Means relies on distances.
6. After clustering, calculate descriptive statistics for each cluster and interpret the groups based on their feature values.

</details>

---

# 70. Integrated Scenario — Anomaly Detection

A bank has transaction data containing:

```text
Transaction amount
Transaction hour
Distance from normal location
Number of transactions in one hour
```

The objective is to identify potentially unusual transactions.

Answer:

1. Is clustering necessarily the best first approach?
2. Could anomaly detection be appropriate?
3. Why might an unusual transaction still be legitimate?
4. Why should an anomaly model be considered a screening tool rather than automatic proof of fraud?

### LLM Hint Prompt

> Analyze a banking transaction anomaly-detection problem. Explain why anomaly detection may be appropriate, but why an anomaly should not automatically be interpreted as fraud.

<details>
<summary>Solution</summary>

1. Clustering is not necessarily the best first approach because the primary question is which observations are unusual rather than which observations form groups.
2. Anomaly detection is appropriate.
3. A transaction may be unusual because of travel, an unusual purchase, a special event, or another legitimate reason.
4. Anomaly detection identifies unusual observations according to the model. Additional investigation is needed to determine why the observation is unusual.

</details>

---

# 71. Integrated Scenario — Association Rules

A supermarket records:

```text
Transaction ID
Product
```

with multiple products per transaction.

The objective is to find products that are frequently purchased together.

Answer:

1. Is this a clustering problem?
2. Is anomaly detection the primary objective?
3. Which technique is appropriate?
4. Which three measures are particularly important?
5. Does a strong rule prove that one product causes another purchase?

### LLM Hint Prompt

> Analyze a supermarket transaction dataset and determine why association rule learning is more appropriate than clustering when the objective is to discover products that frequently occur together. Explain support, confidence, and lift.

<details>
<summary>Solution</summary>

1. No. The objective is not primarily to group transactions or customers.
2. No. The objective is not to identify unusual transactions.
3. Association rule learning is appropriate.
4. Support, confidence, and lift.
5. No. Association indicates co-occurrence and does not prove causation.

</details>

---

# 72. Final Comparison

The most important distinctions are:

```text id="x9w0qk"
CLUSTERING
Question:
What groups exist?

Examples:
K-Means
Hierarchical Clustering


ANOMALY DETECTION
Question:
What observations are unusual?

Example:
Isolation Forest


ASSOCIATION RULE LEARNING
Question:
What items tend to occur together?

Examples:
Apriori
Association Rules


DIMENSIONALITY REDUCTION
Question:
Can the data be represented with fewer dimensions?

Examples:
PCA
t-SNE
```

Each technique solves a different problem.

---

# 73. Final Knowledge Check

Answer the following questions without immediately looking at the solutions.

### Q1

What is the main purpose of K-Means?

### Q2

What does K represent?

### Q3

What is a centroid?

### Q4

Why does inertia usually decrease as K increases?

### Q5

Why can we not simply choose the K with the lowest inertia?

### Q6

What does the elbow method try to identify?

### Q7

What does the silhouette score measure conceptually?

### Q8

Why can feature scaling change K-Means results?

### Q9

What is the main idea behind agglomerative hierarchical clustering?

### Q10

What is a dendrogram?

### Q11

What does linkage mean in hierarchical clustering?

### Q12

How is anomaly detection different from clustering?

### Q13

What does Isolation Forest attempt to identify?

### Q14

What does `contamination` represent conceptually?

### Q15

What is support in association rule learning?

### Q16

What is confidence?

### Q17

What is lift?

### Q18

Does a high lift prove causation?

### Q19

Why should cluster labels such as `0`, `1`, and `2` not be interpreted as meaningful names?

### Q20

Which technique would be most appropriate for discovering products frequently purchased together?

---

# Final Answers

<details>
<summary>Q1 — What is the main purpose of K-Means?</summary>

To divide observations into a chosen number of clusters based on their similarity to cluster centroids.

</details>

<details>
<summary>Q2 — What does K represent?</summary>

K represents the number of clusters that K-Means is instructed to create.

</details>

<details>
<summary>Q3 — What is a centroid?</summary>

A centroid is the mean position of the observations assigned to a cluster.

</details>

<details>
<summary>Q4 — Why does inertia decrease as K increases?</summary>

More clusters give K-Means more centroids and therefore more flexibility to place observations closer to their assigned centers.

</details>

<details>
<summary>Q5 — Why not choose the lowest inertia?</summary>

Because inertia generally improves as more clusters are added. A very large K can fit the data closely without producing meaningful groups.

</details>

<details>
<summary>Q6 — What does the elbow method identify?</summary>

It looks for a point where increasing K begins to produce substantially smaller reductions in inertia.

</details>

<details>
<summary>Q7 — What does silhouette score measure?</summary>

It measures how well observations fit within their assigned clusters while considering their separation from neighboring clusters.

</details>

<details>
<summary>Q8 — Why can scaling change K-Means?</summary>

K-Means uses distances. Features with larger numerical scales can dominate the distance calculation.

</details>

<details>
<summary>Q9 — What is agglomerative hierarchical clustering?</summary>

It starts with individual observations and progressively merges observations or clusters to create a hierarchy.

</details>

<details>
<summary>Q10 — What is a dendrogram?</summary>

A dendrogram is a visualization of the hierarchy created during hierarchical clustering.

</details>

<details>
<summary>Q11 — What is linkage?</summary>

Linkage defines how the distance between two groups or clusters is calculated during hierarchical clustering.

</details>

<details>
<summary>Q12 — How is anomaly detection different from clustering?</summary>

Clustering focuses on finding groups, while anomaly detection focuses on identifying observations that are unusually different from the rest.

</details>

<details>
<summary>Q13 — What does Isolation Forest identify?</summary>

It identifies observations that can be isolated unusually quickly through random recursive partitioning of the feature space.

</details>

<details>
<summary>Q14 — What does contamination represent?</summary>

It represents the assumed or expected proportion of anomalous observations used to determine the anomaly threshold.

</details>

<details>
<summary>Q15 — What is support?</summary>

Support measures how frequently an itemset occurs in the full set of transactions.

</details>

<details>
<summary>Q16 — What is confidence?</summary>

Confidence measures how often the consequent appears among transactions containing the antecedent.

</details>

<details>
<summary>Q17 — What is lift?</summary>

Lift compares the observed confidence of a rule with the overall frequency of the consequent and indicates how much stronger the association is than would be expected under independence.

</details>

<details>
<summary>Q18 — Does high lift prove causation?</summary>

No. Lift measures association and co-occurrence, not causation.

</details>

<details>
<summary>Q19 — Why are cluster labels not meaningful names?</summary>

They are arbitrary identifiers assigned by the algorithm. Cluster `0` does not inherently represent a particular real-world group.

</details>

<details>
<summary>Q20 — Which technique finds frequently purchased products?</summary>

Association rule learning.

</details>

---

# 74. Final Reflection

Consider the following statement:

> **"Unsupervised learning is not about predicting a known answer; it is about finding useful structure in data."**

Explain how each of the following represents a different type of structure:

```text
Clustering
Anomaly Detection
Association Rule Learning
```

### LLM Hint Prompt

> Help me explain the difference between clustering, anomaly detection, and association rule learning using one real-world example for each. Focus on the different questions each technique answers.

<details>
<summary>Possible Answer</summary>

Clustering discovers groups of similar observations.

Anomaly detection identifies observations that appear unusually different from the rest.

Association rule learning identifies items or events that frequently occur together.

For example:

```text
Clustering:
Group customers into similar segments.

Anomaly Detection:
Find unusual credit-card transactions.

Association Rules:
Find products frequently purchased together.
```

</details>

---

# 75. Activity Summary

The activity has extended the basic concepts from Activity 1.

The workflow for clustering is:

```text
Choose features
     ↓
Inspect the data
     ↓
Consider feature scaling
     ↓
Try candidate K values
     ↓
Fit K-Means
     ↓
Examine inertia
     ↓
Examine silhouette score
     ↓
Visualize clusters
     ↓
Interpret cluster profiles
```

Hierarchical clustering adds:

```text
Observations
     ↓
Progressive merging
     ↓
Hierarchy
     ↓
Dendrogram
     ↓
Choose a cut
     ↓
Clusters
```

Anomaly detection changes the question:

```text
Observations
     ↓
Find unusual cases
     ↓
Normal / anomaly
```

Association rule learning uses a different representation:

```text
Transactions
     ↓
Frequent itemsets
     ↓
Rules
     ↓
Support
Confidence
Lift
```

The most important principle is:

> **The appropriate unsupervised-learning technique depends on the question being asked.**

A clustering algorithm is appropriate when the objective is to discover groups.

An anomaly-detection algorithm is appropriate when the objective is to identify unusual observations.

Association rule learning is appropriate when the objective is to discover recurring co-occurrence relationships among items.
