# Activity 1 — Introduction to Unsupervised Learning with Clustering

## Learning Objectives

By the end of this activity, the following concepts should be understood:

* the difference between supervised and unsupervised learning;
* the idea of learning structure without a target variable;
* clustering as a form of unsupervised learning;
* similarity and distance;
* Euclidean distance;
* the basic idea of K-Means clustering;
* the meaning of a cluster centroid;
* the meaning of cluster labels;
* how to visualize clusters;
* the basic idea of hierarchical clustering;
* the difference between K-Means and hierarchical clustering;
* some limitations of clustering.

The activity uses a small synthetic dataset so that the structure of the data can be visualized clearly.

---

# 1. What Is Unsupervised Learning?

In supervised learning, a dataset contains both:

```text
Features
   +
Known target
```

The model learns a relationship between the features and the target.

For example, in regression:

```text
Car characteristics
       ↓
Regression model
       ↓
Predicted MPG
```

In classification:

```text
Flower measurements
       ↓
Classification model
       ↓
Predicted species
```

In both cases, a target value is available during training.

Unsupervised learning is different.

There is no target variable guiding the learning process.

Instead, the algorithm receives data such as:

```text
Feature 1
Feature 2
Feature 3
...
```

and attempts to discover useful structure.

A simplified view is:

```text
Supervised learning

X + known y
     ↓
learn relationship
     ↓
predict y


Unsupervised learning

X only
     ↓
find structure
```

Examples of unsupervised-learning tasks include:

* clustering;
* anomaly detection;
* association rule learning;
* dimensionality reduction;
* density estimation.

This activity focuses on **clustering**.

---

# 2. What Is Clustering?

Clustering attempts to divide observations into groups based on their similarity.

For example, suppose a dataset contains information about customers:

```text
Customer
Age
Income
Spending
```

There may be groups such as:

```text
Group 1:
younger customers with lower income

Group 2:
middle-income customers with moderate spending

Group 3:
higher-income customers with high spending
```

If the dataset does not contain a column called:

```text
Customer_Group
```

a clustering algorithm can attempt to discover such groups from the available features.

The important idea is:

> **Clustering groups observations that are similar according to the information and distance measure used by the algorithm.**

---

# 3. Clustering Does Not Know the "Correct" Groups

A clustering algorithm is not given the correct cluster labels.

For example, the following data could be:

```text
Observation 1
Observation 2
Observation 3
...
```

The algorithm does not know in advance:

```text
Group A
Group B
Group C
```

Instead, it tries to find structure.

This is different from classification.

### Classification

```text
Known:
Observation → Class A

Observation → Class B
```

### Clustering

```text
Observations
     ↓
algorithm discovers groups
```

This distinction is fundamental.

---

# 4. Similarity and Distance

To decide which observations are similar, a clustering algorithm needs a way to measure distance or similarity.

A common distance measure is **Euclidean distance**.

Suppose two observations have two features:

```text
A = (1, 2)

B = (4, 6)
```

The Euclidean distance is:

$$
d(A,B)=
\sqrt{
(4-1)^2+(6-2)^2
}
$$

Calculate the differences:

$$
4-1=3
$$

$$
6-2=4
$$

Therefore:

$$
d(A,B)=
\sqrt{3^2+4^2}
$$

$$
d(A,B)=
\sqrt{9+16}
$$

$$
d(A,B)=
\sqrt{25}
$$

$$
\boxed{d(A,B)=5}
$$

The smaller the distance, the more similar the observations are under this particular distance measure.

---

# 5. Question 1 — Calculate a Distance

Consider:

```text
A = (2, 3)

B = (5, 7)
```

Calculate the Euclidean distance.

### LLM Hint Prompt

> Help me calculate the Euclidean distance between A = (2, 3) and B = (5, 7). Show the formula and guide me through the calculation step by step, but let me try each arithmetic step myself.

<details>
<summary>Solution</summary>

Use:

$$
d(A,B)=
\sqrt{
(x_2-x_1)^2+
(y_2-y_1)^2
}
$$

Therefore:

$$
d=
\sqrt{
(5-2)^2+
(7-3)^2
}
$$

$$
=
\sqrt{3^2+4^2}
$$

$$
=
\sqrt{9+16}
$$

$$
=
\sqrt{25}
$$

$$
\boxed{d=5}
$$

</details>

---

# 6. Why Does Distance Matter in Clustering?

Suppose we have:

```text
A = (2, 3)
B = (2.5, 3.2)
C = (9, 10)
```

A is very close to B.

A is much farther from C.

A clustering algorithm can therefore use distance to identify:

```text
A and B → similar
A and C → less similar
```

This leads to the basic idea:

```text
small distance
     ↓
more similar

large distance
     ↓
less similar
```

Different clustering algorithms use distance or similarity in different ways.

K-Means is based on assigning observations to nearby cluster centers.

---

# 7. Create a Dataset

For the first clustering experiment, use a synthetic dataset.

Scikit-learn provides `make_blobs()` for generating datasets with groups of observations around specified centers. It is designed specifically for clustering examples. ([scikit-learn.org](https://scikit-learn.org/dev/modules/generated/sklearn.datasets.make_blobs.html))

Because the dataset is synthetic, the groups will be easy to visualize.

---

# 8. Install the Required Libraries

This activity is designed for Google Colab.

Run the installation cell separately.

### Code Cell

```python id="p1x5nd"
!pip install -q scikit-learn pandas matplotlib
```

---

# 9. Import the Libraries

### Code Cell

```python id="g7q2vw"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
```

The main components are:

### NumPy

Used for numerical operations.

### pandas

Used for tabular data.

### matplotlib

Used to visualize the observations.

### scikit-learn

Provides:

* the synthetic dataset;
* K-Means;
* hierarchical clustering.

---

# 10. Generate the Data

Run:

### Code Cell

```python id="k3r7hs"
X, _ = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=1.2,
    random_state=42
)
```

The function returns two objects:

```text
X
```

contains the feature values.

The second object contains the generated group information.

It is assigned to:

```python id="h0q6xk"
_
```

because it will not be used by the clustering algorithm.

This is important.

We are deliberately asking an **unsupervised** algorithm to discover groups without giving it the generated group labels.

The current scikit-learn documentation describes `make_blobs` as a function for generating isotropic Gaussian blobs for clustering and supports parameters such as `n_samples`, `centers`, `cluster_std`, and `random_state`. ([scikit-learn.org](https://scikit-learn.org/dev/modules/generated/sklearn.datasets.make_blobs.html))

---

# 11. Inspect the Shape of the Data

### Code Cell

```python id="8m4x4g"
print(X.shape)
```

The expected result is approximately:

```text
(300, 2)
```

This means:

```text
300 observations
2 features
```

Each observation therefore has two values.

For example:

```text
Observation 1 → [1.2, 3.5]
Observation 2 → [7.1, 8.2]
...
```

---

# 12. Question 2 — Understand the Shape

If:

```text
X.shape = (300, 2)
```

answer:

1. How many observations are there?
2. How many features are there?
3. Why is a two-feature dataset convenient for visualization?

### LLM Hint Prompt

> Explain what `X.shape = (300, 2)` means in a machine-learning dataset. Then explain why a dataset with two numerical features is useful for demonstrating clustering visually.

<details>
<summary>Solution</summary>

There are:

* 300 observations;
* 2 features.

A two-feature dataset is convenient because the two features can be plotted directly on the x-axis and y-axis of a two-dimensional graph.

</details>

---

# 13. Visualize the Data

Create a scatter plot.

### Code Cell

```python id="q8oe7s"
plt.figure(figsize=(8, 6))

plt.scatter(
    X[:, 0],
    X[:, 1]
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Synthetic Dataset")

plt.show()
```

Each point represents one observation.

The two coordinates represent the two features.

---

# 14. Question 3 — Examine the Data

Look at the scatter plot.

Answer:

1. Does the data appear to contain groups?
2. Approximately how many groups can be seen?
3. Are the groups labeled?
4. How could an algorithm identify the groups without labels?

### LLM Hint Prompt

> I have a two-dimensional scatter plot containing several visible groups, but no class labels are being used. Help me reason about how an unsupervised clustering algorithm could discover these groups using distances between observations.

<details>
<summary>Solution</summary>

The data contain visible groups.

There are approximately three groups.

The groups do not have labels available to the clustering algorithm.

A clustering algorithm can use the locations of the observations and distances between them to identify groups of nearby observations.

</details>

---

# 15. K-Means Clustering

One of the most commonly used clustering algorithms is **K-Means**.

The basic idea is:

> Divide the observations into `K` groups so that observations within a cluster are relatively close to their cluster center.

K-Means uses **centroids**.

A centroid is the center of a cluster.

The scikit-learn implementation uses `n_clusters` to specify the number of clusters and provides the resulting cluster centers through `cluster_centers_`. ([scikit-learn.org](https://scikit-learn.org/dev/modules/generated/sklearn.cluster.k_means.html))

---

# 16. What Does `K` Mean?

If:

```text
K = 3
```

the algorithm attempts to create:

```text
3 clusters
```

If:

```text
K = 4
```

it attempts to create:

```text
4 clusters
```

The algorithm does not automatically know that the dataset shown above was generated using three centers.

The value of `K` is specified as part of the model configuration.

---

# 17. The Basic K-Means Algorithm

K-Means can be understood as an iterative process.

### Step 1 — Choose K

For example:

$$
K=3
$$

### Step 2 — Initialize centroids

The algorithm starts with initial cluster centers.

### Step 3 — Assign observations

Each observation is assigned to the nearest centroid.

### Step 4 — Recalculate centroids

Each centroid is moved to the center of the observations currently assigned to it.

### Step 5 — Repeat

The assignment and centroid calculations are repeated until the solution stabilizes according to the algorithm's stopping criteria.

Conceptually:

```text
Choose K
   ↓
Choose initial centers
   ↓
Assign points to nearest center
   ↓
Recalculate centers
   ↓
Repeat
```

---

# 18. Create a K-Means Model

Use:

### Code Cell

```python id="e8c8kv"
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init="auto"
)
```

The important arguments are:

### `n_clusters=3`

Requests three clusters.

### `random_state=42`

Makes the random initialization reproducible.

### `n_init="auto"`

Controls the number of K-Means initializations considered by the scikit-learn implementation. The current documentation specifies that the exact number of runs depends on the initialization strategy when `n_init="auto"` is used. ([scikit-learn.org](https://scikit-learn.org/dev/modules/generated/sklearn.cluster.k_means.html))

---

# 19. Fit the Model

Now run:

### Code Cell

```python id="h3rxr1"
kmeans.fit(X)
```

Unlike supervised learning, there is no:

```text
y
```

being supplied.

The algorithm receives:

```text
X
```

and attempts to identify groups from the observations.

---

# 20. Obtain the Cluster Labels

After fitting:

### Code Cell

```python id="9yk0bc"
labels = kmeans.labels_

print(labels[:20])
```

The result might look like:

```text
[1 2 0 1 0 2 2 0 ...]
```

These numbers are **cluster identifiers**.

They are not target classes.

---

# 21. Important: Cluster Numbers Have No Inherent Meaning

Suppose one model gives:

```text
Cluster 0
Cluster 1
Cluster 2
```

Another run might label exactly the same groups:

```text
Cluster 2
Cluster 0
Cluster 1
```

The numbers themselves are arbitrary.

Cluster `0` does not mean:

> "The best cluster."

Cluster `1` does not mean:

> "The second most important cluster."

They are simply identifiers.

---

# 22. Question 4 — Understand Cluster Labels

Suppose two clustering algorithms produce:

```text
Algorithm A:
Cluster 0 → upper-left group
Cluster 1 → lower group
Cluster 2 → upper-right group
```

and:

```text
Algorithm B:
Cluster 1 → upper-left group
Cluster 2 → lower group
Cluster 0 → upper-right group
```

Are the algorithms necessarily producing different clusters?

### LLM Hint Prompt

> Explain why cluster labels such as 0, 1, and 2 should not be interpreted as meaningful class names. Show how two algorithms can assign different numbers to the same groups.

<details>
<summary>Solution</summary>

No.

The numerical cluster labels are arbitrary identifiers.

Two algorithms can assign different label numbers to the same groups.

The actual locations and membership of the observations matter, not whether a group is called `0`, `1`, or `2`.

</details>

---

# 23. Visualize the K-Means Clusters

Create a plot.

### Code Cell

```python id="5x0edx"
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
plt.title("K-Means Clustering")

plt.show()
```

The observations are shown according to their cluster assignment.

The large `X` markers represent the cluster centroids.

---

# 24. What Is a Centroid?

A centroid is the mean position of the observations assigned to a cluster.

Suppose one cluster contains:

```text
(1, 2)
(2, 3)
(3, 4)
```

The centroid is calculated by taking the mean of each feature.

For the first coordinate:

$$
\frac{1+2+3}{3}=2
$$

For the second coordinate:

$$
\frac{2+3+4}{3}=3
$$

So:

$$
\boxed{Centroid=(2,3)}
$$

The centroid represents the center of the cluster according to the feature space.

---

# 25. Question 5 — Calculate a Centroid

Consider the following observations:

```text
A = (2, 4)
B = (4, 6)
C = (6, 8)
```

Calculate the centroid.

### LLM Hint Prompt

> Help me calculate the centroid of the points (2,4), (4,6), and (6,8). Explain why the mean is calculated separately for each feature.

<details>
<summary>Solution</summary>

For the first feature:

$$
\frac{2+4+6}{3}=4
$$

For the second feature:

$$
\frac{4+6+8}{3}=6
$$

Therefore:

$$
\boxed{Centroid=(4,6)}
$$

The mean is calculated separately for each feature because the centroid must have one coordinate for each dimension.

</details>

---

# 26. Inspect the Centroids Numerically

The centroids are stored in:

```python id="9m30hk"
kmeans.cluster_centers_
```

Run:

### Code Cell

```python id="d4k0cw"
print(kmeans.cluster_centers_)
```

If the result is:

```text
[[... ...]
 [... ...]
 [... ...]]
```

there are three rows because:

```text
n_clusters = 3
```

and two columns because:

```text
n_features = 2
```

---

# 27. Question 6 — Understand `cluster_centers_`

If:

```text
cluster_centers_.shape = (3, 2)
```

what does this mean?

### LLM Hint Prompt

> Explain why `cluster_centers_.shape = (3, 2)` when K-Means uses three clusters and the dataset has two features.

<details>
<summary>Solution</summary>

There are three centroids because the algorithm was asked to create three clusters.

Each centroid has two coordinates because each observation has two features.

Therefore:

$$
(3,2)
$$

means:

```text
3 cluster centers
2 feature coordinates per center
```

</details>

---

# 28. How Does K-Means Decide Which Cluster Gets an Observation?

Suppose there are three centroids:

```text
C1
C2
C3
```

and a new observation:

```text
P
```

K-Means conceptually calculates the distance from `P` to each centroid.

For example:

```text
distance(P, C1) = 2.1
distance(P, C2) = 5.4
distance(P, C3) = 1.3
```

The closest centroid is:

```text
C3
```

so the observation is assigned to cluster 3.

This is the central intuition behind K-Means.

---

# 29. Question 7 — Assign an Observation to a Cluster

Suppose an observation has these distances:

```text
Distance to Cluster 0 = 3.2
Distance to Cluster 1 = 1.5
Distance to Cluster 2 = 4.0
```

Which cluster should K-Means assign the observation to?

### LLM Hint Prompt

> I have distances from one observation to three K-Means centroids. Explain how K-Means uses those distances to determine the cluster assignment.

<details>
<summary>Solution</summary>

The observation should be assigned to Cluster 1 because:

$$
1.5
$$

is the smallest distance.

K-Means assigns the observation to the nearest centroid.

</details>

---

# 30. A First Limitation of K-Means

K-Means requires the number of clusters to be specified.

For example:

```python id="4ys6qj"
KMeans(n_clusters=3)
```

requires a choice of:

$$
K=3
$$

But how do we know that 3 is correct?

For the synthetic dataset, the answer can be visually obvious.

For real data, it may not be.

Choosing the number of clusters is an important part of clustering analysis.

This topic will be examined in more detail later.

---

# 31. Experiment with a Different K

Try:

```python id="jry12d"
kmeans_2 = KMeans(
    n_clusters=2,
    random_state=42,
    n_init="auto"
)

labels_2 = kmeans_2.fit_predict(X)
```

Visualize it:

### Code Cell

```python id="2c2f3g"
plt.figure(figsize=(8, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels_2
)

plt.scatter(
    kmeans_2.cluster_centers_[:, 0],
    kmeans_2.cluster_centers_[:, 1],
    marker="X",
    s=200
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means with K = 2")

plt.show()
```

Compare this with the previous result using:

$$
K=3
$$

---

# 32. Question 8 — Changing K

Answer:

1. What changes when `K=2` instead of `K=3`?
2. Are the observations assigned to the same groups?
3. Why does the choice of `K` matter?
4. Can an algorithm automatically know the "correct" number of clusters?

### LLM Hint Prompt

> Help me understand the effect of changing K in K-Means from 3 to 2. Explain why the number of clusters is a modeling decision and why there may not always be one objectively correct value of K.

<details>
<summary>Solution</summary>

Changing K changes the number of groups the algorithm is required to create.

With `K=2`, some groups that were separate under `K=3` may be merged.

The choice matters because it changes the structure discovered by the algorithm.

The data do not always contain one uniquely correct number of clusters, so the choice of K often requires additional analysis.

</details>

---

# 33. Hierarchical Clustering

K-Means is not the only clustering approach.

Another approach is **hierarchical clustering**.

In this activity, the focus is on **agglomerative hierarchical clustering**.

The basic idea is:

```text
Start with every observation as its own cluster
             ↓
Find clusters that are close
             ↓
Merge them
             ↓
Repeat
             ↓
Build a hierarchy
```

Scikit-learn's `AgglomerativeClustering` recursively merges pairs of clusters according to a linkage distance. ([scikit-learn.org](https://scikit-learn.org/1.5/modules/generated/sklearn.cluster.AgglomerativeClustering.html))

---

# 34. K-Means vs Hierarchical Clustering

The approaches start differently.

### K-Means

```text
Choose K
   ↓
Initialize centroids
   ↓
Assign observations
   ↓
Update centroids
```

### Hierarchical clustering

```text
Each observation starts alone
   ↓
Merge nearby clusters
   ↓
Continue merging
   ↓
Create hierarchy
```

A useful conceptual distinction is:

> K-Means builds a fixed number of clusters by repeatedly updating cluster centers.

> Hierarchical clustering builds a hierarchy of merged observations or clusters.

---

# 35. Create an Agglomerative Model

We can use:

### Code Cell

```python id="6wnwge"
hierarchical = AgglomerativeClustering(
    n_clusters=3
)
```

The parameter:

```text
n_clusters=3
```

requests three final clusters.

The current scikit-learn implementation supports `metric`, `linkage`, and other parameters; the default linkage is Ward's method, which uses Euclidean distance. ([scikit-learn.org](https://scikit-learn.org/1.5/modules/generated/sklearn.cluster.AgglomerativeClustering.html))

---

# 36. Fit the Hierarchical Model

### Code Cell

```python id="p9n6v3"
hierarchical_labels = hierarchical.fit_predict(X)
```

Unlike the two-step K-Means example:

```python id="w6fzw0"
kmeans.fit(X)
labels = kmeans.labels_
```

`fit_predict()` performs the fitting and returns the cluster labels.

---

# 37. Visualize the Hierarchical Clusters

### Code Cell

```python id="t8vfhi"
plt.figure(figsize=(8, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=hierarchical_labels
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Hierarchical Clustering")

plt.show()
```

---

# 38. Question 9 — Compare the Two Methods

Compare the K-Means and hierarchical-clustering plots.

Answer:

1. Do both methods produce three clusters?
2. Do the cluster boundaries look identical?
3. Why might two clustering algorithms produce slightly different groupings?
4. Which method uses centroids as part of its basic algorithm?
5. Which method builds a hierarchy?

### LLM Hint Prompt

> Compare K-Means and agglomerative hierarchical clustering conceptually. Focus on how the algorithms create clusters rather than simply comparing the final labels.

<details>
<summary>Solution</summary>

Both methods were asked to produce three clusters.

Their exact groupings may differ because they use different clustering procedures.

K-Means uses cluster centroids and repeatedly assigns observations to nearby centers.

Hierarchical clustering begins with individual observations and progressively merges clusters.

</details>

---

# 39. The Meaning of "Hierarchical"

The word hierarchical means that the clustering process creates multiple levels of grouping.

Imagine:

```text
A   B   C   D
```

Initially:

```text
{A}
{B}
{C}
{D}
```

Then perhaps:

```text
{A,B}
{C}
{D}
```

Then:

```text
{A,B}
{C,D}
```

And eventually:

```text
{A,B,C,D}
```

The process creates a hierarchy of possible groupings.

This is different from K-Means, which directly attempts to produce a chosen number of clusters.

---

# 40. Dendrograms

A hierarchy can be visualized with a **dendrogram**.

A dendrogram shows:

* which observations or groups were merged;
* the order of merging;
* the distance or linkage level at which merges occurred.

A simple conceptual dendrogram might look like:

```text
          ┌───────────────┐
          │               │
      ┌───┴───┐       ┌───┴───┐
      │       │       │       │
      A       B       C       D
```

The vertical level of a merge indicates how far apart the clusters were when they were combined.

For Activity 1, the dendrogram is introduced conceptually. Detailed dendrogram analysis can be left for a later activity.

---

# 41. Optional Dendrogram Demonstration

If SciPy is available in the Colab environment, a dendrogram can be created.

### Code Cell

```python id="wvw3n2"
!pip install -q scipy
```

Then:

### Code Cell

```python id="zctojy"
from scipy.cluster.hierarchy import linkage, dendrogram

Z = linkage(
    X,
    method="ward"
)

plt.figure(figsize=(10, 6))

dendrogram(Z)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Observations")
plt.ylabel("Distance")

plt.show()
```

The `linkage()` function creates the hierarchical clustering structure.

The `dendrogram()` function visualizes it.

For a large dataset, plotting every observation can make the dendrogram difficult to read. The visualization is therefore most useful for understanding the concept at this stage.

---

# 42. Question 10 — Read the Dendrogram Conceptually

Answer:

1. What do the leaves represent?
2. What does a merge represent?
3. What does a higher merge level generally indicate?
4. How could a dendrogram be used to decide how many clusters to keep?

### LLM Hint Prompt

> Explain how to read a hierarchical-clustering dendrogram at a conceptual level. Focus on leaves, merging, merge height, and how cutting the hierarchy can produce different numbers of clusters.

<details>
<summary>Solution</summary>

The leaves represent the original observations.

A merge represents two observations or groups of observations being combined into a larger cluster.

A higher merge generally indicates that the groups being merged are farther apart according to the chosen linkage distance.

A horizontal cut through the hierarchy can produce different numbers of clusters depending on where the cut is made.

</details>

---

# 43. Understanding Cluster Membership

A cluster is not necessarily a natural or permanent category.

It is a grouping produced according to:

* the features supplied;
* the distance measure;
* the clustering algorithm;
* algorithm parameters.

For example:

```text
Features:
Age + Income
```

might produce one grouping.

Using:

```text
Income + Spending
```

could produce a different grouping.

Therefore:

> **Clustering results depend on how the data and problem are represented.**

---

# 44. Question 11 — Change the Features

Suppose customer data contains:

```text
Age
Income
Spending
```

A clustering model using:

```text
Age + Income
```

produces three clusters.

A different model using:

```text
Income + Spending
```

produces different clusters.

Why is this possible?

### LLM Hint Prompt

> Explain why changing the features used by a clustering algorithm can change the resulting clusters. Focus on how the feature space determines distances between observations.

<details>
<summary>Solution</summary>

Clustering is based on relationships between observations in the feature space.

Changing the features changes the coordinates of the observations and therefore changes the distances between them.

As a result, the algorithm may identify different groups.

</details>

---

# 45. A Very Important Issue: Feature Scale

Distance-based clustering is affected by feature scale.

Suppose two features are:

```text
Age:
20–80

Income:
20,000–200,000
```

The numerical scale of income is much larger.

When Euclidean distance is calculated, income can have a much larger influence than age.

For example:

```text
Age difference = 5

Income difference = 50,000
```

The income difference dominates the numerical distance.

This means that feature scaling can be important for clustering methods based on distance.

---

# 46. A Small Example of Scale

Suppose:

```text
Person A:
Age = 30
Income = 50,000

Person B:
Age = 35
Income = 51,000
```

Differences:

```text
Age difference = 5
Income difference = 1,000
```

Without scaling, the income difference is numerically much larger.

A distance calculation can therefore be dominated by income.

Scaling transforms the features so that they are on comparable numerical scales.

---

# 47. Question 12 — Why Can Scale Matter?

Suppose a clustering dataset contains:

```text
Feature A: values between 0 and 10
Feature B: values between 0 and 100,000
```

Answer:

1. Which feature is likely to contribute more to Euclidean distance before scaling?
2. Why?
3. Does this mean Feature B is necessarily more important in the real world?
4. What can be done if both features should contribute more comparably?

### LLM Hint Prompt

> Explain why a feature with values between 0 and 100,000 can dominate Euclidean distance compared with a feature between 0 and 10. Then explain how feature scaling can change the geometry of the data.

<details>
<summary>Solution</summary>

1. Feature B is likely to contribute more.
2. Its numerical differences are much larger.
3. No. Numerical scale does not necessarily represent real-world importance.
4. Features can be scaled, for example using standardization.

</details>

---

# 48. First Clustering Experiment with Scaling

For the current synthetic dataset, scaling is not essential for visualizing the basic idea because both features were generated on comparable scales.

However, the concept is important for real datasets.

A common scikit-learn transformer is:

```python id="7qzn61"
from sklearn.preprocessing import StandardScaler
```

It can standardize features.

For example:

```python id="a2b6cv"
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

The scaled data have features expressed relative to their distribution rather than their original units.

A more detailed experiment with scaling will be part of the more advanced clustering activity.

---

# 49. K-Means Limitations

K-Means is useful, but it has limitations.

### 1. The number of clusters must be specified

```python
KMeans(n_clusters=3)
```

requires a choice of `3`.

### 2. It depends on distance

Distance can be affected by:

* feature scale;
* the choice of representation;
* the selected distance assumptions.

### 3. Cluster shape matters

K-Means works particularly naturally when clusters are reasonably compact and separated around centers.

Highly elongated or unusual shapes can be more challenging.

### 4. Outliers can matter

Extreme observations can influence cluster centers.

### 5. Different initializations can produce different results

K-Means uses initialization, which is why the implementation supports multiple initializations through `n_init`. ([scikit-learn.org](https://scikit-learn.org/dev/modules/generated/sklearn.cluster.k_means.html))

---

# 50. Hierarchical Clustering Limitations

Hierarchical clustering also has limitations.

For example:

* the choice of linkage affects the result;
* distance still matters;
* the method can become computationally expensive for large datasets;
* interpreting the hierarchy can be difficult;
* different choices can produce different structures.

It does, however, provide a useful advantage:

> The hierarchy shows relationships at several levels rather than producing only one fixed partition.

---

# 51. Question 13 — Choosing Between K-Means and Hierarchical Clustering

Consider:

### Problem A

A large dataset contains millions of observations and several numerical features.

### Problem B

A small dataset contains 30 observations and the relationships between the observations are important to inspect.

Which method might be more convenient as a starting point?

Explain why.

### LLM Hint Prompt

> Compare K-Means and hierarchical clustering for a very large dataset versus a very small dataset where the relationships among observations need to be visualized. Discuss computational practicality and interpretability without claiming that one algorithm is universally better.

<details>
<summary>Solution</summary>

K-Means may be a more practical starting point for a very large dataset because it is designed to perform clustering efficiently.

Hierarchical clustering can be attractive for a small dataset when understanding the relationships between observations and viewing the hierarchy is important.

Neither method is universally best. The appropriate choice depends on the data and objective.

</details>

---

# 52. Supervised vs Unsupervised Learning

The main difference can now be summarized.

## Supervised

A target is available.

```text
X + y
 ↓
learn
 ↓
prediction
```

Examples:

```text
Classification
Regression
```

## Unsupervised

The learning process does not use a target.

```text
X
 ↓
discover structure
```

Examples:

```text
Clustering
Anomaly Detection
Association Rule Learning
Dimensionality Reduction
```

---

# 53. Clustering vs Classification

These two approaches are especially easy to confuse.

## Classification

Suppose:

```text
Observation A → Class 0
Observation B → Class 1
Observation C → Class 0
```

The classes are known during training.

The model learns to predict them.

## Clustering

The algorithm receives:

```text
Observation A
Observation B
Observation C
...
```

without target labels.

It attempts to discover groups.

Therefore:

> Classification predicts known categories.

> Clustering discovers groups without using known target categories during the clustering process.

---

# 54. A Note About Labels

A dataset can contain labels even when an unsupervised algorithm does not use them.

For example, the Iris dataset contains known species.

A clustering experiment can deliberately ignore:

```text
species
```

and cluster flowers using:

```text
sepal length
sepal width
petal length
petal width
```

The known species can then be used afterward for analysis or comparison.

The labels were not used to train the clustering algorithm.

This distinction matters because:

> **Unsupervised learning means the target labels do not guide the learning process; it does not necessarily mean that no labels exist anywhere in the dataset.**

---

# 55. Question 14 — Is This Supervised or Unsupervised?

Consider:

```text
Iris measurements
        ↓
K-Means clustering
        ↓
Cluster labels
```

The original Iris species column exists but is not given to K-Means.

Is the clustering process supervised or unsupervised?

### LLM Hint Prompt

> Explain whether a clustering process remains unsupervised when the original dataset contains a known label column but that column is deliberately excluded from the clustering input.

<details>
<summary>Solution</summary>

It remains unsupervised because the species labels are not used to guide the clustering algorithm.

The algorithm receives the feature values and discovers groups without using the known species as a target.

</details>

---

# 56. Cluster Labels Are Not the Same as Target Labels

This distinction is important.

Suppose K-Means returns:

```text
[0, 0, 1, 2, 2, 1, ...]
```

These numbers mean:

> The algorithm assigned these observations to internally discovered clusters.

They do not automatically mean:

```text
0 = setosa
1 = versicolor
2 = virginica
```

There is no guarantee that cluster `0` corresponds to any particular real-world class.

A clustering result must be interpreted based on the feature values and domain context.

---

# 57. Question 15 — Interpret Cluster Numbers

Suppose K-Means is applied to a dataset with known labels, but the labels are not used during clustering.

The result is:

```text
Cluster 0
Cluster 1
Cluster 2
```

Can Cluster 0 automatically be called "Class A"?

Explain.

### LLM Hint Prompt

> Explain why unsupervised cluster labels cannot automatically be treated as known class labels. Discuss the difference between an algorithm-created identifier and a meaningful domain category.

<details>
<summary>Solution</summary>

No.

Cluster numbers are identifiers assigned by the algorithm.

They do not automatically correspond to meaningful domain classes.

A relationship between clusters and known labels would need to be investigated separately.

</details>

---

# 58. A First Look at Cluster Interpretation

Suppose clustering is applied to customer data.

After clustering, the following summary is calculated:

| Cluster | Average Age | Average Income | Average Spending |
| ------- | ----------: | -------------: | ---------------: |
| 0       |          24 |         30,000 |               15 |
| 1       |          42 |         55,000 |               45 |
| 2       |          58 |         80,000 |               20 |

The algorithm only produces cluster identifiers.

The analyst can then interpret the groups:

```text
Cluster 0:
younger, lower income, lower spending

Cluster 1:
middle-aged, medium income, higher spending

Cluster 2:
older, higher income, lower spending
```

The labels are created after examining the characteristics.

This illustrates an important point:

> **Clustering discovers groups; interpretation gives those groups meaning.**

---

# 59. Question 16 — Interpret Clusters

Suppose the following cluster summaries are observed:

| Cluster | Average Age | Average Income | Average Spending |
| ------- | ----------: | -------------: | ---------------: |
| 0       |          22 |         25,000 |               20 |
| 1       |          45 |         60,000 |               70 |
| 2       |          62 |         90,000 |               25 |

Describe each cluster in a meaningful way.

### LLM Hint Prompt

> Help me interpret three customer clusters from their average age, income, and spending. Do not invent information beyond the table. Describe each group using only the evidence provided.

<details>
<summary>Solution</summary>

Cluster 0:

> Younger customers with relatively low income and relatively low spending.

Cluster 1:

> Middle-aged customers with medium income and high spending.

Cluster 2:

> Older customers with high income and relatively low spending.

These descriptions are interpretations of the observed feature averages, not names produced by the clustering algorithm.

</details>

---

# 60. Does Clustering Find "Truth"?

Clustering should not automatically be interpreted as discovering objectively correct categories.

A clustering algorithm finds structure according to:

```text
chosen features
+
chosen algorithm
+
chosen parameters
+
chosen distance/similarity assumptions
```

Different choices can lead to different results.

For example:

```text
K-Means, K=3
```

may produce one grouping.

Changing to:

```text
K-Means, K=4
```

can produce another.

Changing to:

```text
Hierarchical clustering
```

can produce another.

Therefore:

> **A cluster is a useful representation of structure in the data, not necessarily a naturally existing category.**

---

# 61. Question 17 — Is There Always One Correct Clustering?

Answer:

> If K-Means with `K=3` and hierarchical clustering with three clusters produce slightly different groups, which one is automatically correct?

### LLM Hint Prompt

> Explain why different unsupervised algorithms can produce different but defensible clusterings. Focus on assumptions, distance, features, and the purpose of the analysis.

<details>
<summary>Solution</summary>

Neither result is automatically "correct."

Different algorithms use different procedures and assumptions.

The appropriate result depends on:

* the data;
* the chosen features;
* the algorithm;
* the distance/linkage approach;
* and the purpose of the analysis.

Additional evaluation and domain knowledge are often needed.

</details>

---

# 62. Mini Experiment — Change the Cluster Spread

The original dataset used:

```python id="qf9y8r"
cluster_std=1.2
```

Create another dataset with more overlap:

### Code Cell

```python id="o50a9j"
X_overlap, _ = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=2.5,
    random_state=42
)
```

Visualize:

### Code Cell

```python id="q8vxxd"
plt.figure(figsize=(8, 6))

plt.scatter(
    X_overlap[:, 0],
    X_overlap[:, 1]
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Clusters with More Overlap")

plt.show()
```

Now fit K-Means:

### Code Cell

```python id="5v20vx"
kmeans_overlap = KMeans(
    n_clusters=3,
    random_state=42,
    n_init="auto"
)

labels_overlap = kmeans_overlap.fit_predict(
    X_overlap
)
```

Visualize:

### Code Cell

```python id="a0b9fj"
plt.figure(figsize=(8, 6))

plt.scatter(
    X_overlap[:, 0],
    X_overlap[:, 1],
    c=labels_overlap
)

plt.scatter(
    kmeans_overlap.cluster_centers_[:, 0],
    kmeans_overlap.cluster_centers_[:, 1],
    marker="X",
    s=200
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means with Overlapping Groups")

plt.show()
```

---

# 63. Question 18 — Why Does Overlap Matter?

Compare the original dataset with the overlapping dataset.

Answer:

1. Are the groups easier or harder to distinguish?
2. Why?
3. What happens when observations lie between apparent groups?
4. Does K-Means still produce an answer?

### LLM Hint Prompt

> Explain what happens to clustering when groups overlap heavily. Focus on why some observations become ambiguous and why an algorithm can still assign them to clusters.

<details>
<summary>Solution</summary>

The groups become harder to distinguish because observations from different groups are closer together.

Some observations may lie in regions where membership is ambiguous.

K-Means still assigns every observation to one of the requested clusters, even when the boundaries are not obvious.

This is an important limitation: receiving a cluster label does not mean that the observation has an unquestionably natural group membership.

</details>

---

# 64. Important Insight: Clustering Always Produces a Structure

A clustering algorithm can produce clusters even when the data do not contain clearly separated natural groups.

For example, specifying:

```python id="6b6vne"
KMeans(n_clusters=5)
```

will ask the algorithm to produce five clusters.

That does not prove that the data naturally contain five meaningful groups.

Therefore:

> **Clustering results need to be evaluated and interpreted rather than accepted automatically.**

---

# 65. What Happens with Random Data?

Consider data with no obvious groups.

For example:

### Code Cell

```python id="7e3k9q"
rng = np.random.default_rng(42)

X_random = rng.uniform(
    0,
    10,
    size=(300, 2)
)
```

Plot:

### Code Cell

```python id="5j78w2"
plt.figure(figsize=(8, 6))

plt.scatter(
    X_random[:, 0],
    X_random[:, 1]
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Random Data")

plt.show()
```

Now apply K-Means:

### Code Cell

```python id="3h7v7m"
random_kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init="auto"
)

random_labels = random_kmeans.fit_predict(
    X_random
)
```

---

# 66. Question 19 — Does K-Means Prove That Groups Exist?

After clustering the random data, answer:

> If K-Means produces three clusters, does this prove that the random data naturally contain three meaningful groups?

### LLM Hint Prompt

> Explain why K-Means can partition random data into clusters even when the data were not generated from meaningful groups. Focus on the difference between producing a partition and discovering a meaningful structure.

<details>
<summary>Solution</summary>

No.

K-Means is instructed to produce three clusters, so it will partition the observations into three groups even if the data do not contain naturally separated groups.

A clustering output is therefore not automatically evidence that meaningful groups exist.

The result must be evaluated and interpreted.

</details>

---

# 67. What Has Been Learned?

The activity has introduced several levels of unsupervised learning.

### Level 1 — Data

```text
X
```

contains observations without a target guiding the learning process.

### Level 2 — Similarity

Observations can be compared using distance.

### Level 3 — Clustering

Similar observations can be grouped.

### Level 4 — K-Means

K-Means groups observations around centroids.

### Level 5 — Hierarchical clustering

Observations can also be grouped progressively into a hierarchy.

---

# 68. K-Means vs Hierarchical Clustering

| Concept                                    | K-Means                                          | Hierarchical Clustering                  |
| ------------------------------------------ | ------------------------------------------------ | ---------------------------------------- |
| Main idea                                  | Assign observations to clusters around centroids | Repeatedly merge clusters                |
| Requires a chosen number of final clusters | Yes                                              | Usually yes for the final partition      |
| Uses centroids                             | Yes                                              | No centroid-based assignment is required |
| Builds a hierarchy                         | No                                               | Yes                                      |
| Common visualization                       | Cluster scatter plot                             | Dendrogram                               |
| Sensitive to feature scale                 | Yes                                              | Yes                                      |
| Result depends on algorithm settings       | Yes                                              | Yes                                      |

The two methods can produce similar results, but they do not work in the same way.

---

# 69. Regression/Classification vs Clustering

The main difference between supervised and unsupervised learning can now be summarized.

| Supervised Learning  | Unsupervised Learning                   |
| -------------------- | --------------------------------------- |
| Uses a known target  | Does not use a target to guide learning |
| Learns to predict    | Discovers structure                     |
| Regression           | Clustering                              |
| Classification       | Anomaly detection                       |
| Numerical prediction | Association rules                       |
| Class prediction     | Dimensionality reduction                |

For example:

```text
Regression:

X → known numerical y
      ↓
learn prediction


Clustering:

X
↓
discover groups
```

---

# 70. Where Clustering Is Used

Clustering can be applied in many situations.

Examples include:

### Customer segmentation

```text
Customers
   ↓
Clustering
   ↓
Customer groups
```

### Document organization

```text
Documents
   ↓
Similarity
   ↓
Document clusters
```

### Image analysis

```text
Images / image features
   ↓
Clustering
   ↓
groups of similar images
```

### Biological data

```text
Measurements
   ↓
Clustering
   ↓
groups of similar observations
```

The interpretation depends on the application.

---

# 71. What Clustering Does Not Tell Us Automatically

Clustering does not automatically tell us:

* why the groups exist;
* whether the groups are scientifically meaningful;
* whether the number of clusters is correct;
* whether one clustering algorithm is objectively superior;
* whether a cluster corresponds to a real-world category.

These questions require additional analysis.

---

# 72. Final Reflection

Answer the following questions without immediately looking at the solutions.

### Question 20

What is the difference between supervised and unsupervised learning?

### LLM Hint Prompt

> Help me explain supervised versus unsupervised learning using one regression example and one clustering example. Ask me questions that help me identify where the target is used.

<details>
<summary>Solution</summary>

Supervised learning uses a known target during training and learns to predict it.

Unsupervised learning does not use a target to guide the learning process and instead attempts to discover structure in the feature data.

</details>

---

### Question 21

What is clustering?

### LLM Hint Prompt

> Help me define clustering in one or two precise sentences. Focus on grouping observations according to similarity rather than predicting known classes.

<details>
<summary>Solution</summary>

Clustering is an unsupervised-learning approach that groups observations according to their similarity or distance in the selected feature space.

</details>

---

### Question 22

What does `K` mean in K-Means?

### LLM Hint Prompt

> Explain what the K in K-Means represents and why changing K changes the clustering result.

<details>
<summary>Solution</summary>

`K` is the number of clusters the algorithm is asked to create.

Changing K changes the number of groups and can therefore change the cluster assignments.

</details>

---

### Question 23

What is a centroid?

### LLM Hint Prompt

> Explain what a centroid represents in K-Means and calculate a simple centroid for three two-dimensional points if useful.

<details>
<summary>Solution</summary>

A centroid is the mean position of the observations assigned to a cluster.

It has one coordinate for each feature.

</details>

---

### Question 24

Why is distance important in K-Means?

### LLM Hint Prompt

> Explain why K-Means needs a notion of distance and how the nearest centroid determines cluster assignment.

<details>
<summary>Solution</summary>

K-Means assigns observations to the nearest centroid. Therefore, distance determines which cluster an observation is assigned to.

</details>

---

### Question 25

Why can feature scaling matter?

### LLM Hint Prompt

> Explain what happens to Euclidean distance when one feature ranges from 0 to 10 and another ranges from 0 to 100,000. Then explain why scaling may be necessary.

<details>
<summary>Solution</summary>

The feature with the much larger numerical range can dominate the distance calculation.

Scaling can place features on comparable scales so that the clustering result is not determined mainly by numerical units.

</details>

---

### Question 26

What is hierarchical clustering?

### LLM Hint Prompt

> Explain agglomerative hierarchical clustering as a sequence of merges. Compare it with K-Means without going into advanced mathematics.

<details>
<summary>Solution</summary>

Agglomerative hierarchical clustering starts with each observation as its own cluster and progressively merges clusters according to their similarity or linkage distance.

Unlike K-Means, it builds a hierarchy of cluster relationships.

</details>

---

### Question 27

Why is a cluster label such as `0` or `1` not a meaningful class name?

### LLM Hint Prompt

> Explain why cluster identifiers are arbitrary and why they should not automatically be interpreted as known domain categories.

<details>
<summary>Solution</summary>

Cluster numbers are identifiers generated by the algorithm.

They do not have inherent meaning and do not automatically correspond to real-world classes.

</details>

---

# 73. Final Knowledge Check

Answer these questions without looking at the solutions.

1. What is unsupervised learning?
2. What is clustering?
3. What is a feature?
4. What is a target?
5. Why does clustering normally not use a target?
6. What does Euclidean distance measure?
7. What does K represent in K-Means?
8. What is a centroid?
9. How does K-Means assign an observation to a cluster?
10. Why can feature scaling affect K-Means?
11. Why are cluster labels arbitrary?
12. What is the main idea behind agglomerative hierarchical clustering?
13. What is a dendrogram?
14. Why can different algorithms produce different clusters?
15. Does creating three clusters prove that three natural groups exist?
16. What is one important limitation of K-Means?
17. What is one important difference between K-Means and hierarchical clustering?

---

# Final Answers

<details>
<summary>1. What is unsupervised learning?</summary>

Unsupervised learning is machine learning in which the learning process does not use a target variable to guide the discovery of patterns or structure in the data.

</details>

<details>
<summary>2. What is clustering?</summary>

Clustering groups observations according to similarity or distance in the selected feature space.

</details>

<details>
<summary>3. What is a feature?</summary>

A feature is an input variable used to describe an observation.

</details>

<details>
<summary>4. What is a target?</summary>

A target is the value that a supervised-learning model is trained to predict.

</details>

<details>
<summary>5. Why does clustering normally not use a target?</summary>

Because clustering is an unsupervised task. The purpose is to discover groups from the feature data rather than learn to reproduce known target values.

</details>

<details>
<summary>6. What does Euclidean distance measure?</summary>

It measures the straight-line distance between two observations in the selected feature space.

</details>

<details>
<summary>7. What does K represent?</summary>

K represents the number of clusters that K-Means is asked to create.

</details>

<details>
<summary>8. What is a centroid?</summary>

A centroid is the mean position of the observations assigned to a cluster.

</details>

<details>
<summary>9. How does K-Means assign an observation?</summary>

It assigns the observation to the nearest cluster centroid according to the distance measure being used.

</details>

<details>
<summary>10. Why can feature scaling affect K-Means?</summary>

K-Means depends on distances, and a feature with a much larger numerical scale can dominate those distances.

</details>

<details>
<summary>11. Why are cluster labels arbitrary?</summary>

They are identifiers generated by the algorithm. Cluster `0` has no inherent meaning compared with cluster `1`.

</details>

<details>
<summary>12. What is agglomerative hierarchical clustering?</summary>

It starts with individual observations and progressively merges nearby observations or clusters to create a hierarchy.

</details>

<details>
<summary>13. What is a dendrogram?</summary>

A dendrogram is a visual representation of the hierarchy created by hierarchical clustering.

</details>

<details>
<summary>14. Why can different algorithms produce different clusters?</summary>

Different algorithms use different procedures, assumptions, distance/linkage methods, and parameters.

</details>

<details>
<summary>15. Does creating three clusters prove three natural groups exist?</summary>

No. An algorithm can be instructed to create three clusters even when the data do not contain three clearly meaningful groups.

</details>

<details>
<summary>16. Give one K-Means limitation.</summary>

K-Means requires the number of clusters to be specified in advance. Other limitations include sensitivity to scale, outliers, initialization, and cluster shape.

</details>

<details>
<summary>17. What is one key difference between K-Means and hierarchical clustering?</summary>

K-Means works around cluster centroids and repeatedly updates cluster assignments, while hierarchical clustering progressively merges observations or clusters to create a hierarchy.

</details>

---

# 74. Activity Summary

The main workflow introduced in this activity is:

```text
Data without target
       ↓
Explore the feature space
       ↓
Think about similarity
       ↓
Measure distance
       ↓
K-Means
       ↓
Cluster assignments
       ↓
Centroids
       ↓
Visualize and interpret
       ↓
Hierarchical clustering
       ↓
Compare clustering approaches
```

The main conceptual distinction is:

```text
Supervised learning

X + known target
       ↓
learn to predict


Unsupervised learning

X
↓
discover structure
```

The main clustering distinction is:

```text
K-Means

Choose K
   ↓
Use centroids
   ↓
Assign observations
   ↓
Update centroids


Hierarchical clustering

Start with individual observations
   ↓
Merge groups
   ↓
Build hierarchy
```

The important principle is:

> **A clustering algorithm can discover useful structure in data, but the resulting groups are not automatically meaningful or "correct." Their usefulness depends on the data, the features, the algorithm, the parameters, and the purpose of the analysis.**

The next activity can build on this foundation by examining **how to choose K, how scaling affects clustering, how to evaluate clusters, how hierarchical clustering can be explored in greater detail, and how unsupervised learning can also be used for anomaly detection and association rule learning.**
