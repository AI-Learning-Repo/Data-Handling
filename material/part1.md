# Theory — Introduction to Unsupervised Learning and Clustering

This section introduces the concepts required for **Activity 1 — Introduction to Unsupervised Learning with Clustering**.

The main topics are:

* supervised and unsupervised learning;
* clustering;
* similarity and distance;
* Euclidean distance;
* K-Means clustering;
* centroids;
* cluster assignments;
* choosing the number of clusters;
* hierarchical clustering;
* dendrograms;
* interpretation and limitations of clustering.

More advanced topics such as cluster evaluation, feature scaling, anomaly detection, association rules, and dimensionality reduction are covered separately.

---

# 1. Supervised and Unsupervised Learning

Machine learning can be divided into several broad categories.

Two important categories are:

* **supervised learning**
* **unsupervised learning**

The main difference is whether a target variable is used to guide the learning process.

---

# 2. Supervised Learning

In supervised learning, the data contain:

```text
Features
   +
Known target
```

The model learns a relationship between the features and the target.

For example, consider a regression problem:

```text
Weight
Horsepower
Displacement
      ↓
Regression model
      ↓
MPG
```

The model is trained using cars for which the MPG is already known.

The target is:

$$
y=MPG
$$

The model learns how the features are related to this target.

Examples of supervised learning include:

### Regression

Predicting a numerical value.

Examples:

* house price;
* taxi fare;
* MPG;
* temperature.

### Classification

Predicting a category.

Examples:

* spam/not spam;
* survived/did not survive;
* species of an Iris flower.

---

# 3. Unsupervised Learning

In unsupervised learning, there is no target variable guiding the learning process.

The model receives feature data:

```text
X
```

and attempts to discover structure.

Conceptually:

```text
Supervised:

X + known y
     ↓
learn relationship
     ↓
predict y


Unsupervised:

X
↓
discover structure
```

Possible structures include:

* groups of similar observations;
* unusual observations;
* frequently occurring item combinations;
* lower-dimensional representations;
* estimates of the underlying data distribution.

---

# 4. Examples of Unsupervised Learning

Several types of problems fall under unsupervised learning.

## Clustering

Question:

> Which observations form groups?

Example:

```text
Customers
   ↓
Clustering
   ↓
Customer segments
```

## Anomaly Detection

Question:

> Which observations are unusual?

Example:

```text
Credit-card transactions
        ↓
Anomaly detection
        ↓
Unusual transactions
```

## Association Rule Learning

Question:

> Which items frequently occur together?

Example:

```text
Bread + Butter
```

## Dimensionality Reduction

Question:

> Can high-dimensional data be represented with fewer dimensions?

Example:

```text
100 features
    ↓
PCA
    ↓
2 dimensions
```

Activity 1 focuses only on **clustering**.

---

# 5. What Is Clustering?

Clustering is an unsupervised-learning technique used to group similar observations.

The basic idea is:

> Observations that are similar should be placed in the same group, while observations that are less similar should be placed in different groups.

Suppose a dataset contains customer characteristics:

```text
Age
Income
Spending
```

A clustering algorithm might discover groups such as:

```text
Group 1:
younger, lower-income customers

Group 2:
middle-aged, medium-income customers

Group 3:
older, higher-income customers
```

The important point is that these groups were **not provided as target labels**.

The algorithm discovers them from the feature values.

---

# 6. Clustering vs Classification

Clustering and classification can appear similar because both produce groups.

The difference is how the groups are obtained.

## Classification

The classes are known.

For example:

```text
Training data:

Flower A → setosa
Flower B → versicolor
Flower C → virginica
```

The model learns to predict these known classes.

## Clustering

The groups are not provided.

For example:

```text
Flower A
Flower B
Flower C
...
```

The algorithm attempts to discover groups based on their measurements.

Therefore:

> **Classification predicts known categories; clustering discovers groups from the data.**

---

# 7. A Dataset Without a Target

Suppose a customer dataset contains:

| Age | Income | Spending |
| --: | -----: | -------: |
|  22 | 25,000 |       30 |
|  24 | 28,000 |       35 |
|  45 | 65,000 |       80 |
|  47 | 62,000 |       75 |
|  60 | 90,000 |       25 |

There is no column such as:

```text
Customer_Group
```

A clustering algorithm can still examine:

```text
Age
Income
Spending
```

and attempt to discover groups.

This is the fundamental idea behind unsupervised clustering.

---

# 8. Similarity

To create groups, a clustering algorithm needs some definition of **similarity**.

Two observations might be considered similar if their feature values are close.

For example:

```text
A = (2, 3)
B = (2.5, 3.2)
```

These observations are close together.

Another observation:

```text
C = (9, 10)
```

is much farther away.

We might therefore consider:

```text
A and B → similar
A and C → less similar
```

The exact definition of similarity depends on the algorithm and the distance or similarity measure being used.

---

# 9. Distance

One common way to measure similarity is to calculate **distance**.

The intuition is straightforward:

```text
small distance
     ↓
more similar

large distance
     ↓
less similar
```

One of the most familiar distance measures is **Euclidean distance**.

---

# 10. Euclidean Distance in Two Dimensions

Suppose we have two observations:

$$
A=(x_1,y_1)
$$

and:

$$
B=(x_2,y_2)
$$

The Euclidean distance is:

$$
d(A,B)=
\sqrt{
(x_2-x_1)^2+
(y_2-y_1)^2
}
$$

Consider:

$$
A=(1,2)
$$

and:

$$
B=(4,6)
$$

Then:

$$
d=
\sqrt{
(4-1)^2+
(6-2)^2
}
$$

$$
=
\sqrt{
3^2+4^2
}
$$

$$
=
\sqrt{9+16}
$$

$$
=
\sqrt{25}
$$

Therefore:

$$
\boxed{d=5}
$$

---

# 11. Why Does Distance Matter?

Suppose three observations are:

```text
A = (1, 2)
B = (2, 3)
C = (9, 9)
```

The distance from A to B is small.

The distance from A to C is much larger.

A clustering algorithm can use this information to identify groups.

For example:

```text
A ─── B


C
```

suggests that A and B might belong to one group while C belongs elsewhere.

This is only an intuition. Different algorithms use distance differently.

---

# 12. Feature Space

When observations have two numerical features, they can be represented on a two-dimensional graph.

For example:

```text
x-axis → Feature 1
y-axis → Feature 2
```

Each observation becomes a point.

With three features, an observation exists in three-dimensional feature space.

With 100 features, it exists in a 100-dimensional feature space.

Although we cannot easily visualize 100 dimensions directly, the same mathematical concept applies.

---

# 13. Why Two-Dimensional Data Are Useful

A dataset with two numerical features can be plotted directly.

For example:

```text
Feature 2
   ^
   |      ● ●
   |     ● ● ●
   |
   |                   ● ●
   |                  ● ● ●
   |
   +--------------------------> Feature 1
```

The visible groups can provide an initial indication that clustering may be useful.

This is why two-dimensional synthetic data are convenient for introducing clustering.

---

# 14. K-Means Clustering

**K-Means** is one of the most widely used clustering algorithms.

The basic objective is to divide observations into:

$$
K
$$

clusters.

The algorithm tries to create groups in which observations are relatively close to their assigned cluster center.

---

# 15. What Does `K` Mean?

In:

$$
K\text{-Means}
$$

the `K` is the number of clusters to create.

For example:

$$
K=2
$$

means:

```text
Create 2 clusters
```

while:

$$
K=4
$$

means:

```text
Create 4 clusters
```

The algorithm does not automatically know the appropriate value of K.

That is an important issue in practical clustering.

---

# 16. Cluster Centroids

K-Means uses **centroids**.

A centroid is the center of a cluster.

Suppose a cluster contains:

$$
(1,2)
$$

$$
(2,3)
$$

$$
(3,4)
$$

The centroid is found by taking the mean of each feature.

For the first feature:

$$
\frac{1+2+3}{3}=2
$$

For the second:

$$
\frac{2+3+4}{3}=3
$$

Therefore:

$$
\boxed{Centroid=(2,3)}
$$

The centroid represents the center of the observations currently assigned to that cluster.

---

# 17. The K-Means Algorithm

K-Means can be understood as an iterative process.

## Step 1 — Choose K

Suppose:

$$
K=3
$$

## Step 2 — Initialize the centroids

The algorithm chooses initial cluster centers.

## Step 3 — Assign observations

Each observation is assigned to the nearest centroid.

## Step 4 — Recalculate the centroids

The centroid of each cluster is recalculated from its current observations.

## Step 5 — Repeat

The assignment and centroid updates are repeated until the solution stabilizes according to the algorithm's stopping criteria.

Conceptually:

```text
Choose K
   ↓
Initialize centroids
   ↓
Assign observations
   ↓
Update centroids
   ↓
Assign again
   ↓
Update again
   ↓
...
```

---

# 18. Why Is K-Means Called "K-Means"?

The name comes from two ideas.

### K

The number of groups:

$$
K
$$

### Means

Each cluster center is calculated as the mean of the observations assigned to the cluster.

Therefore:

> K-Means creates K groups based on distances to cluster means/centroids.

---

# 19. K-Means in Scikit-Learn

Scikit-learn provides:

```python
from sklearn.cluster import KMeans
```

A model can be created as:

```python
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init="auto"
)
```

The main parameter is:

```python
n_clusters=3
```

which specifies the desired number of clusters.

The `random_state` makes the result reproducible.

The `n_init` parameter controls initialization runs. Current scikit-learn documentation recommends/defines `n_init="auto"` as an automatic choice based on the initialization strategy. ([scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html?utm_source=chatgpt.com))

---

# 20. Fitting K-Means

Unlike supervised learning:

```python
model.fit(X_train, y_train)
```

K-Means does not receive a target.

Instead:

```python
kmeans.fit(X)
```

is sufficient.

The model receives the feature data and discovers the cluster structure.

---

# 21. Cluster Labels

After fitting K-Means, cluster assignments can be obtained using:

```python
kmeans.labels_
```

For example:

```text
[0, 0, 2, 1, 2, 1, 0, ...]
```

These values represent cluster identifiers.

They are **not target classes**.

---

# 22. Cluster Labels Are Arbitrary

Suppose one run produces:

```text
Cluster 0 → left group
Cluster 1 → right group
Cluster 2 → middle group
```

Another run might label exactly the same groups:

```text
Cluster 2 → left group
Cluster 0 → right group
Cluster 1 → middle group
```

The cluster numbers themselves do not have inherent meaning.

Therefore:

> **Cluster 0 is not automatically "better" or "more important" than Cluster 1.**

The meaningful information is the membership and characteristics of the groups.

---

# 23. Cluster Centers in Scikit-Learn

K-Means stores its centroids in:

```python
kmeans.cluster_centers_
```

If there are:

```text
3 clusters
```

and:

```text
2 features
```

then:

```python
kmeans.cluster_centers_.shape
```

will be:

```text
(3, 2)
```

because there are:

* three centroids;
* two coordinates per centroid.

---

# 24. Why Do Centroids Matter?

Suppose the centroids are:

```text
C1 = (2, 3)
C2 = (8, 9)
C3 = (5, 2)
```

For a new observation:

```text
P = (4, 3)
```

the algorithm can calculate its distance to each centroid.

For example:

```text
Distance to C1 = 2
Distance to C2 = 7
Distance to C3 = 3
```

The smallest distance is:

```text
Distance to C1 = 2
```

Therefore the observation is assigned to the cluster represented by `C1`.

This is the central intuition behind K-Means.

---

# 25. K-Means Objective

K-Means attempts to make observations close to their assigned cluster centers.

A simplified mathematical description is:

$$
\sum_{i=1}^{n}
\left\|
x_i-\mu_{c_i}
\right\|^2
$$

where:

* \(x_i\) is observation \(i\);
* \(\mu_{c_i}\) is the centroid of the cluster assigned to observation \(i\);
* \(c_i\) is the assigned cluster.

The algorithm attempts to minimize this within-cluster squared distance.

This quantity is closely related to what scikit-learn reports as **inertia**.

For the introductory clustering activity, the main idea to remember is:

> K-Means tries to create compact groups around their centroids.

---

# 26. Inertia

After fitting K-Means, scikit-learn provides:

```python
kmeans.inertia_
```

Inertia measures the total within-cluster squared distance from observations to their assigned centroids.

A lower value means the observations are, overall, closer to their cluster centers.

However, lower inertia by itself does not tell us the best number of clusters.

---

# 27. Why Does More K Usually Reduce Inertia?

Suppose we use:

$$
K=1
$$

Almost all observations must be represented by one centroid.

Now use:

$$
K=3
$$

There are more centroids.

The observations can therefore be represented more closely.

Now use:

$$
K=10
$$

There are even more centroids.

The algorithm has even greater flexibility.

Therefore:

> Inertia generally decreases when K increases.

This creates an important problem.

If the objective were simply to minimize inertia, we could keep increasing K.

That does not necessarily produce meaningful clusters.

---

# 28. Choosing K

The appropriate number of clusters is often not known in advance.

Possible sources of information include:

* domain knowledge;
* visual inspection;
* elbow method;
* silhouette score;
* comparison of different clusterings.

Activity 1 introduces the idea that K is a modeling choice.

Detailed model-selection techniques are developed further later.

---

# 29. The Elbow Method

The elbow method examines how inertia changes as K increases.

Suppose:

|  K | Inertia |
| -: | ------: |
|  1 |    2000 |
|  2 |    1100 |
|  3 |     600 |
|  4 |     350 |
|  5 |     300 |
|  6 |     270 |

The major improvements occur early.

After approximately:

$$
K=4
$$

the improvements become much smaller.

The curve may therefore look like an elbow around 4.

The idea is:

> Choose a K where additional clusters begin to provide relatively little additional improvement.

The elbow method is a heuristic.

There may be no perfectly clear elbow.

---

# 30. Silhouette Score

The silhouette score provides another way to evaluate cluster separation.

It considers two ideas:

1. how well an observation fits its own cluster;
2. how separated it is from neighboring clusters.

The score ranges from:

$$
-1
$$

to:

$$
1
$$

A value closer to 1 generally indicates that observations are well matched to their own cluster and separated from nearby clusters. A value near zero indicates observations near cluster boundaries. Negative values can indicate potentially poor assignments. ([scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html?utm_source=chatgpt.com))

Detailed silhouette analysis belongs to the more advanced clustering activity.

---

# 31. Feature Scale and Distance

Because K-Means uses distances, the numerical scale of features matters.

Suppose two features are:

```text
Age:
20–80

Income:
20,000–200,000
```

Income has much larger numerical values.

Without scaling, differences in income can dominate the distance calculation.

This can cause clustering to be driven primarily by income.

---

# 32. Why Scaling Changes the Geometry

Imagine two observations:

```text
A:
Age = 30
Income = 40,000

B:
Age = 35
Income = 41,000
```

The raw differences are:

```text
Age difference = 5
Income difference = 1,000
```

When Euclidean distance is calculated, the income difference is numerically much larger.

Scaling can transform the features to comparable scales.

A common method is standardization:

$$
z=
\frac{x-\mu}{\sigma}
$$

This topic becomes important when clustering real-world datasets with differently scaled variables.

---

# 33. Hierarchical Clustering

K-Means is only one clustering algorithm.

Another approach is **hierarchical clustering**.

Hierarchical clustering creates a hierarchy of relationships between observations.

The most common introductory version is:

> **Agglomerative hierarchical clustering**

It starts with individual observations and progressively combines them.

---

# 34. Agglomerative Clustering

The process can be visualized as:

```text
Start:

A   B   C   D
↓   ↓   ↓   ↓

Each observation is its own cluster


Then:

AB   C   D


Then:

AB   CD


Finally:

ABCD
```

The algorithm builds the hierarchy from the bottom upward.

---

# 35. How Is It Different from K-Means?

### K-Means

Starts by selecting:

$$
K
$$

and works toward creating that number of clusters.

### Hierarchical clustering

Starts with individual observations and progressively merges groups.

The conceptual difference is:

```text
K-Means
Choose K
   ↓
Centroids
   ↓
Assign observations
   ↓
Iterate


Hierarchical
Individual observations
   ↓
Merge nearby groups
   ↓
Build hierarchy
```

---

# 36. Linkage

When hierarchical clustering decides which groups to merge, it needs a rule for measuring the distance between groups.

This is called **linkage**.

Common linkage methods include:

* Ward;
* complete;
* average;
* single.

The methods use different definitions of the distance between clusters.

For Activity 1, the most important point is:

> Different linkage methods can produce different hierarchical clusterings.

---

# 37. Agglomerative Clustering in Scikit-Learn

Scikit-learn provides:

```python
from sklearn.cluster import AgglomerativeClustering
```

For example:

```python
hierarchical = AgglomerativeClustering(
    n_clusters=3
)
```

Then:

```python
labels = hierarchical.fit_predict(X)
```

The model assigns each observation to one of the requested final clusters.

The current scikit-learn implementation supports different linkage methods and distance metrics. ([scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html?utm_source=chatgpt.com))

---

# 38. Dendrograms

A major advantage of hierarchical clustering is the ability to visualize the hierarchy using a **dendrogram**.

A simplified dendrogram looks like:

```text
           ┌───────────────┐
           │               │
       ┌───┴───┐       ┌───┴───┐
       │       │       │       │
       A       B       C       D
```

The bottom represents individual observations.

Moving upward represents successive merges.

---

# 39. Understanding Merge Height

Suppose:

```text
A and B
```

are merged at a low height.

This indicates that they are relatively close according to the chosen linkage method.

Suppose:

```text
AB and CD
```

are merged much higher.

This indicates that those two groups were more separated.

Therefore:

> The vertical position of a merge gives information about the distance at which groups were combined.

---

# 40. Cutting a Dendrogram

A dendrogram contains many possible levels of grouping.

For example:

```text
4 individual observations
       ↓
3 groups
       ↓
2 groups
       ↓
1 group
```

A horizontal cut can be used to select a particular number of clusters.

For example:

```text
------------------  ← cut

       │       │
       │       │
      ABC      D
```

The exact interpretation depends on where the hierarchy is cut.

---

# 41. Question: Why Can Hierarchical Clustering Be Useful?

Hierarchical clustering can be particularly useful when the relationships between observations are important.

For example, it can reveal:

```text
Observation A
   and
Observation B
      ↓
very similar

Observation C
   and
Observation D
      ↓
very similar

AB
and
CD
      ↓
more different
```

This multi-level structure is not directly represented by ordinary K-Means output.

---

# 42. Clustering Depends on the Features

Suppose a dataset contains:

```text
Age
Income
Spending
```

Clustering with:

```text
Age + Income
```

may produce one grouping.

Clustering with:

```text
Income + Spending
```

may produce another.

Why?

Because the features define the space in which similarity is measured.

Changing the features changes:

$$
distance
$$

and therefore can change the clusters.

---

# 43. Question — Does Changing the Features Matter?

Suppose:

```text
Model A:
Age + Income

Model B:
Income + Spending
```

produces different clusters.

Is this surprising?

### LLM Hint Prompt

> Explain why changing the features used in a clustering algorithm can change the resulting groups. Focus on how the selected features define the feature space and therefore change distances between observations.

<details>
<summary>Solution</summary>

No.

Clustering is based on relationships between observations in the selected feature space.

Changing the features changes the coordinates of the observations and therefore changes their distances.

As a result, the clustering algorithm may discover different groups.

</details>

---

# 44. Clustering Does Not Automatically Find "True" Groups

This is an important limitation.

Suppose:

```python
KMeans(n_clusters=3)
```

is applied to a dataset.

The algorithm will produce three clusters.

This does not prove that:

> The data naturally contain three objectively correct groups.

The algorithm has been instructed to create three clusters.

The analyst must decide whether those clusters are useful and meaningful.

---

# 45. Cluster Interpretation

Suppose a clustering algorithm produces:

```text
Cluster 0
Cluster 1
Cluster 2
```

The labels alone provide little interpretation.

We need to examine the characteristics of each cluster.

For example:

| Cluster | Average Age | Average Income | Average Spending |
| ------- | ----------: | -------------: | ---------------: |
| 0       |          23 |         25,000 |               20 |
| 1       |          42 |         55,000 |               70 |
| 2       |          60 |         85,000 |               25 |

This allows us to describe the groups.

For example:

```text
Cluster 0:
younger, lower-income, lower-spending

Cluster 1:
middle-aged, medium-income, high-spending

Cluster 2:
older, higher-income, lower-spending
```

The descriptions are interpretations based on the cluster characteristics.

---

# 46. Cluster Labels vs Meaningful Categories

A cluster label such as:

```text
Cluster 1
```

does not mean:

```text
Premium Customer
```

unless the data analysis gives that interpretation.

The algorithm produces:

```text
Cluster 1
```

The analyst examines the data and may later describe it as:

```text
High-income, high-spending customers
```

This distinction is important.

---

# 47. Unsupervised Learning Can Still Be Evaluated

Although clustering does not use target labels during training, the resulting clustering can still be evaluated.

Possible approaches include:

* inertia;
* silhouette score;
* visualization;
* cluster size;
* domain knowledge;
* comparison with known information after clustering.

The evaluation question is different from supervised learning.

Supervised learning asks:

> How accurately did the model predict the known target?

Clustering asks:

> How coherent and useful is the structure that was discovered?

---

# 48. Supervised and Unsupervised Evaluation

### Supervised learning

There is a known answer.

For example:

```text
Actual MPG
Predicted MPG
```

The prediction error can be calculated directly.

### Unsupervised learning

There is no target used during clustering.

The evaluation therefore often focuses on:

```text
cohesion
separation
compactness
stability
interpretability
```

This is one reason unsupervised learning can be more difficult to evaluate objectively.

---

# 49. A Simple Comparison

| Concept                  | Supervised                           | Unsupervised            |
| ------------------------ | ------------------------------------ | ----------------------- |
| Target used for training | Yes                                  | No                      |
| Main objective           | Predict target                       | Discover structure      |
| Example                  | Regression                           | Clustering              |
| Output                   | Prediction                           | Cluster/group structure |
| Evaluation               | Compare prediction with known target | Evaluate structure      |
| Example metric           | RMSE                                 | Silhouette score        |

---

# 50. K-Means vs Hierarchical Clustering

The main differences can be summarized as follows.

| Characteristic                       | K-Means                | Hierarchical Clustering              |
| ------------------------------------ | ---------------------- | ------------------------------------ |
| Basic idea                           | Group around centroids | Progressively merge groups           |
| Centroids                            | Yes                    | Not central to the method            |
| K specified                          | Yes                    | Often specified for final clustering |
| Hierarchy                            | No                     | Yes                                  |
| Dendrogram                           | No                     | Yes                                  |
| Distance matters                     | Yes                    | Yes                                  |
| Feature scaling can matter           | Yes                    | Yes                                  |
| Different settings can change result | Yes                    | Yes                                  |

Neither algorithm is universally better.

The appropriate choice depends on:

* the data;
* the shape of the groups;
* the number of observations;
* computational requirements;
* the purpose of the analysis.

---

# 51. Limitations of K-Means

K-Means is useful, but several limitations should be remembered.

### Number of clusters

K must be selected.

### Feature scale

Different units can distort distances.

### Cluster shape

K-Means is particularly natural for compact groups around centers.

### Outliers

Extreme observations can influence centroids.

### Initialization

Different initial centroids can produce different results, which is why multiple initializations are used.

### Interpretation

The algorithm produces groups but does not automatically tell us why those groups are meaningful.

---

# 52. Limitations of Hierarchical Clustering

Hierarchical clustering also has limitations.

For example:

* different linkage methods can produce different results;
* distance choices matter;
* large datasets can be computationally demanding;
* the resulting hierarchy can be difficult to interpret;
* clusters may not correspond to meaningful real-world categories.

---

# 53. A Broader View of Unsupervised Learning

Clustering is only one part of unsupervised learning.

The major categories introduced in this course can be summarized as:

```text
Unsupervised Learning
        |
        +-- Clustering
        |      ├── K-Means
        |      └── Hierarchical
        |
        +-- Anomaly Detection
        |
        +-- Association Rule Learning
        |
        +-- Dimensionality Reduction
        |
        +-- Density Estimation
               +
            Generative Modeling
```

Each category asks a different question.

---

# 54. Clustering

Question:

> What groups exist in the data?

Example:

```text
Customer data
      ↓
customer segments
```

---

# 55. Anomaly Detection

Question:

> Which observations appear unusual?

Example:

```text
Transaction data
      ↓
unusual transactions
```

A detected anomaly is not automatically an error or fraudulent event.

It simply means that the observation is unusual according to the model.

---

# 56. Association Rule Learning

Question:

> Which items frequently occur together?

Example:

```text
Bread → Butter
```

This represents an association.

It does not establish causation.

---

# 57. Dimensionality Reduction

Question:

> Can the data be represented using fewer dimensions?

For example:

```text
100 features
      ↓
PCA
      ↓
2 dimensions
```

This can make high-dimensional data easier to visualize or analyze.

A reduced representation usually involves some information loss or approximation.

---

# 58. Density Estimation and Generative Modeling

At a high level, density estimation and generative modeling attempt to learn the underlying structure or distribution of data.

For example:

```text
Observed data
      ↓
learn distribution
      ↓
analyze distribution
or
generate new observations
```

Examples include:

* Gaussian Mixture Models;
* Variational Autoencoders;
* Generative Adversarial Networks.

These topics require additional concepts beyond the introductory clustering material.

---

# 59. Why the Data Representation Matters

Unsupervised learning is particularly dependent on how the data are represented.

For example, suppose a person is described by:

```text
Age
Income
Spending
```

The clustering result is based on those measurements.

If the features change:

```text
Age
Income
Number of Purchases
```

the distances change.

If the scales change, the distances change.

If the representation changes, the discovered structure can change.

Therefore:

> **The result of unsupervised learning is strongly connected to the representation of the data.**

---

# 60. A Practical Mental Model

When encountering an unsupervised-learning problem, ask:

### Question 1

What kind of data are available?

### Question 2

What question needs to be answered?

### Question 3

Is the objective:

```text
find groups?
```

or:

```text
find unusual observations?
```

or:

```text
find co-occurring items?
```

or:

```text
reduce dimensions?
```

### Question 4

What similarity, distance, or probability assumptions are appropriate?

### Question 5

How will the discovered structure be evaluated and interpreted?

---

# 61. Key Ideas to Remember

## Unsupervised learning

The learning process does not use a target variable to guide the discovery of structure.

## Clustering

Clustering groups observations according to similarity.

## Distance

Distance provides a way to quantify how close observations are.

## K-Means

K-Means creates `K` clusters around centroids.

## Centroid

A centroid represents the mean location of observations assigned to a cluster.

## Cluster label

A cluster label is an identifier, not automatically a meaningful category.

## Hierarchical clustering

Agglomerative hierarchical clustering progressively merges observations and creates a hierarchy.

## Dendrogram

A dendrogram displays the hierarchy of merges.

## Feature representation

The selected features and their scales influence the clustering result.

## Interpretation

A clustering output does not automatically represent a meaningful real-world grouping.

---

# 62. Final Conceptual Comparison

The entire activity can be summarized as:

```text
Supervised Learning
        ↓
Known target
        ↓
Learn to predict


Unsupervised Learning
        ↓
No target guiding learning
        ↓
Discover structure
```

For clustering:

```text
Observations
     ↓
Similarity / Distance
     ↓
Groups
```

For K-Means:

```text
Choose K
     ↓
Centroids
     ↓
Assign observations
     ↓
Update centroids
     ↓
Repeat
```

For hierarchical clustering:

```text
Individual observations
     ↓
Merge similar groups
     ↓
Hierarchy
     ↓
Dendrogram
```

---

# 63. Final Questions for Review

<details>
<summary>What is the main difference between supervised and unsupervised learning?</summary>

Supervised learning uses a known target during training. Unsupervised learning does not use a target to guide the discovery of structure.

</details>

<details>
<summary>What is clustering?</summary>

Clustering is an unsupervised-learning approach that groups observations according to their similarity or distance in the selected feature space.

</details>

<details>
<summary>Why is distance important in K-Means?</summary>

K-Means assigns observations to the nearest centroid, so distance determines which cluster an observation belongs to.

</details>

<details>
<summary>What does K mean in K-Means?</summary>

K is the number of clusters that the algorithm is asked to create.

</details>

<details>
<summary>What is a centroid?</summary>

A centroid is the mean position of the observations assigned to a cluster.

</details>

<details>
<summary>Why are cluster labels such as 0, 1, and 2 arbitrary?</summary>

They are identifiers generated by the algorithm and do not have inherent real-world meaning.

</details>

<details>
<summary>Why does increasing K usually reduce inertia?</summary>

More clusters provide more centroids, allowing observations to be closer to their assigned centers.

</details>

<details>
<summary>Why can we not simply choose the largest K?</summary>

Because a very large number of clusters can reduce inertia without producing meaningful or useful groups.

</details>

<details>
<summary>What is the elbow method?</summary>

It examines how inertia changes as K increases and looks for a point where additional clusters provide substantially smaller improvements.

</details>

<details>
<summary>What is hierarchical clustering?</summary>

Hierarchical clustering creates a hierarchy by progressively merging observations or groups according to a chosen similarity or linkage rule.

</details>

<details>
<summary>What is a dendrogram?</summary>

A dendrogram is a visual representation of the hierarchy created during hierarchical clustering.

</details>

<details>
<summary>Why can scaling affect clustering?</summary>

Distance-based algorithms are sensitive to numerical scale. A feature with much larger numerical values can dominate the distance calculation.

</details>

<details>
<summary>Does a clustering algorithm automatically discover objectively correct groups?</summary>

No. The result depends on the features, representation, algorithm, parameters, and assumptions. Clusters must be evaluated and interpreted.

</details>

---

# 64. Final Summary

Unsupervised learning begins with a different question from supervised learning.

Instead of:

> What target value should be predicted?

the question may be:

> What structure is present in the data?

Clustering is one way to answer this question.

K-Means identifies groups around centroids.

Hierarchical clustering creates a hierarchy by progressively merging groups.

Both methods depend on how observations are represented and how similarity or distance is defined.

The most important concepts are:

$$
\boxed{
\text{Data}
\rightarrow
\text{Similarity}
\rightarrow
\text{Clusters}
}
$$

and:

$$
\boxed{
\text{Clustering output}
\neq
\text{automatically meaningful categories}
}
$$

The clustering result is a model of structure in the data. Its usefulness depends on whether that structure is coherent, stable, interpretable, and relevant to the problem being investigated.



---

# Background resources:

* https://www.geeksforgeeks.org/machine-learning/unsupervised-learning/
* https://www.geeksforgeeks.org/machine-learning/unsupervised-machine-learning-examples/
* https://www.geeksforgeeks.org/data-science/choosing-the-right-clustering-algorithm-for-your-dataset/
* https://www.geeksforgeeks.org/machine-learning/different-types-clustering-algorithm/
* https://www.geeksforgeeks.org/data-analysis/cluster-analysis/
* https://www.geeksforgeeks.org/machine-learning/clustering-in-machine-learning/
* https://www.interactive-ml.com/k-means.html
* https://www.interactive-ml.com/hclustering.html

