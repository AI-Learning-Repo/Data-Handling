Absolutely. Below is the rewritten theory using your four-module structure. I have retained the important concepts, mathematical explanations, Python examples, and review questions, while reducing repetition and improving the learning progression.

# Activity 1 — Introduction to Unsupervised Learning with Clustering

## Table of Contents

1. Module 1: Unsupervised Foundations & Distance Metrics

   * Supervised and unsupervised learning

   * Clustering and similarity

   * Euclidean distance

   * Feature space and feature scale
2. Module 2: Centroid-Based Clustering (K-Means)

   * K-Means workflow

   * Centroids and cluster assignments

   * Scikit-learn implementation

   * Experiments with `K`

   * Inertia and choosing the number of clusters
3. Module 3: Hierarchical Agglomerative Clustering

   * Hierarchical clustering

   * Agglomerative workflow

   * Linkage methods

   * Dendrograms

   * Comparison with K-Means
4. Module 4: Evaluation, Interpretation & Practical Limitations

   * Evaluating clustering results

   * Interpreting clusters

   * Feature representation

   * Practical limitations

   * Reflection and review

# Module 1: Unsupervised Foundations & Distance Metrics

## Learning objectives

By the end of this module, you should be able to:

* Explain the difference between supervised and unsupervised learning.

* Describe what clustering is and why it is useful.

* Explain similarity and distance between observations.

* Calculate Euclidean distance.

* Describe feature space.

* Explain why feature selection and numerical scale matter.

## 1. Introduction to Unsupervised Learning

Machine learning can be divided into several broad categories.

Two important categories are:

* Supervised learning

* Unsupervised learning

The main difference is whether a target variable is used to guide the learning process.

### Supervised learning

In supervised learning, the data contain features and a known target.

```
Features
   +
Known target
      ↓
Machine learning model
      ↓
Prediction
```

### Unsupervised learning

In unsupervised learning, the data contain features without a target guiding the learning process.

```
Features
   ↓
Unsupervised learning
   ↓
Discover structure
```

The model attempts to identify patterns or relationships within the data.

## 2. Supervised Learning

In supervised learning, the model learns a relationship between input features and a target variable.

For example, consider a regression problem using car data.

```
Weight
Horsepower
Displacement
      ↓
Regression model
      ↓
MPG
```

The target is:

y=MPGy = MPGy=MPG

The model is trained using cars for which the MPG is already known.

### Regression

Regression predicts a numerical value.

Examples include:

* House price

* Taxi fare

* MPG

* Temperature

### Classification

Classification predicts a category.

Examples include:

* Spam / not spam

* Survived / did not survive

* Species of an Iris flower

The important point is that supervised learning uses known target information during training.

## 3. Unsupervised Learning

In unsupervised learning, the model receives feature data but does not receive a target variable to guide the discovery of structure.

Conceptually:

```
Supervised learning:

X + known y
     ↓
Learn relationship
     ↓
Predict y

Unsupervised learning:

X
↓
Discover structure
```

Possible structures include:

* Groups of similar observations

* Unusual observations

* Frequently occurring item combinations

* Lower-dimensional representations

* Estimates of the underlying data distribution

Activity 1 focuses on clustering.

### Examples of unsupervised learning

|
Method

|

Main question

|
| --- | --- |
|

Clustering

|

Which observations form groups?

|
|

Anomaly detection

|

Which observations are unusual?

|
|

Association rule learning

|

Which items frequently occur together?

|
|

Dimensionality reduction

|

Can data be represented using fewer dimensions?

|
|

Density estimation

|

What distribution may have generated the data?

|

These methods address different problems, but they all work without using a target variable to guide the main learning process.

## 4. What Is Clustering?

Clustering is an unsupervised-learning technique used to group similar observations.

The basic idea is:

> Observations that are similar should be placed in the same group, while observations that are less similar should be placed in different groups.

Suppose a dataset contains customer characteristics:

```
Age
Income
Spending
```

A clustering algorithm might discover groups such as:

```
Group 1:
Younger, lower-income customers

Group 2:
Middle-aged, medium-income customers

Group 3:
Older, higher-income customers
```

These groups were not provided as target labels. The algorithm attempts to discover them from the feature values.

### Why is clustering useful?

Clustering can help with:

* Customer segmentation

* Grouping similar products

* Exploring scientific measurements

* Discovering patterns in datasets

* Organising observations into meaningful groups

The usefulness of clustering depends on whether the discovered groups are coherent and relevant to the problem.

## 5. Clustering vs Classification

Clustering and classification can appear similar because both produce groups.

However, the groups are obtained differently.

### Classification

The categories are known.

For example:

```
Training data:

Flower A → setosa
Flower B → versicolor
Flower C → virginica
```

The model learns to predict these known classes.

### Clustering

The groups are not provided.

```
Flower A
Flower B
Flower C
...
```

The algorithm attempts to discover groups based on the measurements.

Therefore:

> Classification predicts known categories; clustering discovers groups from the data.

A cluster label is not automatically equivalent to a known class label.

## 6. A Dataset Without a Target

Suppose a customer dataset contains:

|
Age

|

Income

|

Spending

|
| --- | --- | --- |
|

22

|

25,000

|

30

|
|

24

|

28,000

|

35

|
|

45

|

65,000

|

80

|
|

47

|

62,000

|

75

|
|

60

|

90,000

|

25

|

There is no column such as:

```
Customer_Group
```

A clustering algorithm can still examine:

```
Age
Income
Spending
```

and attempt to discover groups.

This is the fundamental idea behind unsupervised clustering.

## 7. Similarity

To create groups, a clustering algorithm needs some definition of similarity.

Two observations might be considered similar if their feature values are close.

For example:

A=(2,3)A = (2,3)A=(2,3)

B=(2.5,3.2)B = (2.5,3.2)B=(2.5,3.2)

These observations are close together.

Another observation:

C=(9,10)C = (9,10)C=(9,10)

is much farther away.

We might therefore consider:

```
A and B → similar
A and C → less similar
```

The exact definition of similarity depends on the algorithm and the distance or similarity measure being used.

## 8. Distance

One common way to measure similarity is to calculate distance.

The intuition is:

```
Small distance
      ↓
More similar

Large distance
      ↓
Less similar
```

A commonly used distance measure is Euclidean distance.

## 9. Euclidean Distance in Two Dimensions

Suppose we have two observations:

A=(x1,y1)A=(x_1,y_1)A=(x1,y1)

and:

B=(x2,y2)B=(x_2,y_2)B=(x2,y2)

The Euclidean distance is:

d(A,B)=(x2−x1)2+(y2−y1)2d(A,B)= \sqrt{ (x_2-x_1)^2+ (y_2-y_1)^2 }d(A,B)=(x2−x1)2+(y2−y1)2

### Example

Consider:

A=(1,2)A=(1,2)A=(1,2)

and:

B=(4,6)B=(4,6)B=(4,6)

Then:

d=(4−1)2+(6−2)2d= \sqrt{ (4-1)^2+ (6-2)^2 }d=(4−1)2+(6−2)2

=32+42= \sqrt{ 3^2+4^2 }=32+42

=9+16= \sqrt{9+16}=9+16

=25= \sqrt{25}=25

Therefore:

d=5\boxed{d=5}d=5

The Euclidean distance between the two observations is 5.

### Python calculation

Python

Run

```
import numpy as np

A = np.array([1, 2])
B = np.array([4, 6])

distance = np.linalg.norm(A - B)

print(distance)
```

Output:

```
5.0
```

## 10. Why Does Distance Matter?

Suppose three observations are:

```
A = (1, 2)
B = (2, 3)
C = (9, 9)
```

The distance from A to B is small.

The distance from A to C is much larger.

A clustering algorithm can use this information to identify groups.

```
A ─── B


C
```

This suggests that A and B might belong to one group while C belongs elsewhere.

This is only an intuition. Different clustering algorithms use distance differently.

## 11. Feature Space

When observations have two numerical features, they can be represented on a two-dimensional graph.

```
y-axis → Feature 2
x-axis → Feature 1
```

Each observation becomes a point.

For example:

A=(2,3)A=(2,3)A=(2,3)

means that the observation has:

* Feature 1 = 2

* Feature 2 = 3

With three features, an observation exists in three-dimensional feature space.

With 100 features, it exists in a 100-dimensional feature space.

Although we cannot easily visualise 100 dimensions directly, the same mathematical concept applies.

### Two-dimensional feature space

![How to create a sample dataset using Python Scikit-learn?](https://images.openai.com/static-rsc-4/jjIuJSOOos2K-pWPDiMBEQ-MyJwISFtpcZTCwJMO85QT59B312OidEuVJoGsBE-ZEGLn3Nwy-_OW7x2lbE9xPYMHfnCFr4GNTtFKlma4_xNE1JJFJnj1w6-6iuIMQY5vim1mLnrCSm8grJb-6lLoYmyP7yZMWurf8pI2DG-zVyY?purpose=inline)

![Machine Learning Roadmap](https://images.openai.com/static-rsc-4/-m16SWX5wSXdVFdwe6iODMzs6HPgkgryAYmqHTJWHGEvsJfQ_dDpv7nBH5iNc15GjNbW178lO_BLXyK-Bqal1THGojHVRyQO8f7_YnDzOi1iRx7AnWms6FmyU6SZ-dieWNodIncVrr3QwywldiqXFxCZ4fFKkewWhDyed7x9Zrc?purpose=inline)

![Introduction to Machine Learning - NYU ML Summer School](https://images.openai.com/static-rsc-4/piQx4mbbglHYaiD1aT46Vf4vBj6Ob6QokJ1AiF0Ix5AnyS8zgB7Haj_Qt15CzR7uPHYiRi80-HOr0DA9YvzCJsNZc8CujRYTjvZu0ZP2DiUvZ6EhZzfio8wL3gNqBCev_rSqdAArXPwEPmDCxKFLb7ypQlhAvxE95EGA2VVj6cA?purpose=inline)

A dataset with two numerical features can be plotted directly.

For example:

```
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

## 12. Feature Representation and Distance

The selected features define the space in which similarity is measured.

Suppose a dataset contains:

```
Age
Income
Spending
```

Clustering with:

```
Age + Income
```

may produce one grouping.

Clustering with:

```
Income + Spending
```

may produce another.

Why?

Because changing the features changes the coordinates of the observations.

The distances between observations can therefore change.

As a result, the clustering algorithm may discover different groups.

### Important idea

> The result of clustering depends on the features used to represent the observations.

## 13. Feature Scale

Because many clustering methods use distances, the numerical scale of features matters.

Suppose two features are:

```
Age:
20–80

Income:
20,000–200,000
```

Income has much larger numerical values.

Without scaling, differences in income can dominate the distance calculation.

This can cause clustering to be driven primarily by income.

### Example

Imagine two observations:

```
A:
Age = 30
Income = 40,000

B:
Age = 35
Income = 41,000
```

The raw differences are:

```
Age difference = 5
Income difference = 1,000
```

When Euclidean distance is calculated, the income difference is numerically much larger.

This means the income feature can have a much greater influence on the distance.

### Standardisation

A common method is standardisation:

z=x−μσz=\frac{x-\mu}{\sigma}z=σx−μ

where:

* xxx is the original feature value;

* μ\muμ is the feature mean;

* σ\sigmaσ is the feature standard deviation.

Scaling transforms the features to comparable numerical scales.

The choice of scaling method is an important practical consideration in clustering.

## Module 1 — Key ideas

* Supervised learning uses a known target.

* Unsupervised learning discovers structure without a target guiding the main learning process.

* Clustering groups similar observations.

* Distance measures how close observations are.

* Euclidean distance is a common distance measure.

* Feature space represents observations using their feature values.

* Changing the features can change the clustering result.

* Numerical feature scale can influence distance-based clustering.

# Module 2: Centroid-Based Clustering — K-Means

## Learning objectives

By the end of this module, you should be able to:

* Explain what K-Means clustering is.

* Describe the role of centroids.

* Explain the K-Means workflow.

* Implement K-Means using scikit-learn.

* Interpret cluster labels and cluster centres.

* Understand inertia.

* Experiment with different values of K.

## 14. Introduction to K-Means Clustering

K-Means is one of the most widely used clustering algorithms.

The basic objective is to divide observations into:

KKK

clusters.

The algorithm attempts to create groups in which observations are relatively close to their assigned cluster centre.

The central idea is:

> K-Means creates groups around centroids and assigns observations to the nearest centroid.

## 15. What Does `K` Mean?

In:

K-MeansK\text{-Means}K-Means

the `K` is the number of clusters to create.

For example:

K=2K=2K=2

means:

```
Create 2 clusters
```

while:

K=4K=4K=4

means:

```
Create 4 clusters
```

The algorithm does not automatically know the appropriate value of K.

Choosing K is an important modeling decision.

## 16. Cluster Centroids

K-Means uses centroids.

A centroid is the mean position of the observations assigned to a cluster.

Suppose a cluster contains:

(1,2)(1,2)(1,2)

(2,3)(2,3)(2,3)

(3,4)(3,4)(3,4)

The centroid is found by taking the mean of each feature.

### First feature

1+2+33=2\frac{1+2+3}{3}=231+2+3=2

### Second feature

2+3+43=3\frac{2+3+4}{3}=332+3+4=3

Therefore:

Centroid=(2,3)\boxed{\text{Centroid}=(2,3)}Centroid=(2,3)

The centroid represents the centre of the observations currently assigned to that cluster.

### Why are centroids important?

K-Means uses centroids to determine which observations belong to which groups.

An observation is assigned to the nearest centroid.

## 17. The K-Means Algorithm

K-Means can be understood as an iterative process.

### Step 1 — Choose K

Suppose:

K=3K=3K=3

The algorithm will attempt to create three clusters.

### Step 2 — Initialise the centroids

The algorithm chooses initial cluster centres.

### Step 3 — Assign observations

Each observation is assigned to the nearest centroid.

### Step 4 — Recalculate the centroids

The centroid of each cluster is recalculated from its current observations.

### Step 5 — Repeat

The assignment and centroid updates are repeated until the solution stabilises according to the algorithm's stopping criteria.

### Workflow

```
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

### Conceptual example

![Concepts in Machine Learning Class 3: Unsupervised Learning | Concepts in Machine Learning](https://images.openai.com/static-rsc-4/9EWIywzIL9PxDwb69VIoI8TK_eMiQ7eLXWx1vNxI4sAnie8EqjPkmEPkhCehbYN4aEJ-msYS26rn5gqGJ4XDhGhoI6SaVd6P2eQQzNWkxu3yge4ea6QZpKuW5wNtf6M8csu5cMu0aWzb2aCUhHWR2MGwDjlbNnCr8pgGNnlpAj8?purpose=inline)

![K-Means Clustering (with Pyscript demo) - Raden Muhammad Mu’az](https://images.openai.com/static-rsc-4/zbej2pPkR_MUGZxjC-9fKGvNRBJeJNBbGzamzQvwLtATd3IKpSqxkd8ZMvVxKP6sjNnetb4Uhc-sWOjixcy4bFj0ZVgW3imHS2F3rLlqjvUmdLym3ntpgzb_gscTiosYumlZ_NtHQmwe72_HlfBtUBFNGdo4i5IBHFjJqmSlAhw?purpose=inline)

![A K-Means Clustering Approach for Accelerated Path Planning in GMA-DED: The Fast Advanced-Pixel Strategy | MDPI](https://images.openai.com/static-rsc-4/FJemswZGtpI4So9vRy1aathaWtS--b27OT3BloM1wCm4hJnL7hsmkcI66nDu2LUnLtKl3oYmiFIOMansYQCKtqZ6DiTojHGUyatOFpAt-_1CAr82OlH7DYiIo0DuGjonM-RIdfJdS8nNnJPZokKx6xxx2mDF8bAiZeaNhe-yXY4?purpose=inline)

4

The centroids move as observations are reassigned.

The process attempts to create compact groups around their centres.

## 18. Why Is K-Means Called "K-Means"?

The name comes from two ideas.

### K

The number of groups:

KKK

### Means

Each cluster centre is calculated as the mean of the observations assigned to that cluster.

Therefore:

> K-Means creates K groups based on distances to cluster means, also called centroids.

## 19. K-Means in Scikit-Learn

Scikit-learn provides the `KMeans` implementation.

Python

Run

```
from sklearn.cluster import KMeans
```

A model can be created as:

Python

Run

```
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init="auto"
)
```

### Main parameters

#### `n_clusters`

Specifies the desired number of clusters.

Python

Run

```
n_clusters=3
```

#### `random_state`

Controls the random number generation used by the algorithm.

It helps make results reproducible when the same settings and data are used.

#### `n_init`

Controls the number or strategy of initialisation runs.

In current scikit-learn versions, `n_init="auto"` selects the number of runs based on the initialisation strategy.

For exact behaviour, consult the version of the scikit-learn documentation used in your environment.

## 20. Fitting K-Means

Unlike supervised learning:

Python

Run

```
model.fit(X_train, y_train)
```

K-Means does not receive a target.

Instead:

Python

Run

```
kmeans.fit(X)
```

is sufficient.

The model receives the feature data and attempts to discover cluster structure.

### Example

Python

Run

```
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init="auto"
)

kmeans.fit(X)
```

Here:

* `X` contains the feature data.

* `n_clusters=3` requests three clusters.

* The model learns centroids and cluster assignments.

## 21. Cluster Labels

After fitting K-Means, cluster assignments can be obtained using:

Python

Run

```
kmeans.labels_
```

For example:

```
[0, 0, 2, 1, 2, 1, 0, ...]
```

These values represent cluster identifiers.

They are not target classes.

The values `0`, `1`, and `2` simply identify the clusters produced by the algorithm.

### Using `fit_predict`

You can also fit the model and obtain labels in one step:

Python

Run

```
labels = kmeans.fit_predict(X)
```

This returns an array of cluster assignments.

## 22. Cluster Labels Are Arbitrary

Suppose one run produces:

```
Cluster 0 → left group
Cluster 1 → right group
Cluster 2 → middle group
```

Another run might label exactly the same groups:

```
Cluster 2 → left group
Cluster 0 → right group
Cluster 1 → middle group
```

The cluster numbers themselves do not have inherent meaning.

Therefore:

> Cluster 0 is not automatically more important than Cluster 1.

The meaningful information is the membership and characteristics of the groups.

## 23. Cluster Centres in Scikit-Learn

K-Means stores its centroids in:

Python

Run

```
kmeans.cluster_centers_
```

If there are:

```
3 clusters
```

and:

```
2 features
```

then:

Python

Run

```
kmeans.cluster_centers_.shape
```

will be:

```
(3, 2)
```

because there are:

* Three centroids

* Two coordinates per centroid

### Example

Python

Run

```
print(kmeans.cluster_centers_)
```

Possible output:

```
[[2.1 3.4]
 [8.2 7.9]
 [5.0 2.1]]
```

Each row represents one cluster centre.

## 24. Why Do Centroids Matter?

Suppose the centroids are:

```
C1 = (2, 3)
C2 = (8, 9)
C3 = (5, 2)
```

For a new observation:

```
P = (4, 3)
```

the algorithm can calculate its distance to each centroid.

For example:

```
Distance to C1 = 2
Distance to C2 = 7
Distance to C3 = 3
```

The smallest distance is:

```
Distance to C1 = 2
```

Therefore, the observation is assigned to the cluster represented by C1.

This is the central intuition behind K-Means.

## 25. K-Means Objective

K-Means attempts to make observations close to their assigned cluster centres.

A simplified mathematical description is:

∑i=1n∥xi−μci∥2\sum_{i=1}^{n} \left\| x_i-\mu_{c_i} \right\|^2i=1∑n∥xi−μci∥2

where:

* xix_ixi is observation iii;

* μci\mu_{c_i}μci is the centroid of the cluster assigned to observation iii;

* cic_ici is the assigned cluster.

The algorithm attempts to minimise this within-cluster squared distance.

This quantity is closely related to what scikit-learn reports as inertia.

For this introductory activity, the main idea is:

> K-Means tries to create compact groups around their centroids.

## 26. Inertia

After fitting K-Means, scikit-learn provides:

Python

Run

```
kmeans.inertia_
```

Inertia measures the total within-cluster squared distance from observations to their assigned centroids.

A lower value means the observations are, overall, closer to their cluster centres.

### Example

Python

Run

```
print(kmeans.inertia_)
```

Inertia is useful for comparing K-Means solutions, but:

> Lower inertia by itself does not tell us the best number of clusters.

## 27. Why Does More K Usually Reduce Inertia?

Suppose we use:

K=1K=1K=1

Almost all observations must be represented by one centroid.

Now use:

K=3K=3K=3

There are more centroids.

The observations can therefore be represented more closely.

Now use:

K=10K=10K=10

There are even more centroids.

The algorithm has even greater flexibility.

Therefore:

> Inertia generally decreases when K increases.

This creates an important problem.

If the objective were simply to minimise inertia, we could keep increasing K.

That does not necessarily produce meaningful clusters.

## 28. Choosing K

The appropriate number of clusters is often not known in advance.

Possible sources of information include:

* Domain knowledge

* Visual inspection

* Elbow method

* Silhouette score

* Comparison of different clusterings

Activity 1 introduces the idea that K is a modeling choice.

Detailed model-selection techniques can be developed further in later activities.

## 29. The Elbow Method

The elbow method examines how inertia changes as K increases.

Suppose:

|
K

|

Inertia

|
| --- | --- |
|

1

|

2000

|
|

2

|

1100

|
|

3

|

600

|
|

4

|

350

|
|

5

|

300

|
|

6

|

270

|

The major improvements occur early.

After approximately:

K=4K=4K=4

the improvements become much smaller.

The curve may therefore look like an elbow around 4.

The idea is:

> Choose a K where additional clusters begin to provide relatively little additional improvement.

The elbow method is a heuristic.

There may be no perfectly clear elbow.

### Example Python experiment

Python

Run

```
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

inertias = []

for k in range(1, 8):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init="auto"
    )

    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 8), inertias, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()
```

This experiment allows you to observe how inertia changes as K increases.

## 30. Silhouette Score — Introduction

The silhouette score provides another way to evaluate cluster separation.

It considers two ideas:

1. How well an observation fits its own cluster.

2. How separated it is from neighbouring clusters.

The score ranges from:

−1-1−1

to:

111

A value closer to 1 generally indicates that observations are well matched to their own cluster and separated from nearby clusters.

A value near zero indicates observations near cluster boundaries.

Negative values can indicate potentially poor assignments.

### Python example

Python

Run

```
from sklearn.metrics import silhouette_score

score = silhouette_score(X, labels)

print(score)
```

Detailed silhouette analysis belongs to the more advanced clustering activity.

## Module 2 — Practical experiment

Try different values of K using the same dataset.

Python

Run

```
from sklearn.cluster import KMeans

for k in [2, 3, 4]:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init="auto"
    )

    labels = kmeans.fit_predict(X)

    print("K =", k)
    print("Inertia =", kmeans.inertia_)
    print("Centres =")
    print(kmeans.cluster_centers_)
    print()
```

### Questions

1. How does the number of clusters change?

2. How do the centroids change?

3. Does inertia decrease as K increases?

4. Do the discovered groups appear meaningful?

5. Would you automatically choose the largest K?

## Module 2 — Key ideas

* K-Means creates K clusters.

* A centroid is the mean position of a cluster.

* Observations are assigned to the nearest centroid.

* K-Means iteratively updates assignments and centroids.

* Cluster labels are identifiers, not meaningful categories by themselves.

* Inertia measures within-cluster squared distance.

* Increasing K generally reduces inertia.

* Choosing K requires interpretation and evaluation.

# Module 3: Hierarchical Agglomerative Clustering

## Learning objectives

By the end of this module, you should be able to:

* Explain hierarchical clustering.

* Describe agglomerative clustering.

* Explain linkage.

* Interpret a dendrogram.

* Understand merge height.

* Implement hierarchical clustering using scikit-learn.

* Compare hierarchical clustering with K-Means.

## 31. Introduction to Hierarchical Clustering

K-Means is only one clustering algorithm.

Another approach is hierarchical clustering.

Hierarchical clustering creates a hierarchy of relationships between observations.

The most common introductory version is:

> Agglomerative hierarchical clustering

It starts with individual observations and progressively combines them.

Unlike K-Means, the process creates a hierarchy of merging groups.

## 32. Agglomerative Clustering

Agglomerative clustering is a bottom-up approach.

It starts with each observation as its own cluster.

Then it progressively merges groups according to a chosen rule.

### Example

Start:

```
A   B   C   D
↓   ↓   ↓   ↓
```

Each observation is its own cluster.

Then:

```
AB   C   D
```

Then:

```
AB   CD
```

Finally:

```
ABCD
```

The algorithm builds the hierarchy from the bottom upward.

### Conceptual diagram

![Hierarchical Clustering vs K-Means Clustering: How do the Clustering Algorithms Differ? | by Mbali Kalirane | Medium](https://images.openai.com/static-rsc-4/UKoe_sN4NnsYund6lTndC7paHBMFOPpZCNsP4-_xIW45Zo0v1c9dsfdlojpyO2FvSEYBbei1lrN_QT0e4PRM4TdpzzHwOBapI_hhs6tHM7dOhGO6kfj9h8RvPTBD6qtG0YFkzuW91mO291deTyFsfPfDh5mSc9Mqx2F7ofHY99s?purpose=inline)

![Unsupervised Learning: Hierarchical Clustering Using Agglomerative and Divisive Algorithms | by Ashkan Beheshti | Medium](https://images.openai.com/static-rsc-4/0B9yXb7USgXbMnLNDfjbyw_HcCAJv-OFprXjcOZ7tWFQPsnpBRKhiT47oZoPV5a6EqYxNEPdmFiYp_gJnMVfZV1K7UmAPflzs_Cnylb-ddPPVxE5Fe_8ThDPIGphAr3hNycbDOBEuHXy91NVkCxfdvYV8_T4eCGhk5mbhXjEnUk?purpose=inline)

![The Beginner’s Guide to Clustering in Machine Learning | by Utsav Desai | Medium](https://images.openai.com/static-rsc-4/YIVFX04JFGrCRlEgijAMmDTy5jL6DAoX4Z29FxXKR4C8Zbz7prkijlhsSN7M5sZbwGb-92VEI9H_JFjWYSQSLgQUDyDFFvCxYF16Y-eeArwMKX0cPrIa5wdg0UQTTJRdTvnZ1D-xg3bZTbOBVwpPI6qASYbZNC7HQqbGcgonC1k?purpose=inline)

4

## 33. How Is It Different from K-Means?

### K-Means

Starts by selecting:

KKK

and works toward creating that number of clusters.

```
Choose K
   ↓
Centroids
   ↓
Assign observations
   ↓
Iterate
```

### Hierarchical clustering

Starts with individual observations and progressively merges groups.

```
Individual observations
   ↓
Merge nearby groups
   ↓
Build hierarchy
```

The key conceptual difference is:

> K-Means builds groups around centroids, while hierarchical clustering builds a hierarchy through successive merges.

## 34. Linkage

When hierarchical clustering decides which groups to merge, it needs a rule for measuring the distance between groups.

This is called linkage.

Common linkage methods include:

* Single linkage

* Complete linkage

* Average linkage

* Ward linkage

The methods use different definitions of the distance between clusters.

### Single linkage

Measures the distance between the closest pair of observations in two clusters.

### Complete linkage

Measures the distance between the farthest pair of observations in two clusters.

### Average linkage

Uses the average distance between observations in two clusters.

### Ward linkage

Merges clusters based on the increase in within-cluster variance.

For Activity 1, the most important point is:

> Different linkage methods can produce different hierarchical clusterings.


## 35. How Hierarchical Agglomerative Clustering Works

Hierarchical agglomerative clustering starts by treating every data point as its own cluster.

The algorithm then repeatedly:

1. Finds the two closest clusters.

2. Merges them into one larger cluster.

3. Recalculates the distances between the new cluster and the remaining clusters.

4. Continues until all data points belong to one cluster.

This creates a hierarchy of clusters, from many small clusters to one large cluster.

### Example

Suppose we have five data points:

```
A, B, C, D, E
```

At the beginning:

```
{A}, {B}, {C}, {D}, {E}
```

The algorithm may then merge the closest points:

```
{A, B}, {C}, {D}, {E}
```

Next, another pair may be merged:

```
{A, B}, {C, D}, {E}
```

The process continues until only one cluster remains:

```
{A, B, C, D, E}
```

The sequence of merges can be represented using a dendrogram.

### Important idea

Unlike K-Means, hierarchical clustering does not require the number of clusters to be specified at the beginning. The full hierarchy is created first, and the desired number of clusters can be selected later.

## 36. Linkage Methods: How Cluster Distance Is Measured

When hierarchical clustering decides which clusters to merge, it needs a way to measure the distance between two clusters.

This is called the linkage method.

Different linkage methods can produce different clustering results.

### 36.1 Single Linkage

Single linkage measures the distance between the two closest points in the two clusters.

D(A,B)=min⁡a∈A, b∈Bd(a,b)D(A,B)=\min_{a\in A,\ b\in B} d(a,b)D(A,B)=a∈A, b∈Bmind(a,b)

In simple terms:

> The distance between two clusters is the distance between their nearest points.

Single linkage can connect points through long chains. This may create elongated or irregularly shaped clusters.

### 36.2 Complete Linkage

Complete linkage measures the distance between the two farthest points in the clusters.

D(A,B)=max⁡a∈A, b∈Bd(a,b)D(A,B)=\max_{a\in A,\ b\in B} d(a,b)D(A,B)=a∈A, b∈Bmaxd(a,b)

In simple terms:

> The distance between two clusters is the distance between their most distant points.

Complete linkage tends to produce more compact and balanced clusters.

### 36.3 Average Linkage

Average linkage calculates the average distance between all pairs of points across the two clusters.

D(A,B)=1∣A∣∣B∣∑a∈A∑b∈Bd(a,b)D(A,B)= \frac{1}{|A||B|} \sum_{a\in A}\sum_{b\in B}d(a,b)D(A,B)=∣A∣∣B∣1a∈A∑b∈B∑d(a,b)

In simple terms:

> The distance between two clusters is the average distance between their points.

Average linkage provides a compromise between single and complete linkage.

### 36.4 Ward Linkage

Ward linkage merges clusters in a way that minimizes the increase in within-cluster variation.

It generally produces compact, relatively spherical clusters and is often useful when using Euclidean distance.

### Comparison of Linkage Methods

|
Linkage method

|

Distance based on

|

Typical tendency

|
| --- | --- | --- |
|

Single

|

Closest pair of points

|

Can create chains

|
|

Complete

|

Farthest pair of points

|

More compact clusters

|
|

Average

|

Average pairwise distance

|

Balanced compromise

|
|

Ward

|

Increase in within-cluster variance

|

Compact, spherical clusters

|

### Python Example

Python

Run

```
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

labels = model.fit_predict(X)
```

Other linkage methods can be selected when appropriate:

Python

Run

```
model = AgglomerativeClustering(
    n_clusters=3,
    linkage="complete"
)
```

Python

Run

```
model = AgglomerativeClustering(
    n_clusters=3,
    linkage="average"
)
```

The selected linkage method affects the structure of the resulting clusters. Therefore, it is useful to experiment with different methods and compare the results.


----

## 37. Understanding Merge Height

In a dendrogram, the vertical position of a merge provides information about the distance at which groups were combined.

Suppose:

```
A and B
```

are merged at a low height.

This indicates that they are relatively close according to the chosen linkage method.

Suppose:

```
AB and CD
```

are merged much higher.

This indicates that those two groups were more separated according to that linkage method.

Therefore:

> The vertical position of a merge gives information about the distance at which groups were combined.

The meaning of merge height depends on the linkage method and distance metric.

## 38. Cutting a Dendrogram

A dendrogram contains many possible levels of grouping.

For example:

```
4 individual observations
       ↓
3 groups
       ↓
2 groups
       ↓
1 group
```

A horizontal cut can be used to select a particular number of clusters.

### Example

```
           ┌───────────┐
           │           │
       ┌───┴───┐       │
       │       │       │
       A       B       C
```

A cut below the final merge could produce:

```
Cluster 1: A, B
Cluster 2: C
```

A cut above all merges would produce:

```
Cluster 1: A, B, C
```

The exact interpretation depends on where the hierarchy is cut.

### Important idea

Hierarchical clustering creates a hierarchy first. A chosen number of clusters can then be obtained by cutting that hierarchy.

## 39. Why Can Hierarchical Clustering Be Useful?

Hierarchical clustering can be useful when the relationships between observations are important.

For example:

```
Observation A
   and
Observation B
      ↓
Very similar

Observation C
   and
Observation D
      ↓
Very similar

AB
and
CD
      ↓
More different
```

This multi-level structure is not directly represented by ordinary K-Means output.

### Example

Imagine a dataset of biological measurements.

Some observations may be very similar, while others form broader groups.

Hierarchical clustering can help reveal:

* Closely related observations.

* Groups within larger groups.

* The order in which observations or groups merge.

This can be useful for exploratory analysis.

## 40. Visualising a Dendrogram in Python

Scikit-learn provides the clustering algorithm, while SciPy provides a convenient function for constructing dendrograms.

Python

Run

```
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

Z = linkage(X, method="ward")

plt.figure(figsize=(10, 5))

dendrogram(Z)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Observations")
plt.ylabel("Merge distance")

plt.show()
```

### Explanation

Python

Run

```
Z = linkage(X, method="ward")
```

This calculates the hierarchical merging structure.

Python

Run

```
dendrogram(Z)
```

This visualises the hierarchy.

The resulting diagram can be examined to understand how observations were progressively merged.

### Note

The dendrogram is calculated using the selected linkage method. If the linkage method changes, the hierarchy and merge heights may also change.

## 41. Hierarchical Clustering Experiment

Use the same dataset that was used for K-Means.

Python

Run

```
from sklearn.cluster import AgglomerativeClustering

hierarchical = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

hierarchical_labels = hierarchical.fit_predict(X)

print(hierarchical_labels)
```

Now compare the output with K-Means.

Python

Run

```
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init="auto"
)

kmeans_labels = kmeans.fit_predict(X)

print("K-Means labels:")
print(kmeans_labels)

print("Hierarchical labels:")
print(hierarchical_labels)
```

### Questions

1. Do both algorithms produce the same number of clusters?

2. Do they assign the same observations to each group?

3. What happens if the linkage method changes?

4. What does the dendrogram reveal about the observations?

5. Why might two clustering algorithms produce different results?

## 42. K-Means vs Hierarchical Clustering

The main differences can be summarised as follows.

|
Characteristic

|

K-Means

|

Hierarchical Clustering

|
| --- | --- | --- |
|

Basic idea

|

Group around centroids

|

Progressively merge groups

|
|

Centroids

|

Yes

|

Not central to the method

|
|

K specified

|

Yes

|

Often specified for final clustering

|
|

Hierarchy

|

No

|

Yes

|
|

Dendrogram

|

No

|

Yes

|
|

Distance matters

|

Yes

|

Yes

|
|

Feature scaling can matter

|

Yes

|

Yes

|
|

Different settings can change result

|

Yes

|

Yes

|

### K-Means

```
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

### Hierarchical clustering

```
Individual observations
   ↓
Merge nearby groups
   ↓
Build hierarchy
   ↓
Dendrogram
```

Neither algorithm is universally better.

The appropriate choice depends on:

* The data.

* The shape of the groups.

* The number of observations.

* Computational requirements.

* The purpose of the analysis.

## Module 3 — Key ideas

* Hierarchical clustering creates a hierarchy of relationships.

* Agglomerative clustering starts with individual observations.

* Groups are progressively merged.

* Linkage defines how distances between groups are measured.

* Different linkage methods can produce different results.

* A dendrogram displays the hierarchy of merges.

* Merge height provides information about the distance at which groups were combined.

* A dendrogram can be cut to obtain a chosen number of clusters.

* Hierarchical clustering and K-Means use different approaches to forming groups.

# Module 4: Evaluation, Interpretation & Practical Limitations

## Learning objectives

By the end of this module, you should be able to:

* Explain why clustering results need to be evaluated.

* Understand inertia and silhouette score.

* Interpret cluster characteristics.

* Explain why cluster labels are arbitrary.

* Describe how feature selection and scaling affect clustering.

* Identify practical limitations of K-Means and hierarchical clustering.

* Reflect on whether discovered clusters are meaningful.

## 43. Choosing the Number of Clusters

The appropriate number of clusters is often not known in advance.

For K-Means, the analyst must choose a value of K.

For hierarchical clustering, the analyst can select a number of final clusters or examine a dendrogram to understand possible groupings.

Possible sources of information include:

* Domain knowledge.

* Visual inspection.

* Elbow method.

* Silhouette score.

* Comparison of different clusterings.

* Cluster interpretability.

### Important idea

> A clustering algorithm can produce groups, but the analyst must decide whether those groups are useful and meaningful.

## 44. Inertia and Its Limitations

Recall that K-Means attempts to minimise within-cluster squared distance.

This is closely related to inertia:

Python

Run

```
kmeans.inertia_
```

A lower inertia means that observations are, overall, closer to their assigned centroids.

However, lower inertia alone does not identify the best K.

### Why?

Increasing K generally gives the algorithm more centroids.

This allows observations to be represented more closely.

Therefore:

```
More clusters
      ↓
Usually lower inertia
```

But this does not necessarily mean:

```
More clusters
      ↓
More meaningful groups
```

A very large K could create many small groups without providing useful insight.

## 45. Silhouette Score

The silhouette score provides another way to evaluate clustering.

It considers:

1. How well an observation fits its own cluster.

2. How separated it is from neighbouring clusters.

The score ranges from:

−1-1−1

to:

111

### Interpretation

|
Silhouette value

|

General interpretation

|
| --- | --- |
|

Close to 1

|

Observation is well matched to its cluster and separated from nearby clusters

|
|

Around 0

|

Observation may be near a cluster boundary

|
|

Negative

|

Observation may be assigned to an unsuitable cluster

|

A higher silhouette score can indicate better separation under the metric used, but it is not a guarantee that the clusters are meaningful for the real-world problem.

### Python example

Python

Run

```
from sklearn.metrics import silhouette_score

score = silhouette_score(X, labels)

print(score)
```

### Important distinction

The silhouette score evaluates a particular clustering according to its distance-based structure.

It does not automatically establish that the discovered groups correspond to meaningful real-world categories.

## 46. Comparing Different Values of K

A practical experiment is to calculate inertia and silhouette score for several values of K.

Python

Run

```
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

for k in [2, 3, 4, 5]:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init="auto"
    )

    labels = kmeans.fit_predict(X)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(X, labels)

    print("K =", k)
    print("Inertia =", inertia)
    print("Silhouette =", silhouette)
    print()
```

### Questions

1. Does inertia decrease as K increases?

2. Which values of K produce relatively high silhouette scores?

3. Is there a clear elbow?

4. Do the clusters appear meaningful when visualised?

5. Does the result support one particular K, or is the choice uncertain?

## 47. Interpreting Cluster Results

Suppose a clustering algorithm produces:

```
Cluster 0
Cluster 1
Cluster 2
```

The labels alone provide little interpretation.

We need to examine the characteristics of each cluster.

For example:

|
Cluster

|

Average Age

|

Average Income

|

Average Spending

|
| --- | --- | --- | --- |
|

0

|

23

|

25,000

|

20

|
|

1

|

42

|

55,000

|

70

|
|

2

|

60

|

85,000

|

25

|

This allows us to describe the groups.

### Possible descriptions

```
Cluster 0:
Younger, lower-income, lower-spending customers.

Cluster 1:
Middle-aged, medium-income, higher-spending customers.

Cluster 2:
Older, higher-income, lower-spending customers.
```

These descriptions are interpretations based on the cluster characteristics.

They are not automatically target labels.

## 48. Cluster Labels vs Meaningful Categories

A cluster label such as:

```
Cluster 1
```

does not mean:

```
Premium Customer
```

unless the data analysis gives that interpretation.

The algorithm produces:

```
Cluster 1
```

The analyst examines the data and may later describe it as:

```
High-income, high-spending customers
```

This distinction is important.

### Key principle

> Cluster labels identify groups; interpretation gives those groups meaning.

Cluster numbers can change between different runs without changing the underlying grouping.

## 49. Cluster Size

Cluster size is another useful aspect of interpretation.

Suppose a clustering algorithm produces:

```
Cluster 0 → 100 observations
Cluster 1 → 25 observations
Cluster 2 → 5 observations
```

This tells us that the groups are not equally sized.

The analyst should ask:

* Is the small cluster meaningful?

* Does it represent a special subgroup?

* Could it contain unusual observations?

* Is the imbalance caused by the algorithm or the underlying data?

Cluster size alone does not determine whether a cluster is useful.

It is one part of the interpretation.

### Python example

Python

Run

```
import numpy as np

unique_labels, counts = np.unique(
    labels,
    return_counts=True
)

for cluster, count in zip(unique_labels, counts):
    print("Cluster", cluster, ":", count, "observations")
```

## 50. Clustering Depends on the Features

Suppose a dataset contains:

```
Age
Income
Spending
```

Clustering with:

```
Age + Income
```

may produce one grouping.

Clustering with:

```
Income + Spending
```

may produce another.

Why?

Because the features define the space in which similarity is measured.

Changing the features changes:

distance\text{distance}distance

and therefore can change the clusters.

### Example

Consider two models:

```
Model A:
Age + Income

Model B:
Income + Spending
```

If the models produce different clusters, this is not surprising.

The observations have different coordinates in each feature space.

Therefore, the distance relationships change.

### Important idea

> Changing the selected features can change the structure discovered by clustering.

## 51. Feature Scale and Practical Clustering

Feature scale is particularly important for distance-based methods.

Suppose:

```
Age:
20–80

Income:
20,000–200,000
```

Income has much larger numerical values.

Without scaling, income differences can dominate Euclidean distance.

This may cause clustering to be driven primarily by income rather than age.

### Standardisation

A common scaling method is:

z=x−μσz=\frac{x-\mu}{\sigma}z=σx−μ

where:

* xxx is the original feature value.

* μ\muμ is the feature mean.

* σ\sigmaσ is the feature standard deviation.

### Python example

Python

Run

```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

The scaled data can then be used for clustering:

Python

Run

```
kmeans.fit(X_scaled)
```

### Important note

Scaling is not always required in every clustering problem.

The appropriate choice depends on:

* The units of the features.

* The meaning of the variables.

* The distance measure.

* The purpose of the analysis.

## 52. Limitations of K-Means

K-Means is useful, but several limitations should be remembered.

### Number of clusters

K must be selected.

The algorithm does not automatically know the objectively correct number of groups.

### Feature scale

Different units can distort distances.

### Cluster shape

K-Means is particularly natural for compact groups around centres.

It may not represent every possible cluster shape well.

### Outliers

Extreme observations can influence centroids.

### Initialization

Different initial centroids can produce different results.

This is one reason multiple initialisation runs are used.

### Interpretation

The algorithm produces groups but does not automatically tell us why those groups are meaningful.

### Feature selection

Changing the features can change the clustering result.

## 53. Limitations of Hierarchical Clustering

Hierarchical clustering also has limitations.

For example:

* Different linkage methods can produce different results.

* Distance choices matter.

* Large datasets can be computationally demanding.

* The resulting hierarchy can be difficult to interpret.

* Clusters may not correspond to meaningful real-world categories.

* Early merging decisions can influence the resulting hierarchy.

### Important idea

> Hierarchical clustering produces a structured representation of relationships, but the hierarchy still depends on the selected features, distance measure, and linkage method.

## 54. Unsupervised Learning Can Still Be Evaluated

Although clustering does not use target labels during training, the resulting clustering can still be evaluated.

Possible approaches include:

* Inertia.

* Silhouette score.

* Visualisation.

* Cluster size.

* Domain knowledge.

* Comparison of different clusterings.

* Stability across different settings.

The evaluation question is different from supervised learning.

### Supervised learning asks:

> How accurately did the model predict the known target?

### Clustering asks:

> How coherent and useful is the structure that was discovered?

## 55. Supervised and Unsupervised Evaluation

### Supervised learning

There is a known answer.

For example:

```
Actual MPG
Predicted MPG
```

The prediction error can be calculated directly.

### Unsupervised learning

There is no target used during clustering.

The evaluation therefore often focuses on:

```
Cohesion
Separation
Compactness
Stability
Interpretability
```

This is one reason unsupervised learning can be more difficult to evaluate objectively.

### Comparison

|
Concept

|

Supervised

|

Unsupervised

|
| --- | --- | --- |
|

Target used for training

|

Yes

|

No

|
|

Main objective

|

Predict target

|

Discover structure

|
|

Example

|

Regression

|

Clustering

|
|

Output

|

Prediction

|

Cluster/group structure

|
|

Evaluation

|

Compare prediction with known target

|

Evaluate structure

|
|

Example metric

|

RMSE

|

Silhouette score

|

## 56. Clustering Does Not Automatically Find "True" Groups

This is an important limitation.

Suppose:

Python

Run

```
KMeans(n_clusters=3)
```

is applied to a dataset.

The algorithm will produce three clusters.

This does not prove that:

> The data naturally contain three objectively correct groups.

The algorithm has been instructed to create three clusters.

The analyst must decide whether those clusters are useful and meaningful.

### Example

A customer dataset might produce:

```
Cluster 0
Cluster 1
Cluster 2
```

These clusters could be mathematically coherent but not useful for a business decision.

Alternatively, they could reveal meaningful customer segments.

The algorithm alone cannot establish the practical significance.

## 57. Broader View of Unsupervised Learning

Clustering is only one part of unsupervised learning.

The major categories introduced in this course can be summarised as:

```
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

### Clustering

> What groups exist in the data?

Example:

```
Customer data
      ↓
Customer segments
```

### Anomaly detection

> Which observations appear unusual?

Example:

```
Transaction data
      ↓
Unusual transactions
```

A detected anomaly is not automatically an error or fraudulent event.

It simply means that the observation is unusual according to the model.

### Association rule learning

> Which items frequently occur together?

Example:

```
Bread → Butter
```

This represents an association.

It does not establish causation.

### Dimensionality reduction

> Can the data be represented using fewer dimensions?

For example:

```
100 features
      ↓
PCA
      ↓
2 dimensions
```

This can make high-dimensional data easier to visualise or analyse.

### Density estimation and generative modeling

At a high level, density estimation and generative modeling attempt to learn the underlying structure or distribution of data.

Examples include:

* Gaussian Mixture Models.

* Variational Autoencoders.

* Generative Adversarial Networks.

These topics require additional concepts beyond the introductory clustering material.

## 58. A Practical Mental Model

When encountering an unsupervised-learning problem, ask:

### Question 1 — What kind of data are available?

Are the observations described using numerical features, categorical features, or another representation?

### Question 2 — What question needs to be answered?

Is the objective to:

```
Find groups?
```

or:

```
Find unusual observations?
```

or:

```
Find co-occurring items?
```

or:

```
Reduce dimensions?
```

### Question 3 — What similarity or distance measure is appropriate?

For example:

* Euclidean distance.

* Another numerical distance.

* A similarity measure appropriate for the data.

### Question 4 — What algorithm should be used?

For clustering, possible choices include:

* K-Means.

* Hierarchical clustering.

* Other clustering methods.

### Question 5 — How will the result be evaluated?

Consider:

* Compactness.

* Separation.

* Stability.

* Interpretability.

* Domain knowledge.

### Question 6 — Are the discovered groups useful?

A clustering result should be examined rather than accepted automatically.

## 59. Final Reflection

Clustering is an exploratory technique.

It can reveal patterns in data, but the result depends on the modeling choices.

For example:

```
Features
   ↓
Distance
   ↓
Algorithm
   ↓
Parameters
   ↓
Clusters
```

Changing any of these can affect the result.

### Reflection questions

1. What did K-Means discover in the dataset?

2. What did hierarchical clustering discover?

3. Did both algorithms produce similar groups?

4. How did changing K affect the result?

5. How did changing the features affect the result?

6. How did feature scaling affect the result?

7. Are the clusters meaningful for the problem?

8. What limitations should be considered?

## Module 4 — Key ideas

* Clustering results should be evaluated and interpreted.

* Inertia measures within-cluster squared distance.

* Silhouette score provides information about cohesion and separation.

* Lower inertia alone does not identify the best K.

* Cluster labels are identifiers, not meaningful categories.

* Cluster characteristics help explain the groups.

* Feature selection and scaling influence distance-based clustering.

* K-Means and hierarchical clustering have different assumptions and limitations.

* Clustering does not automatically discover objectively correct real-world groups.

# Final Review Questions

## Question 1 — Supervised vs unsupervised learning

What is the main difference between supervised and unsupervised learning?

## Question 2 — Clustering

What is clustering?

## Question 3 — Euclidean distance

Calculate the Euclidean distance between:

A=(1,2)A=(1,2)A=(1,2)

and:

B=(4,6)B=(4,6)B=(4,6)

## Question 4 — K-Means

What does K mean in K-Means?

## Question 5 — Centroids

What is a centroid?

## Question 6 — Cluster labels

Why are cluster labels such as 0, 1, and 2 arbitrary?

## Question 7 — Inertia

Why does increasing K usually reduce inertia?

## Question 8 — Choosing K

Why can we not simply choose the largest K?

## Question 9 — Elbow method

What is the elbow method?

## Question 10 — Hierarchical clustering

What is hierarchical clustering?

## Question 11 — Dendrograms

What is a dendrogram?

## Question 12 — Linkage

What is linkage in hierarchical clustering?

## Question 13 — Feature scaling

Why can scaling affect clustering?

## Question 14 — Feature representation

Why can changing the features change the clustering result?

## Question 15 — Interpretation

Does a clustering algorithm automatically discover objectively correct groups?

# Final Summary

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

Data→Similarity→Clusters\boxed{ \text{Data} \rightarrow \text{Similarity} \rightarrow \text{Clusters} }Data→Similarity→Clusters

For K-Means:

```
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

```
Individual observations
     ↓
Merge similar groups
     ↓
Hierarchy
     ↓
Dendrogram
```

The key distinction is:

Clustering output≠Automatically meaningful categories\boxed{ \text{Clustering output} \neq \text{Automatically meaningful categories} }Clustering output=Automatically meaningful categories

The clustering result is a model of structure in the data.

Its usefulness depends on whether that structure is:

* Coherent.

* Stable.

* Interpretable.

* Relevant to the problem being investigated.

The main lesson: Clustering is not simply about running an algorithm and obtaining groups. It is about understanding how the data representation, similarity measure, algorithm, and modeling choices influence the structure that is discovered.

## Background resources

The following resources can support further study:

### Scikit-learn documentation

* KMeans — scikit-learn 

* AgglomerativeClustering — scikit-learn 

* silhouette_score — scikit-learn 

### Additional learning resources

* Unsupervised Learning — GeeksforGeeks 

* Clustering in Machine Learning — GeeksforGeeks 

* Interactive K-Means 

* Interactive Hierarchical Clustering
