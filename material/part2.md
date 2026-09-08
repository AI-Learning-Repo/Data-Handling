# Theory — Unsupervised Learning: Clustering, Anomaly Detection, and Association Rules

## 1. From Activity 1 to Activity 2

Activity 1 introduced the basic idea of unsupervised learning through clustering.

The main idea was:

> **Find structure in data without using a target variable to guide the learning process.**

Activity 2 extends this idea by asking more practical questions:

* How do we decide how many clusters to create?
* What happens when features have very different scales?
* How can we measure whether a clustering result is reasonable?
* How can hierarchical clustering help us understand different levels of grouping?
* How can unsupervised learning identify unusual observations?
* How can unsupervised learning discover relationships between items?

These questions lead to several important unsupervised learning techniques.

---

# 2. Choosing the Number of Clusters

## 2.1 Why does `K` matter?

K-Means requires the number of clusters to be specified in advance.

For example:

```python
KMeans(n_clusters=3)
```

means that the algorithm will divide the observations into three clusters.

The choice of `K` is important because different values can produce very different results.

For example, suppose customer data contains two natural groups:

* low-spending customers
* high-spending customers

Using:

```python
KMeans(n_clusters=2)
```

may represent the structure reasonably well.

However, using:

```python
KMeans(n_clusters=8)
```

forces the algorithm to divide the same data into eight groups, even if those groups are not naturally meaningful.

This illustrates an important principle:

> **A clustering algorithm can produce clusters even when the chosen number of clusters does not represent the underlying structure well.**

---

## 2.2 The Elbow Method

One common way to investigate possible values of `K` is the **elbow method**.

The idea is to fit K-Means for several values of `K` and examine the **inertia**.

For example:

```python
inertias = []

for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init="auto"
    )
    model.fit(X)

    inertias.append(model.inertia_)
```

The inertia values can then be plotted.

### What is inertia?

Inertia measures how closely observations are grouped around their cluster centers.

Conceptually:

> **Lower inertia means that observations are, on average, closer to their assigned centroids.**

A simplified representation is:

$$
\text{Inertia}
=
\sum_{i=1}^{n}
\left\|x_i-c_{z_i}\right\|^2
$$

where:

* \(x_i\) is an observation,
* \(c_{z_i}\) is the centroid of the cluster assigned to the observation,
* the squared distance measures how far the observation is from that centroid.

Increasing `K` generally decreases inertia.

Why?

Because more clusters give the algorithm more centroids to represent the observations.

For example:

```text
K = 2  → larger groups → higher inertia
K = 3  → more flexibility → lower inertia
K = 4  → lower inertia
K = 5  → lower inertia
...
```

Therefore, simply choosing the value with the smallest inertia is not useful.

The goal is to look for a point where the reduction in inertia begins to slow substantially.

This point is often called the **elbow**.

### Example

Suppose the results are:

| K | Inertia |
| - | ------: |
| 1 |    1200 |
| 2 |     600 |
| 3 |     350 |
| 4 |     300 |
| 5 |     270 |
| 6 |     250 |

The improvement from:

```text
K = 1 → 2
```

is large.

The improvement from:

```text
K = 2 → 3
```

is also substantial.

After that, the improvement becomes smaller.

A reasonable choice may therefore be around `K = 3`.

However, the elbow method is not a mathematical guarantee that the selected value is correct.

---

### Question

Why does inertia normally decrease as `K` increases?

#### LLM Hint Prompt

```text
Explain why K-Means inertia generally decreases when the number
of clusters increases. Use a simple example involving points and
centroids rather than advanced mathematics.
```

<details>
<summary>Show answer</summary>

When `K` increases, K-Means has more centroids available to represent the data.

With more centroids, observations can usually be assigned to a closer centroid, reducing the total squared distance between observations and their assigned centroids.

Therefore, inertia normally decreases as `K` increases.

However, a lower inertia does not automatically mean that the clustering is more meaningful, because increasing `K` can simply create more and smaller clusters.

</details>

---

# 3. Silhouette Score

The elbow method focuses on how compact the clusters are.

Another useful measure is the **silhouette score**.

The silhouette score considers both:

1. how close an observation is to observations in its own cluster;
2. how far that observation is from observations in other clusters.

The score ranges approximately from:

$$
-1 \text{ to } 1
$$

Interpretation:

| Silhouette Score | General Interpretation                     |
| ---------------- | ------------------------------------------ |
| Close to 1       | Observation is well matched to its cluster |
| Around 0         | Observation lies near a cluster boundary   |
| Below 0          | Observation may fit another cluster better |

A higher average silhouette score generally indicates better-separated and more cohesive clusters.

Example:

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, labels)

print(score)
```

The score can also be calculated for several values of `K`.

```python
scores = []

for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init="auto"
    )

    labels = model.fit_predict(X)

    score = silhouette_score(X, labels)

    scores.append(score)
```

This allows different candidate values of `K` to be compared.

---

## 3.1 Elbow vs Silhouette

The two approaches answer slightly different questions.

### Inertia

Asks approximately:

> How compact are the clusters?

### Silhouette score

Asks approximately:

> How cohesive are the clusters internally, and how well separated are they from other clusters?

Because they measure different properties, it is useful to consider both.

A practical clustering analysis might therefore examine:

```text
candidate K values
        ↓
inertia / elbow
        ↓
silhouette scores
        ↓
visual inspection
        ↓
domain interpretation
```

The final choice should not depend on a single metric alone.

---

### Question

Suppose `K = 3` has a slightly higher silhouette score than `K = 4`, but `K = 4` produces groups that are much more useful for the application. Which value should automatically be selected?

#### LLM Hint Prompt

```text
Explain whether clustering metrics should always determine the final
choice of K. Include the role of domain knowledge and interpretation.
```

<details>
<summary>Show answer</summary>

No.

Clustering metrics provide quantitative evidence, but they do not automatically determine whether a clustering solution is useful.

A clustering solution should also be examined through visualization, domain knowledge, cluster sizes, and the meaning of the resulting groups.

A slightly lower metric value can sometimes correspond to a more useful or interpretable solution.

</details>

---

# 4. Why Feature Scaling Matters

Many clustering algorithms rely on distances.

This creates an important issue when features have very different numerical scales.

Consider two features:

```text
Age:             18–80
Annual Income:   20,000–200,000
```

Suppose distance is calculated directly.

A difference of:

```text
Age = 10
```

is numerically much smaller than:

```text
Income = 50,000
```

The income feature may therefore dominate the distance calculation.

This means that clustering may be driven primarily by income rather than by the combination of both features.

---

## 4.1 Standardization

One common solution is **standardization**.

A standardization transformation is commonly written as:

$$
z =
\frac{x-\mu}{\sigma}
$$

where:

* \(x\) is the original value,
* \(\mu\) is the feature mean,
* \(\sigma\) is the feature standard deviation.

In scikit-learn:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

The transformed features are placed on a more comparable scale.

---

## 4.2 Why scaling can change clusters

Consider:

```text
Feature A: 1–10
Feature B: 1–100000
```

Without scaling, Feature B can dominate Euclidean distance.

After scaling, both features contribute more comparably.

Therefore:

> **Scaling can change the geometry of the feature space, which can change the clustering result.**

This is particularly important for K-Means and other distance-based methods.

---

### Question

Why can applying `StandardScaler` change the K-Means clusters?

#### LLM Hint Prompt

```text
Explain how feature scaling changes distances between observations
and why that can cause K-Means to assign observations to different
clusters.
```

<details>
<summary>Show answer</summary>

K-Means uses distances when assigning observations to centroids.

Scaling changes the numerical contribution of each feature to those distances.

Therefore, the nearest centroid for some observations can change after scaling, which can lead to different cluster assignments and different centroids.

</details>

---

# 5. Hierarchical Clustering in More Detail

Activity 1 introduced hierarchical clustering as an alternative to K-Means.

A common form is **agglomerative hierarchical clustering**.

The process begins with:

```text
Each observation = its own cluster
```

The algorithm then repeatedly merges the most appropriate clusters.

Conceptually:

```text
Individual observations
        ↓
Small groups
        ↓
Larger groups
        ↓
Even larger groups
        ↓
One large hierarchy
```

This produces a hierarchy of cluster relationships.

---

# 6. Linkage Methods

When hierarchical clustering decides which clusters should be merged, it needs a method for measuring the distance between clusters.

This is called **linkage**.

Different linkage methods define cluster-to-cluster distance differently.

Common approaches include:

### Single linkage

Measures the distance between the closest pair of observations in the two clusters.

Conceptually:

```text
Cluster A ●────● Cluster B
           ↑
     closest pair
```

This can produce elongated or chain-like clusters.

---

### Complete linkage

Measures the distance between the farthest pair of observations in the two clusters.

This tends to favor more compact groups.

---

### Average linkage

Uses the average distance between observations in the two clusters.

This provides a compromise between some characteristics of single and complete linkage.

---

### Ward linkage

Ward linkage attempts to merge clusters while minimizing the increase in within-cluster variance.

It is often used with numeric data where compact clusters are appropriate.

Example:

```python
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

labels = model.fit_predict(X)
```

---

# 7. Dendrograms

A major advantage of hierarchical clustering is that the hierarchy can be visualized using a **dendrogram**.

A dendrogram shows:

* which observations or groups are merged;
* the order in which merges occur;
* the distance or linkage value at which the merges occur.

Conceptually:

```text
          ┌───────────────┐
          │               │
      ┌───┴───┐       ┌───┴───┐
      │       │       │       │
      A       B       C       D
```

A horizontal cut through a dendrogram can produce a chosen number of clusters.

This creates an important difference from K-Means:

> K-Means begins with a chosen number of clusters, while hierarchical clustering can reveal a hierarchy of possible groupings.

The hierarchy can therefore help investigate different numbers of clusters.

---

## 7.1 Why the dendrogram is useful

Imagine that four observations form a very close group and two other observations form another group.

The dendrogram may visually reveal this structure before a final number of clusters is selected.

This can be useful when the appropriate value of `K` is not obvious.

---

### Question

Why can a dendrogram provide information that a basic K-Means result does not?

#### LLM Hint Prompt

```text
Compare K-Means with hierarchical clustering and explain why a
dendrogram can reveal multiple levels of grouping rather than only
one final partition.
```

<details>
<summary>Show answer</summary>

A standard K-Means model returns a partition based on the specified number of clusters.

A dendrogram represents a hierarchy of merges. It shows how small groups are progressively combined into larger groups.

Therefore, different levels of the hierarchy can be examined instead of seeing only one final partition.

</details>

---

# 8. Evaluating Clustering Results

Unsupervised learning creates a special evaluation problem.

In supervised learning, predictions can often be compared directly against known target values.

For example:

```text
Predicted price
        vs.
Actual price
```

In clustering, there may be no true cluster labels.

The algorithm creates the groups itself.

Therefore, evaluation often combines several forms of evidence.

## 8.1 Internal evaluation

Metrics calculated using the data and cluster assignments.

Examples:

* inertia;
* silhouette score.

## 8.2 Visual evaluation

Plots can reveal:

* separated groups;
* overlapping groups;
* unusual points;
* elongated structures;
* clusters of very different sizes.

## 8.3 Domain evaluation

The groups should make sense for the application.

For example, suppose customer clustering creates:

```text
Cluster 0 → low spending, low activity
Cluster 1 → high spending, high activity
Cluster 2 → high spending, low activity
```

These groups may be useful because their characteristics can support interpretation or decision-making.

---

# 9. Cluster Profiles

Cluster labels such as:

```text
Cluster 0
Cluster 1
Cluster 2
```

do not automatically have a meaningful interpretation.

The numerical labels are simply identifiers.

To interpret the clusters, it is useful to calculate summaries for each group.

For example:

```python
df.groupby("cluster")[["age", "income", "spending"]].mean()
```

This might produce:

| Cluster | Age | Income | Spending |
| ------- | --: | -----: | -------: |
| 0       |  24 | 32,000 |       18 |
| 1       |  46 | 78,000 |       72 |
| 2       |  63 | 51,000 |       35 |

Possible interpretation:

```text
Cluster 0 → younger, lower-income, lower-spending group
Cluster 1 → higher-income, higher-spending group
Cluster 2 → older, moderate-income, moderate-spending group
```

The interpretation comes from the characteristics of the observations, not from the cluster number itself.

---

# 10. Anomaly Detection

Clustering is not the only type of unsupervised learning.

Another important task is **anomaly detection**.

An anomaly is an observation that is unusually different from the general pattern of the data.

Examples include:

* an unusually large financial transaction;
* unusual login behavior;
* abnormal sensor readings;
* unexpected network activity;
* unusual purchasing behavior.

The objective is not necessarily to create groups.

Instead, the goal is:

> **Identify observations that appear unusual compared with the normal pattern.**

---

# 11. Isolation Forest

One common anomaly detection method is **Isolation Forest**.

The main idea is different from K-Means.

K-Means asks:

> Which cluster should this observation belong to?

Isolation Forest asks approximately:

> How easily can this observation be separated from the other observations?

Unusual observations are often easier to isolate because they are located in sparse or unusual regions of the feature space.

Example:

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

predictions = model.fit_predict(X)
```

The output commonly uses:

```text
1  → inlier / normal observation
-1 → anomaly
```

The exact anomaly detection workflow depends on the data and application.

---

## 11.1 Anomaly Detection Is Not the Same as Removing Outliers

An important distinction is:

> **An unusual observation is not automatically an error.**

For example, suppose transaction data contains:

```text
€20
€35
€42
€50
€4800
```

The €4800 transaction is unusual.

It could be:

* a data-entry error;
* a legitimate large purchase;
* fraudulent activity.

An anomaly detection model identifies unusual behavior, but interpretation still requires context.

---

### Question

Why should an anomaly not automatically be deleted from a dataset?

#### LLM Hint Prompt

```text
Explain the difference between a statistical anomaly and a data error.
Give an example where an unusual observation could be valid.
```

<details>
<summary>Show answer</summary>

An anomaly is unusual relative to the rest of the data, but unusual does not necessarily mean incorrect.

For example, a very large transaction may be unusual but completely legitimate.

Deleting anomalies without understanding their meaning can remove important information.

</details>

---

# 12. Association Rule Learning

Association rule learning focuses on relationships among items or events.

It is particularly common with **transactional data**.

A classic example is market-basket analysis.

Suppose transactions look like:

```text
Transaction 1 → Bread, Milk
Transaction 2 → Bread, Butter
Transaction 3 → Bread, Butter, Milk
Transaction 4 → Milk, Eggs
```

The goal may be to discover patterns such as:

```text
Bread → Butter
```

This does not necessarily mean that buying bread causes someone to buy butter.

Instead, it indicates an observed association in the data.

---

# 13. Frequent Itemsets

Association rule learning usually begins by identifying **frequent itemsets**.

An itemset is a collection of items that occurs together.

Examples:

```text
{Bread}
{Milk}
{Bread, Butter}
{Bread, Milk}
{Bread, Butter, Milk}
```

An itemset becomes interesting when it occurs frequently enough.

This leads to the concept of **support**.

---

# 14. Support

Support measures how frequently an itemset appears in the dataset.

Conceptually:

$$
Support(A)
=
\frac{\text{transactions containing }A}
{\text{total transactions}}
$$

For example, if:

```text
100 transactions
```

and:

```text
25 contain Bread and Butter
```

then:

$$
Support(Bread, Butter)=0.25
$$

or:

```text
25%
```

Support helps identify patterns that occur often enough to be worth investigating.

---

# 15. Confidence

Confidence measures how often the consequent appears when the antecedent appears.

For a rule:

$$
A \rightarrow B
$$

confidence is:

$$
Confidence(A\rightarrow B)
=
\frac{Support(A\cup B)}
{Support(A)}
$$

Suppose:

```text
40 transactions contain Bread
30 of those also contain Butter
```

Then:

$$
Confidence(Bread\rightarrow Butter)
=
\frac{30}{40}
=
0.75
$$

Therefore, the confidence is:

```text
75%
```

This means that among transactions containing bread, 75% also contain butter.

---

# 16. Lift

Confidence alone can be misleading.

Suppose almost everyone in the dataset buys butter.

Then a rule such as:

```text
Bread → Butter
```

could have high confidence simply because butter is already very common.

**Lift** helps account for this.

Conceptually:

$$
Lift(A\rightarrow B)
=
\frac{Confidence(A\rightarrow B)}
{Support(B)}
$$

Interpretation is commonly:

```text
Lift > 1  → positive association
Lift ≈ 1  → little or no association
Lift < 1  → negative association
```

For example:

```text
Lift = 1.8
```

suggests that the items occur together more often than would be expected under a simple independence assumption.

Lift should therefore be considered together with support and confidence rather than interpreted alone.

---

## 16.1 Association Does Not Mean Causation

This distinction is essential.

Suppose the data reveals:

```text
Coffee → Sugar
```

This does not prove:

```text
Buying coffee causes people to buy sugar.
```

The data only indicates an association.

Other explanations may exist.

For example:

* the items may be used together;
* both may be influenced by another factor;
* the data may come from a particular population;
* the pattern may be specific to the observed transactions.

---

### Question

Why does a strong association rule not prove causation?

#### LLM Hint Prompt

```text
Explain the difference between association and causation using
a market-basket example. Explain why co-occurrence alone cannot
establish a cause-and-effect relationship.
```

<details>
<summary>Show answer</summary>

Association means that two events or items occur together more frequently than expected according to the chosen measure.

Causation means that one event directly contributes to producing another.

Association rule learning only identifies patterns of co-occurrence. It does not perform a causal experiment or prove that one item causes another.

</details>

---

# 17. Association Rules in Practice

Association rule learning is useful in contexts such as:

* retail analysis;
* product recommendation;
* website navigation;
* content recommendation;
* purchasing behavior;
* event co-occurrence analysis.

A typical workflow is:

```text
Transaction data
       ↓
Frequent itemsets
       ↓
Candidate rules
       ↓
Support filtering
       ↓
Confidence filtering
       ↓
Lift analysis
       ↓
Interpretation
```

The final rules should be examined in context.

A mathematically strong rule may not necessarily be operationally useful.

---

# 18. Comparing Major Unsupervised Learning Tasks

Several unsupervised learning methods may look similar because they all work without a target variable, but they solve different problems.

| Task                     | Main Goal                                       | Typical Output               | Example               |
| ------------------------ | ----------------------------------------------- | ---------------------------- | --------------------- |
| Clustering               | Find groups                                     | Cluster assignments          | Customer segments     |
| Anomaly Detection        | Find unusual observations                       | Anomaly / normal label       | Fraud investigation   |
| Association Rules        | Find co-occurrence patterns                     | Rules                        | Bread → Butter        |
| Dimensionality Reduction | Represent data with fewer dimensions            | Reduced features             | Visualization         |
| Density Estimation       | Model data distribution                         | Density/probability estimate | Distribution modeling |
| Generative Modeling      | Learn structure for generating new observations | Generated samples/model      | Synthetic data        |

The choice of technique should therefore begin with the question:

> **What structure are we trying to discover?**

---

# 19. Clustering vs Anomaly Detection

These two tasks are particularly easy to confuse.

### Clustering

Focuses on:

```text
Which observations belong together?
```

### Anomaly detection

Focuses on:

```text
Which observations do not fit the normal pattern?
```

For example:

Suppose most transactions form several groups based on amount and frequency.

Clustering may identify:

```text
Cluster A → small frequent purchases
Cluster B → medium purchases
Cluster C → large regular purchases
```

Anomaly detection may instead identify:

```text
Transaction X → unusually different from normal behavior
```

The same dataset can therefore be used for different unsupervised tasks depending on the question.

---

# 20. Clustering vs Association Rules

Clustering works primarily with observations represented by features in a feature space.

Association rules often work with transaction-style data such as:

```text
Customer 1 → Bread, Milk
Customer 2 → Bread, Butter
Customer 3 → Milk, Eggs
```

The questions are different.

Clustering asks:

> Which observations are similar?

Association rule learning asks:

> Which items tend to occur together?

For example:

```text
Clustering:
Customer A ≈ Customer B

Association:
Bread ↔ Butter
```

---

# 21. Dimensionality Reduction

Dimensionality reduction is another unsupervised learning task.

The purpose is to represent data using fewer dimensions while attempting to preserve important structure.

Suppose a dataset contains:

```text
100 features
```

A dimensionality reduction method may create:

```text
2 or 3 transformed dimensions
```

These lower-dimensional representations can be useful for:

* visualization;
* simplifying data;
* noise reduction;
* preprocessing;
* exploring structure.

A common technique is **Principal Component Analysis (PCA)**.

The mathematical details of PCA involve concepts such as variance, covariance, and orthogonal directions, but the main conceptual idea is:

> **Find a lower-dimensional representation that retains as much important variation in the data as possible.**

Dimensionality reduction is particularly useful when datasets contain many features and cannot be visualized directly.

---

# 22. Density Estimation and Generative Modeling

Another group of unsupervised approaches attempts to learn the underlying structure or distribution of the data.

### Density estimation

Attempts to estimate how likely different regions of the feature space are.

This can help answer questions such as:

> Where are observations concentrated?

### Generative modeling

Attempts to learn enough structure from the data to generate new observations that resemble the original data distribution.

Examples of broader generative modeling approaches include:

* Gaussian Mixture Models;
* Variational Autoencoders;
* Generative Adversarial Networks.

The important distinction is that generative modeling goes beyond simply identifying clusters or anomalies.

The objective includes learning a representation or distribution that can support generation of new samples.

---

# 23. Why Unsupervised Learning Requires Interpretation

One of the major characteristics of unsupervised learning is that there may be no simple "correct answer."

Suppose K-Means produces:

```text
Cluster 0
Cluster 1
Cluster 2
```

The algorithm has completed its optimization objective, but this does not automatically mean that the clusters correspond to meaningful real-world categories.

Interpretation may require:

* examining feature distributions;
* visualizing the clusters;
* comparing cluster sizes;
* inspecting cluster profiles;
* checking stability;
* using domain knowledge.

This creates a useful distinction:

> **A mathematically valid clustering is not necessarily a practically useful clustering.**

---

# 24. Important Limitations

The techniques used in Activity 2 have several limitations.

## K-Means

K-Means can be sensitive to:

* the chosen `K`;
* feature scaling;
* initialization;
* outliers;
* cluster geometry.

It generally works best when groups are reasonably compact and represented well by their centroids.

---

## Hierarchical Clustering

Hierarchical clustering can be affected by:

* linkage method;
* distance measure;
* scaling;
* noisy observations.

Different linkage methods may produce different hierarchies.

---

## Anomaly Detection

Anomaly detection can produce false positives.

An observation may be unusual without being erroneous or fraudulent.

The proportion of observations expected to be anomalous can also affect the modeling process.

---

## Association Rules

Association rules can produce a very large number of patterns.

Not every statistically interesting rule is useful.

Rules therefore need to be filtered and interpreted using measures such as:

* support;
* confidence;
* lift;
* domain relevance.

---

# 25. A General Unsupervised Learning Workflow

A useful general workflow is:

```text
Understand the problem
        ↓
Identify the structure to discover
        ↓
Inspect and clean the data
        ↓
Select relevant features
        ↓
Consider feature scaling
        ↓
Choose an unsupervised method
        ↓
Fit the model
        ↓
Evaluate the discovered structure
        ↓
Visualize and interpret
        ↓
Check whether the result is useful
```

Different tasks then use different techniques.

For clustering:

```text
Data
 ↓
Scaling
 ↓
K selection
 ↓
Clustering
 ↓
Metrics
 ↓
Visualization
 ↓
Cluster interpretation
```

For anomaly detection:

```text
Data
 ↓
Feature preparation
 ↓
Anomaly model
 ↓
Anomaly scores / labels
 ↓
Inspection of unusual observations
```

For association rules:

```text
Transactions
 ↓
Frequent itemsets
 ↓
Rules
 ↓
Support
 ↓
Confidence
 ↓
Lift
 ↓
Interpretation
```

---

# 26. Key Concepts to Remember

The main concepts from Activity 2 can be summarized as follows:

### K-Means

Divides observations into `K` clusters using distance to centroids.

### Inertia

Measures the total within-cluster squared distance.

Lower is more compact, but lower inertia alone does not determine the best `K`.

### Elbow Method

Examines how inertia decreases as `K` increases and looks for a point where improvements begin to diminish.

### Silhouette Score

Measures how well observations fit within their clusters relative to other clusters.

Higher values generally indicate better-defined clustering.

### Feature Scaling

Places features on comparable scales so that large numerical units do not dominate distance calculations.

### Hierarchical Clustering

Builds a hierarchy of clusters through repeated merging.

### Dendrogram

Visualizes the hierarchy and can help investigate different possible numbers of clusters.

### Anomaly Detection

Identifies observations that are unusually different from the normal pattern.

### Isolation Forest

Uses repeated random partitioning to identify observations that are easier to isolate.

### Association Rule Learning

Finds co-occurrence relationships between items or events.

### Support

Measures how frequently an itemset occurs.

### Confidence

Measures how often the consequent occurs when the antecedent occurs.

### Lift

Measures how much more often two items occur together compared with what would be expected based on the consequent's overall frequency.

### Association vs Causation

A discovered relationship does not prove that one event causes another.

---

# 27. Questions and Answers

## Question 1

Why is choosing `K` an important part of K-Means?

<details>
<summary>Answer</summary>

K-Means creates exactly the number of clusters specified by `K`.

A value that is too small may combine genuinely different groups, while a value that is too large may split meaningful groups into smaller artificial groups.

Therefore, the selected value should be investigated using methods such as the elbow method, silhouette score, visualization, and domain knowledge.

</details>

---

## Question 2

Why is inertia alone not enough to choose `K`?

<details>
<summary>Answer</summary>

Inertia generally decreases as `K` increases.

Therefore, the smallest inertia will usually occur when many clusters are used.

This does not mean that the resulting clustering is the most meaningful.

The elbow method, silhouette score, visualization, and interpretation provide additional evidence.

</details>

---

## Question 3

Why can feature scaling affect clustering?

<details>
<summary>Answer</summary>

Distance-based methods are affected by the numerical scale of features.

A feature with much larger numerical values can dominate the distance calculation.

Scaling places features on more comparable scales and can therefore change distances, cluster assignments, and centroids.

</details>

---

## Question 4

What is the main difference between K-Means and hierarchical clustering?

<details>
<summary>Answer</summary>

K-Means directly partitions the data into a specified number of clusters using centroids.

Hierarchical clustering builds a hierarchy of clusters through repeated merging or splitting.

A dendrogram can be used to visualize this hierarchy.

</details>

---

## Question 5

What is the purpose of a dendrogram?

<details>
<summary>Answer</summary>

A dendrogram visualizes the sequence of cluster merges in hierarchical clustering.

It helps reveal relationships between observations and allows different levels of the hierarchy to be examined.

</details>

---

## Question 6

What is an anomaly?

<details>
<summary>Answer</summary>

An anomaly is an observation that is unusually different from the general pattern of the data.

An anomaly is not automatically an error. It may represent a valid but unusual event.

</details>

---

## Question 7

What is the difference between clustering and anomaly detection?

<details>
<summary>Answer</summary>

Clustering identifies groups of similar observations.

Anomaly detection identifies observations that differ substantially from the normal pattern.

Clustering focuses on group structure, while anomaly detection focuses on unusual behavior.

</details>

---

## Question 8

What does support measure in association rule learning?

<details>
<summary>Answer</summary>

Support measures how frequently an itemset appears in the dataset.

It is the proportion of transactions that contain the specified itemset.

</details>

---

## Question 9

What does confidence measure?

<details>
<summary>Answer</summary>

Confidence measures how frequently the consequent appears among transactions that contain the antecedent.

For a rule \(A \rightarrow B\), it measures how often \(B\) occurs when \(A\) occurs.

</details>

---

## Question 10

Why is lift useful?

<details>
<summary>Answer</summary>

Lift compares the observed confidence of a rule with the overall frequency of the consequent.

It helps determine whether the association appears stronger than might be expected simply because the consequent is common.

</details>

---

## Question 11

Does a rule such as `Bread → Butter` mean that buying bread causes someone to buy butter?

<details>
<summary>Answer</summary>

No.

The rule describes an association or co-occurrence pattern.

It does not establish a cause-and-effect relationship.

</details>

---

## Question 12

Why can unsupervised learning be difficult to evaluate?

<details>
<summary>Answer</summary>

There may be no known target labels representing the correct groups.

Therefore, evaluation often combines quantitative metrics, visualization, data inspection, and domain interpretation rather than relying on a single prediction error measure.

</details>

---

# 28. Final Knowledge Check

Use the following questions to verify understanding.

### Conceptual

1. What is the purpose of the `K` parameter in K-Means?
2. Why does inertia generally decrease as `K` increases?
3. What does the elbow method attempt to identify?
4. What does the silhouette score measure?
5. Why can feature scaling change clustering results?
6. What is hierarchical clustering?
7. What does a dendrogram represent?
8. What is the difference between clustering and anomaly detection?
9. What is Isolation Forest designed to detect?
10. What is an association rule?
11. What does support measure?
12. What does confidence measure?
13. What does lift measure?
14. Why does association not imply causation?
15. Why is domain interpretation important in unsupervised learning?

### Applied

16. A K-Means model has very high inertia. What information would be needed before deciding that the model is poor?
17. Two features have ranges of `0–5` and `0–1,000,000`. What problem could this create for distance-based clustering?
18. A silhouette score is close to zero. What might that indicate?
19. A point is identified as an anomaly. Why should it not automatically be removed?
20. An association rule has very high confidence but very low lift. What might this suggest?
21. A dendrogram shows several possible levels of grouping. How could this help in selecting a number of clusters?
22. A clustering result produces groups that are mathematically well separated but difficult to interpret. Should the result automatically be considered useful? Explain.
23. The same dataset produces different cluster assignments before and after scaling. Is that necessarily a problem? Explain.
24. A rule `A → B` has high confidence. What additional information should be examined before concluding that the rule is useful?
25. Why can the same dataset support different unsupervised learning tasks depending on the problem being investigated?

---

# 29. Summary

Activity 2 expands unsupervised learning from simply creating clusters to evaluating and interpreting discovered structure.

The major ideas are:

```text
K-Means
   ↓
Choosing K
   ↓
Inertia + Elbow
   ↓
Silhouette Score
   ↓
Feature Scaling
   ↓
Hierarchical Clustering
   ↓
Dendrograms
   ↓
Anomaly Detection
   ↓
Isolation Forest
   ↓
Association Rules
   ↓
Support + Confidence + Lift
```

The central idea remains the same:

> **Unsupervised learning searches for structure in data without using a target variable to guide the learning process.**

However, discovering structure is only the beginning.

A meaningful analysis must also consider:

```text
Is the structure stable?
        ↓
Is it measurable?
        ↓
Is it interpretable?
        ↓
Does it make sense in context?
        ↓
Is it useful for the intended purpose?
```

This is what turns an unsupervised learning result from a numerical output into a meaningful analysis.
