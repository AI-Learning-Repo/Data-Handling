# Theory Part 1: Unsupervised Learning and Clustering

This reading explains the concepts behind [Activity 1](activity1.md). Sections 1–6 follow the activity's organization so you can move between a practical experiment and its explanation. Section 7 provides a broader view of unsupervised learning; it is theory only and has no associated lab tasks.

The focus is understanding: what each method receives, how it forms groups, what its outputs mean, and why those outputs need interpretation. Detailed clustering metrics and model selection belong to Part 2.

## Table of contents

- [1. From prediction to discovering structure](#1-from-prediction-to-discovering-structure)
  - [1.1 Connect to classification and regression](#11-connect-to-classification-and-regression)
  - [1.2 Understand the clustering question](#12-understand-the-clustering-question)
- [2. Representing customers and measuring similarity](#2-representing-customers-and-measuring-similarity)
  - [2.1 Understand the observations and features](#21-understand-the-observations-and-features)
  - [2.2 Define the feature space](#22-define-the-feature-space)
  - [2.3 Understand Euclidean distance](#23-understand-euclidean-distance)
  - [2.4 Consider units and scale](#24-consider-units-and-scale)
- [3. Understanding and applying K-Means](#3-understanding-and-applying-k-means)
  - [3.1 Understand K and centroids](#31-understand-k-and-centroids)
  - [3.2 Follow the assignment and update cycle](#32-follow-the-assignment-and-update-cycle)
  - [3.3 Interpret fitting and model outputs](#33-interpret-fitting-and-model-outputs)
  - [3.4 Read centroids and cluster plots](#34-read-centroids-and-cluster-plots)
  - [3.5 Understand what changes with K](#35-understand-what-changes-with-k)
- [4. Building a hierarchy of clusters](#4-building-a-hierarchy-of-clusters)
  - [4.1 Understand agglomerative merging](#41-understand-agglomerative-merging)
  - [4.2 Understand linkage and compare procedures](#42-understand-linkage-and-compare-procedures)
  - [4.3 Read a dendrogram](#43-read-a-dendrogram)
- [5. Interpreting results and recognizing limitations](#5-interpreting-results-and-recognizing-limitations)
  - [5.1 Interpret cluster profiles](#51-interpret-cluster-profiles)
  - [5.2 Understand overlapping groups](#52-understand-overlapping-groups)
  - [5.3 Distinguish a partition from natural groups](#53-distinguish-a-partition-from-natural-groups)
  - [5.4 Recognize the limitations](#54-recognize-the-limitations)
- [6. Consolidation and preparation for Part 2](#6-consolidation-and-preparation-for-part-2)
  - [6.1 Compare the two methods](#61-compare-the-two-methods)
  - [6.2 Check your understanding](#62-check-your-understanding)
  - [6.3 Prepare for evaluation](#63-prepare-for-evaluation)
- [7. A broader view of unsupervised learning — theory only](#7-a-broader-view-of-unsupervised-learning--theory-only)
  - [7.1 Match the task to the question](#71-match-the-task-to-the-question)
  - [7.2 Anomaly detection](#72-anomaly-detection)
  - [7.3 Association rule learning](#73-association-rule-learning)
  - [7.4 Dimensionality reduction](#74-dimensionality-reduction)
  - [7.5 Density estimation and generative modeling](#75-density-estimation-and-generative-modeling)
- [References](#references)

## 1. From prediction to discovering structure

### 1.1 Connect to classification and regression

A **feature** is an input variable describing an observation. A **target** is the value a supervised model is trained to predict.

In classification, the target is a category, such as a flower species. In regression, it is a numerical value, such as MPG. Both learn from examples containing features and known targets.

| Learning setting | Training information | Purpose |
|---|---|---|
| Supervised | Features X and known targets y | Learn to predict a target |
| Unsupervised | Features X without a target guiding learning | Discover structure in the features |

The distinction is how the learning process uses information. A dataset may contain known labels while a clustering experiment deliberately excludes them. The fitting process remains unsupervised if those labels do not guide it.

### 1.2 Understand the clustering question

Clustering asks: **Which observations belong together according to a chosen definition of similarity?**

In customer segmentation, the observations are customers. Features such as income and spending describe them. The model creates group assignments without receiving a correct customer-group target.

Classification instead starts from categories defined in the training labels and learns to predict them. Although both procedures may return numbers, those numbers have different roles. A class prediction refers to an existing target category; a cluster number identifies a group created by the algorithm.

Clustering can support customer segmentation, document organization, or the exploration of biological measurements. The meaning of a group depends on the application and on the measurements supplied.

## 2. Representing customers and measuring similarity

### 2.1 Understand the observations and features

In Activity 1, a row represents a customer. We select annual income and spending score from the Mall Customers dataset.

Income is expressed in thousands of dollars. Spending score is a feature supplied by the dataset provider. It is not a target in this experiment: both variables help define customer similarity.

An identifier such as `CustomerID` usually should not be included in a distance calculation. The difference between IDs 100 and 101 does not tell us that those customers have similar behavior. Categorical variables require a suitable representation and similarity definition; leaving them out of this first example is a simplification.

Inspecting missing values and data types comes before clustering. Missing measurements and unsuitable inputs can prevent fitting or make comparisons misleading. Even dropping incomplete rows is a choice: we should know which observations were excluded.

### 2.2 Define the feature space

Each observation becomes a point whose coordinates are its feature values. Two features define a two-dimensional space; 20 features define a 20-dimensional space.

For the customer example:

- horizontal coordinate: annual income;
- vertical coordinate: spending score;
- one point: one customer.

A two-dimensional scatter plot shows both selected coordinates directly. A plot using only two features from a larger feature set shows a partial view of the relationships the model uses.

Changing features changes the space. Clustering on age and income may produce different groups from clustering on income and spending, even when the customer rows are identical. Neither result is automatically incorrect: the two representations ask different similarity questions.

### 2.3 Understand Euclidean distance

For points $A=(x_A,y_A)$ and $B=(x_B,y_B)$, Euclidean distance is:

$$
d(A,B)=\sqrt{(x_B-x_A)^2+(y_B-y_A)^2}
$$

The calculation combines the coordinate differences into a straight-line distance. For $A=(1,2)$ and $B=(4,6)$:

$$
d(A,B)=\sqrt{3^2+4^2}=\sqrt{25}=5
$$

The same principle extends to more features: square the difference for each feature, add the squared differences, and take the square root.

Distance provides a numerical definition of similarity, not a universal judgment about the observations. Two customers close in income and spending score could differ in interests, needs, or characteristics that were never measured.

**Worked check:** For $A=(2,3)$ and $B=(5,7)$, what is the distance?

<details>
<summary>Calculation</summary>

```math
d(A,B)=\sqrt{(5-2)^2+(7-3)^2}=\sqrt{9+16}=5
```

</details>

### 2.4 Consider units and scale

Euclidean distance is sensitive to numerical units. Consider two observations that differ by 5 years in age and 1,000 dollars in income. Their squared contributions to distance are 25 and 1,000,000. Income dominates this comparison because of its numerical scale.

Expressing the same income difference as 1 thousand dollars changes that contribution to 1. The people did not change, but their coordinates did.

Standardization expresses each value relative to its feature's mean and standard deviation:

$$
z=\frac{x-\mu}{\sigma}
$$

Here $\mu$ is the feature mean and $\sigma$ its standard deviation. For a nonconstant feature, standardization gives it mean zero and standard deviation one on the data used to fit the transformation.

Scaling makes numerical contributions more comparable. It does not prove that every feature deserves equal influence or guarantee meaningful clusters. Feature selection and weighting remain modeling decisions.

Activity 1 uses original units to make the first plots and centroids easy to interpret. Part 2 investigates how scaling changes the results.

## 3. Understanding and applying K-Means

### 3.1 Understand K and centroids

**K** is the number of clusters requested. A **centroid** is the mean position of the observations assigned to a cluster.

For points `(1,2)`, `(2,3)`, and `(3,4)`:

$$
\mu=\left(\frac{1+2+3}{3},\frac{2+3+4}{3}\right)=(2,3)
$$

Each coordinate is averaged separately. With 20 features, each centroid has 20 coordinates. A centroid is a calculated location and need not be one of the original observations.

The name K-Means reflects these two ideas: K groups represented by their means.

### 3.2 Follow the assignment and update cycle

K-Means alternates between two operations after initialization:

1. **Assignment:** give each observation to its nearest current centroid.
2. **Update:** replace each centroid with the mean of its newly assigned observations.

Updating centers can change which centroid is nearest to a point, so assignment is repeated. Fitting stops when a convergence criterion or iteration limit is reached.

The objective is to reduce the **sum of squared distances from observations to their assigned centroids**. This favors compact groups around centers. It does not establish that these groups are meaningful categories.

A worked assignment makes the procedure concrete. Consider point $P=(4,3)$ and three centroids:

| Centroid | Coordinates | Euclidean distance from P |
|---|---|---|
| C1 | (2, 3) | 2 |
| C2 | (8, 9) | √52 ≈ 7.21 |
| C3 | (5, 2) | √2 ≈ 1.41 |

P is assigned to **C3**, the nearest centroid. After all assignments, each group's mean is recalculated.

<details>
<summary>Check the distances</summary>

```math
\begin{aligned}
d(P,C_1)&=\sqrt{(4-2)^2+(3-3)^2}=2 \\
d(P,C_2)&=\sqrt{(4-8)^2+(3-9)^2}=\sqrt{52} \\
d(P,C_3)&=\sqrt{(4-5)^2+(3-2)^2}=\sqrt{2}
\end{aligned}
```

</details>

Different initial centroids can lead to different final partitions. Repeating the procedure from several initializations helps find a better solution, but does not guarantee the global optimum.

### 3.3 Interpret fitting and model outputs

The important connection between code and concept is:

| Lab expression | Conceptual meaning |
|---|---|
| `KMeans(n_clusters=3, random_state=42, n_init=10)` | Request three groups, set a reproducible random seed, and try ten initializations |
| `kmeans.fit(X)` | Fit the model to features without a supplied target |
| `kmeans.labels_` | One group identifier per fitted observation |
| `kmeans.cluster_centers_` | Coordinates of the learned centroids |
| `model.fit_predict(X)` | Fit the model and return group identifiers |

For three clusters and two features, the centroid array has shape `(3, 2)`.

Cluster numbers have no inherent ordering. One run could identify the left-hand group as `0`, while another calls it `2`. If exactly the same observations remain together, the grouping is unchanged despite the different names.

The [K-Means reference](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) provides implementation details. For conceptual revision, focus on what fitting receives and what the outputs represent.

### 3.4 Read centroids and cluster plots

A cluster plot colors observations by assignment. A centroid marker shows a group's mean position. The colors and numbers identify groups; they do not explain the groups by themselves.

For illustration, a centroid `(70,25)` in the lab's original units represents mean income of 70 thousand dollars and mean spending score of 25. It does not mean every group member has those values.

If a model is fitted on standardized data, its centroid coordinates are expressed in standardized units. They must be transformed back, or interpreted through summaries of the original observations, before describing them as income or spending values.

### 3.5 Understand what changes with K

Changing K changes the number of centers available to represent the observations. Both the centers and memberships may change.

K-Means fits a new partition for each K. A two-cluster solution need not be obtained simply by merging two groups from a three-cluster solution. This is a key difference from a hierarchical tree, whose groupings are nested.

A requested K is a modeling choice. The customer dataset does not provide a known generation count, so three clusters in Activity 1 is a starting experiment. Selecting a defensible value requires the evaluation ideas introduced in Part 2.

## 4. Building a hierarchy of clusters

### 4.1 Understand agglomerative merging

Hierarchical clustering represents groupings at several levels. The version used in Activity 1 is **agglomerative**: it starts with individual observations and repeatedly merges groups.

A possible sequence is:

```text
{A} {B} {C} {D} → {A,B} {C} {D} → {A,B} {C,D} → {A,B,C,D}
```

Each later grouping contains earlier groups. An agglomerative merge is not undone: observations that join stay together as we move up the hierarchy.

Hierarchical clustering can also be **divisive**, starting with one group and splitting it into smaller groups. This is background terminology here; the practical method in Activity 1 uses merging.

### 4.2 Understand linkage and compare procedures

Distance between individual observations is not enough to define which groups should merge. A **linkage criterion** defines how candidate cluster merges are compared.

The activity uses **Ward linkage**, which chooses the merge giving the smallest increase in within-cluster sum of squares. Detailed comparisons with single, complete, and average linkage belong to Part 2.

K-Means reassigns individual observations around evolving centers. Agglomerative clustering commits to a sequence of merges. Because their procedures differ, the same features and requested number of groups can produce different memberships.

For a fair comparison, keep the observations, features, and preprocessing consistent. Otherwise, a difference might be caused by representation rather than the clustering procedure.

The lab's `AgglomerativeClustering(n_clusters=3, linkage="ward")` requests three final groups. Conceptually, a full hierarchy can be built before choosing where to cut it. Ward linkage in this implementation uses Euclidean geometry; see the [agglomerative reference](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html).

### 4.3 Read a dendrogram

A dendrogram is a tree diagram of a hierarchy. Read it through three elements:

- **Leaves:** the original observations in an untruncated tree.
- **Branches and joins:** which observations or groups merge.
- **Merge height:** the value of the linkage criterion associated with a merge.

The horizontal placement of leaves organizes the drawing; it is not a feature axis. Merge height must be interpreted according to linkage. For Ward, avoid treating every height as a simple distance between two individual observations.

The activity's four-point example has A = `(1,1)`, B = `(1,2)`, C = `(8,8)`, and D = `(8,9)`. A and B form one close pair, and C and D form another. Combining the pairs requires a much higher merge.

A **horizontal cut** selects a level in the tree. Observations joined below the cut are in the same cluster. A cut between the pair merges and the final merge yields `{A,B}` and `{C,D}`.

A large gap between merge heights can suggest a candidate cut, but does not prove that its group count is correct. A full customer tree contains many leaves; the small example makes the reading rules clearer before moving to that larger tree.

## 5. Interpreting results and recognizing limitations

### 5.1 Interpret cluster profiles

A **cluster profile** summarizes the observations in a group, for example using size, mean income, and mean spending score.

This illustrative table is not a claimed output of the customer experiment:

| Group | Customers | Mean income (thousands of dollars) | Mean spending score |
|---|---:|---:|---:|
| A | 40 | 30 | 20 |
| B | 55 | 75 | 80 |
| C | 35 | 80 | 25 |

Group B can be described as having relatively high mean income and spending score. We cannot conclude from these means that all its members are alike or that they share a particular motive.

Group size helps identify small segments; means help describe central tendencies. Neither establishes quality alone. In a fuller analysis, distributions and variation within each group also matter.

Labels are produced by the model. Descriptions are interpretations supported by the observed features.

### 5.2 Understand overlapping groups

Synthetic experiments let us change one factor while holding others fixed. Activity 1 holds the generation centers and sample count fixed while increasing spread.

Tightly concentrated groups are easier to separate visually. With more spread, groups overlap and some observations become ambiguous. K-Means still assigns each observation to one of its requested clusters.

Generation labels describe the synthetic process; clustering labels describe the model's partition. They are different pieces of information. The model in the activity receives only features, not generation labels.

The lesson transfers as a limitation: overlap makes group membership harder to interpret. It does not imply that real customers follow the same synthetic distribution.

### 5.3 Distinguish a partition from natural groups

K-Means can partition observations sampled uniformly across a square. Producing three colored regions therefore does not prove that three natural populations exist.

This distinction matters because plots can make an imposed partition look persuasive. The algorithm has answered its optimization question, but we still need to assess the result against the purpose of the analysis.

Avoid generalizing that every clustering algorithm must create a chosen number of groups. The fixed-K behavior demonstrated here belongs to K-Means and the final group count requested from our agglomerative model.

### 5.4 Recognize the limitations

| Consideration | K-Means | Agglomerative clustering |
|---|---|---|
| Features and scale | Define the distances to centers | Affect merge decisions |
| Number of groups | K must be chosen | A final partition requires a cut or group count |
| Shape and overlap | Compact groups around centers are a natural fit; complex geometry can be difficult | Result depends strongly on linkage and geometry |
| Outliers | Can move centroids and change assignments | Can affect the merge hierarchy |
| Algorithm choices | Initialization can change the final result | Linkage and early irreversible merges can change the result |
| Size of dataset | Often a practical starting point for large numerical datasets | A full hierarchy may be expensive and difficult to inspect |
| Meaning | Requires interpretation | Requires interpretation |

Neither method is universally best. Dataset size alone is not enough to decide: the intended question, representation, and grouping structure also matter.

## 6. Consolidation and preparation for Part 2

### 6.1 Compare the two methods

The central procedural difference is:

| K-Means | Agglomerative hierarchical clustering |
|---|---|
| Choose K and initialize centers | Start with individual observations |
| Assign points and update centers repeatedly | Merge groups repeatedly using linkage |
| Return a partition and its centroids | Provide nested relationships and a final partition |
| Inspect centers and group membership | Inspect membership and a dendrogram |

Both discover groups without target labels guiding fitting. Both depend on how similarity is represented.

### 6.2 Check your understanding

Use these questions for conceptual revision. The activity contains the practical checkpoints and calculations.

1. A dataset contains a category column that is excluded from clustering. Does its existence make the clustering supervised?
2. Can a centroid be a location where no observation exists?
3. Two plots use different colors but group exactly the same observations together. Are the partitions different?
4. Why can changing a feature's units change a result?
5. How does a hierarchy provide information beyond one K-Means partition?
6. Does increasing overlap prevent K-Means from returning labels?
7. What evidence supports a description such as “higher-income customers”?

<details>
<summary>Suggested answers</summary>

1. No. What matters is whether the category labels guide the fitting process.
2. Yes. A mean is a calculated location and need not equal an observed point.
3. No. Group membership is unchanged; the visual identifiers differ.
4. Units alter coordinate differences and their relative contributions to distance.
5. A hierarchy shows nested group relationships and permits examining different cut levels.
6. No. It still assigns points, even where membership is ambiguous.
7. The actual income values or summaries for that group, compared with other groups. Its numerical label supplies no such evidence.

</details>

### 6.3 Prepare for evaluation

In supervised learning, predictions can be compared with known targets. In clustering, there may be no reference grouping to treat as the correct answer.

Evaluation therefore combines several forms of evidence:

- **Cohesion:** are observations within a group close under the chosen representation?
- **Separation:** are different groups distinct?
- **Stability:** do small changes in data or initialization substantially change the grouping?
- **Interpretability and usefulness:** can we describe the groups, and do they help answer the intended question?

When relevant reference labels exist, comparing them with clusters afterward is another source of evidence. Those labels need not coincide with the structure the selected features reveal.

Part 2 develops inertia, the elbow method, silhouette scores, scaling experiments, and detailed linkage analysis. Here, retain the main principle: **producing groups is the beginning of analysis, not sufficient evidence of a useful result.**

## 7. A broader view of unsupervised learning — theory only

This section provides awareness of other tasks. It introduces their questions and outputs without adding coding exercises to Activity 1.

### 7.1 Match the task to the question

| Task | Main question | Typical output |
|---|---|---|
| Clustering | Which observations belong together? | Group assignments, possibly centers or a hierarchy |
| Anomaly detection | Which observations are unusual? | Anomaly scores and flags |
| Association rule learning | Which items tend to occur together? | Rules with measures of association |
| Dimensionality reduction | Can we represent observations using fewer coordinates? | A reduced feature representation |
| Density estimation | Where is the data concentrated? | An estimated probability density |
| Generative modeling | Can we produce new samples resembling the observed data? | A model that can generate samples |

These tasks can overlap. For example, a distribution model may support both density estimation and generating samples. Select a method according to the question, rather than treating every unsupervised method as a grouping algorithm.

### 7.2 Anomaly detection

Anomaly detection identifies observations that appear unusual relative to the patterns a model has learned. Examples include unusual customer profiles, sensor readings, or account activity.

An unusual observation is not automatically an error or evidence of wrongdoing. It may be a rare valid event. A flag indicates a case to investigate; it does not explain why the case is unusual.

Isolation Forest is one example: its central intuition is that unusual observations can often be isolated with relatively few random partitions. Part 2 develops this topic. Anomaly detection can also use supervised approaches when labeled examples are available; here we are introducing its unsupervised form.

### 7.3 Association rule learning

Association rule learning looks for items or events that occur together. In market-basket data, each transaction contains a collection of products.

A rule such as “Bread → Butter” describes how often butter occurs among transactions containing bread. It does not mean that buying bread causes a person to buy butter.

Common measures describe how frequent the pattern is (**support**), how often the consequent occurs given the antecedent (**confidence**), and how the co-occurrence compares with independence (**lift**). The main distinction from clustering is the question: co-occurring items rather than groups of similar customers.

### 7.4 Dimensionality reduction

Dimensionality reduction represents observations using fewer coordinates while trying to retain useful structure. A dataset with 100 features might be represented in two dimensions for visualization, or in a smaller number of dimensions for further analysis.

**Principal Component Analysis (PCA)** is a common example. It constructs new axes from combinations of the original features and keeps directions that explain the greatest variance. These components are transformed features, not cluster labels and not simply a selection of original columns.

For example, measurements of height, leg length, and arm length may contain related information. A component could summarize a shared direction of variation across these measurements. Reducing dimensions usually discards some information, and large variance is not automatically the same as relevance to an application.

Other methods, such as t-SNE, are often used to explore data visually. The broad lesson is that a two-dimensional representation can help us inspect high-dimensional observations, but apparent groups in the projection do not by themselves prove meaningful clusters in the original data.

At this stage, remember the distinction: **clustering groups observations; dimensionality reduction changes their representation.** No PCA calculations or implementation are required in Activity 1.

### 7.5 Density estimation and generative modeling

Density estimation models where observations are concentrated in feature space. It describes an estimated distribution rather than simply assigning each observation to a group.

Generative modeling uses learned patterns to produce new samples. A Gaussian mixture model is an example that can estimate a density and generate samples; its mixture components can also support a form of clustering.

More advanced examples include variational autoencoders and generative adversarial networks. Their details are outside this introductory material. Generative models can be trained in several learning settings, so “generative” is not a synonym for “unsupervised.”

## References

- [scikit-learn: K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [scikit-learn: Agglomerative clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
- [SciPy: Linkage and hierarchical clustering](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)
- [scikit-learn: Principal Component Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

