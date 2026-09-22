# Theory Part 2: Evaluating Clusters and Detecting Anomalies

[Activity 1](activity1.md) introduced how K-Means and agglomerative clustering form groups. This reading explains how to judge a grouping, how preprocessing influences the result, and how anomaly detection answers a different question.

Sections 1–6 follow [Activity 2](activity2.md). Section 7 keeps broader topics in the theory: association rules, dimensionality reduction, density estimation, and generative modeling. Its worked association-rule example is further reading, not an additional Activity 2 coding task.

## Table of contents

- [1. Preparing data and investigating scale](#1-preparing-data-and-investigating-scale)
  - [1.1 Understand the automobile example](#11-understand-the-automobile-example)
  - [1.2 Understand standardization](#12-understand-standardization)
  - [1.3 Separate fitting coordinates from display coordinates](#13-separate-fitting-coordinates-from-display-coordinates)
  - [1.4 Reuse fitted preprocessing appropriately](#14-reuse-fitted-preprocessing-appropriately)
- [2. Choosing and evaluating candidate cluster counts](#2-choosing-and-evaluating-candidate-cluster-counts)
  - [2.1 Understand inertia and its limits](#21-understand-inertia-and-its-limits)
  - [2.2 Interpret an elbow plot](#22-interpret-an-elbow-plot)
  - [2.3 Understand silhouette scores](#23-understand-silhouette-scores)
  - [2.4 Combine evidence to choose a candidate](#24-combine-evidence-to-choose-a-candidate)
- [3. Exploring hierarchical clustering in more detail](#3-exploring-hierarchical-clustering-in-more-detail)
  - [3.1 Understand linkage criteria](#31-understand-linkage-criteria)
  - [3.2 Compare methods on a consistent basis](#32-compare-methods-on-a-consistent-basis)
  - [3.3 Read and cut a dendrogram](#33-read-and-cut-a-dendrogram)
- [4. Interpreting and checking clustering results](#4-interpreting-and-checking-clustering-results)
  - [4.1 Use profiles and inspect individual fit](#41-use-profiles-and-inspect-individual-fit)
  - [4.2 Distinguish reproducibility from stability](#42-distinguish-reproducibility-from-stability)
  - [4.3 Make a defensible recommendation](#43-make-a-defensible-recommendation)
- [5. Detecting unusual customer profiles](#5-detecting-unusual-customer-profiles)
  - [5.1 Define the anomaly question and the observation](#51-define-the-anomaly-question-and-the-observation)
  - [5.2 Understand Isolation Forest and its outputs](#52-understand-isolation-forest-and-its-outputs)
  - [5.3 Interpret flags using the full feature set](#53-interpret-flags-using-the-full-feature-set)
  - [5.4 Understand contamination and evaluate screening](#54-understand-contamination-and-evaluate-screening)
- [6. Consolidation and conceptual review](#6-consolidation-and-conceptual-review)
  - [6.1 Compare the questions and evidence](#61-compare-the-questions-and-evidence)
  - [6.2 Check your understanding](#62-check-your-understanding)
  - [6.3 Explain a complete analysis](#63-explain-a-complete-analysis)
- [7. Broader techniques and further reading](#7-broader-techniques-and-further-reading)
  - [7.1 Match the technique to the question](#71-match-the-technique-to-the-question)
  - [7.2 Understand association rules through one consistent example](#72-understand-association-rules-through-one-consistent-example)
  - [7.3 Understand dimensionality reduction](#73-understand-dimensionality-reduction)
  - [7.4 Understand density estimation and generative modeling](#74-understand-density-estimation-and-generative-modeling)
- [References](#references)

## 1. Preparing data and investigating scale

### 1.1 Understand the automobile example

Activity 2 clusters automobiles using **horsepower** and **weight**. A row is a car; its selected measurements are the features. MPG is available in the source dataset but is not supplied as a target or clustering feature.

This differs from the earlier regression problem. Predicting MPG uses known MPG values during training. Clustering horsepower and weight asks which cars have similar measurements without a target guiding the grouping.

These are observed automobile data. There is no known synthetic generation count that establishes four clusters. A visible pattern can suggest candidates, but a continuous distribution can also be partitioned.

Inspection comes first: confirm the meaning of a row, data types, missing values, and feature ranges. Dropping incomplete rows changes the observations included in the analysis, so record the number removed and consider whether the remaining sample is appropriate.

### 1.2 Understand standardization

Distance-based algorithms respond to coordinate differences. If weight varies by thousands of pounds while horsepower varies by hundreds, weight may contribute much more to raw Euclidean distances.

Standardization transforms a feature value using its mean and standard deviation:

$$
z=\frac{x-\mu}{\sigma}
$$

For example, if a measurement is 150, the feature mean is 100, and its standard deviation is 25:

$$
z=\frac{150-100}{25}=2
$$

The value is two standard deviations above the mean. For a nonconstant feature, standardization produces mean zero and standard deviation one on the data used to fit it.

It does not make a feature normally distributed, remove outliers, or prove that every feature deserves equal importance. Means and standard deviations can themselves be affected by extreme values. Standardization is a representation choice, not an automatic quality guarantee.

### 1.3 Separate fitting coordinates from display coordinates

A model fitted on standardized features uses standardized distances. Its centroids are also in that coordinate system.

It is valid to show its assignments on a plot of the original measurements, provided the same rows are used in the same order. To plot the centroids on those axes, transform them back to original units first.

| Operation | Representation in Activity 2 |
|---|---|
| Main clustering fits | Standardized horsepower and weight |
| Inertia and silhouette calculations | The same standardized features |
| Scatter-plot axes | Original horsepower and weight |
| Cluster profiles | Original measurements grouped by the fitted labels |

The raw-versus-scaled experiment deliberately changes the fitting representation while holding the observations and K fixed. Differences show sensitivity to feature scale; they do not prove that either partition is the true one.

Raw and standardized inertia values are not directly comparable as quality scores because their distance units differ. More generally, comparisons across changed features or representations answer different geometric questions.

### 1.4 Reuse fitted preprocessing appropriately

When exploring one dataset, clustering and its internal diagnostics can use that dataset. A supervised-style train/test split is not required for every descriptive clustering exercise.

If the objective includes judging performance on unseen observations, separate the data appropriately. Learn scaling and imputation parameters from the training/reference data, then reuse them on validation or test data.

Fitting a new scaler to the test data changes the coordinate system. Fitting preprocessing to the entire dataset before a held-out evaluation lets evaluation data influence the analysis.

Reproducible preprocessing is necessary, but evaluation also needs to match what the system will actually do with future observations.

## 2. Choosing and evaluating candidate cluster counts

### 2.1 Understand inertia and its limits

K-Means minimizes the within-cluster sum of squared distances, reported as inertia:

$$
I=\sum_{i=1}^{n}\left\|x_i-\mu_{c_i}\right\|^2
$$

Here:

- $x_i$ is observation i in the fitting feature space;
- $c_i$ identifies the cluster assigned to that observation;
- $\mu_{c_i}$ is that cluster's centroid;
- the squared distance measures how far the observation lies from its center.

For a fixed dataset and representation, lower inertia means lower total squared distance to centers. It is a sum, so its magnitude also depends on the number of observations, features, and their scales. An isolated value such as “inertia = 500” is not a universal good or bad rating.

Adding centers gives the model more flexibility. The optimal achievable inertia cannot increase as K increases, although separately fitted runs can occasionally fail to achieve that pattern because K-Means can reach different local solutions.

With a centroid at every observation, the distance contributions are zero. Therefore, minimizing inertia alone gives no useful penalty for creating too many groups.

**Worked check:** Three observations have distances 1, 2, and 3 to their assigned centers. What do they contribute to inertia?

<details>
<summary>Calculation</summary>

```math
I_{\text{three observations}}=1^2+2^2+3^2=14
```

The contribution is 14, not the sum of the unsquared distances, 6.

</details>

### 2.2 Interpret an elbow plot

An elbow plot displays inertia against K. It looks for a point after which extra clusters provide smaller reductions.

Consider these illustrative results, not claimed automobile output:

| K | Inertia |
|---:|---:|
| 1 | 1200 |
| 2 | 600 |
| 3 | 350 |
| 4 | 300 |
| 5 | 270 |
| 6 | 250 |

The reductions from 1 to 2 and 2 to 3 are large. After K = 3, improvements are much smaller. Three is therefore a reasonable candidate to investigate.

This is a heuristic. The curve may have several plausible bends or no clear bend. Reporting uncertainty is more accurate than forcing a unique answer from an ambiguous curve.

### 2.3 Understand silhouette scores

Silhouette evaluates each observation using two average distances:

- **a:** the average distance to the other observations in its own cluster;
- **b:** the smallest average distance to any other cluster, computed by considering each competing cluster separately.

For an ordinary nonsingleton cluster membership:

$$
s=\frac{b-a}{\max(a,b)}
$$

The definition uses point-to-point distances, not just distances to centroids. The software assigns zero to a point in a singleton cluster; the ordinary formula does not have a within-cluster average for that case.

| Silhouette | Interpretation under the selected distance |
|---|---|
| Near 1 | The observation is much closer to its own cluster |
| Near 0 | Own and nearest competing cluster have similar average distances |
| Below 0 | A competing cluster is closer on average |

The score reported for a partition is the mean of its individual values. At least two clusters and fewer clusters than observations are required. For the exact interface and conventions, see the [silhouette reference](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html).

**Worked check:** If a = 2 and b = 5, what is the silhouette? What would change if a = 5 and b = 2?

<details>
<summary>Calculation and interpretation</summary>

```math
s_1=\frac{5-2}{5}=0.6
```

The observation fits its own cluster better under this comparison.

```math
s_2=\frac{2-5}{5}=-0.6
```

A competing cluster is closer on average. This is evidence of a poor geometric fit, not a verified real-world misclassification.

</details>

A high mean does not establish natural categories, and it can hide poorly matched points or small groups. Silhouette also favors some kinds of compact, separated geometry over other useful structures. Interpret it alongside the data and task.

### 2.4 Combine evidence to choose a candidate

Inertia and silhouette answer different questions. An elbow asks when improvements in compactness diminish; silhouette compares within-group distances with competing-group distances.

Consider another illustrative table:

| K | Inertia | Mean silhouette |
|---:|---:|---:|
| 2 | 900 | 0.41 |
| 3 | 600 | 0.56 |
| 4 | 470 | 0.49 |
| 5 | 390 | 0.38 |

K = 5 has the lowest inertia, but K = 3 has the highest silhouette. Three is a useful candidate, not an automatically proven optimum.

Inspect its plots, group sizes, profiles, and stability. A slightly lower silhouette may be acceptable if another partition serves a clear application purpose better. Explain that tradeoff rather than presenting the preferred count as an objective discovery.

A small group is also ambiguous: it may be a valid rare segment, a set of unusual observations, or an effect of the algorithm and K. Equal-sized groups are not a universal requirement.

## 3. Exploring hierarchical clustering in more detail

### 3.1 Understand linkage criteria

Agglomerative clustering begins with individual observations and repeatedly merges groups. Linkage defines how the candidate merges are ranked.

| Linkage | Definition | Useful intuition |
|---|---|---|
| Single | Minimum distance between any cross-group pair | One close connection can join groups |
| Complete | Maximum distance between any cross-group pair | Even the farthest pair matters |
| Average | Mean distance across all cross-group pairs | Uses the overall cross-group relationship |
| Ward | Increase in within-cluster sum of squares | Prefers merges with a small increase in within-group variation |

Single linkage can connect distant regions through chains of close observations. Complete linkage is more affected by distant cross-group pairs. Ward is a variance-based criterion and is used with Euclidean geometry in this activity.

Linkage is not the same as the point-to-point distance metric. Euclidean distance compares individual observations; linkage uses distances or variance to compare groups.

**Small example:** Two candidate groups have cross-group distances 1, 2, 8, and 9. Their single-linkage value is 1, complete-linkage value is 9, and average-linkage value is 5. Ward's criterion cannot be obtained merely by choosing the minimum, maximum, or mean of that list.

### 3.2 Compare methods on a consistent basis

Changing linkage can change the first merge decisions and therefore the final hierarchy. Agglomerative merges are not undone later.

When comparing linkage choices, hold the observations, feature representation, and final group count fixed. Otherwise, a result may differ because several things changed at once.

Silhouette can compare different clustering algorithms when it is calculated on the same data representation and distance definition. It is not restricted to K-Means. K-Means inertia, however, is tied to a centroid-based squared-distance objective; do not treat it as a universal score for every unsupervised task.

A practical comparison combines scores with membership plots, group sizes, and interpretation. A high score alone does not establish that one linkage is universally better.

### 3.3 Read and cut a dendrogram

A dendrogram shows how observations or groups join as the linkage height increases. A horizontal cut keeps observations together when their branches join below the cut.

Lowering the cut generally leaves more groups because fewer merges are accepted. Raising it generally produces fewer groups. A large gap between successive merge heights may suggest a useful candidate cut, but still needs evaluation.

Activity 2 displays a **truncated** tree for readability. Some displayed leaves therefore represent already merged groups, not individual cars. Parenthesized leaf counts show how many observations are represented. This differs from the four-observation untruncated tree in Activity 1.

The activity positions a cut between merges to illustrate a candidate group count. It does not automatically search for the largest gap or discover the best partition. Equal merge heights can make a desired count unavailable using a single distance threshold; inspect the number actually produced.

SciPy's `fcluster` converts a hierarchy into assignments at a specified cut. Its numerical group identifiers need not match those produced by a separately fitted model. See the [SciPy cut reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html).

**Background distinction:** Agglomerative clustering builds upward by merging. Divisive hierarchical clustering begins with one group and works downward by splitting. The activity implements the agglomerative approach; divisive clustering remains a conceptual comparison.

## 4. Interpreting and checking clustering results

### 4.1 Use profiles and inspect individual fit

A cluster number has no inherent meaning. A profile describes the observations assigned to it using quantities such as group size and mean feature values.

For the automobile example, a profile can support “relatively heavy cars with higher horsepower.” It cannot, from those features alone, support claims about reliability, price, or driver preferences.

Means can hide substantial variation. Inspecting per-cluster silhouette averages and the fraction of negative individual silhouettes helps reveal groups that the overall mean conceals.

For example, a large well-separated group can dominate the overall average even when a smaller group is poorly separated. A negative silhouette suggests a competing group is closer on average; there is no supplied true label proving that the assignment is wrong.

### 4.2 Distinguish reproducibility from stability

**Reproducibility** means the same setup can produce the same result again. A fixed random seed helps with this.

**Stability** asks how much the result changes when something changes slightly. Relevant checks include different initializations, modest measurement perturbations, or resampling the observations.

Activity 2 checks only sensitivity to initialization: it changes the seed while holding K, features, preprocessing, and the number of initializations fixed. Agreement is useful evidence about that particular change, not a complete stability assessment.

Compare which observations remain together, allowing arbitrary cluster IDs to change. Similar inertia or silhouette values alone do not establish identical memberships.

A stable partition can still be unhelpful. An algorithm may consistently divide a continuous cloud of points into the same regions without discovering meaningful categories.

### 4.3 Make a defensible recommendation

A useful recommendation explains the question, representation, candidate solution, supporting evidence, and limitations.

| Evidence | What it helps assess | What it cannot establish alone |
|---|---|---|
| Inertia and elbow | Compactness and diminishing returns as K grows | One objectively correct K |
| Silhouette | Within-group similarity versus competing groups | Application usefulness or true categories |
| Plots and group sizes | Shape, overlap, small groups, unusual regions | All relationships in a high-dimensional space |
| Profiles | Characteristics of each group | Causes or motives |
| Stability checks | Sensitivity to specified changes | Meaningful natural populations |
| Domain interpretation | Relevance to the intended purpose | Geometric quality by itself |

If relevant reference labels exist, they may be used for an external comparison after clustering. The fit can remain unsupervised, but a known category system need not match the structure discovered from the selected features. Raw cluster IDs should not be compared directly with class numbers as though they had matching meanings.

## 5. Detecting unusual customer profiles

### 5.1 Define the anomaly question and the observation

Clustering asks which observations belong together. Anomaly detection asks which observations look unusual relative to the patterns a model has learned.

The activity's credit-card dataset contains aggregated **customer profiles**. Each row describes usage patterns such as balances, purchases, and cash advances. It is not a collection of labeled individual fraudulent transactions.

An unusual profile could reflect a legitimate rare behavior, a data-quality problem, or another case needing context. A flag does not identify the cause.

The activity excludes customer IDs from fitting but retains them for investigation. It fills missing numerical measurements with feature medians. Imputed values can affect detection, so the original measurements should remain available for review. Future-data evaluation requires fitting such preprocessing only on the training/reference data.

### 5.2 Understand Isolation Forest and its outputs

Isolation Forest builds random partitions. A tree repeatedly chooses a feature and a split value. Observations that separate from others after relatively few splits have short isolation paths.

Across many trees, unusual observations often have shorter paths than observations in common regions. This provides the model's isolation-based notion of unusualness; it is not a nearest-centroid calculation.

Because this method is based on splits rather than Euclidean distances, the scaling rationale used for K-Means does not transfer directly. Standardization is not required merely to balance Euclidean contributions. Feature selection, missing-value handling, and the meaning of measurements still matter.

Separate three outputs:

| Output | Meaning in this activity |
|---|---|
| `score_samples` value | A model score; lower means more unusual in this implementation |
| `1` or `-1` prediction | An inlier or anomaly flag after applying a threshold |
| Confirmed event | A conclusion requiring external investigation or trusted labels |

A score is not a probability that a customer committed fraud. A prediction of `1` is also not proof that the customer has no unusual or problematic behavior.

The [Isolation Forest reference](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) documents its score and threshold conventions.

### 5.3 Interpret flags using the full feature set

The model in Activity 2 uses all selected numerical profile features. Its balance-versus-purchases scatter plot displays only two.

A flagged customer may look ordinary on those axes but unusual on other features or their combinations. The plot is a partial view, not a full explanation of the model's judgment.

Review the original measurements, missingness, and relevant context. Do not automatically delete flagged observations: rare valid cases may be precisely what the analysis is meant to reveal.

### 5.4 Understand contamination and evaluate screening

With a numeric setting such as `contamination=0.02`, the implementation chooses a score cutoff targeting approximately 2% flagged observations in the fitted data. The setting is an assumption for thresholding, not an estimate verified against real anomaly labels.

Changing that value while keeping features and random construction fixed changes the decision threshold. A larger setting generally flags more observations. It does not establish that more real anomalies have appeared.

The same fraction is not guaranteed for future observations because their distribution may differ. The separate `"auto"` setting uses the implementation's automatic cutoff rule; it does not mean that the true real-world anomaly prevalence is known.

Without verified case labels, inspect the flagged observations, sensitivity to thresholds, stability, and relevance to the task. Counts alone cannot establish detection accuracy.

When trustworthy labels are available for the event of interest, evaluation can use ideas from the classification lecture:

- **Precision:** among flagged cases, how many are confirmed cases of interest?
- **Recall:** among all actual cases of interest in the evaluated data, how many are flagged?

Those labels must come from independent evidence, not the model's own flags. Finding more flags is not automatically better: false alarms and missed cases have different costs. Reviewing only flagged cases also leaves uncertainty about missed cases among the unflagged observations.

## 6. Consolidation and conceptual review

### 6.1 Compare the questions and evidence

| Question | Suitable task | Main output | Interpretation needed |
|---|---|---|---|
| Which cars have similar horsepower and weight? | Clustering | Group assignments and profiles | Are the groups coherent and useful? |
| How are groups related at different levels? | Hierarchical clustering | A hierarchy and possible cuts | Which level suits the purpose? |
| Which customer profiles are unusual? | Anomaly detection | Scores and flags | Why are those cases unusual, and do they matter? |

An isolated observation still receives a K-Means group assignment. An anomaly model asks a different question about it. Neither output, by itself, supplies a causal explanation.

### 6.2 Check your understanding

Use these questions for closed-book conceptual revision:

1. Why is the lowest inertia not enough to choose K?
2. What do a and b mean in the silhouette formula?
3. Why can raw and standardized inertia values not be compared directly?
4. How do single and complete linkage differ?
5. What does a cut through a dendrogram select?
6. Can a fixed random seed prove stability?
7. Why does a small cluster need investigation rather than automatic rejection?
8. What does contamination control in the activity's Isolation Forest model?
9. Why can a flagged point look ordinary in a two-feature plot?
10. What additional evidence is needed to calculate precision or recall?

<details>
<summary>Suggested answers</summary>

1. Additional centroids provide flexibility to reduce distances without guaranteeing useful groups.
2. a is mean distance to other points in the same cluster; b is the smallest mean distance to a competing cluster.
3. They use different coordinate systems and distance scales.
4. Single uses the closest cross-group pair; complete uses the farthest.
5. It selects the merges accepted below a chosen height and therefore a partition.
6. No. Repeating an identical setup tests reproducibility; stability requires investigating changes.
7. It could be a useful rare segment or an artifact of the modeling choices.
8. A score threshold targeting an assumed fraction of flagged training observations.
9. Other features used by the model may explain its unusualness.
10. Trusted labels for the actual cases of interest, independent of the predicted flags.

</details>

### 6.3 Explain a complete analysis

A coherent explanation follows the decisions that determine the result:

**Question → observations and features → preprocessing → model → evidence → interpretation and limitations.**

For clustering, justify a candidate grouping using several kinds of evidence. For anomaly detection, explain the threshold and the investigation needed after a flag.

The central principle is that unsupervised outputs describe structure according to a model. Their usefulness has to be assessed in the context of the question.

## 7. Broader techniques and further reading

These topics remain in the theory. They do not add implementation exercises to Activity 2. [Part 1's overview](part1.md#7-a-broader-view-of-unsupervised-learning--theory-only) introduces their purposes; the following examples develop the distinctions.

### 7.1 Match the technique to the question

| Objective | Technique | Output |
|---|---|---|
| Group similar observations | Clustering | Assignments, centers, or a hierarchy |
| Identify unusual observations | Anomaly detection | Scores and flags |
| Find items that occur together | Association rule learning | Co-occurrence rules and measures |
| Represent observations with fewer coordinates | Dimensionality reduction | Transformed features |
| Model where observations are concentrated | Density estimation | An estimated probability density |
| Produce new samples resembling observed data | Generative modeling | A model capable of sampling |

These tasks can overlap. One model may support several purposes, but they are not interchangeable questions.

### 7.2 Understand association rules through one consistent example

An association rule describes co-occurrence. In market-basket data, each row is a transaction containing a set of products. An **itemset** is a collection such as `{Bread, Butter}`. Apriori is one method for finding itemsets that meet a minimum frequency requirement.

A rule `Bread → Butter` asks how often butter appears in transactions containing bread. Its direction does not establish a time order or cause.

For this example, suppose there are 100 transactions:

| | Butter present | Butter absent | Total |
|---|---:|---:|---:|
| Bread present | 30 | 10 | 40 |
| Bread absent | 20 | 40 | 60 |
| Total | 50 | 50 | 100 |

**Support** is the proportion of transactions containing an itemset. In this table:

$$
\operatorname{support}(\{\text{Bread}\})=\frac{40}{100}=0.40
$$

$$
\operatorname{support}(\{\text{Butter}\})=\frac{50}{100}=0.50
$$

$$
\operatorname{support}(\{\text{Bread},\text{Butter}\})=\frac{30}{100}=0.30
$$

The combined itemset means transactions containing **both** items. Its support is also commonly called the rule's support.

**Confidence** is the fraction of transactions containing the antecedent that also contain the consequent:

$$
\operatorname{confidence}(\text{Bread}\rightarrow\text{Butter})=\frac{30}{40}=0.75
$$

Thus 75% of bread-containing transactions also contain butter. Reversing the rule gives a different confidence:

$$
\operatorname{confidence}(\text{Butter}\rightarrow\text{Bread})=\frac{30}{50}=0.60
$$

**Lift** compares confidence with the consequent's overall support:

$$
\operatorname{lift}(A\rightarrow B)=\frac{\operatorname{confidence}(A\rightarrow B)}{\operatorname{support}(B)}
$$

<details>
<summary>Calculate lift in both directions</summary>

$$
\operatorname{lift}(\text{Bread}\rightarrow\text{Butter})=\frac{0.75}{0.50}=1.5
$$

$$
\operatorname{lift}(\text{Butter}\rightarrow\text{Bread})=\frac{0.60}{0.40}=1.5
$$

The confidence values differ, but the lifts are equal. For the same data, lift is symmetric because it is the joint occurrence proportion divided by the product of the two individual occurrence proportions.

</details>

| Lift | Meaning in the observed data |
|---|---|
| Greater than 1 | Co-occurrence is more frequent than under independence |
| Equal to 1 | Co-occurrence matches independence |
| Less than 1 | Co-occurrence is less frequent than under independence |

High confidence can arise simply because the consequent is common. Lift helps account for that baseline, but support and sample size still matter: a strong-looking rule based on very few transactions may be unreliable.

These definitions are consistent with the [mlxtend association-rule reference](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/).

**Interpretation check:** Does lift = 1.5 prove that buying bread causes buying butter?

<details>
<summary>Answer</summary>

No. It describes co-occurrence relative to an independence baseline. Shared preferences, shopping occasions, or other factors could explain the pattern. Association rules alone do not establish causation.

</details>

### 7.3 Understand dimensionality reduction

Dimensionality reduction transforms observations into fewer coordinates. It can help visualize many features, reduce redundancy, or prepare a representation for later analysis.

PCA constructs new axes from combinations of the original features, retaining directions that explain the greatest variance. The new coordinates are components, not customer categories. Retaining two components is not the same as asking for two clusters.

Reduction generally loses information. Directions with high variance need not contain everything relevant to an application. A plot showing only two components may also conceal differences visible in the original feature space.

Other visualization methods, such as t-SNE, emphasize different aspects of structure. Visually separated regions in a projection are evidence to investigate, not automatic proof of useful clusters in the original data. The [PCA reference](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) gives further detail; no derivation or implementation is required here.

### 7.4 Understand density estimation and generative modeling

Density estimation models how observations are distributed across feature space. Generative modeling uses learned patterns to produce new samples.

A Gaussian mixture model can estimate a density, generate samples, and provide component-membership probabilities useful for a form of clustering. This illustrates that the technique families overlap.

Variational autoencoders and generative adversarial networks are more advanced examples of generative approaches. Their architecture and training details are outside this lecture. Generative models can use several learning settings, so “generative” is not a synonym for “unsupervised.”

## References

- [scikit-learn: K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [scikit-learn: StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [scikit-learn: Silhouette score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
- [SciPy: Hierarchical linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)
- [SciPy: Cutting a hierarchy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html)
- [scikit-learn: Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [mlxtend: Association-rule measures](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/)
- [scikit-learn: PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
