# Activity 2: Evaluating Clusters and Detecting Anomalies

Activity 1 showed how clustering forms groups. This activity asks how to judge those groups and how to identify unusual observations. You will first investigate automobile clusters, then switch explicitly to customer credit-card profiles for anomaly detection.

Run the code in order in a fresh Google Colab notebook. Predict each experiment's outcome before running it, and attempt questions before opening solutions. The [Part 2 theory](part2.md) follows the same six main sections.

## Table of contents

- [1. Preparing data and investigating scale](#1-preparing-data-and-investigating-scale)
  - [1.1 Load and inspect automobile data](#11-load-and-inspect-automobile-data)
  - [1.2 Standardize the features](#12-standardize-the-features)
  - [1.3 Compare raw and scaled clustering](#13-compare-raw-and-scaled-clustering)
  - [1.4 Keep preprocessing consistent](#14-keep-preprocessing-consistent)
- [2. Choosing and evaluating candidate cluster counts](#2-choosing-and-evaluating-candidate-cluster-counts)
  - [2.1 Understand inertia](#21-understand-inertia)
  - [2.2 Build an elbow plot](#22-build-an-elbow-plot)
  - [2.3 Add silhouette scores](#23-add-silhouette-scores)
  - [2.4 Inspect a candidate partition](#24-inspect-a-candidate-partition)
- [3. Exploring hierarchical clustering in more detail](#3-exploring-hierarchical-clustering-in-more-detail)
  - [3.1 Compare linkage criteria](#31-compare-linkage-criteria)
  - [3.2 Fit and evaluate the alternatives](#32-fit-and-evaluate-the-alternatives)
  - [3.3 Build and cut a dendrogram](#33-build-and-cut-a-dendrogram)
- [4. Interpreting and checking clustering results](#4-interpreting-and-checking-clustering-results)
  - [4.1 Build profiles in original units](#41-build-profiles-in-original-units)
  - [4.2 Check sensitivity to initialization](#42-check-sensitivity-to-initialization)
  - [4.3 Make a defensible clustering recommendation](#43-make-a-defensible-clustering-recommendation)
- [5. Detecting unusual customer profiles](#5-detecting-unusual-customer-profiles)
  - [5.1 Change the question and inspect the data](#51-change-the-question-and-inspect-the-data)
  - [5.2 Fit an Isolation Forest](#52-fit-an-isolation-forest)
  - [5.3 Inspect scores and flagged observations](#53-inspect-scores-and-flagged-observations)
  - [5.4 Investigate the threshold choice](#54-investigate-the-threshold-choice)
- [6. Consolidation and conceptual review](#6-consolidation-and-conceptual-review)
  - [6.1 Compare the questions and outputs](#61-compare-the-questions-and-outputs)
  - [6.2 Check your understanding](#62-check-your-understanding)
  - [6.3 Explain an analysis from start to finish](#63-explain-an-analysis-from-start-to-finish)

## Learning objectives

By the end of this activity, you should be able to:

- explain and demonstrate how scaling affects distance-based clustering;
- interpret inertia, an elbow plot, and silhouette scores;
- compare hierarchical linkage methods and select a dendrogram cut;
- combine metrics, profiles, visual evidence, and stability when judging clusters;
- explain Isolation Forest, anomaly scores, and the contamination threshold;
- distinguish an unusual observation from an error or confirmed fraud.

## 1. Preparing data and investigating scale

### 1.1 Load and inspect automobile data

We use the automobile dataset `mpg`, with **horsepower** and **weight** as clustering features. This connects to the earlier regression lecture, but the question is now which cars have similar measurements. We do not predict MPG or supply it as a training target.

This is observed automobile data, not synthetic blobs. There is no known generation count that tells us how many clusters to expect.

```python
!pip install -q numpy pandas matplotlib seaborn scipy scikit-learn
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.ensemble import IsolationForest
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
```

```python
cars = sns.load_dataset("mpg")
car_features = ["horsepower", "weight"]
X_cars_df = cars[car_features].dropna().copy()
X_cars = X_cars_df.to_numpy()

print("Original rows:", len(cars))
print("Rows removed:", len(cars) - len(X_cars_df))
print("Clustering data shape:", X_cars.shape)
display(X_cars_df.describe())

plt.figure(figsize=(8, 5))
plt.scatter(X_cars_df["horsepower"], X_cars_df["weight"], alpha=0.6)
plt.xlabel("Horsepower")
plt.ylabel("Weight (lb)")
plt.title("Automobiles before clustering")
plt.show()
```

The earlier notebook recorded 392 complete rows and two selected features. Check your output rather than assuming a fixed count. Dropping rows with missing selected measurements is our introductory cleaning choice.

**Question 1 — Read the data before choosing a model**

Do you see clearly separated groups, a continuous pattern, or a mixture? Which feature has larger numerical variation? Does the plot establish that four clusters are correct?

<details>
<summary>Optional LLM hint</summary>

Ask: “Help me describe a horsepower-versus-weight scatter plot without assuming a correct cluster count. Ask me about gaps, overlap, and feature units.”

</details>

<details>
<summary>Possible answer</summary>

Use the observed plot to describe the pattern. Weight varies over a much larger numerical scale than horsepower. The plot may suggest candidate groupings, but these data were not generated with four known centers. Four is only a possible modeling choice.

</details>

### 1.2 Standardize the features

K-Means and our hierarchical methods use distances. Without scaling, weight differences can dominate horsepower differences because of the units.

Standardization uses each feature's mean and standard deviation:

$$
z=\frac{x-\mu}{\sigma}
$$

```python
car_scaler = StandardScaler()
X_cars_scaled = car_scaler.fit_transform(X_cars_df)

scaled_summary = pd.DataFrame(X_cars_scaled, columns=car_features)
print("Means after scaling:")
print(scaled_summary.mean().round(6))
print("Standard deviations after scaling:")
print(scaled_summary.std(ddof=0).round(6))
```

The means should be approximately zero and standard deviations approximately one. `ddof=0` matches the standard-deviation convention used by this scaler. You do not need to memorize that argument for conceptual revision.

### 1.3 Compare raw and scaled clustering

**Predict first:** Could cluster membership change even though we have exactly the same cars?

Use K = 4 as a controlled starting comparison, not a final decision:

```python
raw_model = KMeans(n_clusters=4, random_state=42, n_init=10)
raw_labels = raw_model.fit_predict(X_cars)

scaled_model = KMeans(n_clusters=4, random_state=42, n_init=10)
scaled_labels = scaled_model.fit_predict(X_cars_scaled)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
for ax, group_labels, title in [
    (axes[0], raw_labels, "Fitted in original units"),
    (axes[1], scaled_labels, "Fitted after standardization")
]:
    ax.scatter(X_cars[:, 0], X_cars[:, 1], c=group_labels, cmap="tab10", alpha=0.7)
    ax.set_xlabel("Horsepower")
    ax.set_title(title)
axes[0].set_ylabel("Weight (lb)")
plt.tight_layout()
plt.show()
```

Both plots use original units for readability. Only the representation used for fitting differs. Compare which observations belong together; cluster numbers and colors may be permuted between fits.

**Question 2 — Explain the change**

Why might the groups differ? Does standardization guarantee a more meaningful result? Why would comparing raw inertia directly with standardized inertia be misleading?

<details>
<summary>Solution</summary>

Standardization changes the features' relative contributions to distance. It can therefore change centers and assignments. Comparable numerical scales do not guarantee that the selected features or groups are useful.

Raw and standardized inertia are measured in different coordinate systems. Their numerical values are not directly comparable as evidence that one representation is better.

</details>

### 1.4 Keep preprocessing consistent

For the remaining clustering experiments, use **`X_cars_scaled` consistently for fitting and metrics**. Plotting those assignments against original units is fine because the rows correspond.

We are exploring one dataset, so these examples do not require a separate test set. If the goal later includes evaluating unseen observations, fit preprocessing on the training/reference data and reuse that fitted transformation. Do not learn a separate scaler from the test data.

## 2. Choosing and evaluating candidate cluster counts

### 2.1 Understand inertia

K-Means tries to reduce the sum of squared distances from observations to their assigned centroids. This is reported as **inertia**:

$$
I=\sum_{i=1}^{n}\left\|x_i-\mu_{c_i}\right\|^2
$$

Here $x_i$ is observation i, and $\mu_{c_i}$ is its assigned centroid. In this experiment, both are expressed in standardized coordinates.

```python
print("Inertia for the scaled K = 4 model:", scaled_model.inertia_)
```

Lower inertia means a smaller total squared distance on the same data representation. It does not establish the best K: more centers provide greater flexibility.

**Question 3 — Why not keep increasing K?**

Suppose K = 2, 3, and 4 give inertia values 900, 600, and 470. Why is K = 4 not automatically best? What happens in the ideal limiting case of one centroid at every observation?

<details>
<summary>Optional LLM hint</summary>

Ask: “Explain why adding centroids can reduce within-cluster squared distances without creating useful groups. Let me reason about one center per observation.”

</details>

<details>
<summary>Solution</summary>

More centers allow points to be represented more closely. With one centroid at every observation, each point has zero distance to its center, giving zero inertia. That partition usually provides little useful summarization.

Inertia alone rewards extra clusters; it does not balance compactness against a useful number of groups.

</details>

### 2.2 Build an elbow plot

Fit candidates from K = 1 to K = 8. Keep these models so later comparisons use the same fitted results.

```python
car_models = {}
car_labels_by_k = {}
inertia_rows = []

for k in range(1, 9):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    group_labels = model.fit_predict(X_cars_scaled)
    car_models[k] = model
    car_labels_by_k[k] = group_labels
    inertia_rows.append({"K": k, "Inertia": model.inertia_})

inertia_table = pd.DataFrame(inertia_rows)
plt.figure(figsize=(7, 4))
plt.plot(inertia_table["K"], inertia_table["Inertia"], marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia in standardized feature space")
plt.title("Elbow plot for automobile clustering")
plt.show()
```

Look for a bend after which additional clusters provide smaller reductions. A gradual curve without a clear elbow is a valid result, not a failed experiment.

**Question 4 — Propose a candidate**

Which K values would you investigate further? Identify the part of the curve supporting your choice. If there is no clear elbow, explain that uncertainty.

<details>
<summary>Possible answer</summary>

Choose candidates where improvement appears to slow, using your actual plot. There is no predetermined correct elbow for these automobile data. A defensible answer explains the pattern and recognizes ambiguity rather than claiming that the method proves a unique K.

</details>

### 2.3 Add silhouette scores

Silhouette compares an observation's average distance to its own cluster with its average distance to the nearest competing cluster. Values range from −1 to 1:

| Value | Interpretation |
|---|---|
| Near 1 | Much closer to its own group than to competing groups |
| Near 0 | Similar average distances; groups overlap or membership is ambiguous |
| Negative | On average, closer to a competing group |

The reported score is the mean over observations. It uses point-to-point distances, not just distances to centroids. At least two clusters are needed, and the number of clusters must be smaller than the number of observations. See the [silhouette reference](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html).

```python
silhouette_rows = []
for k in range(2, 9):
    score = silhouette_score(X_cars_scaled, car_labels_by_k[k])
    silhouette_rows.append({"K": k, "Silhouette": score})

silhouette_table = pd.DataFrame(silhouette_rows)
selection_table = inertia_table.merge(silhouette_table, on="K", how="left")
display(selection_table.round(3))

plt.figure(figsize=(7, 4))
plt.plot(silhouette_table["K"], silhouette_table["Silhouette"], marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Mean silhouette score")
plt.title("Silhouette scores for automobile clustering")
plt.show()
```

The missing silhouette value for K = 1 is intentional: there is no competing cluster.

**Question 5 — Combine the evidence**

Which K has the highest silhouette? Does it agree with your elbow candidates? If they disagree, what would you inspect next?

<details>
<summary>Optional LLM hint</summary>

Ask: “Help me compare elbow and silhouette results. Explain what each measures and ask me what additional evidence would help if they disagree.”

</details>

<details>
<summary>Possible answer</summary>

Report the values from your output. Elbow analysis examines diminishing reductions in inertia; silhouette compares within-group and between-group distances. They need not favor the same K.

Inspect the candidate plots, group sizes, profiles, and stability. The highest mean silhouette is evidence about geometry, not proof that a grouping is useful for an application.

</details>

### 2.4 Inspect a candidate partition

Choose a candidate from K = 2 through K = 8. The default below keeps the exercise runnable; replace it with another candidate in that range if your reasoning supports it.

```python
chosen_k = 3  # Candidate for inspection, not a declared optimum.
assert chosen_k in range(2, 9), "Choose an integer K from 2 through 8."
chosen_model = car_models[chosen_k]
chosen_labels = car_labels_by_k[chosen_k]
centers_original = car_scaler.inverse_transform(chosen_model.cluster_centers_)

print(pd.Series(chosen_labels).value_counts().sort_index())
plt.figure(figsize=(8, 5))
plt.scatter(X_cars[:, 0], X_cars[:, 1], c=chosen_labels, cmap="tab10", alpha=0.7)
plt.scatter(
    centers_original[:, 0], centers_original[:, 1],
    marker="X", s=200, c="black", label="Centroids"
)
plt.xlabel("Horsepower")
plt.ylabel("Weight (lb)")
plt.title(f"Candidate automobile partition: K = {chosen_k}")
plt.legend()
plt.show()
```

We transform the centroids back before plotting them in original units. A very small group may be a valid small segment or a sign to investigate; cluster size alone does not decide quality.

## 3. Exploring hierarchical clustering in more detail

### 3.1 Compare linkage criteria

Agglomerative clustering repeatedly merges groups. **Linkage** defines how candidate merges are compared.

| Linkage | Merge criterion |
|---|---|
| Single | Minimum pairwise distance between points in the two groups |
| Complete | Maximum pairwise distance between points in the two groups |
| Average | Mean of the cross-group pairwise distances |
| Ward | Increase in within-cluster sum of squares caused by merging |

**Question 6 — Reason about linkage**

Why can a chain of nearby observations connect otherwise separated regions under single linkage? Would complete linkage evaluate that connection in the same way?

<details>
<summary>Solution</summary>

Single linkage only needs a close pair connecting two groups. Repeated local connections can form a long chain. Complete linkage considers the farthest pair across the groups, so distant ends of that chain affect its decision.

</details>

### 3.2 Fit and evaluate the alternatives

Use the same scaled features and candidate group count as the K-Means comparison:

```python
linkage_methods = ["ward", "complete", "average", "single"]
hierarchical_results = []
hierarchical_labels_by_method = {}
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)

for ax, method in zip(axes.flat, linkage_methods):
    model = AgglomerativeClustering(n_clusters=chosen_k, linkage=method)
    group_labels = model.fit_predict(X_cars_scaled)
    hierarchical_labels_by_method[method] = group_labels
    counts = pd.Series(group_labels).value_counts()
    hierarchical_results.append({
        "Linkage": method,
        "Silhouette": silhouette_score(X_cars_scaled, group_labels),
        "Smallest group": counts.min(),
        "Largest group": counts.max()
    })
    ax.scatter(X_cars[:, 0], X_cars[:, 1], c=group_labels, cmap="tab10", alpha=0.7)
    ax.set_title(method.capitalize())
    ax.set_xlabel("Horsepower")
    ax.set_ylabel("Weight (lb)")
plt.tight_layout()
plt.show()
display(pd.DataFrame(hierarchical_results).round(3))
```

**Question 7 — Interpret different results**

Which methods produce similar memberships? Does any method produce a very small group? Compare the plots and silhouette scores with the chosen K-Means result. Why does the largest score not settle the choice by itself?

<details>
<summary>Possible answer</summary>

Use the observed outputs. Different linkage criteria can create different merge sequences, memberships, and sizes. A higher silhouette supports better separation under this representation, but profiles, stability, and the purpose of the analysis also matter.

Compare membership patterns rather than requiring the same colors or numerical IDs across models.

</details>

### 3.3 Build and cut a dendrogram

Build a Ward hierarchy from the same scaled observations. To keep its upper structure readable, display only the last 12 merged groups.

```python
Z_cars = linkage(X_cars_scaled, method="ward")

lower_height = Z_cars[-chosen_k, 2]
upper_height = Z_cars[-(chosen_k - 1), 2]
cut_height = (lower_height + upper_height) / 2

plt.figure(figsize=(11, 5))
dendrogram(Z_cars, truncate_mode="lastp", p=12, show_leaf_counts=True)
plt.axhline(cut_height, color="red", linestyle="--", label="Candidate cut")
plt.xlabel("Displayed leaves (parentheses indicate grouped observations)")
plt.ylabel("Ward linkage height")
plt.title("Automobile hierarchy: upper part of the tree")
plt.legend()
plt.show()

cut_labels = fcluster(Z_cars, t=cut_height, criterion="distance")
print("Clusters at this cut:", len(np.unique(cut_labels)))
print(pd.Series(cut_labels).value_counts().sort_index())
```

The dashed height is chosen between merges to illustrate the candidate count; it is **not an automatic search for the best cut**. Tied merge heights can prevent a distance cut from yielding the requested count, so inspect the printed result. SciPy's returned identifiers need not match scikit-learn's identifiers.

Because this tree is truncated, a displayed leaf may represent several observations. A parenthesized number is a group size, not a customer or car ID. Merge height is a Ward linkage value, not simply a raw weight difference.

**Question 8 — Interpret the hierarchy**

What would a lower cut generally do to the number of groups? Why might a large gap between merge heights suggest a candidate cut? Why should the cut still be evaluated?

<details>
<summary>Optional LLM hint</summary>

Ask: “Help me reason about a horizontal dendrogram cut. Explain which merges are accepted below the cut and ask me what changes as the cut moves downward.”

</details>

<details>
<summary>Solution</summary>

A lower cut accepts fewer merges and generally leaves more groups. A large gap suggests that combining groups across that interval requires a larger change in the linkage criterion.

This is evidence about the hierarchy's geometry. It does not prove a unique correct grouping or its usefulness for the application.

</details>

## 4. Interpreting and checking clustering results

### 4.1 Build profiles in original units

Return to the chosen K-Means model. Keep its assignments associated with the original automobile rows, then inspect both profiles and silhouette values within each group.

```python
car_profiles_data = X_cars_df.copy()
car_profiles_data["Cluster"] = chosen_labels
car_profiles_data["Silhouette"] = silhouette_samples(X_cars_scaled, chosen_labels)

car_profiles = car_profiles_data.groupby("Cluster").agg(
    Cars=("horsepower", "size"),
    MeanHorsepower=("horsepower", "mean"),
    MeanWeight=("weight", "mean"),
    MeanSilhouette=("Silhouette", "mean"),
    NegativeSilhouetteFraction=("Silhouette", lambda s: (s < 0).mean())
)
display(car_profiles.round(3))
```

**Question 9 — Look beyond the overall score**

Describe each group using horsepower and weight. Does a positive overall silhouette guarantee that every car fits its assigned group well? What does a negative silhouette tell you, and what does it not tell you?

<details>
<summary>Solution</summary>

A group can be described by its observed measurements, such as relatively lighter cars with lower horsepower. Do not infer reliability, price, or driver preferences from these two features alone.

An average can hide poorly matched observations. A negative silhouette indicates greater average similarity to a competing cluster under the selected distance. It is not a known ground-truth error label.

</details>

### 4.2 Check sensitivity to initialization

A result is more convincing if it does not change drastically under small, reasonable changes. Start by repeating K-Means with different random seeds while keeping features, K, and the number of initializations fixed.

```python
stability_rows = []
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
for ax, seed in zip(axes, [1, 42, 100]):
    model = KMeans(n_clusters=chosen_k, random_state=seed, n_init=10)
    group_labels = model.fit_predict(X_cars_scaled)
    stability_rows.append({
        "Seed": seed,
        "Inertia": model.inertia_,
        "Silhouette": silhouette_score(X_cars_scaled, group_labels)
    })
    ax.scatter(X_cars[:, 0], X_cars[:, 1], c=group_labels, cmap="tab10", alpha=0.7)
    ax.set_title(f"Seed = {seed}")
    ax.set_xlabel("Horsepower")
axes[0].set_ylabel("Weight (lb)")
plt.tight_layout()
plt.show()
display(pd.DataFrame(stability_rows).round(3))
```

Compare memberships visually, allowing label colors to change. Similar scores alone do not prove identical memberships. With ten initializations, these runs may be very similar; that is useful evidence about this particular sensitivity check.

**Question 10 — What has stability established?**

If the runs agree, have we proved the groups are meaningful? What other changes could we investigate?

<details>
<summary>Solution</summary>

Agreement suggests robustness to these initialization choices. It does not establish natural categories or usefulness. A fuller stability study could vary the sample of observations, slightly perturb measurements, or compare justified feature choices.

</details>

### 4.3 Make a defensible clustering recommendation

Write a short recommendation for someone exploring automobile groups. Include:

- the features and preprocessing used;
- your candidate K and algorithm;
- evidence from metrics, plots, sizes, profiles, and initialization checks;
- a limitation or unresolved question.

The aim is a reasoned recommendation, not a claim that a metric found the one true answer. Finish this clustering analysis before moving to the next dataset and task.

## 5. Detecting unusual customer profiles

### 5.1 Change the question and inspect the data

Clustering asked which observations belong together. **Anomaly detection** asks which observations appear unusual relative to the patterns in the data.

We now use the supplied credit-card dataset. Each row contains **aggregated customer-level usage**, not an individual transaction. A flagged row is an unusual customer profile, not a confirmed fraudulent transaction.

```python
credit_url = (
    "https://raw.githubusercontent.com/AI-Learning-Repo/"
    "Data-Handling/refs/heads/week5/datasets/credit_card.csv"
)
credit_card = pd.read_csv(credit_url)
credit_features = credit_card.drop(columns=["CUST_ID"]).select_dtypes(include="number")
credit_features = credit_features.dropna(axis=1, how="all").copy()

print("Customer rows and selected features:", credit_features.shape)
print("Missing measurements:")
print(credit_features.isna().sum())
display(credit_features.head())

X_credit = credit_features.fillna(credit_features.median())
```

Keep identifiers in `credit_card` for locating observations, but exclude them from fitting. Entirely missing numerical columns are excluded; remaining missing values are filled with feature medians. Imputation is a modeling choice and should be recorded when investigating flagged cases.

For a first view, plot two named features. The model will use all selected numerical features.

```python
plt.figure(figsize=(8, 5))
plt.scatter(X_credit["BALANCE"], X_credit["PURCHASES"], alpha=0.3, s=12)
plt.xlabel("Balance")
plt.ylabel("Purchases")
plt.title("Customer profiles: a two-feature view")
plt.show()
```

**Question 11 — Identify the limits of the plot**

Which combinations look unusual? Can a two-feature plot reveal every unusual profile when the model uses many features?

<details>
<summary>Possible answer</summary>

Describe points that stand out in the displayed balance and purchases measurements. Other customers may be unusual because of features not shown. A two-feature view cannot fully explain every flag from a model fitted on many features.

</details>

### 5.2 Fit an Isolation Forest

Isolation Forest creates random partitions of feature space. Unusual observations often require fewer splits to isolate than observations in common regions. We use its output as a screening result.

Unlike our distance-based clustering methods, it does not require standardization simply to prevent large units from dominating Euclidean distance. We use the imputed numerical features directly.

```python
isolation_model = IsolationForest(contamination=0.02, random_state=42)
credit_flags = isolation_model.fit_predict(X_credit)

credit_results = credit_card.copy()
credit_results["Flag"] = credit_flags
credit_results["NormalityScore"] = isolation_model.score_samples(X_credit)
print(credit_results["Flag"].value_counts())
```

The implementation returns `1` for an inlier and `-1` for a flagged anomaly. For **`score_samples`**, lower values mean more unusual observations; these are not probabilities of fraud. The numeric contamination setting determines a threshold targeting approximately 2% flagged training observations, rather than discovering the true anomaly prevalence. See the [Isolation Forest reference](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).

**Question 12 — Interpret contamination**

If the model flags about 2% of customers, has it discovered that 2% are fraudulent? Explain the role of the setting.

<details>
<summary>Optional LLM hint</summary>

Ask: “Explain the difference between a threshold chosen using an assumed anomaly fraction and evidence that a flagged customer is fraudulent.”

</details>

<details>
<summary>Solution</summary>

No. The setting controls how scores are converted into flags on the fitted data. It does not verify the actual proportion of unusual cases or identify confirmed fraud. The selected features describe customer profiles, and flagged cases need investigation.

</details>

### 5.3 Inspect scores and flagged observations

```python
flagged = credit_flags == -1
plt.figure(figsize=(8, 5))
plt.scatter(
    X_credit.loc[~flagged, "BALANCE"], X_credit.loc[~flagged, "PURCHASES"],
    alpha=0.25, s=12, label="Not flagged"
)
plt.scatter(
    X_credit.loc[flagged, "BALANCE"], X_credit.loc[flagged, "PURCHASES"],
    marker="x", c="red", s=35, label="Flagged profile"
)
plt.xlabel("Balance")
plt.ylabel("Purchases")
plt.title("Isolation Forest flags in a two-feature view")
plt.legend()
plt.show()

display(credit_results.nsmallest(5, "NormalityScore")[[
    "CUST_ID", "BALANCE", "PURCHASES", "CASH_ADVANCE", "Flag", "NormalityScore"
]])
```

The table shows the original measurements, including any original missing values, so they can be reviewed. Fitting used the imputed values in `X_credit`.

**Question 13 — Decide what to do next**

A flagged customer appears close to other customers in the plot. Is that necessarily a mistake? Should flagged rows automatically be removed?

<details>
<summary>Solution</summary>

The model uses more features than the plot shows, so a flag may depend on other measurements or combinations. Check those features, data quality, and imputed values before interpreting the result.

Do not automatically remove a flagged row. It may be a valid unusual profile, an error, or another case requiring context. A model flag does not supply the explanation.

</details>

### 5.4 Investigate the threshold choice

Keep the features and random seed fixed while varying contamination:

```python
threshold_rows = []
for fraction in [0.01, 0.02, 0.05]:
    model = IsolationForest(contamination=fraction, random_state=42)
    flags = model.fit_predict(X_credit)
    threshold_rows.append({
        "Contamination setting": fraction,
        "Flagged customers": int((flags == -1).sum()),
        "Flagged fraction": float((flags == -1).mean())
    })
display(pd.DataFrame(threshold_rows))
```

**Question 14 — Evaluate a screening result**

Why does the count increase as contamination increases? Which setting is best? Without verified case labels, can these counts establish precision or recall?

<details>
<summary>Solution</summary>

Increasing contamination changes the cutoff to flag a larger share of the fitted observations. With the data and random construction held fixed, this is a threshold change, not evidence that more real fraud appeared.

There is no best setting from the counts alone. Selection depends on case review, the costs of missed cases and false alarms, and available review capacity. Precision and recall require trusted labels for what is actually being detected; predicted flags cannot serve as their own ground truth.

</details>

## 6. Consolidation and conceptual review

### 6.1 Compare the questions and outputs

| Task | Question | Outputs to interpret | Evidence still needed |
|---|---|---|---|
| Clustering | Which observations belong together? | Assignments, centers or a hierarchy, profiles | Geometry, stability, and usefulness |
| Anomaly detection | Which observations are unusual? | Scores and threshold-based flags | Case investigation and, when available, verified outcomes |

K-Means assigns even an isolated point to a cluster. Isolation Forest addresses whether a profile is unusual. These outputs answer different questions.

### 6.2 Check your understanding

Answer without writing a complete program:

1. Why must preprocessing be considered before interpreting clustering metrics?
2. Why does inertia alone favor additional clusters?
3. What does silhouette add, and what can its average hide?
4. How can linkage change a hierarchy?
5. Does a stable partition prove meaningful natural groups?
6. What does Isolation Forest try to isolate, and why can unusual observations be easier to isolate?
7. What is the difference between a score, a flag, and a confirmed real-world event?
8. Why should a test dataset use the training scaler rather than its own fitted scaler?

<details>
<summary>Suggested answers</summary>

1. Preprocessing defines the coordinates and distances used by the metrics. Changing units or features changes the comparison.
2. Additional centroids provide more flexibility to reduce within-cluster squared distances.
3. It compares within-cluster similarity with competing clusters. A mean can hide individual observations or groups that fit poorly.
4. Linkage changes how candidate merges are ranked, which can change the merge sequence and final memberships.
5. No. Stability is one useful property, not proof of a useful interpretation.
6. It isolates observations using random splits. Profiles in unusual regions often require fewer splits than those in common regions.
7. A score ranks unusualness, a threshold converts it into a flag, and a confirmed event requires external evidence.
8. Reusing the training transformation preserves the learned coordinate system and avoids fitting preprocessing on evaluation data.

</details>

### 6.3 Explain an analysis from start to finish

Choose one of the two tasks and write a short explanation covering the question, selected features, preprocessing, model, evaluation, and limitations.

For clustering, support your chosen grouping with several kinds of evidence. For anomaly detection, explain how flagged cases would be investigated. Consult the matching [Part 2 theory](part2.md) when revising the concepts.
