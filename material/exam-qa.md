# Potential Exam Questions: Unsupervised Learning

This question bank supports a **closed-book exam focused on concepts and core ideas** from [Part 1](part1.md) and [Part 2](part2.md). It is a collection of possible questions, not a single exam paper. Some questions assess related ideas in different ways; select a balanced subset rather than using them all.

No question requires writing a complete program, recalling installation commands, remembering dataset column names, or reproducing exact notebook output. Short code is supplied where it helps assess understanding. Quantitative questions use small examples; formulas are provided where the aim is interpretation rather than formula recall.

Sections A–E assess the core clustering and anomaly-detection material. Section F checks high-level awareness of the broader techniques covered in the theory. Section G contains integrated alternatives. Suggested answers and marking guidance follow the questions.

## Table of contents

- [A. Foundations and data representation](#a-foundations-and-data-representation)
- [B. K-Means and its outputs](#b-k-means-and-its-outputs)
- [C. Choosing and evaluating clusters](#c-choosing-and-evaluating-clusters)
- [D. Hierarchical clustering and dendrograms](#d-hierarchical-clustering-and-dendrograms)
- [E. Anomaly detection](#e-anomaly-detection)
- [F. Broader techniques: awareness only](#f-broader-techniques-awareness-only)
- [G. Integrated reasoning questions](#g-integrated-reasoning-questions)
- [Suggested answers and marking guidance](#suggested-answers-and-marking-guidance)

## A. Foundations and data representation

### A.1 Supervised versus unsupervised learning

**A1 — Multiple choice**

A flower dataset contains measurements and known species. An analyst excludes species from the inputs, fits K-Means to the measurements, and compares the groups with species afterward. Which statement best describes the fitting process?

A) It is supervised because labels exist in the source dataset.  
B) It is classification because the output contains group numbers.  
C) It is unsupervised because species labels do not guide fitting.  
D) It becomes supervised when the analyst compares the results with species.

**A2 — Identify the task**

For each problem, identify classification, regression, or clustering. State the target, or explain why no target guides fitting.

1. Predict a house's sale price from examples with known prices.
2. Predict whether an email is spam from labeled examples.
3. Group customers by their purchase behavior without supplied customer-group labels.

### A.2 Features, identifiers, and scale

**A3 — Choose meaningful inputs**

A table contains `CustomerID`, age, annual income, and number of purchases.

(a) Why would a customer identifier normally be excluded from distance-based clustering?  
(b) Could adding age to an income-and-purchases clustering change the groups? Explain.  
(c) Does leaving a feature out of one experiment prove that it can never be useful?

**A4 — Explain a change of units**

An analyst changes annual income from thousands of euros to euros before fitting K-Means, without scaling the features. Age remains in years.

(a) Could the groups change even though the customers have not changed? Why?  
(b) What could standardization help with?  
(c) Does standardization guarantee meaningful clusters?

## B. K-Means and its outputs

### B.1 Distance, centroids, and the algorithm

**B1 — Calculate and interpret distance**

For two observations A = `(2,3)` and B = `(5,7)`, use:

$$
d(A,B)=\sqrt{(x_B-x_A)^2+(y_B-y_A)^2}
$$

(a) Calculate the Euclidean distance.  
(b) Explain how K-Means uses distance when assigning an observation.  
(c) Does being close in two selected features imply that the observations are similar in every respect?

**B2 — Complete one K-Means cycle**

Four observations each have one numerical feature: `0, 2, 8, 10`. The initial centroids are C1 = 0 and C2 = 8.

(a) Assign each observation to its nearest initial centroid.  
(b) Calculate the updated centroids.  
(c) Explain why K-Means normally repeats assignment and updating.  
(d) Must a centroid coincide with an observed value?

**B3 — Explain a limitation**

Choose two of the following and explain how each can affect K-Means: overlapping groups, elongated groups, outliers, initialization.

For at least one, explain why receiving a cluster label does not establish an unambiguous natural category.

### B.2 Read outputs without writing a program

**B4 — Interpret supplied code**

Assume X contains 100 observations and two numerical features. You are told that `labels_` contains group assignments and `cluster_centers_` contains center coordinates.

```python
model = KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(X)
labels = model.labels_
centers = model.cluster_centers_
```

(a) How many groups are requested?  
(b) How many group assignments should `labels` contain?  
(c) How many rows and columns should `centers` contain? Explain.  
(d) Is a known target supplied to fitting?  
(e) Why might repeating the fit with different initial centers be useful?

Do not write new code or explain every API argument.

**B5 — Compare group identifiers**

The following assignments refer to the same six observations in the same order:

```text
Run A: 0, 0, 1, 1, 2, 2
Run B: 2, 2, 0, 0, 1, 1
```

(a) Do these runs produce different partitions? Explain.  
(b) Would a label such as `2` automatically mean high-income customers?  
(c) What information would support a meaningful description of that group?

## C. Choosing and evaluating clusters

### C.1 Inertia, elbow, and silhouette

**C1 — Calculate inertia**

Three observations have distances 1, 2, and 3 to their assigned centroids. Inertia is the sum of squared distances to assigned centroids.

(a) Calculate these observations' contribution to inertia.  
(b) Why does increasing K generally reduce inertia on the same data representation?  
(c) Why does this make the smallest inertia insufficient for selecting K?

**C2 — Interpret competing evidence**

All models below use the same observations, features, and preprocessing. The values are illustrative, not notebook output.

| K | Inertia | Mean silhouette |
|---:|---:|---:|
| 1 | 1200 | Not defined |
| 2 | 650 | 0.61 |
| 3 | 330 | 0.55 |
| 4 | 300 | 0.48 |
| 5 | 280 | 0.40 |

(a) Which K has the lowest inertia?  
(b) Which K has the highest silhouette?  
(c) Around which K is there a plausible elbow? Explain using the changes in inertia.  
(d) Recommend one candidate for further investigation. Justify it using the evidence and give one additional check.  
(e) Why is silhouette not defined for K = 1?

**C3 — Understand silhouette**

For an observation, a is its average distance to other points in its own cluster, and b is its smallest average distance to a competing cluster. Use:

$$
s=\frac{b-a}{\max(a,b)}
$$

(a) Calculate s when a = 2 and b = 5. Interpret the sign.  
(b) What would a negative silhouette suggest?  
(c) Does silhouette use only distances to centroids?  
(d) Does a high mean silhouette guarantee that every observation fits its assigned cluster well?

### C.2 Fair comparisons, stability, and interpretation

**C4 — Identify an unfair comparison**

The same dataset gives inertia 900 before standardization and inertia 90 after standardization. An analyst concludes: “Standardization made the clustering ten times better.”

(a) Explain why this conclusion is not justified.  
(b) State one way to investigate the effect of scaling.  
(c) If evaluating unseen observations, should their scaler be fitted separately? Explain.

**C5 — True or false, with reasons**

For each statement, give true or false and explain briefly. Correct false statements.

1. Repeating K-Means with the same data and random seed is sufficient evidence of stability under changes to the data.
2. Different random seeds can produce different K-Means partitions.
3. A stable partition must correspond to meaningful real-world categories.

**C6 — Interpret profiles carefully**

A clustering gives the following summary. These are mean measurements, not values shared by every member.

| Group | Number of cars | Mean horsepower | Mean weight (lb) |
|---|---:|---:|---:|
| A | 80 | 70 | 2000 |
| B | 15 | 160 | 3800 |
| C | 65 | 110 | 2900 |

(a) Describe group B using only the evidence shown.  
(b) Is B automatically a bad cluster because it is small?  
(c) Can you conclude that B contains unreliable cars?  
(d) Give one additional piece of evidence that would help evaluate the grouping.

## D. Hierarchical clustering and dendrograms

### D.1 Understand the procedure and linkage

**D1 — Compare K-Means and agglomerative clustering**

Explain one difference in how they form groups and one difference in the information their outputs provide. What happens to an earlier merge as agglomerative clustering continues?

**D2 — Calculate linkage values**

The distances between every cross-group pair of observations in two clusters are `1, 2, 8, 9`.

(a) Give the single-linkage value.  
(b) Give the complete-linkage value.  
(c) Give the average-linkage value.  
(d) Explain why single linkage can form chains connecting distant regions.  
(e) Does Ward linkage simply select the minimum of these distances? Explain its criterion in words.

### D.2 Interpret a hierarchy

**D3 — Read and cut a dendrogram**

The diagram shows four observations. A and B merge at height 1, C and D merge at height 2, and the two pairs merge at height 8. Heights refer to the chosen linkage criterion; the drawing is schematic.

```text
Height
  8        +-------------------+
           |                   |
  2        |               +---+---+
           |               |       |
  1    +---+---+           |       |
       |       |           |       |
       A       B           C       D
```

(a) List the groups at a horizontal cut of height 1.5.  
(b) List the groups at height 3.  
(c) How many groups remain at height 9?  
(d) Why might the gap between heights 2 and 8 suggest a useful candidate cut?  
(e) Does that gap prove an objectively correct number of groups?

**D4 — Multiple choice**

A truncated dendrogram displays a leaf labeled `(12)`. You are told that parenthesized labels indicate observation counts. What does the label mean?

A) The displayed leaf represents a merged group of 12 observations.  
B) The leaf must be original observation number 12.  
C) The group was merged at a distance of 12.  
D) The model must have exactly 12 final clusters.

**D5 — Distinguish merging from splitting**

Which approach is agglomerative and which is divisive?

1. Start with each observation alone and repeatedly combine groups.
2. Start with every observation in one group and repeatedly divide groups.

Explain which approach was implemented in the activities. No implementation of a divisive algorithm is required.

## E. Anomaly detection

### E.1 Understand Isolation Forest and thresholds

**E1 — Explain the core idea**

Describe how Isolation Forest uses random partitions to identify unusual observations. Why can an unusual observation require fewer splits to isolate than an observation in a common region? How does this question differ from K-Means group assignment?

**E2 — Interpret an assumed anomaly fraction**

An Isolation Forest is fitted to 1,000 customer profiles with numeric contamination set to 0.02. About 20 profiles are flagged.

(a) What role does the contamination setting play?  
(b) Has the model established that 20 customers committed fraud?  
(c) If contamination changes to 0.05 with the data and random construction held fixed, what would you generally expect to happen to the number flagged? Why?  
(d) Must exactly 2% of future profiles be flagged by the original model?

**E3 — Distinguish a score from a probability**

For this question, lower model scores mean more unusual profiles. Customer A scores −0.70 and customer B scores −0.40.

(a) Which customer is ranked as more unusual?  
(b) Is A's score a 70% probability of fraud?  
(c) What else is needed to convert scores into binary flags?

### E.2 Interpret and evaluate flagged cases

**E4 — Investigate a flag**

A model uses 15 numerical customer-profile features. A flagged customer appears close to many unflagged customers in a plot of balance versus purchases.

(a) Is the flag necessarily a mistake? Explain.  
(b) Give two checks you would make before interpreting it.  
(c) Should the row automatically be deleted?  
(d) Why does the fact that each row is an aggregated customer profile matter when reporting the result?

**E5 — Evaluate with independent labels**

This is a hypothetical evaluation dataset, not labels supplied by the lab. Independent review provides trustworthy labels for all 1,000 profiles. Ten are confirmed cases of interest. A model flags 20 profiles, including five of those ten confirmed cases.

Use:

$$
\text{Precision}=\frac{\text{confirmed cases among flags}}{\text{all flags}}
$$

$$
\text{Recall}=\frac{\text{confirmed cases among flags}}{\text{all actual confirmed cases}}
$$

(a) Calculate precision and recall.  
(b) Could you calculate these quantities using the model's flags alone as the correct answers?  
(c) Would simply increasing the number of flags prove that the detector improved?

## F. Broader techniques: awareness only

These questions assess the purposes and limitations of the theory-only topics. They do not require algorithm implementations, PCA derivations, or calculations of association-rule measures.

### F.1 Select and distinguish tasks

**F1 — Match the question to the task**

Choose the most directly relevant task for each objective: clustering, anomaly detection, association rule learning, or dimensionality reduction.

1. Discover groups of customers with similar purchasing behavior.
2. Screen for unusual sensor observations.
3. Find products frequently bought together.
4. Represent observations with 100 numerical features using two coordinates for visualization.

**F2 — Multiple choice, then explain**

PCA transforms a dataset with 20 features into two components. Which statement is most accurate?

A) It has created two customer clusters.  
B) It has selected the two original columns with the clearest names.  
C) It has preserved all information and guaranteed natural groups.  
D) It has created two transformed coordinates that retain selected directions of variation.

Then explain one limitation of judging clusters only from this two-dimensional plot.

### F.2 Interpret other kinds of structure

**F3 — Association and causation**

A supermarket finds that butter appears in many transactions containing bread.

(a) Is this primarily a clustering question or an association-rule question?  
(b) Why can high confidence be uninformative if butter is bought in almost every transaction?  
(c) What comparison does lift add, at a high level?  
(d) Does a strong rule prove that buying bread causes buying butter?

**F4 — Multiple choice**

Which statement correctly distinguishes density estimation and generative modeling?

A) Density estimation assigns known target classes; generative modeling only creates clusters.  
B) Density estimation models where observations are concentrated; generative modeling supports producing new samples. A model can support both.  
C) Density estimation and generative modeling are always mutually exclusive.  
D) A generative model must always be trained without any labels or conditioning information.

## G. Integrated reasoning questions

These longer questions can replace several shorter questions on the same topics.

### G.1 Plan and justify an analysis

**G1 — Customer segmentation**

A company has customer ID, age, annual income, and purchase frequency. It wants to understand useful customer groups and has no predefined segment labels.

Outline an analysis in words. Cover:

(a) the task and input selection;  
(b) preprocessing and the role of scale;  
(c) one suitable starting clustering method and why it is a candidate;  
(d) how to investigate the number or level of groups;  
(e) how to evaluate and interpret the result, including a limitation.

There is no requirement to select a particular K or write code.

**G2 — Choose a method for the purpose**

A researcher has 40 observations and wants to inspect relationships at several nested levels. Another analyst has a very large numerical dataset and wants a manageable partition around group centers.

For each case, suggest one of the two clustering methods taught and justify it. Give one reason why dataset size alone is not enough to make the choice.

### G.2 Critique conclusions

**G3 — Identify unsupported claims**

An analyst reports:

> “K-Means returned five groups, proving five natural customer types. The mean silhouette is 0.61, so every customer is correctly grouped. Isolation Forest flagged 2% of profiles, proving that exactly 2% are fraudulent. We will delete them.”

Identify four unsupported conclusions. For each, explain the problem and state a more defensible interpretation or next step.

## Suggested answers and marking guidance

Award credit for correct reasoning expressed in the student's own words. For calculations, distinguish the method from arithmetic and allow reasonable rounding. For recommendations, accept justified alternatives consistent with the evidence; do not require a single preferred K or an exact phrase from the notes.

For true/false items, require a reason. For multiple-choice items followed by an explanation, assess both parts. The points below identify what a good answer should contain; they are not a fixed mark allocation for a complete paper.

### Answers A: Foundations and representation

<details>
<summary>A1–A4: Show answers</summary>

**A1:** C. Species labels do not guide fitting. Their existence or later use for comparison does not change the information used in the fit.

**A2:**

1. Supervised regression; target: sale price.
2. Supervised classification; target: spam/not spam.
3. Unsupervised clustering; no customer-group target guides fitting. Group assignments are outputs discovered from the features.

**A3:**

(a) Differences between IDs normally have no meaningful relationship to customer similarity.  
(b) Yes. Adding age changes coordinates and distances, potentially changing memberships.  
(c) No. Usefulness depends on the problem and representation; an exclusion in one example is not a universal rule.

**A4:**

(a) Yes. Income differences become 1,000 times larger, increasing their influence relative to age in raw distances.  
(b) Standardization can put numerical variation on comparable scales.  
(c) No. It does not guarantee relevant features, appropriate weighting, or useful groups.

</details>

### Answers B: K-Means

<details>
<summary>B1–B5: Show answers</summary>

**B1:**

```math
d(A,B)=\sqrt{(5-2)^2+(7-3)^2}=\sqrt{9+16}=5
```

K-Means assigns an observation to its nearest current centroid. Similarity is relative to the selected features and units; unmeasured characteristics may differ.

**B2:**

(a) C1 receives 0 and 2; C2 receives 8 and 10.  
(b) The updated centers are:

```math
C_1=\frac{0+2}{2}=1,\qquad C_2=\frac{8+10}{2}=9
```

(c) Moving centers can change the next nearest-centroid assignments. Repetition seeks a stable solution under a stopping criterion. In this example, the assignments remain unchanged after the update.  
(d) No. Neither 1 nor 9 is one of the supplied observations.

**B3:** Accept two explained effects, for example:

- Overlap makes memberships ambiguous even though K-Means assigns every observation.
- Elongated or complex group shapes may be poorly represented by partitions around centers.
- Outliers can shift means and change centers or assignments.
- Different initial centers can lead to different final partitions.

A list without explaining the effects is incomplete. Receiving a label is not proof of a natural group or a uniquely justified assignment.

**B4:**

(a) Three groups.  
(b) 100 assignments, one per fitted observation.  
(c) Three rows and two columns: one centroid per group, with one coordinate per feature.  
(d) No target is supplied.  
(e) Different starts may lead to different solutions; trying several can help find a lower-inertia result, without guaranteeing a global optimum.

**B5:**

(a) No. Observations 1–2, 3–4, and 5–6 remain together in both runs; only their IDs change.  
(b) No. Cluster numbers have no inherent meaning.  
(c) Actual feature values and group profiles, such as income distributions or means, support a description.

</details>

### Answers C: Evaluation

<details>
<summary>C1–C6: Show answers</summary>

**C1:**

```math
I=1^2+2^2+3^2=14
```

Additional centroids give more flexibility to represent points closely. One centroid at each observation can produce zero inertia but little useful summarization. Lower inertia alone therefore does not balance compactness against a useful number of clusters.

**C2:**

(a) K = 5, inertia 280.  
(b) K = 2, silhouette 0.61.  
(c) K = 3 is a plausible elbow: inertia decreases by 550, then 320, but subsequent improvements are only 30 and 20.  
(d) Accept K = 2 supported by silhouette or K = 3 supported by the elbow, with recognition of the tradeoff and an additional check. Useful checks include plots, sizes, profiles, stability, and application relevance. A claim that either value is proven correct is not justified.  
(e) There is no competing cluster with which to compare within-cluster distances.

**C3:**

```math
s=\frac{5-2}{\max(2,5)}=\frac{3}{5}=0.6
```

(a) It is positive because the point is closer on average to its own group than to the nearest competing group.  
(b) A competing group is closer on average under the chosen distance; this is not a verified ground-truth error.  
(c) No. It uses average point-to-point distances.  
(d) No. A high average can hide poorly matched observations or small groups.

**C4:**

(a) The two inertia values use different coordinate scales, so their ratio is not a quality-improvement factor.  
(b) Compare memberships and profiles, inspect plots, or investigate appropriate diagnostics while being explicit about the changed representation.  
(c) No. Fit preprocessing on the training/reference data and reuse it, preserving the coordinate system and avoiding fitting on held-out evaluation data.

**C5:**

1. False. An identical setup tests reproducibility; stability requires investigating changes such as initialization or data perturbations.
2. True. K-Means can reach different solutions from different initial centers.
3. False. A method can consistently produce a partition that is not meaningful for the application.

**C6:**

(a) B is a smaller group with relatively high mean horsepower and weight.  
(b) No. It may represent a valid small segment or something requiring further investigation.  
(c) No. Reliability is not measured in this table.  
(d) Accept a relevant check such as within-group variation, silhouette values, plots, sensitivity checks, or domain usefulness.

</details>

### Answers D: Hierarchical clustering

<details>
<summary>D1–D5: Show answers</summary>

**D1:** K-Means alternates individual assignments and centroid updates. Agglomerative clustering repeatedly merges groups using linkage. K-Means provides a chosen partition and centers; hierarchical clustering also represents nested relationships that can be examined at different cuts. Earlier agglomerative merges are not undone.

**D2:**

(a) Single: 1.  
(b) Complete: 9.  
(c) Average:

```math
\frac{1+2+8+9}{4}=5
```

(d) A close cross-group pair can connect groups; repeated local connections can form long chains.  
(e) No. Ward chooses the merge with the smallest increase in within-cluster sum of squares.

**D3:**

(a) Three groups: `{A,B}`, `{C}`, `{D}`.  
(b) Two groups: `{A,B}`, `{C,D}`.  
(c) One group: `{A,B,C,D}`.  
(d) The pairs form at low heights, while combining them requires a much larger linkage value. A cut in the gap preserves the pairs.  
(e) No. It suggests a candidate based on the hierarchy, which still needs evaluation and interpretation.

**D4:** A. In this truncated display, the leaf stands for a merged group containing 12 observations.

**D5:** Procedure 1 is agglomerative; procedure 2 is divisive. The activities implement agglomerative clustering. Both can represent nested groups, but they build the hierarchy in opposite directions.

</details>

### Answers E: Anomaly detection

<details>
<summary>E1–E5: Show answers</summary>

**E1:** Isolation Forest builds random partitions and uses how easily observations are isolated across trees. Observations in unusual regions often require fewer splits than those surrounded by many similar observations. K-Means asks which centroid-based group receives a point; anomaly detection asks how unusual the observation is.

**E2:**

(a) The numeric setting determines a score threshold targeting approximately 2% flagged training observations.  
(b) No. These are unusual profiles under a model, not independently confirmed fraud.  
(c) Generally more profiles, approximately 50 in the fitted data. The cutoff changes to flag a larger fraction; this is not evidence that more real cases appeared.  
(d) No. Future data may have a different distribution, so their flagged fraction need not equal the training threshold assumption.

**E3:**

(a) A, because −0.70 is lower than −0.40 under the stated convention.  
(b) No. The score is not a calibrated fraud probability.  
(c) A decision threshold is needed to convert scores into flags.

**E4:**

(a) No. The model uses features not shown in the two-dimensional plot.  
(b) Accept two distinct checks: inspect the other feature values, review original missing values and imputation, check measurement quality, investigate relevant customer context, or inspect threshold sensitivity.  
(c) No. An unusual observation can be valid and important.  
(d) The result concerns unusual aggregated customer profiles; it does not identify or confirm an individual fraudulent transaction.

**E5:**

```math
\text{Precision}=\frac{5}{20}=0.25=25\%
```

```math
\text{Recall}=\frac{5}{10}=0.50=50\%
```

(b) No. Evaluation requires independent labels for the actual cases of interest; predicted flags cannot be their own ground truth.  
(c) No. More flags may include additional false alarms. Judge the tradeoff using verified outcomes and the purpose of screening.

</details>

### Answers F: Broader techniques

<details>
<summary>F1–F4: Show answers</summary>

**F1:**

1. Clustering.
2. Anomaly detection.
3. Association rule learning.
4. Dimensionality reduction.

**F2:** D. PCA creates transformed coordinates. Reduction may discard information, and visible separation in a projection does not prove meaningful groups in the original feature space.

**F3:**

(a) Association-rule learning.  
(b) Butter may appear often with bread simply because it appears in almost all transactions.  
(c) Lift compares co-occurrence with what would be expected from the items' individual frequencies under independence.  
(d) No. Co-occurrence does not establish causation; common preferences or shopping occasions may explain it.

**F4:** B. Density estimation and sample generation have different objectives, but one distribution model can support both. Generative models can also use labels or conditioning, so generative does not necessarily mean unsupervised.

</details>

### Answers G: Integrated reasoning

<details>
<summary>G1–G3: Show answers and assessment guidance</summary>

**G1:** A strong answer includes:

- Unsupervised clustering to investigate groups without predefined segment targets.
- Exclusion of ID from distances and a reasoned selection of meaningful measurements.
- Inspection of missing values and numerical scales; justified preprocessing.
- A suitable starting method, such as K-Means for numerical groups around centers or hierarchical clustering for nested relationships.
- Investigation of group count through appropriate evidence, such as elbow and silhouette for K-Means or candidate dendrogram cuts for a hierarchy.
- Evaluation through plots, sizes, profiles, and stability, followed by interpretation tied to the business question.
- A relevant limitation, such as scale sensitivity, overlapping groups, initialization, linkage, or uncertain application usefulness.

Accept alternative defensible choices. Do not require a particular K. Merely naming an algorithm without explaining the decisions is incomplete.

**G2:** Agglomerative hierarchical clustering is a reasonable candidate for the small research dataset because a dendrogram reveals nested relationships. K-Means is a reasonable starting candidate for a large numerical dataset when a partition around centers is useful. Size alone is insufficient: feature representation, cluster geometry, distance/linkage assumptions, and the purpose also matter.

**G3:** Assess these four corrections:

1. **Five returned groups do not prove five natural types.** K-Means was instructed to form five groups. Investigate candidate counts and application relevance.
2. **A mean silhouette of 0.61 does not establish every assignment as correct.** It summarizes geometric fit and may hide negative individual values. Inspect per-group and per-observation results.
3. **A 2% flagged fraction does not prove 2% fraud.** Flags depend on the model and threshold. Independent evidence is required to establish real events.
4. **Automatic deletion is not justified.** Investigate original measurements, data quality, and context; flagged profiles may be valid and informative.

Credit explanations connecting each correction to the relevant concept, rather than a bare list of “false” statements.

</details>
