# Classification Metrics: Why Accuracy Is Not Enough

Suppose we build a classification model that predicts whether an email is **Spam** or **Not Spam**.

We have 100 emails:

* 90 are Not Spam
* 10 are Spam

---

## 1. Start with Accuracy

The simplest question we can ask is:

> **How many predictions did the model get correct?**

Suppose the model makes these predictions:

* 85 normal emails correctly classified as Not Spam
* 5 normal emails incorrectly classified as Spam
* 8 spam emails correctly classified as Spam
* 2 spam emails incorrectly classified as Not Spam

So the model gets:

$$
85 + 8 = 93
$$

predictions correct out of 100.

Therefore:

$$
Accuracy = \frac{93}{100} = 93\%
$$

So we might initially say:

> "The model has 93% accuracy. That sounds quite good."

But accuracy doesn't tell us **what kinds of mistakes** the model is making.

---

# 2. Why Accuracy Is Not Always Enough

Consider a different situation.

Suppose we have **1,000 patients**:

* 990 do not have a disease
* 10 have the disease

Now imagine a model that predicts:

> **Everyone is healthy.**

The model correctly identifies all 990 healthy people.

Therefore:

$$
Accuracy = \frac{990}{1000}=99\%
$$

The model has **99% accuracy**.

But it identifies **none of the 10 people who have the disease**.

So we have a problem.

The accuracy looks excellent, but the model is completely failing at identifying the cases we may care about most.

This is especially common when the classes are **imbalanced**.

For example:

* 990 negative cases
* 10 positive cases

A model can get very high accuracy simply by predicting the majority class most of the time.

So we need more information.

---

# 3. Recall: How Many Positive Cases Did We Find?

Now introduce **recall**.

Recall asks:

> **Of all the cases that were actually positive, how many did the model find?**

For example, suppose there are 10 patients who actually have the disease.

The model correctly identifies 8 of them but misses 2.

Then:

$$
Recall = \frac{8}{8+2}
$$

$$
Recall = 80\%
$$

So the model has **80% recall**.

We can interpret this as:

> "The model found 80% of the actual positive cases."

### Why is recall useful?

Recall is particularly important when **missing a positive case is costly or dangerous**.

Examples:

* Disease detection
* Fraud detection
* Detecting defective products
* Security threat detection

In these situations, we may want to avoid **false negatives**.

---

# 4. False Negatives

A **false negative** happens when:

> The actual class is positive, but the model predicts negative.

For example:

**Patient actually has disease → Model says no disease**

This is a false negative.

Recall helps us understand how often we are successfully finding the positive cases rather than missing them.

A useful way to remember it:

> **Recall = How many of the actual positives did we find?**

---

# 5. Precision: Can We Trust Positive Predictions?

Now consider the opposite question.

Suppose our model says:

> "These 10 emails are spam."

How many of those 10 are actually spam?

This is **precision**.

Suppose:

* 8 really are spam
* 2 are actually normal emails

Then:

$$
Precision = \frac{8}{8+2}
$$

$$
Precision = 80\%
$$

So:

> **When the model predicts Spam, it is correct 80% of the time.**

Precision is therefore concerned with **false positives**.

A false positive happens when:

> The actual class is negative, but the model predicts positive.

For example:

**Normal email → Model says spam**

---

# 6. Precision and Recall Answer Different Questions

At this point, we have two different questions.

### Recall

> **Of all the actual positives, how many did we find?**

### Precision

> **Of everything we predicted as positive, how many were actually positive?**

Consider a disease detection model.

### High recall, low precision

The model finds almost everyone who has the disease.

But it also tells many healthy people that they might have the disease.

So:

* Few actual cases are missed
* Many false alarms occur

### High precision, low recall

The model is very careful before predicting disease.

When it predicts disease, it is usually correct.

But it misses many people who actually have the disease.

So:

* Few false alarms
* Many actual cases are missed

Neither situation is automatically better. It depends on the application.

---

# 7. Confusion Matrix

Now that we have discussed the different types of predictions, we can organize them into a **confusion matrix**.

For a binary classification problem:

|                        |     Actual Positive |     Actual Negative |
| ---------------------- | ------------------: | ------------------: |
| **Predicted Positive** |  True Positive (TP) | False Positive (FP) |
| **Predicted Negative** | False Negative (FN) |  True Negative (TN) |

Let's use our spam example.

Suppose we have 100 emails:

|                        | Actual Spam | Actual Not Spam |
| ---------------------- | ----------: | --------------: |
| **Predicted Spam**     |           8 |               2 |
| **Predicted Not Spam** |           2 |              88 |

Now we can identify:

* **TP = 8** → Spam correctly identified as spam
* **TN = 88** → Normal correctly identified as normal
* **FP = 2** → Normal incorrectly classified as spam
* **FN = 2** → Spam incorrectly classified as normal

The confusion matrix is useful because it shows **what the model is getting right and what it is getting wrong**.

---

# 8. All the Metrics Come From the Confusion Matrix

Once we have TP, TN, FP, and FN, we can calculate our metrics.

### Accuracy

Overall correctness:

$$
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
$$

For our example:

$$
Accuracy = \frac{8+88}{100}=96\%
$$

---

### Recall

How many actual positives did we find?

$$
Recall = \frac{TP}{TP+FN}
$$

$$
Recall = \frac{8}{8+2}=80\%
$$

---

### Precision

How many predicted positives were actually positive?

$$
Precision = \frac{TP}{TP+FP}
$$

$$
Precision = \frac{8}{8+2}=80\%
$$

So we now have:

| Metric    |  Result |
| --------- | ------: |
| Accuracy  | **96%** |
| Precision | **80%** |
| Recall    | **80%** |

Notice that the **96% accuracy** doesn't tell us that the model has **80% precision and 80% recall**.

That is why reporting accuracy alone can be misleading.

---

# 9. F1 Score

Sometimes we want one number that summarizes the balance between **precision and recall**.

This is where the **F1 score** is useful.

The formula is:

$$
F1 = 2 \times
\frac{Precision \times Recall}
{Precision + Recall}
$$

In our example:

$$
Precision = 80\%
$$

$$
Recall = 80\%
$$

Therefore:

$$
F1 = 80\%
$$

The important intuition is:

> **F1 is high when both precision and recall are high.**

For example:

| Precision | Recall |   F1 |
| --------: | -----: | ---: |
|       90% |    90% |  90% |
|       90% |    50% | ~64% |
|       50% |    90% | ~64% |
|       10% |    90% | ~18% |

So a model cannot get a high F1 score simply by having excellent precision while having very poor recall, or vice versa.

---

# 10. Putting Everything Together

The progression can be presented to the class like this:

### Step 1 — Accuracy

> **How many predictions are correct overall?**

Useful, but can be misleading with imbalanced classes.

↓

### Step 2 — Recall

> **Of all the actual positive cases, how many did we find?**

Useful when missing positive cases is important.

↓

### Step 3 — Precision

> **Of all the cases we predicted as positive, how many were actually positive?**

Useful when false positives are important.

↓

### Step 4 — Confusion Matrix

> **What types of correct and incorrect predictions are we making?**

Shows:

**TP, TN, FP, FN**

↓

### Step 5 — F1 Score

> **How well are we balancing precision and recall?**

Combines precision and recall into a single measure.

---

## A simple summary for the class

| Metric               | Main question                                        |
| -------------------- | ---------------------------------------------------- |
| **Accuracy**         | How often is the model correct overall?              |
| **Recall**           | How many actual positives did we find?               |
| **Precision**        | How many predicted positives were actually positive? |
| **Confusion Matrix** | What types of mistakes is the model making?          |
| **F1**               | How well are we balancing precision and recall?      |

The main lesson is:

> **Accuracy tells us how often the model is correct overall, but it does not tell us what kind of errors the model is making. Precision, recall, and the confusion matrix give us a much clearer picture of classification performance.**

---

## Links

- [F1 Score in Machine Learning](https://www.geeksforgeeks.org/machine-learning/f1-score-in-machine-learning/)