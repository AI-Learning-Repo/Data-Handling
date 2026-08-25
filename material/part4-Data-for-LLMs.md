# Data and Datasets for LLMs

## Table of Contents

1. [Introduction to Dataset Engineering](#1-introduction-to-dataset-engineering)
2. [Data-Centric vs. Model-Centric AI](#2-data-centric-vs-model-centric-ai)
3. [Data Curation: Quality, Coverage, and Quantity](#3-data-curation-quality-coverage-and-quantity)
4. [Data Acquisition and Annotation](#4-data-acquisition-and-annotation)
5. [Data Augmentation and Synthesis](#5-data-augmentation-and-synthesis)
6. [Limitations of AI-Generated Data](#6-limitations-of-ai-generated-data)
7. [Model Distillation](#7-model-distillation)
8. [Data Processing Pipeline](#8-data-processing-pipeline)

---

## 1. Introduction to Dataset Engineering

**Dataset engineering** is the process of designing, collecting, curating, processing, and maintaining datasets for training and evaluating AI systems. The objective is not simply to collect as much data as possible, but to develop data that is appropriate for the task, sufficiently representative, high quality, and practical to obtain and maintain.

For modern AI systems, dataset engineering can involve data scientists, machine-learning engineers, domain experts, annotators, data engineers, and evaluation specialists.

Data requirements also differ substantially between the different stages of model development.

> **Pre-training vs. Post-training**
>
> * **Pre-training** is the stage in which a model is trained on very large datasets to learn general patterns in language, code, images, or other modalities. For language models, the data can contain billions or trillions of tokens.
>
> * **Post-training** refers to the set of techniques used to adapt a pretrained model to follow instructions, perform particular tasks, use tools, improve reasoning or other capabilities, and behave according to desired requirements. Post-training can involve supervised fine-tuning (SFT), preference optimization, reinforcement learning, and other methods.
>
> This chapter focuses primarily on the **data used for post-training and evaluation**, while also discussing data used for pre-training and synthetic-data generation.

---

## 2. Data-Centric vs. Model-Centric AI

AI systems can be improved from both the **model side** and the **data side**.

### Model-Centric AI

**Model-centric AI** emphasizes improvements to the model, learning algorithm, optimization process, or inference procedure while treating the available dataset as relatively fixed.

Examples include:

* changing the model architecture,
* modifying the loss function,
* changing optimization methods,
* tuning hyperparameters,
* changing training procedures,
* improving inference strategies.

A classic example is the development of increasingly effective neural-network architectures on established benchmark datasets such as ImageNet.

### Data-Centric AI

**Data-centric AI** emphasizes systematic improvements to the data used to train, fine-tune, and evaluate a model.

Examples include:

* correcting incorrect labels,
* improving annotation guidelines,
* removing duplicates,
* filtering low-quality examples,
* increasing coverage of important cases,
* improving data diversity,
* balancing underrepresented categories,
* adding domain-specific examples,
* improving evaluation datasets.

For example, a team might keep the architecture of an open-weight model such as Llama relatively unchanged while experimenting with different training or fine-tuning datasets to improve performance on a particular domain or task.

### Both Approaches Are Important

Data-centric and model-centric development are **complementary rather than mutually exclusive**.

In practice, AI teams often iterate through a cycle such as:

**Model → Data → Evaluation → Error Analysis → Model/Data Improvement**

The appropriate balance depends on the problem, the available resources, and the limitations identified during evaluation.

---

## 3. Data Curation: Quality, Coverage, and Quantity

**Data curation** is the process of selecting, organizing, filtering, improving, and validating data for a particular purpose.

For LLMs, curation can involve both adding useful examples and removing examples that are irrelevant, incorrect, duplicated, unsafe, or otherwise unsuitable.

For example, suppose a conversational model frequently produces unnecessarily critical or unsolicited comments about a user's writing style. If such behavior is associated with particular training examples, the dataset can be reviewed to determine whether those examples should be removed, corrected, or replaced with examples demonstrating the desired behavior.

The goal of curation is to ensure that the dataset provides useful signals for the capabilities and behaviors that the model is expected to learn.

Three important dimensions are:

1. **Quality**
2. **Coverage**
3. **Quantity**

---

### 3.1 Reasoning Data

Some LLM training and post-training datasets contain examples designed to improve problem-solving and reasoning capabilities.

These examples may include:

* mathematical problems,
* programming problems,
* scientific questions,
* multi-step decision tasks,
* worked solutions,
* intermediate reasoning or verification steps.

For example:

**Question:**
The cafeteria has 23 apples. It uses 20 apples for lunch and then buys 6 more. How many apples does it have?

**Solution:**
23 − 20 = 3
3 + 6 = 9

Such examples can provide a training signal for multi-step problem solving.

However, there are many different approaches to training reasoning capabilities. A dataset containing explicit intermediate reasoning is only one possible approach, and the internal reasoning of a deployed model does not necessarily need to be exposed to the user.

---

### 3.2 Tool-Use Data

LLMs can also be trained or post-trained to interact with external tools such as:

* search systems,
* calculators,
* databases,
* code interpreters,
* APIs,
* software applications.

Tool-use data can contain examples showing:

1. the user's request,
2. the decision to use a particular tool,
3. the tool arguments,
4. the tool result,
5. the model's subsequent response.

For example:

```text
User: What is the current exchange rate?

Model: Use currency tool.

Tool arguments:
{"from": "EUR", "to": "USD"}

Tool result:
1 EUR = ...

Model:
The current rate is ...
```

The exact representation depends on the model and the tool-calling framework.

---

### 3.3 Data Quality

Data quality refers to whether examples are accurate, relevant, consistent, and suitable for the intended task.

A smaller, carefully curated dataset can sometimes be more useful than a much larger dataset containing substantial noise. However, there is **no universal rule** that a smaller dataset will always outperform a larger one.

The value of additional data depends on:

* its quality,
* its relevance,
* its diversity,
* its relationship to the target task,
* the model being trained,
* the training method,
* the evaluation methodology.

#### Example: LIMA

The LIMA study demonstrated that a relatively small set of carefully selected instruction-response examples could substantially improve the instruction-following behavior of a pretrained language model.

The broader lesson is not that "1,000 examples are enough," but that **carefully selected post-training data can have a substantial effect on model behavior**.

---

### Characteristics of High-Quality Data

High-quality datasets commonly have several characteristics:

1. **Relevant:** The examples represent the task or capability being targeted.

2. **Correct:** The information, labels, or expected responses are accurate.

3. **Consistent:** Similar cases are treated according to consistent criteria.

4. **Representative:** The dataset reflects the important cases and populations encountered in the intended application.

5. **Diverse where necessary:** The dataset includes meaningful variations in language, users, tasks, contexts, and edge cases.

6. **Well-formatted:** The data is stored in a format appropriate for the intended training or evaluation process.

7. **Deduplicated:** Unnecessary duplicates are identified and handled appropriately.

8. **Privacy- and license-aware:** Data is collected and processed in accordance with applicable privacy, licensing, and usage requirements.

---

### 3.4 Data Coverage

**Data coverage** describes how well a dataset represents the situations that the model is expected to encounter.

Coverage can involve:

* different tasks,
* different domains,
* different languages,
* different user groups,
* different writing styles,
* different levels of difficulty,
* common cases,
* rare cases,
* edge cases.

For example, a multilingual e-commerce assistant may need examples covering different languages, cultural contexts, product categories, and styles of user interaction.

If users frequently write short messages, long messages, messages containing spelling errors, or messages containing code, the evaluation and training data should reflect these variations when they are relevant to the application.

### Example: Mathematics and Coding Data

Large language model developers have reported deliberately including high-quality mathematics and coding data in training mixtures.

This illustrates an important principle:

> **The composition of training data can influence capabilities beyond the narrow subject represented by the data.**

However, the relationship between a particular data source and a particular capability is complex and depends on the model, training method, and other data in the training mixture.

---

### 3.5 Data Quantity

The amount of data required varies substantially according to the task and training method.

For example:

* **Supervised Fine-Tuning (SFT):** A model is trained on examples such as instruction-response pairs. The required number of examples can range from relatively small datasets to very large datasets depending on the task.

* **Parameter-Efficient Fine-Tuning (PEFT):** Methods such as LoRA update only a relatively small number of trainable parameters while keeping most of the pretrained model fixed. PEFT can reduce memory and computational requirements compared with full fine-tuning.

* **Full Fine-Tuning:** A larger portion or all of the model's parameters may be updated. This can require substantially more computational resources.

There is no fixed number of examples that guarantees success. Dataset size should be determined through experimentation and evaluation.

---

### Diminishing Returns and Multi-Stage Training

Adding more examples does not necessarily produce proportional improvements.

For example, the first set of high-quality examples may address major weaknesses in a model, while later examples may cover cases that the model already handles well.

This can produce **diminishing returns**.

Training can also be performed in multiple stages. For example:

1. train or adapt a model using a broad dataset,
2. identify the target domain,
3. use a smaller domain-specific dataset for additional adaptation,
4. evaluate the resulting model.

For example, a model might first learn general sentiment-related patterns from a broad collection of text and then be adapted using a smaller dataset of product reviews.

The important principle is:

> **Dataset composition and training order can matter in addition to dataset size.**

---

### Start With a Small Pilot Dataset

Before investing heavily in dataset creation, it is often useful to create a small, high-quality pilot dataset.

The pilot can help determine:

* whether the task is clearly defined,
* whether annotation guidelines are understandable,
* whether the model can learn the intended behavior,
* whether the evaluation method is appropriate,
* whether additional data is likely to address the observed errors.

If a small, carefully designed experiment produces no measurable improvement, it is worth investigating the training setup, task definition, data quality, or evaluation procedure before simply collecting more data.

---

## 4. Data Acquisition and Annotation

The goal of data acquisition is to obtain data that is relevant, sufficiently diverse, high quality, and legally and ethically appropriate for the intended use.

Potential sources include:

* public datasets,
* licensed datasets,
* internally generated data,
* expert annotations,
* user interactions,
* synthetic data,
* simulated environments,
* human demonstrations.

### Application Data and the Data Flywheel

Data generated through an AI application can become an important source of information for future development.

For example:

**Users → Application → Feedback/Interactions → Error Analysis → New Data → Training/Evaluation → Improved Application**

This is sometimes described as a **data flywheel**.

However, application data is not automatically suitable for training. Organizations need to consider:

* user consent,
* privacy,
* data retention,
* licensing,
* representativeness,
* data quality,
* security,
* applicable regulations.

---

### Combining Multiple Data Sources

Dataset creation often combines several sources rather than relying on one dataset.

For example, a team might construct an 11,000-example dataset as follows:

1. **Source:** Start with an open dataset containing 10,000 examples.

2. **Filter:** Remove 1,000 irrelevant or unusable examples.

3. **Audit:** Identify examples whose answers are incorrect or otherwise unsuitable.

4. **Human annotation:** Have qualified annotators correct or replace problematic responses.

5. **Identify gaps:** Determine which important topics or task types are underrepresented.

6. **Synthetic generation:** Generate additional examples to address those gaps.

7. **Validation:** Have humans or automated procedures review the generated examples.

The resulting dataset is not simply "collected." It has been **constructed through an iterative engineering process**.

---

### Annotation Guidelines

One of the most difficult parts of human annotation is often defining what constitutes a high-quality response.

For example, annotators may need to decide:

* Is the response factually correct?
* Is it relevant?
* Is it sufficiently complete?
* Is it unnecessarily verbose?
* Does it follow the user's instructions?
* Is it safe?
* How should borderline cases be labeled?

Clear annotation guidelines are important because inconsistent labels can introduce noise into the training data.

Good annotation processes therefore typically include:

* detailed guidelines,
* examples,
* edge cases,
* annotator training,
* quality checks,
* disagreement analysis,
* periodic revision of the guidelines.

---

### Public Dataset Resources

Before creating a dataset from scratch, it is often useful to investigate existing datasets.

Examples include:

1. **Hugging Face Datasets** — public and community datasets for machine learning and AI.

2. **Kaggle Datasets** — datasets covering many machine-learning and data-science applications.

3. **Google Dataset Search** — a search engine for discovering datasets.

4. **Data.gov** — U.S. government open-data resources.

5. **ICPSR** — research datasets, particularly in the social sciences.

6. **UCI Machine Learning Repository** — widely used datasets for machine learning research and education.

7. **OpenML** — a platform for sharing datasets and machine-learning experiments.

8. **AWS Open Data** — datasets made available through Amazon Web Services.

9. **TensorFlow Datasets** — datasets prepared for use with TensorFlow and other ML workflows.

10. **Stanford Large Network Dataset Collection (SNAP)** — datasets for graph and network research.

Always verify the **license, provenance, quality, and permitted uses** of a dataset before using it.

---

## 5. Data Augmentation and Synthesis

When suitable human-generated data is limited or expensive to obtain, additional examples can sometimes be produced through **data augmentation** or **synthetic data generation**.

### Data Augmentation

**Data augmentation** modifies existing data to create additional training examples while attempting to preserve the relevant information.

Examples include:

* flipping or cropping images,
* changing image brightness,
* introducing controlled noise,
* paraphrasing text,
* translating text,
* perturbing numerical values.

The appropriate augmentation method depends on the task. An augmentation is useful only if the transformation preserves the properties that the model needs to learn.

### Synthetic Data

**Synthetic data** is data generated artificially rather than directly collected from the target real-world population.

It can be used to:

* increase dataset size,
* cover rare cases,
* create examples for difficult tasks,
* simulate environments,
* reduce exposure of sensitive information,
* generate training examples when human data is expensive.

Synthetic data should still be evaluated for correctness, diversity, bias, and suitability for the target task.

---

### 5.1 Rule-Based Generation and Perturbation

Before modern generative models, synthetic data was often generated using rules, templates, simulations, and controlled perturbations.

For example, a fraud-detection dataset might use simulated transactions with different:

* merchants,
* transaction amounts,
* dates,
* locations,
* transaction patterns.

Image datasets can also be augmented by changing orientation, scale, lighting, or adding controlled noise.

The key requirement is that the transformation should preserve or intentionally modify the target label in a predictable way.

---

### 5.2 Simulation

Simulation can generate training data for situations that are expensive, dangerous, or difficult to reproduce in the physical world.

Examples include:

* autonomous-driving environments,
* robotics,
* industrial processes,
* physics-based environments,
* game-playing agents.

A simulated environment can generate many interactions at relatively low cost.

However, simulated data may differ from real-world data. This is sometimes described as the **sim-to-real gap**.

---

### 5.3 AI-Powered Data Synthesis

Modern generative models can also be used to generate training data.

Examples include:

* generating instruction-response pairs,
* generating coding problems,
* generating synthetic conversations,
* generating tool-use trajectories,
* generating variations of existing examples,
* generating examples for rare or difficult cases.

AI-generated data can increase scale, but it should normally be accompanied by appropriate **verification and filtering**.

---

### 5.4 Self-Play and Environment Interaction

An AI system can sometimes generate training data through interaction with an environment or through self-play.

For example, game-playing systems can play against copies of themselves or against other agents, producing large numbers of training experiences.

OpenAI's Dota 2 system is a well-known example of large-scale self-play. The system generated enormous amounts of gameplay experience through simulated matches.

The broader principle is:

> **When an environment can be simulated cheaply, interaction with the environment can become a source of training data.**

---

### 5.5 Instruction Generation

A powerful language model can be used to generate additional instruction examples from a smaller set of seed examples.

A typical process is:

**Human-written seed examples → Generative model → Candidate instructions/responses → Filtering and validation**

The Stanford Alpaca project is an early and influential example of this approach. The project used a set of instruction examples and a language model to generate a larger instruction-following dataset.

The important lesson is that **small amounts of human-created data can sometimes be used to guide large-scale synthetic-data generation**, but the generated examples still require validation.

---

### 5.6 Reverse Instruction Generation

Another approach starts with a trusted source rather than an invented answer.

For example:

1. Begin with an existing document or answer.
2. Ask a language model to generate a question that could lead to that answer.
3. Validate the resulting question-answer pair.
4. Add the pair to the dataset if it satisfies the required criteria.

Using a trusted source can reduce the risk of introducing unsupported facts, but it does **not guarantee** that the generated example is correct or useful.

---

### 5.7 Translation and Back-Translation

Translation can be used to create multilingual training data.

For example:

**English → Target language → English**

This can produce additional examples while preserving some of the original semantic content.

However, machine translation can introduce:

* incorrect terminology,
* grammatical errors,
* cultural problems,
* loss of meaning,
* unnatural phrasing.

Therefore, translated data should be evaluated, particularly for specialized or low-resource languages.

---

### Example: Synthetic Coding Data

Coding is an attractive area for synthetic-data generation because some properties can be automatically tested.

A possible pipeline is:

1. Generate a programming problem.
2. Generate a candidate solution.
3. Generate or provide test cases.
4. Execute the code.
5. Reject solutions that fail the tests.
6. Optionally generate explanations or documentation.
7. Perform additional validation.

This approach illustrates an important principle:

> **Synthetic data becomes more useful when the generated examples can be checked against objective criteria.**

---

### 5.8 Data Verification

Synthetic data should be verified before being treated as high-quality training data.

Possible verification methods include:

#### Functional verification

For code or mathematical problems, automatically check whether the generated solution satisfies known tests or constraints.

Passing automated tests provides evidence of correctness, but does not guarantee correctness for every possible situation.

#### Human evaluation

Experts or trained annotators can evaluate examples where correctness or quality cannot easily be verified automatically.

#### Model-based evaluation

Another AI model can be used as an evaluator or **judge** to score candidate examples.

This can be useful for large datasets, but model-based evaluation has limitations, including:

* evaluator bias,
* inconsistent judgments,
* sensitivity to presentation,
* difficulty detecting subtle errors.

For important applications, model-based evaluation should therefore be combined with other validation methods where appropriate.

---

## 6. Limitations of AI-Generated Data

Synthetic data can reduce the cost and time required to create training examples, but it does not eliminate the need for high-quality human-generated or independently sourced data.

Several limitations are important.

### 6.1 Quality Control

If the model generating synthetic data makes an error, that error may be reproduced in the resulting dataset.

This creates a version of the familiar principle:

> **Poor-quality generated data can produce poor-quality training signals.**

The risk is especially important when the generated data cannot be automatically verified.

---

### 6.2 Superficial Imitation

A model trained on generated examples may reproduce the patterns and style of the teacher model without necessarily acquiring the intended underlying capability.

For example, a teacher model may generate apparently convincing mathematical explanations. A student model trained on those explanations may learn the format and language of mathematical reasoning while still making mathematical errors on new problems.

This is one reason evaluation on **new, independently constructed problems** is important.

---

### 6.3 Model Collapse and Distributional Degradation

Repeatedly generating data with AI systems and then training new models primarily on that generated data can, under some conditions, reduce diversity and distort the data distribution.

This phenomenon is often discussed as **model collapse**.

One concern is that rare or unusual examples can become increasingly underrepresented when generated data is repeatedly filtered toward highly probable patterns.

The risk depends on:

* the amount of synthetic data,
* how it is generated,
* how it is filtered,
* how much original data remains in the mixture,
* the diversity of the source data,
* the training process.

Maintaining high-quality, independently sourced data can help preserve diversity and coverage.

---

### 6.4 Data Provenance and Licensing

Synthetic data also raises questions about **data provenance**.

Important questions include:

* What data was used to create the generating model?
* What model generated the synthetic examples?
* What license applies to the source data?
* What terms apply to the generating model?
* Can the generated data legally be used for the intended purpose?
* Could the generated data reproduce copyrighted or sensitive material?

These questions should be addressed through appropriate legal and data-governance processes rather than assuming that synthetic data is automatically free of licensing or privacy concerns.

---

### 6.5 Benchmark Contamination

Synthetic data can also create evaluation problems.

If information from a benchmark or test set is included in training data, directly or indirectly, a model may perform well on that benchmark because of memorization or contamination rather than because it has acquired the intended capability.

Training and evaluation datasets should therefore be separated carefully, and their provenance should be monitored where possible.

---

## 7. Model Distillation

**Knowledge distillation** is a family of techniques in which a **teacher model** provides training signals to a **student model**.

The objective is often to create a smaller or more efficient model that retains useful capabilities of a larger model.

A teacher can provide information such as:

* predictions,
* probability distributions,
* labels,
* generated responses,
* demonstrations,
* other training signals.

### Why Use Distillation?

Large models can be expensive to run because of:

* memory requirements,
* inference latency,
* computational cost,
* energy consumption.

A smaller model can be more suitable for:

* mobile devices,
* embedded systems,
* high-volume applications,
* latency-sensitive applications,
* cost-sensitive deployments.

### Example: DistilBERT

DistilBERT is a well-known example of knowledge distillation applied to BERT.

The published work reported that DistilBERT was substantially smaller and faster than BERT while retaining much of its performance on several language-understanding benchmarks.

The important lesson is not a universal percentage of retained capability, but that **distillation can trade some model size and computational cost for a smaller and faster model**.

---

### Synthetic Data vs. Distillation

These concepts are related but should not be treated as identical.

**Synthetic data generation** means creating artificial training examples.

**Knowledge distillation** means transferring useful information from a teacher model to a student model.

A teacher model may generate synthetic examples as part of a distillation process, but synthetic data can also be generated for purposes that have nothing to do with distillation.

---

### Licensing and Terms of Use

When using outputs from proprietary AI systems for training another model, always check the applicable:

* terms of service,
* model license,
* data license,
* contractual restrictions.

Some providers restrict certain uses of model outputs, while others permit them under specified conditions.

Therefore, **model output should not automatically be assumed to be available for unrestricted training use**.

---

### Can a Student Outperform the Teacher?

A student model can sometimes outperform a teacher on a particular benchmark or task.

This does not necessarily mean that the student has acquired all of the teacher's capabilities.

Possible reasons include:

* the student is specialized for the target task,
* low-quality teacher outputs have been filtered,
* additional high-quality data is used,
* the student has a different architecture or training procedure,
* the benchmark favors the student's specialization.

Therefore, distillation should be evaluated against clearly defined target capabilities rather than assuming that the student is simply a smaller copy of the teacher.

---

## 8. Data Processing Pipeline

Once data has been collected, annotated, or generated, it must be processed before it is used for training or evaluation.

A typical dataset pipeline may include:

**Raw Data → Inspection → Filtering → Deduplication → Quality Checks → Formatting → Splitting → Training/Evaluation**

The exact order can vary depending on the application.

### Data Engineering Practices

1. **Keep the raw source data.**
   Preserve an immutable copy so that processing can be repeated or audited.

2. **Version the dataset.**
   Record which transformations and sources produced each dataset version.

3. **Test processing scripts on a small sample first.**
   Verify the results before processing the complete dataset.

4. **Track data provenance.**
   Record where examples came from and what transformations were applied.

---

### Step 1: Inspect the Data

Before applying large-scale transformations, inspect representative examples.

Look for:

* unexpected formats,
* missing values,
* unusual distributions,
* incorrect labels,
* duplicated examples,
* annotation inconsistencies,
* inappropriate content,
* unexpected language or domain distributions.

Manual inspection remains useful even when much of the processing is automated.

---

### Step 2: Deduplicate Data

Duplicates can affect training and evaluation.

Potential problems include:

* over-representing particular examples,
* increasing memorization,
* distorting the data distribution,
* contaminating evaluation sets.

Deduplication can involve:

* exact matching,
* hashing,
* near-duplicate detection,
* similarity-based methods.

Deduplication should be performed carefully because legitimate repeated patterns are not always unwanted duplicates.

---

### Step 3: Clean and Filter Data

Cleaning and filtering can address:

* corrupted text,
* unwanted HTML or markup,
* malformed records,
* irrelevant documents,
* duplicate content,
* inappropriate material,
* privacy-sensitive information,
* low-quality annotations.

Filtering should be based on explicit criteria rather than removing data simply because it looks unusual.

Rare examples can be important for evaluation and robustness.

---

### Step 4: Format the Data

Training frameworks generally expect data to follow a particular schema.

For example, a conversational dataset might represent an interaction as:

```json
{
  "messages": [
    {"role": "user", "content": "What is a dataset?"},
    {"role": "assistant", "content": "A dataset is a collection of examples..."}
  ]
}
```

The exact format depends on the model and training framework.

For instruction tuning, the dataset may contain fields such as:

* instruction,
* input,
* response,

or a conversation represented as a sequence of messages.

The training pipeline then converts these records into the tokenized representation required by the model.

It is therefore important to ensure that the **training data format is compatible with the model's expected chat or training template**.

---

## Summary

Dataset engineering for LLMs is broader than collecting large amounts of text.

It includes:

* understanding the purpose of the dataset,
* distinguishing pre-training from post-training data,
* improving data quality,
* ensuring appropriate coverage,
* acquiring and annotating data,
* generating synthetic data,
* validating generated examples,
* understanding the limitations of synthetic data,
* using teacher models and distillation where appropriate,
* processing and formatting datasets,
* maintaining provenance and versioning,
* evaluating the resulting model on representative data.

A useful way to view the process is:

**Data Sources → Curation → Annotation/Generation → Verification → Processing → Training → Evaluation → Error Analysis → Data Improvement**

The central principle is:

> **For LLM development, data is not simply an input to the model. It is an engineering artifact that must be designed, evaluated, processed, and continuously improved.**

---

## Ref

- [AI Engineering (Chapter 8), By Chip Huyen](https://metropolia.finna.fi/Record/nelli15.36974248300041)