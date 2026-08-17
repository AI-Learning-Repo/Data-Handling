# About Data Handling and Machine Learning

This course provides the foundation for the two courses that follow it:

1. **Data Handling and Machine Learning — 5 ECTS**
2. **Neural Networks — 5 ECTS**
3. **Neural Network Project — 5 ECTS**

The three courses form a connected learning journey. In this first course, you will learn how to work with data and how data can be used to build machine learning systems. In the following Neural Networks course, you will move beyond classical machine learning and study how neural networks and modern deep learning systems work. The Neural Network Project then gives you an opportunity to apply that knowledge in a larger practical project.

The central idea of this course is simple:

> **AI systems are fundamentally dependent on data: how we acquire it, represent it, process it, validate it, and use it.**

You will explore this idea from two complementary perspectives. During the morning sessions, you will work with **structured data and classical machine learning**. During the afternoon sessions, you will work with **unstructured and text data and modern AI systems**, including fine-tuning and retrieval-augmented generation (RAG).

# Prerequisites

Before starting this course, you should have a working knowledge of:

* **Python programming**
* **Databases and data management**
<!-- * **SQL**
* **NoSQL databases**, such as MongoDB -->

You do not need previous experience with machine learning, deep learning, or LLMs. The course introduces the relevant machine learning and modern AI concepts from the beginning.

# Mindset

Machine learning is not simply about choosing an algorithm and training a model.

Successful AI systems depend heavily on the quality and structure of their data. Before a model can make useful predictions, we need to understand where the data comes from, how it should be represented, whether it contains problems, how it should be processed, and how we can determine whether the resulting system actually works.

This course therefore takes a **data-first approach to AI**.

You will learn to think about questions such as:

* Where does the data come from?
* What does one data sample look like?
* How should the data be represented?
* How do we clean and process it?
* How should we split and validate it?
* What can go wrong?
* How do we measure whether a model is useful?
* Which machine learning approach is appropriate for a particular problem?
* What ethical and practical issues should we consider?

The goal is not only to teach you how to use machine learning tools, but also to develop your ability to **reason about data and make informed decisions when building AI systems**.

# Course Description

**Data Handling and Machine Learning** is a **5 ECTS** course and the first course in a three-course AI package.

The course introduces the fundamental concepts, methods, and tools needed to work with data and build machine learning solutions. You will gain hands-on experience in acquiring, processing, manipulating, storing, representing, and evaluating data, as well as applying machine learning methods to real-world problems.

The course is divided into two complementary parts.

### Part 1: Structured Data and Classical Machine Learning

In the morning sessions, you will work with structured data and classical machine learning.

You will learn how to:

* inspect and understand datasets
* perform exploratory data analysis
* preprocess and clean data
* perform feature engineering
* work with larger datasets using DuckDB
* apply classification
* apply regression
* perform clustering
* perform association analysis
* validate machine learning models
* visualize and interpret results
* build simple machine learning web applications

You will work with practical datasets and use machine learning to solve problems rather than studying algorithms only from a theoretical perspective.

### Part 2: Unstructured Data and Modern AI Systems

In the afternoon sessions, the focus shifts from structured data to **text and other unstructured data**.

You will explore how modern AI systems work with text data, beginning with datasets and data representation and progressing toward modern NLP and LLM-based applications.

You will learn about:

* Hugging Face datasets
* synthetic data
* data curation
* JSON and JSONL
* tokenization
* Byte Pair Encoding (BPE)
* fine-tuning
* Parameter-Efficient Fine-Tuning (PEFT)
* QLoRA
* Qwen 1B
* semantic embeddings
* vector databases
* Retrieval-Augmented Generation (RAG)

You will use **Qwen SLM as a practical vehicle for exploring modern AI data workflows**. The goal is not to study neural networks or transformer architecture in depth at this stage. Those topics are covered in the following **Neural Networks** course.

You will instead focus on the data side of modern AI systems: how to prepare data, represent it, fine-tune an existing model, retrieve relevant information, and evaluate the resulting system.

# From Classical NLP to Modern AI

An important part of the course is understanding how different approaches represent and use text.

As part of your independent study, you will explore **Bag of Words and TF-IDF**. This provides a foundation for comparing traditional NLP approaches with modern methods.

In the later project work, you will compare approaches such as:

**TF-IDF → embeddings → fine-tuning → RAG**

You will consider not only how these approaches work, but also **when each approach is appropriate and what its limitations are**.

In particular, you will learn to reason about questions such as:

* When is a simple TF-IDF approach sufficient?
* When is fine-tuning appropriate?
* When is RAG a better solution?
* What are the trade-offs between these approaches?
* How can we evaluate whether a more advanced approach actually improves the result?

# Mini Project

The afternoon sessions include a collaborative mini project carried out in groups.

The project is developed through two sprints.

### Sprint 1 — Fine-Tuned Model

During the first part of the project, you will:

* identify a problem and a suitable dataset
* prepare and curate data
* create an appropriate training dataset
* fine-tune a Qwen 1B model
* save and use the fine-tuned model
* evaluate the result
* present your work

Possible project domains include areas such as **finance, security, and health**, depending on the selected problem and available data.

### Sprint 2 — RAG and Method Comparison

During the second part of the project, you will extend your understanding of modern AI systems by:

* creating embeddings
* working with a vector database
* implementing a RAG workflow
* evaluating retrieval and generated results
* comparing RAG with fine-tuning
* comparing modern approaches with TF-IDF
* explaining which approach is most suitable for the selected problem and why

The objective is not simply to build an AI application. You should be able to **justify your technical choices based on the characteristics of the data and the problem**.

# Learning Outcomes

By the end of this course, you will be able to:

* explain the fundamental concepts and possibilities of data handling and machine learning
* acquire, manipulate, preprocess, and manage structured and unstructured data
* perform exploratory data analysis and basic feature engineering
* apply classification, regression, clustering, and association analysis to appropriate problems
* select suitable machine learning methods based on the characteristics of a problem
* validate and evaluate machine learning models using appropriate metrics
* visualize and interpret machine learning results
* work with datasets using tools such as pandas, scikit-learn, DuckDB, and Hugging Face
* explain how text can be represented using traditional and modern NLP approaches
* explain the role of tokenization and BPE in modern language models
* prepare datasets for fine-tuning
* perform high-level fine-tuning of an existing Qwen 1B model using PEFT/QLoRA
* explain the basic principles of embeddings and semantic retrieval
* build a basic RAG workflow using a vector database
* compare TF-IDF, fine-tuning, and RAG for different types of problems
* evaluate the quality and limitations of data and AI systems
* recognize ethical, legal, and responsible-use considerations related to data and AI
* communicate and justify decisions made when developing machine learning and AI solutions


# Schedule

The course runs for **eight weeks**, with teaching taking place one day per week on Tuesdays.

Each Tuesday is divided into two parts.

### Morning — Structured Data and Classical Machine Learning

**09:00–12:00**

| Week  | Topics                                                                 |
| ----- | ---------------------------------------------------------------------- |
| **1** | Classical ML overview, Python review, pandas basics                    |
| **2** | Data processing, feature engineering, data handling                    |
| **3** | Classification, evaluation, EDA, Underfitting/ overfitting                                        |
| **4** | Linear and polynomial regression, decision tree regression, evaluation |
| **5** | Clustering and Big Data with DuckDB                                    |
| **6** | Association analysis, **classification web application**                   |
| **7** | Ethics, complete ML workflow, **regression web application**               |
| **8** | **Exam**                                                               |

### Afternoon — Data for Modern AI Systems

**13:00–16:00**

| Week  | Topics                                                              |
| ----- | ------------------------------------------------------------------- |
| **1** | Tools, mini project, project ideas                                  |
| **2** | Hugging Face datasets, synthetic data, data curation                |
| **3** | Data representation, JSON/JSONL, tokenization, BPE                  |
| **4** | Qwen 1B, PEFT, QLoRA, high-level fine-tuning                        |
| **5** | **Sprint 1 presentations — fine-tuned model**                       |
| **6** | Semantic embeddings                                                 |
| **7** | Vector databases and RAG                                            |
| **8** | **Sprint 2 presentations — RAG, TF-IDF and fine-tuning comparison** |

# Course Mechanics

The course follows a practical and interactive learning approach.

Each morning and afternoon session combines **short theory segments with hands-on practice**. Rather than spending long periods in traditional lectures, you will repeatedly move between concepts and practical exercises.

A typical session consists of:

* **35 minutes — Theory**
* **30 minutes — Practice**
* **Break**
* **35 minutes — Theory**
* **30 minutes — Practice**

The practical work is an important part of the course. You will work with real datasets, experiment with machine learning methods, visualize and validate results, and build working applications.

The afternoon sessions additionally include collaborative group work through the two project sprints.

The course is worth **5 ECTS**. The total workload is approximately **135 hours**, corresponding to 27 hours per ECTS. In addition to scheduled teaching, you should reserve time every week for independent study, preparation, assignments, and group project work.

# Technologies and Tools

Throughout the course, you will work with tools and technologies commonly used in modern data science and AI development, including:

* **Python**
* **Google Colab**
* **pandas**
* **scikit-learn**
* **DuckDB**
* **Hugging Face Datasets**
* **Hugging Face Transformers**
* **Qwen 1B**
* **PEFT**
* **QLoRA**
* **Vector databases**
* **RAG**

The tools are used as practical means of understanding the underlying data and machine learning concepts. The focus is not on memorizing specific APIs, but on understanding how to use these tools to solve data and AI problems.

# Materials

The main learning resources include:

* [Python for Data Analysis, 3rd Edition - Wes McKinney](https://metropolia.finna.fi/Record/nelli15.5590000000474271)
* [AI Engineering - Chip Huyen](https://metropolia.finna.fi/Record/nelli15.39388935400041)
* [Data Science for Beginners - Microsoft](https://github.com/microsoft/Data-Science-For-Beginners)
* [ML for Beginners - Microsoft](https://github.com/microsoft/ML-For-Beginners)

Additional lecture material, examples, datasets, exercises, and project instructions will be provided during the course.

Some topics will be covered through **independent study**. This includes supporting concepts such as TF-IDF, selected data-handling concepts, validation fundamentals, and other preparation material needed for the practical sessions.

Assignments and preparation material will be provided in advance so that you can come prepared for each session.

# Teaching Strategy

The teaching strategy combines short lectures, guided practical exercises, independent study, and collaborative project work.

The course follows a **flipped and practice-oriented approach**. Classroom time is used primarily for concepts that benefit from explanation, discussion, experimentation, and feedback. Supporting concepts can be studied independently, allowing classroom sessions to focus on applying knowledge to real data and real problems.

Throughout the course, we repeatedly return to the same fundamental questions:

> **Where did the data come from?**
> **What does one data sample look like?**
> **How should it be represented?**
> **How should it be processed?**
> **How do we validate it?**
> **What can go wrong?**
> **How do we know whether the result is useful?**
> **What are the ethical and practical implications?**

The morning and afternoon sessions approach these questions from different perspectives:

**Morning:**

> *How do we make structured data useful for machine learning?*

**Afternoon:**

> *How do we make unstructured and text data useful for modern AI systems?*

By studying both perspectives, you will develop a broader understanding of the role of data in AI.

# Relationship to the AI Course Package

This course is the **first course in a three-course AI package**.

### Course 1 — Data Handling and Machine Learning — 5 ECTS

You learn how to work with data and how data can be used to build and evaluate classical machine learning and modern AI systems.

↓

### Course 2 — Neural Networks — 5 ECTS

You build on these foundations to study neural networks and the underlying mechanisms of modern deep learning, including the concepts needed to understand neural language models and transformers in greater depth.

↓

### Course 3 — Neural Network Project — 5 ECTS

You apply the knowledge from the previous courses to a larger practical neural network project.

The courses are therefore intentionally connected. **Data Handling and Machine Learning establishes the data and machine learning foundation; Neural Networks opens the model itself; and the Neural Network Project brings the knowledge together in practice.**

# Assessment

The course includes an exam in **Week 8**.

Potential exam questions will be provided progressively throughout the course. This gives you an opportunity to identify the most important concepts early and use them as a guide for your independent study.

The exam focuses on your understanding of the concepts, methods, tools, and decision-making principles covered throughout the course.

The practical mini project is assessed through the two project sprints, where you will demonstrate your ability to work with data, build an AI solution, evaluate it, and explain your technical choices.

### Weight Distribution & Track Options

You can choose **one of three options** to pass the course.

**Option A** (recommended): Best for students who want a collaborative, hands-on learning experience.

* **In-class Pair Programming — 38%**
* **Group Mini Project (2 Sprints) — 28%**
* **Feedback — 7%**
* **Exam — 27%**

**Option B**: For students who prefer not participate in pair programming.

- **Individual Tasks – 20%**
- **Group Mini Project (2 Sprints) – 28%**
- **Paper Exam – 25%**
- **Coding Exam – 27%**

**Option C**: For students with a strong background in Machine Learning who prefer an exam-only evaluation.

- **Paper Exam – 50%**
- **Coding Exam – 50%**

### Overall Grade Thresholds

The course is designed to enable every student to succeed, with the possibility of achieving the highest grade through consistent effort, active participation, and demonstrated understanding.

| Grade | Required percentage |
| ----- | ------------------: |
| **5** |                 93% |
| **4** |                 83% |
| **3** |                 75% |
| **2** |                 67% |
| **1** |                 61% |


# Principles for Using AI

AI tools are an important part of modern data and software development, and this course provides opportunities to use them as learning and development tools.

However, using an AI tool is not the same as understanding the solution it produces.

When using AI systems, you should:

* understand and verify generated code and suggestions
* critically evaluate generated data and outputs
* validate models rather than assuming that they work
* understand the limitations of AI-generated information
* be transparent about AI use when required
* consider privacy, copyright, bias, and other responsible-AI issues

The purpose of using AI in this course is to **strengthen your ability to work with AI systems, not to replace your own understanding and reasoning**.

# By the End of the Course

By the end of the course, you should be able to look at a new data or AI problem and ask the right questions before reaching for a model.

You should be able to reason from:

**Problem → Data → Representation → Processing → Model → Validation → Evaluation → Application**

and understand that the best AI solution is not necessarily the most complex one.

Sometimes a classical machine learning model is appropriate. Sometimes TF-IDF is enough. Sometimes fine-tuning is useful. Sometimes RAG is the better solution.

Your task is to understand **why**.
