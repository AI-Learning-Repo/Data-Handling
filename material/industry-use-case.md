# Model-Centric vs. Data-Centric AI: Why Data Matters

## Before We Start: Data Is Becoming a Valuable AI Skill

If you think AI is only about building better models, take a look at what is happening around **AI training data, expert evaluation, and data infrastructure**.

### Industry examples

* [This 24 Year Old Built A Multibillion-Dollar AI Training Empire In Eight Months — Forbes](https://www.forbes.com/sites/annatong/2025/12/04/this-24-year-old-built-a-multibillion-dollar-ai-training-empire-in-eight-months/?utm_source=chatgpt.com)
* [micro1 — Data lab to train frontier models & evaluate agents](https://www.micro1.ai/?utm_source=chatgpt.com)
* [micro1 — AI Training & Expert Opportunities](https://www.micro1.ai/experts/opportunities?utm_source=chatgpt.com)

These are not just interesting business stories. They illustrate an important trend: **AI companies increasingly need high-quality data, expert feedback, realistic training environments, and rigorous evaluation.**

For example, Forbes reported in December 2025 that micro1, led by 24-year-old Ali Ansari, had pivoted from AI recruitment into AI data annotation and training. The company had reportedly exceeded $100 million in annualized revenue and was receiving investment interest at a valuation of around $2.5 billion. Forbes also reported that some highly specialized experts working on AI training were being paid **$60–$170 per hour**, with some medical and finance experts earning as much as **$500 per hour**.

These numbers should **not** be interpreted as guaranteed salaries. Rather, they demonstrate that a new category of work is emerging around the development, evaluation, and improvement of AI systems.

And this is exactly why we should pay attention to **data**.

---

# Two Complementary Approaches to Building AI

When developing machine learning and artificial intelligence systems, there are two important ways to think about improvement:

* **Model-Centric AI:** The main focus is on improving the model while treating the available dataset as relatively fixed. Engineers experiment with model architectures, hyperparameters, optimization methods, loss functions, training strategies, inference techniques, and other algorithmic choices.

* **Data-Centric AI:** The main focus is on systematically improving the data used to train, fine-tune, and evaluate the system. This includes improving data quality, labeling accuracy, consistency, diversity, coverage, relevance, and representativeness.

These approaches are **not competitors**. In modern AI development, successful systems typically require both.

The important lesson is:

> **A powerful model trained or evaluated on poor data can still produce poor results.**

---

## Why Data Matters So Much

As AI models become increasingly capable, the question is no longer simply:

> **“Can we build a bigger or better model?”**

It is increasingly also:

> **“Do we have the right data to teach, test, and improve that model?”**

Data can become a bottleneck in several ways.

### 1. Quality and Label Accuracy

Incorrect labels, ambiguous instructions, inconsistent judgments, duplicated examples, and systematic errors can all reduce model performance.

For example, imagine training a model to identify whether a legal contract contains a particular type of clause.

If human annotators disagree about what qualifies as that clause, the model receives contradictory signals.

Improving the model architecture may not solve the underlying problem.

**Improving the data may.**

---

### 2. Domain Expertise

General web data is not enough for every AI application.

A model designed to work with:

* legal documents,
* financial analysis,
* medical reports,
* software engineering,
* scientific research,
* robotics,
* cybersecurity,

may require specialized examples and expert evaluation.

This is one reason companies are increasingly interested in **domain experts who can evaluate and improve AI outputs**.

micro1, for example, currently describes its work in terms of expert human data, real-world training environments, and contextual evaluations. Its research areas include legal reasoning, pathology-report reasoning, and financial reasoning.

---

### 3. AI Agents Need More Than Static Text

Modern AI systems are increasingly expected to **perform tasks**, not simply generate text.

An AI agent may need to:

1. understand a goal,
2. plan a sequence of actions,
3. use tools,
4. interact with software,
5. recover from errors,
6. evaluate its own results,
7. complete the task successfully.

Training such systems can require realistic interaction data and environments rather than simply collecting more documents from the internet.

micro1 currently describes its Realm product as providing reinforcement-learning environments designed to generate human data for agentic actions and improve model reasoning.

---

### 4. Robotics Creates an Even Bigger Data Challenge

For language models, enormous amounts of text already exist on the internet.

For robots, the situation is different.

There is no equivalent of “the internet” containing every possible demonstration of a human:

* opening a door,
* folding clothes,
* repairing something,
* preparing food,
* picking up an object,
* navigating a physical environment.

This means some robotics data has to be **created and captured from the real world**.

Forbes reported that micro1 was developing robotics datasets by having people record themselves performing everyday physical tasks.

This illustrates an important principle:

> **The more AI moves into the physical and real-world environment, the more valuable high-quality real-world data becomes.**

---

# Data-Centric AI Is More Than Data Labeling

When people hear “data-centric AI,” they sometimes think only about labeling images or classifying text.

The field is much broader.

Data-centric AI can include:

* data collection,
* data cleaning,
* deduplication,
* filtering,
* labeling,
* expert annotation,
* data selection,
* data balancing,
* synthetic data generation,
* human preference data,
* reinforcement-learning environments,
* tool-use trajectories,
* evaluation datasets,
* benchmark design,
* error analysis,
* dataset versioning,
* quality control.

In other words:

> **Data engineering becomes part of AI engineering.**

---

# A New Career Opportunity

This is especially important for students.

You do not necessarily have to invent a new neural-network architecture to contribute to AI.

There are opportunities for people who understand:

* software engineering,
* mathematics,
* statistics,
* finance,
* law,
* medicine,
* linguistics,
* science,
* robotics,
* data engineering,
* human-computer interaction,

because AI systems need people who can **generate, evaluate, and improve high-quality data**.

micro1's current opportunities page explicitly describes its expert opportunities as using professional expertise to train next-generation AI models, with areas including intelligence, robotics, and related work.

This is an important career message:

> **Your domain expertise can become part of the AI development process.**

---

# The Bigger Picture

The AI development process is increasingly becoming a cycle:

**Model → Data → Evaluation → Error Analysis → Better Data → Better Model → Better Evaluation**

Rather than thinking:

> “AI = building the model”

we should think:

> **“AI = building a system in which models, data, evaluation, and human expertise continuously improve one another.”**

The exciting part for students is that **data is not merely something the model consumes. Data itself is becoming an engineering asset, a research problem, and a career opportunity.**

And that is why, when learning AI, you should pay attention not only to **how models work**, but also to:

**Where does the data come from?
Who creates it?
How is it labeled?
How do we know it is correct?
What data is missing?
How do we evaluate it?
And how can better data make the AI system better?**
