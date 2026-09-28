# Mini-Project Showcase & Jigsaw Challenge


## Phase 1: Home Team Strategy & AI Defense Prep

**30 minutes**

Meet with your project group and prepare for the breakout tables.

### 1. Rehearse Your 12-Minute Deck

Walk through your slides:

**Streamlit prototype → Fine-tuning → RAG → Privacy & Governance → Failures & Reflections**

Make sure **every member** is ready to share their screen and present the project.

### 2. Prepare Your Technical Defense: AI Permitted

Review the **Course Question Pool** below.

During this preparation phase, you are **encouraged to use AI tools** such as ChatGPT or Claude to:

* test each other with questions;
* clarify concepts;
* draft possible answers;
* check your understanding;
* connect course concepts to your own project and results.

Prepare a **1-page Technical Defense Cheat Sheet** for your team members.

**Important:** AI is a preparation tool. During your actual presentation and technical defense, you must explain concepts in your own words.

**Live AI use is not permitted during the breakout presentation/defense.**

Also remember: AI-generated answers should be checked against the course material and your **actual project results**.

### 3. Assign Breakout Tables

Look up your Group Number in the Table Assignment Chart.

**Send exactly one representative to each table where your group is listed.**

**Table Assignment Chart**

|    Table    | Capacity | Groups Present at This Table                      |
| :---------: | :------: | :------------------------------------------------ |
| **Table 1** |     5    | **Group 2, Group 4, Group 7, Group 8, Group 10**  |
| **Table 2** |     5    | **Group 1, Group 5, Group 9, Group 10, Group 11** |
| **Table 3** |     5    | **Group 4, Group 5, Group 7, Group 10, Group 11** |
| **Table 4** |     5    | **Group 2, Group 3, Group 6, Group 9, Group 10**  |
| **Table 5** |     5    | **Group 1, Group 3, Group 7, Group 8, Group 11**  |
| **Table 6** |     5    | **Group 2, Group 4, Group 5, Group 6, Group 10**  |
| **Table 7** |     5    | **Group 2, Group 6, Group 7, Group 9**   |
| **Table 8** |     4    | **Group 1, Group 4, Group 5, Group 6**            |

**Important:** Each group must send one representative to every table where its group number appears.

---

## Phase 2: The Jigsaw Breakout Tables

**75 minutes**

At about **10:10**, disperse to your assigned table.

Each table follows the same format:

**10-122 minutes presentation + technical Q&A = 15 minutes per presenter.**

### Your Two Roles

#### 1. As Presenter: approximately 10-12 min + Q&A

* Share your screen.
* Present your group's project.
* Demo your prototype where appropriate.
* Explain your RAG and/or fine-tuning approach.
* Answer questions from your peers.
* Be honest about limitations, failures, and things that did not work.

**Live AI use is not permitted during your presentation or defense.**

A good technical answer does not have to claim that everything worked. Explaining a limitation or failure accurately is part of demonstrating understanding.

#### 2. As Active Listener

While other students present:

* Listen actively.
* Use the Course Question Pool to help formulate questions.
* Ask questions that connect the project to the core course concepts.
* Look for useful ideas, limitations, and lessons that you could transfer to your own work.

### Listener Requirement

During the breakout session, **every student must:**

* ask at least **one substantive technical question**;
* identify at least **one RAG insight** from another project;
* identify at least **one fine-tuning lesson** from another project;
* record at least **one limitation or failure** they learned from another project.

Focus on asking questions that help the group understand the technical choices.

---

## Special Note for Tables 7 and  8

Tables 7 and  8 have four students, so the presentation/Q&A portion takes approximately **60 minutes**.

Use the remaining **15 minutes** for the **Table Synthesis Challenge**.

Together, identify:

1. **One RAG insight** from the projects you heard.
2. **One fine-tuning lesson** from the projects you heard.
3. **One notable failure or limitation.**
4. **One sentence explaining when RAG and fine-tuning would be appropriate.**

Be prepared to share your synthesis if asked.

---

## Course Question Pool

Use these questions during Phase 1 preparation and during Q&A at your breakout tables.

### A. Core Comparison

#### Mandatory question

Every presenter should be prepared to answer:

> **In simple terms, what is the main difference between RAG and fine-tuning for your specific use case?**

#### Additional comparison question

> **Give an example of a scenario where fine-tuning might be preferred over RAG, and vice versa.**

### B. Data & Governance

* **Did you split your data into training/testing data? If so, how did you do it?**
* What makes training data "high quality"? How did you clean or prepare your data?
* What are the data privacy implications of fine-tuning a model on proprietary data versus retrieving it via RAG?
* How did you ensure sensitive information was not exposed?

### C. LLM & Prototype Basics

* How did building the early Streamlit prototype clarify the problem you were trying to solve?
* What causes hallucinations in language models, and did you encounter them?
* Why can an LLM produce a confident answer that is nevertheless incorrect?

### D. Fine-Tuning Qwen

* What did the training loss indicate about your fine-tuning run?
* Did the fine-tuned model show signs of overfitting?
* What could the fine-tuned model do that the base model could not?

### E. RAG Implementation

* What happens when the vector database retrieves irrelevant chunks?
* What was the primary limitation of your retrieval pipeline?
* What could cause a RAG system to give a poor answer even when the correct information exists somewhere in the knowledge base?

---

## Peer Scoring

The purpose of peer scoring is to evaluate how clearly and accurately each project is explained and defended.

### 1. Presentation Score: Out of 10

Listeners rate the presenter using the project template:

* Prototype
* Fine-tuning
* RAG
* Privacy & Governance
* Failures & Honest Evaluation

Use this general scale:

| Score    | General meaning                                                                                    |
| -------- | -------------------------------------------------------------------------------------------------- |
| **9–10** | Clear, accurate, strong technical explanation; answers questions well and acknowledges limitations |
| **7–8**  | Good understanding with some minor gaps                                                            |
| **5–6**  | Basic understanding but noticeable gaps                                                            |
| **3–4**  | Significant misunderstandings or weak explanation                                                  |
| **1–2**  | Unable to explain or defend important parts of the project                                         |

A presenter should **not** be penalized simply for admitting that something failed. Explaining a failure honestly and accurately is part of good technical evaluation.

The listeners' ratings are averaged to produce the presenter's base Presentation Score.

### 2. Peer Badges: +2 Points Each

At the end of the breakout round, each table awards two badges.

**Master Explainer: +2 pt**s

Awarded to the presenter who demonstrated the strongest combination of:

* clarity;
* technical accuracy;
* ability to answer questions;
* honest explanation of limitations and failures.

**Concept Connector: +2 pts**

Awarded to **two listeners** who asked the question that most effectively connected a project to the core course concepts (**Max 2 badges per table**).

The best question is not necessarily the most difficult question. It is a question that helps the group understand something important about **data, LLMs, RAG, fine-tuning, privacy, or model limitations**.

---

## How Final Scores Are Calculated

Because project groups have different numbers of members, we use a **Team Average** rather than a raw total.

$$
\text{Final Group Score}=\frac{\text{Total Points Earned by All Group Members}}{\text{Number of Members in the Group}}
$$

This prevents larger groups from receiving an automatic advantage simply because they have more representatives.

**Example**

Suppose Group A has three members:

* Student A1: **8.7** Presentation + **2** Concept Connector = **10.7**
* Student A2: **8.0** Presentation + **2** Master Explainer = **10.0**
* Student A3: **9.0** Presentation + **2** Concept Connector = **11.0**

Then:

$$
\text{Group A Score}=\frac{10.7+10.0+11.0}{3}=\mathbf{10.57}
$$

---

## Random Check

After the Jigsaw round, some students will receive **one question selected at random from the Course Question Pool**.

This is an **individual check of understanding**.

* No AI.
* Answer in your own words.
* You may use what you learned from the Jigsaw discussion.
* The question will be selected randomly, so be prepared for questions across the different course topics.

You may be asked about:

* data;
* LLM basics;
* RAG;
* fine-tuning;
* privacy/governance;
* limitations and failure modes;
* RAG vs fine-tuning.

The purpose is not to memorize a particular answer. It is to check whether you can **explain the core concepts in your own words after learning from your classmates**.

---

## Phase 3 & 4: Plenary Finalist Showcase

After the Jigsaw round, the scores will be compiled and the **Top 3 Finalist Groups** will be identified.

**Finalist presentations**

**11:35–12:00: Finalist Group 1**

**12:00–13:00: Lunch**

**13:00–14:00: Finalist Groups 2 & 3**

The finalist groups will present to the full class, followed by questions, feedback, and concluding remarks.

---

## What Is the Goal of Today's Activity?

The goal is not simply to give a presentation.

By the end of the Jigsaw, you should be able to:

1. Explain the main ideas behind your own project.
2. Explain basic RAG and fine-tuning concepts in your own words.
3. Compare RAG and fine-tuning at an introductory level.
4. Identify common limitations and failure modes.
5. Ask useful technical questions about another team's work.
6. Learn from the different approaches taken by your classmates.
