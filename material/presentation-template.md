# Mini-Project Presentation Template

**Presentation length: 12 minutes**

Your presentation should tell the story of your project from the initial idea to the final prototype. Do not only demonstrate what you built. Explain **what you tried, what happened, what you learned, what did not work, and what you would change**.

Please use the following structure.

---

### Slide 1 — Project Overview

**Time: ~1 minute**

Introduce your project briefly.

Include:

* Project title
* Group members
* Domain
* Problem you wanted to solve
* Intended user or use case
* Type/source of data you used

Answer:

> **What problem were we trying to solve, and why did we think this was a suitable problem for experimenting with AI?**

Keep this brief. The presentation is primarily about your **development and learning**, not about explaining the domain.

---

### Slide 2 — Phase 1: Initial Prototype

**Time: ~1 minute**

Show your initial **Streamlit prototype**.

Explain:

* What did your first prototype do?
* Which approach did you initially use?
* What could the user do with it?
* What worked?
* What did not work?
* What did you learn from building the first version?

Show at least **one example of an input and output**.

Then answer:

> **How did building the prototype change or clarify our understanding of the problem?**

---

### Slide 3 — Phase 2: Fine-Tuning Qwen

**Time: ~2 minutes**

Explain your fine-tuning experiment.

Include:

#### Data

* Where did your data come from?
* How much data did you use?
* How did you clean or prepare it?
* What did an example training instance look like?

#### Fine-tuning

* Which Qwen model did you use?
* What was the purpose of fine-tuning?
* What did you want the model to learn?

#### Results

Show examples of the model **before and after fine-tuning**, where possible.

Discuss:

* What improved?
* What did not improve?
* Did the model behave differently than expected?
* Did you encounter problems during training?

Also discuss limitations such as:

* small dataset
* quality of training data
* computational limitations
* hallucinations
* overfitting
* insufficient evaluation data

Answer:

> **What did we learn from fine-tuning that we could not learn from simply building the prototype?**

---

### Slide 4 — Phase 3: RAG

**Time: ~2 minutes**

Explain your RAG implementation.

Your RAG system may be relatively simple. You do **not** need to discuss advanced chunking techniques unless you actually used them.

Show your pipeline, for example:

**Your JSONL data → embeddings → vector database/retrieval → relevant information → Qwen → final answer**

Explain:

* What information was stored in your JSONL data?
* How did you create embeddings?
* Where/how did you store the vectors?
* How did retrieval work?
* How was the retrieved information given to Qwen?
* How did the final answer reach the user?

Show **one example where RAG worked well**.

Then, if possible, show **one example where RAG did not work well**.

Discuss:

* Was the correct information retrieved?
* Was irrelevant information retrieved?
* Did Qwen use the retrieved information correctly?
* Did the system hallucinate?
* What limitations did you observe?

Answer:

> **What did RAG add to our project?**

---

### Slide 5 — Comparing Our Approaches

**Time: ~2 minutes**

Compare the approaches you actually implemented.

At minimum, compare:

**Base Qwen vs. Fine-tuned Qwen vs. RAG + Qwen**

You can use a table such as:

|                                   | Initial/Base Qwen | Fine-tuned Qwen | RAG + Qwen |
| --------------------------------- | ----------------- | --------------- | ---------- |
| How does it access information?   |                   |                 |            |
| What does it learn from our data? |                   |                 |            |
| Response quality                  |                   |                 |            |
| Hallucinations/errors             |                   |                 |            |
| Data requirements                 |                   |                 |            |
| Training required?                |                   |                 |            |
| Updating information              |                   |                 |            |
| Computational requirements        |                   |                 |            |
| Main advantage                    |                   |                 |            |
| Main limitation                   |                   |                 |            |

Do not simply say:

> "RAG is better."

Instead, explain:

> **What did our experiment teach us about when fine-tuning and RAG might be useful?**

For example, you might conclude that fine-tuning was useful for changing the model's behavior or output style, while RAG was useful for providing the model with information from an external knowledge source.

Your conclusion should be based on **your observations and evidence from the project**.

---

### Slide 6 — Evaluation: How Do We Know It Worked?

**Time: ~1 minute**

Explain how you evaluated your systems.

Answer:

* What examples/test data did you use?
* Did you use the same questions/examples for different approaches?
* What made an answer "good"?
* What were the limitations of your evaluation?

---

### Slide 7 — Data Privacy, Governance & Responsible AI

**Time: ~1.5 minutes**

Reflect on the data you used in your actual project.

#### Data privacy

Discuss:

* Does the data contain personal information?
* Does it contain sensitive information?
* Could individuals be identified?
* Was the data anonymized?
* Who should have access to the data?

#### Data governance

Discuss:

* Where did the data come from?
* Were you allowed to use it?
* What license or usage restrictions apply?
* Who owns the data?
* Where was it stored?
* Who had access to it?
* Was it uploaded to any external service?
* How should the data be managed in a real-world deployment?

#### AI-specific consideration

Reflect on:

> **What are the privacy and governance differences between fine-tuning a model on data and using that data as an external RAG knowledge source?**

You don't need to provide a sophisticated security architecture. The goal is to demonstrate that you understand that **having access to data does not automatically mean that you can use it without considering privacy, permissions, ownership, security, and governance.**

---

### Slide 8 — Reflection on the Development Process

**Time: ~1 minute**

Your project was divided into different phases/milestones:

**Prototype → Fine-tuning → RAG**

Reflect on this process.

Answer:

* Why was it useful to have different milestones?
* Which milestone was most useful?
* Did an earlier phase reveal a problem that affected a later phase?
* Did your original project idea change?
* Did the data turn out to be different from what you expected?
* Were there things you should have tested earlier?
* Was anything unnecessary?
* What would you change about the sequence of milestones?

Most importantly:

> **If you were starting the project again, how would you organize the milestones differently?**

This is where you should demonstrate what you learned about **iterative development**, rather than simply describing what you did.

---

### Slide 9 — Final Reflection

**Time: ~30–60 seconds**

End with three short reflections.

#### 1. What did we learn?

Give **one or two important technical lessons**.

#### 2. What surprised us?

Describe something you did not expect.

#### 3. What would we do next?

If you had more time, what would you improve or investigate?

Be specific.

Instead of:

> "We would improve the model."

Say something like:

> "We would create a larger independent evaluation set and investigate whether the improvement we observed after fine-tuning generalizes to new examples."

---

# Required Demonstrations

In addition to the slides, your presentation should contain **evidence from your experiments**.

You should show:

### At least one example from the prototype

**Input → Output**

### At least one example from fine-tuning

**Before fine-tuning → After fine-tuning**

### At least one example from RAG

**Question → Retrieved information → Final answer**

### At least one limitation or failure

Show something that **did not work as expected** and explain what you think caused the problem.

> **A good project presentation does not need to show that everything worked. Showing and explaining a failure can demonstrate important learning.**

---

# Suggested 12-Minute Timing

| Section                             |      Time |
| ----------------------------------- | --------: |
| 1. Project overview                 |      1:00 |
| 2. Initial prototype                |      1:00 |
| 3. Fine-tuning                      |      2:00 |
| 4. RAG                              |      2:00 |
| 5. Comparison                       |      2:00 |
| 6. Evaluation                       |      1:00 |
| 7. Data privacy & governance        |      1:30 |
| 8. Development/milestone reflection |      1:00 |
| 9. Final reflection                 |      0:30 |
| **Total**                           | **12:00** |
