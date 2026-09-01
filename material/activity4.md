# Lab: Preparing Data for Fine-Tuning

## From Raw Data to a Clean Training Dataset

### Overview

Before we can fine-tune a language model, we need to prepare the data that the model will learn from.

In this lab, we will focus **only on data preparation**.

We will **not** work with Qwen, chat templates, tokenization, or fine-tuning today. Those steps will be covered next week.

Our goal today is to produce a clean dataset that is ready for the next stage.

The overall process is:

```text
Raw / Existing Data
        ↓
Inspect
        ↓
Clean
        ↓
Choose a task
        ↓
Create training examples
        ↓
Choose a representation
        ↓
Convert to JSONL
        ↓
Validate
        ↓
READY FOR NEXT WEEK
```

---

# Learning Objectives

By the end of this lab, you should be able to:

* identify common formats used to store datasets;
* load and inspect a dataset;
* understand the difference between CSV, JSON, and JSONL;
* identify whether an existing dataset is suitable for fine-tuning;
* transform an existing dataset into a new structure;
* create instruction/input/output examples;
* create conversational `messages` examples;
* save a dataset as JSONL;
* identify common problems in training data;
* understand how synthetic data can be generated using an LLM;
* understand the basic idea of **distillation** for generating training data;
* create and validate a small domain-specific synthetic dataset.

---

# Part 1: What Does a Training Example Look Like?

A fine-tuning dataset consists of many examples.

At its simplest:

```text
Input → Desired Output
```

For example:

```text
Input:
What is hypertension?

Output:
Hypertension is a condition in which blood pressure is persistently elevated.
```

Another example:

```text
Input:
Classify this security event:

Multiple failed login attempts were detected from the same IP address.

Output:
brute_force_attempt
```

Before collecting thousands of examples, we need to decide:

> **What behavior do we want the model to learn?**

This is the first decision in data preparation.

---

# Part 2: Different Ways to Represent Data

There are many ways to represent the same information.

For example, this information:

```text
Question: What is hypertension?

Answer: Hypertension is persistently elevated blood pressure.
```

could be represented as a table:

| question              | answer                                                |
| --------------------- | ----------------------------------------------------- |
| What is hypertension? | Hypertension is persistently elevated blood pressure. |

Or as JSON:

```json
{
  "question": "What is hypertension?",
  "answer": "Hypertension is persistently elevated blood pressure."
}
```

Or as an instruction example:

```json
{
  "instruction": "Answer the following health question.",
  "input": "What is hypertension?",
  "output": "Hypertension is persistently elevated blood pressure."
}
```

Or as a conversation:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is hypertension?"
    },
    {
      "role": "assistant",
      "content": "Hypertension is persistently elevated blood pressure."
    }
  ]
}
```

The information is similar, but the **representation is different**.

---

# Part 3: CSV, JSON and JSONL

Today we will work with three important data formats.

## CSV

CSV is useful for tabular data.

Example:

```text
question,answer
"What is hypertension?","Hypertension is persistently elevated blood pressure."
"What is diabetes?","Diabetes is a metabolic disorder..."
```

CSV works particularly well when every example has a simple, fixed set of columns.

---

## JSON

JSON represents structured data.

For example:

```json
{
  "question": "What is hypertension?",
  "answer": "Hypertension is persistently elevated blood pressure."
}
```

JSON can represent more complex structures than CSV.

---

## JSONL

JSONL means **JSON Lines**.

Each line contains one JSON object.

```json
{"question":"What is hypertension?","answer":"Hypertension is persistently elevated blood pressure."}
{"question":"What is diabetes?","answer":"Diabetes is a metabolic disorder..."}
{"question":"What is asthma?","answer":"Asthma is a condition affecting the airways..."}
```

This is particularly convenient for machine-learning datasets because each line represents one example.

> **Remember:**
>
> JSON describes structured data.
>
> JSONL is a convenient way of storing many individual JSON records in one file.

---

# Part 4: Task 1: Inspect a Real Dataset

Instead of creating all our data from scratch, we will first work with a **real dataset from Hugging Face**.

Hugging Face provides many datasets covering areas such as:

* health;
* finance;
* cybersecurity;
* question answering;
* classification;
* summarization;
* instruction following.

You can browse datasets at the Hugging Face Datasets Hub.

[Hugging Face Datasets](https://huggingface.co/datasets)

<!-- For this exercise, your instructor will provide a dataset or dataset name. -->

Load it using:

```python
from datasets import load_dataset

dataset = load_dataset("DATASET_NAME")
```

Inspect the dataset:

```python
print(dataset)
```

Then inspect the available columns:

```python
print(dataset["train"].column_names)
```

Finally, inspect one example:

```python
print(dataset["train"][0])
```

---

## Questions

Answer the following:

1. How many examples are in the training set?
2. What columns are available?
3. What does one example look like?
4. Which field contains the input?
5. Which field contains the desired output?
6. Is there an instruction?
7. Is the dataset already suitable for our intended task?
8. What would need to be changed?

---

# Part 5: Task 2: Dataset Detective

Not every dataset you find will be ready for your project.

Consider this example:

```json
{
  "text": "The patient reports severe headache.",
  "label": "symptom"
}
```

Suppose our goal is to train a model to classify health-related text.

This dataset might be useful, but we need to decide how to represent the training examples.

For example:

```json
{
  "instruction": "Classify the following health-related text.",
  "input": "The patient reports severe headache.",
  "output": "symptom"
}
```

Or:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Classify the following health-related text:\nThe patient reports severe headache."
    },
    {
      "role": "assistant",
      "content": "symptom"
    }
  ]
}
```

### Your task

For the dataset you selected:

1. Identify its current structure.
2. Decide what the desired training-example structure should be.
3. Explain why you chose that structure.
4. Identify what transformations are necessary.

---

# Part 6: Task 3: Convert an Existing Dataset

Suppose we have this dataset:

```json
{
  "question": "What is hypertension?",
  "answer": "Hypertension is persistently elevated blood pressure."
}
```

We want to convert it into:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is hypertension?"
    },
    {
      "role": "assistant",
      "content": "Hypertension is persistently elevated blood pressure."
    }
  ]
}
```

Write a conversion function:

```python
def convert_example(example):
    return {
        "messages": [
            {
                "role": "user",
                "content": example["question"]
            },
            {
                "role": "assistant",
                "content": example["answer"]
            }
        ]
    }
```

Apply it to your dataset:

```python
formatted_dataset = dataset.map(convert_example)
```

Inspect the result:

```python
print(formatted_dataset["train"][0])
```

---

## Think about the transformation

You have converted:

```text
question + answer
        ↓
user + assistant
        ↓
messages
```

This is an example of **data transformation**.

The original dataset did not have to be in the final format.

That is a normal part of preparing data for machine learning.

---

# Part 7: Task 4: Save the Dataset as JSONL

Now save the prepared examples as JSONL.

```python
import json

with open("train.jsonl", "w", encoding="utf-8") as f:
    for example in formatted_dataset["train"]:
        f.write(
            json.dumps(example, ensure_ascii=False) + "\n"
        )
```

You should now have:

```text
train.jsonl
```

Open the file and inspect several lines.

You should see something similar to:

```json
{"messages":[{"role":"user","content":"What is hypertension?"},{"role":"assistant","content":"Hypertension is persistently elevated blood pressure."}]}
{"messages":[{"role":"user","content":"What is diabetes?"},{"role":"assistant","content":"Diabetes is a metabolic disorder..."}]}
```

---

# Part 8: Task 5: Validate the JSONL

A file being called `train.jsonl` does not mean that it is a good training dataset.

We need to validate it.

Start with a basic structural check:

```python
import json

with open("train.jsonl", "r", encoding="utf-8") as f:
    for line_number, line in enumerate(f, start=1):
        try:
            example = json.loads(line)
            print("Line", line_number, "OK")
        except json.JSONDecodeError:
            print("Line", line_number, "ERROR")
```

Now check the structure.

For example:

```python
def validate_example(example):
    if "messages" not in example:
        return False

    if not isinstance(example["messages"], list):
        return False

    if len(example["messages"]) < 2:
        return False

    return True
```

---

# Part 9: Find Bad Training Examples

A dataset can be valid JSON and still be a **bad training dataset**.

Consider these examples.

### Example A

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is diabetes?"
    },
    {
      "role": "assistant",
      "content": "Diabetes is a group of metabolic disorders characterized by elevated blood glucose."
    }
  ]
}
```

### Example B

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is diabetes?"
    },
    {
      "role": "assistant",
      "content": "I don't know."
    }
  ]
}
```

### Example C

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is diabetes?"
    },
    {
      "role": "assistant",
      "content": "Diabetes is definitely caused by eating sugar."
    }
  ]
}
```

### Discuss

Which examples would you include?

Why?

What makes an example a **high-quality training example**?

Think about:

* correctness;
* relevance;
* consistency;
* completeness;
* clarity;
* formatting;
* duplication;
* inappropriate content;
* contradictions.

> **Important:** A syntactically correct dataset is not necessarily a high-quality dataset.

---

# Part 10: Synthetic Data

Sometimes we do not have enough training examples.

One possible solution is to create **synthetic data**.

Synthetic data is artificially generated data designed to resemble the examples we want to train on.

For example, suppose we want to build a health-information assistant.

We could ask an LLM to generate:

```text
Question:
What is hypertension?

Answer:
...
```

We could generate hundreds of variations of questions and answers.

However:

> **Generated data should not automatically be trusted.**

Synthetic data needs to be reviewed and validated just like real data.

---

# Part 11: Synthetic Data Pipeline

A useful way to think about synthetic data generation is:

```text
Define the task
      ↓
Define what a good example looks like
      ↓
Ask an LLM to generate examples
      ↓
Inspect examples
      ↓
Remove bad examples
      ↓
Validate structure
      ↓
Save as JSONL
```

The LLM is a **data-generation tool**, not the final authority on whether the data is correct.

---

# Part 12: Distillation

Another approach is **distillation**.

The basic idea is:

> Use a stronger model as a teacher to generate examples that can later be used to train a smaller model.

For example:

```text
             Teacher LLM
            /     |     \
           ↓      ↓      ↓
       Example  Example  Example
           \      |      /
            ↓     ↓     ↓
          Synthetic Dataset
                 ↓
          Smaller Model
```

For example, a teacher model could be asked:

```text
You are creating training data for a health-information assistant.

Generate a user question and a high-quality educational answer.

Return:
- question
- answer
```

The resulting examples can then be converted into our chosen dataset format.

---

# Part 13: (**Optional**) Generate Synthetic Data Using an API

You can generate a small synthetic dataset using an LLM API.

You may use either:

* OpenAI API
* Gemini API

You do **not** need to build a complicated application.

The basic idea is:

```text
Python program
      ↓
API request
      ↓
LLM generates examples
      ↓
Python receives examples
      ↓
Save as JSONL
```

<!-- For this week's homework, focus on the **data**, not the API engineering. -->

Your final output should be a JSONL file containing your generated examples.

---

# Part 14: In-Class Activity: Build a Dataset with ChatGPT

Now we will create a dataset interactively using an AI assistant such as ChatGPT.

We will work with two domains:

## Domain A: Health

The goal is to create a small educational health-information dataset.

Possible tasks include:

* answering basic health questions;
* explaining medical terminology;
* extracting information from a provided text;
* classifying health-related text;
* summarizing health information.

For example:

```text
User:
What is hypertension?

Assistant:
[high-quality educational answer]
```

Another possible task:

```text
User:
Extract the symptoms mentioned in this text:

"The patient reports headache and fatigue."

Assistant:
{
  "symptoms": ["headache", "fatigue"]
}
```

### Important

For this exercise, the dataset should focus on **general educational information**, not diagnosis or personalized medical advice.

---

# Domain B: Cybersecurity

The second dataset will focus on cybersecurity.

Possible tasks include:

* classifying security events;
* explaining security concepts;
* extracting indicators from security alerts;
* summarizing incidents;
* categorizing security events.

For example:

```text
User:
Classify this security event:

"Multiple failed login attempts were detected from the same IP address."

Assistant:
brute_force_attempt
```

Or:

```text
User:
Extract the indicators from this security alert.

Assistant:
{
  "ip_addresses": ["..."],
  "domains": ["..."],
  "event_type": "..."
}
```

---

# Part 15: Prompting ChatGPT to Generate Data

Your first task is **not** to ask:

> "Give me 100 examples."

Instead, first define the task carefully.

For example:

```text
We are creating a supervised fine-tuning dataset
for an educational health-information assistant.

Generate examples where:

- the user asks a general health question;
- the assistant provides a clear educational answer;
- answers should not diagnose individuals;
- answers should not provide personalized medical treatment;
- the output should be concise and factually responsible.

Return each example with:
- question
- answer
```

Then inspect the generated examples.

Ask yourself:

> **Would I want a model to learn to behave this way?**

---

# Part 16: Improve the Generation Prompt

Now ask the AI assistant to generate examples with stricter requirements.

For example:

```text
Generate 10 examples.

Requirements:

1. Every example must contain a question and answer.
2. Questions must be different from one another.
3. Answers must directly answer the question.
4. Do not invent statistics or unsupported claims.
5. Use clear educational language.
6. Keep answers between 30 and 80 words.
7. Return valid JSON.
```

Compare the results.

### Discuss

How did changing the prompt affect the dataset?

Which prompt produced better examples?

What problems still remain?

---

# Part 17: Human Validation

Now manually review the generated dataset.

For each example, ask:

| Question                                        | Yes/No |
| ----------------------------------------------- | ------ |
| Is the example relevant?                        |        |
| Is the answer correct?                          |        |
| Does the answer actually answer the question?   |        |
| Is the format consistent?                       |        |
| Is the example duplicated?                      |        |
| Is the wording clear?                           |        |
| Would we want the model to learn this response? |        |

Remove examples that fail important checks.

This is an essential part of synthetic-data generation.

> **Generation is only one step. Validation is equally important.**

---

# Part 18: Final Dataset

At the end of today's lab, your dataset should look conceptually like:

```text
project/
│
├── raw_data/
│   └── original_dataset
│
├── processed_data/
│   └── train.jsonl
│
└── notes/
    └── data_decisions.md
```

The `train.jsonl` file should contain clean, consistent training examples.

For example:

```json
{"messages":[{"role":"user","content":"What is hypertension?"},{"role":"assistant","content":"..."}]}
{"messages":[{"role":"user","content":"What is diabetes?"},{"role":"assistant","content":"..."}]}
{"messages":[{"role":"user","content":"What is asthma?"},{"role":"assistant","content":"..."}]}
```

---

# Part 19: Final Challenge

Choose **one** of the two domains:

* Health
* Cybersecurity

Then complete the following:

### Step 1: Define the task

Write one sentence:

> "We want the model to learn how to __________."

### Step 2: Define the input

What will the user provide?

### Step 3: Define the output

What should the assistant produce?

### Step 4: Choose a representation

For example:

```text
instruction/input/output
```

or:

```text
messages
```

### Step 5: Create examples

Use either:

* an existing dataset from Hugging Face;
* manually created examples;
* AI-generated synthetic examples;
* or a combination.

### Step 6: Convert

Convert your examples into a consistent structure.

### Step 7: Save

Save them as:

```text
train.jsonl
```

### Step 8: Validate

Check:

* valid JSON;
* consistent structure;
* correct roles;
* no missing values;
* no obvious duplicates;
* appropriate answers;
* consistent output format;
* no examples that should not be learned.

---

# Deliverable

At the end of the lab, submit:

### 1. `train.jsonl`

A clean dataset containing your training examples.

### 2. `data_notes.md`

Briefly describe:

* where the data came from;
* what task the dataset represents;
* what transformations you performed;
* what format you selected and why;
* whether you used synthetic data;
* how you validated the data;
* approximately how many examples you kept and removed.

### 3. One reflection

Answer:

> **What was the most difficult part of preparing the dataset, and why?**

---

# What We Are NOT Doing Today

Today we stop here:

```text
                    TODAY
                      ↓
Raw Data
   ↓
Clean Data
   ↓
Training Examples
   ↓
JSON / JSONL
   ↓
Validation
   ↓
       ✓ READY
```

Next week we will continue:

```text
             NEXT WEEK
                  ↓
        Prepared JSONL dataset
                  ↓
        Qwen conversation format
                  ↓
         Chat template
                  ↓
            Tokenization
                  ↓
           Fine-tuning
                  ↓
            Evaluation
```

So today's objective is simple:

> **Prepare good data first. Train the model later.**

A well-formatted, consistent, high-quality dataset is the foundation for everything we will do next week.
