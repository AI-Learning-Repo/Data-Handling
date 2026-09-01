# Lab: Data Representation and JSONL

## Objective

This short lab focuses on understanding how training examples are represented and stored.

The goal is to distinguish between:

* **data representation** — how one example is structured;
* **file format** — how examples are stored;
* **JSON vs. JSONL** — two common ways of storing structured data.

Model-specific formatting, Qwen, chat templates, tokenization, and fine-tuning are **not covered in this lab**.

---

## 1. Representing One Training Example

Consider the following task:

> Classify a health-related sentence.

A simple training example can be represented as:

```json
{
  "instruction": "Classify the text.",
  "input": "The patient reports a headache.",
  "output": "symptom"
}
```

Here:

* `instruction` defines the task;
* `input` contains the information to process;
* `output` contains the desired result.

### Exercise 1

Create three training examples for the same task.

Use:

```text
instruction
input
output
```

Keep the structure consistent across all three examples.

---

# 2. The Same Data in Different Representations

The same example could instead be represented as:

```json
{
  "question": "What symptom is reported?",
  "answer": "headache"
}
```

Or as a conversational example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What symptom is reported?"
    },
    {
      "role": "assistant",
      "content": "headache"
    }
  ]
}
```

These are different **data representations**.

### Exercise 2

Take one of your examples from Exercise 1 and represent it using:

1. `question` / `answer`
2. `messages`

Compare the three representations.

**Question:** What information is the same, and what has changed?

---

# 3. JSON

A JSON file can contain one structured object:

```json
{
  "instruction": "Classify the text.",
  "input": "The patient reports a headache.",
  "output": "symptom"
}
```

It can also contain multiple objects inside an array:

```json
[
  {
    "instruction": "Classify the text.",
    "input": "The patient reports a headache.",
    "output": "symptom"
  },
  {
    "instruction": "Classify the text.",
    "input": "The patient has a fever.",
    "output": "symptom"
  }
]
```

---

# 4. JSONL

JSONL stands for **JSON Lines**.

In JSONL, each line is a separate JSON object.

```json
{"instruction":"Classify the text.","input":"The patient reports a headache.","output":"symptom"}
{"instruction":"Classify the text.","input":"The patient has a fever.","output":"symptom"}
{"instruction":"Classify the text.","input":"The patient was discharged.","output":"outcome"}
```

The important difference is:

```text
JSON
└── one complete JSON structure

JSONL
├── JSON object
├── JSON object
├── JSON object
└── ...
```

### Exercise 3

Create a file called:

```text
examples.jsonl
```

Add your three training examples from Exercise 1.

There should be **one example per line**.

---

# 5. Reading JSONL with Python

Use Python to read the file:

```python
import json

with open("examples.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        example = json.loads(line)
        print(example)
```

Notice that each line becomes a Python dictionary.

---

# 6. Validate the JSONL File

A JSONL file should contain valid JSON on every line.

Try:

```python
import json

with open("examples.jsonl", "r", encoding="utf-8") as f:
    for line_number, line in enumerate(f, start=1):
        try:
            example = json.loads(line)
            print(f"Line {line_number}: OK")
        except json.JSONDecodeError:
            print(f"Line {line_number}: ERROR")
```

### Exercise 4

Intentionally introduce an error into one line of your JSONL file.

Run the validation code.

What happens?

Fix the error and run the validation again.

---

# 7. Converting a Representation

Suppose the source data looks like this:

```json
{
  "question": "What is hypertension?",
  "answer": "Hypertension is persistently elevated blood pressure."
}
```

We want to convert it to:

```json
{
  "instruction": "Answer the health-related question.",
  "input": "What is hypertension?",
  "output": "Hypertension is persistently elevated blood pressure."
}
```

Write a Python function:

```python
def convert_example(example):
    return {
        "instruction": "Answer the health-related question.",
        "input": example["question"],
        "output": example["answer"]
    }
```

### Exercise 5

Create five examples using the `question` / `answer` structure.

Write Python code that converts all five examples into:

```text
instruction
input
output
```

Then save the converted examples as JSONL.

---

# 8. Reflection

Answer the following questions:

### Question 1

What is the difference between **data representation** and **file format**?

### Question 2

What is the difference between JSON and JSONL?

### Question 3

Why might JSONL be convenient for a dataset containing many training examples?

### Question 4

If a dataset is provided as CSV, does that mean it cannot be used for fine-tuning?

Explain why or why not.

### Question 5

Why is it useful to separate:

```text
data representation
```

from:

```text
file format
```

---

## Final Takeaway

The key distinction is:

```text
How is one example organized?
        ↓
Data representation

How are many examples stored?
        ↓
File format
```

For example:

```text
instruction / input / output
        ↓
       JSONL
```

or:

```text
messages
        ↓
       JSONL
```

The representation and the storage format are separate decisions.

In later work, the prepared dataset can be transformed again for a particular model and training pipeline.
