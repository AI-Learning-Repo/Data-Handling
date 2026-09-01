# Data Representation and File Formats

Before preparing data for fine-tuning, it is important to distinguish between **how a training example is structured**, **how examples are stored**, and **how a particular model expects those examples to be formatted**.

These are related concepts, but they are not the same thing.

## 1. Three Layers of Data Formatting

There are three useful layers to distinguish:

| Layer                   | Examples                                           | What it describes                                           |
| ----------------------- | -------------------------------------------------- | ----------------------------------------------------------- |
| **Data representation** | `instruction`, `input`, `output`, `messages`       | How a training example is organized                         |
| **File format**         | JSON, JSONL, CSV, Parquet                          | How one or many examples are stored                         |
| **Model/chat format**   | Qwen chat template, other model-specific templates | How structured examples are rendered for a particular model |

Consider the following example:

```json
{
  "instruction": "Classify the text.",
  "input": "The patient reports a headache.",
  "output": "symptom"
}
```

This describes the **structure of one training example**.

It contains three fields:

* `instruction` — what task should be performed;
* `input` — the information on which the task is performed;
* `output` — the desired result.

This is a **data representation**.

It can be stored in a JSON file, but the concepts of *data representation* and *JSON* should not be treated as synonymous.

---

# 2. JSON

**JSON** (JavaScript Object Notation) is a structured data format.

A single training example can be represented as a JSON object:

```json
{
  "instruction": "Classify the text.",
  "input": "The patient reports a headache.",
  "output": "symptom"
}
```

JSON can also represent multiple objects inside a single file, usually as an array:

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

Here, the file contains a JSON array containing multiple JSON objects.

JSON is useful when data needs to represent nested or hierarchical structures.

For example:

```json
{
  "id": 101,
  "patient": {
    "age": 42,
    "gender": "female"
  },
  "symptoms": [
    "headache",
    "fatigue"
  ]
}
```

The structure can contain:

* strings;
* numbers;
* Boolean values;
* arrays;
* nested objects.

---

# 3. JSONL

**JSONL** means **JSON Lines**.

Instead of storing many examples inside one JSON array, JSONL stores **one JSON object per line**.

For example:

```json
{"instruction":"Classify the text.","input":"The patient reports a headache.","output":"symptom"}
{"instruction":"Classify the text.","input":"The patient has a fever.","output":"symptom"}
{"instruction":"Classify the text.","input":"The patient was discharged.","output":"outcome"}
```

Each line is a complete JSON object.

Conceptually:

```text
Line 1 → Example 1
Line 2 → Example 2
Line 3 → Example 3
Line 4 → Example 4
...
```

This makes JSONL particularly convenient for datasets containing many independent examples.

---

# 4. JSON vs. JSONL

The difference can be summarized as follows.

### JSON

A JSON file containing multiple examples might look like:

```json
[
  {
    "question": "What is hypertension?",
    "answer": "Hypertension is persistently elevated blood pressure."
  },
  {
    "question": "What is diabetes?",
    "answer": "Diabetes is a metabolic disorder."
  }
]
```

The entire file is one JSON structure: an array.

### JSONL

The same information in JSONL could be:

```json
{"question":"What is hypertension?","answer":"Hypertension is persistently elevated blood pressure."}
{"question":"What is diabetes?","answer":"Diabetes is a metabolic disorder."}
```

Each line is a separate JSON object.

### The key difference

> **JSON commonly represents a complete structured document, while JSONL represents a sequence of independent JSON objects, one per line.**

Both formats can represent the same underlying information.

The choice between them is often determined by how the data will be processed.

---

# 5. Why JSONL Is Common for Machine-Learning Datasets

A fine-tuning dataset may contain thousands or millions of examples.

With JSONL, each example is an independent record:

```text
example 1
example 2
example 3
...
example 100000
```

This structure is convenient for processing datasets incrementally.

It is also easy to:

* append new examples;
* process examples line by line;
* inspect individual records;
* identify malformed records;
* convert records into another structure.

For example:

```python
import json

with open("train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        example = json.loads(line)
        print(example)
```

Each iteration processes one training example.

---

# 6. JSONL Does Not Define the Training Task

An important distinction is that **JSONL tells us how examples are stored, not what the model should learn**.

For example, these are both valid JSONL datasets:

### Classification

```json
{"text":"The server received thousands of login attempts.","label":"possible_brute_force"}
{"text":"The employee clicked a suspicious email link.","label":"possible_phishing"}
```

### Question answering

```json
{"question":"What is hypertension?","answer":"Hypertension is persistently elevated blood pressure."}
{"question":"What is diabetes?","answer":"Diabetes is a metabolic disorder."}
```

Both are JSONL.

However, they represent different tasks and different data structures.

Therefore:

> **JSONL describes the storage format. It does not describe the learning task.**

---

# 7. Data Representation

Before choosing a file format, it is useful to decide how a **single training example** should be represented.

For example, an instruction-following dataset could use:

```json
{
  "instruction": "Classify the text.",
  "input": "The patient reports a headache.",
  "output": "symptom"
}
```

Another dataset might use:

```json
{
  "question": "What is hypertension?",
  "answer": "Hypertension is persistently elevated blood pressure."
}
```

A conversational dataset could instead use:

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

These are different **data representations**.

The representation should be chosen according to the task and the requirements of the training pipeline.

---

# 8. `instruction`, `input`, and `output`

A common representation for instruction-style data is:

```json
{
  "instruction": "Classify the text.",
  "input": "The patient reports a headache.",
  "output": "symptom"
}
```

The fields have distinct roles.

### Instruction

Describes the task:

```text
Classify the text.
```

### Input

Provides the information needed to perform the task:

```text
The patient reports a headache.
```

### Output

Provides the desired result:

```text
symptom
```

This representation can be useful as an intermediate format because it makes the task structure explicit.

---

# 9. Conversational Representation: `messages`

For conversational tasks, a training example can instead be represented using a sequence of messages:

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

The `messages` structure represents a conversation.

Each message has at least:

```text
role
content
```

Typical roles include:

* `system`
* `user`
* `assistant`

For example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a health information assistant."
    },
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

This is still a **data representation**.

It is not yet the model's final token sequence.

---

# 10. Data Representation and File Format Are Independent

The same data representation can be stored in different file formats.

For example, this conversational example:

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

could be stored as one JSON object.

A collection of such examples could be stored as JSONL:

```json
{"messages":[{"role":"user","content":"What is hypertension?"},{"role":"assistant","content":"Hypertension is persistently elevated blood pressure."}]}
{"messages":[{"role":"user","content":"What is diabetes?"},{"role":"assistant","content":"Diabetes is a metabolic disorder."}]}
```

The **data representation** is `messages`.

The **file format** is JSONL.

These are two different concepts.

---

# 11. Other Common File Formats

JSON and JSONL are not the only formats encountered when working with datasets.

### CSV

CSV represents tabular data:

```text
question,answer
"What is hypertension?","Hypertension is persistently elevated blood pressure."
"What is diabetes?","Diabetes is a metabolic disorder."
```

CSV is convenient when the data naturally consists of rows and columns.

### Parquet

Parquet is a column-oriented storage format commonly used for larger datasets and data-processing workflows.

A dataset may therefore be encountered in:

```text
.csv
.json
.jsonl
.parquet
```

The file extension tells us something about **how the data is stored**, but not necessarily how an individual training example should be structured.

Hugging Face Datasets supports loading datasets from several common formats, including CSV, JSON/JSONL, Parquet, and text. [Hugging Face Datasets — Loading data](https://huggingface.co/docs/datasets/loading?utm_source=chatgpt.com)

---

# 12. Converting Between Formats

In practical data preparation, it is common for the source dataset to have a different structure from the desired training dataset.

For example, a source CSV might contain:

```text
question,answer
What is hypertension?,Hypertension is persistently elevated blood pressure.
What is diabetes?,Diabetes is a metabolic disorder.
```

The desired representation might be:

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

The conversion process is:

```text
CSV
 ↓
Read columns
 ↓
Construct a training example
 ↓
Create messages
 ↓
Save as JSONL
```

For example:

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

This type of conversion is a normal part of dataset preparation.

---

# 13. A Dataset Does Not Have to Start in the "Correct" Format

Real datasets are often collected for purposes other than fine-tuning.

For example, a dataset may originally look like:

```text
text | category
```

or:

```text
question | answer
```

or:

```text
document | summary
```

or:

```text
instruction | response
```

The dataset can often be transformed into the representation required by a particular training pipeline.

Therefore, finding a dataset in the "wrong" format does not necessarily mean that it cannot be used.

The important questions are:

1. Does the dataset contain useful information?
2. Does it contain enough information to construct the desired training examples?
3. Can the examples be transformed into a consistent structure?
4. Are the resulting examples appropriate for the intended task?

---

# 14. The Three Layers in Practice

Consider a dataset downloaded from an external source.

It might begin as:

```text
CSV
```

The columns could be:

```text
question
answer
```

The data can then be transformed into:

```text
messages
```

For example:

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

The collection of examples can then be stored as:

```text
train.jsonl
```

At this point, the data is represented as structured conversational examples and stored in JSONL.

The model-specific formatting step comes later.

---

# 15. Model Format Is a Separate Layer

A chat model may require a particular representation of a conversation when the data is prepared for training or inference.

For example:

```text
messages
   ↓
model-specific chat template
   ↓
formatted sequence
```

Different model families can use different conventions.

Therefore, a dataset containing:

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

does not by itself specify the exact special-token sequence that a particular model will use.

That is determined by the model's tokenizer and chat template.

This distinction becomes important when working with Qwen and other chat models.

**Model-specific chat formatting and tokenization will be covered separately.**

---

# 16. The Complete Data Preparation Pipeline

It is useful to visualize the different layers as a sequence:

```text
Raw / Existing Dataset
        ↓
      CSV
        ↓
Data Transformation
        ↓
instruction / input / output
        ↓
Data Transformation
        ↓
messages
        ↓
       JSONL
        ↓
   Data Validation
        ↓
Prepared Dataset
```

A later model-specific stage can then be added:

```text
Prepared Dataset
        ↓
Model-specific chat template
        ↓
Tokenization
        ↓
Fine-tuning
```

The important point is that these are **different stages**.

---

# 17. Common Confusion: "Qwen Requires JSONL"

It is tempting to say:

> "Qwen requires JSONL."

This is not the most useful way to think about the problem.

JSONL is a **file format**.

`messages` is a **data representation**.

A Qwen chat template is a **model-specific formatting mechanism**.

These concepts should be kept separate.

A better description is:

> **JSONL is a convenient way to store training examples; `messages` is a conversational representation; and the model's chat template converts that representation into the model-specific format used for training or inference.**

This distinction also makes it easier to work with other models and other datasets.

---

# 18. Practical Example

Suppose the original data is:

```csv
question,answer
"What is hypertension?","Hypertension is persistently elevated blood pressure."
"What is diabetes?","Diabetes is a metabolic disorder."
```

### Step 1 — Source file

```text
CSV
```

### Step 2 — Data representation

Transform each row into:

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

### Step 3 — Storage format

Store each example as one JSON object per line:

```json
{"messages":[{"role":"user","content":"What is hypertension?"},{"role":"assistant","content":"Hypertension is persistently elevated blood pressure."}]}
{"messages":[{"role":"user","content":"What is diabetes?"},{"role":"assistant","content":"Diabetes is a metabolic disorder."}]}
```

The result is:

```text
train.jsonl
```

### Step 4 — Validation

Check that:

* every line is valid JSON;
* required fields are present;
* roles are correct;
* content is not empty;
* examples are relevant;
* answers are appropriate;
* formatting is consistent.

### Step 5 — Later model-specific processing

Only after the dataset has been prepared do we apply the formatting and tokenization required by the selected model.

---

# 19. Key Takeaways

The most important distinctions are:

### Data representation

Describes **what one training example looks like**.

Examples:

```text
instruction / input / output
```

or:

```text
messages
```

### File format

Describes **how examples are stored**.

Examples:

```text
JSON
JSONL
CSV
Parquet
```

### Model/chat format

Describes **how the structured example is rendered for a particular model**.

Examples include model-specific chat templates.

---

A useful mental model is:

```text
              WHAT IS THE EXAMPLE?
                       │
                       ▼
              Data representation
           instruction / messages / ...
                       │
                       ▼
              HOW IS IT STORED?
                       │
                       ▼
              File format
          JSON / JSONL / CSV / ...
                       │
                       ▼
           WHICH MODEL WILL USE IT?
                       │
                       ▼
            Model-specific format
                       │
                       ▼
                 Tokenization
                       │
                       ▼
                  Fine-tuning
```

> **Good data preparation starts before model-specific formatting.**
>
> First define the task and the structure of the examples. Then choose an appropriate storage format, transform and validate the data, and only afterward apply the formatting required by the target model.
