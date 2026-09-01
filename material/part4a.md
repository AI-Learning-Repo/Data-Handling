# Preparing and Formatting Data for Fine-Tuning

Before we fine-tune a model, we need to answer an important question:

> **What exactly should one training example look like?**

Fine-tuning is not simply a matter of collecting a large amount of text and giving it to a model. The data must be structured so that the model can clearly learn:

* what the **instruction** is,
* what information the **user provides**,
* what the **assistant should produce**,
* where a conversation starts and ends,
* and how different roles are represented.

This week, our focus is therefore **data preparation and formatting**.

Next week, we will use the prepared dataset for fine-tuning.

---

# 1. What is a fine-tuning example?

A fine-tuning dataset is usually made of many examples showing the model:

> **Given this input, produce this desired output.**

For example, suppose we want to fine-tune a model to classify financial news.

A simple training example might look like:

```text
Instruction:
Classify the sentiment of the financial news.

Input:
"Company X reported stronger-than-expected quarterly earnings."

Output:
"positive"
```

Another example could be:

```text
Instruction:
Extract the financial information from the text.

Input:
"Company X reported revenue of $2.4 billion in Q2."

Output:
{
  "company": "Company X",
  "revenue": "$2.4 billion",
  "quarter": "Q2"
}
```

The important point is that **the dataset represents the task we want the model to learn**.

The quality of these examples is often more important than simply increasing the number of examples.

---

# 2. From raw data to training data

Real-world data rarely arrives in the format required for fine-tuning.

For example, imagine that we start with a spreadsheet:

| News                                     | Sentiment |
| ---------------------------------------- | --------- |
| Company profits increased significantly. | positive  |
| The company announced major losses.      | negative  |
| Revenue remained stable.                 | neutral   |

This is useful information, but it is not yet necessarily a good chat fine-tuning dataset.

We need to transform the raw information into **training examples**.

For example:

```json
{
  "instruction": "Classify the sentiment of the financial news.",
  "input": "Company profits increased significantly.",
  "output": "positive"
}
```

The same information can then be represented as a conversation:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Classify the sentiment of the financial news:\nCompany profits increased significantly."
    },
    {
      "role": "assistant",
      "content": "positive"
    }
  ]
}
```

The underlying information is the same.

The **representation is different**.

This distinction becomes very important when we prepare data for different model families.

---

# 3. Common formats for fine-tuning data

There is no single universal format for all fine-tuning datasets.

Different libraries, tasks, and model families may use different representations.

Some common approaches include:

### A. Instruction / input / output

A traditional supervised fine-tuning representation is:

```json
{
  "instruction": "Translate the following sentence into French.",
  "input": "The market is recovering.",
  "output": "Le marché se redresse."
}
```

This format is easy to understand and is still useful as an **intermediate data format**.

---

### B. Prompt / completion

Another representation is:

```json
{
  "prompt": "Translate the following sentence into French:\nThe market is recovering.",
  "completion": "Le marché se redresse."
}
```

This represents the example as one prompt followed by the expected completion.

---

### C. Chat messages

Modern chat models commonly represent examples using roles:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful translation assistant."
    },
    {
      "role": "user",
      "content": "Translate: The market is recovering."
    },
    {
      "role": "assistant",
      "content": "Le marché se redresse."
    }
  ]
}
```

This representation is particularly useful for chat-oriented models.

**Qwen fine-tuning commonly uses this messages-based representation.**

---

# 4. Why prompt format matters

Chat models are not trained on an abstract concept of:

> user says something → assistant answers

Instead, the conversation is ultimately represented as a sequence of tokens.

The model is trained to recognize particular patterns that represent:

* the beginning of a message,
* the role of the speaker,
* the message content,
* the end of a message,
* and sometimes the beginning of the assistant's generation.

Therefore, **the structure used to train the model matters**.

For example, Qwen models use a chat template that represents conversations using special control tokens.

A simplified representation looks like:

```text
<|im_start|>system
You are a helpful assistant.
<|im_end|>

<|im_start|>user
What is the revenue?
<|im_end|>

<|im_start|>assistant
The revenue was $2.4 billion.
<|im_end|>
```

These special tokens are not ordinary text.

They help the model understand the structure of the conversation.

> **The important idea:** the JSON representation of a conversation is not necessarily the exact sequence of tokens that the model receives.

There is an additional formatting step.

---

# 5. Qwen and the Chat Template

For this course, we will work with **Qwen**, so we will pay particular attention to how Qwen expects conversational data to be formatted.

Instead of manually writing Qwen's special tokens, we should normally let the tokenizer apply the model's chat template.

For example:

```python
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "What is the revenue?"
    },
    {
        "role": "assistant",
        "content": "The revenue was $2.4 billion."
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False
)
```

The tokenizer converts the structured `messages` object into the model-specific textual/token representation.

Conceptually:

```text
messages
   ↓
chat template
   ↓
Qwen-specific formatted conversation
   ↓
tokenizer
   ↓
input token IDs
   ↓
model
```

This is an important distinction:

> **Your dataset format and the model's internal training format are related, but they are not necessarily the same thing.**

---

# 6. Why use `apply_chat_template()`?

We could manually construct the Qwen prompt:

```python
text = (
    "<|im_start|>user\n"
    "What is the revenue?"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)
```

But this is usually a bad idea.

The exact template can vary between model versions.

The tokenizer already knows the template associated with the model.

Therefore, the safer approach is:

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False
)
```

This gives us the formatting expected by that model.

### General rule

> **When a model provides a chat template, use the tokenizer's chat template rather than manually recreating it.**

This makes our data preparation more robust and reduces formatting errors.

---

# 7. Different models can use different formats

One reason students sometimes get confused about fine-tuning is that examples from one model family cannot necessarily be copied directly to another model.

For example, different model families may use different conventions.

A model may use:

```text
[INST]
...
[/INST]
```

Another may use special role tokens.

Qwen uses its own chat-template conventions, including tokens such as:

```text
<|im_start|>
<|im_end|>
```

The exact formatting should be determined by the model's tokenizer and chat template.

Therefore:

> **Do not assume that a dataset formatted for one model is automatically formatted correctly for another model.**

The same underlying training example may need to be rendered differently depending on the target model.

---

# 8. Dataset format vs. prompt format

These two concepts should not be confused.

Consider this dataset record:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is 2 + 2?"
    },
    {
      "role": "assistant",
      "content": "4"
    }
  ]
}
```

This is a **structured dataset representation**.

After applying the Qwen chat template, it becomes a model-specific formatted sequence.

Conceptually:

```text
Structured dataset
       ↓
messages
       ↓
Qwen chat template
       ↓
special tokens + message content
       ↓
tokenization
       ↓
token IDs
```

This distinction will be important when we build our dataset.

---

# 9. What should the training examples contain?

A good supervised fine-tuning example should make the desired behavior clear.

For example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Extract the company and revenue from this text:\nCompany X reported revenue of $2.4 billion."
    },
    {
      "role": "assistant",
      "content": "{\"company\":\"Company X\",\"revenue\":\"$2.4 billion\"}"
    }
  ]
}
```

Notice that the example contains:

1. A clear task.
2. A realistic input.
3. A desired answer.
4. A consistent output format.

This is what the model learns from.

If our examples are inconsistent, the model receives inconsistent signals.

---

# 10. Consistency is extremely important

Suppose we are training a model to extract financial information.

Example 1:

```json
{
  "company": "Company X",
  "revenue": "$2.4 billion"
}
```

Example 2:

```text
Company: Company Y
Revenue: $1.8 billion
```

Example 3:

```json
{
  "name": "Company Z",
  "sales": "$900 million"
}
```

All three contain similar information, but the output format is inconsistent.

If our goal is structured extraction, this makes the learning task less clear.

A better dataset uses a consistent schema:

```json
{
  "company": "Company Z",
  "revenue": "$900 million"
}
```

The model can then learn the desired output pattern more easily.

> **Fine-tuning examples should be consistent with the behavior we want at inference time.**

---

# 11. Formatting is part of the task design

Data formatting is not merely a technical preprocessing step.

It is also part of **defining what the model should learn**.

For example, suppose we want a model to answer financial questions.

We could train it to produce:

```text
The company generated approximately $2.4 billion in revenue.
```

Or we could train it to produce:

```json
{
  "revenue": 2400000000,
  "currency": "USD"
}
```

These are different target behaviors.

The model will learn patterns from the examples we provide.

Therefore, when preparing a dataset, we should ask:

> **What exact behavior do we want the model to reproduce?**

Then we design the examples around that behavior.

---

# 12. The assistant answer is especially important

In supervised fine-tuning, the desired assistant response is the target behavior.

For example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Summarize this financial report in one sentence."
    },
    {
      "role": "assistant",
      "content": "The company reported strong revenue growth despite higher operating costs."
    }
  ]
}
```

The model learns from the relationship between the input and the desired response.

Therefore, poor answers create poor training signals.

If the dataset contains:

* incorrect answers,
* irrelevant information,
* contradictory instructions,
* inconsistent formatting,
* or low-quality generated examples,

the model may learn those patterns as well.

This is why **data quality matters before fine-tuning begins**.

---

# 13. What about the system message?

Chat datasets can also contain a system message.

For example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a financial analysis assistant."
    },
    {
      "role": "user",
      "content": "What is the main risk mentioned in the report?"
    },
    {
      "role": "assistant",
      "content": "The main risk is declining demand in the European market."
    }
  ]
}
```

The system message can be useful when the behavior should remain consistent across many examples.

However, we should not add system messages simply because they are available.

We should ask:

> **Is this behavior something we actually want the model to learn?**

If every example uses the same system instruction, it becomes part of the training pattern.

---

# 14. Training format and inference format should match

This is one of the most important principles of the activity.

Suppose we fine-tune the model using:

```text
system → user → assistant
```

with a particular output style.

At inference time, we should provide a compatible structure.

For example, if the training examples teach the model to return JSON:

```json
{
  "company": "Company X",
  "revenue": "$2.4 billion"
}
```

we should not suddenly expect a completely different response style without considering how the model was trained.

Similarly, if the model was trained using a particular conversational structure, we should use the model's chat template when constructing inference prompts.

### Key principle

> **The format used during training should be compatible with the format used during inference.**

---

# 15. Why we should not manually memorize Qwen's special tokens

You may see Qwen prompts containing tokens such as:

```text
<|im_start|>
<|im_end|>
```

It is useful to understand what these tokens represent.

However, you should generally **not build your training pipeline around manually typing these tokens**.

Instead:

```python
tokenizer.apply_chat_template(...)
```

should be responsible for rendering the conversation according to the model's expected format.

This gives us a useful division of responsibilities:

```text
We define:
    What the conversation means

The tokenizer defines:
    How that conversation is represented for the model
```

---

# 16. A useful mental model

Think of the process as translating between several representations.

### Level 1 — Raw domain data

```text
Company X reported revenue of $2.4 billion.
```

### Level 2 — Training example

```text
Instruction:
Extract the revenue.

Input:
Company X reported revenue of $2.4 billion.

Output:
$2.4 billion
```

### Level 3 — Structured chat data

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Extract the revenue:\nCompany X reported revenue of $2.4 billion."
    },
    {
      "role": "assistant",
      "content": "$2.4 billion"
    }
  ]
}
```

### Level 4 — Qwen chat template

```text
<|im_start|>user
Extract the revenue:
Company X reported revenue of $2.4 billion.
<|im_end|>
<|im_start|>assistant
$2.4 billion
<|im_end|>
```

### Level 5 — Tokens

The tokenizer converts the formatted conversation into token IDs that can be processed by the model.

```text
Text
 ↓
Tokenizer
 ↓
[ token_1, token_2, token_3, ... ]
```

This is ultimately what the model trains on.

---

# 17. Why we will examine multiple formats this week

Our goal is not to memorize a collection of JSON structures.

Instead, we want to understand **why different formats exist and when they are appropriate**.

During the activity, we will compare:

| Format                   | Example use                      | Important idea                             |
| ------------------------ | -------------------------------- | ------------------------------------------ |
| Instruction/Input/Output | Traditional instruction datasets | Separates task, input, answer              |
| Prompt/Completion        | Completion-style training        | Represents prompt → target                 |
| Chat messages            | Chat/instruction models          | Represents conversation roles              |
| Model chat template      | Qwen and other chat models       | Converts messages to model-specific format |
| Token IDs                | Actual model input               | What the model ultimately processes        |

The key question is always:

> **What representation does my target model expect?**

For this course, the answer will ultimately be determined by the **Qwen tokenizer and its chat template**.

---

# 18. Preparing data for our mini-project

Later, we will build a synthetic dataset using AI.

For example, a finance project might generate examples such as:

```text
User:
Analyze the following financial statement and identify the main risk.

Assistant:
The main risk is declining operating margin...
```

Or structured extraction examples:

```text
User:
Extract the following financial information...

Assistant:
{
  "revenue": ...,
  "profit": ...,
  "currency": ...
}
```

The important point is that AI-generated examples are **not automatically good training examples**.

We will need to:

1. Define the task.
2. Define the desired output.
3. Define the schema or conversation format.
4. Generate examples.
5. Validate the generated examples.
6. Remove poor or contradictory examples.
7. Convert them into the format expected by Qwen.
8. Inspect the final training examples before fine-tuning.

So the workflow for the course will be:

```text
Domain problem
      ↓
Define desired behavior
      ↓
Design training examples
      ↓
Generate / collect data
      ↓
Clean and validate
      ↓
Format as chat messages
      ↓
Apply Qwen chat template
      ↓
Inspect tokenized examples
      ↓
Fine-tune
```

---

# 19. The three questions we should always ask

Before fine-tuning any model, ask:

### Question 1 — What behavior are we teaching?

For example:

> Extract financial entities from a document.

### Question 2 — What should a correct answer look like?

For example:

```json
{
  "company": "...",
  "revenue": "...",
  "profit": "..."
}
```

### Question 3 — How does our target model expect the conversation to be formatted?

For Qwen, we use the tokenizer's chat template:

```python
tokenizer.apply_chat_template(...)
```

These three questions connect **data design**, **dataset formatting**, and **model-specific formatting**.

---

# 20. Key Takeaways

Before moving to fine-tuning, you should understand the following:

* A fine-tuning dataset is a collection of examples describing the behavior we want the model to learn.
* Raw data is usually not ready for fine-tuning.
* Training examples can be represented in several formats.
* Chat models commonly use structured `messages` containing roles such as `system`, `user`, and `assistant`.
* Different model families can use different chat formats.
* **Qwen uses a model-specific chat template.**
* We should normally use:

```python
tokenizer.apply_chat_template(...)
```

rather than manually constructing Qwen's special-token format.

* The dataset representation and the final token sequence are different stages of the pipeline.
* Consistency between examples is important.
* The quality of the assistant's target responses is critical.
* The training format should be compatible with the inference format.
* Synthetic data must be validated before it is used for fine-tuning.

> **The goal of this week's activity is not yet to fine-tune the model.**
>
> The goal is to learn how to turn raw/domain information into **high-quality, consistent, correctly formatted training data** that a Qwen model can learn from.
