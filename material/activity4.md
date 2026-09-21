# Activity 4: Augmented Generation with Qwen 2.5 (RAG Part 2)

In this lab, you will:

1. prepare the Google Colab GPU runtime,
2. load the base `Qwen/Qwen2.5-1.5B-Instruct` model and tokenizer without adapters,
3. test the untrained model on MediCore Hospital questions to observe baseline failure,
4. connect the ChromaDB retrieval engine built in Lab 5A,
5. build a pure Python RAG pipeline without external orchestration frameworks (no LangChain),
6. format retrieved context into Qwen’s native ChatML structure using `tokenizer.apply_chat_template()`,
7. enforce strict grounding prompts to eliminate hallucinations,
8. demonstrate real-time factual updates without fine-tuning or retraining.

> [!NOTE]
> Last week, we baked knowledge directly into model weights using LoRA adapters. In this lab, the model weights remain completely frozen and untrained. All domain knowledge will be retrieved from ChromaDB at inference time and injected directly into the prompt context.

---

### Prerequisite: Enable GPU in Google Colab

Generating tokens with `Qwen2.5-1.5B-Instruct` requires a GPU for low-latency generation.

1. In the top menu, click **Runtime** > **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU**.
3. Click **Save**.

---

## Core Workflow

### Step 0: Install Dependencies

Install the Hugging Face ecosystem alongside ChromaDB and sentence transformers:

```python
# [Cell 0] Install Dependencies
!pip install -q \
    "opentelemetry-api>=1.39.0,<=1.42.1" \
    "opentelemetry-sdk>=1.39.0,<=1.42.1" \
    transformers \
    accelerate \
    chromadb \
    sentence-transformers
```

#### Code Explanation:
<details>
<summary><b>Code Explanation:</b></summary>

* `transformers` & `accelerate`: The core Hugging Face libraries used to load and execute causal language models on GPU hardware.
* `torch`: PyTorch, the underlying tensor library running the neural network calculations.
* `chromadb` & `sentence-transformers`: The retrieval layer used to perform semantic search over domain documents.
</details>

---

### Step 1: Load the Base Model and Tokenizer

Load the exact same starting base model used in Activity 4. Notice that we are **not** attaching any PEFT or LoRA adapters.

```python
# [Cell 1] Load Base Model and Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# 1. Load the tokenizer and ensure padding tokens are mapped
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 2. Load the base model in FP16 precision directly onto the GPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype=torch.float16
)

# Enable KV caching for fast inference generation
model.config.use_cache = True

print("Base model loaded successfully into GPU memory.")
```

#### Code Explanation:
<details>
<summary><b>Code Explanation:</b></summary>

* `model_name = "Qwen/Qwen2.5-1.5B-Instruct"`: The 1.5-billion parameter instruction-tuned model.
* `torch_dtype=torch.float16`: Loads the model in 16-bit precision, consuming approximately ~3 GB of VRAM.
* `device_map="cuda"`: Automatically places all transformer layers on the Colab T4 GPU.
* `model.config.use_cache = True`: Unlike training (where KV-caching is turned off for backpropagation), we explicitly enable it here to accelerate sequential token generation during inference.
</details>

---

### Step 2: Baseline Evaluation (Without Context)

Before introducing retrieval, query the base model about MediCore Hospital to establish a performance baseline.

```python
# [Cell 2] Baseline Test Without Context
test_query = "Who leads the neurology department at MediCore Hospital?"

messages = [
    {"role": "user", "content": test_query}
]

# Format using Qwen's ChatML template
prompt_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

# Generate response using greedy decoding
output_ids = model.generate(
    **inputs,
    max_new_tokens=60,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id
)

# Strip out the input prompt and decode newly generated tokens
response = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("BASELINE UNGROUNDED ANSWER:\n", response)
```

#### Code Explanation:
<details>
<summary><b>Code Explanation:</b></summary>

* `messages = [...]`: Represents standard conversational user input.
* `tokenizer.apply_chat_template(..., add_generation_prompt=True)`: Appends `<|im_start|>assistant\n` so the model knows it is its turn to speak.
* `do_sample=False`: Employs greedy decoding. Because this model was never trained on MediCore Hospital and has no reference context, it will either hallucinate a fictitious doctor or state that it lacks information.
</details>

---

#### 🧪 Checkpoint Challenge 1: The Knowledge Limit

> **Question:**  
> Why did the model fail to answer or invent a fictional name? In Activity 4, the model knew that Dr. Elena Varga led the neurology department after Cell 7. Why does it not know now?

💡 **Gemini Prompt Hint:**  
> *"Explain the difference between parametric memory updated during LoRA fine-tuning versus an untrained base foundation model encountering private enterprise entities."*

<details>
<summary><b>Answer & Explanation</b></summary>

In Activity 4, you performed Parameter-Efficient Fine-Tuning (LoRA), which modified the internal weight matrices of the model so that the fact was stored in its *parametric memory*. 

In this lab, we loaded the clean, untouched base model (`Qwen2.5-1.5B-Instruct`). Because MediCore Hospital is a private synthetic entity that did not exist in Qwen’s public pre-training corpus, the model has no internal knowledge of it and must either admit ignorance or hallucinate based on statistical probabilities.
</details>

---

### Step 3: Populate the ChromaDB Vector Index

Re-create the retrieval collection established in Lab 5A. This will act as our external *non-parametric* memory.

```python
# [Cell 3] Ingest MediCore Knowledge into ChromaDB
import json
import chromadb
from chromadb.utils import embedding_functions

# 1. Download dataset if missing
!wget -nc -q https://raw.githubusercontent.com/AI-Learning-Repo/Data-Handling/refs/heads/week4/datasets/MediCore.json

# 2. Initialize in-memory Chroma client
chroma_client = chromadb.Client()
st_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = chroma_client.get_or_create_collection(
    name="medicore_rag",
    embedding_function=st_fn
)

# 3. Read and index records
with open("MediCore.json", "r", encoding="utf-8") as f:
    lines = [json.loads(line.strip()) for line in f]

collection.add(
    documents=[item["completion"] for item in lines],
    ids=[f"fact_{idx}" for idx in range(len(lines))],
    metadatas=[{"prompt": item["prompt"]} for item in lines]
)

print(f"ChromaDB ready: {collection.count()} knowledge chunks indexed.")
```

#### Code Explanation:
<details>
<summary><b>Code Explanation:</b></summary>

* `SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")`: Converts incoming text chunks into 384-dimensional vectors.
* `collection.add(...)`: Stores the factual statements from `MediCore.json` alongside unique IDs and metadata.
</details>

---

### Step 4: The Pure Python RAG Pipeline

Connect retrieval to generation. The process follows a clear sequence:
1. Embed the user query and retrieve the top-$k$ nearest context passages from ChromaDB.
2. Concatenate these passages into a clean context block.
3. Inject the context into Qwen’s ChatML prompt alongside strict grounding instructions.
4. Execute `model.generate()`.

```python
# [Cell 4] Context Retrieval and Grounded Generation
user_query = "Who leads the neurology department at MediCore Hospital?"

# 1. Retrieve the top 2 most relevant passages
search_results = collection.query(
    query_texts=[user_query],
    n_results=2
)
retrieved_chunks = search_results["documents"][0]
context_block = "\n".join([f"- {chunk}" for chunk in retrieved_chunks])

print("--- RETRIEVED CONTEXT ---")
print(context_block)
print("--------------------------\n")

# 2. Build the Grounded ChatML Message Structure
messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant for MediCore Hospital. "
            "Answer the user's question relying ONLY on the provided context below. "
            "If the answer cannot be determined from the context, state: 'Information not available.'"
        )
    },
    {
        "role": "user",
        "content": f"Context:\n{context_block}\n\nQuestion: {user_query}"
    }
]

# 3. Apply Chat Template
formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 4. Generate Answer
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )

grounded_response = tokenizer.decode(
    output_tokens[0][inputs.input_ids.shape[1]:], 
    skip_special_tokens=True
).strip()

print("GROUNDED RAG ANSWER:\n", grounded_response)
```

#### Code Explanation:
<details>
<summary><b>Code Explanation:</b></summary>

* `collection.query(..., n_results=2)`: Retrieves the top 2 semantic matches based on vector proximity.
* `context_block = "\n".join(...)`: Combines individual retrieved passages into a structured text string.
* `messages = [...]`: Places the instructions into the `system` role and pairs the retrieved context with the user question into the `user` role.
* `torch.no_grad()`: Disables gradient tracking during generation, which saves GPU memory and speeds up execution.
* The model produces the correct name (**Dr. Elena Varga**) not because its weights changed, but because it extracted the fact directly from the injected prompt text.
</details>

---

#### 🧪 Checkpoint Challenge 2: Inspecting the Raw Prompt

> **Question:**  
> What does the exact text string passed to the GPU actually look like after `tokenizer.apply_chat_template` runs? Print `formatted_prompt` to the console. Where do the ChatML tags (`<|im_start|>`, `<|im_end|>`) appear relative to your retrieved context?

💡 **Gemini Prompt Hint:**  
> *"Explain how tokenizer.apply_chat_template in Hugging Face wraps user and system roles with ChatML control tokens for Qwen models."*

<details>
<summary><b>Answer & Explanation</b></summary>

Running `print(formatted_prompt)` reveals the underlying structure:

```text
<|im_start|>system
You are a helpful assistant for MediCore Hospital...<|im_end|>
<|im_start|>user
Context:
- Dr. Elena Varga leads the neurology department at MediCore Hospital.
- The neurology department handles brain and nervous system diseases...

Question: Who leads the neurology department at MediCore Hospital?<|im_end|>
<|im_start|>assistant
```

This makes it clear that RAG does not alter model mechanics. It is simply a structured prompt-assembly technique that formats retrieved facts directly into the assistant's immediate context window.
</details>

---

### Step 5: Mitigating Hallucination with Negative Constraints

A major vulnerability of generative models is generating convincing answers to questions about missing information. Test how the system prompt prevents hallucination when a topic is absent from the database.

```python
# [Cell 5] Negative Constraint Evaluation
unrecorded_query = "What is the name of MediCore Hospital's chief veterinary surgeon?"

# 1. Retrieve context for a non-existent topic
results = collection.query(query_texts=[unrecorded_query], n_results=2)
context_block = "\n".join([f"- {c}" for c in results["documents"][0]])

# 2. Format grounded prompt
messages = [
    {
        "role": "system",
        "content": (
            "You are a medical assistant for MediCore Hospital. "
            "Answer the question using ONLY the provided context. "
            "If the answer is not explicitly mentioned in the context, reply: 'Information not available.'"
        )
    },
    {
        "role": "user",
        "content": f"Context:\n{context_block}\n\nQuestion: {unrecorded_query}"
    }
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=40, do_sample=False)

reply = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

print(f"QUERY: {unrecorded_query}")
print(f"MODEL REFUSAL: {reply}")
```

#### Code Explanation:
<details>
<summary><b>Code Explanation:</b></summary>

* Even though ChromaDB will still return its 2 "closest" vectors (because vector search always returns the nearest items, regardless of absolute relevance), none of those retrieved chunks mention a veterinary department.
* Because the system prompt strictly instructed the model not to invent information, the model respects the negative constraint and outputs the designated fallback phrase.
</details>

---

### Step 6: Instant Knowledge Updates (The Core Advantage of RAG)

In Activity 4, updating an existing fact required retraining the LoRA adapter for several epochs. With RAG, updating the vector database instantly updates the model's generated output.

```python
# [Cell 6] Dynamic Knowledge Update Without Training
update_query = "Who leads the neurology department at MediCore Hospital?"

# 1. Update the document in ChromaDB
# Scenario: Dr. Elena Varga retired; Dr. Arto Virtanen was appointed
collection.update(
    ids=["fact_56"],
    documents=["Dr. Arto Virtanen leads the neurology department at MediCore Hospital."]
)

# 2. Re-run retrieval with the exact same query
results = collection.query(query_texts=[update_query], n_results=1)
updated_context = results["documents"][0][0]

messages = [
    {
        "role": "system", 
        "content": "Answer the question strictly based only on the provided context."
    },
    {"role": "user", "content": f"Context:\n{updated_context}\n\nQuestion: {update_query}"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50, do_sample=False)

updated_response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

print("UPDATED RAG RESPONSE:\n", updated_response)
```

#### Code Explanation:
<details>
<summary><b>Code Explanation:</b></summary>

* `collection.update(...)`: Replaces the underlying document text and recalculates the 384-dimensional vector in milliseconds.
* When the prompt is reassembled, the updated name is injected into the context window.
* Without any parameter updates, optimizer execution, or backpropagation, the model immediately outputs **Dr. Arto Virtanen**.
</details>

---

#### 🧪 Checkpoint Challenge 3: Synthesis — Architectural Trade-offs

> **Question:**  
> Contrast this step with your experience in Activity 4 (LoRA fine-tuning). If MediCore Hospital had 50 department heads that changed monthly, why is RAG a more sustainable architectural choice than fine-tuning? Conversely, when would fine-tuning still be necessary?

💡 **Gemini Prompt Hint:**  
> *"Contrast the maintenance overhead of RAG vs fine-tuning for rapidly changing factual data versus style/tone adaptation."*

<details>
<summary><b>Answer & Explanation</b></summary>

* **Why RAG is superior for frequently changing data:**  
  Updating records in a vector database takes milliseconds, requires negligible compute, and introduces zero risk of **catastrophic forgetting** (where updating weights causes the model to forget prior knowledge). Fine-tuning on constantly shifting facts is computationally expensive and slow to deploy.

* **When Fine-Tuning is still necessary:**  
  RAG cannot teach a model fundamentally new syntax, behavioral styles, custom token representations, or strict linguistic conventions (such as always responding in a specific medical triage tone or outputting compact raw binary data). Modern production architectures therefore use **fine-tuning to define how the model behaves** and **RAG to supply what the model knows**.
</details>

---

## Appendix: Structured JSON Generation in RAG

Just as in Activity 4 (Step 11), real-world systems often require RAG responses wrapped in structured JSON schemas for downstream API integration.

```python
# [Appendix Cell] Python-Enforced JSON Output with Source Citations
import json

def generate_rag_json(query):
    # 1. Retrieve
    retrieved = collection.query(query_texts=[query], n_results=2)
    chunks = retrieved["documents"][0]
    sources = retrieved["ids"][0]
    
    # 2. Generate
    context_str = "\n".join([f"[{sid}] {c}" for sid, c in zip(sources, chunks)])
    messages = [
        {"role": "system", "content": "Answer the question using the context. Keep it under 20 words."},
        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        
    answer_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # 3. Python-enforced structured envelope
    payload = {
        "query": query,
        "answer": answer_text,
        "sources_used": sources
    }
    
    return json.dumps(payload, indent=2)

print(generate_rag_json("What is the address of MediCore Hospital?"))
```

---

## Lab Summary

In this lab, you:
1. loaded the base instruction-tuned `Qwen2.5-1.5B-Instruct` model and verified its baseline ignorance regarding MediCore Hospital,
2. connected the **ChromaDB** retrieval index created in Lab 5A directly into an end-to-end inference workflow,
3. implemented a complete RAG system in pure Python without high-level framework abstractions,
4. formatted retrieved documents into Qwen’s native ChatML structure using `tokenizer.apply_chat_template()`,
5. used negative constraint instructions in the system prompt to eliminate hallucinations,
6. demonstrated instant knowledge updates via database modifications, bypassing the need for model retraining.