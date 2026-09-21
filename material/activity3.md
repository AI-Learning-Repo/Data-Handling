# Activity 3: Semantic Search and Vector Retrieval with ChromaDB  (RAG Part 1)

In this lab, you will:

1. prepare the Colab environment for vector operations,
2. understand how text embeddings convert natural language into mathematical coordinates,
3. calculate semantic similarity between concepts without using keyword matching,
4. initialize and configure **ChromaDB**, an embedded vector database,
5. load and index domain knowledge from `MediCore.json`,
6. execute semantic queries using top-$k$ similarity retrieval,
7. filter query results using document metadata,
8. explore how to extract raw text from unstructured formats (PDF and DOCX).

> [!NOTE]
> This lab focuses entirely on the **Retrieval ("R")** component of Retrieval-Augmented Generation (RAG). You will not generate responses with an LLM in this lab. Instead, you will build and evaluate the semantic search engine that supplies retrieved context to Qwen in the next lab.

---

### Prerequisite: Compute Environment

Vector generation and database indexing in this lab use small embedding models (`all-MiniLM-L6-v2`). These run efficiently on standard CPU runtimes. However, if you already have a **T4 GPU** runtime active in Colab, the libraries will automatically take advantage of it.

---

## Core Workflow

### Step 0: Install Dependencies

Run this cell to install the vector database and embedding libraries:

```python
# [Cell 0] Install Dependencies
!pip install -q \
    "opentelemetry-api>=1.39.0,<=1.42.1" \
    "opentelemetry-sdk>=1.39.0,<=1.42.1" \
    chromadb \
    sentence-transformers \
    pypdf \
    python-docx
```

#### Code Explanation:

<details>
<summary><b>Code Explanation:</b></summary>

* `chromadb`: An open-source, lightweight vector database designed to store document chunks, compute embeddings, and perform fast similarity search.
* `sentence-transformers`: A PyTorch-based framework that provides access to pre-trained transformer models engineered specifically to produce dense vector representations of sentences and paragraphs.
* `opentelemetry-api` & `opentelemetry-sdk`: Observability and telemetry libraries used internally by ChromaDB to trace operations and log metrics. We explicitly pin their versions (`>=1.39.0,<=1.42.1`) to maintain compatibility with Google Colab's pre-installed environment packages (specifically `google-adk`) and prevent pip resolver dependency conflicts.
* `pypdf` & `python-docx`: Lightweight text extraction libraries used in the appendix to read unstructured documents.

</details>

---

### Step 1: Semantic Intuition and Vector Embeddings

Before working with a vector database, examine what an embedding actually looks like and how mathematical distance relates to semantic meaning.

```python
# [Cell 1] Vector Embedding Intuition
from sentence_transformers import SentenceTransformer
import numpy as np

# Load a lightweight, industry-standard sentence embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Three sentences: two share meaning, one is unrelated
sentences = [
    "The physician examined the patient.",
    "A doctor checked the sick individual.",
    "The sports car drove down the highway."
]

# Generate dense vector embeddings (384 floating-point numbers each)
embeddings = embed_model.encode(sentences)

print(f"Embedding shape for each sentence: {embeddings[0].shape}")
print(f"First 5 dimensions of sentence 1:\n{embeddings[0][:5]}\n")

# Define Cosine Similarity calculation
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])

print(f"Similarity between 'physician' and 'doctor' sentences: {sim_1_2:.4f}")
print(f"Similarity between 'physician' and 'sports car' sentences: {sim_1_3:.4f}")
```

#### Code Explanation:
<details>
<summary><b>Code Explanation</b></summary>

* `SentenceTransformer("all-MiniLM-L6-v2")`: Loads a compact 22-million parameter model. It converts any arbitrary string into a fixed-size array of 384 numbers.
* `embeddings[0].shape`: Shows `(384,)`. Regardless of whether an input sentence has 3 words or 50 words, its embedding vector always has the exact same coordinate dimension.
* `cosine_similarity(v1, v2)`: Measures the cosine of the angle between two vectors in 384-dimensional space.
  * A value near `1.0` means the two sentences point in nearly the identical direction in concept-space (high semantic similarity).
  * A value near `0.0` means the vectors are orthogonal (no semantic relationship).
</details>

---

#### 🧪 Checkpoint Challenge 1: Exploring Vector Geometry

> **Question:**  
> Sentence 1 (*"The physician examined the patient"*) and Sentence 2 (*"A doctor checked the sick individual"*) share almost no common words except "the". Why does their cosine similarity score remain close to `0.80` or higher, while a standard SQL `LIKE %doctor%` or Python string search would score this as a zero match?

💡 **Gemini Prompt Hint:**  
> *"Explain how dense sentence embeddings capture semantic similarity between synonyms compared to lexical/keyword matching. Keep the explanation concise and technical."*

<details>
<summary><b>Answer & Explanation</b></summary>

Keyword search (lexical matching) looks only for identical character sequences. Because `"physician"` $\neq$ `"doctor"` and `"examined"` $\neq$ `"checked"`, keyword algorithms register zero overlap.

Dense embedding models like `all-MiniLM-L6-v2` were trained on hundreds of millions of sentence pairs using contrastive learning. The model places words and sentences that appear in similar contexts near one another in a continuous 384-dimensional vector space. As a result, the coordinates for medical synonyms point in nearly the same mathematical direction, producing a high cosine similarity score.
</details>

---

### Step 2: Initialize ChromaDB and the Embedding Function

A vector database automates vector computation, indexing, and storage. ChromaDB can run entirely in-memory inside your notebook session.

```python
# [Cell 2] Initialize ChromaDB
import chromadb
from chromadb.utils import embedding_functions

# 1. Instantiate an in-memory client
chroma_client = chromadb.Client()

# 2. Configure Chroma to automatically use all-MiniLM-L6-v2 for all additions and queries
st_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# 3. Create a collection (equivalent to a table in SQL)
collection = chroma_client.get_or_create_collection(
    name="medicore_knowledge",
    embedding_function=st_embedding_function
)

print(f"Collection '{collection.name}' initialized successfully.")
```

#### Code Explanation:
<details>
<summary><b>Code Explanation</b></summary>

* `chromadb.Client()`: Creates an ephemeral, in-memory instance of ChromaDB that lives within the Colab RAM. For permanent disk storage, `chromadb.PersistentClient(path="./my_vectordb")` would be used instead.
* `embedding_functions.SentenceTransformerEmbeddingFunction(...)`: Links the embedding model directly to the collection. Whenever you submit raw text to Chroma, it runs the embedding step internally. You do not need to manually call `.encode()`.
* `get_or_create_collection(...)`: Creates a storage partition. If the collection already exists, it loads it; if not, it creates a new one.
</details>

---

### Step 3: Load and Ingest the MediCore Knowledge Base

Download the same `MediCore.json` dataset used in Activity 4 and ingest it into ChromaDB. Each entry will be indexed as a standalone document chunk with metadata.

```python
# [Cell 3a] Download the dataset if not already present
!wget -nc -q https://raw.githubusercontent.com/AI-Learning-Repo/Data-Handling/refs/heads/week4/datasets/MediCore.json
```

```python
# [Cell 3b] Ingest Data into ChromaDB
import json

# 1. Read the JSON file
with open("MediCore.json", "r", encoding="utf-8") as f:
    lines = f.readlines()

documents = []
metadatas = []
ids = []

for idx, line in enumerate(lines):
    item = json.loads(line.strip())
    
    # Store the factual completion as the searchable text
    documents.append(item["completion"])
    
    # Extract simple metadata: associate the question prompt as context
    metadatas.append({"source_prompt": item["prompt"], "doc_index": idx})
    ids.append(f"medicore_fact_{idx}")

# 2. Add records to the vector collection in a single batch
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print(f"Successfully indexed {collection.count()} document chunks into ChromaDB.")
```

#### Code Explanation:
<details>
<summary><b>Code Explanation</b></summary>

* `documents.append(item["completion"])`: The actual text that will be embedded and searched. In our dataset, each `completion` is a self-contained factual statement about MediCore Hospital (e.g., *"Dr. Elena Varga leads the neurology department at MediCore Hospital."*).
* `metadatas`: A list of key-value dictionaries attached to each document. Metadata is not transformed into vectors, but it allows for filtering results later (e.g., filtering by department or category).
* `ids`: Every item in ChromaDB requires a unique string identifier.
* `collection.add(...)`: Passes the text chunks to the configured embedding function, calculates their vectors, and writes them into Chroma's search index.
</details>

---

#### 🧪 Checkpoint Challenge 2: Understanding Vector Chunks

> **Question:**  
> In `activity4.md` (LoRA fine-tuning), we trained the model on pairs of `{"prompt": ..., "completion": ...}` formatted with `<|im_start|>` and `<|im_end|>` ChatML tags. Why do we store only the factual completion sentences in ChromaDB here, without any ChatML tags?

💡 **Gemini Prompt Hint:**  
> *"In a RAG retrieval pipeline, why should documents stored in a vector database be clean factual prose rather than chat-templated prompts? Consider embedding distance and relevance."*

<details>
<summary><b>Answer & Explanation</b></summary>

Embedding models measure semantic similarity between concepts. If you embed special tokens like `<|im_start|>assistant`, they add noise to the vector representation without adding semantic value. 

Furthermore, during RAG retrieval, the user asks a factual question (e.g., *"Who leads neurology?"*). The vector for this query should align directly with the vector for the factual answer (*"Dr. Elena Varga leads the neurology department..."*). Special prompt formatting is only applied later when constructing the generation prompt for the LLM.
</details>

---

### Step 4: Semantic Querying (Top-$k$ Retrieval)

Query ChromaDB using natural language queries that differ from the exact sentences stored in the database.

```python
# [Cell 4] Query ChromaDB
query_text = "Who is in charge of brain and nervous system conditions?"

# Retrieve the top 2 closest semantic matches
results = collection.query(
    query_texts=[query_text],
    n_results=2
)

# Inspect the returned results
print(f"QUERY: {query_text}\n")
for i in range(len(results["documents"][0])):
    doc = results["documents"][0][i]
    distance = results["distances"][0][i]
    doc_id = results["ids"][0][i]
    print(f"Rank {i+1} [Distance: {distance:.4f}] [ID: {doc_id}]:")
    print(f"Content: {doc}\n")
```

#### Code Explanation:
<details>
<summary><b>Code Explanation</b></summary>

* `collection.query(...)`:
  1. Automatically embeds `query_text` into a 384-dimensional query vector using `all-MiniLM-L6-v2`.
  2. Measures the distance between the query vector and every document vector in the collection.
  3. Returns the top `n_results` closest matches.
* `results["distances"]`: By default, ChromaDB calculates **Squared $L2$ Distance** (Euclidean) or Cosine Distance. Lower distance values indicate higher similarity.
</details>

---

#### 🧪 Checkpoint Challenge 3: Evaluating Semantic Retrieval

> **Question:**  
> Notice that the query asked about *"brain and nervous system conditions"*, but the top returned result mentions *"Dr. Elena Varga leads the neurology department"* and *"The neurology department handles brain and nervous system diseases"*.  
> 
> Try modifying `query_text` to: `"Where do the helicopters land?"`  
> What does ChromaDB return, and what is its distance score? Does it work even though the word "pad" or "helipad" was not in your query?

💡 **Gemini Prompt Hint:**  
> *"In ChromaDB vector retrieval, run a conceptual query that uses no matching nouns from the underlying document and explain how nearest-neighbor search identifies the correct concept."*

<details>
<summary><b>Answer & Explanation</b></summary>

When you query `"Where do the helicopters land?"`, ChromaDB returns:
`"MediCore Hospital has a rooftop helipad for emergency patient transport."`

Even though the word "helipad" was not present in the query, the embedding model maps the phrase "helicopters land" to virtually the same area in vector space as "helipad". This illustrates why semantic retrieval is significantly more flexible for question answering than traditional keyword searching.
</details>

---

### Step 5: Updating Knowledge in Real Time (The RAG Advantage)

In Activity 4, updating an existing fact required retraining the model. In a vector database, changing a fact is an instantaneous database operation.

```python
# [Cell 5] Instant Knowledge Update (CEO Leadership Transition)

target_id = "medicore_fact_71"
query = "Who is the CEO of MediCore Hospital?"

# Step 1: Ensure the record is set to the baseline value (allows clean re-runs)
collection.update(
    ids=[target_id],
    documents=["The CEO of MediCore Hospital is Juhani Aho."]
)

# Step 2: Query the database before modification
print("--- BEFORE UPDATE ---")
search_before = collection.query(query_texts=[query], n_results=1)
doc_before = search_before["documents"][0][0]
dist_before = search_before["distances"][0][0]
print(f"Rank 1 [Distance: {dist_before:.4f}]: {doc_before}")

# Step 3: Mutate the record in the vector database
# Scenario: Juhani Aho retires; Milla Kallio is appointed as the new CEO
collection.update(
    ids=[target_id],
    documents=["The CEO of MediCore Hospital is Milla Kallio."]
)

# Step 4: Query the database after modification
print("\n--- AFTER UPDATE ---")
search_after = collection.query(query_texts=[query], n_results=1)
doc_after = search_after["documents"][0][0]
dist_after = search_after["distances"][0][0]
print(f"Rank 1 [Distance: {dist_after:.4f}]: {doc_after}")
```

#### Code Explanation:
<details>
<summary><b>Code Explanation:</b></summary>

* `target_id = "medicore_fact_71"`: Refers to the specific index assigned to the CEO record during ingestion in Step 3 (`lines[71]`).
* `collection.update(ids=[target_id], documents=[...])`: 
  1. Locates the existing document key in the ChromaDB index.
  2. Passes the updated text string (`"The CEO of MediCore Hospital is Milla Kallio."`) through `all-MiniLM-L6-v2` to compute a new 384-dimensional vector coordinate.
  3. Overwrites both the document text and the vector in the index.
* **Why the result is clean at $k=1$:** Unlike the neurology department (which had a general department description competing with the leadership record), the CEO topic has only one reference in the database. As a result, the distance score is very low ($\approx 0.12$), and the document holds Rank 1 with no semantic interference.
* **Idempotence:** Step 1 explicitly sets the document back to *"Juhani Aho"* before querying. This guarantees that whether a student runs the cell once or ten times in succession, the output will always demonstrate a clear before-and-after transition.

</details>

---

## Appendix: Handling Unstructured Documents (PDF & DOCX)

In real applications, source knowledge rarely arrives in a pre-parsed `.json` file. It typically lives in `.pdf` manuals, `.docx` policies, or plain text files.

Here's an  end-to-end proof-of-Concept. It programmatically generates a sample .docx and .pdf file in Colab, extracts their text, chunks them, loads them into ChromaDB, and performs a semantic search with source attribution.

```python
# [Appendix Cell] End-to-End Proof of Concept: Unstructured Files (DOCX & PDF) to ChromaDB
!pip install -q reportlab

import os
import docx
from pypdf import PdfReader
from reportlab.pdfgen import canvas
import chromadb
from chromadb.utils import embedding_functions

# =====================================================================
# Step 1: Create sample DOCX and PDF files directly in Colab
# =====================================================================

# 1a. Create sample Word Document
doc_path = "hospital_policy.docx"
doc = docx.Document()
doc.add_heading("MediCore Hospital Acute Care Protocols", level=1)
doc.add_paragraph(
    "All patients arriving with acute chest pain must receive an immediate 12-lead ECG "
    "within 10 minutes of registration. The attending cardiologist must be paged immediately."
)
doc.add_paragraph(
    "Emergency stroke patients require an immediate non-contrast head CT scan. "
    "Thrombolytic therapy must be evaluated within 45 minutes of door arrival."
)
doc.save(doc_path)

# 1b. Create sample PDF Document
pdf_path = "surgical_protocols.pdf"
c = canvas.Canvas(pdf_path)
c.drawString(72, 750, "MediCore Hospital Surgical Division Policy:")
c.drawString(72, 730, "Robotic-assisted surgery suites require full UV-C terminal sterilization.")
c.drawString(72, 710, "Surgeons must complete 3D virtual simulation before operating with the MediBot.")
c.save()

print("Generated sample files: hospital_policy.docx, surgical_protocols.pdf")

# =====================================================================
# Step 2: Extraction Functions
# =====================================================================

def extract_from_docx(file_path):
    d = docx.Document(file_path)
    paragraphs = [p.text.strip() for p in d.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)

def extract_from_pdf(file_path):
    reader = PdfReader(file_path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text.strip())
    return "\n".join(pages_text)

docx_text = extract_from_docx(doc_path)
pdf_text = extract_from_pdf(pdf_path)

# =====================================================================
# Step 3: Fixed-Size Text Chunking with Overlap
# =====================================================================

def chunk_text(text, source_name, chunk_size=150, overlap=30):
    chunks = []
    metadatas = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            metadatas.append({"source": source_name, "char_start": start})
        start += (chunk_size - overlap)
    return chunks, metadatas

docx_chunks, docx_meta = chunk_text(docx_text, source_name="hospital_policy.docx")
pdf_chunks, pdf_meta = chunk_text(pdf_text, source_name="surgical_protocols.pdf")

all_chunks = docx_chunks + pdf_chunks
all_metadatas = docx_meta + pdf_meta
all_ids = [f"unstructured_chunk_{i}" for i in range(len(all_chunks))]

# =====================================================================
# Step 4: Ingest into a Dedicated ChromaDB Collection
# =====================================================================

chroma_client = chromadb.Client()
st_embed = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

unstructured_collection = chroma_client.create_collection(
    name="unstructured_demo",
    embedding_function=st_embed
)

unstructured_collection.add(
    documents=all_chunks,
    metadatas=all_metadatas,
    ids=all_ids
)

print(f"Indexed {unstructured_collection.count()} chunks from DOCX and PDF into ChromaDB.\n")

# =====================================================================
# Step 5: Query Across Document Types
# =====================================================================

query = "What is the procedure for emergency chest pain?"

results = unstructured_collection.query(
    query_texts=[query],
    n_results=1
)

retrieved_doc = results["documents"][0][0]
source_file = results["metadatas"][0][0]["source"]
distance = results["distances"][0][0]

print(f"QUERY: {query}")
print(f"MATCH FROM SOURCE: [{source_file}] (Distance: {distance:.4f})")
print(f"CONTENT: {retrieved_doc}")
```


---

## Summary and Next Steps

In this lab, you:
1. converted sentences into 384-dimensional vectors and evaluated semantic distance using cosine similarity,
2. indexed the `MediCore.json` factual database into an in-memory **ChromaDB** collection,
3. performed natural language queries to retrieve context without exact keyword matching,
4. demonstrated how vector database updates immediately change retrieval results with zero retraining overhead,
5. explored how raw text is extracted from `.pdf` and `.docx` files and divided into chunks.

In **Activity 4**, you will connect this retrieval mechanism to the base `Qwen2.5-1.5B-Instruct` model to produce grounded, hallucination-free answers.

---

## Links

- [RAG with Python Cookbook](https://github.com/polzerdo55862/RAG-with-Python-Cookbook/tree/main)