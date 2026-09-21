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
!pip install -q chromadb sentence-transformers pypdf python-docx
```

#### Code Explanation:
<details>
<summary><b>Code Explanation</b></summary>

* `chromadb`: An open-source, lightweight vector database designed to store document chunks, compute embeddings, and perform fast similarity search.
* `sentence-transformers`: A PyTorch-based framework that provides access to pre-trained transformer models engineered specifically to produce dense vector representations of sentences and paragraphs.
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
# [Cell 5] Instant Knowledge Update
print("BEFORE UPDATE:")
initial_search = collection.query(query_texts=["Who leads the neurology department?"], n_results=1)
print(initial_search["documents"][0][0])

# Scenario: Dr. Elena Varga has stepped down; Dr. Arto Virtanen is the new department head
collection.update(
    ids=["medicore_fact_56"],  # The ID corresponding to the neurology lead record
    documents=["Dr. Arto Virtanen leads the neurology department at MediCore Hospital."]
)

print("\nAFTER UPDATE:")
updated_search = collection.query(query_texts=["Who leads the neurology department?"], n_results=1)
print(updated_search["documents"][0][0])
```

#### Code Explanation:
<details>
<summary><b>Code Explanation</b></summary>

* `collection.update(...)`: Replaces the specified text chunk and recomputes its vector embedding immediately.
* This operation takes only a few milliseconds. In the next lab, when Qwen reads from this collection, it will immediately generate answers based on the new personnel without needing any model retraining.
</details>

---

## Appendix: Handling Unstructured Documents (PDF & DOCX)

In real applications, source knowledge rarely arrives in a pre-parsed `.json` file. It typically lives in `.pdf` manuals, `.docx` policies, or plain text files.

Below are standalone proofs-of-concept showing how to extract text from these file types so they can be chunked and indexed into ChromaDB.

### A. Extracting Text from a PDF (`pypdf`)

```python
# [Appendix Cell A] PDF Extraction Proof-of-Concept
from pypdf import PdfReader
import io

# 1. Create a minimal in-memory PDF for demonstration purposes
# (In practice, you would pass a path like: reader = PdfReader("hospital_policy.pdf"))
from pypdf import PdfWriter
writer = PdfWriter()
writer.add_blank_page(width=200, height=200)
pdf_stream = io.BytesIO()
writer.write(pdf_stream)
pdf_stream.seek(0)

# 2. Extract text page-by-page
def extract_text_from_pdf(file_source):
    reader = PdfReader(file_source)
    extracted_text = []
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            extracted_text.append(text)
            
    return "\n".join(extracted_text)

# Example usage:
# full_text = extract_text_from_pdf("my_policy.pdf")
print("PDF extraction function defined successfully.")
```

### B. Extracting Text from a Word Document (`python-docx`)

```python
# [Appendix Cell B] DOCX Extraction Proof-of-Concept
import docx

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    full_text = []
    
    # Extract text from every paragraph
    for para in doc.paragraphs:
        if para.text.strip():  # Skip empty lines
            full_text.append(para.text.strip())
            
    return "\n".join(full_text)

print("DOCX extraction function defined successfully.")
```

### C. Basic Fixed-Size Text Chunking

Once text is extracted from a PDF or DOCX file, it is usually too long to embed as a single vector. You divide it into smaller segments:

```python
# [Appendix Cell C] Simple Text Chunking Strategy
sample_long_document = """
MediCore Hospital Emergency Protocol:
All patients arriving with acute chest pain must undergo an immediate ECG within 10 minutes of arrival.
The triage nurse must assign an emergency severity index (ESI) of level 2 or higher.
The attending cardiologist on duty must be notified immediately via the direct emergency line.
Blood samples for cardiac troponin testing must be drawn at bedside upon triage completion.
"""

def chunk_text(text, chunk_size=150, overlap=30):
    """Splits text into chunks of roughly chunk_size characters with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

chunks = chunk_text(sample_long_document, chunk_size=120, overlap=20)

print(f"Divided long text into {len(chunks)} chunks:")
for i, c in enumerate(chunks):
    print(f"Chunk {i+1}: {c}")
```

#### Code Explanation:
<details>
<summary><b>Code Explanation</b></summary>

* `chunk_size`: The maximum character or token length of each chunk. Small chunks (e.g., 100–300 words) ensure the embedding model focuses on specific facts rather than diluting meaning across multiple topics.
* `overlap`: Keeps a small portion of overlapping text between adjacent chunks (e.g., 20–50 characters). This prevents sentences or thoughts from being abruptly cut in half across a boundary, ensuring context is preserved across splits.
</details>

---

## Summary and Next Steps

In this lab, you:
1. converted sentences into 384-dimensional vectors and evaluated semantic distance using cosine similarity,
2. indexed the `MediCore.json` factual database into an in-memory **ChromaDB** collection,
3. performed natural language queries to retrieve context without exact keyword matching,
4. demonstrated how vector database updates immediately change retrieval results with zero retraining overhead,
5. explored how raw text is extracted from `.pdf` and `.docx` files and divided into chunks.

In **Activity 4**, you will connect this retrieval mechanism to the base `Qwen2.5-1.5B-Instruct` model to produce grounded, hallucination-free answers.