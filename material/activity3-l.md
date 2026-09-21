# Activity 3: Semantic Search and Vector Retrieval with ChromaDB

## Learning Objectives

In this lab, you will:

1. Prepare the Colab environment for vector operations.
2. Understand how text embeddings convert natural language into mathematical coordinates.
3. Calculate semantic similarity between concepts without relying only on keyword matching.
4. Initialize and configure ChromaDB, a vector database.
5. Load and index domain knowledge from `MediCore.json`.
6. Execute semantic queries using top-k similarity retrieval.
7. Filter query results using document metadata.
8. Explore how to extract raw text from PDF and DOCX files.
9. Understand how document updates affect the retrieval system.

> **Scope:** This lab focuses on the Retrieval (R) component of Retrieval-Augmented Generation. You will not generate responses with an LLM in this lab. Instead, you will build and explore the semantic search system that will supply context to Qwen in the next lab.

---

# Step 0: Install Dependencies

Run the following cell to install the required libraries.

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

### Code Explanation

| Library | Purpose |
| :--- | :--- |
| `chromadb` | Vector database for storing documents, embeddings, and performing similarity search. |
| `sentence-transformers` | Provides pretrained models for generating sentence and document embeddings. |
| `opentelemetry-api` and `opentelemetry-sdk` | Observability and telemetry dependencies used by parts of the Python ecosystem, including integrations used by ChromaDB. |
| `pypdf` | Extracts text from text-based PDF files. |
| `python-docx` | Reads and extracts content from Microsoft Word DOCX files. |

### Technical note about version pinning

The original lab pins specific OpenTelemetry versions. These constraints may have been selected for compatibility with a particular Colab environment.

However, they should not be presented as a universal requirement for all environments. Dependency compatibility can change over time.

If the installation works without restrictive version pins, you can avoid unnecessary constraints. If the pins are needed for a specific environment, explain that they are environment-specific.

---

# Step 1: Semantic Intuition and Vector Embeddings

Before using a vector database, we will explore how sentences are converted into vectors and how their similarity can be measured.

## Code

```python
# [Cell 1] Vector Embedding Intuition

from sentence_transformers import SentenceTransformer
import numpy as np

# Load a lightweight sentence embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Three sentences: two share meaning, one is unrelated
sentences = [
    "The physician examined the patient.",
    "A doctor checked the sick individual.",
    "The sports car drove down the highway."
]

# Generate dense vector embeddings
embeddings = embed_model.encode(sentences)

print(f"Embedding shape for each sentence: {embeddings[0].shape}")
print(f"First 5 dimensions of sentence 1:\n{embeddings[0][:5]}\n")

# Define Cosine Similarity calculation
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )

sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])

print(
    f"Similarity between 'physician' and 'doctor' sentences: "
    f"{sim_1_2:.4f}"
)

print(
    f"Similarity between 'physician' and 'sports car' sentences: "
    f"{sim_1_3:.4f}"
)
```

## Code Explanation

### 1. Loading the model

```python
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
```

This loads a pretrained sentence embedding model.

The model generates a vector representation for each input sentence. With this model, the output embedding has 384 dimensions.

The model is useful for introductory semantic similarity experiments. Its performance should not be assumed to be optimal for all medical, technical, or multilingual retrieval tasks.

### 2. Generating embeddings

```python
embeddings = embed_model.encode(sentences)
```

The `encode()` method converts the list of sentences into numerical vectors.

The output can be viewed as an array containing one vector per sentence.

```python
embeddings[0].shape
```

Expected shape:

```text
(384,)
```

This means that the first sentence is represented by a vector containing 384 values.

### 3. Calculating cosine similarity

```python
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )
```

This function computes the cosine similarity between two vectors.

The formula is:

$$\text{cosine similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

The function compares the direction of the vectors rather than simply comparing their raw numerical values.

### Important observation

You should not expect a guaranteed similarity value such as `0.80`. The exact score depends on the model and the input text.

Run the code and record the actual values returned in your environment.

## Checkpoint Challenge 1: Exploring Vector Geometry

### Question

Sentence 1:
> The physician examined the patient.

Sentence 2:
> A doctor checked the sick individual.

The sentences share very few exact words. Why can their cosine similarity still be relatively high, while a basic keyword search might not identify a match?

### Answer and Explanation

Keyword search generally looks for matching words or character sequences. A basic search for `doctor` will not necessarily match the word `physician`.

A sentence embedding model learns representations from training data that can capture relationships between words and sentences. As a result, terms and expressions with similar meanings may be represented in similar regions of the embedding space.

This can result in a high cosine similarity between the two sentences.

However, the exact similarity value is not guaranteed, and a high score does not prove that two sentences are factually equivalent.

---

# Step 2: Initialize ChromaDB and the Embedding Function

A vector database stores documents and supports similarity-based retrieval.

In this lab, we use ChromaDB.

## What is ChromaDB?

ChromaDB is a vector database that can store documents, embeddings, and metadata. It also supports querying the database using natural-language text.

Instead of manually generating an embedding for every query, we can configure ChromaDB with an embedding function.

When we add documents or perform a text query, ChromaDB uses the configured embedding function to generate the relevant embeddings.

## Code

```python
# [Cell 2] Initialize ChromaDB

import chromadb
from chromadb.utils import embedding_functions

# 1. Instantiate an in-memory client
chroma_client = chromadb.Client()

# 2. Configure the embedding function
st_embedding_function = (
    embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)

# 3. Create or retrieve a collection
collection = chroma_client.get_or_create_collection(
    name="medicore_knowledge",
    embedding_function=st_embedding_function
)

print(f"Collection '{collection.name}' initialized successfully.")
```

## Code Explanation

### 1. Creating a ChromaDB client

```python
chroma_client = chromadb.Client()
```

This creates an in-memory ChromaDB client.

The data is associated with the current runtime session. It should not be treated as permanent storage.

For persistent local storage, you could use:

```python
chroma_client = chromadb.PersistentClient(
    path="./my_vectordb"
)
```

This provides a persistent storage configuration rather than a temporary in-memory client.

### 2. Configuring the embedding function

```python
st_embedding_function = (
    embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)
```

This connects ChromaDB to the Sentence Transformers model.

When text is added to the collection, the embedding function is used to create embeddings.

When a query is performed using `query_texts`, the same embedding function is used to represent the query.

**Important:** The same compatible embedding model should be used for the stored documents and queries. Changing the embedding model can change the vector space and affect retrieval behavior.

### 3. Creating a collection

```python
collection = chroma_client.get_or_create_collection(
    name="medicore_knowledge",
    embedding_function=st_embedding_function
)
```

A collection is a logical group of documents and their associated data.

The collection stores:
* Document IDs.
* Document text.
* Embeddings.
* Metadata.

`get_or_create_collection()` returns an existing collection if one with the same name is available, or creates a new one if it does not exist.

### Distance metric note

The original lab describes ChromaDB's distance as both squared L2 and cosine distance. That is inaccurate because the distance metric depends on the collection configuration.

If the lab needs a specific distance metric, configure it explicitly when creating the collection and explain its meaning. Also remember that reusing an existing collection name may return a collection with its original configuration.

---

# Step 3: Load and Ingest the MediCore Knowledge Base

We will now load the `MediCore.json` dataset and store its factual information in ChromaDB.

Each entry will be indexed as a document with metadata.

## Step 3a: Download the dataset

```python
# [Cell 3a] Download the dataset if not already present

!wget -nc -q \
    https://raw.githubusercontent.com/AI-Learning-Repo/Data-Handling/refs/heads/week4/datasets/MediCore.json
```

### Explanation

This command downloads the dataset from the specified GitHub repository.

The `-nc` option prevents `wget` from downloading the file again if the file already exists locally.

**Important:** Confirm that the dataset URL is accessible and that the file format matches the ingestion code below.

## Step 3b: Ingest data into ChromaDB

The original code assumes that `MediCore.json` is in JSON Lines (JSONL/NDJSON) format.

In this format, each line contains one JSON object.

Example:

```json
{"prompt": "Who leads neurology?", "completion": "Dr. Elena Varga leads neurology."}
{"prompt": "Where do helicopters land?", "completion": "The hospital has a rooftop helipad."}
```

If the file is a standard JSON array instead, the ingestion code must be adjusted.

### Code

```python
# [Cell 3b] Ingest Data into ChromaDB

import json

# 1. Read the JSON Lines file
with open("MediCore.json", "r", encoding="utf-8") as f:
    lines = f.readlines()

documents = []
metadatas = []
ids = []

for idx, line in enumerate(lines):
    if not line.strip():
        continue

    item = json.loads(line.strip())

    # Store the factual completion as searchable text
    documents.append(item["completion"])

    # Store the original prompt as metadata
    metadatas.append({
        "source_prompt": item["prompt"],
        "doc_index": idx
    })

    # Generate a unique ID
    ids.append(f"medicore_fact_{idx}")

# 2. Add records to the vector collection
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print(
    f"Successfully indexed {collection.count()} "
    "document chunks into ChromaDB."
)
```

## Code Explanation

### 1. Reading the data

```python
with open("MediCore.json", "r", encoding="utf-8") as f:
    lines = f.readlines()
```

This reads the file line by line.

It is appropriate if the file contains one JSON object per line.

If the file contains a single JSON array, `json.load(f)` would be more appropriate.

### 2. Selecting the document text

```python
documents.append(item["completion"])
```

The code stores the factual completion as the searchable document.

For example:

```text
The neurology department handles brain and nervous system diseases.
```

This allows the retrieval system to compare the user's query against the factual information.

### Why not store the ChatML formatting?

The original dataset was used for fine-tuning, where prompts and completions may have been formatted using ChatML tokens.

The retrieval database does not need those training-specific chat markers.

For example, the following is not necessary for the retrieval text:

```text
<|im_start|>assistant
The neurology department handles diseases.
<|im_end|>
```

Instead, the database can store clean source text.

*Clarification:* ChatML tokens are not automatically harmful in every embedding workflow, but they generally do not provide useful factual content for this retrieval task. Keeping the stored text clean simplifies indexing and makes the document content easier to inspect.

### 3. Storing metadata

```python
metadatas.append({
    "source_prompt": item["prompt"],
    "doc_index": idx
})
```

Metadata is additional structured information associated with a document.

It is useful for:
* Tracking the original question.
* Identifying the source record.
* Filtering documents.
* Providing source attribution.
* Debugging retrieval results.

Metadata is not necessarily embedded together with the document. It is stored as structured information that can be used for filtering and reference.

### 4. Document IDs

```python
ids.append(f"medicore_fact_{idx}")
```

Every record added to the collection needs a unique ID.

IDs make it possible to:
* Identify a specific document.
* Update a document.
* Delete a document.
* Track a document across experiments.

### 5. Adding documents to ChromaDB

```python
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)
```

ChromaDB processes the documents using the configured embedding function and stores the resulting data.

## Checkpoint Challenge 2: Understanding Vector Documents

### Question

In the previous fine-tuning activity, the model was trained on pairs of:

```json
{
  "prompt": "...",
  "completion": "..."
}
```

Why do we store the factual completion rather than the entire ChatML-formatted training example?

### Answer and Explanation

The retrieval system's goal is to find relevant source information.

The factual completion contains the information that we want to retrieve and provide as context to the generation model.

The training format is designed for a specific model training process. It is not necessary to reproduce that format in the stored retrieval document.

Using clean factual text can make the document representation easier to inspect and reduce irrelevant formatting content.

However, including the prompt or other contextual information can also be useful in some retrieval systems. The appropriate choice depends on the structure of the knowledge base and the retrieval task.

---

# Step 4: Semantic Querying (Top-k Retrieval)

We will now query ChromaDB using a natural-language question.

The query will not necessarily use the exact wording of the stored documents.

## Code

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

    print(
        f"Rank {i+1} "
        f"[Distance: {distance:.4f}] "
        f"[ID: {doc_id}]:"
    )

    print(f"Content: {doc}\n")
```

## Code Explanation

### 1. Defining the query

```python
query_text = "Who is in charge of brain and nervous system conditions?"
```

This is the question we want to answer using the knowledge base.

The query is written in natural language and does not have to exactly match the wording of the stored document.

### 2. Retrieving the top-k documents

```python
results = collection.query(
    query_texts=[query_text],
    n_results=2
)
```

The parameter `n_results=2` asks ChromaDB to return up to two matching results for the query.

This is called top-k retrieval.

Here:

$$k = 2$$

The system returns the two highest-ranked candidates according to its similarity search configuration.

### 3. Understanding the output

The results include information such as:
* `documents`: The retrieved document text.
* `ids`: The document identifiers.
* `distances`: The distance values.
* `metadatas`: The associated metadata, if requested or returned by the API.

Lower distance values indicate closer vectors under the configured distance metric.

**Important:** A low distance does not guarantee that a document answers the question correctly. Relevance should be evaluated by inspecting the returned content.

### Why does semantic retrieval work?

The embedding model represents the query and documents in a vector space. The system compares the query representation with the stored document representations.

A query containing:
> brain and nervous system conditions

may retrieve a document containing:
> neurology department handles brain and nervous system diseases

because the embedding model can capture related semantic patterns.

## Checkpoint Challenge 3: Evaluating Semantic Retrieval

### Question 1

The query asks:
> Who is in charge of brain and nervous system conditions?

Why might the system retrieve a document about the neurology department?

### Answer

The query refers to a medical specialty involving the brain and nervous system. The word `neurology` is associated with that subject.

The embedding model can identify semantic relationships between the query and the stored document, even if the exact phrasing differs.

The retrieved document should still be inspected to determine whether it contains the specific answer to the question.

### Question 2

Try modifying the query:

```python
query_text = "Where do the helicopters land?"
```

Run the retrieval code.

What does ChromaDB return, and what is the distance score?

### Expected learning outcome

The database may retrieve a document describing the hospital's rooftop helipad if that document exists and the embedding model identifies the relationship between the query and the stored text.

However, the exact result and distance score must be observed by running the code. The original lab's assertion that a specific document will always be returned is not guaranteed across different models, datasets, or configurations.

### Suggested investigation

Compare these queries:

```python
queries = [
    "Where do the helicopters land?",
    "Where is the emergency helicopter landing area?",
    "What is the location of the hospital helipad?"
]

for query in queries:
    results = collection.query(
        query_texts=[query],
        n_results=2
    )

    print(f"\nQUERY: {query}")

    for i, doc in enumerate(results["documents"][0]):
        print(
            f"Rank {i + 1}: "
            f"{doc} "
            f"(Distance: {results['distances'][0][i]:.4f})"
        )
```

This experiment helps students understand that different queries can change the ranking of retrieved documents.

---

# Step 5: Updating Knowledge in Real Time

One advantage of a retrieval-based knowledge system is that we can update stored information without retraining the generation model.

For example, imagine that the hospital CEO changes.

The original record:
> The CEO of MediCore Hospital is Juhani Aho.

The updated record:
> The CEO of MediCore Hospital is Milla Kallio.

The retrieval system should use the updated information after the record has been modified.

## Code

```python
# [Cell 5] Instant Knowledge Update

target_id = "medicore_fact_71"

query = "Who is the CEO of MediCore Hospital?"

# Step 1: Reset the record to the baseline value
collection.update(
    ids=[target_id],
    documents=[
        "The CEO of MediCore Hospital is Juhani Aho."
    ]
)

# Step 2: Query the database before modification
print("--- BEFORE UPDATE ---")

search_before = collection.query(
    query_texts=[query],
    n_results=1
)

doc_before = search_before["documents"][0][0]
dist_before = search_before["distances"][0][0]

print(
    f"Rank 1 [Distance: {dist_before:.4f}]: "
    f"{doc_before}"
)

# Step 3: Update the record
collection.update(
    ids=[target_id],
    documents=[
        "The CEO of MediCore Hospital is Milla Kallio."
    ]
)

# Step 4: Query the database after modification
print("\n--- AFTER UPDATE ---")

search_after = collection.query(
    query_texts=[query],
    n_results=1
)

doc_after = search_after["documents"][0][0]
dist_after = search_after["distances"][0][0]

print(
    f"Rank 1 [Distance: {dist_after:.4f}]: "
    f"{doc_after}"
)
```

## Code Explanation

### 1. Selecting the document ID

```python
target_id = "medicore_fact_71"
```

This identifies the record to be updated.

*Technical accuracy note:* The assumption that ID `medicore_fact_71` corresponds to the CEO record must be verified using the actual dataset. The index number alone does not establish the record's content.

A safer approach is to inspect the ingested data or search for the CEO record before assigning the target ID.

### 2. Updating a document

```python
collection.update(
    ids=[target_id],
    documents=[
        "The CEO of MediCore Hospital is Milla Kallio."
    ]
)
```

The database updates the document associated with the specified ID.

The embedding for the updated text needs to reflect the new content. ChromaDB handles the embedding update using its configured embedding function.

### 3. Why is retraining not required?

The embedding model and generation model are not being fine-tuned.

Instead:
1. The document text is changed.
2. The vector database updates the document's stored representation.
3. A later query can retrieve the updated document.

This is different from retraining the LLM or embedding model itself.

### 4. Avoid assuming a fixed distance

The original lab states that the CEO query should produce a distance of approximately `0.12`. This is not guaranteed.

The distance depends on:
* The embedding model.
* The selected distance metric.
* The query.
* The documents in the collection.
* The ChromaDB configuration.

The correct teaching objective is to observe whether the updated document is retrieved, not to expect a fixed numerical score.

### 5. Repeated execution

The baseline reset is intended to make the demonstration repeatable.

However, the code assumes that:
* The target ID exists.
* The collection has been initialized.
* The document can be updated.
* The collection has not been replaced or populated with conflicting records.

Therefore, the demonstration is designed to be repeatable under these conditions, rather than being guaranteed to work in every possible runtime state.

---

# Appendix: Handling Unstructured Documents (PDF & DOCX)

In real-world applications, knowledge is often stored in unstructured documents rather than a clean JSON dataset.

Examples include:
* PDF manuals.
* DOCX policies.
* Internal reports.
* Technical documentation.
* Employee guidelines.
* Medical protocols.

Before these documents can be used for retrieval, we need to extract their text, divide the text into useful chunks, and store the chunks in the vector database.

## Workflow

1. **Source documents:** PDF and DOCX files
2. **Text extraction:** Read the document content
3. **Chunking:** Split text into smaller sections
4. **Vector indexing:** Generate embeddings and store the chunks
5. **Semantic retrieval:** Search the indexed document chunks

## Appendix Cell: Install ReportLab

```python
!pip install -q reportlab
```

ReportLab is used to generate the example PDF document.

## Step 1: Create Sample DOCX and PDF Files

```python
import os
import docx
from pypdf import PdfReader
from reportlab.pdfgen import canvas

# ==========================================
# 1a. Create sample Word Document
# ==========================================

doc_path = "hospital_policy.docx"

doc = docx.Document()

doc.add_heading(
    "MediCore Hospital Acute Care Protocols",
    level=1
)

doc.add_paragraph(
    "All patients arriving with acute chest pain must receive "
    "an immediate 12-lead ECG within 10 minutes of registration. "
    "The attending cardiologist must be paged immediately."
)

doc.add_paragraph(
    "Emergency stroke patients require an immediate "
    "non-contrast head CT scan. Thrombolytic therapy must be "
    "evaluated within 45 minutes of door arrival."
)

doc.save(doc_path)

# ==========================================
# 1b. Create sample PDF Document
# ==========================================

pdf_path = "surgical_protocols.pdf"

c = canvas.Canvas(pdf_path)

c.drawString(
    72,
    750,
    "MediCore Hospital Surgical Division Policy:"
)

c.drawString(
    72,
    730,
    "Robotic-assisted surgery suites require full "
    "UV-C terminal sterilization."
)

c.drawString(
    72,
    710,
    "Surgeons must complete 3D virtual simulation "
    "before operating with the MediBot."
)

c.save()

print(
    "Generated sample files: "
    "hospital_policy.docx, surgical_protocols.pdf"
)
```

### Code Explanation

This code generates two example documents.

* **DOCX:** Contains hospital acute care protocols.
* **PDF:** Contains surgical division policies.

These files are created specifically for the demonstration, so students can test the extraction pipeline without needing external documents.

### Important limitation

The generated PDF uses a simple text layout. Real PDFs may contain:
* Scanned images.
* Tables.
* Multi-column layouts.
* Headers and footers.
* Complex formatting.

`pypdf` can extract text from many text-based PDFs, but scanned documents may require OCR. Extraction quality should be inspected before the text is indexed.

## Step 2: Define Text Extraction Functions

```python
# ==========================================
# Step 2: Extraction Functions
# ==========================================

def extract_from_docx(file_path):
    d = docx.Document(file_path)

    paragraphs = [
        p.text.strip()
        for p in d.paragraphs
        if p.text.strip()
    ]

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
```

## Code Explanation

### DOCX extraction

```python
docx.Document(file_path)
```

This opens the Word document.

The code iterates through its paragraphs and extracts non-empty text.

The paragraphs are then joined into a single string.

### PDF extraction

```python
reader = PdfReader(file_path)
```

This opens the PDF using `pypdf`.

The code loops through each page and calls:

```python
page.extract_text()
```

If text is found, it is added to the list of extracted pages.

### Limitation

The function does not extract every possible PDF element. For example, it may not correctly reconstruct tables or text embedded inside images.

## Step 3: Fixed-Size Text Chunking with Overlap

Large documents should usually be divided into smaller chunks before embedding.

**Why?**

Embedding an entire long document can make retrieval less precise. A query about a specific procedure may retrieve a large document containing many unrelated topics.

Chunking creates smaller searchable units.

### Code

```python
# ==========================================
# Step 3: Text Chunking
# ==========================================

def chunk_text(
    text,
    source_name,
    chunk_size=150,
    overlap=30
):
    chunks = []
    metadatas = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

            metadatas.append({
                "source": source_name,
                "char_start": start
            })

        start += (chunk_size - overlap)

    return chunks, metadatas


docx_chunks, docx_meta = chunk_text(
    docx_text,
    source_name="hospital_policy.docx"
)

pdf_chunks, pdf_meta = chunk_text(
    pdf_text,
    source_name="surgical_protocols.pdf"
)

all_chunks = docx_chunks + pdf_chunks
all_metadatas = docx_meta + pdf_meta

all_ids = [
    f"unstructured_chunk_{i}"
    for i in range(len(all_chunks))
]
```

## Code Explanation

### Chunk size

```python
chunk_size=150
```

The code uses a chunk size of 150 characters.

This is a small value chosen for demonstration.

It is not a universal optimal chunk size. In a real application, the appropriate chunk size depends on:
* The type of documents.
* The embedding model.
* The language.
* The retrieval task.
* The expected context size.
* The document structure.

### Overlap

```python
overlap=30
```

Overlap means that adjacent chunks share some text.

For example, if a document is split into chunks of 150 characters with an overlap of 30 characters, the next chunk begins 120 characters after the previous chunk's start.

Overlap helps reduce the risk of losing context at chunk boundaries.

### Important limitation of character-based chunking

This approach may split a sentence or paragraph in the middle.

For example:

```text
Chunk 1:
Patients arriving with acute chest pain must receive an immediate

Chunk 2:
12-lead ECG within 10 minutes of registration.
```

This can be problematic because the second chunk may lose context.

*Recommended improvement for a more advanced lab:* Introduce sentence-aware or paragraph-aware chunking and preserve document structure. Character-based chunking is acceptable as an introductory demonstration, but its limitations should be explained.

## Step 4: Ingest into a Dedicated ChromaDB Collection

```python
# ==========================================
# Step 4: Create a Dedicated Collection
# ==========================================

chroma_client = chromadb.Client()

st_embed = (
    embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)

unstructured_collection = (
    chroma_client.create_collection(
        name="unstructured_demo",
        embedding_function=st_embed
    )
)

unstructured_collection.add(
    documents=all_chunks,
    metadatas=all_metadatas,
    ids=all_ids
)

print(
    f"Indexed {unstructured_collection.count()} "
    "chunks from DOCX and PDF into ChromaDB.\n"
)
```

## Code Explanation

This code creates a separate collection for the unstructured document demonstration.

The original MediCore collection contains the factual knowledge base, while this collection contains chunks extracted from the DOCX and PDF files.

Separating collections helps students distinguish between different experiments.

### Why do we store metadata?

The metadata includes:

```python
{
    "source": source_name,
    "char_start": start
}
```

This allows the system to identify which source file produced the chunk.

In a production pipeline, you may also want to store:
* Page number.
* Document title.
* Section heading.
* Document version.
* Source URL.
* Access permissions.

This makes it easier to trace retrieved content back to the original source.

## Step 5: Query Across Document Types

```python
# ==========================================
# Step 5: Query Across Document Types
# ==========================================

query = "What is the procedure for emergency chest pain?"

results = unstructured_collection.query(
    query_texts=[query],
    n_results=1
)

retrieved_doc = results["documents"][0][0]
source_file = results["metadatas"][0][0]["source"]
distance = results["distances"][0][0]

print(f"QUERY: {query}")

print(
    f"MATCH FROM SOURCE: [{source_file}] "
    f"(Distance: {distance:.4f})"
)

print(f"CONTENT: {retrieved_doc}")
```

## Code Explanation

### Query

```python
query = "What is the procedure for emergency chest pain?"
```

The query asks for information about acute chest pain procedures.

The retrieval system searches the document chunks and returns the closest match.

### Source attribution

```python
source_file = results["metadatas"][0][0]["source"]
```

This retrieves the source file name from the metadata.

Source attribution is important because it helps users understand where the retrieved information came from.

### Limitation

The code retrieves only one chunk:

```python
n_results=1
```

The single chunk may not contain the complete procedure.

In a real RAG application, you might retrieve multiple chunks and combine them with additional processing, while keeping the context within the model's available input length.

---

# Metadata Filtering

Metadata filtering allows us to restrict retrieval to documents that satisfy a structured condition.

For example, imagine the database contains documents from different hospital departments:

```python
{
    "department": "neurology",
    "source_prompt": "Who leads neurology?"
}
```

We could search only within documents whose metadata contains:

```text
department = neurology
```

### Example

```python
filtered_results = collection.query(
    query_texts=[
        "Who leads the neurology department?"
    ],
    n_results=3,
    where={
        "department": "neurology"
    }
)
```

**Important:** This example only works if the `department` metadata field was added during ingestion. The original ingestion code stores `source_prompt` and `doc_index`, but does not create a `department` field.

### Recommended metadata improvement

If the dataset includes a department or category field, you could add it during ingestion:

```python
metadatas.append({
    "source_prompt": item["prompt"],
    "doc_index": idx,
    "department": item["department"]
})
```

This requires that the `department` key exists in the dataset. Do not add this line without checking the actual schema.

---

# Summary and Next Steps

In this lab, you:

1. Learned what RAG is and why retrieval is needed.
2. Understood the difference between keyword search and semantic search.
3. Converted sentences into 384-dimensional embedding vectors.
4. Calculated cosine similarity between sentence embeddings.
5. Initialized ChromaDB and configured a sentence embedding function.
6. Loaded the MediCore knowledge base into a vector database.
7. Performed natural-language queries using top-k retrieval.
8. Explored how metadata can support filtering and source attribution.
9. Updated a document without retraining the language model.
10. Extracted text from PDF and DOCX files.
11. Divided extracted text into chunks and indexed those chunks.
12. Discussed the limitations of distance metrics, chunking, and text extraction.

## Connection to the next lab

In the next activity, you will connect the retrieval system to the Qwen2.5-1.5B-Instruct model.

The next lab will implement the Augmented Generation (A + G) stage of RAG.

The general workflow will be:

```text
User question
      |
      v
Retrieve relevant documents from ChromaDB
      |
      v
Construct a prompt containing the retrieved context
      |
      v
Pass the prompt to Qwen
      |
      v
Generate an answer grounded in the retrieved information
```

The retrieval system provides the information, while the generation model uses that information to formulate the response.

**Key takeaway:** A strong RAG system depends on both retrieval quality and generation quality. An LLM cannot reliably answer from context that the retrieval system failed to find or extracted incorrectly.


<!-- 
----
# Suggested Additional Experiment: Evaluate Retrieval Quality

The lab currently demonstrates that retrieval works, but it does not formally evaluate retrieval quality.

## Create test queries

```python
test_queries = [
    "Who leads the neurology department?",
    "Where do helicopters land?",
    "What is the hospital's emergency chest pain procedure?"
]
```

For each query, you should:
1. Run the query against the database.
2. Inspect the top-k retrieved documents.
3. Decide whether the documents are relevant.
4. Record the distance values.
5. Identify cases where the correct information is missing from the top-k results.

**Why is this useful?**

A system that returns documents is not necessarily a good retrieval system.

Retrieval quality should be evaluated based on whether relevant documents appear in the results.

For a more advanced exercise:

| Metric | Meaning |
| :--- | :--- |
| **Recall@k** | Whether a relevant document appears in the top-k results. |
| **Precision@k** | The proportion of the top-k results that are relevant. |
| **MRR** | How high the first relevant result appears in the ranking. |

These metrics require a set of queries with known relevant documents.

-->
