# Lab: Building a Retrieval-Augmented Generation (RAG) System with Qwen 2.5, Sentence Embeddings, and ChromaDB

## Learning Objectives

By the end of this lab, the learner will be able to:

1. Explain the purpose of Retrieval-Augmented Generation (RAG).
2. Explain why a language model can benefit from external documents.
3. Extract text from external documents such as PDFs.
4. Split documents into useful chunks.
5. Generate sentence embeddings.
6. Store embeddings and metadata in ChromaDB.
7. Retrieve relevant document chunks for a question.
8. Construct a RAG prompt for Qwen 2.5.
9. Generate an answer grounded in retrieved documents.
10. Evaluate retrieval quality and answer grounding.
11. Compare different RAG configurations.
12. Build a small RAG application using a custom document collection.

---

# 1. Introduction to Retrieval-Augmented Generation

A language model generates responses from information encoded in its parameters. However, many applications require the model to answer questions using information that is:

* specific to an organization,
* contained in private documents,
* recently updated,
* too large to include in the original training data,
* or supplied by the user at runtime.

**Retrieval-Augmented Generation (RAG)** addresses this problem by retrieving relevant information from an external knowledge source and providing that information to the language model as context.

A simplified RAG pipeline is:

```text
External Documents
       ↓
Text Extraction
       ↓
Document Chunking
       ↓
Sentence Embeddings
       ↓
ChromaDB
       ↓
User Question
       ↓
Question Embedding
       ↓
Similarity Search
       ↓
Relevant Document Chunks
       ↓
Prompt + Retrieved Context
       ↓
Qwen 2.5
       ↓
Generated Answer
```

RAG therefore separates two responsibilities:

* **Retrieval:** Find relevant information.
* **Generation:** Use that information to produce a natural-language answer.

Qwen 2.5 does not need a special "RAG mode." RAG is an application architecture around the language model.

### Q&A

**Question:** What problem does RAG solve?

**Hint for asking Gemini:**

> "Explain what Retrieval-Augmented Generation solves and why a language model might need external documents."

**Answer: Detailed explanation**

RAG allows a language model to use information from an external knowledge base at inference time. Instead of relying only on information stored in the model's parameters, the application retrieves relevant documents and places them into the prompt.

For example, a company could store its employee handbook in a vector database. When an employee asks:

```text
How many vacation days are available after five years of employment?
```

the RAG system can retrieve the relevant section of the handbook and provide it to Qwen as context.

The model then generates the answer using that retrieved information.

---

# 2. Understanding the RAG Architecture

A complete RAG system normally has two stages.

## Stage A: Building the knowledge base

```text
Documents
   ↓
Extract text
   ↓
Clean text
   ↓
Split into chunks
   ↓
Generate embeddings
   ↓
Store in ChromaDB
```

This stage is usually performed before users ask questions.

## Stage B: Answering a question

```text
User question
      ↓
Generate question embedding
      ↓
Search ChromaDB
      ↓
Retrieve relevant chunks
      ↓
Build prompt
      ↓
Qwen 2.5
      ↓
Answer
```

This second stage occurs whenever a user submits a question.

### Q&A

**Question:** Why are documents converted into embeddings before being stored?

**Hint for asking Gemini:**

> "Explain why text documents are converted into vector embeddings in a RAG system."

**Answer: Detailed explanation**

Text cannot be directly compared using ordinary numerical similarity operations. An embedding model converts text into a numerical vector representing its semantic meaning.

For example:

```text
"How can a password be reset?"
```

and

```text
"What is the procedure for changing a forgotten password?"
```

contain different words but have similar meanings.

A good sentence-embedding model produces vectors that are relatively close to each other.

This allows the retrieval system to search by **meaning**, rather than relying only on exact keyword matches.

---

# 3. Preparing the Google Colab Environment

For this lab, a GPU runtime is recommended.

In Google Colab:

**Runtime → Change runtime type → GPU**

Install the required libraries.

```python
!pip install -q transformers accelerate sentence-transformers chromadb pypdf
```

Import the libraries:

```python
import os
import json
import pandas as pd
import chromadb
import torch

from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
```

Check whether a GPU is available:

```python
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

### Q&A

**Question:** Why is a GPU useful for this lab?

**Hint for asking Gemini:**

> "Explain which parts of a RAG pipeline benefit from GPU acceleration."

**Answer: Detailed explanation**

The language model is usually the most computationally expensive part of the pipeline. Generating tokens with Qwen 2.5 is substantially faster on a GPU than on a CPU.

The embedding model can also benefit from GPU acceleration when processing a large number of document chunks.

ChromaDB itself does not require a GPU.

A simplified view is:

| Component           | GPU useful? |
| ------------------- | ----------- |
| PDF text extraction | No          |
| Chunking            | No          |
| Sentence embeddings | Yes         |
| ChromaDB            | No          |
| Qwen generation     | Yes         |

---

# 4. Working with External Documents

A RAG system should use documents that are separate from the language model itself.

Possible sources include:

* PDF files
* TXT files
* Markdown files
* JSON files
* CSV files
* HTML/web pages
* technical documentation
* manuals
* reports
* research papers
* product documentation

For the main exercise, use a PDF containing useful information.

The document should contain enough information to support several questions.

Examples include:

* a public technical manual,
* a public policy document,
* a public report,
* a textbook chapter,
* an open-access research paper,
* or a fictional document created specifically for the lab.

Avoid using confidential or personally identifiable information.

---

# 5. Uploading a PDF to Google Colab

Use the following code to upload a PDF.

```python
from google.colab import files

uploaded = files.upload()

pdf_files = list(uploaded.keys())

print("Uploaded files:")
for filename in pdf_files:
    print("-", filename)
```

Select a PDF from the local computer.

For example:

```text
example_document.pdf
```

The file is now available in the Colab environment.

---

# 6. Extracting Text from a PDF

The `pypdf` library can extract text from text-based PDF documents.

```python
from pypdf import PdfReader

pdf_path = pdf_files[0]

reader = PdfReader(pdf_path)

print("Number of pages:", len(reader.pages))
```

Extract the text:

```python
pages = []

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""

    pages.append({
        "page": page_number,
        "text": text
    })

print(pages[0]["text"][:1000])
```

Each page is stored together with its page number.

This is useful because metadata can later tell the user where retrieved information came from.

### Important limitation

`pypdf` works well with PDFs containing selectable text.

A scanned PDF may contain images rather than actual text. In that situation:

```text
PDF
 ↓
Image
 ↓
OCR
 ↓
Text
```

is required before the RAG pipeline can use the content effectively.

### Q&A

**Question:** Why should page numbers be preserved during PDF extraction?

**Hint for asking Gemini:**

> "Explain why metadata such as page numbers and filenames are important in a RAG system."

**Answer: Detailed explanation**

Page numbers provide provenance information.

Suppose ChromaDB retrieves a paragraph from page 17. The application can associate the retrieved chunk with:

```text
Source: example_document.pdf
Page: 17
```

This makes the system easier to evaluate and allows users to inspect the original source.

Metadata can also contain:

```text
filename
page
section
document type
URL
chunk ID
```

This information does not normally form the semantic vector itself. Instead, it describes the stored document chunk.

---

# 7. Document Chunking

A complete document is often too large to place into a single prompt.

Instead, the document is divided into smaller pieces called **chunks**.

For example:

```text
Document
│
├── Chunk 1
├── Chunk 2
├── Chunk 3
├── Chunk 4
└── Chunk 5
```

A simple character-based chunking function can be used for the first implementation.

```python
def chunk_text(text, chunk_size=700, overlap=100):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks
```

Test the function:

```python
sample_text = pages[0]["text"]

chunks = chunk_text(sample_text)

print("Number of chunks:", len(chunks))
print("\nFirst chunk:\n")
print(chunks[0])
```

For a complete PDF, chunk each page separately:

```python
all_chunks = []

for page_data in pages:

    page_chunks = chunk_text(
        page_data["text"],
        chunk_size=700,
        overlap=100
    )

    for chunk_number, chunk in enumerate(page_chunks):

        all_chunks.append({
            "text": chunk,
            "page": page_data["page"],
            "chunk": chunk_number,
            "source": pdf_path
        })

print("Total chunks:", len(all_chunks))
```

Inspect a few chunks:

```python
for item in all_chunks[:3]:
    print("Source:", item["source"])
    print("Page:", item["page"])
    print("Chunk:", item["chunk"])
    print(item["text"][:500])
    print("-" * 80)
```

### Q&A

**Question:** What happens if chunks are too large or too small?

**Hint for asking Gemini:**

> "Explain the trade-off between small and large chunks in a RAG retrieval system."

**Answer: Detailed explanation**

If chunks are too large, a retrieved chunk may contain a large amount of irrelevant information. This increases prompt length and can make it harder for the model to identify the important information.

If chunks are too small, important information may be divided between multiple chunks.

For example:

```text
Chunk 1:
The system provides three levels of access...

Chunk 2:
Administrators have access to all configuration settings...
```

A question about administrator permissions may require both chunks.

Chunk overlap helps reduce this problem.

With:

```text
chunk_size = 700
overlap = 100
```

the final 100 characters of one chunk can appear at the beginning of the next chunk.

The best chunk size depends on the document structure and retrieval task.

---

# 8. Generating Sentence Embeddings

Installations already include Sentence Transformers, so the next step is to load an embedding model.

```python
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)
```

Test the model:

```python
sentence = "Retrieval-Augmented Generation uses external documents."

embedding = embedding_model.encode(sentence)

print("Embedding type:", type(embedding))
print("Embedding dimensions:", len(embedding))
```

The embedding is a numerical vector.

For example:

```text
[0.012, -0.083, 0.144, ...]
```

The `all-MiniLM-L6-v2` model produces 384-dimensional embeddings.

Generate embeddings for all chunks:

```python
texts = [item["text"] for item in all_chunks]

embeddings = embedding_model.encode(
    texts,
    show_progress_bar=True
)

print("Number of embeddings:", len(embeddings))
print("Embedding shape:", embeddings.shape)
```

### Q&A

**Question:** What is the difference between an embedding model and a language model?

**Hint for asking Gemini:**

> "Compare the roles of a sentence embedding model and a generative language model in RAG."

**Answer: Detailed explanation**

An embedding model converts text into numerical vectors that are useful for measuring semantic similarity.

A generative language model such as Qwen takes text and generates new text.

In this lab:

```text
Sentence Transformer
        ↓
Text → Vector
```

while:

```text
Qwen 2.5
        ↓
Prompt → Generated answer
```

They therefore perform different tasks.

---

# 9. Understanding Semantic Similarity

Suppose the knowledge base contains:

```text
Users can reset their password from the account settings page.
```

The user asks:

```text
Where can I change a forgotten password?
```

There may not be an exact keyword match for every word.

However, the meanings are related.

Embeddings allow the system to represent both pieces of text as vectors:

```text
Document vector
      ↕
Semantic similarity
      ↕
Question vector
```

A common similarity measure is cosine similarity.

Conceptually:

```text
similarity(question, document)
```

produces a score indicating how semantically related the two pieces of text are.

### Q&A

**Question:** Why is semantic similarity often more useful than exact keyword matching?

**Hint for asking Gemini:**

> "Explain the difference between keyword search and semantic vector search using a simple example."

**Answer: Detailed explanation**

Keyword search looks for matching words.

Semantic search attempts to identify related meanings.

For example:

```text
Question:
How do I reset my password?
```

and:

```text
Document:
Users who have forgotten their credentials can change them
from the account settings page.
```

share little exact vocabulary but have a strong semantic relationship.

An embedding model can represent these meanings in a vector space where related sentences tend to be closer.

---

# 10. Installing and Initializing ChromaDB

ChromaDB will be used as the vector database.

```python
import chromadb

client = chromadb.PersistentClient(
    path="./chroma_db"
)
```

Create a collection:

```python
collection = client.get_or_create_collection(
    name="rag_documents"
)
```

A ChromaDB collection can contain:

* document text,
* embeddings,
* IDs,
* metadata.

---

# 11. Storing Document Chunks in ChromaDB

Create unique IDs for the chunks:

```python
ids = [
    f"{item['source']}_page_{item['page']}_chunk_{item['chunk']}"
    for item in all_chunks
]
```

Prepare metadata:

```python
metadatas = [
    {
        "source": item["source"],
        "page": item["page"],
        "chunk": item["chunk"]
    }
    for item in all_chunks
]
```

Store the chunks:

```python
collection.upsert(
    ids=ids,
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)
```

Check the collection:

```python
print("Documents in collection:", collection.count())
```

Inspect stored records:

```python
result = collection.get(
    limit=3,
    include=["documents", "metadatas"]
)

for document, metadata in zip(
    result["documents"],
    result["metadatas"]
):
    print(metadata)
    print(document[:300])
    print("-" * 80)
```

### Q&A

**Question:** What information should be stored alongside an embedding?

**Hint for asking Gemini:**

> "Explain which metadata is useful to store with document chunks in a vector database."

**Answer: Detailed explanation**

At minimum, a useful record contains:

```text
ID
Document text
Embedding
Metadata
```

Metadata can include:

```text
source filename
page number
section
document type
URL
creation date
```

The embedding is used for retrieval, while metadata helps identify and filter the source.

---

# 12. Retrieving Relevant Chunks

The retrieval process begins with the user's question.

Example:

```python
question = "What does the document say about the main system requirements?"
```

Convert the question into an embedding:

```python
question_embedding = embedding_model.encode(
    question
)
```

Search ChromaDB:

```python
results = collection.query(
    query_embeddings=[question_embedding.tolist()],
    n_results=4
)
```

Inspect the results:

```python
for i, document in enumerate(results["documents"][0]):

    print(f"Result {i + 1}")
    print("Source:", results["metadatas"][0][i])
    print(document)
    print("-" * 80)
```

The `n_results` parameter controls how many chunks are retrieved.

For example:

```python
n_results=3
```

retrieves three chunks.

### Q&A

**Question:** What does `n_results` control?

**Hint for asking Gemini:**

> "Explain how the number of retrieved chunks affects a RAG system."

**Answer: Detailed explanation**

`n_results` controls how many candidate chunks are returned from the vector database.

A small value such as:

```python
n_results=2
```

may provide highly focused context but could miss useful information.

A larger value such as:

```python
n_results=10
```

may retrieve more supporting information but can also introduce irrelevant content and increase prompt length.

The appropriate value depends on the document collection and question type.

---

# 13. Loading Qwen 2.5

Load the Qwen 2.5 1.5B Instruct model using Transformers.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

model.eval()
```

Check the model device:

```python
print(model.device)
```

The model can now generate text.

---

# 14. Testing Qwen Without RAG

Before adding retrieval, test the language model itself.

```python
def generate_answer(prompt, max_new_tokens=200):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )
```

Test it:

```python
prompt = """
Explain the purpose of Retrieval-Augmented Generation
in two or three sentences.
"""

print(generate_answer(prompt))
```

This demonstrates the generation component before external documents are introduced.

### Q&A

**Question:** Why test the language model before adding retrieval?

**Hint for asking Gemini:**

> "Explain why components of a RAG pipeline should be tested independently before combining them."

**Answer: Detailed explanation**

A RAG system contains several independent components:

```text
Document extraction
Chunking
Embeddings
Vector database
Retrieval
Prompt construction
Generation
```

If the final system produces a poor answer, testing each component independently makes it easier to identify the source of the problem.

For example:

* Incorrect extracted text → document processing problem.
* Irrelevant retrieved chunks → embedding or retrieval problem.
* Relevant chunks but incorrect answer → generation or prompting problem.

---

# 15. Building the RAG Prompt

The retrieved documents must be included in the prompt sent to Qwen.

First, create a function to format the retrieved context.

```python
def build_context(results):

    context_parts = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    for document, metadata in zip(documents, metadatas):

        source = metadata["source"]
        page = metadata["page"]

        context_parts.append(
            f"[Source: {source}, Page: {page}]\n"
            f"{document}"
        )

    return "\n\n".join(context_parts)
```

Create a RAG prompt:

```python
def build_rag_prompt(question, context):

    return f"""
You are answering a question using the provided document context.

Rules:
1. Use the provided context as the primary source of information.
2. Do not invent facts that are not supported by the context.
3. If the context does not contain enough information, say:
   "I don't know based on the provided documents."
4. Give a concise and direct answer.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
```

---

# 16. Creating the Complete RAG Function

The following function combines:

1. question embedding,
2. ChromaDB retrieval,
3. context construction,
4. prompt construction,
5. Qwen generation.

```python
def rag_answer(
    question,
    n_results=4,
    max_new_tokens=200
):

    # 1. Embed the question
    question_embedding = embedding_model.encode(
        question
    )

    # 2. Retrieve relevant chunks
    results = collection.query(
        query_embeddings=[
            question_embedding.tolist()
        ],
        n_results=n_results
    )

    # 3. Build context
    context = build_context(results)

    # 4. Build prompt
    prompt = build_rag_prompt(
        question,
        context
    )

    # 5. Generate answer
    answer = generate_answer(
        prompt,
        max_new_tokens=max_new_tokens
    )

    return {
        "question": question,
        "answer": answer,
        "context": context,
        "results": results
    }
```

Test it:

```python
question = "What are the main requirements described in the document?"

result = rag_answer(question)

print("QUESTION:")
print(result["question"])

print("\nANSWER:")
print(result["answer"])
```

---

# 17. Displaying Retrieved Sources

A RAG application should make the retrieved sources visible during evaluation.

```python
for i, metadata in enumerate(
    result["results"]["metadatas"][0]
):

    print(
        f"Retrieved source {i + 1}: "
        f"{metadata['source']}, "
        f"page {metadata['page']}, "
        f"chunk {metadata['chunk']}"
    )
```

This makes it possible to inspect whether the answer is based on appropriate evidence.

### Q&A

**Question:** Why should retrieved sources be inspected instead of evaluating only the final answer?

**Hint for asking Gemini:**

> "Explain why retrieval quality and generation quality should be evaluated separately in RAG."

**Answer: Detailed explanation**

A fluent answer can still be incorrect.

For example:

```text
Retrieved context:
Information about software installation.

Question:
What are the hardware requirements?
```

If Qwen generates a plausible answer despite receiving irrelevant context, the generation may appear successful while retrieval has failed.

RAG evaluation should therefore examine at least two things separately:

```text
Retrieval quality
        +
Answer quality
```

---

# 18. Testing Questions That Are Supported by the Documents

Create several questions whose answers should be present in the uploaded document.

```python
questions = [
    "What is the main purpose of the system?",
    "What are the important requirements?",
    "What process does the document describe?"
]

for question in questions:

    result = rag_answer(question)

    print("=" * 80)
    print("QUESTION:", question)
    print("ANSWER:", result["answer"])
```

The retrieved chunks should be inspected for each question.

---

# 19. Testing Questions That Are Not Supported

A good RAG system should not confidently invent information when the knowledge base does not contain an answer.

For example:

```python
question = "What is the population of Mars according to this document?"

result = rag_answer(question)

print(result["answer"])
```

The expected behavior is something similar to:

```text
I don't know based on the provided documents.
```

The exact wording may vary.

### Q&A

**Question:** Why is an "I don't know" behavior important in RAG?

**Hint for asking Gemini:**

> "Explain hallucination in RAG systems and why an explicit unknown-answer policy is useful."

**Answer: Detailed explanation**

Retrieval does not guarantee that the retrieved information answers the question.

If the system always generates an answer, the language model may fill gaps using information that was not retrieved from the knowledge base.

An explicit instruction such as:

```text
If the context does not contain enough information, say that
the answer is not available in the provided documents.
```

reduces this behavior.

It does not completely eliminate hallucination, but it establishes a clear grounding rule for the application.

---

# 20. Improving Document Ingestion

The basic implementation can be extended to support multiple document types.

## TXT and Markdown

```python
def load_text_file(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()
```

## JSON

```python
def load_json_file(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    return json.dumps(
        data,
        indent=2,
        ensure_ascii=False
    )
```

## CSV

```python
def load_csv_file(path):

    df = pd.read_csv(path)

    return df.to_string(
        index=False
    )
```

This demonstrates an important property of RAG:

> The language model does not need to directly understand every original file format. The ingestion pipeline converts the information into text that can be chunked, embedded, retrieved, and supplied to the model.

---

# 21. Comparing Different Data Sources

A useful RAG knowledge base should contain more than one source.

For example:

```text
knowledge_base/
│
├── manual.pdf
├── policy.pdf
├── faq.txt
├── product_notes.md
└── specifications.csv
```

Each chunk should retain its source metadata.

For example:

```python
{
    "source": "manual.pdf",
    "page": 4,
    "chunk": 2
}
```

or:

```python
{
    "source": "faq.txt",
    "page": 1,
    "chunk": 5
}
```

This allows the retrieval system to search across multiple documents while retaining provenance.

### Q&A

**Question:** Why is it useful for a RAG system to retrieve information from multiple documents?

**Hint for asking Gemini:**

> "Explain how multi-document RAG differs from RAG using a single document."

**Answer: Detailed explanation**

A multi-document knowledge base can answer questions that require information distributed across several sources.

For example:

```text
manual.pdf
    ↓
Technical information

policy.pdf
    ↓
Usage rules

faq.txt
    ↓
Common questions
```

The same retrieval system can search all three sources.

Metadata allows the application to identify which source contributed each retrieved chunk.

---

# 22. Experiment: Changing Chunk Size

The initial configuration uses:

```python
chunk_size=700
overlap=100
```

Create alternative configurations:

```python
chunk_sizes = [
    (300, 50),
    (700, 100),
    (1200, 150)
]
```

For each configuration:

1. Recreate the chunks.
2. Generate embeddings.
3. Store them in a separate ChromaDB collection.
4. Ask the same set of questions.
5. Compare the retrieved results.

Record observations such as:

| Chunk configuration | Retrieval quality | Context completeness | Prompt size |
| ------------------- | ----------------- | -------------------- | ----------- |
| 300 / 50            |                   |                      |             |
| 700 / 100           |                   |                      |             |
| 1200 / 150          |                   |                      |             |

### Q&A

**Question:** Is there one universally correct chunk size?

**Hint for asking Gemini:**

> "Explain why there is no universal optimal chunk size for every RAG application."

**Answer: Detailed explanation**

Different documents have different structures.

A technical manual may benefit from relatively large chunks because explanations span several paragraphs.

A FAQ document may work better with smaller chunks because each question-and-answer pair is already a meaningful unit.

Therefore chunk size should be treated as an experimental parameter rather than a fixed universal value.

---

# 23. Experiment: Changing the Number of Retrieved Chunks

Test different values:

```python
retrieval_values = [1, 2, 4, 6, 8]
```

For each value, run the same questions:

```python
for k in retrieval_values:

    result = rag_answer(
        "What is the main purpose of the system?",
        n_results=k
    )

    print("=" * 80)
    print("Retrieved chunks:", k)
    print(result["answer"])
```

Compare:

* answer correctness,
* irrelevant context,
* missing information,
* response length,
* generation speed.

### Q&A

**Question:** Why can retrieving more documents sometimes make an answer worse?

**Hint for asking Gemini:**

> "Explain why increasing the number of retrieved chunks does not always improve RAG answers."

**Answer: Detailed explanation**

More context is not automatically better.

Suppose the first two retrieved chunks directly answer the question, but the next six contain unrelated information.

The language model now has to process additional material and determine which information matters.

This can increase:

* prompt length,
* irrelevant information,
* ambiguity,
* processing time.

The retrieval stage should therefore aim for **relevant context**, not simply maximum context.

---

# 24. Experiment: Base Qwen vs. Qwen + RAG

The same question can be given to Qwen in two different ways.

## Without retrieval

```text
Question
   ↓
Qwen
   ↓
Answer
```

## With retrieval

```text
Question
   ↓
ChromaDB
   ↓
Relevant documents
   ↓
Qwen
   ↓
Answer
```

Run the same question using both configurations.

```python
question = "What does the document say about the system requirements?"

base_prompt = f"""
Answer the following question:

{question}
"""

base_answer = generate_answer(base_prompt)

rag_result = rag_answer(question)

print("BASE MODEL:")
print(base_answer)

print("\nRAG:")
print(rag_result["answer"])
```

Record the results.

| Configuration | Correct? | Grounded? | Useful? |
| ------------- | -------: | --------: | ------: |
| Qwen          |          |           |         |
| Qwen + RAG    |          |           |         |

### Q&A

**Question:** Why might RAG improve an answer without changing the language model's parameters?

**Hint for asking Gemini:**

> "Explain how RAG can change a model's available information without changing the model weights."

**Answer: Detailed explanation**

RAG changes the information available to the model at inference time.

The language model remains the same, but the prompt now contains relevant external information.

Conceptually:

```text
Qwen parameters
       +
retrieved context
       ↓
answer
```

The model does not need to have memorized every document in advance.

---

# 25. Optional Extension: Combining Fine-Tuning and RAG

A RAG system and model fine-tuning solve different problems.

RAG is primarily useful for providing **external knowledge**.

Fine-tuning is primarily useful for changing **behavior, style, formatting, or task specialization**.

They can therefore be combined:

```text
External documents
       ↓
     RAG
       ↓
retrieved context
       ↓
fine-tuned Qwen
       ↓
answer
```

For example, a model could be trained to consistently produce a particular answer format while RAG supplies the current factual information.

This combination is optional for the lab.

### Q&A

**Question:** What is the conceptual difference between fine-tuning and RAG?

**Hint for asking Gemini:**

> "Compare fine-tuning and RAG and explain what problem each one is designed to solve."

**Answer: Detailed explanation**

Fine-tuning changes model parameters through additional training.

RAG does not require changing the model parameters. Instead, it retrieves external information and places it into the model's input context.

A simplified comparison is:

| Technique         | Main purpose                                      |
| ----------------- | ------------------------------------------------- |
| Fine-tuning       | Adapt model behavior                              |
| RAG               | Provide external knowledge                        |
| Fine-tuning + RAG | Adapt behavior while supplying external knowledge |

---

# 26. Evaluating the RAG System

A RAG system should not be evaluated only by asking whether the final answer sounds good.

At least four aspects should be considered.

## 1. Retrieval relevance

Did the system retrieve useful chunks?

## 2. Answer correctness

Is the answer factually correct according to the source?

## 3. Grounding

Is the answer supported by the retrieved context?

## 4. Unknown behavior

Does the system avoid confidently answering questions that are unsupported?

Create an evaluation table:

| Question | Retrieved relevant chunks? | Answer correct? | Grounded? | Unknown handled? |
| -------- | -------------------------: | --------------: | --------: | ---------------: |
| Q1       |                            |                 |           |                  |
| Q2       |                            |                 |           |                  |
| Q3       |                            |                 |           |                  |
| Q4       |                            |                 |           |                  |
| Q5       |                            |                 |           |                  |

### Q&A

**Question:** Why is grounding different from answer correctness?

**Hint for asking Gemini:**

> "Explain the difference between a correct answer and a grounded answer in RAG."

**Answer: Detailed explanation**

An answer can be correct for the wrong reason.

For example, suppose the retrieved document says:

```text
The product was released in 2024.
```

The model answers:

```text
The product was released in 2024.
```

This is grounded.

However, if the retrieved context contains no information about the release date and the model produces the same answer from its general knowledge, the answer may be factually correct but not grounded in the retrieved evidence.

RAG evaluation therefore considers both correctness and evidence.

---

# 27. Final RAG Pipeline

At this stage, the complete system can be represented as:

```text
                  ┌───────────────────┐
                  │ External Documents│
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Text Extraction   │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Chunking          │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Sentence Encoder  │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ ChromaDB          │
                  └─────────┬─────────┘
                            │
                    User Question
                            │
                            ▼
                  ┌───────────────────┐
                  │ Question Embedding│
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Similarity Search │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Retrieved Context │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ RAG Prompt        │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Qwen 2.5-1.5B     │
                  └─────────┬─────────┘
                            │
                            ▼
                     Final Answer
```

### Q&A

**Question:** Which components belong to retrieval and which belong to generation?

**Hint for asking Gemini:**

> "Divide a RAG pipeline into retrieval and generation components and explain the role of each."

**Answer: Detailed explanation**

The retrieval side includes:

```text
Document extraction
Chunking
Embedding
Vector database
Question embedding
Similarity search
```

The generation side includes:

```text
Prompt construction
Qwen inference
Answer generation
```

The retrieved context connects these two parts.

---

# 28. Mini Project

Build a RAG application using a custom collection of documents.

## Requirements

The project should contain at least:

* 3 different documents,
* 20 or more document chunks,
* sentence embeddings,
* ChromaDB,
* Qwen 2.5-1.5B-Instruct,
* a question-answering interface,
* source metadata,
* retrieval results,
* an unknown-question test.

Possible document collection:

```text
Document 1: Technical manual
Document 2: Product documentation
Document 3: FAQ
```

The documents should contain information that can be queried.

## Required functionality

The application should:

1. Load the documents.
2. Extract their text.
3. Split the text into chunks.
4. Generate embeddings.
5. Store the chunks in ChromaDB.
6. Accept a user question.
7. Retrieve the most relevant chunks.
8. Build a prompt containing the retrieved context.
9. Generate an answer with Qwen.
10. Display the answer.
11. Display the sources used.
12. Demonstrate behavior when the answer is not present.

---

# 29. Mini Project Evaluation

Use the following rubric.

| Component               | Suggested marks |
| ----------------------- | --------------: |
| Document ingestion      |              10 |
| Chunking strategy       |              10 |
| Embedding generation    |              15 |
| ChromaDB implementation |              15 |
| Retrieval               |              15 |
| Qwen generation         |              15 |
| Source metadata         |              10 |
| Evaluation and analysis |              10 |
| **Total**               |         **100** |

The analysis should discuss at least:

* why the selected chunk size was used,
* why the selected number of retrieved chunks was used,
* whether retrieval was relevant,
* whether answers were grounded,
* examples where the system failed,
* possible improvements.

---

# 30. Suggested Gemini Questions for Troubleshooting

Gemini can be used as a learning assistant while implementing the notebook.

### ChromaDB problem

**Hint:**

> "My ChromaDB collection returns an error when I upsert document embeddings. Here is my code and the full error message. What is causing the error and what should I change?"

### PDF extraction problem

**Hint:**

> "My PDF text extraction returns empty strings for several pages. Explain the likely causes and suggest how I can determine whether the PDF requires OCR."

### Retrieval problem

**Hint:**

> "My RAG system retrieves irrelevant chunks even though the answer exists in the document. Analyze my chunking, embedding, and retrieval configuration."

### Generation problem

**Hint:**

> "The retrieved chunks are relevant, but Qwen produces an incorrect answer. Analyze the prompt and explain how the generation step could be improved."

### Hallucination problem

**Hint:**

> "My RAG model answers questions that are not supported by the retrieved documents. Explain possible causes and suggest changes to the prompt and evaluation process."

---

# 31. Final Questions

### Question 1

What are the main stages of a RAG pipeline?

**Hint for asking Gemini:**

> "Summarize the complete RAG pipeline from external documents to a generated answer."

**Answer: Detailed explanation**

The main stages are:

```text
Documents
→ extraction
→ chunking
→ embeddings
→ vector storage
→ question embedding
→ retrieval
→ context construction
→ prompt
→ language-model generation
→ answer
```

---

### Question 2

What is the role of ChromaDB?

**Hint for asking Gemini:**

> "Explain the role of ChromaDB in a sentence-embedding-based RAG system."

**Answer: Detailed explanation**

ChromaDB stores document chunks, their embeddings, and metadata. It provides a mechanism for finding vectors that are semantically similar to a question embedding.

It therefore acts as the retrieval layer between the document collection and the language model.

---

### Question 3

Why are sentence embeddings needed?

**Hint for asking Gemini:**

> "Explain why a RAG system converts documents and questions into sentence embeddings."

**Answer: Detailed explanation**

Embeddings provide a numerical representation of semantic meaning.

The system embeds both:

```text
document chunks
```

and:

```text
user questions
```

It can then compare their vectors and retrieve chunks that are semantically related to the question.

---

### Question 4

Why does RAG not require changing Qwen's model weights?

**Hint for asking Gemini:**

> "Explain how RAG provides external information to a language model without modifying its parameters."

**Answer: Detailed explanation**

RAG supplies external information through the input prompt.

The model receives:

```text
Question
+
Retrieved context
```

instead of receiving only the question.

The Qwen parameters remain unchanged during normal RAG inference.

---

### Question 5

What is the most important principle when evaluating a RAG system?

**Hint for asking Gemini:**

> "What are the most important criteria for evaluating retrieval-augmented generation systems?"

**Answer: Detailed explanation**

A useful RAG system should retrieve relevant information and generate an answer that is supported by that information.

The evaluation should therefore consider:

```text
Retrieval relevance
        +
Answer correctness
        +
Grounding
        +
Unknown-question behavior
```

A fluent answer alone is not sufficient evidence that the RAG system is working correctly.

---

# Conclusion

This lab implemented a complete RAG pipeline using:

```text
External Documents
       ↓
PDF/Text Extraction
       ↓
Chunking
       ↓
Sentence Embeddings
       ↓
ChromaDB
       ↓
Semantic Retrieval
       ↓
Context-Aware Prompt
       ↓
Qwen 2.5-1.5B-Instruct
       ↓
Grounded Answer
```

The central idea is that the language model and the knowledge base have separate roles. The embedding model and ChromaDB identify relevant information, while Qwen uses the retrieved information to generate a response.

The next step is to experiment with **different documents, chunk sizes, embedding models, retrieval counts, prompts, and evaluation questions** and determine which configuration produces the most reliable grounded answers.

## Reference material

This lab is adapted from the RAG and vector-database concepts covered in Microsoft's **Generative AI for Beginners** course:

<Link url="https://github.com/microsoft/generative-ai-for-beginners/tree/main/05-advanced-prompts/02-retrieval-augmented-generation">Microsoft: Generative AI for Beginners: Retrieval-Augmented Generation</Link>
