# Laboratory Exercise: Sentence Embeddings and ChromaDB

## 1. Objectives

The objective of this laboratory exercise is to introduce sentence embeddings, semantic similarity, and vector databases using ChromaDB.

After completing the exercise, you should be able to:

1. Explain the purpose of text embeddings.
2. Generate sentence embeddings using Sentence Transformers.
3. Calculate similarity between text embeddings.
4. Distinguish between keyword search and semantic search.
5. Explain the purpose of a vector database.
6. Create and use a ChromaDB collection.
7. Store documents and metadata in ChromaDB.
8. Perform semantic similarity searches.
9. Interpret retrieved documents and similarity/distance information.
10. Explain the role of ChromaDB and sentence embeddings in a RAG system.

---

# 2. Background

## 2.1 Text Search

A conventional keyword-based search identifies documents according to matching words or terms.

For example, a query such as:

```text
programming language used for data science
```

may retrieve a document containing the words `programming`, `language`, and `data science`.

Keyword matching does not directly represent the semantic relationship between different expressions. For example:

```text
automobile
car
vehicle
```

are related concepts, although they are different words.

Semantic search addresses this limitation by representing text as numerical vectors and comparing those vectors.

The general process is:

```text
Text
  ↓
Embedding model
  ↓
Vector representation
  ↓
Similarity comparison
  ↓
Relevant documents
```

---

# 3. Sentence Embeddings

An embedding is a numerical representation of an object such as text, an image, or an audio signal.

A sentence embedding represents a sentence or a larger text segment as a vector.

For example:

```text
Python is commonly used for data science.
```

may be represented by a vector of the following form:

```text
[0.021, -0.143, 0.782, 0.091, ...]
```

The individual values do not normally have an independent human-interpretable meaning. The vector as a whole is used to represent the text in a numerical space.

A sentence embedding model is trained so that semantically related texts tend to have similar representations.

For example:

```text
Python is used for data science.
Python is commonly used by data scientists.
```

are expected to have more similar embeddings than:

```text
Python is used for data science.
Paris is the capital of France.
```

---

# 4. Sentence Transformers

In this exercise, sentence embeddings are generated using the Sentence Transformers library.

Install the required packages:

```python
!pip install -q sentence-transformers chromadb scikit-learn
```

Import the required libraries:

```python
import chromadb

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
```

Load the embedding model:

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
```

Encode a sentence:

```python
sentence = "Python is used for data science."

embedding = model.encode(sentence)

print(embedding)
```

Inspect the dimensionality:

```python
print(embedding.shape)
```

The output should be:

```text
(384,)
```

The model therefore represents each input text as a vector with 384 dimensions.

---

# 5. Exercise 1: Comparing Sentence Embeddings

Create the following collection of sentences:

```python
sentences = [
    "Python is a popular programming language.",
    "Python is commonly used for data science.",
    "Machine learning allows computers to learn patterns from data.",
    "Paris is the capital city of France."
]
```

Generate embeddings:

```python
embeddings = model.encode(sentences)

print(embeddings.shape)
```

Calculate cosine similarity:

```python
similarities = cosine_similarity(embeddings)

print(similarities)
```

The result is a matrix in which each row and column corresponds to one sentence.

Display the values together with the corresponding sentences:

```python
for i in range(len(sentences)):
    print(f"\nSentence {i}: {sentences[i]}")
    
    for j in range(len(sentences)):
        print(
            f"  Similarity with sentence {j}: "
            f"{similarities[i][j]:.3f}"
        )
```

### Task

Examine the similarity values and identify:

1. The two sentences with the strongest semantic relationship.
2. The two sentences with the weakest semantic relationship.

Briefly explain whether the numerical results correspond to your expectations.

---

# 6. Exercise 2: Creating Your Own Examples

Create at least five sentences covering two or three different topics.

For example:

```python
my_sentences = [
    "Python is useful for data analysis.",
    "I use Python to process datasets.",
    "The weather is cold in winter.",
    "Machine learning models learn patterns from data.",
    "Artificial intelligence includes machine learning."
]
```

Generate embeddings and calculate the cosine similarity matrix.

```python
my_embeddings = model.encode(my_sentences)

my_similarities = cosine_similarity(my_embeddings)

print(my_similarities)
```

### Task

Identify:

* two pairs of sentences that should have high semantic similarity;
* one pair that should have low semantic similarity.

Compare the expected relationships with the calculated values.

---

# 7. Vector Databases

A vector database is a database designed to store and retrieve vector representations efficiently.

For a small number of documents, embeddings can be stored in ordinary Python data structures. For larger collections, a dedicated vector database provides functionality for storing, indexing, and searching vectors.

A simplified vector-search workflow is:

```text
Documents
    ↓
Embedding model
    ↓
Embeddings
    ↓
Vector database
    ↓
Similarity search
    ↓
Relevant documents
```

A vector database normally stores more than the vector itself. A record can include:

* a unique identifier;
* the original document or text;
* an embedding;
* metadata.

---

# 8. ChromaDB

ChromaDB is a vector database that can be used to store and retrieve embeddings together with their associated documents and metadata.

In this exercise, ChromaDB is used to implement semantic search.

A conceptual representation of a stored record is:

```text
ID:
doc1

Document:
Python is a popular programming language.

Embedding:
[0.12, -0.31, 0.72, ...]

Metadata:
{
    "topic": "programming",
    "source": "python.txt"
}
```

ChromaDB organizes related records into collections.

---

# 9. Creating a ChromaDB Collection

Create a ChromaDB client:

```python
client = chromadb.Client()
```

Create a collection:

```python
collection = client.create_collection(
    name="computer_science"
)
```

The collection will contain the documents used in the semantic search experiments.

---

# 10. Adding Documents

Create a small document collection:

```python
documents = [
    "Python is a popular programming language used for data science.",
    "Machine learning allows computers to learn patterns from data.",
    "Neural networks are computational models inspired by the human brain.",
    "Paris is the capital city of France.",
    "The Eiffel Tower is one of the most famous landmarks in Paris."
]
```

Create unique identifiers:

```python
ids = [
    "doc1",
    "doc2",
    "doc3",
    "doc4",
    "doc5"
]
```

Create metadata:

```python
metadatas = [
    {"topic": "programming"},
    {"topic": "machine learning"},
    {"topic": "deep learning"},
    {"topic": "geography"},
    {"topic": "travel"}
]
```

Add the records to the collection:

```python
collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)
```

The collection now contains the documents and their associated metadata and embeddings.

---

# 11. Semantic Search

Define a query:

```python
query = "What programming language is useful for data science?"
```

Query the collection:

```python
results = collection.query(
    query_texts=[query],
    n_results=3
)
```

Inspect the retrieved documents:

```python
print(results["documents"])
```

Inspect the identifiers:

```python
print(results["ids"])
```

Inspect the metadata:

```python
print(results["metadatas"])
```

Inspect the distances:

```python
print(results["distances"])
```

The retrieved documents are ranked according to their distance from the query embedding.

The exact interpretation of the numerical distance depends on the distance metric configured for the collection. In general, documents that are closer according to the selected metric are considered more similar to the query.

---

# 12. Exercise 3: Semantic Search

Test the collection with the following queries:

```python
queries = [
    "Which language can be used for analysing data?",
    "How do computers learn from data?",
    "What are neural networks?",
    "Where is the Eiffel Tower located?",
    "Which city is the capital of France?"
]
```

For each query, retrieve the three most relevant documents.

### Task

Create a table with the following columns:

| Query | Top Result | Relevant?        |
| ----- | ---------- | ---------------- |
| ...   | ...        | Yes/No/Partially |

Evaluate the relevance of the top result for each query.

---

# 13. Creating a Search Function

The search operation can be placed in a Python function:

```python
def semantic_search(query, n_results=3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    for i, document in enumerate(results["documents"][0]):
        print(f"\nResult {i + 1}")
        print("-" * 50)
        print(document)
        print("Metadata:", results["metadatas"][0][i])
```

Test the function:

```python
semantic_search(
    "How can computers learn from examples?"
)
```

Test several additional queries and inspect the retrieved documents.

---

# 14. Metadata

Metadata is information associated with a document that describes or identifies the document.

For example:

```python
{
    "source": "machine_learning.pdf",
    "page": 15,
    "section": "Introduction"
}
```

Metadata is useful when the source of retrieved information needs to be identified.

In a RAG system, metadata can also be used to:

* identify the source document;
* identify a page or section;
* filter documents;
* display citations or references;
* distinguish between different document collections.

---

# 15. Exercise 4: Metadata

Create a new collection:

```python
collection2 = client.create_collection(
    name="documents_with_sources"
)
```

Create the documents:

```python
documents2 = [
    "Python is widely used for data analysis.",
    "Machine learning algorithms learn patterns from data.",
    "Neural networks are used in many deep learning applications."
]
```

Create metadata:

```python
metadatas2 = [
    {
        "source": "python_guide.pdf",
        "page": 4,
        "topic": "programming"
    },
    {
        "source": "ml_guide.pdf",
        "page": 10,
        "topic": "machine learning"
    },
    {
        "source": "deep_learning.pdf",
        "page": 18,
        "topic": "deep learning"
    }
]
```

Add the documents:

```python
collection2.add(
    documents=documents2,
    ids=["a", "b", "c"],
    metadatas=metadatas2
)
```

Perform a search:

```python
results = collection2.query(
    query_texts=["How do computers learn patterns?"],
    n_results=2
)
```

Display the results:

```python
for i, document in enumerate(results["documents"][0]):
    print("Document:")
    print(document)
    print()
    
    print("Metadata:")
    print(results["metadatas"][0][i])
    
    print("-" * 50)
```

### Task

Explain why the source and page information would be useful in a document retrieval system.

---

# 16. Retrieval and RAG

The semantic search system developed in this laboratory represents the retrieval component of a RAG system.

The current workflow is:

```text
User query
    ↓
Query embedding
    ↓
ChromaDB
    ↓
Relevant documents
```

A complete RAG system adds a language model:

```text
User query
    ↓
Query embedding
    ↓
ChromaDB
    ↓
Relevant document chunks
    ↓
Prompt containing retrieved information
    ↓
Language model
    ↓
Generated answer
```

In the next laboratory, Qwen 2.5 will be used as the language model.

The distinction between the components is:

| Component            | Main function                          |
| -------------------- | -------------------------------------- |
| Sentence Transformer | Generates embeddings                   |
| ChromaDB             | Stores and retrieves vectors/documents |
| Qwen 2.5             | Generates natural-language responses   |

---

# 17. Final Exercise

Create a semantic search system using a collection of at least **10 documents**.

The documents should cover a technical topic such as:

* artificial intelligence;
* machine learning;
* programming;
* databases;
* cybersecurity;
* cloud computing;
* computer networks;
* data science.

Each document must have:

1. a unique identifier;
2. document text;
3. at least two metadata fields.

For example:

```python
{
    "source": "ai_introduction.pdf",
    "topic": "artificial intelligence"
}
```

Your implementation must:

1. Create a ChromaDB client.
2. Create a collection.
3. Add the documents.
4. Add metadata.
5. Perform semantic searches.
6. Retrieve the top three results for each query.
7. Display the retrieved documents and metadata.
8. Test at least five queries.
9. Evaluate the relevance of the retrieved results.

For each query, record whether the top result is:

* relevant;
* partially relevant;
* not relevant.

---

# 18. Questions

Answer the following questions after completing the exercises.

### Question 1

What is an embedding?

### Question 2

What is a sentence embedding?

### Question 3

Why are embeddings useful for semantic search?

### Question 4

What is the difference between keyword search and semantic search?

### Question 5

What is a vector database?

### Question 6

What is ChromaDB used for?

### Question 7

What information can be associated with a document in ChromaDB?

### Question 8

What is metadata, and why can it be useful in RAG?

### Question 9

What does `n_results=3` specify in a ChromaDB query?

### Question 10

Does ChromaDB generate the final natural-language answer?

### Question 11

What is the purpose of the Sentence Transformer in this laboratory?

### Question 12

What additional component is required to turn this semantic search system into a basic RAG system?

---

# 19. Sample Solutions

## Answer 1

An embedding is a numerical vector representation of an object such as text. It allows the object to be represented in a mathematical space where relationships between objects can be measured.

## Answer 2

A sentence embedding is a vector representation of a sentence or text segment generated by an embedding model.

## Answer 3

Embeddings allow text to be compared numerically. Semantically similar texts tend to have similar vector representations, which makes semantic search possible.

## Answer 4

Keyword search primarily relies on matching terms between a query and documents. Semantic search represents the query and documents as vectors and retrieves documents according to their semantic similarity.

## Answer 5

A vector database is a database designed to store and retrieve vector representations efficiently. It can also store associated documents, identifiers, and metadata.

## Answer 6

ChromaDB is used to store and retrieve embeddings and their associated documents and metadata. In this laboratory, it is used to implement semantic search.

## Answer 7

A ChromaDB record can contain an identifier, document text, an embedding, and metadata.

## Answer 8

Metadata is additional information associated with a document. Examples include the source filename, page number, document type, topic, or section. In RAG, metadata can be used to identify sources, filter documents, and provide source information.

## Answer 9

`n_results=3` specifies that the query should return the three highest-ranked results according to the configured similarity or distance metric.

## Answer 10

No. ChromaDB is a vector database. It retrieves information but does not generate the final natural-language response.

## Answer 11

The Sentence Transformer generates vector representations of the input text. These embeddings allow queries and documents to be compared based on semantic similarity.

## Answer 12

A language model is required to generate an answer from the retrieved information. In the next laboratory, Qwen 2.5 will perform this generation step.

---

# 20. Key Concepts

The main concepts from this laboratory can be summarized as follows:

```text
                    TEXT
                      ↓
             Sentence Transformer
                      ↓
                  EMBEDDING
                      ↓
                 ChromaDB
                      ↓
              Semantic Search
                      ↓
            Relevant Documents
```

For RAG, a generation component is added:

```text
                    TEXT
                      ↓
             Sentence Transformer
                      ↓
                  ChromaDB
                      ↓
            Relevant Documents
                      ↓
                   Qwen
                      ↓
              Generated Answer
```

The next laboratory will extend this process to external documents such as PDF, TXT/Markdown, JSON, and CSV files and will implement the complete RAG pipeline.
