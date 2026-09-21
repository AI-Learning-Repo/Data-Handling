# Retrieval-Augmented Generation (RAG)

## 1. Knowledge Representation in Large Language Models

LLM-based systems can combine knowledge encoded in model parameters (**parametric knowledge**) with information stored externally and retrieved at inference time (**non-parametric memory**). Understanding this distinction is essential when designing systems for domain-specific applications.

```text
                    ┌─────────────────────────────────────────┐
                    │          LLM-Based System                │
                    └─────────────────────────────────────────┘
                                  │           │
                   ┌──────────────┘           └──────────────┐
                   ▼                                         ▼
        ┌───────────────────────┐                 ┌────────────────────────┐
        │   Parametric Memory   │                 │ Non-Parametric Memory  │
        ├───────────────────────┤                 ├────────────────────────┤
        │ • Encoded in weights  │                 │ • Stored externally    │
        │ • Learned during      │                 │ • Databases / indexes  │
        │   training/fine-tuning│                 │ • Can be updated       │
        │ • Expensive to modify │                 │   independently         │
        │ • Example: LoRA       │                 │ • Example: vector DB   │
        └───────────────────────┘                 └────────────────────────┘
```

These two forms of memory are not mutually exclusive. A model using RAG continues to rely on its parametric knowledge, linguistic capabilities, and learned reasoning patterns while additionally conditioning its generation on information retrieved from an external corpus.

### 1.1 Parametric Memory

Parametric memory refers to information and capabilities encoded in the parameters of a neural network. During pre-training and subsequent fine-tuning, optimization modifies the model's weight matrices $W$. In Activity 4, when Qwen 2.5 was fine-tuned using LoRA, the original model weights were kept frozen while trainable low-rank adapter parameters represented an update to the model's behavior:

$$W' = W + \Delta W$$

where $W$ represents the original model parameters and $\Delta W$ represents the learned LoRA adaptation.

Parametric knowledge is therefore stored implicitly within the numerical parameters of the model rather than as individually addressable records.

**Limitations of Parametric Memory:**

* **Knowledge Cutoff:** A model's pretrained knowledge reflects the data available during its training process. Information created or changed after training is not automatically incorporated into its parameters.
* **Domain Isolation:** Proprietary or organization-specific information, such as private APIs, internal personnel directories, confidential procedures, or private databases, is generally unavailable to a pretrained model unless that information is subsequently provided through fine-tuning, prompting, retrieval, or another mechanism.
* **Update Latency and Cost:** Modifying parametric knowledge typically requires additional training or fine-tuning. This involves data preparation, optimization, checkpoint management, evaluation, and computational resources.
* **Catastrophic Forgetting:** Fine-tuning on a narrow distribution can sometimes reduce performance on previously learned tasks or distributions, particularly when the new training data or optimization process is strongly specialized.
* **Hallucination:** A language model generates tokens according to learned probability distributions. When its parameters contain incomplete, ambiguous, or incorrect associations about a subject, the model may generate plausible-sounding but factually unsupported information.

Parametric memory therefore provides powerful generalization and learned behavior, but it is not naturally designed for frequent updates to individual facts.

### 1.2 Non-Parametric Memory

Non-parametric memory stores information separately from the neural network's parameters. Instead of encoding every domain-specific fact into model weights, information can be maintained in an external, indexable data source such as a relational database, document repository, search engine, or vector database.

At inference time, an information-retrieval system can retrieve relevant external information and provide it to the LLM through its context window.

The model is therefore not limited to the information encoded in its parameters. It can condition its response on external evidence supplied at inference time.

A useful distinction is:

```text
Parametric memory
    → information learned into model parameters

Non-parametric memory
    → information stored externally and retrieved when needed
```

Non-parametric memory is particularly useful when information changes frequently or must remain outside the model's weights for security, governance, or operational reasons.

### 1.3 The RAG Paradigm

Retrieval-Augmented Generation (RAG) was introduced in its modern neural formulation by Lewis et al. (2020) as an approach that combines a parametric generator with non-parametric external memory.

A basic RAG system dynamically retrieves relevant information from an external corpus and supplies that information to the language model before generation.

A simplified RAG architecture can be represented as:

```text
                 User Query
                     │
                     ▼
              ┌─────────────┐
              │  Retrieval  │
              └──────┬──────┘
                     │
              Relevant Documents
                     │
                     ▼
              ┌─────────────┐
              │ Augmentation│
              └──────┬──────┘
                     │
             Query + Context
                     │
                     ▼
              ┌─────────────┐
              │ Generation  │
              └──────┬──────┘
                     │
                     ▼
                   Answer
```

A canonical RAG pipeline consists of three conceptual phases:

1. **Retrieval ($R$):** Given a query $q$, a retrieval system searches an external corpus $\mathcal{D}$ and selects relevant documents or document chunks:

$$\{d_1, d_2, \dots, d_k\} \subseteq \mathcal{D}$$

2. **Augmentation ($A$):** The retrieved documents are combined with the original query using a structured prompt template.

3. **Generation ($G$):** The language model generates an output conditioned on the query and retrieved context:

$$P(y \mid q, d_1, \dots, d_k)$$

This three-stage model represents a basic RAG architecture. Modern systems may additionally perform query rewriting, hybrid retrieval, metadata filtering, reranking, iterative retrieval, or evidence verification.

---

## 2. Dense Vector Embeddings and Semantic Distance

Traditional information retrieval often relies on **lexical or sparse retrieval**, including algorithms such as BM25 and TF-IDF. These methods represent documents using sparse term-based representations and calculate relevance partly from token overlap, term frequency, and inverse document frequency.

Lexical retrieval can perform poorly when a query and a relevant document use different vocabulary to express related concepts. For example:

```text
Query:
"Who treats brain conditions?"

Document:
"Dr. Elena Varga leads the neurology department."
```

There may be little direct lexical overlap even though the two statements are semantically related.

Dense retrieval addresses this problem by representing text as numerical vectors in a continuous embedding space.

### 2.1 The Embedding Function

An embedding model is a neural model that maps text into a fixed-dimensional dense vector representation:

$$f: \text{Text} \rightarrow \mathbb{R}^{d}$$

where $d$ is the dimensionality of the embedding space.

For example, `all-MiniLM-L6-v2` produces 384-dimensional sentence embeddings.

Internally, the embedding model processes a sequence of tokens or subword units rather than individual characters. Depending on the model and training procedure, embeddings are learned using objectives designed to place semantically related texts closer together in the resulting representation space. Contrastive learning is one common approach.

Conceptually:

```text
"The doctor examined the patient."
                  │
                  ▼
       Embedding Model
                  │
                  ▼
[0.042, -0.128, 0.891, ..., -0.015]

"A physician checked the patient."
                  │
                  ▼
       Embedding Model
                  │
                  ▼
[0.040, -0.125, 0.885, ..., -0.012]

"A sports car drove down the road."
                  │
                  ▼
       Embedding Model
                  │
                  ▼
[-0.512, 0.781, -0.044, ..., 0.319]
```

The objective is for semantically related texts to tend to occupy nearby regions of the embedding space. However, geometric proximity should be interpreted as a learned measure of similarity rather than as a perfect representation of semantic meaning.

For example, the model may learn that expressions such as *doctor* and *physician* are related because similar linguistic patterns occur throughout its training data. The resulting embeddings can therefore identify relationships that are not obvious from exact token matching.

### 2.2 Mathematical Distance Metrics

Once two text sequences have been converted into vectors:

$$\mathbf{u}, \mathbf{v} \in \mathbb{R}^{d}$$

their relationship can be measured using a distance or similarity function.

Vector databases commonly support metrics such as cosine similarity, inner product, and Euclidean distance.

#### Cosine Similarity

Cosine similarity measures the angle between two vectors and is independent of their magnitude:

$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

or equivalently:

$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\sum_{i=1}^{d} u_i v_i}{\sqrt{\sum_{i=1}^{d} u_i^2} \sqrt{\sum_{i=1}^{d} v_i^2}}$$

The mathematical range is:

$$[-1, 1]$$

A value of $1$ indicates that the vectors point in the same direction, while $0$ indicates that they are orthogonal in the embedding space.

Importantly, a cosine similarity of $0$ should not automatically be interpreted as "no semantic relationship." It means only that the vectors are orthogonal according to that particular embedding representation.

Cosine distance can be defined as:

$$D_{\text{cosine}}(\mathbf{u}, \mathbf{v}) = 1 - \text{Cosine Similarity}(\mathbf{u}, \mathbf{v})$$

#### Dot Product (Inner Product)

The dot product is:

$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{d} u_i v_i$$

If both vectors are normalized to unit length:

$$\|\mathbf{u}\|_2 = \|\mathbf{v}\|_2 = 1$$

then:

$$\mathbf{u} \cdot \mathbf{v} = \text{Cosine Similarity}(\mathbf{u}, \mathbf{v})$$

Thus, normalized inner-product search and cosine-similarity search can produce equivalent rankings.

#### Squared Euclidean ($L_2$) Distance

Euclidean distance measures the straight-line distance between two vectors. Squared Euclidean distance is:

$$D_{L_2}(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2^2$$

or:

$$D_{L_2}(\mathbf{u}, \mathbf{v}) = \sum_{i=1}^{d} (u_i - v_i)^2$$

The range is:

$$[0, \infty)$$

A value of $0$ indicates identical vectors.

The choice of metric matters because different embedding models and normalization strategies can make different metrics more or less appropriate.

---

## 3. Vector Databases and Approximate Nearest Neighbor (ANN) Search

Storing and searching large collections of dense vectors requires specialized indexing techniques. A conventional relational database can store vectors, but ordinary B-tree indexes are not designed to efficiently solve high-dimensional nearest-neighbor search.

Vector databases therefore provide specialized indexes and search algorithms designed for vector similarity.

```text
Query Vector
     │
     ▼
┌──────────────────────────────────────────────┐
│              Vector Database                 │
│                                              │
│   Vector index: HNSW / IVF / other methods  │
│                                              │
│   [Doc 1] •                                  │
│             [Doc 2] •                        │
│                    [Doc 3] •                │
│                                [Doc 4] •    │
│                                              │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
               Candidate vectors
                       │
                       ▼
                Top-k results
                       │
                       ▼
             Document IDs + payloads
```

### 3.1 Exhaustive vs. Approximate Search

A **$k$-Nearest Neighbors ($k$-NN)** search can be performed exhaustively by comparing the query vector against every vector in the collection.

If there are $N$ vectors and each vector has dimensionality $d$, the basic comparison cost is approximately:

$$\mathcal{O}(N \cdot d)$$

This approach provides exact nearest-neighbor results but becomes increasingly expensive as the number of vectors grows.

**Approximate Nearest Neighbor (ANN)** methods reduce search cost by avoiding exhaustive comparison with every vector. The trade-off is that the returned neighbors may not always be the mathematically exact nearest neighbors, meaning recall can decrease depending on the index and its configuration.

Common ANN approaches include:

* **HNSW (Hierarchical Navigable Small World):** Constructs a graph-based index in which vectors are connected to neighboring vectors across multiple hierarchical levels. Search navigates the graph toward regions containing vectors similar to the query.
* **IVF (Inverted File):** Partitions the vector space into clusters and searches a selected subset of those clusters rather than scanning every vector.

ANN methods can provide substantial practical speed improvements, but there is **no universal $\mathcal{O}(\log N)$ guarantee** for ANN search. Actual performance depends on the algorithm, index configuration, dimensionality, dataset distribution, hardware, and desired recall.

The practical trade-off can therefore be summarized as:

```text
More exhaustive search
        │
        ├── Higher recall
        └── Higher computational cost

More approximate search
        │
        ├── Faster retrieval
        └── Potentially lower recall
```

### 3.2 ChromaDB Core Architecture

ChromaDB is an open-source vector database designed for storing and querying embeddings and associated document data.

A simplified ChromaDB data model contains several important components:

1. **Collections:** Logical groups of records, roughly analogous to tables in a relational database.
2. **Documents:** Human-readable text associated with records.
3. **Embeddings:** Dense numerical vector representations associated with the documents. These can be supplied by the application or generated through an embedding function.
4. **Metadata:** Structured key-value information associated with records, such as:

```python
{
    "department": "neurology",
    "author": "admin"
}
```

Metadata can be used to apply deterministic filtering constraints alongside similarity search.

For example:

```text
Query:
"Who treats neurological conditions?"

Vector similarity
        +
department == "neurology"
        ↓
Relevant filtered candidates
        ↓
Top-k results
```

This allows semantic similarity and structured database constraints to work together.

---

## 4. Document Ingestion and Chunking

In production environments, information is rarely stored as individual, perfectly sized text records. Important information often resides in PDFs, DOCX files, web pages, databases, spreadsheets, and other heterogeneous sources.

A RAG system therefore requires an ingestion pipeline that converts source material into representations suitable for retrieval.

```text
┌──────────────┐
│   Raw Files  │
│ PDF / DOCX   │
│ JSON / HTML  │
└──────┬───────┘
       │
       ▼
┌────────────────┐
│ Text / Structure│
│   Extraction    │
└──────┬─────────┘
       │
       ▼
┌────────────────┐
│    Chunking    │
│    Strategy    │
└──────┬─────────┘
       │
       ▼
┌────────────────┐
│   Embedding    │
│   & Indexing   │
└────────────────┘
```

### 4.1 Text Extraction

Before vectorization, unstructured files generally need to be converted into a representation that preserves the relevant textual and structural information.

* **PDF Files (`pypdf`):** A PDF parser can extract text from the document's internal objects. The quality of the resulting text depends on how the PDF was created and how its layout is represented.
* **Word Documents (`python-docx`):** A DOCX file is an Office Open XML package. Libraries such as `python-docx` can traverse document elements and extract paragraph and other structural content.
* **JSON:** Structured fields can be extracted directly using a JSON parser such as Python's `json` module.

The LLM is generally **file-format agnostic but representation-sensitive**.

For example, if a PDF contains:

```text
Drug        Dosage
A           10 mg
B           20 mg
```

and the extraction process produces:

```text
Drug Dosage A 10 mg B 20 mg
```

the model does not know that the source was a PDF, but the loss of table structure may still affect its ability to interpret the information correctly.

Therefore, ingestion quality is a critical component of RAG performance.

### 4.2 The Need for Chunking

Large documents cannot always be embedded effectively as single units. Embedding models have input limits, and a single vector may be insufficiently specific to represent a long document containing many unrelated topics.

For example, a 50-page document could contain:

```text
Page 1–10    → Neurology
Page 11–20   → Cardiology
Page 21–30   → Administration
Page 31–40   → Billing
Page 41–50   → Emergency procedures
```

Representing the entire document with one vector would force a single embedding to represent all of these concepts.

The document is therefore divided into smaller semantic units called **chunks**.

```text
Document:

[──────── Chunk 1 ────────]
                 [──────── Chunk 2 ────────]
                                  [──────── Chunk 3 ────────]
                  ◄── overlap ──►
```

Chunking is not merely a method for satisfying context-window limits. It is also an **information-retrieval design decision**.

The goal is to create chunks that are:

* sufficiently small to retrieve precisely,
* sufficiently large to preserve necessary context,
* semantically coherent,
* compatible with the embedding model's input limits.

### 4.3 Chunking Parameters and Trade-offs

**Chunk Size:** The amount of text assigned to each chunk, typically measured in tokens or characters.

* **Too small:** A chunk may contain insufficient context to establish what a statement means.
* **Too large:** A chunk may contain multiple unrelated topics, reducing retrieval specificity and consuming more context tokens during generation.

There is no universally optimal chunk size. The appropriate value depends on the document structure, embedding model, retrieval task, and downstream generation model.

For example:

```text
Legal document:
Section-based chunking may be preferable.

Technical documentation:
Heading + subsection chunking may be preferable.

Conversation:
Turn- or topic-based chunking may be preferable.
```

**Chunk Overlap:** A portion of text shared between adjacent chunks.

```text
Chunk 1:
[A B C D E F]

Chunk 2:
          [E F G H I J]

Overlap:
          [E F]
```

Overlap reduces the chance that information spanning a chunk boundary will become separated from the context necessary to interpret it.

However, excessive overlap increases storage requirements and can cause redundant retrieval results.

The embedding model's own input limitations must also be considered. For example, `all-MiniLM-L6-v2` has a model-specific input limitation and truncates inputs beyond its supported sequence length. Therefore, chunk sizes must be selected with the embedding model's tokenizer and maximum input length in mind.

---

## 5. Augmented Generation and Prompt Engineering

Once relevant chunks have been retrieved, they must be incorporated into the LLM's input in a structured manner.

The purpose of prompt construction is to clearly distinguish:

```text
System instructions
        +
Retrieved evidence
        +
User question
        ↓
Language model
```

The retrieved material should be treated as evidence rather than as instructions that automatically override the system's intended behavior.

### 5.1 ChatML Formatting in Qwen 2.5

Instruction-tuned chat models are trained using specific conversational formats. Qwen 2.5 uses a chat-template format based on special control tokens to distinguish system, user, and assistant messages.

Conceptually:

```text
<|im_start|>system
[System instructions]
<|im_end|>

<|im_start|>user
[Retrieved context]
[User question]
<|im_end|>

<|im_start|>assistant
```

When using Qwen 2.5 through its tokenizer, the recommended approach is to construct the conversation as structured messages and apply the tokenizer's chat template.

Conceptually:

```python
tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True
)
```

Using the chat template ensures that the input is formatted according to the conversational structure expected by the model.

Setting:

```python
add_generation_prompt=True
```

adds the assistant-generation marker expected by the chat format. This does not function as a literal mathematical switch; rather, it provides the conversational structure on which the model was trained and signals that the following tokens should constitute the assistant response.

### 5.2 Controlling Hallucinations via Evidence Constraints

Language models can generate information that is not supported by retrieved documents. A system prompt can reduce this behavior by instructing the model to use the supplied context as its evidence source.

For example:

> "Answer the question using only the provided context. If the answer cannot be determined from the context, state that the information is not available."

This instruction biases the model toward a closed-domain question-answering behavior.

However, it is important to distinguish **instructional constraints** from **hard guarantees**.

A system prompt does not mathematically prevent hallucination. The model can still generate unsupported information even when explicitly instructed not to do so.

Therefore, robust grounded-generation systems may combine prompt instructions with:

* retrieval score thresholds,
* metadata filtering,
* reranking,
* citation requirements,
* evidence extraction,
* answer verification,
* structured output validation,
* post-generation checks.

The overall objective is not simply to tell the model not to hallucinate, but to design the entire pipeline so that unsupported answers are less likely.

### 5.3 Deterministic Decoding

During retrieval evaluation, generation should generally be made as reproducible as practical.

Setting:

```python
do_sample=False
```

selects greedy decoding in standard transformer-generation implementations. At each step, the token with the highest conditional probability is selected:

$$w_t = \arg\max_w P(w \mid w_{1:t-1}, q, d_1, \dots, d_k)$$

Greedy decoding reduces one source of experimental variance because it does not sample alternative tokens from the probability distribution.

This is useful when comparing retrieval configurations because repeated runs can be made more consistent.

However, deterministic decoding does **not** mean that every generation error can be attributed directly to retrieval quality. Errors may also originate from:

* model capabilities,
* prompt design,
* context ordering,
* conflicting evidence,
* context length,
* generation strategy,
* tokenizer behavior,
* insufficiently informative retrieved documents.

Therefore, deterministic decoding should be understood as a method for **reducing experimental variance**, rather than as a guarantee that retrieval is the only source of error.

---

## 6. Architectural Synthesis: RAG vs. LoRA Fine-Tuning

RAG and Parameter-Efficient Fine-Tuning (LoRA) address different aspects of domain adaptation.

A useful conceptual distinction is:

```text
                  Domain Adaptation
                         │
              ┌──────────┴──────────┐
              │                     │
        Change model             Supply evidence
          behavior                 at runtime
              │                     │
            LoRA                    RAG
              │                     │
        "How should I            "What information
         respond?"                should I consider?"
```

This distinction should not be interpreted as an absolute boundary. LoRA can encode domain knowledge, while retrieved information can influence behavior. The distinction is primarily about where the adaptation is stored and how it is updated.

### 6.1 Direct Comparison

| Architectural Dimension | LoRA Fine-Tuning | RAG Pipeline |
| :--- | :--- | :--- |
| **Knowledge Placement** | **Parametric:** learned through model or adapter parameters | **Non-parametric:** maintained in external data stores |
| **Update Mechanism** | Requires additional optimization/training | External documents and indexes can be updated independently of model weights |
| **Update Latency** | Typically requires substantially more computation and validation | Often much faster because the generator does not need retraining |
| **Auditability & Citations** | Internal parameter knowledge is difficult to trace to individual training examples | Retrieved documents/chunks can be logged, referenced, and cited |
| **Hallucination** | Fine-tuning does not inherently provide source-grounded generation | Retrieval can improve grounding when relevant evidence is retrieved, but does not eliminate hallucination |
| **Token Cost** | No retrieval context is required unless used with RAG | Retrieved chunks consume context-window tokens |
| **Inference Architecture** | Model forward pass with adapted parameters | Retrieval/indexing stage followed by model generation |
| **Best Suited For** | Learned behavior, formatting, style, task-specific patterns | Dynamic facts, external knowledge, access-controlled information, and source-grounded answers |

### 6.2 LoRA: Changing How the Model Behaves

LoRA is particularly useful when the desired change concerns **learned behavior**.

Examples include:

* response style,
* domain-specific terminology,
* output structure,
* specialized task behavior,
* formatting conventions,
* repeated task-specific transformations.

For example, a model could be fine-tuned to consistently produce:

```json
{
  "diagnosis": "...",
  "confidence": 0.0,
  "evidence": []
}
```

The model learns the pattern through training rather than receiving a new example every time it generates an answer.

### 6.3 RAG: Supplying What the Model Should Consider

RAG is particularly useful when the information itself is external, dynamic, proprietary, or frequently updated.

Examples include:

* current clinic schedules,
* internal company policies,
* product documentation,
* personnel directories,
* revised regulations,
* private databases,
* customer-specific records.

Instead of modifying model parameters, the system retrieves the relevant information at inference time.

### 6.4 The Hybrid Enterprise Architecture

In production systems, LoRA and RAG can be combined.

A hybrid architecture can be represented as:

```text
[Raw User Query]
        │
        ▼
┌──────────────────────────────────────────────────────┐
│ 1. Retrieval System                                  │
│                                                      │
│ Vector / hybrid search                              │
│ Metadata filtering                                  │
│ Optional reranking                                  │
└───────────────────────┬──────────────────────────────┘
                        │
                        ▼
             Retrieved evidence chunks
                        │
                        ▼
┌──────────────────────────────────────────────────────┐
│ 2. Fine-Tuned Generator                             │
│                                                      │
│ Base Model + LoRA Adapter                            │
│                                                      │
│ • Specialized response behavior                     │
│ • Formatting requirements                            │
│ • Domain terminology                                 │
│ • Output schema                                      │
└───────────────────────┬──────────────────────────────┘
                        │
                        ▼
              Grounded model response
                        │
                        ▼
              Optional verification
```

The conceptual division is:

> **LoRA can teach the model how to process and present information.**

> **RAG can provide the information that needs to be processed.**

For example:

```text
LoRA:
"Answer in a formal clinical style and output validated JSON."

RAG:
"Here are the current clinic schedules and relevant patient-specific records."

Combined:
"Use the retrieved information and present the answer according to the learned output format."
```

This combination allows model behavior and external knowledge to be managed through separate mechanisms.

---

## 7. Review and Technical Questions & Answers

### Question 1: Does the generative language model care whether the source document was originally a PDF, DOCX, or JSON file?

**Answer:**

Generally, no. The generation component is largely **file-format agnostic**, but it is **representation-sensitive**.

By the time information reaches the language model, source files have normally undergone upstream extraction and normalization.

For example:

1. Parsers such as `pypdf`, `python-docx`, or `json.loads()` extract relevant content from the source representation.
2. The ingestion pipeline converts this content into text and/or structured records suitable for indexing.
3. The embedding model converts the selected text into vectors for retrieval.
4. The prompt assembly process places retrieved content into the LLM's context.

The model's tokenizer processes the resulting representation as a sequence of tokens. It generally does not receive a signal saying:

```text
"This sentence originally came from a PDF."
```

However, the distinction between file format and extracted representation is important.

If a PDF parser incorrectly extracts a table:

```text
Original:
Drug       Dosage
A          10 mg
B          20 mg
```

into:

```text
Drug Dosage A 10 mg B 20 mg
```

the model does not fail because it cannot understand PDFs. It fails because information about the original structure was lost during preprocessing.

Therefore:

> **The model is largely file-format agnostic, but it is not representation-agnostic.**

The quality of extraction and preservation of important structure can have a direct effect on retrieval and generation quality.

---

### Question 2: Why did semantic retrieval in Activity 3 find the neurology department lead when queried with "Who is in charge of brain conditions?", while lexical search could fail?

**Answer:**

Lexical retrieval methods such as BM25 primarily depend on relationships between query terms and document terms. If a query contains:

```text
"charge"
"brain"
"conditions"
```

while a document contains:

```text
"Dr. Elena Varga leads the neurology department."
```

there may be little direct lexical overlap.

A lexical retrieval system may therefore assign the document a low relevance score, depending on its tokenization, preprocessing, stemming, query expansion, and other configuration.

Dense retrieval uses a different representation.

The embedding model maps both the query and document into a continuous vector space:

$$q = f(\text{query})$$

$$d = f(\text{document})$$

The vector database then calculates their similarity.

Because the embedding model has learned representations in which semantically related expressions tend to be closer together, a query about *brain conditions* can potentially retrieve a document about *neurology* even when the exact words do not occur in both texts.

Conceptually:

```text
"brain conditions"
       │
       ▼
   Embedding
       │
       ▼
   Vector A
       │
       │   high similarity
       ▼
   Vector B
       │
       ▼
"neurology department"
```

The important point is that dense retrieval is not performing literal synonym substitution. Rather, the embedding model has learned a representation in which related linguistic concepts can occupy nearby regions of vector space.

This allows semantic retrieval to recover relevant documents despite vocabulary differences.

---

### Question 3: In Activity 4, why was it necessary to pass retrieved context through `tokenizer.apply_chat_template()` rather than simply concatenating it to the model input?

**Answer:**

Instruction-tuned chat models are trained using structured conversational formats. Qwen 2.5 uses special control tokens to distinguish system, user, and assistant messages.

The model therefore expects an input structure conceptually similar to:

```text
<|im_start|>system
[System instructions]
<|im_end|>

<|im_start|>user
[Retrieved context]
[User question]
<|im_end|>

<|im_start|>assistant
```

Simply concatenating text can still result in a sequence that the model can process, but it may not provide the conversational structure expected by the model's instruction-tuning process.

Using:

```python
tokenizer.apply_chat_template()
```

provides two important benefits:

1. It formats the system, user, and assistant messages according to the model's expected chat template.
2. With `add_generation_prompt=True`, it appends the assistant-generation marker expected by the model's chat format.

The generation marker should not be interpreted as a literal mathematical trigger. Rather, it reproduces the conversational structure on which the model was trained and indicates that the next generated tokens should constitute the assistant's response.

Therefore, the chat template helps ensure that retrieved context and the user's question are presented within the instruction-following format expected by the model.

---

### Question 4: Nearest-neighbor search always returns $k$ documents, even when the query topic does not exist in the database. How was hallucination prevented in Activity 4 when querying about non-existent services?

**Answer:**

A nearest-neighbor search does not inherently understand the concept of "no relevant document."

If a collection contains:

```text
Document A → similarity 0.91
Document B → similarity 0.83
Document C → similarity 0.79
```

and a completely unrelated query produces:

```text
Document A → similarity 0.31
Document B → similarity 0.29
Document C → similarity 0.27
```

the database may still return A, B, and C as the nearest neighbors.

This means that:

> **The nearest document is not necessarily a relevant document.**

A RAG system therefore needs some mechanism for determining whether the retrieved evidence is sufficiently relevant.

A simple architecture is:

```text
Query
  │
  ▼
Retriever
  │
  ▼
Top-k candidates
  │
  ▼
Relevance threshold / reranking
  │
  ├── Sufficient evidence ──► Generator
  │
  └── Insufficient evidence ─► "Information not available."
```

In the described activity, the system prompt provided a negative constraint such as:

> "Answer the question using ONLY the provided context. If the answer cannot be determined from the context, state: 'Information not available.'"

This instruction encourages the model to abstain when the retrieved context does not support an answer.

However, the prompt does not provide a formal guarantee against hallucination. A more robust production system would combine the instruction with retrieval thresholds, reranking, metadata filtering, evidence verification, or other grounding mechanisms.

Thus, the important principle is:

> **RAG reduces the model's dependence on unsupported parametric knowledge by supplying external evidence, but retrieval and generation must both be controlled to achieve reliable grounding.**

---

### Question 5: When Dr. Elena Varga was replaced by Dr. Arto Virtanen, how did the mechanical update process in RAG contrast with the LoRA approach?

**Answer:**

The two approaches store the information in fundamentally different locations.

**In LoRA Fine-Tuning:**

The information can become encoded into learned model or adapter parameters. Updating the information generally requires modifying the training data and performing another optimization process.

A typical workflow is:

```text
Updated training data
        │
        ▼
Training / fine-tuning
        │
        ▼
New adapter parameters
        │
        ▼
Validation
        │
        ▼
Deploy updated adapter
```

This requires substantially more computation and operational work than modifying an external database record.

**In RAG:**

The personnel information remains in the external knowledge store.

A simplified update is:

```text
Old record:
"Dr. Elena Varga leads neurology."

        ↓

Database update

        ↓

New record:
"Dr. Arto Virtanen leads neurology."

        ↓

New embedding / updated index

        ↓

Future retrieval
```

The LLM's parameters do not need to change.

The exact implementation depends on the database and indexing configuration, but the fundamental advantage is that the external knowledge source can be updated independently of the generator's weights.

Therefore, if the fact changes frequently, RAG can avoid repeatedly retraining the language model.

---

### Question 6: Why is greedy decoding (`do_sample=False`) used when evaluating RAG pipelines instead of stochastic sampling (`do_sample=True`)?

**Answer:**

Stochastic decoding introduces variation into generation by sampling from the model's probability distribution.

For example, temperature, top-$p$, and related methods can cause the model to select different tokens across otherwise identical runs.

With greedy decoding:

```python
do_sample=False
```

the decoder selects the token with the highest probability at each generation step:

$$w_t = \arg\max_w P(w \mid w_{1:t-1}, q, d_1, \dots, d_k)$$

This reduces generation randomness and makes repeated experiments more reproducible.

For example, suppose two retrieval systems are being compared:

```text
System A → retrieves documents X, Y, Z
System B → retrieves documents X, Y, W
```

If generation is stochastic, differences in answers could result from either:

```text
retrieval differences
        +
random generation differences
```

Using greedy decoding reduces the second source of variation:

```text
retrieval differences
        +
more deterministic generation
```

This makes it easier to attribute observed differences to changes in the retrieval or prompting pipeline.

However, greedy decoding does not eliminate every source of experimental variation or guarantee higher factual accuracy. It is primarily useful for **reproducibility and controlled evaluation**.

---

## 8. Architectural Design Principles

The preceding sections lead to several general principles for designing RAG systems.

### 8.1 Retrieval Quality Is as Important as Generation Quality

A powerful language model cannot reliably answer a question from evidence that was never retrieved.

The pipeline can therefore be represented as:

```text
                 RAG QUALITY
                     │
          ┌──────────┴──────────┐
          │                     │
    Retrieval quality      Generation quality
          │                     │
    "Did we retrieve       "Did the model
     the right evidence?"   use it correctly?"
```

A retrieval failure can occur before the LLM has any opportunity to generate a correct answer.

### 8.2 Retrieval Does Not Automatically Mean Grounding

A system can technically be called RAG while still producing unsupported answers.

For example:

```text
Query
  ↓
Retrieve irrelevant document
  ↓
Give document to LLM
  ↓
LLM produces confident answer
```

This is still a retrieval-augmented pipeline, but it is not necessarily a reliable grounded-generation system.

Grounded RAG therefore requires attention to:

* retrieval relevance,
* evidence quality,
* context construction,
* model adherence to evidence,
* abstention behavior,
* citation/provenance,
* evaluation methodology.

### 8.3 The "Top-k" Parameter Is Not a Guarantee of Relevance

Increasing $k$ does not necessarily improve answer quality.

If:

$$k = 3$$

the model receives three retrieved passages.

If:

$$k = 20$$

it receives twenty passages.

The additional passages may contain:

* useful evidence,
* redundant information,
* unrelated information,
* conflicting information,
* distracting information.

Therefore, retrieval systems should optimize not simply for the number of documents retrieved but for the **quality and relevance of the evidence supplied to the generator**.

---

## 9. Final Conceptual Model

The complete architecture can be summarized as follows:

```text
                         USER QUERY
                              │
                              ▼
                    ┌──────────────────┐
                    │ Query Processing │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Retrieval     │
                    │                  │
                    │ Dense / Sparse / │
                    │ Hybrid Search    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Filtering /      │
                    │ Reranking        │
                    └────────┬─────────┘
                             │
                             ▼
                    Relevant Evidence
                             │
                             ▼
                    ┌──────────────────┐
                    │   Augmentation   │
                    │                  │
                    │ Query + Context  │
                    │ + Instructions   │
                    └────────┬─────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │       LLM Generator         │
              │                              │
              │ Base Model + Optional LoRA  │
              └──────────────┬───────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Generation     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Verification /   │
                    │ Validation       │
                    └────────┬─────────┘
                             │
                             ▼
                       FINAL ANSWER
```

The central architectural distinction is therefore:

$$\boxed{\text{LoRA modifies learned model behavior}}$$

while:

$$\boxed{\text{RAG supplies external evidence at inference time}}$$

and a robust production system can combine both:

$$\boxed{\text{Base LLM} + \text{LoRA behavior adaptation} + \text{External retrieval} + \text{Evidence-aware generation}}$$

This separation allows frequently changing knowledge to remain in an external data layer while learned response behavior remains in the model parameters.

---

## 10. References and Further Reading

1. [RAG with Python Cookbook](https://github.com/polzerdo55862/RAG-with-Python-Cookbook/tree/main)
2. 