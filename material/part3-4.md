# Retrieval-Augmented Generation (RAG)


## 1. Knowledge Representation in Large Language Models

Large language models (LLMs) acquire and store knowledge through two distinct mechanisms: **parametric memory** and **non-parametric memory**. Understanding this distinction is essential when designing systems for domain-specific applications.

```
                    ┌─────────────────────────────────────────┐
                    │       Large Language Model System       │
                    └─────────────────────────────────────────┘
                                   │           │
                 ┌─────────────────┘           └─────────────────┐
                 ▼                                               ▼
     ┌───────────────────────┐                       ┌───────────────────────┐
     │   Parametric Memory   │                       │ Non-Parametric Memory │
     ├───────────────────────┤                       ├───────────────────────┤
     │ • Stored in weights   │                       │ • Stored externally   │
     │ • Static after train  │                       │ • Vector databases    │
     │ • Expensive to update │                       │ • Updated in real time│
     │ • Example: LoRA (W)   │                       │ • Example: ChromaDB   │
     └───────────────────────┘                       └───────────────────────┘
```

### 1.1 Parametric Memory
Parametric memory refers to the information encoded directly into the neural network's mathematical weight matrices ($W$) during training. In Activity 4, when you fine-tuned Qwen 2.5 using LoRA, you altered these weight parameters ($\Delta W$).

**Limitations of Parametric Memory:**
* **Knowledge Cutoff:** An LLM cannot know information created after its final training epoch.
* **Domain Isolation:** Proprietary or organizational data (such as hospital records, private APIs, or internal personnel directories) does not exist in public pre-training corpora.
* **Update Latency and Cost:** Modifying parametric memory requires running backpropagation across batches of data, requiring dedicated compute time and validation.
* **Catastrophic Forgetting:** When a model is optimized on a narrow distribution of new facts, it risks degrading its performance on previously learned distributions.
* **Hallucination:** When parametric memory contains weak or incomplete statistical correlations regarding a topic, the autoregressive generation process generates plausible-sounding but factually false tokens.

### 1.2 Non-Parametric Memory
Non-Parametric memory decouples knowledge storage from the neural network's parameters. Instead of forcing the model to memorize facts, knowledge is stored in an external, indexable data store (such as a relational database or vector store). At inference time, the model acts strictly as a reasoning engine over the data supplied to it in its input context.

### 1.3 The RAG Paradigm
First formalized by Lewis et al. (2020) [1], **Retrieval-Augmented Generation (RAG)** is an architecture that dynamically fetches relevant non-parametric data and supplies it directly to the model's prompt before token generation begins. 

A RAG pipeline consists of three sequential phases:
1. **Retrieval ($R$):** Given a query $q$, a retrieval model queries an external corpus $\mathcal{D}$ and extracts the $k$ most relevant document chunks: 
   $$\{d_1, d_2, \dots, d_k\} \subset \mathcal{D}$$
2. **Augmentation ($A$):** The retrieved documents are combined with $q$ into a unified, structured prompt context string using a specified template.
3. **Generation ($G$):** The augmented prompt is passed to the LLM, which generates a conditioned response:
   $$P(y \mid q, d_1, \dots, d_k)$$

---

## 2. Dense Vector Embeddings and Semantic Distance

Traditional information retrieval relies on **lexical (sparse) search**, such as BM25 or TF-IDF. These algorithms index exact tokens and compute relevance using term frequency and inverse document frequency. Lexical search fails when queries and target documents express identical concepts using different vocabularies (e.g., *"physician"* vs. *"doctor"*).

RAG resolves this limitation through **dense semantic retrieval**, which transforms text into continuous vector representations.

### 2.1 The Embedding Function
An embedding model is a neural network (typically an encoder architecture, such as BERT or MiniLM) trained via contrastive learning to map a sequence of text characters into a dense, continuous vector space:

$$f: \text{Text} \to \mathbb{R}^d$$

Where $d$ is the dimensionality of the vector space (for example, $d = 384$ for `all-MiniLM-L6-v2`, or $d = 1536$ for larger commercial models).

```
"The doctor examined the patient."  ──► [ 0.042, -0.128,  0.891, ..., -0.015 ] (384 dimensions)
"A physician checked the patient."  ──► [ 0.040, -0.125,  0.885, ..., -0.012 ] (384 dimensions)
"A sports car drove down the road." ──► [-0.512,  0.781, -0.044, ...,  0.319 ] (384 dimensions)
```

In this space, semantic similarity corresponds to geometric proximity: vectors pointing in similar directions represent texts with similar meanings.

### 2.2 Mathematical Distance Metrics
Once two text sequences are converted to vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$, their similarity is determined mathematically. Vector databases typically support three core metrics:

#### Cosine Similarity
Measures the cosine of the angle $\theta$ between two vectors, normalizing for vector magnitude:

$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^d u_i v_i}{\sqrt{\sum_{i=1}^d u_i^2} \sqrt{\sum_{i=1}^d v_i^2}}$$

* Output range: $[-1.0, 1.0]$.
* A value of $1.0$ indicates identical orientation; $0.0$ indicates orthogonality (no semantic correlation).
* **Cosine Distance** is defined as:
  $$D_{\text{cosine}}(\mathbf{u}, \mathbf{v}) = 1 - \text{Cosine Similarity}(\mathbf{u}, \mathbf{v})$$

#### Dot Product (Inner Product)
$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^d u_i v_i$$

If all embeddings are normalized to unit length ($\|\mathbf{u}\|_2 = 1$), the dot product is mathematically equivalent to cosine similarity, but requires fewer floating-point operations.

#### Squared Euclidean ($L_2$) Distance
Measures the straight-line geometric distance between two points in $\mathbb{R}^d$:

$$D_{L2}(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2^2 = \sum_{i=1}^d (u_i - v_i)^2$$

* Output range: $[0, \infty)$.
* A value of $0.0$ indicates identical vectors. ChromaDB uses squared $L_2$ distance by default.

---

## 3. Vector Databases and Approximate Nearest Neighbor (ANN) Search

Storing and searching millions of dense vectors requires specialized data infrastructure. Relational databases (e.g., PostgreSQL without vector extensions) are optimized for scalar values and B-tree indexes, which degrade when searching high-dimensional vector spaces.

```
Query Vector (q)
       │
       ▼
┌───────────────────────────────────────────────────────────┐
│                    Vector Database                        │
│                                                           │
│   [Doc 1] •                   [Doc 4] •                   │
│               [Doc 2] •                                   │
│                           ◄─── ANN Search identifies      │
│                                nearest cluster (Doc 2, 3) │
│       [Doc 3] •                                           │
└───────────────────────────────────────────────────────────┘
       │
       ▼
Top-k Document IDs and Payloads Returned
```

### 3.1 Exhaustive vs. Approximate Search
* **$k$-Nearest Neighbors ($k$-NN):** Performs a brute-force linear scan across every vector in the database, calculating distance against the query vector.
  * Time Complexity: $\mathcal{O}(N \cdot d)$, where $N$ is the total number of documents and $d$ is vector dimensionality.
  * Unviable for production datasets containing millions of items.
* **Approximate Nearest Neighbors (ANN):** Trade a negligible degree of search recall for logarithmic search speeds ($\mathcal{O}(\log N)$). 
  * Common algorithms include **HNSW (Hierarchical Navigable Small World graphs)** and **IVF (Inverted File indexes)**. These group vectors into spatial clusters, allowing the database to search only the most relevant clusters.

### 3.2 ChromaDB Core Architecture
ChromaDB is an open-source, embedded vector database designed for direct integration with Python workflows. Its data model is organized into four layers:

1. **Collections:** The primary organizational unit, analogous to a table in SQL. Each collection uses a specified distance metric and embedding function.
2. **Documents:** The raw human-readable text payloads.
3. **Embeddings:** The dense vectors associated with each document, either supplied manually or computed automatically by Chroma's internal embedding worker.
4. **Metadata:** Unindexed key-value pairs associated with each document (e.g., `{"department": "neurology", "author": "admin"}`) used for deterministic boolean filtering during vector queries.

---

## 4. Document Ingestion and Chunking

In production environments, information is rarely packaged as individual, single-sentence records. Information typically resides in unstructured documents such as PDFs, DOCX files, and raw text files. Preparing these files for RAG requires an ingestion pipeline:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Raw Files   │ ──► │     Text     │ ──► │   Chunking   │ ──► │  Embedding   │
│ (PDF / DOCX) │     │  Extraction  │     │   Strategy   │     │  & Indexing  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

### 4.1 Text Extraction
Before vectorization, unstructured files must have their text layers extracted into plain UTF-8 strings:
* **PDF Files (`pypdf`):** Reads the internal PDF layout trees, parses stream objects, and extracts raw string sequences while ignoring binary font data and inline images.
* **Word Documents (`python-docx`):** Parses the underlying Office Open XML (`.docx`) file package, traversing the XML DOM to extract text from `<w:p>` (paragraph) elements.

### 4.2 The Need for Chunking
LLMs have a finite context window (e.g., 2,048 or 8,192 tokens), and embedding models typically have even smaller input limits (e.g., 256 or 512 tokens for `all-MiniLM-L6-v2`). Furthermore, passing an entire 50-page document as a single embedding dilutes the semantic signal, as a single 384-dimensional vector cannot represent dozens of divergent topics simultaneously.

Text must therefore be partitioned into **chunks**:

```
Document Text:
[--- Chunk 1 (150 chars) ---]
                  [--- Chunk 2 (150 chars) ---]
                                    [--- Chunk 3 (150 chars) ---]
                  ◄─ Overlap ─►     ◄─ Overlap ─►
```

### 4.3 Chunking Parameters and Trade-offs
* **Chunk Size:** The number of characters or tokens per passage.
  * *Too small (e.g., 20 tokens):* Chunks lack sufficient context for the generation model to comprehend the broader meaning.
  * *Too large (e.g., 1000 tokens):* The embedding averages too many distinct concepts together, degrading retrieval precision.
* **Chunk Overlap:** A sliding window parameter that preserves a specified number of characters or tokens across chunk boundaries.
  * *Purpose:* Prevents semantic continuity from breaking if an important sentence happens to cross the boundary between two adjacent chunks.

---

## 5. Augmented Generation and Prompt Engineering

Once the top-$k$ document chunks are retrieved from the vector store, they must be formatted into a structured prompt that the language model can parse without confusing reference data with user instructions.

### 5.1 ChatML Formatting in Qwen 2.5
Modern chat models do not take unstructured text streams. They rely on explicit control tokens to delineate roles. Qwen 2.5 implements the **ChatML** format, which structures inputs using distinct blocks:

```text
<|im_start|>system
[System Instructions and Behavioral Constraints]<|im_end|>
<|im_start|>user
[Injected Context + User Question]<|im_end|>
<|im_start|>assistant
```

When constructing a RAG prompt, the retrieved chunks are formatted into a designated section within the user message (or system context), followed by the user's specific inquiry.

### 5.2 Controlling Hallucinations via Negative Constraints
Language models are trained to continue text patterns. When an answer is missing from the retrieved context, a default model will often generate a statistically plausible answer using its parametric weights.

To enforce factual boundaries, the system prompt must include **negative constraints**:

> *"Answer the question using ONLY the provided context. If the answer cannot be determined from the context, state that the information is not available."*

This constraint alters the model's objective: it converts the task from open-domain text generation to a **closed-domain extraction and summarization** task.

### 5.3 Deterministic Decoding
During retrieval evaluation, token generation should remain deterministic. Setting `do_sample=False` activates **greedy decoding**, where the model selects the single highest-probability token at each generation step:

$$w_t = \arg\max_w P(w \mid w_{1:t-1})$$

This eliminates stochastic variation, ensuring that generation quality reflects the retrieved context rather than random sampling variance.

---

## 6. Architectural Synthesis: RAG vs. LoRA Fine-Tuning

RAG and Parameter-Efficient Fine-Tuning (LoRA) address domain adaptation from opposite architectural directions. They are complementary techniques suited to different operational constraints.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Design Decision Matrix                          │
└────────────────────────────────────────────────────────────────────────┘
                    │                                   │
                    ▼                                   ▼
         What needs modification?           What needs modification?
         THE MODEL'S BEHAVIOR                THE MODEL'S KNOWLEDGE
                    │                                   │
                    ▼                                   ▼
          Use LoRA Fine-Tuning                      Use RAG
  • Tone and voice                     • Proprietary enterprise facts
  • Complex output format (JSON)       • Rapidly updating information
  • Specialized terminology/jargon     • High risk of hallucination
  • Low inference-time token budget    • Strict auditing/citation needs
```

### 6.1 Direct Comparison

| Architectural Dimension | LoRA Fine-Tuning (Activity 4) | RAG Pipeline (Activity 5) |
| :--- | :--- | :--- |
| **Knowledge Placement** | **Parametric:** Embedded in adapter weights ($\Delta W$) | **Non-Parametric:** Maintained in external database |
| **Update Latency** | **High:** Requires batch preparation, backpropagation, and checkpoint validation | **Instantaneous:** Millisecond-level database CRUD operations |
| **Auditability & Citations** | **None:** The model cannot output deterministic source pointers for its internal weights | **Complete:** The exact chunk IDs and text inputs are known and loggable |
| **Hallucination Risk** | **Moderate to High:** Subject to parametric drift and statistical approximations | **Low:** Constrained directly by the injected context window |
| **Token Cost / Context Usage** | **Zero Overhead:** The prompt contains only the user query | **High Overhead:** Chunks consume tokens from the model's context window |
| **Inference Compute** | Standard model forward pass | Two-stage: Embedding search + forward pass over extended context |
| **Best Suited For** | Form, style, syntax, specialized linguistic behavior | Dynamic facts, access-controlled data, external knowledge |

### 6.2 The Hybrid Enterprise Architecture
In production systems, these two methodologies are rarely mutually exclusive. Production AI architectures use a **hybrid pattern**:

```
[Raw User Query]
       │
       ▼
┌────────────────────────────────────────────────────────────────┐
│ 1. Vector Database (ChromaDB / Milvus / Qdrant)                │
│    Retrieves verified, current domain knowledge chunks         │
└────────────────────────────────────────────────────────────────┘
       │
       ▼  (Retrieved Context Chunks)
┌────────────────────────────────────────────────────────────────┐
│ 2. Fine-Tuned Model (Base Model + LoRA Adapter)                │
│    • Adapter enforces tone, compliance rules, and JSON schema  │
│    • Base model reasons over the retrieved context             │
└────────────────────────────────────────────────────────────────┘
       │
       ▼
[Deterministic, Schema-Compliant, Factually Grounded Output]
```

1. **LoRA Fine-Tuning** trains the model on **how to process and format information** (e.g., adopting a formal clinical tone, refusing disallowed categories, outputting validated JSON).
2. **RAG** supplies the model with **what information to process** (e.g., current clinic schedules, specific patient files, revised regulatory documents).

---

## 7. References and Further Reading

1. 
2. 