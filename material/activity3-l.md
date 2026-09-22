# Activity 3: Find Relevant Information with Semantic Search

**RAG Part 1 — longer version**

<!-- How can a search for *"Where do helicopters land?"* find a sentence about a *"rooftop helipad"*? -->

In this activity, you will build a search system for **MediCore, a fictional hospital**. It returns stored text that may help answer a question. In Activity 4, a language model will use retrieved text to generate an answer.

By the end, you should be able to:

- Explain what an embedding represents.
- Retrieve and inspect the top matching passages.
- Explain why a close match may not contain the answer.
- Update a stored fact without training a model.

This is a standalone lab: Steps 0–6 use the same core cells as [the short version](activity3.md). Continue with six extensions on vector geometry, keyword search, filtering, chunking, file extraction, and evaluation.

If you already ran the short version in the same notebook, continue from Cell 8. Otherwise start from Cell 0. Extension 5 is optional; Extension 6 can run without it.

## Before you start

Open a new Google Colab notebook. Copy each Python block into a separate code cell and run them in order. A **CPU runtime is enough**; no GPU or API key is needed. Internet access is needed to install packages and download the model and dataset.

You only need basic Python lists, dictionaries, and loops. The first model download may take a few minutes.

| Component | Its job |
| :--- | :--- |
| MiniLM embedding model | Convert text into a list of numbers |
| Chroma vector database | Store records and search their vectors |
| Qwen language model, in Activity 4 | Generate an answer from the question and retrieved text |

There is **no answer generation or model training** in this activity.

## Step 0: Install the libraries

```python
# Cell 0 — Run before importing the libraries.
%pip install -q "opentelemetry-api>=1.39.0,<=1.42.1" "opentelemetry-sdk>=1.39.0,<=1.42.1" "chromadb>=1.0,<2" "sentence-transformers>=3,<6"
```

Chroma manages the searchable collection. Sentence Transformers loads the embedding model. The OpenTelemetry bounds retain the original lab's Colab dependency workaround; these are supporting libraries, not concepts you need to study here. Package compatibility depends on the current Colab environment, so these ranges are not a guarantee or a frozen environment.

If Colab requests a session restart after installation, restart it and continue from Cell 1. If installation reports a dependency conflict, inspect the message before continuing.

## Step 1: Understand an embedding

An **embedding** is a list of numbers produced by a trained model to represent an input. For text search, the model has learned patterns that often place related texts near one another.

Imagine representing each word with just two numbers:

![Illustrative 2D vectors: doctor and physician have similar directions, as do car and automobile.](img/activity3-embedding-map.svg)

**The coordinates in this picture are invented for teaching.** They are not MiniLM output. The axes have no assigned meanings such as "medical" or "mechanical."

The arrows start at the same origin. Doctor and physician point in similar directions; car and automobile form another pair. This helps us picture **cosine similarity**, which compares vector directions. Longer arrows alone would not mean greater cosine similarity.

Real MiniLM embeddings have **384 coordinates**, so we cannot draw their full geometry on a page. One coordinate does not usually have a simple label.

### Inspect real sentence embeddings

**Predict:** Which pair below should be more similar?

```python
# Cell 1 — Real embeddings.
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = SentenceTransformer(MODEL_NAME, device="cpu")

sentences = [
    "The physician examined the patient.",
    "A doctor checked the sick individual.",
    "The sports car drove down the highway.",
]
vectors = embed_model.encode(sentences)

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print("Array shape:", vectors.shape)
print("First five coordinates of sentence 1:", vectors[0][:5])
print("Medical pair:", round(cosine_similarity(vectors[0], vectors[1]), 3))
print("Medical/car pair:", round(cosine_similarity(vectors[0], vectors[2]), 3))
```

**Inspect:** The shape should be `(3, 384)`: three sentences, with 384 numbers each. Compare the two scores; do not look for a particular decimal value.

Cosine similarity ranges from −1 to 1. Larger values indicate more aligned vectors. It is **not a probability that two texts mean the same thing**, and zero is not proof of no semantic relationship.

**Explain:** Why can two sentences with different words have a relatively high similarity?

<details>
<summary>Discussion answer</summary>

The model learned patterns connecting related expressions during training. Similar wording is not required, although similar meaning is not guaranteed either. It can make mistakes with negation, names, numbers, or unfamiliar terms.

</details>

## Step 2: Inspect the source data

The dataset contains question-and-answer records. We will search the `completion` text and keep the original `prompt` as descriptive metadata.

```python
# Cell 2 — Download and read JSON Lines.
import json
from pathlib import Path
from urllib.request import urlretrieve

DATA_URL = (
    "https://raw.githubusercontent.com/AI-Learning-Repo/"
    "Data-Handling/refs/heads/week4/datasets/MediCore.json"
)
data_path = Path("MediCore.json")
if not data_path.exists():
    urlretrieve(DATA_URL, data_path)

with data_path.open(encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

print("Number of records:", len(records))
print(json.dumps(records[0], indent=2, ensure_ascii=False))
```

Despite its `.json` filename, this is **JSON Lines**: each nonempty line is a separate JSON object. That is why the code reads one line at a time.

For this small dataset, each answer is treated as one **document**, or searchable passage. We do not split it further. In other datasets, a short answer such as "Yes" would need its question or heading included to make sense.

## Step 3: Build the searchable collection

A collection groups documents, their embeddings, and identifying information.

| Stored item | Purpose |
| :--- | :--- |
| Document | The text we want to retrieve |
| Embedding | Numbers used for similarity search |
| ID | A unique label used to get or update a record |
| Metadata | Extra fields, such as the original question or source |

```python
# Cell 3 — Create a fresh in-memory collection and index the records.
import hashlib
import chromadb
from chromadb.utils import embedding_functions

# A question-based ID stays the same if the dataset rows are reordered.
def record_id(question):
    return "fact_" + hashlib.sha256(question.encode("utf-8")).hexdigest()

ids = [record_id(item["prompt"]) for item in records]
if len(ids) != len(set(ids)):
    raise ValueError("Repeated questions found; inspect the dataset before indexing.")

documents = [item["completion"] for item in records]
metadatas = [
    {"source": "MediCore.json", "source_prompt": item["prompt"]}
    for item in records
]

chroma_client = chromadb.EphemeralClient()
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=MODEL_NAME,
    device="cpu",
)

# Re-running this cell resets ONLY this activity's collection.
collection_name = "activity3_medicore_cosine"
existing_names = [c.name for c in chroma_client.list_collections()]
if collection_name in existing_names:
    chroma_client.delete_collection(collection_name)

collection = chroma_client.create_collection(
    name=collection_name,
    embedding_function=embedding_fn,
    configuration={"hnsw": {"space": "cosine"}},
)
collection.add(documents=documents, metadatas=metadatas, ids=ids)
print("Indexed records:", collection.count())
```

Chroma calls MiniLM to embed the documents. Later, it uses the **same embedding model** for questions, making the vectors comparable. Metadata is stored separately; the original question is not embedded in this example.

We explicitly choose **cosine distance**:

> cosine distance = 1 − cosine similarity

A smaller distance means a closer vector match. For example, a similarity of 0.8 corresponds to a distance of 0.2. Neither number means "80% correct."

The `hnsw` setting names Chroma's search index; you do not need its internal algorithm for this lesson. An index helps search efficiently without requiring every query to compare against every stored vector.

This collection is temporary. A new Colab runtime needs to run the setup again. Re-running Cell 3 restores the records from the downloaded dataset, including undoing the later CEO edit.

## Step 4: Retrieve and read the evidence

**Top-k retrieval** requests the `k` closest matches. Here, `k=2`.

```python
# Cell 4 — A helper that prints and returns the retrieved records.
def search(query, k=2):
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "distances", "metadatas"],
    )
    print("\nQUESTION:", query)
    for rank, (doc_id, text, distance) in enumerate(
        zip(results["ids"][0], results["documents"][0], results["distances"][0]),
        start=1,
    ):
        print(f"\nRank {rank} | cosine distance {distance:.3f}")
        print("ID:", doc_id)
        print(text)
    return results

results = search("Who leads the neurology department at MediCore Hospital?")
```

The API accepts several questions at once. We supplied a list containing one question, so `results["documents"][0]` means **the documents returned for the first question**. The first document inside that list is rank 1.

**Inspect:** Does a returned passage actually name the department leader? A description of what neurology treats is related to the topic, but may not answer who leads it.

Now change the wording and the number of results:

```python
# Cell 5 — Compare wording and k.
results = search("Who is in charge of brain and nervous system conditions?", k=2)
results = search("Where do helicopters land?", k=1)
results = search("Where do helicopters land?", k=3)
```

**Record:** For each question, write down the best supporting passage and its rank. If none answers the question, say so. Results and distances may vary with the query, data, and library versions.

**Explain:** Did increasing `k` add useful evidence, irrelevant text, or both?

## Step 5: Search for missing information

A search system can return close matches even when none contains an answer.

```python
# Cell 6 — Read the results rather than assuming they answer the question.
results = search("What is the name of MediCore Hospital's chief veterinary surgeon?")
```

**Inspect:** Do any passages explicitly name a person in that role? A passage about another surgeon is insufficient evidence.

If the retrieved passages do not contain the answer, that alone does not establish that the whole database lacks it. There are two possibilities: the fact is absent, or retrieval missed it.

Chroma returns candidate passages; it does not decide whether a question has been answered. A distance score alone does not settle that decision.

## Step 6: Change a fact without training

Suppose the fictional hospital appoints a new CEO. We can edit its stored record and create a new embedding without changing the embedding model's parameters.

```python
# Cell 7 — Locate the record by its question, then compare before and after.
ceo_question = "Who is in charge of brain and nervous system conditions?"
matches = [item for item in records if item["prompt"] == ceo_question]
if len(matches) != 1:
    raise ValueError("Expected one CEO record. Inspect the dataset before updating.")

ceo_id = record_id(ceo_question)
original_text = matches[0]["completion"]

# Restore the original first, so re-running this cell repeats the experiment.
collection.update(ids=[ceo_id], documents=[original_text])
print("BEFORE UPDATE")
before = search(ceo_question, k=2)

collection.update(
    ids=[ceo_id],
    documents=["The CEO of MediCore Hospital is Milla Kallio."],
)
print("\nSTORED RECORD AFTER UPDATE")
print(collection.get(ids=[ceo_id], include=["documents"])["documents"][0])

print("\nSEARCH AFTER UPDATE")
after = search(ceo_question, k=2)
print("\nUpdated record retrieved:", ceo_id in after["ids"][0])
```

Two separate checks matter: **was the record changed, and was it retrieved?** Updating storage does not guarantee a particular rank.

Only the temporary Chroma record changed. The downloaded file and model parameters did not. In Activity 4, retrieved text will be added to a language model's prompt; it still needs to use that evidence correctly.

## Check your understanding

Before moving on, explain in your own words:

1. What does MiniLM produce, and what does Chroma return?
2. Why can a query match a passage that uses different words?
3. Does a low distance guarantee that a passage answers the question?
4. What changes when we update a document?
5. Why have we not yet built a complete RAG system?

<details>
<summary>Suggested answers</summary>

1. MiniLM produces numerical vectors. Chroma returns stored documents and associated information.
2. The model can represent related expressions with similar vector directions.
3. No. Read the passage and check whether it supports the required answer.
4. The stored text and its embedding change. The model parameters do not.
5. We have built retrieval. Activity 4 adds the retrieved evidence to a prompt and generates an answer.

</details>

## Extension 1: Calculate similarity in two dimensions

The illustration in Step 1 uses made-up vectors. Here we construct similar vectors with two coordinates each. You can inspect all their numbers.

**Predict:** Which pair should have the smallest cosine distance?

```python
# Cell 8 — Toy coordinates, not outputs of the embedding model.
angles = {"doctor": 20, "physician": 27, "car": 120, "automobile": 127}
toy_vectors = {}
for word, degrees in angles.items():
    radians = np.deg2rad(degrees)
    toy_vectors[word] = np.array([np.cos(radians), np.sin(radians)])
    print(word, np.round(toy_vectors[word], 3))

for left, right in [
    ("doctor", "physician"),
    ("car", "automobile"),
    ("doctor", "car"),
]:
    similarity = cosine_similarity(toy_vectors[left], toy_vectors[right])
    print(f"{left} / {right}: similarity={similarity:.3f}, distance={1-similarity:.3f}")

print(
    "After making the doctor vector three times longer:",
    round(cosine_similarity(3 * toy_vectors["doctor"], toy_vectors["physician"]), 3),
)
```

**Explain:** Why does multiplying one vector by three leave its cosine similarity unchanged?

<details>
<summary>Discussion answer</summary>

Multiplying by a positive number changes the vector's length but not its direction. Cosine similarity compares directions. The toy numbers were selected by us; the real embedding model learns its representation during training.

A negative score means directions are more than 90 degrees apart. It does not mean the words are linguistic opposites. For example, "hot" and "cold" can be related in text even though they are antonyms.

</details>

Now replace the real sentences in Cell 1 with your own pair of paraphrases and an unrelated sentence. Predict their relative similarities, then compare the results.

## Extension 2: Compare literal matching with semantic search

A simple keyword search checks whether a word or phrase occurs in the text. It can be useful for exact names or identifiers. Semantic search can help when the wording differs.

This small experiment searches the **same three passages** in two ways.

```python
# Cell 9 — Literal substring matching versus embedding similarity.
demo_passages = [
    "A physician examined the patient.",
    "A mechanic repaired the car.",
    "A librarian sorted the books.",
]
demo_query = "doctor"
demo_vectors = embed_model.encode(demo_passages)
query_vector = embed_model.encode(demo_query)

literal_matches = [
    text for text in demo_passages if demo_query.lower() in text.lower()
]
print("Literal matches:", literal_matches)

scores = [cosine_similarity(query_vector, v) for v in demo_vectors]
for index in np.argsort(scores)[::-1]:
    print(f"Similarity {scores[index]:.3f} | {demo_passages[index]}")
```

**Inspect:** Does the most similar passage fit the query? Why did the literal search return a different result?

**Try:** Change `demo_query` to `"mechanic"` and then to `"vehicle repair"`. Which search works well in each case?

This substring search is deliberately simple. It does not represent every keyword search system: more advanced systems can use stemming, synonyms, or ranked matching. Semantic search is also imperfect; exact identifiers and small numerical differences may need exact matching.

## Extension 3: Filter using metadata

Suppose two hospital sites have different reception hours. A similar passage from the wrong site would be a poor answer.

Use a separate, explicitly invented dataset to see how **filtering and similarity search work together**.

```python
# Cell 10 — A small, separate collection with site metadata.
site_name = "activity3_sites_cosine"
if site_name in [c.name for c in chroma_client.list_collections()]:
    chroma_client.delete_collection(site_name)

site_collection = chroma_client.create_collection(
    name=site_name,
    embedding_function=embedding_fn,
    configuration={"hnsw": {"space": "cosine"}},
)
site_collection.add(
    ids=["north_reception", "south_reception", "north_parking", "south_parking"],
    documents=[
        "Reception is open from 08:00 to 16:00.",
        "Reception is open from 10:00 to 18:00.",
        "Visitor parking is next to the north entrance.",
        "Visitor parking is behind the south building.",
    ],
    metadatas=[
        {"site": "north", "source": "teaching_example"},
        {"site": "south", "source": "teaching_example"},
        {"site": "north", "source": "teaching_example"},
        {"site": "south", "source": "teaching_example"},
    ],
)

for site in [None, "north", "south"]:
    options = {} if site is None else {"where": {"site": site}}
    result = site_collection.query(
        query_texts=["When is reception open?"],
        n_results=2,
        include=["documents", "metadatas", "distances"],
        **options,
    )
    print("\nSITE FILTER:", site or "none")
    for text, metadata in zip(result["documents"][0], result["metadatas"][0]):
        print(metadata["site"], "|", text)
```

**Inspect:** The filtered results should all have the requested `site`. They can still include a parking passage because we requested two results.

**Explain:** What does the filter guarantee, and what does it not guarantee?

<details>
<summary>Discussion answer</summary>

The filter restricts eligible records to the specified metadata value. It does not guarantee that each result answers the question. Here, the application supplies the site explicitly; the embedding model does not infer the filter.

</details>

Metadata is useful only if it is accurate and available. These labels belong to this teaching example; they were not present in the original MediCore dataset.

## Extension 4: See why chunk boundaries matter

MediCore answers are already short. Longer source documents must often be divided into **chunks** so that search can retrieve a useful passage.

Consider this invented policy:

> Visitors collect a badge at reception. The badge must be returned before leaving.

If one chunk contains only the second sentence, a reader may not know where the badge came from. If one chunk contains an entire unrelated handbook, the useful detail can be hard to retrieve.

The next experiment compares two ways to split exactly the same text. These small character counts make boundaries easy to see; they are not recommended production settings.

```python
# Cell 11 — A simple character-based chunker.
def chunk_text(text, chunk_size, overlap):
    if not 0 <= overlap < chunk_size:
        raise ValueError("Use 0 <= overlap < chunk_size.")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks

policy_text = (
    "Visitors collect a badge at reception. "
    "The badge must be returned before leaving. "
    "The library lends books for fourteen days. "
    "Bicycle parking is beside the east entrance."
)
chunk_query = "Where do visitors get their badge, and when must they return it?"

for size, overlap in [(50, 0), (110, 30)]:
    chunks = chunk_text(policy_text, size, overlap)
    chunk_vectors = embed_model.encode(chunks)
    query_vector = embed_model.encode(chunk_query)
    scores = [cosine_similarity(query_vector, v) for v in chunk_vectors]
    best_index = int(np.argmax(scores))

    print(f"\nCHUNK SIZE={size} characters, OVERLAP={overlap} characters")
    for index, chunk in enumerate(chunks):
        print(index, repr(chunk))
    print("Closest chunk:", chunks[best_index])
    print("Cosine similarity:", round(scores[best_index], 3))
```

**Inspect:** For each setting, does the closest chunk contain both parts of the answer? Are any words cut in half?

**Explain:** Why might overlap help? What extra text does it repeat?

The trade-off is that small chunks can lose context, while large chunks can combine unrelated topics. Overlap repeats text to help preserve information near boundaries, but can also produce redundant results.

A character is not a token. MiniLM truncates inputs beyond its supported token limit (256 word pieces by default), so a long input can be only partly represented. In real applications, use the model's tokenizer to check length and prefer meaningful boundaries such as sentences or paragraphs. See the [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

## Extension 5: Retrieve text from PDF and DOCX files

This optional extension creates tiny sample files **inside Colab**. You do not need to upload anything. Their contents are fictional administrative examples.

The flow is:

```text
PDF / DOCX → extract text → inspect text → create chunks → embed → search
```

### Create the example files

```python
# Cell 12 — Libraries needed only for this extension.
%pip install -q pypdf python-docx reportlab
```

```python
# Cell 13 — Create two small files.
import docx
from pypdf import PdfReader
from reportlab.pdfgen import canvas

word_path = "activity3_visitor_guide.docx"
word_file = docx.Document()
word_file.add_paragraph(
    "Visitors collect a badge at the main reception desk. "
    "They must return the badge before leaving."
)
word_file.add_paragraph(
    "Visitor parking is beside the east entrance. "
    "Bicycle racks are beside the library."
)
word_file.save(word_path)

pdf_path = "activity3_library_guide.pdf"
pdf_file = canvas.Canvas(pdf_path)
pdf_file.drawString(72, 750, "The hospital library is on the second floor.")
pdf_file.drawString(72, 730, "Library books can be borrowed for fourteen days.")
pdf_file.save()

print("Created:", word_path, "and", pdf_path)
```

### Inspect the extracted text

```python
# Cell 14 — Keep source information with each extracted passage.
extracted = []

for paragraph_number, paragraph in enumerate(docx.Document(word_path).paragraphs, start=1):
    text = paragraph.text.strip()
    if text:
        extracted.append({
            "text": text,
            "source": word_path,
            "location": f"paragraph {paragraph_number}",
        })

for page_number, page in enumerate(PdfReader(pdf_path).pages, start=1):
    text = (page.extract_text() or "").strip()
    if text:
        extracted.append({
            "text": text,
            "source": pdf_path,
            "location": f"page {page_number}",
        })

for item in extracted:
    print("\n", item["source"], "|", item["location"])
    print(item["text"])
```

**Inspect before indexing:** Is all the intended text present and readable?

This DOCX reader handles paragraphs, not tables or every possible document element. This PDF reader handles extractable text; scanned images need an additional OCR step. The simple examples avoid complicated layouts.

### Index and retrieve

```python
# Cell 15 — Reuse the chunker and embedding function.
file_documents, file_metadatas, file_ids = [], [], []
for passage_number, item in enumerate(extracted):
    for chunk_number, text in enumerate(chunk_text(item["text"], 200, 30)):
        file_documents.append(text)
        file_metadatas.append({
            "source": item["source"],
            "location": item["location"],
        })
        file_ids.append(f"passage_{passage_number}_chunk_{chunk_number}")

file_collection_name = "activity3_files_cosine"
if file_collection_name in [c.name for c in chroma_client.list_collections()]:
    chroma_client.delete_collection(file_collection_name)

file_collection = chroma_client.create_collection(
    name=file_collection_name,
    embedding_function=embedding_fn,
    configuration={"hnsw": {"space": "cosine"}},
)
file_collection.add(
    ids=file_ids,
    documents=file_documents,
    metadatas=file_metadatas,
)

for query in ["Where do visitors collect a badge?", "How long can I borrow a library book?"]:
    result = file_collection.query(
        query_texts=[query],
        n_results=2,
        include=["documents", "metadatas", "distances"],
    )
    print("\nQUESTION:", query)
    for text, metadata, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        print(f"\n{metadata['source']} | {metadata['location']} | distance={distance:.3f}")
        print(text)
```

These tiny passages may fit in a single chunk with the chosen settings. Longer passages would be split.

**Explain:** Why preserve the filename and page or paragraph? They let a reader locate the original evidence. A filename alone does not prove that the passage answers the question.

## Extension 6: Evaluate retrieval with a small question set

One successful example does not show that search works reliably. Choose questions and expected evidence **before** inspecting their retrieved results.

Here we use three known records. Each test names a source question so we can locate the expected record without relying on row numbers.

```python
# Cell 16 — Measure whether a designated supporting record appears in top-k.
test_cases = [
    {
        "query": "Who runs MediCore Hospital as CEO?",
        "source_question": "Who is the CEO of MediCore Hospital?",
    },
    {
        "query": "Who heads neurology at MediCore Hospital?",
        "source_question": "Who leads the neurology department at MediCore Hospital?",
    },
    {
        "query": "Where can a helicopter land at MediCore?",
        "source_question": "Does MediCore Hospital have a helicopter landing pad?",
    },
]

for case in test_cases:
    matches = [r for r in records if r["prompt"] == case["source_question"]]
    if len(matches) != 1:
        raise ValueError(f"Check the source record for: {case['source_question']}")
    case["expected_id"] = record_id(case["source_question"])

for k in [1, 3]:
    hits = 0
    print(f"\nTOP-{k}")
    for case in test_cases:
        result = collection.query(query_texts=[case["query"]], n_results=k)
        hit = case["expected_id"] in result["ids"][0]
        hits += int(hit)
        print("Expected record found:", hit, "|", case["query"])
        for passage in result["documents"][0]:
            print("  ", passage)
    print(f"Hit rate: {hits}/{len(test_cases)} = {hits / len(test_cases):.0%}")
```

This **hit rate** counts the proportion of questions whose designated supporting record was retrieved. It is a small classroom check, not a general performance estimate. Another passage might also contain a valid answer; read the output and investigate misses.

The CEO record may now name Milla Kallio after Cell 7. Its ID is unchanged because the ID is based on the question, not the answer.

Add a manual evidence check:

| Question type | Did a passage answer the question? | Supporting excerpt or reason it was insufficient |
| :--- | :--- | :--- |
| Known fact | | |
| Same fact, different wording | | |
| Chief veterinary surgeon | | |
| "Who is responsible for the department?" | | |

The last question is ambiguous. A relevant-sounding result does not resolve which department the user meant.

**Explain:** Why should the missing-information case not be counted as a normal retrieval hit? There is no designated supporting record to retrieve; we instead check whether the returned passages lack the required evidence. We are not testing an LLM's refusal behavior yet.

## What to take to Activity 4

You should now be able to show:

- A query, its retrieved passages, and the distance metric used.
- One example where different wording still retrieves useful evidence.
- One example where a nearest match is insufficient.
- The effect of a metadata filter.
- How chunk boundaries can change available evidence.
- A stored update and whether search retrieves it.

The next activity adds **augmentation** (putting retrieved text into the prompt) and **generation** (producing an answer). Retrieval quality remains important: the generator needs suitable evidence to work with.

## References

- [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- [Chroma collection configuration](https://docs.trychroma.com/docs/collections/configure).
- [Chroma updates](https://docs.trychroma.com/docs/collections/update-data).
- [Chroma metadata filtering](https://docs.trychroma.com/docs/querying-collections/metadata-filtering).
