# Retrieval-Augmented Generation: Find Evidence, Then Answer

**Beginner theory for Activities 3 and 4**

A language model can write a convincing answer without having the information needed to answer correctly. **Retrieval-Augmented Generation (RAG)** gives it relevant material to work with: search an external source, add the results to the prompt, and then generate an answer.

Our examples use **MediCore, a fictional hospital**. The people, policies, and changes in the activities belong to that teaching scenario.

| Reading path | Practical activity |
| :--- | :--- |
| Sections 1–5: finding relevant information | [Activity 3 short](activity3.md) or [Activity 3 long](activity3-l.md) |
| Sections 6–10: generating and checking an answer | [Activity 4 short](activity4.md) or [Activity 4 long](activity4-l.md) |
| Optional detail at the end | Extra background for the longer activities |

You do not need the optional mathematics to understand the core workflow.

## 1. Start with a question the model cannot reliably answer alone

Suppose a user asks:

> Who leads the neurology department at MediCore Hospital?

The model has general language abilities, but we should not assume it has reliable knowledge of this fictional organization. Without evidence, it might express uncertainty, guess a name, or produce a general response.

Our dataset contains this statement:

> Dr. Elena Varga leads the neurology department at MediCore Hospital.

A RAG system tries to find that statement and give it to the model before asking for an answer.

| Stage | What happens in this example? |
| :--- | :--- |
| **Retrieval** | Search the knowledge source for passages relevant to the question |
| **Augmentation** | Put the retrieved passages into the prompt alongside the question |
| **Generation** | Ask the language model to produce an answer using that context |

An intended answer is "Dr. Elena Varga." We still need to check whether the right evidence was retrieved and whether the generated answer follows it.

Activity 3 builds retrieval. Activity 4 connects it to augmentation and generation.

## 2. Separate preparing the knowledge from answering a question

A searchable knowledge source is prepared before it can be used:

```text
PREPARE THE KNOWLEDGE
Source material → extract text → create passages → embed → store and index

ANSWER A QUESTION
Question → embed and search → retrieve passages
                                      ↓
               Instructions + passages + original question
                                      ↓
                           Generate an answer
```

**Embedding** means converting text into a numerical representation. We will examine that representation in the next section. **Indexing** organizes the stored vectors to make searching practical.

The source does not need to be embedded again for every question. New or changed passages need new embeddings; each new question also needs an embedding.

There are different jobs in the lab:

| Component | Input → output |
| :--- | :--- |
| MiniLM embedding model | Text → a vector of numbers |
| Chroma vector database | Query vector → matching stored records |
| Our Python code | Question and retrieved passages → a model prompt |
| Qwen Instruct language model | Prompt → generated answer |

Chroma can call MiniLM automatically when we supply text. Qwen does not search Chroma by itself: our code performs the retrieval.

RAG can also use keyword search, database queries, or other retrieval methods. These activities use vector search to introduce semantic retrieval.

## 3. Embeddings: representing text with numbers

An **embedding** is a list of numbers, also called a **vector**, produced by a model. A text embedding model learns patterns that can make related texts have similar representations.

Compare:

- "The physician examined the patient."
- "A doctor checked the sick individual."
- "The sports car drove down the highway."

We expect the first two to be more similar to one another than to the third. Their wording differs, but they describe a similar event. That expectation is something to test, not a promised score.

### A two-dimensional picture

Imagine that each input has only two coordinates:

![Invented 2D vectors showing similar directions for doctor and physician, and for car and automobile.](img/activity3-embedding-map.svg)

These coordinates illustrate the same arrangement:

| Word | First coordinate | Second coordinate |
| :--- | ---: | ---: |
| doctor | 0.940 | 0.342 |
| physician | 0.891 | 0.454 |
| car | −0.500 | 0.866 |
| automobile | −0.602 | 0.799 |

**We invented these coordinates for teaching.** They are rounded points on a unit circle, not model outputs or a projection of actual embeddings. The axes do not have assigned meanings such as "medical knowledge" or "speed."

The vectors start at the same origin. Doctor and physician point in similar directions; car and automobile form another pair.

The model used in the lab, `all-MiniLM-L6-v2`, produces **384 coordinates** for a sentence or short paragraph. Two coordinates can be drawn on a page; 384 cannot be shown in full that way. The word map is an analogy for the geometry, while the lab searches sentence embeddings. [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

### Similarity and distance

**Cosine similarity** compares vector directions. Making a vector longer without changing its direction does not change this similarity.

Our collections use **cosine distance**:

> cosine distance = 1 − cosine similarity

| Illustrative similarity | Corresponding distance | Interpretation |
| ---: | ---: | :--- |
| 0.90 | 0.10 | More aligned |
| 0.60 | 0.40 | Less aligned than the first pair |
| 0.20 | 0.80 | Less aligned than either pair above |

These numbers are illustrations, not measured lab results. **Higher similarity and lower distance indicate a closer vector match. Neither is a probability that an answer is correct.**

Embedding similarity is imperfect. Opposite claims such as "Reception is open" and "Reception is closed" can share vocabulary and topic. Names, dates, and numbers may differ in ways that matter greatly to a question but do not produce a large distance.

### How this differs from keyword matching

A literal search for `doctor` does not match the word `physician`. Embeddings may connect those expressions through learned patterns.

Keyword search remains useful for exact names, codes, or identifiers. More advanced keyword systems can also handle variants or synonyms. Neither approach is always best.

**Pause:** Would a passage mentioning neurology necessarily tell you who leads it? Similar topic and sufficient evidence are different things.

## 4. Retrieval: return candidates, then read them

In the labs, a Chroma **collection** stores related records. A record contains:

| Item | Purpose |
| :--- | :--- |
| Document | The passage text we want to retrieve |
| Embedding | The numbers used to compare it with a question |
| ID | A unique label used to identify or update the record |
| Metadata | Extra information, such as its source or original question |

The `completion` field in `MediCore.json` supplies the searchable text. Its `prompt` field is retained as metadata. We embed the answer text, not that metadata.

Documents and questions use the same embedding model in these labs. Merely having the same number of coordinates would not make vectors from unrelated models comparable.

### What top-k means

**Top-k retrieval** requests the `k` closest matches. With `k=2`, we ask for two candidate passages. Fewer may be available if the collection or filtered set is small.

Both activities explicitly select cosine distance, so lower returned scores mean closer matches under that metric. Chroma supports other distance choices; scores from different metrics should not be interpreted interchangeably. [Chroma configuration](https://docs.trychroma.com/docs/collections/configure)

For a question about a department leader, the results might include:

| Passage | Does it answer who leads the department? |
| :--- | :--- |
| "The neurology department treats brain and nervous system conditions." | No: related topic, but no leader is named |
| "Dr. Elena Varga leads the neurology department at MediCore Hospital." | Yes: the required fact is present |

Increasing `k` can help retrieve missing evidence, but can also add irrelevant, repetitive, or conflicting text.

### What if the answer is missing?

For a question about MediCore's chief veterinary surgeon, the nearest records might describe other surgeons. Their presence does not establish that the requested role exists.

If no returned passage answers the question, either the knowledge source lacks the fact or retrieval missed it. Inspecting a few results does not establish absence from the whole source.

### Metadata filtering

Suppose reception hours differ between the north and south sites. A filter such as `site = "north"` restricts eligible records to that site; similarity search then finds matches within that restriction.

The filter guarantees the requested metadata condition, not that every result answers the question. The longer Activity 3 demonstrates this with a separate teaching dataset whose records have site labels.

## 5. Passages and chunks: preserve enough context

MediCore's short answers can each serve as one searchable passage. A long handbook usually needs to be divided into smaller pieces called **chunks**.

Consider:

> Visitors collect a badge at reception. The badge must be returned before leaving.

If retrieval returns only the second sentence, it misses where visitors collect the badge. A useful chunk should preserve enough context for the intended question.

| Choice | Possible problem |
| :--- | :--- |
| Very small chunks | Missing explanations, headings, or related facts |
| Very large chunks | Several topics in one vector and extra text in the prompt |
| Overlapping chunks | Better continuity, but repeated text and redundant results |

**Overlap** repeats some text across adjacent chunks. It can help when a fact spans a boundary, but no single chunk size or overlap works best for every source.

Input limits also matter. MiniLM truncates text beyond 256 word pieces by default. A word piece is a tokenizer unit, not necessarily a whole word or character; a long passage may therefore be only partly represented. [MiniLM input limits](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

For files, extraction comes first. PDF and DOCX content must become usable text in this pipeline. A scanned PDF may need **OCR**, which recognizes text in images. Tables and layout can also be damaged during extraction. Read the extracted text before indexing it. [pypdf extraction limitations](https://pypdf.readthedocs.io/en/stable/user/extract-text.html)

**Activity 3 connection:** The short lab retrieves prepared records. The long lab adds filtering, chunking, and file extraction. At the end of either lab, the output is still retrieved text; no language model has generated an answer.

## 6. Augmentation: put the evidence into the prompt

Once passages have been retrieved, our code assembles the model's input.

A simplified prompt for the earlier question is:

```text
Instructions:
Answer using the supplied context.
If the context does not support an answer, say information is unavailable.

Context:
[S1] Dr. Elena Varga leads the neurology department at MediCore Hospital.

Question:
Who leads the neurology department at MediCore Hospital?
```

This is **augmentation**: adding retrieved information to the question and instructions. Qwen receives the passage text, not the retrieval vectors.

The roles have different purposes:

- **System message:** the intended behavior, such as answering from evidence.
- **User message:** the context and question in this lab.
- **Assistant message:** the generated response.

The tokenizer's **chat template** formats those messages with the role markers the model expects. For Qwen, markers include `<|im_start|>` and `<|im_end|>`. The lab uses `apply_chat_template()` instead of assembling these markers by hand. The formatted text is then converted to token IDs. [Hugging Face chat templates](https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating)

**Nothing in this step trains the model.** Adding a name to the prompt makes it available for this response; it does not update model parameters. Our lab calls do not automatically include previous answers as conversation history.

## 7. Generation: produce an answer and check its claims

The language model generates a continuation of its input, one token at a time. A **token** can represent a word, part of a word, punctuation, or a control marker.

Activity 4 uses the already pretrained and instruction-tuned `Qwen2.5-1.5B-Instruct` checkpoint. It has language abilities before the lab starts; it has not been trained on MediCore by our notebook. [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

### Grounding and correctness

An answer is **grounded** when its factual claims are supported by the supplied evidence.

Suppose the only evidence names the neurology leader:

| Generated answer | Assessment |
| :--- | :--- |
| "Dr. Elena Varga leads the department." | Supported by the passage |
| "Dr. Elena Varga leads it and has worked there for twenty years." | The added employment history is unsupported |
| "Dr. Arto Virtanen leads the department." | Contradicts the supplied passage |

**Correctness** asks whether the answer agrees with the intended facts. In these labs we use the dataset as our reference. In a real system, a source can itself be outdated or wrong, so an answer can faithfully repeat it and still be incorrect about the world.

### Missing, ambiguous, or conflicting evidence

Grounding instructions encourage the model to avoid inventing facts, but cannot guarantee compliance.

When evidence is missing, acknowledging insufficient information is appropriate. When a question is vague, asking for clarification may be better. When sources name different leaders without a date or priority rule, choosing one confidently is not justified.

The model may fail at any of these behaviors. Check the actual output.

### Generation settings do not guarantee truth

The lab sets `do_sample=False` and `num_beams=1` for **greedy decoding**, which selects the most likely next token. This removes sampling randomness, making comparisons easier; errors can still come from retrieval, instructions, or model behavior.

`max_new_tokens` limits the answer length. The **context window** limits how much input and generated text the model can handle together. Activity 4 uses a 4,096-token classroom budget, not the model's maximum supported length. [Generation settings](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation)

## 8. Updating a fact: follow it through the system

The labs change the fictional CEO from Juhani Aho to Milla Kallio:

```text
Edit the stored sentence
        ↓
Recompute its embedding
        ↓
Retrieve again
        ↓
Assemble a fresh prompt
        ↓
Generate and check a new answer
```

The embedding model and language model parameters stay unchanged. The text and its stored vector change.

Check each stage:

1. Does the database contain the edited sentence?
2. Did the new search return that record?
3. Is the edited text in the new prompt?
4. Does the generated answer agree with it?

A database edit does not rewrite an earlier answer or guarantee a particular retrieval rank.

In Activity 3, the edit remains in its temporary collection until reset. Activity 4 rebuilds a separate collection from the downloaded file and restores the original CEO at the end of its update experiment. Neither lab edits the downloaded dataset file. This makes notebook state explicit when comparing results.

## 9. Evaluate retrieval and generation separately

A useful answer needs both suitable evidence and a model that uses it well.

| What to inspect | Question to ask |
| :--- | :--- |
| Retrieval | Did we retrieve a passage containing the needed fact? |
| Prompt construction | Was that passage actually included in the model input? |
| Generation | Does the answer use the fact correctly without unsupported additions? |

Test more than one easy question: include known facts, paraphrases, different vocabulary, missing information, and ambiguity.

For known-fact questions, the longer labs check **hit rate at k**: the fraction whose designated supporting record appears among the retrieved results. Finding the expected record for two of three questions gives a hit rate of 2/3. This is a retrieval measure, not answer accuracy.

An exact check for "Information not available." measures whether that phrase appeared. It does not determine whether the response was appropriate; a paraphrase may also be acceptable.

### What source references and JSON prove

A reference such as `[S1]` lets a reader inspect a passage. It does not prove that the passage supports the claim. In Activity 4, these labels are local to each response, while the stored IDs identify database records.

Similarly, Python's `json.dumps()` creates valid JSON around the answer. It does not validate the answer's facts. The long lab uses `retrieved_sources` to describe the passages supplied, rather than claiming every source was used correctly.

**Activity 4 connection:** The short lab compares answers before and after adding evidence and grounding instructions. The long lab also holds instructions fixed while varying only the supplied context, helping separate the effects of those changes.

## 10. How RAG relates to fine-tuning

Models can use both information learned during training and information supplied in the current prompt.

| Approach | What changes? | Common reason to use it |
| :--- | :--- | :--- |
| Prompting | Instructions and examples in the input | Guide the response for a task |
| RAG | Evidence retrieved and placed in the input | Supply external, changing, or traceable facts |
| LoRA fine-tuning | Trainable adapter parameters | Adapt recurring behavior, style, or task performance |

In LoRA training, the original model weights are typically frozen while adapter parameters are learned. During later inference, no training occurs.

The distinction is practical rather than absolute: prompting can influence style, and fine-tuning can encode facts. A system may combine fine-tuning with RAG.

For a personnel directory that changes monthly, updating external records can be more convenient than repeatedly training on new facts. It still requires checking the source, retrieval, and answer.

## Check your understanding

Discuss these before looking at the suggested answers:

1. Why might a search for "helicopter landing" retrieve a passage about a helipad?
2. Does cosine distance 0.2 mean an answer is 80% likely to be correct?
3. A passage describes neurology but names no leader. Does it answer "Who leads neurology?"
4. What does augmentation give Qwen: stored embedding vectors or passage text?
5. If the correct fact is in the prompt but the answer is wrong, what would you inspect next?
6. Why can a cited, grounded answer still be wrong about the real world?

<details>
<summary>Suggested answers</summary>

1. An embedding model can give related expressions similar representations despite different wording.
2. No. It is a geometric score under the selected metric, not a probability of correctness.
3. No. Being related to the topic is insufficient.
4. Passage text, together with instructions and the question.
5. The instructions, conflicting or distracting context, and the model's use of the evidence.
6. A citation can point to an irrelevant passage, and even genuinely supporting evidence can be outdated or incorrect.

</details>

## Optional detail for the longer activities

### The cosine calculation

For two nonzero vectors, cosine similarity is:

$$
\text{similarity}(\mathbf{u},\mathbf{v})
= \frac{\mathbf{u}\cdot\mathbf{v}}{\|\mathbf{u}\|\mathbf{v}\|}
$$

The dot product multiplies corresponding coordinates and adds the results. Dividing by the vector lengths makes the score depend on direction.

The range is −1 to 1. Zero means perpendicular vectors in that representation, not proof of no semantic relationship. A negative value means directions more than 90 degrees apart, not necessarily opposite word meanings.

For the unrounded teaching vectors, doctor and physician differ by 7 degrees, giving similarity approximately 0.993 and distance approximately 0.007. Those values follow from our invented geometry, not learned language knowledge.

Other metrics include dot product and squared Euclidean distance. Normalizing vectors means making each length equal to one; on normalized vectors, dot product equals cosine similarity. The labs choose cosine explicitly so there is one scoring convention to interpret.

### Why use a search index?

An exact search can compare the question vector with every stored vector. Larger collections often use **approximate nearest-neighbor search**, which can reduce work while sometimes missing a mathematically closer match.

Chroma's local HNSW index organizes connections between nearby vectors. Its search settings affect the trade-off between speed and retrieval quality. You do not need to implement that index for these activities. [Chroma index documentation](https://docs.trychroma.com/docs/collections/configure)

### Two names for where information is stored

**Parametric knowledge** refers to information and abilities encoded in learned model parameters.

**Non-parametric memory**, in RAG discussions, refers to an external information store that can be retrieved independently of those parameters. These terms name the distinction we have already used between learned model capabilities and stored records.

### Extensions beyond these labs

**Hybrid retrieval** combines keyword and vector search. A **reranker** scores retrieved candidates again to improve their ordering. Neither replaces checking whether the selected passages support an answer.

Retrieved text may also contain instructions directed at the model. Treat it as evidence to inspect; merely adding a system instruction does not guarantee that the model will ignore such content.

For this introduction, the essential habit is to inspect the question, retrieved passages, final prompt, and generated claims as separate parts of one process.

## Further reading

- [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2): the embedding model used in Activity 3.
- [Chroma documentation](https://docs.trychroma.com/docs/collections/configure): distance metrics and collection indexes.
- [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct): the generator used in Activity 4.
- [Hugging Face chat templates](https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating): formatting conversations for a model.
