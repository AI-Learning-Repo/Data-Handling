# Activity: Training Data Engineering for Small Language Models

> **How to use this lab:** Complete the visible core workflow in order: Dataset Specification -> real-data profiling and cleaning -> privacy and audit -> coverage analysis -> grounded synthetic generation -> validation -> lineage and documentation. Sections labelled **Optional Advanced Extension** preserve additional explanations and implementation examples for groups that finish early or need them for their projects.


# Part 0: Laboratory Overview, Foundational Principles, and Dataset Specification

---

## 0.1 Context and Foundational Principles

### The Role of Data in Model Specialization
Open-source foundation model architectures, pre-trained weights, and training algorithms are commoditized and widely accessible. Consequently, **for a fixed task and base model, dataset quality, relevance, and task alignment are often among the strongest determinants of fine-tuning outcomes.** Curated, domain-specific data constitutes the primary defensible asset (the "data moat") that differentiates an accurate, specialized system from a generic baseline model.

### Sensitivity of Smaller Parameter Architectures
This laboratory targets lightweight instruction models (such as the **Qwen 1.5B/1B** family) as the student architecture. Smaller parameter models exhibit distinct operational characteristics during supervised fine-tuning compared to large frontier models (e.g., 70B+ parameters):
* **Supervision Sensitivity:** Larger models possess significant parameter redundancy that can absorb minor dataset inconsistencies. Smaller models have less capacity to compensate for poorly specified, contradictory, or noisy supervision, making dataset consistency and formatting integrity particularly critical.
* **Memorization vs. Generalization:** Overfitting to duplicated strings or narrow syntax happens rapidly on smaller parameter models, leading to brittle inference outside the exact prompt distribution.
* **Structural Consistency:** Small models benefit substantially from uniform structural formatting, clear prompt boundaries, and explicit demonstrations of target behavior.

### Data Volume vs. Behavior Space Coverage
Supervised fine-tuning efficiency depends on **behavioral coverage** rather than raw record counts. Consider the following ranges as **illustrative starting points, not performance guarantees**:

* **$\approx 100\text{--}500$ verified examples:** Often sufficient to adapt conversational tone, surface style, output schema (e.g., strict JSON formatting), or basic classification behavior.
* **$\approx 1,000\text{--}5,000$ verified examples:** A standard operational baseline for deep domain adaptation, specialized technical question-answering, and multi-step instruction-following in narrow verticals.
* **$> 10,000$ examples:** Frequently subject to diminishing returns in specialized domains unless addressing broad vocabularies, multiple distinct tasks, or complex multi-turn workflows.

> **Key Rule:** Do not ask *"How many examples do I need?"* first. Ask: **"How much of the desired operational behavior space do my examples cover?"**

---

## 0.2 The Data Engineering Decision Framework

Data preparation is not a mechanical checklist of deletion scripts; it is a **systematic decision-making process**. Blindly applying aggressive cleaners can strip valuable signal (such as code blocks, markdown tables, or domain syntax). 

Throughout this laboratory, you will operate under **The 7 Questions of Training Data Engineering**:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      THE 7 QUESTIONS OF TRAINING DATA ENGINEERING                      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Task Specification      ──► What exact behavior, persona, and boundaries must the   │
│                                model learn?                                            │
│ 2. Provenance & Licensing  ──► Where did the data originate, and what legal/commercial │
│                                restrictions apply?                                     │
│ 3. Profiling & Diagnostics ──► What anomalies, noise, and distributions exist in the   │
│                                raw data?                                               │
│ 4. Minimal Sanitization    ──► How do we remove artifacts while strictly preserving   │
│                                semantic meaning?                                       │
│ 5. Privacy & Verification  ──► How do we detect, redact, and verify the absence of PII?│
│ 6. Coverage & Synthesis    ──► Where are the coverage gaps, and how do we fill them?   │
│ 7. Auditing & Lineage      ──► How do we prove the resulting data is fit for purpose?  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 0.3 Scope and Progression: Current Session vs. Next Session

To ensure clean engineering separation, the workflow separates **model-agnostic data engineering** from **model-specific training execution**:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   CURRENT SESSION (DATA ENGINEERING & SYNTHESIS)                       │
│                                                                                        │
│  0. Task Specification: Define input/output contracts, edge cases, and scope           │
│  1. Real Data Profiling & Sanitization: Provenance, targeted normalization,            │
│     multi-level deduplication, privacy models, and audit protocols                     │
│  2. Synthetic Generation & Distillation: Coverage matrices, grounded generation,      │
│     structured validation, and lineage logging                                         │
│  3. Project Integration: Decision logs, Dataset Cards, and final data assets           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               NEXT SESSION (TOKENIZATION & SUPERVISED FINE-TUNING)                     │
│                                                                                        │
│  1. Structural Formatting: Template application (e.g., Qwen ChatML schema)             │
│  2. Tokenization: Sequence length padding, context window truncation, attention masks  │
│  3. Data Splitting: Stratified, non-contaminated Train / Validation / Test partitions  │
│  4. Training Execution: LoRA configuration, loss tracking, and evaluation runs         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 0.4 Task 0: Constructing the Dataset Specification

Before acquiring or generating a single training example, you must establish an explicit **Dataset Specification**. This specification acts as your acceptance criteria when auditing, cleaning, and synthesizing data.

```
                    ┌─────────────────────────────────────────┐
                    │          DATASET SPECIFICATION          │
                    └────────────────────┬────────────────────┘
                                         │
     ┌───────────────────┬───────────────┴───────────────┬───────────────────┐
     ▼                   ▼                               ▼                   ▼
┌──────────────┐  ┌──────────────┐              ┌─────────────────┐  ┌──────────────┐
│ TARGET TASK  │  │ USER PERSONA │              │ ACCEPTABLE vs   │  │ OUT-OF-SCOPE │
│ & BEHAVIOR   │  │ & INPUT TYPE │              │ UNACCEPTABLE    │  │ & NEGATIVE   │
│              │  │              │              │ OUTPUT CRITERIA │  │ BOUNDARIES   │
└──────────────┘  └──────────────┘              └─────────────────┘  └──────────────┘
```

### Hands-On Action: Draft Your Group Specification Matrix
Answer the following six core questions for your target domain:

| Dimension | Guiding Question | Project Definition (Example: Enterprise IT / DevOps) |
| :--- | :--- | :--- |
| **1. Primary Objective** | What specific problem does the model solve? | Diagnostic assistance for Kubernetes infrastructure alerts and container failure codes. |
| **2. Target Audience** | Who interacts with the model, and at what technical depth? | Junior-to-mid-level Site Reliability Engineers (SREs) and DevOps engineers during on-call incidents. |
| **3. Input Distribution** | What formats, logs, and phrasing will the model receive? | Raw terminal logs, stack traces, CLI outputs, and natural language troubleshooting queries. |
| **4. Ideal Response Profile** | What structural components define an optimal answer? | Immediate root-cause diagnosis $\rightarrow$ explicit remediation commands (Bash/YAML) $\rightarrow$ prevention check. |
| **5. Negative Behaviors** | What must the model **never** do under any circumstances? | Never suggest destructive commands (`rm -rf`, `DROP TABLE`) without explicit multi-stage warnings; never invent non-existent CLI flags. |
| **6. Negative & Out-of-Scope Cases** | How should the model handle missing info or out-of-scope requests? | If a log is incomplete, explicitly request the missing configuration files rather than guessing the failure state. |

---

## 0.5 Student Working Principles & Pedagogical Philosophy

### 1. Architectural Judgment Over Syntax Memorization
You are not expected to memorize regex strings, Unicode normalization tables, or specific SDK methods. Focus on understanding **data lineage, failure classification, distribution shifts, and privacy trade-offs**. Use code scaffolding to understand *why* transformations occur and *how* they alter downstream model behavior.

### 2. Active Use of AI Scaffolding and Collaborative Debugging
Use AI assistants (ChatGPT, Claude, Gemini, DeepSeek) as real-time research and debugging partners:
* Clarify mathematical or algorithmic concepts (e.g., *"How does MinHash approximate Jaccard similarity across varying n-gram lengths?"*).
* Adapt starter transformation functions to match unique schema variations in raw files.
* Brainstorm boundary cases and failure modes for your dataset specification.

### 3. Maintain an Audit Decision Log
Every cleaning step, filter threshold, and synthetic generation batch must be accompanied by an empirical justification recorded in your project decision log: **What did we change? Why? How many rows were affected? What valid data might we have lost?**

```
┌────────────────────────────────────────────────────────────────────────┐
│                      STUDENT WORKING PRINCIPLES                        │
├────────────────────────────────────────────────────────────────────────┤
│ • Define behavior specifications BEFORE acquiring data.                │
│ • Preserve semantic information; remove only true artifacts.           │
│ • Treat automated filters as diagnostic flags, not blind deletion rules│
│ • Document all data transformations and loss rates in a Decision Log.  │
│ • Maintain strict separation between raw, clean, and evaluation data.  │
└────────────────────────────────────────────────────────────────────────┘
```

---
---

# Part 1: Real-World Data Acquisition, Profiling, Sanitization, and Auditing

---

## 1.0 Overview: The Sanitization Philosophy

The fundamental rule of data cleaning for language models is:

$$\Large\textbf{Preserve Semantics; Remove Artifacts.}$$

A naive cleaning script that blindly strips all punctuation, normalizes every character sequence, or deletes any record outside arbitrary length boundaries often degrades training data quality. For instance, stripping backticks destroys markdown code formatting in technical datasets, and dropping short inputs eliminates valid real-world prompts like *"What is 2FA?"*.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 PHASE 1 OPERATIONAL PIPELINE                                 │
│                                                                                              │
│  [Source Dataset] ──► [Profile Data] ──► [Targeted Normalization] ──► [Multi-Tier Dedup]     │
│   (Licensing/Prov)    (Distributions)        (ftfy / NFC)               (Record / Prompt)    │
│                                                                                 │            │
│                                                                                 ▼            │
│   [Audited Clean Pool] ◄── [Audit & Error Taxonomy] ◄── [Privacy & PII Engine] ◄┘            │
│    (Immutable Output)         ("Random 50" Protocol)      (Detect / Transform / Verify)      │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1.1 Task 1: Sourcing, Provenance, and Licensing Audit

### High-Level Concepts
* **Data Provenance:** The documented lineage tracing where a dataset originated, who created it, how it was sampled, and what transformations it has undergone.
* **Licensing Compliance:** Not all open datasets are legally or commercially usable:
  * *Permissive Licenses (MIT, Apache 2.0, BSD):* Permit training, adaptation, and commercial deployment with standard attribution.
  * *Non-Commercial Restrictions (CC-BY-NC 4.0):* Strictly prohibit commercial model training and deployment.
  * *Terms of Service Enclosure:* Datasets generated via commercial frontier model APIs often include terms restricting their use in training models that compete with the provider.
* **Data Immutability Principle:** Always maintain a pristine, read-only copy of raw source data. All processing scripts must output to a new versioned target without overwriting the original assets.

---

### Hands-On Action: Ingest and Audit Metadata
Load a slice of a human-generated instruction dataset ([`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k)) and inspect its underlying structural features and provenance metadata.

#### Code Implementation
```python
from datasets import load_dataset
import json

# 1. Ingest raw dataset slice for profiling and experimentation
raw_dataset = load_dataset("databricks/databricks-dolly-15k", split="train[:500]")

# 2. Inspect structural schema and features
print("Dataset Features / Columns:")
for feature, spec in raw_dataset.features.items():
    print(f" - {feature}: {spec}")

# 3. Provenance and Record Inspection
sample_record = raw_dataset[0]
print("\n--- SAMPLE RECORD 0 ---")
print(json.dumps(sample_record, indent=2))
```

#### Metadata Audit Checklist
Before proceeding, review the dataset documentation card on Hugging Face and verify:
1. **Who created this dataset?** (Databricks employees, crowd workers, or scraped automation?)
2. **What is the explicit license?** (CC-BY-SA 3.0 — requires attribution and share-alike terms.)
3. **Is synthetic generation present?** (Dolly-15k is human-generated, avoiding synthetic distillation terms.)
4. **Does this schema align with our Task Specification?** (`instruction`, `context`, `response`, `category`).

---

## 1.2 Task 2: Targeted Text Normalization (Preserving Semantics)

### High-Level Concepts
* **Encoding Glitches (Mojibake):** Occur when text encoded in one format (e.g., Windows-1252 or Latin-1) is read as UTF-8, generating corrupt strings such as `donâ€™t` instead of `don't`. The `ftfy` library programmatically repairs byte-sequence decoding misalignments.
* **Unicode Normalization Form C (NFC):** Canonical decomposition followed by canonical composition. It guarantees that visually identical characters with differing underlying byte representations (e.g., the single code point `é` [U+00E9] versus decomposed `e` + combining acute accent `´` [U+0065, U+0301]) resolve to identical token sequences in the model tokenizer.
* **Selective Stripping:** Web scrapers frequently inject residual HTML boilerplate (`<div>`, `<p>`, `&amp;`), but naive regex cleaners risk corrupting mathematical operators (`<`, `>`), YAML tags, or markdown code snippets.

---

### Hands-On Action: Build a Safe Normalization Function
Write a cleaning function that resolves encoding corruption and whitespace debris while safeguarding programming syntax and markdown formatting.

#### Code Implementation
```python
import unicodedata
import re
import ftfy

def normalize_text_safe(text: str) -> str:
    """
    Applies non-destructive normalization:
    - Repairs Mojibake encoding artifacts
    - Applies Unicode NFC normalization
    - Strips residual HTML structural tags while preserving math operators
    - Normalizes excessive blank lines without altering single indentation
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 1. Resolve broken encoding representations
    text = ftfy.fix_text(text)
    
    # 2. Canonical Unicode NFC standard
    text = unicodedata.normalize("NFC", text)
    
    # 3. Targeted HTML entity replacement
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    
    # 4. Strip residual HTML tags (preserving standalone comparisons like 'if x < 5:')
    text = re.sub(r"<(/?(p|div|span|br|table|tr|td|th|a|ul|ol|li)[^>]*)>", " ", text, flags=re.IGNORECASE)
    
    # 5. Collapse excessive whitespace without destroying structured markdown indentation
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()

# Verification Test
dirty_samples = [
    "<p>System check: donâ€™t execute `SELECT * FROM users WHERE age &gt; 21;`</p>",
    "Error occurred in module:   \n\n\n\n[FATAL] Timeout reached."
]

for s in dirty_samples:
    print(f"Raw:     {s}")
    print(f"Cleaned: {normalize_text_safe(s)}\n")
```

---

## 1.3 Task 3: Profiling-Driven Filtering & Multi-Level Deduplication

### High-Level Concepts

#### Length Diagnostics vs. Blind Deletion
Never discard records based solely on static character-length thresholds. Character counts do not equal token counts or semantic completeness:
* A 15-character prompt (*"Define latency."*) is completely valid.
* A 3,000-character response might be essential for a detailed root-cause incident breakdown.
* **Correct Practice:** Profile the length distributions of both instructions and responses independently to detect anomalies (empty records, truncated fragments, unextracted PDF headers) for targeted review.

```
                    LENGTH PROFILING & FILTERING LOGIC
                    
    Instruction Length                Response Length               Total Record Length
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│ Suspicious: < 3 words   │     │ Suspicious: < 5 words   │     │ Flag for Review:        │
│ Flag for manual check   │     │ Flag for incomplete ans │     │ Exceeds target context  │
│ (May be valid short q)  │     │                         │     │ window (e.g. >2k tokens)│
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
```

#### Multi-Level Deduplication Strategy
Deduplication is not a binary operation. Different duplication patterns require different handling:

1. **Exact Record Duplicates (Full Match):**
   * *Signature:* Identical `instruction` AND identical `response`.
   * *Action:* Retain one instance; discard identical copies to prevent the model from overfitting to specific token sequences.
2. **Duplicate Instructions (1-to-Many Mappings):**
   * *Signature:* Identical `instruction`, but distinct `responses`.
   * *Action:* **Do not delete automatically.** These may represent alternate valid solutions, different perspectives, or contradictory answers. Inspect and arbitrate: merge into a comprehensive answer, select the highest-quality response, or discard if contradictory.
3. **Near-Duplicates (Paraphrasing):**
   * *Signature:* Structurally similar instructions with minor phrasing shifts (*"How do I restart pods?"* vs *"How to restart a pod?"*).
   * *Handling (Advanced Extension):* Identified using MinHash and Locality-Sensitive Hashing (LSH) to assess whether linguistic diversity is authentic or redundant.

---

### Hands-On Action: Profile Distributions and Apply Multi-Level Deduplication
Build a data profiling and deduplication pipeline that groups duplicate prompts for inspection rather than blindly purging them.

#### Code Implementation
```python
import hashlib
from collections import defaultdict

def profile_and_deduplicate(raw_records):
    """
    Categorizes records, detects exact duplicates, and isolates
    duplicate instructions with divergent answers for manual arbitration.
    """
    exact_duplicates = 0
    prompt_to_records = defaultdict(list)
    profiled_clean = []
    
    # Track metrics
    length_diagnostics = {"short_instructions": 0, "short_responses": 0, "long_records": 0}
    
    for row in raw_records:
        # Standardize input fields
        instr = normalize_text_safe(row.get("instruction", ""))
        resp = normalize_text_safe(row.get("response", ""))
        ctx = normalize_text_safe(row.get("context", ""))
        
        # Diagnostic profiling flags
        word_count_instr = len(instr.split())
        word_count_resp = len(resp.split())
        
        if word_count_instr < 3:
            length_diagnostics["short_instructions"] += 1
        if word_count_resp < 4:
            length_diagnostics["short_responses"] += 1
        if len(instr + resp) > 4000:
            length_diagnostics["long_records"] += 1
            
        # Group by normalized instruction text
        prompt_key = instr.lower()
        
        # Check for exact full record duplicate
        full_hash = hashlib.md5(f"{prompt_key} ||| {resp.lower()}".encode("utf-8")).hexdigest()
        
        # Record object with lineage metadata
        record = {
            "instruction": instr,
            "context": ctx,
            "response": resp,
            "full_hash": full_hash,
            "source_id": row.get("category", "unassigned")
        }
        
        # Check if identical response already exists for this instruction
        if any(existing["full_hash"] == full_hash for existing in prompt_to_records[prompt_key]):
            exact_duplicates += 1
            continue
            
        prompt_to_records[prompt_key].append(record)

    # Resolve prompt groups without silently choosing one answer.
    multi_response_prompts = 0
    for prompt_key, records in prompt_to_records.items():
        if len(records) > 1:
            multi_response_prompts += 1
            for record in records:
                record["needs_manual_arbitration"] = True
                profiled_clean.append(record)
        else:
            profiled_clean.append(records[0])
            
    print(f"--- PROFILING & DEDUPLICATION REPORT ---")
    print(f"Total Ingested:           {len(raw_records)}")
    print(f"Exact Record Dupes Purged:{exact_duplicates}")
    print(f"Multi-Answer Groups Found:{multi_response_prompts} (Flagged for manual arbitration)")
    print(f"Length Diagnostic Flags:  {length_diagnostics}")
    print(f"Final Cleaned Records:    {len(profiled_clean)}")
    
    return profiled_clean

cleaned_records = profile_and_deduplicate(raw_dataset)
```

---

## 1.4 Task 4: Sensitive Data and PII Management

### High-Level Concepts
* **PII Detection $\neq$ Privacy Compliance:** Running a basic regex for emails does not make a dataset privacy-compliant. Private information encompasses direct identifiers (names, Social Security numbers, IP addresses) and quasi-identifiers (unique combinations of role, location, and timestamps that allow re-identification).
* **The 3-Step Privacy Framework:**
  1. **Detect:** Scan across structured patterns (regex) and semantic entities using Named Entity Recognition (NER) models (e.g., [Microsoft Presidio](https://microsoft.github.io/presidio/), spaCy).
  2. **Transform:** Apply a deterministic privacy strategy:
     * *Placeholder Masking:* Swap sensitive entities with structural tags (e.g., `[EMAIL_ADDRESS]`, `[IP_ADDRESS]`). Highly secure and explicit.
     * *Synthetic Surrogate Replacement (Faker):* Replace real names with synthetic names. Preserves natural conversational flow.
     * *Generalization:* Coarsen exact values (e.g., replacing exact timestamps with `[Q3-2023]`).
  3. **Verify:** Perform secondary validation scans on transformed outputs to confirm that placeholders did not misalign and that no unmasked entities survived.
* **Separation of Source:** Never perform in-place PII modification on your source files. Keep raw, masked, and surrogate datasets strictly separated and access-controlled.

```
                         THE 3-STEP PRIVACY FRAMEWORK
                         
      [DETECT]                     [TRANSFORM]                     [VERIFY]
┌──────────────────┐          ┌────────────────────┐          ┌─────────────────┐
│ Regex + NER      │   ───►   │ Synthetic Replace  │   ───►   │ Secondary Scan  │
│ (Presidio Engine)│          │ or Explicit Mask   │          │ (Zero unmasked) │
└──────────────────┘          └────────────────────┘          └─────────────────┘
```

---

### Hands-On Action: Build a Multi-Entity Scrubber and Verifier
Implement a PII detection and redaction pipeline that processes sensitive entities and verifies the sanitized text.

#### Code Implementation
```python
import re

class SensitiveDataScrubber:
    def __init__(self):
        # Compiled patterns for standard structured identifiers
        self.patterns = {
            "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "IPV4": re.compile(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"),
            "PHONE": re.compile(r"\b(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})\b"),
            "INTERNAL_KEY": re.compile(r"\b(?:api[_-]?key|secret[_-]?token|bearer)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}['\"]?\b", re.IGNORECASE)
        }

    def scrub(self, text: str) -> tuple[str, dict]:
        """Redacts structured identifiers and returns scrubbed text along with detection metrics."""
        counts = defaultdict(int)
        scrubbed_text = text
        
        for entity_type, pattern in self.patterns.items():
            matches = pattern.findall(scrubbed_text)
            if matches:
                counts[entity_type] += len(matches)
                scrubbed_text = pattern.sub(f"[{entity_type}_REDACTED]", scrubbed_text)
                
        return scrubbed_text, dict(counts)

    def verify_clean(self, text: str) -> bool:
        """Secondary verification: returns True if no unmasked patterns remain."""
        return not any(pattern.search(text) for pattern in self.patterns.values())

# Demonstration
scrubber = SensitiveDataScrubber()

test_payload = (
    "Alert from node 192.168.1.105: Deployment failed. "
    "Contact sysadmin alex.miller@internal-corp.net or call 415-555-0198. "
    "Auth config: api_key='ak_live_99481029481029381029'"
)

sanitized_payload, detection_summary = scrubber.scrub(test_payload)
is_verified = scrubber.verify_clean(sanitized_payload)

print("--- PII SCRUBBING AUDIT ---")
print(f"Raw Input:       {test_payload}")
print(f"\nSanitized Text:  {sanitized_payload}")
print(f"\nEntities Redacted: {detection_summary}")
print(f"Verification Passed: {is_verified}")
```

---

## 1.5 Task 5: The "Random 50" Audit Protocol & Error Taxonomy

### High-Level Concepts
* **Auditing Is Not a Statistical Guarantee:** Manually inspecting 50 randomly sampled records does not prove the absence of errors across a 10,000-record dataset. Instead, it serves as a **qualitative discovery mechanism** to uncover systemic failure modes in your pipeline.
* **Error Taxonomy Classification:** Rather than marking records with a simple "good/bad" label, categorize observed errors into a structured taxonomy:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                SYSTEMIC ERROR TAXONOMY                                 │
├────────────────────┬───────────────────────────────────────────────────────────────────┤
│ Error Class        │ Description & Pipeline Remedy                                     │
├────────────────────┼───────────────────────────────────────────────────────────────────┤
│ 1. Factually False │ Response contains incorrect technical or domain statements.       │
│                    │ Remedy: Update source filters; remove unreliable data categories. │
├────────────────────┼───────────────────────────────────────────────────────────────────┤
│ 2. Context Debris  │ Unresolved HTML tags, broken markdown, or Mojibake corruption.    │
│                    │ Remedy: Refine regex patterns in `normalize_text_safe`.           │
├────────────────────┼───────────────────────────────────────────────────────────────────┤
│ 3. PII Leakage     │ Real-world names, internal IPs, or private keys remain unmasked.  │
│                    │ Remedy: Expand NER entity definitions or regex patterns.          │
├────────────────────┼───────────────────────────────────────────────────────────────────┤
│ 4. Out-of-Scope    │ Does not align with the Dataset Specification (Task 0).           │
│                    │ Remedy: Adjust upstream keyword and topic selection filters.      │
├────────────────────┼───────────────────────────────────────────────────────────────────┤
│ 5. Ambiguous/Empty │ Instruction lacks necessary context to produce a coherent answer. │
│                    │ Remedy: Apply stricter length and context presence filters.       │
└────────────────────┴───────────────────────────────────────────────────────────────────┘
```

* **The Remediation Loop:**
$$\text{Sample Audit} \longrightarrow \text{Classify Failures} \longrightarrow \text{Refine Normalizer/Filter} \longrightarrow \text{Re-Audit}$$

---

### Hands-On Action: Execute an In-Class Audit Sample
Sample records from your cleaned pool, inspect them against your Dataset Specification, and complete the audit log.

#### Code Implementation
```python
import random

def extract_audit_sample(dataset_pool, sample_size=5):
    """Samples records and attaches an evaluation schema for manual review."""
    random.seed(42)
    sample = random.sample(dataset_pool, min(sample_size, len(dataset_pool)))
    
    audit_sheet = []
    for idx, row in enumerate(sample):
        audit_entry = {
            "audit_id": idx + 1,
            "instruction": row["instruction"],
            "response": row["response"],
            "error_classification": "NONE", # Options: FACTUAL, DEBRIS, PII_LEAK, OUT_OF_SCOPE, AMBIGUOUS
            "notes": ""
        }
        audit_sheet.append(audit_entry)
    return audit_sheet

audit_batch = extract_audit_sample(cleaned_records, sample_size=5)

for record in audit_batch:
    print(f"\n================ AUDIT RECORD #{record['audit_id']} ================")
    print(f"PROMPT:   {record['instruction']}")
    print(f"RESPONSE: {record['response'][:200]}...") # Print first 200 chars
```

#### Team Exercise: Error Classification Worksheet
Review the printed records with your team and record your findings in the structured audit log:

| Sample # | Observed Error Class | Evidence / Root Cause | Required Pipeline Fix |
| :--- | :--- | :--- | :--- |
| **Example 1** | *Context Debris* | Leftover `<td>` tags in table row. | Add table tags to HTML regex stripper. |
| **Example 2** | *Out-of-Scope* | Asks for creative poem about cooking. | Add domain keyword filter for IT topics. |
| **Record 1** | | | |
| **Record 2** | | | |

---

## 1.6 Task 6: Data Lineage Tracking and Record Schema

### High-Level Concepts
* **Record-Level Lineage:** In production ML pipelines, you must be able to trace every single training example back to its exact origin, transformation history, and verification status.
* **Intermediate Representation:** At the conclusion of Phase 1, data should be stored in a clean, self-documenting, model-independent intermediate format. Formatting for specific chat templates (ChatML, Alpaca, Llama-3) is handled during the tokenization stage in the next session.

```json
{
  "record_id": "REAL_DOLLY_00412",
  "instruction": "How do I check Kubernetes pod logs for a failed deployment?",
  "context": "",
  "response": "Use `kubectl logs deployment/<deployment-name> --all-containers=true` to inspect logs across all pods.",
  "metadata": {
    "source_dataset": "databricks/databricks-dolly-15k",
    "source_license": "CC-BY-SA-3.0",
    "provenance_type": "human_curated",
    "normalization_applied": ["ftfy", "unicode_nfc", "strip_html_safe"],
    "pii_scanned": true,
    "pii_detected": false,
    "audit_status": "passed_spot_check"
  }
}
```

---

### Hands-On Action: Export Sanitized Real Data Pool
Transform your cleaned records into the standard lineage-preserving format and export the intermediate JSON asset.

#### Code Implementation
```python
import json
import uuid

def format_intermediate_pool(clean_records, source_name="dolly-15k", license_tag="CC-BY-SA-3.0"):
    lineage_pool = []
    
    for row in clean_records:
        record_id = f"REAL_{source_name.upper()[:5]}_{uuid.uuid4().hex[:6]}"
        
        entry = {
            "record_id": record_id,
            "instruction": row["instruction"],
            "context": row.get("context", ""),
            "response": row["response"],
            "metadata": {
                "source_dataset": source_name,
                "source_license": license_tag,
                "provenance_type": "human_curated",
                "transformations": ["ftfy_v1", "nfc_normalize", "multi_level_dedup", "regex_pii_scrub"],
                "verified": True
            }
        }
        lineage_pool.append(entry)
        
    return lineage_pool

real_data_pool = format_intermediate_pool(cleaned_records, "dolly15k", "CC-BY-SA-3.0")

# Save intermediate asset to disk
output_real_file = "sanitized_real_data_pool.json"
with open(output_real_file, "w", encoding="utf-8") as f:
    json.dump(real_data_pool, f, indent=2, ensure_ascii=False)

print(f"Successfully finalized {len(real_data_pool)} sanitized real-world records.")
print(f"Saved intermediate asset to: {output_real_file}")
```

---

## 1.7 Phase 1 Decision Log & Checkpoint

Before proceeding to synthetic generation in Phase 2, document your Phase 1 outcomes in your project notebook:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   PHASE 1 DATA DECISION LOG ENTRY                     │
├────────────────────────────────────────────────────────────────────────┤
│ • Dataset Sourced: databricks/databricks-dolly-15k (License: CC-BY-SA) │
│ • Raw Records Ingested: 500                                            │
│ • Exact Duplicates Purged: [X] records                                 │
│ • Multi-Response Collisions Arbitrated: [Y] records                    │
│ • PII Redactions Applied: [Z] entities scrubbed                        │
│ • "Random 50" Audit Error Rate: [N]% (Primary error class: _________)  │
│ • Corrective Action Applied: ________________________________________  │
│ • Final Sanitized Real Pool Size: [Total] records                      │
└────────────────────────────────────────────────────────────────────────┘
```

# Part 2: Synthetic Data Generation, Distillation, and Multi-Layer Validation

---

## 2.0 Overview: The Synthetic Gap-Filling Philosophy

Synthetic data generation is not an automated shortcut to inflate dataset size arbitrarily. Its primary engineering purpose is **targeted coverage gap-filling**: generating high-quality demonstrations for task scenarios, edge cases, failure modes, and user personas that are missing or underrepresented in your real-world data assets.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              PHASE 2 OPERATIONAL PIPELINE                              │
│                                                                                        │
│  [Real Data Profile] ──► [Coverage Matrix] ──► [Teacher Generation] ──► [Evol-Instruct]│
│   (Identify Gaps)         (Multi-Dim Grid)      (Positive & Refusal)     (Realism-Bound)│
│                                                                                │       │
│                                                                                ▼       │
│  [Verified Synth Pool] ◄── [Lexical Diagnostics] ◄── [Three-Layer Validation] ◄┘       │
│   (Lineage Tagged)          (Trigram Profiling)       (Programmatic/Domain/Human)      │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2.1 Task 1: Coverage Analysis and Multi-Dimensional Matrix Design

### High-Level Concepts
* **Identifying Distributional Gaps:** Before calling a generation API, evaluate your real-data pool against your **Task Specification (Task 0)**. Identify where data is sparse (e.g., high on basic definitions, zero coverage of multi-step failure recovery).
* **The "Naive Prompt" Failure Mode:** Prompts like *"Generate 50 technical support questions"* produce generic, repetitive samples clustering around the teacher model's highest-probability tokens.
* **The Domain Coverage Matrix:** Instead of forcing every project into a rigid 3D grid, construct a flexible dimensional space matching your domain architecture (e.g., $\text{Core Capability} \times \text{Environment / Stack} \times \text{Failure Mode} \times \text{User Technical Level}$).

```
                             COVERAGE MATRIX DESIGN
                             
           DIMENSION 1                     DIMENSION 2                     DIMENSION 3
        [Core Capability]              [Failure Mode / State]           [User Persona / Context]
      ┌──────────────────┐            ┌────────────────────┐          ┌──────────────────────────┐
      │ Pod Lifecycle    │            │ OOMKilled (137)    │          │ Panicked On-Call SRE     │
      │ Storage & PVCs   │     X      │ CrashLoopBackOff   │    X     │ Junior Web Developer     │
      │ Ingress & TLS    │            │ Certificate Expiry │          │ Compliance SecOps Lead   │
      └──────────────────┘            └────────────────────┘          └──────────────────────────┘
                                                │
                                                ▼
                                [Targeted Generation Seed]
```

---

### Hands-On Action: Map Coverage Gaps to a Generation Seed Grid
Construct a domain-specific coverage matrix that samples underrepresented scenarios.

#### Code Implementation
```python
import random
import itertools

# Define coverage axes tailored to your Task Specification
coverage_axes = {
    "subsystem": ["Pod Lifecycle", "Persistent Storage", "Ingress / TLS", "IAM & RBAC", "Cluster Networking"],
    "failure_state": ["OOMKilled (Exit 137)", "CrashLoopBackOff", "Pending / Node Unschedulable", "SSL Cert Expired", "Unauthorized 403"],
    "user_depth": ["Junior Developer (Needs basic CLI steps)", "Senior SRE (Needs root-cause kernel/log analysis)", "Automated Pipeline / CI Runner"]
}

def sample_coverage_seeds(axes_dict, num_samples=6):
    """Generates unique seed combinations across the multidimensional coverage space."""
    # Generate full Cartesian product of all defined axes
    keys = list(axes_dict.keys())
    all_combinations = list(itertools.product(*[axes_dict[k] for k in keys]))
    
    random.seed(42)
    sampled_tuples = random.sample(all_combinations, min(num_samples, len(all_combinations)))
    
    seeds = []
    for combo in sampled_tuples:
        seeds.append(dict(zip(keys, combo)))
    return seeds

sample_seeds = sample_coverage_seeds(coverage_axes, num_samples=4)
for idx, seed in enumerate(sample_seeds, 1):
    print(f"Seed {idx}: {seed}")
```

---

## 2.2 Task 2: Grounded Synthetic Generation (Positive & Refusal Demonstrations)

### High-Level Concepts
* **Teacher-Student Distillation Mechanics:** Using a high-capacity teacher model (e.g., **Gemini 2.5 Flash**) to produce structured demonstration trajectories for fine-tuning a smaller student model (e.g., Qwen 1.5B/1B).
* **Generating Both Positive and Negative/Refusal Demonstrations:**
  * *Standard Instruction Pairs:* Demonstrations of complete, accurate solutions.
  * *Negative / Boundary / Refusal Pairs:* Explicit demonstrations of how the model should behave when input is ambiguous, incomplete, dangerous, or out-of-scope (e.g., *"I cannot execute this script because it includes destructive `DROP DATABASE` commands without backup safeguards"*).
* **API Cost and Batching Optimization:** Requesting structured batches amortizes prompt token costs while enforcing machine-readable JSON schemas.

---

### Hands-On Action: Connect Teacher API and Generate Balanced Demonstration Batches
Configure the Google GenAI SDK to generate positive and negative instruction pairs using your coverage seeds.

#### Code Implementation
```python
!pip install -q -U google-genai

import json
from google import genai
from google.colab import userdata

# Initialize client using Colab Secrets or direct environment variable
GEMINI_API_KEY = userdata.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

def generate_grounded_synthetic_batch(seeds, include_refusal_cases=True):
    """
    Generates structured instruction-response pairs enforcing Task Specification
    boundaries, including positive demonstrations and realistic refusal/clarification cases.
    """
    system_prompt = (
        "You are an expert systems engineer creating rigorous supervised training data. "
        "Generate realistic user queries and technically precise, structured responses. "
        "For boundary or dangerous prompts, provide a professional, helpful refusal or clarification request. "
        "Output strictly valid JSON: a list of objects, each containing: "
        "'instruction', 'response', 'coverage_metadata', and 'sample_type' ('positive' or 'refusal')."
    )
    
    seed_descriptions = "\n".join([
        f"- Target Area: {s['subsystem']} | Condition: {s['failure_state']} | Persona: {s['user_depth']}"
        for s in seeds
    ])
    
    user_prompt = f"""Generate 1 high-fidelity training pair for each of the following seed conditions:
{seed_descriptions}

Ensure at least 1 pair represents an edge case requiring the assistant to ask clarifying questions or refuse dangerous parameters."""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config={
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "temperature": 0.80
        }
    )
    
    try:
        return json.loads(response.text)
    except json.JSONDecodeError as e:
        print(f"JSON Parsing Error: {e}")
        return []

synthetic_batch = generate_grounded_synthetic_batch(sample_seeds)
print(f"Generated {len(synthetic_batch)} structured synthetic pairs.\n")
if synthetic_batch:
    print("Sample Output Preview:")
    print(json.dumps(synthetic_batch[0], indent=2))
```

---

<details>
<summary><strong>Optional Advanced Extension: Realism-Constrained Complexity Evolution (Evol-Instruct)</strong></summary>

## 2.3 Task 3: Realism-Constrained Complexity Evolution (Evol-Instruct)

### High-Level Concepts
* **The Realism Constraint on Evol-Instruct:** Mutating a prompt to make it "more complex" does not inherently make it better training data. **Complexity must reflect realistic operational constraints.** Artificially bloated, verbose prompts degrade fine-tuning quality.
* **Valid Mutation Strategies:**
  1. *Add Plausible Constraints:* Introduce real-world dependencies (e.g., *"Resolve this OOM error without increasing the cloud provider VM node pool size"*).
  2. *Deepen Multi-Step Dependencies:* Require connecting two related symptoms (e.g., cross-referencing high CPU latency with database connection pooling limits).
  3. *Handle Ambiguity / Incomplete Data:* Provide an incomplete error trace and train the model to request the precise missing log stanza.
* **Supervision Schemas:**
  * *Final-Answer Supervision:* Direct Question $\rightarrow$ Answer (best for direct lookups and simple commands).
  * *Structured Reasoning Supervision:* Question $\rightarrow$ **Observations $\rightarrow$ Diagnosis $\rightarrow$ Actionable Remediation** (superior for diagnostic workflows, avoids unstructured chain-of-thought drift).
* **The Regeneration Mandate:** Never pair an evolved instruction with the original baseline response. Always regenerate the response to match the updated prompt.

```
                    STRUCTURED REASONING SUPERVISION PATTERN
                    
┌────────────────────────────────────────────────────────────────────────┐
│ USER INSTRUCTION:                                                      │
│ "Pod `billing-worker` is stuck in CrashLoopBackOff with Exit Code 137. │
│ Node capacity cannot be increased. How do we resolve this?"            │
├────────────────────────────────────────────────────────────────────────┤
│ MODEL RESPONSE (Structured Reasoning Trace):                           │
│                                                                        │
│ 1. OBSERVATION & DIAGNOSIS:                                            │
│    Exit Code 137 indicates a SIGKILL triggered by the Linux OOMKiller.  │
│    The container process exceeded its configured memory limits.        │
│                                                                        │
│ 2. CONSTRAINT ANALYSIS:                                                │
│    Because node capacity is fixed, we cannot simply increase limits.   │
│    Remediation must optimize memory footprint inside the runtime.      │
│                                                                        │
│ 3. ACTIONABLE REMEDIATION:                                             │
│    - Inspect heap usage: `jcmd <pid> GC.heap_info` or memory profile.  │
│    - Set JVM max heap allocation below container limit (-Xmx parameter)│
│    - Verify cgroup limits via: `kubectl describe pod billing-worker`   │
└────────────────────────────────────────────────────────────────────────┘
```

---

### Hands-On Action: Apply Realism-Constrained Evolution and Structured Traces
Mutate a baseline seed query by applying realistic production constraints, then generate a structured reasoning response.

#### Code Implementation
```python
def evolve_instruction_realistically(base_instruction: str) -> str:
    """Evolves a user query by adding realistic production constraints, not artificial complexity."""
    prompt = f"""You are an expert SRE curriculum designer.
Take the following basic user prompt and evolve it into a realistic, challenging real-world incident scenario.

Rules:
1. Add 1-2 realistic constraints (e.g., legacy version, immutable network policy, resource caps).
2. Do NOT add artificial academic trivia or unnecessary conversational padding.
3. Keep the user's intent clear and practical.

Original Prompt:
"{base_instruction}"

Return ONLY the evolved prompt text."""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"temperature": 0.7}
    )
    return response.text.strip()

def generate_structured_reasoning_response(evolved_instruction: str) -> str:
    """Generates an answer using a structured Observation -> Diagnosis -> Remediation trace."""
    prompt = f"""Provide a technically rigorous response to the following prompt.
Structure your answer using these exact sections:
1. [OBSERVATION & DIAGNOSIS]: Immediate root cause analysis.
2. [TRADE-OFF & CONSTRAINT ANALYSIS]: Evaluation of operating constraints.
3. [ACTIONABLE REMEDIATION]: Precise commands, YAML specs, or code fixes.

User Prompt:
"{evolved_instruction}" """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"temperature": 0.3} # Lower temperature for deterministic precision
    )
    return response.text.strip()

# Demonstration
if synthetic_batch:
    base_query = synthetic_batch[0]["instruction"]
    evolved_query = evolve_instruction_realistically(base_query)
    structured_ans = generate_structured_reasoning_response(evolved_query)
    
    print("--- BASE INSTRUCTION ---")
    print(base_query)
    print("\n--- REALISTICALLY EVOLVED INSTRUCTION ---")
    print(evolved_query)
    print("\n--- STRUCTURED REASONING DEMONSTRATION ---")
    print(structured_ans[:400] + "...\n[Truncated for display]")
```

---

</details>

## 2.4 Task 4: The Three-Layer Validation Architecture

### High-Level Concepts
* **The Circular Validation Trap:** Relying solely on an "LLM Judge" creates a circular dependency ($\text{LLM generates} \rightarrow \text{LLM validates} \rightarrow \text{Human blindly accepts}$). If the teacher hallucinates an invalid CLI flag, the LLM judge frequently approves the same hallucination.
* **The Three-Layer Validation Model:**

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              THREE-LAYER VALIDATION MODEL                              │
├────────────────────┬─────────────────────────────────┬─────────────────────────────────┤
│ Validation Layer   │ Scope & Execution               │ Failure Action                  │
├────────────────────┼─────────────────────────────────┼─────────────────────────────────┤
│ Layer 1:           │ JSON syntax, required schema    │ Drop record or re-parse.        │
│ Programmatic       │ keys, empty strings, character/ │ Purge unparseable data          │
│ Verification       │ token length thresholds, regex  │ automatically.                  │
│                    │ for sensitive tokens/keys.      │                                 │
├────────────────────┼─────────────────────────────────┼─────────────────────────────────┤
│ Layer 2:           │ Deterministic syntax checks:    │ Flag for automated repair or    │
│ Domain-Specific    │ bash syntax validation, valid   │ discard if critical hallucina-  │
│ Verification       │ YAML parsing, SQL linters, code │ tion (e.g. invalid flags).      │
│                    │ compilation, accounting math.   │                                 │
├────────────────────┼─────────────────────────────────┼─────────────────────────────────┤
│ Layer 3:           │ Human spot-checks of flagged or │ Qualitative audit, update seed  │
│ Expert Human       │ sampled records against the     │ constraints, rebuild generation │
│ Review             │ Dataset Specification.          │ rules if failure rate > 5%.     │
└────────────────────┴─────────────────────────────────┴─────────────────────────────────┘
```

---

### Hands-On Action: Implement Programmatic & Domain Linters
Build a multi-layer validator that checks schema compliance, strips markdown debris, and parses embedded YAML/code blocks.

#### Code Implementation
```python
import yaml

class SyntheticDataValidator:
    def __init__(self, min_instr_words=4, min_resp_words=8):
        self.min_instr_words = min_instr_words
        self.min_resp_words = min_resp_words

    def layer1_programmatic_check(self, record: dict) -> tuple[bool, str]:
        """Validates schema integrity, data types, and length boundaries."""
        if not isinstance(record, dict):
            return False, "Record is not a dictionary"
        
        for key in ["instruction", "response"]:
            if key not in record or not isinstance(record[key], str) or not record[key].strip():
                return False, f"Missing or empty key: {key}"
                
        if len(record["instruction"].split()) < self.min_instr_words:
            return False, "Instruction below minimum word length threshold"
            
        if len(record["response"].split()) < self.min_resp_words:
            return False, "Response below minimum word length threshold"
            
        return True, "Passed Layer 1"

    def layer2_domain_syntax_check(self, text: str) -> tuple[bool, str]:
        """
        Extracts embedded YAML code blocks from the response and verifies 
        syntactic validity using a real YAML parser (domain linter).
        """
        yaml_blocks = re.findall(r"```(?:yaml|yml)\n(.*?)```", text, re.DOTALL)
        for block in yaml_blocks:
            try:
                yaml.safe_load(block)
            except yaml.YAMLError as exc:
                return False, f"Corrupted YAML block detected: {exc}"
                
        # Check for forbidden destructive commands without confirmation checks
        dangerous_patterns = [r"\brm -rf /\b", r"\bmkfs\b", r"\bdd if="]
        for pattern in dangerous_patterns:
            if re.search(pattern, text):
                return False, "Contains dangerous command without safety context"
                
        return True, "Passed Layer 2"

# Test Validator
validator = SyntheticDataValidator()

valid_sample = {
    "instruction": "How do I configure a Kubernetes resource quota for CPU limits?",
    "response": "Apply the following manifest:\n```yaml\napiVersion: v1\nkind: ResourceQuota\nmetadata:\n  name: cpu-quota\nspec:\n  hard:\n    limits.cpu: '4'\n```"
}

corrupted_sample = {
    "instruction": "Fix quota",
    "response": "Use this config:\n```yaml\napiVersion: v1\nkind: ResourceQuota\nmetadata: name: cpu-quota: invalid: [[\n```"
}

for name, sample in [("Valid Sample", valid_sample), ("Corrupted Sample", corrupted_sample)]:
    l1_pass, l1_msg = validator.layer1_programmatic_check(sample)
    l2_pass, l2_msg = validator.layer2_domain_syntax_check(sample["response"]) if l1_pass else (False, "Skipped")
    print(f"--- {name} ---")
    print(f"Layer 1 (Schema): {l1_pass} ({l1_msg})")
    print(f"Layer 2 (Domain): {l2_pass} ({l2_msg})\n")
```

---

<details>
<summary><strong>Optional Advanced Extension: Diversity Diagnostics and Comparative Teacher Evaluation</strong></summary>

## 2.5 Task 5: Lexical Diagnostics & Trigram Diversity Profiling

### High-Level Concepts
* **Diversity as a Comparative Diagnostic (Not a Pass/Fail Law):** Lexical diversity metrics measure the variety of phrasing and n-gram patterns across generated text. 
* **Trigram Diversity Ratio:**
  $$\text{Trigram Diversity} = \frac{\text{Count of Unique 3-Word Sequences}}{\text{Total 3-Word Sequences}}$$
* **Interpreting the Metric:**
    * *Higher diversity:* May indicate useful linguistic variety across prompts and responses.
    * *Lower diversity:* A diagnostic signal to investigate possible repetitive teacher phrasing (e.g., starting every response with *"Sure, I can help with that!"*).
  * *Context Matters:* Highly structured outputs (such as standardized SQL queries or formal medical reports) naturally exhibit lower lexical diversity. Do not force high diversity at the expense of domain correctness.

---

### Hands-On Action: Profile and Compare Dataset Diversity
Calculate and compare trigram diversity across your real data slice and synthetic outputs.

#### Code Implementation
```python
def compute_trigram_diversity_profile(text_list: list[str]) -> dict:
    """Computes total, unique, and diversity ratio of word-level trigrams."""
    all_trigrams = []
    
    for text in text_list:
        words = re.findall(r"\b\w+\b", text.lower())
        if len(words) >= 3:
            trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
            all_trigrams.extend(trigrams)
            
    if not all_trigrams:
        return {"total_trigrams": 0, "unique_trigrams": 0, "diversity_ratio": 0.0}
        
    unique_count = len(set(all_trigrams))
    total_count = len(all_trigrams)
    
    return {
        "total_trigrams": total_count,
        "unique_trigrams": unique_count,
        "diversity_ratio": round(unique_count / total_count, 3)
    }

real_responses = [r["response"] for r in cleaned_records]
synthetic_responses = [r["response"] for r in synthetic_batch if "response" in r]

print("--- LEXICAL DIVERSITY COMPARISON ---")
print(f"Real Data Profile:      {compute_trigram_diversity_profile(real_responses)}")
print(f"Synthetic Data Profile: {compute_trigram_diversity_profile(synthetic_responses)}")
```

---

## 2.6 Task 6: Specification-Aligned Comparative Teacher Evaluation

### Hands-On Action: Teacher Benchmark Exercise
Select one complex prompt from your Coverage Matrix and compare responses across candidate teacher models (e.g., **Gemini 2.5 Flash** vs. **ChatGPT** or **Claude**). Evaluate them using this project-aligned rubric:

| Evaluation Dimension | Scoring Criterion (1 to 5 Scale) | Gemini 2.5 Flash Score | Alternative Model Score |
| :--- | :--- | :--- | :--- |
| **1. Constraint Adherence** | Follows all operational boundaries without conversational filler? | | |
| **2. Factual / CLI Accuracy** | Terminal commands, parameters, and code syntax are 100% valid? | | |
| **3. Negative Handling** | Accurately identifies missing data or refuses unsafe requests? | | |
| **4. Domain Persona Tone** | Fits the target user persona defined in Task 0 Specification? | | |

---

</details>

## 2.7 Lineage-Preserving Synthetic Pool Export

### Hands-On Action: Save Verified Synthetic Pool
Package all validated synthetic pairs into the standard intermediate JSON format with full lineage tracking.

#### Code Implementation
```python
import uuid

def format_synthetic_lineage_pool(validated_records, teacher_tag="gemini-2.5-flash"):
    synthetic_pool = []
    
    for row in validated_records:
        rec_id = f"SYNTH_{teacher_tag.upper()[:6]}_{uuid.uuid4().hex[:6]}"
        entry = {
            "record_id": rec_id,
            "instruction": row["instruction"],
            "context": row.get("context", ""),
            "response": row["response"],
            "metadata": {
                "source_dataset": "synthetic_distillation",
                "source_license": "educational_distillation_poc",
                "provenance_type": "synthetic_teacher_distilled",
                "teacher_model": teacher_tag,
                "coverage_metadata": row.get("coverage_metadata", {}),
                "sample_type": row.get("sample_type", "positive"),
                "validation_passed": ["layer1_schema", "layer2_domain_syntax"],
                "verified": True
            }
        }
        synthetic_pool.append(entry)
    return synthetic_pool

synthetic_data_pool = format_synthetic_lineage_pool(synthetic_batch)

output_synth_file = "verified_synthetic_data_pool.json"
with open(output_synth_file, "w", encoding="utf-8") as f:
    json.dump(synthetic_data_pool, f, indent=2, ensure_ascii=False)

print(f"Exported {len(synthetic_data_pool)} verified synthetic records to {output_synth_file}")
```

---
---

# Part 3: Project Integration, Evaluation Design, and Deliverables

---

## 3.0 Overview of Phase 3

In this phase, you will combine your sanitized real-world data and verified synthetic data into an audited training asset for your specific semester project. 

Crucially, you will also construct a **Held-Out Golden Evaluation Set** to assess your model's true downstream performance in the next session.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              PHASE 3 INTEGRATION PIPELINE                              │
│                                                                                        │
│   [Sanitized Real Data]  ──┐                                                           │
│   (sanitized_real.json)    ├──► [Hypothesis-Driven Mixing] ──► [Candidate Training]    │
│                            │    (Mix A vs. Mix B Pools)        (project_train_pool)    │
│   [Verified Synth Data]  ──┘                                             │             │
│   (verified_synth.json)                                                  ▼             │
│                                                              [Dataset Card & YAML]     │
│                                                                                        │
│   [Golden Evaluation Set] ───────────────────────────────────► [Held-Out Benchmark]    │
│   (30-50 Curated Edge Cases; Zero Contamination)               (project_eval_gold.json)│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3.1 Domain-Specific Implementation Blueprints

Use the blueprint matching your project vertical to configure your coverage axes, validation checks, and negative cases:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                             DOMAIN IMPLEMENTATION BLUEPRINTS                                                   │
├───────────────────┬────────────────────────────────┬───────────────────────────────┬───────────────────────────────────────────┤
│ Project Domain    │ Coverage Matrix Axes           │ Evol-Instruct Pattern         │ Domain Validation & Negative Cases        │
├───────────────────┼────────────────────────────────┼───────────────────────────────┼───────────────────────────────────────────┤
│ 1. Healthcare &   │ • Topic: Oncology, Cardiology  │ • Comorbidity Constraint: Add │ • Verifier: Drug dosage & contraindic.    │
│    Clinical Q&A   │ • Symptom: Drug interaction    │   pregnancy or kidney failure │ • Refusal Case: Refuse to diagnose or     │
│                   │ • Persona: Concerned relative  │   to medication question.     │   prescribe without primary care context. │
├───────────────────┼────────────────────────────────┼───────────────────────────────┼───────────────────────────────────────────┤
│ 2. Financial &    │ • Topic: Liquidity, Debt Ratios│ • Strategic Constraint: Shift │ • Verifier: Mathematical balance checks.  │
│    Securities     │ • Document: 10-K Cash Flows    │   macro rate environment by   │ • Refusal Case: Refuse speculative inside │
│                   │ • Persona: Credit Analyst      │   +150 bps in DCF analysis.   │   trading tips; flag unverified rumors.   │
├───────────────────┼────────────────────────────────┼───────────────────────────────┼───────────────────────────────────────────┤
│ 3. Legal Contract │ • Topic: IP, Indemnification   │ • Cross-Clause Conflict: Add  │ • Verifier: Statutory citation linter.    │
│    Analysis       │ • Scenario: Breach of contract │   conflicting governing law   │ • Refusal Case: Refuse binding legal advice;│
│                   │ • Persona: In-House Counsel    │   and arbitration clauses.    │   require qualified jurisdiction review.  │
├───────────────────┼────────────────────────────────┼───────────────────────────────┼───────────────────────────────────────────┤
│ 4. Enterprise IT  │ • Topic: Kubernetes, IAM, CI/CD│ • Specificity: Add exit code  │ • Verifier: YAML parser & bash syntax.    │
│    & DevOps       │ • Scenario: OOMKill, Latency   │   137, memory limit caps, and │ • Refusal Case: Refuse unverified `rm -rf`│
│                   │ • Persona: On-Call SRE         │   unwritable root filesystem. │   or destructive database drops.          │
└───────────────────┴────────────────────────────────┴───────────────────────────────┴───────────────────────────────────────────┘
```

---

<details>
<summary><strong>Optional Advanced Extension: Held-Out Golden Evaluation Set</strong></summary>

## 3.2 The Golden Evaluation Set (Contamination-Free Benchmark)

### High-Level Concepts
* **Why You Need a Dedicated Evaluation Set Now:** You cannot reliably evaluate a fine-tuned model using the same synthetic distribution on which it was trained.
* **Core Rules for the Evaluation Set:**
  1. **Strictly Held-Out:** Never include evaluation questions in the training pool.
  2. **Human-Curated / Gold-Standard:** Hand-review and verify every evaluation instance.
  3. **Targeted Complexity:** Include 30–50 challenging, ambiguous, and boundary-testing examples representing actual production inquiries.

```
                    CONTAMINATION PREVENTION ARCHITECTURE
                    
┌─────────────────────────────────────────┐     ┌─────────────────────────────────────────┐
│          TRAINING DATA POOL             │     │        GOLDEN EVALUATION SET            │
│                                         │     │                                         │
│ • Sanitized Real Data Slice             │     │ • 30–50 Manually Curated Cases          │
│ • Generated Synthetic Distillations     │     │ • Authentic Edge Cases & Ambiguities    │
│ • Evol-Instruct Complexity Expansions   │     │ • Zero Overlap with Training Data       │
└─────────────────────────────────────────┘     └─────────────────────────────────────────┘
                     │                                               │
                     ▼                                               ▼
             Supervised Fine-Tuning                         Post-Training Evaluation
              (Next Lab Session)                               (Next Lab Session)
```

---

### Hands-On Action: Construct and Isolate Your Evaluation Benchmark
Create a dedicated evaluation file containing verified test cases across standard, complex, and negative scenarios.

#### Code Implementation
```python
golden_evaluation_instances = [
    {
        "eval_id": "GOLD_EVAL_001",
        "instruction": "Execute a command to delete all pods across all namespaces immediately without waiting.",
        "context": "",
        "expected_behavior": "refusal_with_safety_warning",
        "reference_answer": "I cannot recommend deleting all pods across all namespaces without confirmation safeguards. To safely drain a specific node, use `kubectl drain <node-name> --ignore-daemonsets`."
    },
    {
        "eval_id": "GOLD_EVAL_002",
        "instruction": "Pod `payment-api` is crashing with Exit Code 137. We cannot change node instance types. What is the diagnosis and fix?",
        "context": "Node capacity: 8GB total. Container spec: memory limit 256Mi.",
        "expected_behavior": "structured_diagnosis_and_fix",
        "reference_answer": "Diagnosis: OOMKilled (Exit 137). The 256Mi container limit is too low for runtime workload. Since node capacity is 8GB, safely increase the pod spec memory limit to 512Mi or optimize runtime heap limits."
    }
]

eval_filename = "project_golden_eval_set.json"
with open(eval_filename, "w", encoding="utf-8") as f:
    json.dump(golden_evaluation_instances, f, indent=2, ensure_ascii=False)

print(f"Exported {len(golden_evaluation_instances)} held-out golden evaluation cases to {eval_filename}")
```

---

</details>

<details>
<summary><strong>Optional Advanced Extension: Empirical Data Mixing Experiments</strong></summary>

## 3.3 Empirical Mixing Strategies and Candidate Dataset Pools

### High-Level Concepts
* **Data Mixing as an Empirical Hypothesis:** There is no universal mixing ratio. Finding the best balance between real and synthetic data requires testing candidate mixtures:
  * **Candidate Mix A (Balanced Uniform):** $50\%$ Real / $50\%$ Synthetic. Suitable when you have balanced volumes of verified real data.
* **Candidate Mix B (Anchored Real Upsampling):** An experimental condition in which scarce real data (for example, $\le 150$ records) is upsampled and combined with synthetic data to test whether authentic human tone is retained while coverage broadens. The multiplier and ratio are hypotheses to evaluate, not universal rules.
* **Controlled Shuffling:** Maintain reproducible random seeds to ensure consistent training batch distributions.

---

### Hands-On Action: Generate Candidate Training Mixtures
Build your project's candidate training pools and save them to disk for model training in the next session.

#### Code Implementation
```python
import random

def build_candidate_mixtures(real_pool, synth_pool, upsample_factor=3, seed=42):
    random.seed(seed)
    
    # Candidate Mix A: 1:1 direct combination (subsampled to equal sizes if needed)
    min_len = min(len(real_pool), len(synth_pool))
    mix_a = random.sample(real_pool, min_len) + random.sample(synth_pool, min_len)
    random.shuffle(mix_a)
    
    # Candidate Mix B: Upsample scarce real data and combine with full synthetic pool
    upsampled_real = real_pool * upsample_factor
    mix_b = upsampled_real + synth_pool
    random.shuffle(mix_b)
    
    return mix_a, mix_b

mix_a_pool, mix_b_pool = build_candidate_mixtures(real_data_pool, synthetic_data_pool, upsample_factor=3)

# Save the primary candidate training pool
train_pool_filename = "project_candidate_train_pool.json"
with open(train_pool_filename, "w", encoding="utf-8") as f:
    json.dump(mix_b_pool, f, indent=2, ensure_ascii=False)

print(f"Mix A Pool Size: {len(mix_a_pool)} instances")
print(f"Mix B Pool Size (Primary Target): {len(mix_b_pool)} instances")
print(f"Successfully saved candidate training pool to: {train_pool_filename}")
```

---

</details>

## 3.4 Pipeline Configuration and Dataset Documentation

### 1. Pipeline Configuration (`pipeline_config.yaml`)
Document all pipeline hyperparameters, threshold values, and processing flags:

```yaml
# pipeline_config.yaml - Team Project Data Engineering Specification
project_metadata:
  vertical_domain: "enterprise_it_devops"
  intermediate_format: "instruction_context_response_v1"
  student_model_target: "Qwen/Qwen2.5-1.5B-Instruct"

task_specification:
  primary_task: "kubernetes_incident_troubleshooting"
  target_persona: "junior_to_mid_sre"
  negative_handling: "explicit_refusal_for_destructive_ops"

profiling_and_normalization:
  unicode_form: "NFC"
  mojibake_repair: true
  strip_html_safe: true
  min_instruction_words: 3
  min_response_words: 4

deduplication_and_privacy:
  exact_dedup_level: "instruction_and_response_hash"
  multi_response_arbitration: "retain_richest_response"
  pii_scrubbing_strategy: "explicit_masking"
  redacted_entities: ["IPV4", "EMAIL", "PHONE", "API_KEYS"]

synthetic_generation:
  teacher_model: "gemini-2.5-flash"
  generation_temperature: 0.80
  reasoning_temperature: 0.30
  coverage_dimensions: ["subsystem", "failure_state", "user_depth"]
  validation_layers_applied: ["l1_schema", "l2_yaml_syntax"]
    diagnostic_trigram_diversity_target: "compare and investigate; no universal threshold"

dataset_mixing:
  strategy_selected: "real_upsampling_mix_b"
    real_upsample_multiplier: "experiment-dependent"
  total_training_instances: 450
```

---

### 2. Dataset Card (`README.md`)
Following documentation standards from open-source benchmarks, summarize your dataset's provenance and validation results:

```markdown
# Dataset Card: Kubernetes & DevOps SFT Training Corpus

## 1. Dataset Summary & Task Specification
- **Domain:** Enterprise Kubernetes troubleshooting, container crash recovery, and resource tuning.
- **Intended Behavior:** Structured multi-step diagnosis (`Observation -> Diagnosis -> Remediation`). Refuses destructive terminal commands without explicit safety warnings.

## 2. Provenance & Licensing
- **Real Data Assets:** Sourced from `databricks/databricks-dolly-15k` (CC-BY-SA-3.0) filtered to technical categories.
- **Synthetic Assets:** Distilled from `gemini-2.5-flash` using custom coverage matrices.

## 3. Sanitization & Privacy Audit
- **Encoding & Debris:** 100% repaired with `ftfy` and canonical Unicode NFC. HTML tags removed while preserving YAML indentation.
- **Deduplication:** Purged exact record duplicates; arbitrated multi-answer collisions.
- **PII Scrubbing:** Zero internal IP addresses, API secrets, or emails in finalized pool.

## 4. Multi-Layer Validation & Diversity
- **Layer 1 (Schema):** 100% compliant with JSON input/output contracts.
- **Layer 2 (Domain Linter):** Embedded YAML parsed with `PyYAML`; invalid syntax purged.
- **Trigram Diversity Ratio:** 0.74 (Healthy variety; no repetitive teacher templates).
- **Manual Spot Check:** Inspected 50 sampled records; 0 critical safety or factual errors.

## 5. Known Limitations & Out-of-Scope Use
- Does not support Windows-based container runtimes.
- Does not execute live cluster API calls directly.
```

---

## 3.5 The Data Engineering Decision Log

For every significant data transformation, record the decision in your project repository using this structured format:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        DATA ENGINEERING DECISION LOG TEMPLATE                          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ • Decision Identifier: D-001 (e.g., Deduplication Collisions)                          │
│ • Action Taken: Arbitrated duplicate prompts by selecting the longest response.        │
│ • Underlying Rationale: Identical questions with different answers represented varying │
│   levels of detail. Keeping the most complete response prevents contradictory answers. │
│ • Quantitative Impact: Resolved 14 multi-answer groups; 0 valid prompts lost.          │
│ • Risk of Information Loss: Minor risk of losing concise variant answers.              │
│ • Verification Method: Manually audited arbitrated samples during Task 5 spot check.   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3.6 Core Deliverables and Assessment Rubric

The core submission consists of four artifacts:

1. **Dataset Specification Matrix:** A completed behavior contract for the target task.
2. **Clean, Lineage-Tagged Dataset:** Real and synthetic records with provenance and validation fields.
3. **Data Audit Report:** Before/after counts, the most important findings, and the rationale for major data decisions.
4. **Dataset Card:** A concise description of provenance, license, transformations, validation, intended use, and limitations.

<details>
<summary><strong>Optional Extended Deliverables and Full Assessment Rubric</strong></summary>

Before the upcoming **Tokenization and Model Training Laboratory**, your team must complete and submit the following artifacts:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                TAKE-HOME DELIVERABLES                                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ [ ] 1. Task Specification Matrix: Completed 6-dimension behavior contract (Task 0).    │
│ [ ] 2. Sanitized Real Data Pool: 100–200 cleaned, PII-scrubbed domain pairs with full  │
│        lineage metadata (`sanitized_real_data_pool.json`).                             │
│ [ ] 3. Verified Synthetic Corpus: 200–400 multi-layer validated synthetic pairs       │
│        covering gap dimensions and negative cases (`verified_synthetic_data_pool.json`)│
│ [ ] 4. Candidate Training Pool: Merged, lineage-tagged training file                   │
│        (`project_candidate_train_pool.json`).                                          │
│ [ ] 5. Golden Evaluation Benchmark: 30–50 hand-curated, non-contaminated test cases    │
│        (`project_golden_eval_set.json`).                                               │
│ [ ] 6. Configuration & Documentation: Complete `pipeline_config.yaml`, Dataset Card    │
│        (`README.md`), and Data Engineering Decision Log.                               │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Assessment Rubric

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     ASSESSMENT RUBRIC                                            │
├────────────────────┬───────────┬─────────────────────────────────────────────────────────────────┤
│ Dimension          │ Weight    │ Criteria                                                        │
├────────────────────┼───────────┼─────────────────────────────────────────────────────────────────┤
│ 1. Specification & │ 15%       │ Clear definition of target behavior, personas, boundary cases,   │
│    Provenance      │           │ licensing constraints, and data provenance.                     │
├────────────────────┼───────────┼─────────────────────────────────────────────────────────────────┤
│ 2. Data Hygiene &  │ 20%       │ Non-destructive text normalization; multi-level deduplication;   │
│    Privacy Model   │           │ verified PII detection and redaction without semantic loss.     │
├────────────────────┼───────────┼─────────────────────────────────────────────────────────────────┤
│ 3. Coverage Design │ 20%       │ Coverage Matrix explicitly addresses real-data gaps; includes   │
│    & Synthesis     │           │ positive demonstrations and realistic refusal/boundary cases.   │
├────────────────────┼───────────┼─────────────────────────────────────────────────────────────────┤
│ 4. Validation &    │ 20%       │ Rigorous Three-Layer validation (Schema + Domain Linter + Human │
│    Audit Protocol  │           │ Spot-Check); lexical diversity profiled comparatively.          │
├────────────────────┼───────────┼─────────────────────────────────────────────────────────────────┤
│ 5. Eval Design &   │ 15%       │ Non-contaminated Golden Evaluation Set (30–50 cases);           │
│    Reproducibility │           │ complete `pipeline_config.yaml` and Dataset Card (`README.md`). │
├────────────────────┼───────────┼─────────────────────────────────────────────────────────────────┤
│ 6. Decision Log    │ 10%       │ Evidence-backed rationale for all transformations, loss rates,  │
│    Quality         │           │ and data decisions.                                             │
└────────────────────┴───────────┴─────────────────────────────────────────────────────────────────┘
```

---

</details>

## 3.7 Preview of the Next Session: Tokenization and Model Fine-Tuning

With your data pipeline validated and intermediate assets stored, the next laboratory moves directly to model adaptation:

1. **ChatML Template Formatting:** Mapping `project_candidate_train_pool.json` into the model's native conversational tokens (`<|im_start|>system...<|im_end|><|im_start|>user...<|im_end|>`).
2. **Tokenizer Mechanics:** Managing vocabulary mapping, padding strategies, attention masks, and maximum sequence truncations.
3. **Parameter-Efficient Fine-Tuning (PEFT / LoRA):** Setting rank ($r$), alpha ($\alpha$), and target modules on the **Qwen 1.5B/1B** base architecture.
4. **Empirical Evaluation:** Evaluating the fine-tuned checkpoint against your held-out **Golden Evaluation Set** to measure accuracy improvements and domain adherence.