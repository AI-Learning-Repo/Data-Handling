# Theory: Principles of Training Data Engineering for Language Model Specialization

---

## Overview

Adapting a general-purpose language model to a specialized professional domain requires more than adjusting hyperparameters or running training scripts. The success of parameter-efficient fine-tuning depends primarily on the **relevance, structural consistency, and accuracy of the underlying dataset**.

This theoretical companion explains the concepts behind the lab's required workflow: **Specify -> Inspect -> Clean -> Protect -> Audit -> Identify gaps -> Generate a small batch -> Verify -> Document.** It also preserves advanced techniques for later study; these are useful when a project's scale or risk justifies them, but they are not prerequisites for completing the introductory lab.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        DATA-CENTRIC ENGINEERING LIFECYCLE                              │
│                                                                                        │
│ [1. Specify] ──► [2. Inspect] ──► [3. Clean & Protect] ──► [4. Audit]                  │
│ Define boundaries   Profile provenance         Preserve semantics / redact PII          │
│                                                               │                        │
│                                                               ▼                        │
│ [7. Document] ◄── [6. Verify] ◄── [5. Coverage & Small-Batch Generation]               │
│ Lineage and decisions  Programmatic / human    Fill named behavioral gaps               │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Module 1: Foundations of Supervised Fine-Tuning & Model Sensitivity

### 1.1 Pre-Training vs. Supervised Fine-Tuning (SFT)
Large Language Models (LLMs) are trained in two distinct phases:

1. **Pre-Training (Knowledge Acquisition):** Models process trillions of tokens of unstructured text using an autoregressive next-token prediction objective. The model learns grammar, world knowledge, reasoning patterns, and statistical distributions of language.
2. **Supervised Fine-Tuning (Behavioral Adaptation):** Models are trained on curated pairs of prompts and demonstrations (instruction-response pairs). SFT does not primarily teach the model entirely new languages or foundational facts; rather, it **adapts the model’s tone, format, constraints, and operational behavior** to align with specific tasks.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ PRE-TRAINING: Next-token prediction over massive text corpora (Knowledge Base)          │
│ "The capital of France is [Paris]..."                                                  │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ SUPERVISED FINE-TUNING: Structured demonstration learning (Behavioral Alignment)       │
│ User: "Extract all server hostnames from this log: ..."                                │
│ Assistant: "['srv-auth-01', 'srv-db-02']"                                              │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 The Sensitivity of Smaller Parameter Models (1B–3B)
Smaller architectures (such as **Qwen 1.5B/1B**) offer practical advantages: they run on commodity hardware, have low inference latency, and are cost-effective to deploy. However, their reduced parameter capacity changes how they respond to training data:

* **Lower Capacity for Noise Absorption:** A 70-billion-parameter model has substantial internal redundancy, allowing it to generalize despite occasional formatting errors or noisy labels in the training set. A 1B model has less capacity to compensate for inconsistencies and tends to learn spurious correlations from noisy data.
* **Rapid Memorization:** If training instances contain exact phrase duplicates, smaller models tend to memorize specific token sequences rather than abstracting the underlying task logic.
* **Strict Formatting Requirements:** Small models depend on consistent input-output schemas. Variations in prompt markers, system prompt phrasing, or JSON layouts increase the risk of output formatting failures during deployment.

---

## Module 2: The Dataset Specification & Task Alignment

Before collecting or modifying data, you must define the operational contract for the target system. 

```
                    ┌─────────────────────────────────────────┐
                    │          DATASET SPECIFICATION          │
                    └────────────────────┬────────────────────┘
                                         │
     ┌───────────────────┬───────────────┴───────────────┬───────────────────┐
     ▼                   ▼                               ▼                   ▼
┌──────────────┐  ┌──────────────┐              ┌─────────────────┐  ┌──────────────┐
│  Target Task │  │ User Persona │              │ Ideal Response  │  │ Out-of-Scope │
│  & Inputs    │  │  & Tone      │              │ Profile & Schema│  │ & Refusals   │
└──────────────┘  └──────────────┘              └─────────────────┘  └──────────────┘
```

### 2.1 The Components of a Dataset Specification
A formal specification prevents teams from collecting data that is technically clean but misaligned with production goals:

1. **Input Distribution:** What exact data types will the model encounter? (e.g., raw stack traces, markdown tables, customer queries).
2. **User Persona & Depth:** Is the model speaking to an end customer, a junior engineer, or an automated parser?
3. **Acceptance Criteria:** What structural elements define an optimal response? (e.g., direct answers first, followed by reproducible code blocks).
4. **Boundary & Refusal Definitions:** How must the model respond when presented with incomplete, impossible, or dangerous inputs?

### 2.2 The Importance of Negative and Refusal Examples
Models trained exclusively on successful query-response pairs often develop a bias toward compliance: they attempt to answer even when the request is impossible, nonsensical, or unsafe.

To prevent hallucinations, a well-curated dataset must include **negative demonstrations**:
* **Ambiguous Inputs:** Teaching the model to ask clarifying questions rather than guessing.
* **Missing Information:** Training the model to state what additional parameters are needed.
* **Unsafe or Out-of-Scope Requests:** Demonstrating polite, clear refusals for unsupported or destructive actions.

---

## Module 3: Real-World Data Sanitization Principles

Data cleaning is governed by a core engineering heuristic:

$$\Large\textbf{Preserve Semantics; Remove Artifacts.}$$

Aggressive, indiscriminate cleaning scripts can strip away necessary technical syntax. For example, stripping angle brackets (`<`, `>`) breaks code comparisons, and removing backticks destroys markdown syntax highlighting.

```
Raw Input:      "<p>Check if `x < 10` before proceeding.</p>"
                                │
                                ▼
Bad Cleaning:   "Check if x  10 before proceeding."     (Stripped all brackets; corrupted meaning)
                                │
                                ▼
Safe Cleaning:  "Check if `x < 10` before proceeding."   (Stripped HTML tags; preserved math/code)
```

### 3.1 Encoding Mechanics & Unicode Normalization
Text sourced from heterogeneous environments (web scrapes, legacy logs, exported databases) frequently contains character encoding anomalies:

* **Mojibake:** Occurs when bytes written in one character encoding (e.g., Windows-1252) are decoded using another (e.g., UTF-8). For example, the right single quotation mark `’` (bytes `0xE2 0x80 0x99` in UTF-8) may be misread as `â€™`. Automated libraries like `ftfy` use heuristic byte analysis to restore the original characters.
* **Unicode Canonical Equivalence (NFC):** In Unicode, certain glyphs can be represented in multiple ways. For instance, the character `é` can be stored as a single code point (`U+00E9`, Latin Small Letter E with Acute) or as a composite pair (`U+0065` [Letter E] + `U+0301` [Combining Acute Accent]). 

```
Composed Form (NFC):      [ U+00E9 ]        ──► Tokenizer sees: Token ID 4125
Decomposed Form (NFD):    [ U+0065, U+0301 ] ──► Tokenizer sees: Token ID 101, Token ID 782
```

Although both render identically to the human eye, a model's tokenizer treats them as distinct token sequences. Applying **Unicode Normalization Form C (NFC)** standardizes all characters into canonical composed forms, ensuring consistent tokenization.

---

### 3.2 Profiling-Driven Length Diagnostics
Static character filters (such as deleting any string with fewer than 20 characters) risk removing valid data while letting problematic entries through:

| Metric | Diagnostic Purpose | What Outliers Usually Indicate |
| :--- | :--- | :--- |
| **Instruction Word Count** | Measures prompt complexity | Extremely low word counts ($\le 2$ words) may be incomplete prompts or button labels; high word counts may be unparsed context dumps. |
| **Response Word Count** | Measures answer completeness | Extremely short responses may indicate truncated generation or unhelpful answers (e.g., *"Yes"*); long responses may exceed context windows. |
| **Instruction-to-Response Ratio** | Identifies imbalance | Very long instructions with 1-word responses often signal misaligned classification tasks mixed into QA datasets. |

Rather than applying rigid deletion thresholds, use length profiling to **flag suspicious records for inspection**.

---

### 3.3 Multi-Level Deduplication Mechanics
Deduplication is not a simple binary check. Real-world datasets contain different categories of duplication that require different handling:

```
                                  DEDUPLICATION TAXONOMY
                                             │
             ┌───────────────────────────────┼───────────────────────────────┐
             ▼                               ▼                               ▼
   [Exact Record Duplicate]       [Duplicate Instructions]           [Near-Duplicates]
   Identical prompt AND answer.   Identical prompt, different ans.   Paraphrased versions.
   Action: Purge exact copies.    Action: Group & arbitrate.         Action: Profile diversity.
```

1. **Exact Record Duplicates:** Both the prompt and the response match an existing entry character-for-character. Retaining exact duplicates causes smaller models to overfit to those specific token sequences.
2. **Duplicate Instructions (1-to-Many Collisions):** The exact same prompt appears multiple times with different responses. This often happens when combining datasets from multiple sources. **Do not automatically delete these.** They may represent complementary perspectives or direct factual contradictions. Flag the group for manual review and record the final decision; response length alone is not a quality criterion.
3. **Near-Duplicates (Paraphrasing):** Phrasings differ slightly (*"How do I restart a pod?"* vs. *"What is the command to restart a pod?"*). While minor phrasing variations can help generalization, near-identical duplicate blocks add training overhead without improving coverage.

---

## Module 4: Privacy Engineering & Data Governance

### 4.1 Neural Network Memorization
Neural language models are capable of memorizing verbatim sequences from their training sets, particularly when those sequences are repeated or associated with rare tokens. If unredacted data contains Personally Identifiable Information (PII), proprietary source code, internal hostnames, or API credentials, the fine-tuned model may emit those secrets during standard inference.

```
Training Data Leak:   "Internal proxy configuration for staging: auth_token=secret_xyz_88192"
                                                │ (Model fine-tuned on unredacted data)
                                                ▼
Inference Prompt:     "How do I configure the staging proxy?"
Model Output:         "Use configuration with auth_token=secret_xyz_88192..." (Security Incident)
```

### 4.2 The Detect $\rightarrow$ Transform $\rightarrow$ Verify Model
Privacy engineering follows a three-stage lifecycle:

```
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│       1. DETECT         │     │      2. TRANSFORM       │     │       3. VERIFY         │
│ Regex: Structured items │     │ Redaction: Explicit tag │     │ Secondary scan to catch │
│ (IPs, emails, keys)     │ ──► │ Replacement: Fake names │ ──► │ unmasked entities or    │
│ Optional: NER for names │     │ Generalize: Coarse bins │     │ malformed placeholders. │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
```

1. **Detection:** In the core lab, use high-precision regular expressions for structured entities such as IP addresses, email addresses, phone numbers, and secret keys.
2. **Transformation:**
   * *Placeholder Masking (`[EMAIL_REDACTED]`):* Secure and explicit, but can disrupt natural sentence flow if overused.
   * *Synthetic Surrogate Replacement:* Replacing real names with synthetic equivalents (e.g., using `Faker`). This preserves natural language structure while removing the original identity.
   * *Generalization:* Converting specific values into broader categories (e.g., converting a specific date to a year or quarter).
3. **Verification:** Transformed outputs must pass through a secondary validation scan to confirm that no raw patterns remain unmasked.

<details>
<summary><strong>Optional Advanced Extension: Semantic PII Detection</strong></summary>

Unstructured entities such as names, physical locations, and organization names may require Named Entity Recognition (NER) models such as Microsoft Presidio or spaCy. Synthetic surrogate replacement can preserve natural language while removing original identities, but requires additional verification.

</details>

---

## Module 5: Synthetic Data Generation & Distillation Mechanics

### 5.1 Teacher-Student Distillation
Distillation uses a high-capacity foundation model (the **Teacher**, such as Gemini 2.5 Flash) to generate demonstrations that are then used to train a smaller model (the **Student**, such as Qwen 1.5B). This process captures the teacher's domain knowledge and formatting discipline into a compact, cost-effective student model.

```
┌─────────────────────────────────────────┐
│     TEACHER MODEL (Gemini 2.5 Flash)    │
│  High parameter count, rich reasoning   │
└────────────────────┬────────────────────┘
                     │  Generates domain-grounded demonstrations
                     ▼
┌─────────────────────────────────────────┐
│        VERIFIED TRAINING DATASET        │
│  Filtered, linted, and audited examples │
└────────────────────┬────────────────────┘
                     │  Supervised fine-tuning loop
                     ▼
┌─────────────────────────────────────────┐
│       STUDENT MODEL (Qwen 1.5B/1B)      │
│  Fast, private, low-cost local model    │
└─────────────────────────────────────────┘
```

### 5.2 Why Naive Generation Fails: The Mode Collapse Problem
Prompting an LLM with generic requests (e.g., *"Generate 100 customer support questions"*) results in **mode collapse**: the model draws repeatedly from its highest-probability paths, producing variations of the same handful of common scenarios.

```
Generic Prompt: "Generate 50 DevOps questions"
Output Cluster: 
 - "How do I list pods in Kubernetes?" (x12)
 - "How do I check logs in Kubernetes?" (x15)
 - "How do I describe a service?" (x11)
Result: High token volume, near-zero coverage of complex scenarios or edge cases.
```

### 5.3 Coverage Matrices: Systematic Gap-Filling
To ensure broad coverage across the problem space, construct a multidimensional **Coverage Matrix**. By sampling across orthogonal dimensions, you prompt the teacher to generate examples throughout the domain's operational space.

```
                    MULTIDIMENSIONAL SAMPLING ARCHITECTURE
                    
  Dimension A (Domain Subsystem):  [ Storage ]   [ Ingress ]   [ Compute ]   [ IAM ]
                                        │             │             │           │
                                        ▼             ▼             ▼           ▼
  Dimension B (Failure Mode):      [ Timeout ]   [ Cert Exp ]  [ OOMKill ]   [ 403 Forbidden ]
                                        │             │             │           │
                                        ▼             ▼             ▼           ▼
  Dimension C (User Experience):   [ Junior ]    [ Automated ] [ On-Call ]   [ Auditor ]
                                        │             │             │           │
                                        └─────────────┼─────────────┘           │
                                                      ▼                         ▼
                                        Systematic Seed Prompts        Broad Domain Coverage
```

---

<details>
<summary><strong>Optional Advanced Extension: Complexity Evolution and Structured Reasoning</strong></summary>

### 5.4 Complexity Evolution (Evol-Instruct) with Realism Constraints
Evol-Instruct is a technique for progressively increasing the difficulty and depth of simple seed prompts. However, making a prompt more complex only improves training data if the complexity is **operationally realistic**.

```
┌────────────────────────────────────────────────────────────────────────┐
│ BASE SEED PROMPT:                                                      │
│ "How do I fix a pod stuck in CrashLoopBackOff?"                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ UNCONSTRAINED / ARTIFICIAL EVOLUTION (Anti-Pattern):                   │
│ "Explain how a 17th-century sailor would fix a CrashLoopBackOff while   │
│ translating every third word into Latin."                              │
│ Result: Complex, but useless for production training.                  │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ REALISM-CONSTRAINED EVOLUTION (Best Practice):                         │
│ "Pod `auth-api` is crashing with Exit Code 137 under memory limit caps.│
│ Node resources cannot be scaled. How do you isolate the memory leak    │
│ and configure JVM heap settings to resolve it?"                        │
│ Result: High cognitive demand reflecting real operational constraints. │
└────────────────────────────────────────────────────────────────────────┘
```

### 5.5 Supervision Formats: Direct Answers vs. Structured Reasoning
Depending on your target task, choose the appropriate supervision schema:

1. **Direct Supervision:** Clean, direct mappings from instruction to output. Best for lookup queries, translation, and structured API format conversion.
2. **Structured Reasoning Supervision:** Decomposes complex diagnostics into clear sections:
   $$\text{Observation \& Diagnosis} \longrightarrow \text{Constraint Analysis} \longrightarrow \text{Actionable Remediation}$$
   This trains the student model to reason methodically through technical problems without drifting into unverified internal monologues.

</details>

---

## Module 6: Quality Validation & Diagnostic Metrics

<details>
<summary><strong>Optional Advanced Extension: LLM-as-a-Judge</strong></summary>

### 6.1 The Circular Validation Trap
A common mistake in synthetic data generation is relying entirely on another LLM as an automated judge without external verification. If a teacher model hallucinates an invalid command-line flag or incorrect mathematical derivation, an LLM judge often approves the same error because it shares similar underlying statistical priors.

```
                    THE CIRCULAR VALIDATION FAILURE MODE
                    
┌────────────────────────┐         ┌────────────────────────┐         ┌────────────────────────┐
│   Teacher Generates    │  ───►   │    LLM Judge Checks    │  ───►   │ Human Blindly Trusts   │
│  "Use `kubectl --force │         │   "Looks plausible,    │         │  Model learns invalid  │
│   --delete-all-nodes`" │         │    approved!"          │         │  syntax during SFT.    │
└────────────────────────┘         └────────────────────────┘         └────────────────────────┘
```

</details>

### 6.2 The Three-Layer Validation Architecture
To prevent hallucinations from entering your training set, use a multi-tiered validation model combining programmatic, domain-specific, and human reviews:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              THREE-LAYER VALIDATION MODEL                              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Layer 1: Programmatic Checks                                                           │
│ • Valid JSON / YAML syntax parsing                                                     │
│ • Key presence verification (`instruction`, `response`, `metadata`)                   │
│ • Word and character count sanity checks                                               │
│ • Regular expression scans for sensitive tokens                                        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Layer 2: Deterministic Domain Verification                                             │
│ • Code syntax compilation and linters (e.g., Python AST, SQL parsing, Bash linters)    │
│ • Mathematical formula balance checks                                                  │
│ • Static schema validation against official API references                             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Layer 3: Expert Human Audit ("Random 50" Protocol)                                     │
│ • Qualitative spot-checks across sampled subsets                                       │
│ • Categorization of errors into structured taxonomies                                  │
│ • Iterative refinement of upstream generation prompts                                  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

<details>
<summary><strong>Optional Advanced Extension: Lexical Diversity Diagnostics</strong></summary>

### 6.3 Lexical Diversity Diagnostics (Trigram Profiling)
To measure whether a generation pipeline is producing varied language rather than repetitive boilerplate, analyze the distribution of **word-level trigrams** (three-word sequences):

$$\text{Trigram Diversity Ratio} = \frac{\text{Count of Unique Trigrams}}{\text{Total Trigrams}}$$

```
Response A: "To resolve this issue, you must first inspect the logs."
Trigrams:   ("to", "resolve", "this"), ("resolve", "this", "issue"), ("this", "issue", "you"), ...

Response B: "To resolve this issue, you must verify the network config."
Trigrams:   ("to", "resolve", "this") [DUPLICATE], ("resolve", "this", "issue") [DUPLICATE], ...
```

* **How to use this metric:** Treat trigram diversity as a **comparative diagnostic**, not a rigid pass/fail rule. If your synthetic pool has a diversity ratio of $0.42$ while your real data baseline is $0.75$, the teacher model is likely using repetitive templates.
* **Domain Context:** Highly structured domains (e.g., formal legal disclaimers, strict SQL schemas) naturally have lower trigram diversity than conversational domains.

</details>

---

## Module 7: Data Lineage, Mixing, and Evaluation Design

### 7.1 Record-Level Lineage
Every training example should carry metadata documenting its origin, generation parameters, and verification history:

```
┌────────────────────────────────────────────────────────────────────────┐
│ RECORD LINEAGE SCHEMA                                                  │
├────────────────────────────────────────────────────────────────────────┤
│ • record_id: "SYNTH_GEMINI_a9f14b"                                     │
│ • source: "synthetic"                                                   │
│ • source_id: "seed_networking_tls_expired_03"                           │
│ • generation_method: "coverage_matrix"                                  │
│ • teacher_model: "gemini-2.5-flash"                                    │
│ • coverage_axes: {"subsystem": "Networking", "failure": "TLS Expired"} │
│ • validation_checks_passed: ["schema_json", "linter_yaml_v2"]          │
│ • human_audited: false                                                 │
└────────────────────────────────────────────────────────────────────────┘
```

Record-level lineage ensures reproducibility and allows you to isolate and remove specific data subsets if a particular generation batch turns out to be flawed.

---

<details>
<summary><strong>Optional Advanced Extension: Data Mixing Experiments</strong></summary>

### 7.2 Data Mixing Strategies
When combining real-world data and synthetic generation pools, compare candidate mixtures empirically rather than relying on fixed ratios:

```
                       CANDIDATE MIXING ARCHITECTURES
                       
      CANDIDATE MIX A: BALANCED               CANDIDATE MIX B: ANCHORED UPSAMPLING
┌──────────────────────────────────┐        ┌──────────────────────────────────┐
│ Verified Real Data (50%)         │        │ Real Data (Upsampled 3x) [30%]   │
├──────────────────────────────────┤        ├──────────────────────────────────┤
│ Verified Synthetic Data (50%)    │        │ Broad Synthetic Pool [70%]       │
└──────────────────────────────────┘        └──────────────────────────────────┘
```

* **Uniform Mixing (Mix A):** Best when you have access to a substantial pool of clean, verified real-world data.
* **Anchored Upsampling (Mix B):** When real-world data is scarce but highly authentic, upsampling it can be tested as an experimental condition while synthetic data broadens coverage. The multiplier and ratio are hypotheses to evaluate, not universal rules.

</details>

---

<details>
<summary><strong>Optional Advanced Extension: Held-Out Golden Evaluation Set</strong></summary>

### 7.3 Evaluation Design: The Golden Benchmark
Fine-tuned models should never be evaluated solely on synthetic samples generated by the training pipeline. Doing so evaluates the model on the teacher's stylistic habits rather than true task competence.

Before fine-tuning, construct a **Held-Out Golden Evaluation Set**:
1. **Zero Contamination:** Maintain strict separation between training and evaluation instances. Evaluation queries must never appear in the training pool.
2. **Hand-Curated and Verified:** Every evaluation instance should be manually reviewed for factual accuracy and clarity.
3. **Representative Difficulty:** Include standard operational tasks, complex multi-step scenarios, ambiguous requests, and invalid/destructive commands to test model boundaries.

```
┌─────────────────────────────────────────┐     ┌─────────────────────────────────────────┐
│        TRAINING POOL (Mix A or B)       │     │        GOLDEN EVALUATION SET            │
│                                         │     │                                         │
│ • Sanitized Real Data Pool              │     │ • 30–50 Hand-Curated Golden Cases       │
│ • Verified Synthetic Demonstrations     │     │ • Authentic Edge Cases & Ambiguities    │
│ • Evol-Instruct Complexity Expansions   │     │ • Zero Contamination with Training Pool │
└─────────────────────────────────────────┘     └─────────────────────────────────────────┘
                     │                                               │
                     ▼                                               ▼
          Supervised Fine-Tuning                            Empirical Evaluation
              (Next Session)                                   (Next Session)
```

</details>

---

## Summary of Core Theoretical Principles

| Concept | Common Misconception | Correct Engineering Principle |
| :--- | :--- | :--- |
| **Data Cleaning** | Remove all non-alphanumeric characters and short text. | **Preserve semantics; remove artifacts.** Protect code blocks, math operators, and domain syntax. |
| **Deduplication** | Delete any row with a duplicate question. | **Differentiate duplicate types.** Purge exact matches; flag multi-answer collisions for manual review. |
| **Privacy / PII** | A regex for email addresses guarantees compliance. | **Apply a 3-step privacy model:** Detect (core: regex; optional: NER) $\rightarrow$ Transform (Mask/Surrogate) $\rightarrow$ Verify. |
| **Synthetic Data** | Generate massive volumes using open-ended prompts. | **Target coverage gaps** using structured multidimensional Coverage Matrices. |
| **Quality Control** | An LLM judge approving an answer confirms it is correct. | **Use programmatic checks, domain checks, and human audits.** LLM judging is optional additional evidence. |
| **Documentation** | A clean text file is enough. | **Preserve lineage and decisions:** record source, source ID, transformations, and validation status for each example. |