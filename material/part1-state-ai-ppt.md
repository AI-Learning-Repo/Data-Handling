# From Data to AI Agents: Why This 3-Course Package Matters

## Slide 1 — AI Engineering Today: From AGI Vision to Practical Systems

### The long-term vision is broad. Today's systems are usually specialized.

* **AGI:** A long-term vision of systems capable of broad, human-level reasoning across domains.
* **Today's reality:** Most deployed AI systems are **specialized systems** designed for particular tasks and domains.
* **Engineering challenge:** Build the **simplest sufficient system** for the problem—not necessarily the largest model available.

**Key message:**

> AI engineering today is less about building one model that does everything and more about building the right system for a specific problem.

---

## Slide 2 — Real-World Constraints Change the Choice of Model

### Capability is only one part of the decision.

* **Compute:** Large open-weight models can require substantial GPU memory, infrastructure, and energy to operate.
* **Cost:** Cloud APIs can become expensive as usage grows, particularly with long conversations, retrieval, and repeated agent interactions.
* **Efficiency:** Quantization techniques such as **GGUF** and **4-bit AWQ** make smaller models practical on more accessible hardware.
* **Control:** Local or private deployment can provide greater control over costs, data, and the operating environment.

**Key message:**

> The best model is not necessarily the biggest model. It is the model that meets the requirements of the application.

---

## Slide 3 — Privacy, Governance, and Control Matter Too

### AI systems also have to fit their operating environment.

* **Regulation:** AI systems increasingly need to meet requirements related to transparency, safety, and accountability.
* **Privacy:** Organizations working with sensitive information may need to limit how data is sent to external AI services.
* **Customization:** Some applications benefit from models adapted to a specific domain, vocabulary, or task.
* **Output control:** Watermarking and other provider-controlled mechanisms can also influence how organizations think about model ownership, provenance, and control.

**Key message:**

> For some applications, greater control over the model and its environment can be as important as model capability.

---

## Slide 4 — Start With the Problem, Not the Model

### Different problems require different AI architectures.

```text
                    What is the problem?
                             │
            ┌────────────────┴────────────────┐
            │                                 │
     Structured data                   Text / language
            │                                 │
     Classical ML                      What capability is needed?
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │                         │                         │
             Prediction               Semantic search          Knowledge access
                    │                         │                         │
                ML / NLP                  Embeddings                   RAG
                                                                        │
                                                              Does the model need
                                                                  adaptation?
                                                                        │
                                                                   Fine-tuning
                                                                        │
                                                              Are local / resource
                                                              constraints important?
                                                                        │
                                                                       SLM
```

### Important:

**RAG, fine-tuning, model size, and deployment location are separate decisions.**

A system can use:

* RAG without fine-tuning
* Fine-tuning without RAG
* A large model with RAG
* A small model without fine-tuning
* A local model for privacy or cost reasons

**Key message:**

> AI engineering is about selecting and combining the right components for the problem.

---

## Slide 5 — What Does This Mean for Students?

### AI engineering requires more than prompt engineering.

To make these decisions, three layers of knowledge are needed:

```text
        DATA
          │
          ▼
       MODELS
          │
          ▼
       SYSTEMS
```

### The 3-course package follows the same progression:

|       | Course                           | Main question                                                 |
| ----- | -------------------------------- | ------------------------------------------------------------- |
| **1** | Data Handling & Machine Learning | **How do we work with data and build reliable ML solutions?** |
| **2** | Neural Networks                  | **How do modern neural networks work?**                       |
| **3** | Neural Network Project           | **How do we turn models into useful AI systems?**             |

**15 ECTS across two periods**

---

## Slide 6 — Period 1: Build the Foundation

### Course 1 — Data Handling and Machine Learning

**5 ECTS**

**Students work with:**

* Data preparation and cleaning
* Feature transformation
* Machine-learning models
* Evaluation and validation
* Data pipelines

**Why first?**

> Good AI systems start with good data.

The concepts developed here provide the foundation for later work with neural networks, retrieval, and model adaptation.

---

## Slide 7 — Period 2: Understand the Models

### Course 2 — Neural Networks

**5 ECTS**

**Students learn:**

* Neural-network fundamentals
* Optimization and loss functions
* Backpropagation
* Attention mechanisms
* Transformer architectures

**Why it matters:**

> Understanding how models work makes it easier to reason about their behavior, limitations, optimization, and failure modes.

---

## Slide 8 — Period 2: Build the Systems

### Course 3 — Neural Network Project

**5 ECTS**

**Students apply their knowledge to:**

* RAG systems
* Parameter-efficient fine-tuning (LoRA / QLoRA)
* Model quantization
* Local model deployment
* Tool and API integration
* AI agents

```text
                     AI SYSTEM
                         │
                ┌────────┴────────┐
                │                 │
              Model             Tools
                │                 │
            RAG / FT       APIs / Functions
                │                 │
                └────────┬────────┘
                         │
                       Agent
                         │
                    AI Assistant
```

**Why it matters:**

> A useful AI assistant is not just a neural network. It is a system that combines models, data, retrieval, tools, memory, and evaluation.

---

## Slide 9 — The Learning Path

### From data to complete AI systems

```text
       DATA
         │
         ▼
   MACHINE LEARNING
         │
         ▼
  NEURAL NETWORKS
         │
         ▼
 MODEL ADAPTATION
         │
         ▼
   RAG / QUANTIZATION
         │
         ▼
    AGENTS & TOOLS
         │
         ▼
     AI SYSTEMS
```

### The progression is simple:

**Work with data → Understand models → Build systems**

By the end of the package, students have the foundations needed to evaluate different AI approaches and make informed engineering decisions as the technology continues to evolve.
