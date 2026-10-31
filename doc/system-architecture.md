# Research Assistant Architecture

This document models the current `llm_agents` implementation. A bounded agent loop can use native tool calling from either a local Ollama model or an OpenAI GPT model. The selected provider controls whether questions/tool results stay local or are sent to a hosted API.

## System Boundary

The project has two related but distinct flows:

1. Interactive research: arXiv paper records are collected or PDFs are ingested, indexed as text, and queried through a bounded tool loop. The current local arXiv corpus has battery-degradation/cycle-life search results; the no-argument CLI demo still uses a small generic RAG sample corpus.
2. Offline evaluation: the benchmark runner uses M3SciQA annotations to measure retrieval and citation-ID matching, and asks the configured LLM to choose tools. Image-grounded QA retrieves public `evidence_reference` passages with BM25 and supplies the passages, paper IDs, question, and image to the selected vision model. The GPT validation run uses hosted **OpenAI** only; Ollama remains an optional, separate local provider.

The project does not include or have local access to the NASA PCoE battery dataset. An attempted archive download was incomplete and unusable; it is not part of the project data. Consequently, no battery lifetime regression experiment or score is claimed.

```mermaid
flowchart LR
    User[User question] --> CLI[main.py CLI]
    CLI --> Assistant[ResearchAssistant]
    Assistant --> Agent[Agent bounded loop]
    Agent --> LLM[Ollama or OpenAI tool-calling LLM]
    LLM --> Agent
    Agent --> Registry[ToolRegistry]
    Registry --> Search[Search tool]
    Registry --> Calc[Calculator tool]
    Registry --> Trend[Publication trend tool]
    Registry --> Figure[Figure inspection tool]
    Search --> Candidate[Token-overlap candidate filter]
    Candidate --> BM25[BM25 Reranker]
    Candidate -. no matches .-> Empty[Empty evidence result]
    BM25 --> Papers[ResearchPaper corpus]
    Figure --> Assets[PDF-extracted local images]
    Figure -. optional local inference .-> Ollama[Ollama vision API]
    Agent --> Trace[Tool observations / trace]
    Trace --> UserAnswer[Answer returned to CLI]

    Collector[arXiv API collector] --> PaperStore[JSONL paper store]
    PDF[Local PDF] --> Ingest[PyMuPDF ingestion]
    Ingest --> PaperStore
    Ingest --> Assets
    PaperStore --> Papers

    M3Data[Local M3SciQA JSONL + images] --> Eval[evaluate-agent / benchmark]
    M3Data --> Evidence[Public evidence_reference passages]
    Evidence --> QARR[BM25 QA evidence retriever]
    QARR --> QAPrompt[Image + question + retrieved passages]
    QAPrompt --> VisionQA[Configured vision QA model]
    Eval --> Metrics[JSON metrics / JSONL predictions]
    Eval --> LLM
    Eval --> BM25
```

Dashed Ollama usage is optional. The M3SciQA evaluator's calls to the vision model do not pass through `Agent`; the separate interactive `inspect_figures` tool does.

The multimodal QA benchmark groups public `evidence_reference` passages by `reference_arxiv_id`, retrieves up to `--rag-top-k` positive-scoring passages for each question, and includes them with their arXiv IDs in the prompt. Answer and explanation labels are excluded. This is annotation-assisted RAG evaluation within M3SciQA, not open-corpus retrieval over collected PDFs.

## UML Class Model

```mermaid
classDiagram
    class LLM {
        <<interface-like base>>
        +decide(question, observations) dict
    }
    class OllamaToolCallingLLM {
        +decide(question, observations, tools) dict
        +model str
        +host str
    }
    class Agent {
        -llm LLM
        -tools ToolRegistry
        -reranker Reranker
        -max_steps int
        -top_k int
        +run(question) dict
    }
    class Orchestrator {
        -corpus List~str~
        +run(question) dict
        +_retrieve(query) list
    }
    class ResearchAssistant {
        -papers List~ResearchPaper~
        -documents List~str~
        -reranker Reranker
        -agent Agent
        +run(question) dict
        +search(query, top_k) list
        +publication_trend() dict
        +inspect_figures(query, top_k) dict
    }
    class ToolRegistry {
        -tools Dict~str, Tool~
        +register(tool) None
        +call(name, arguments) Any
        +describe() list
    }
    class Tool {
        +name str
        +description str
        +func Callable
        +call(**kwargs) Any
    }
    class Reranker {
        +rerank(query, docs, top_k) list
    }
    class ResearchPaper {
        +paper_id str
        +title str
        +authors List~str~
        +year Optional~int~
        +abstract str
        +full_text str
        +figures List~dict~
        +indexed_text() str
    }
    class VisionModule {
        <<module: vision.py>>
        +analyze_image_with_ollama(path, prompt, model) str
        +analyze_images_with_ollama(paths, prompt, model) str
    }

    LLM <|-- OllamaToolCallingLLM
    Agent <|-- Orchestrator
    ResearchAssistant *-- Agent : constructs
    ResearchAssistant o-- ResearchPaper : indexes
    ResearchAssistant *-- Reranker : owns
    ResearchAssistant ..> ToolRegistry : registers tools
    Agent --> LLM : asks for next action
    Agent --> ToolRegistry : calls selected tool
    Agent --> Reranker : reranks search results
    ToolRegistry o-- Tool : stores
    ResearchAssistant ..> VisionModule : optional figure analysis
```

`LLM` is an interface-like base class; the Ollama and OpenAI implementations use the configured model and registry JSON schemas to choose one next tool action or return an answer. `Orchestrator` inherits the adaptive `Agent` loop directly and supplies corpus retrieval; `ResearchAssistant` separately creates an `Agent` and registers retrieval, trend, calculator, and figure-inspection tools. The registry stores callable tool wrappers, not independent autonomous agents.

## Interactive Question Sequence

```mermaid
sequenceDiagram
    actor User
    participant CLI as main.py
    participant RA as ResearchAssistant
    participant R as ConfiguredToolCallingLLM
    participant A as Agent / Orchestrator
    participant TR as ToolRegistry
    participant RR as Reranker
    participant V as vision.py Ollama adapter

    User->>CLI: ask --question Q
    CLI->>RA: run(Q)
    RA->>A: run(Q)
    A->>R: decide(Q, observations, tool schemas)
    R-->>A: tool call or answer
    opt model selected a tool
        A->>TR: call(selected tool, args)
        opt selected tool is search
            A->>RR: rerank candidates
            RR-->>A: scored documents
        end
        opt selected tool is inspect_figures
            TR->>RA: inspect_figures(Q)
            RA->>RR: rank figure/page contexts
            RR-->>RA: ranked assets
            opt vision model configured
                RA->>V: analyze selected image + prompt
                V-->>RA: visual description
            end
        end
        TR-->>A: tool observation
        A->>R: decide(Q, observations including this result)
        R-->>A: one next tool call or grounded answer
    end
    A-->>RA: answer + trace
    RA-->>CLI: result
    CLI-->>User: answer and used tools
```

The loop is bounded by `Agent.max_steps` tool actions. Every decision chooses at most one action; after executing it, the agent passes the full observation trace to the LLM before asking for the next action. This allows later choices to adapt to earlier results instead of following a preplanned tool sequence. The agent permits one final model decision after the action budget is consumed, so it can synthesize an answer from the last observation but cannot call another tool. Batched tool calls are rejected. Search results are BM25 reranked after execution. If token-overlap filtering finds no candidates, search returns an empty list rather than the unrelated corpus. The LLM is instructed to identify missing or conflicting evidence, but this is not calibrated abstention or guaranteed conflict resolution; generated claims are not independently entailment-checked.

## Evaluation Flow

```mermaid
sequenceDiagram
    actor Evaluator
    participant CLI as evaluate-agent CLI
    participant E as evaluate_m3sciqa_agent
    participant D as selected M3SciQA split JSONL
    participant RR as BM25 Reranker
    participant R as ConfiguredToolCallingLLM
    participant O as JSON result

    Evaluator->>CLI: evaluate-agent --split validation or test --top-k 5
    CLI->>E: evaluate(data_dir, split, 5)
    E->>D: load rows
    loop each row with reference evidence and paper ID
        E->>RR: rank evidence corpus using question
        RR-->>E: top five paper IDs
        E->>E: Recall@5, reciprocal rank, top-1 ID correctness
    end
    loop each figure/table row
        E->>R: decide(question, tool schemas)
        R-->>E: first selected tool
        E->>E: compare to inspect_figures expectation
    end
    E->>O: write metrics and definitions
    O-->>Evaluator: JSON output
```

The candidate evidence corpus is formed from the selected split's public `evidence_reference` annotations grouped by `reference_arxiv_id`; answer and explanation fields are excluded. Because reference annotations create the candidate evidence set, this evaluation measures ranking within annotated evidence, not open-corpus retrieval. The validation split has answer labels; the test split does not. The four annotation-based retrieval/routing metrics can run on both splits, but answer correctness or answer-generation EM/F1 cannot be scored on test.

## Storage and Data Flow

- `papers.py` discovers arXiv metadata, normalizes paper records, serializes JSONL, and ingests local PDFs with PyMuPDF.
- `ResearchPaper.indexed_text()` combines paper metadata, abstract/full text, figure context, and source URL into a searchable string.
- `ResearchAssistant` builds its in-memory searchable document list from paper records.
- `_retrieve()` performs first-stage token-overlap filtering; `Reranker` scores and sorts candidates with BM25.
- `benchmark.py` handles M3SciQA dataset/evaluation operations; `evaluation.py` computes the four agent-evaluation metrics.
- Local corpora and extracted images live under `data/`; metric outputs live under `results/`. Existing dataset files are not changed by the evaluation commands.

## Design Choices and Limitations

- **Small, inspectable runtime:** arXiv collection and JSON/XML processing use the Python standard library; PyMuPDF handles PDFs; BM25 and tokenization are implemented locally. This keeps setup modest, but there is no sentence-transformer embedding model or FAISS index.
- **Adaptive tool orchestration:** `Agent` asks the LLM for one next action, executes it, and supplies its result before the next decision. A maximum action budget bounds execution, while a final decision can synthesize an answer after the budget. Unsupported, malformed, or batched calls raise an error; no automatic retry policy is implemented.
- **LLM tool selection:** Ollama and OpenAI implementations receive function schemas from `ToolRegistry`. Results depend on model and prompt. The saved 0.3407/0.3560 scores are from the former keyword router. The pre-adaptive GPT run scored 0.1615 first-tool accuracy with three errors; the fresh run after sequential replanning scored 0.0996 strict first-tool accuracy (45/452 correct, zero recorded tool-call errors). The benchmark evaluates only the first choice per row, not later replanning. See the [fresh tool-calling metrics](../results/tool-calling/validation/agentic-run-2026-10-31/metrics.json).
- **Two-stage lexical retrieval:** `_retrieve()` filters by token overlap, then BM25 ranks candidates. No overlap yields an empty result, enabling the LLM to recognize missing evidence; relevant documents without shared query terms can still be missed. No dense retrieval or vector-only/hybrid ablation has been run.
- **Citation scope:** interactive retrieval returns source identifiers/URLs with the selected evidence text. The M3SciQA citation metric only checks whether the top-ranked arXiv ID equals `reference_arxiv_id`; it does not judge whether the text entails the answer. The M3SciQA retrieval corpus is formed from annotated `evidence_reference` passages, so its Recall/MRR results measure ranking within annotated evidence, not open-corpus retrieval.
- **Multimodal boundary:** PDF ingestion extracts selectable text and embedded raster images; optional page rendering captures vector content. Scanned PDFs have no OCR. Interactive figure inspection can request a local Ollama vision description; answer grounding and cross-paper visual reasoning are not independently verified.
- **Evaluation labels:** M3SciQA validation has answer labels, while the public test answers are blank/hidden. Test retrieval, ID-match, and routing metrics are calculable from released annotations; EM/F1 or answer correctness is not. Never interpret empty-reference test scores as zero model quality.
- **Evidence robustness:** prompts instruct the LLM to state when support is missing or sources conflict. This is instruction-level guidance, not a contradiction detector or calibrated abstention mechanism, and has not been validated on a dedicated robustness set.
- **Provider and data privacy:** with Ollama, questions, tool results, and optional vision images are sent to the local service. With OpenAI, agent questions/tool results or vision images are sent to the hosted API. arXiv collection and dataset download require network access.
