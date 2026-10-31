# Research Assistant Architecture

This document models the current `llm_agents` implementation. It describes the implemented system rather than an aspirational LLM architecture: tool routing is deterministic and rule-based; no hosted LLM or agent framework is required.

## System Boundary

The project has two related but distinct flows:

1. Interactive research: arXiv paper records are collected or PDFs are ingested, indexed as text, and queried through a bounded tool loop. The current local arXiv corpus has battery-degradation/cycle-life search results; the no-argument CLI demo still uses a small generic RAG sample corpus.
2. Offline evaluation: the benchmark runner uses M3SciQA annotations to measure retrieval, citation-ID matching, and routing. M3SciQA inference can also send local images to **Ollama**, but that inference path is separate from the text agent loop.

The project does not include or have local access to the NASA PCoE battery dataset. An attempted archive download was incomplete and unusable; it is not part of the project data. Consequently, no battery lifetime regression experiment or score is claimed.

```mermaid
flowchart LR
    User[User question] --> CLI[main.py CLI]
    CLI --> Assistant[ResearchAssistant]
    Assistant --> Router[ResearchLLM rule router]
    Router --> Agent[Agent bounded loop]
    Agent --> Registry[ToolRegistry]
    Registry --> Search[Search tool]
    Registry --> Calc[Calculator tool]
    Registry --> Trend[Publication trend tool]
    Registry --> Figure[Figure inspection tool]
    Search --> Candidate[Token-overlap candidate filter]
    Candidate --> BM25[BM25 Reranker]
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
    Eval --> Metrics[JSON metrics / JSONL predictions]
    Eval --> Router
    Eval --> BM25
```

Dashed Ollama usage is optional. The M3SciQA evaluator's calls to the vision model do not pass through `Agent`; the separate interactive `inspect_figures` tool does.

## UML Class Model

```mermaid
classDiagram
    class LLM {
        <<interface-like base>>
        +decide(question, observations) dict
    }
    class RuleBasedLLM {
        +decide(question, observations) dict
    }
    class ResearchLLM {
        +decide(question, observations) dict
    }
    class Agent {
        -llm LLM
        -tools ToolRegistry
        -reranker Reranker
        -max_steps int
        -top_k int
        +run(question) dict
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

    LLM <|-- RuleBasedLLM
    RuleBasedLLM <|-- ResearchLLM
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

`LLM` is an interface-like base class; `RuleBasedLLM` and `ResearchLLM` make decisions. The registry stores callable tool wrappers, not independent autonomous agents. `ResearchAssistant` creates `Agent` and registers closures/methods for retrieval, trends, and figure inspection.

## Interactive Question Sequence

```mermaid
sequenceDiagram
    actor User
    participant CLI as main.py
    participant RA as ResearchAssistant
    participant R as ResearchLLM
    participant A as Agent
    participant TR as ToolRegistry
    participant RR as Reranker
    participant V as vision.py Ollama adapter

    User->>CLI: ask --question Q
    CLI->>RA: run(Q)
    RA->>A: run(Q)
    A->>R: decide(Q, observations=[])
    alt visual keywords detected
        R-->>A: inspect_figures(query=Q)
        A->>TR: call(inspect_figures, args)
        TR->>RA: inspect_figures(Q)
        RA->>RR: rank figure/page contexts
        RR-->>RA: ranked assets
        opt vision model configured
            RA->>V: analyze selected image + prompt
            V-->>RA: visual description
        end
        RA-->>TR: figure summary with page/path/context
    else trend keywords detected
        R-->>A: publication_trend()
        A->>TR: call(publication_trend, args)
        TR->>RA: count papers by year and fit slope
        RA-->>TR: trend result + corpus-scope caveat
    else search or calculation
        R-->>A: search(...) or calculator(...)
        A->>TR: call(selected tool, args)
        opt selected tool is search
            A->>RR: rerank candidates
            RR-->>A: scored documents
        end
    end
    A->>R: decide(Q, observations)
    R-->>A: answer
    A-->>RA: answer + trace
    RA-->>CLI: result
    CLI-->>User: answer and used tools
```

The loop is bounded by `Agent.max_steps`. `Agent` appends each tool's name, arguments, and result to `trace`; search results are BM25 reranked after tool execution. The current rule-based router makes one primary tool selection; it does not yet plan arbitrary multi-step workflows or invoke a generative answer model.

## Evaluation Flow

```mermaid
sequenceDiagram
    actor Evaluator
    participant CLI as evaluate-agent CLI
    participant E as evaluate_m3sciqa_agent
    participant D as selected M3SciQA split JSONL
    participant RR as BM25 Reranker
    participant R as ResearchLLM
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
        E->>R: decide(question, empty observations)
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
- `benchmark.py` handles SciFact and M3SciQA dataset/evaluation operations; `evaluation.py` computes the four agent-evaluation metrics.
- Local corpora and extracted images live under `data/`; metric outputs live under `results/`. Existing dataset files are not changed by the evaluation commands.

## Design Choices and Limitations

- **Small, inspectable runtime:** arXiv collection and JSON/XML processing use the Python standard library; PyMuPDF handles PDFs; BM25 and tokenization are implemented locally. This keeps setup modest, but there is no sentence-transformer embedding model or FAISS index.
- **Bounded tool orchestration:** `Agent` applies a maximum-step loop and records tool calls/results in a trace. The trace is useful for debugging and evaluation; tool failures are not yet normalized into a structured retry/abstention policy.
- **Deterministic tool selection:** `ResearchLLM` routes using explicit regex/keyword cues instead of model tool-calling. This is easy to reproduce but brittle: measured tool-selection accuracy is 0.3407 on validation and 0.3560 on test under the benchmark's visual-tool label rule.
- **Two-stage lexical retrieval:** `_retrieve()` first filters by token overlap, then BM25 ranks candidates. This is lightweight, but a relevant document with no shared query tokens is lost before BM25; no dense retrieval or vector-only/hybrid ablation has been run.
- **Citation scope:** interactive retrieval returns source identifiers/URLs with the selected evidence text. The M3SciQA citation metric only checks whether the top-ranked arXiv ID equals `reference_arxiv_id`; it does not judge whether the text entails the answer. The M3SciQA retrieval corpus is formed from annotated `evidence_reference` passages, so its Recall/MRR results measure ranking within annotated evidence, not open-corpus retrieval.
- **Multimodal boundary:** PDF ingestion extracts selectable text and embedded raster images; optional page rendering captures vector content. Scanned PDFs have no OCR. Interactive figure inspection can request a local Ollama description, but the default deterministic agent does not synthesize a complete, multi-paper, evidence-cited answer from the vision output.
- **Evaluation labels:** M3SciQA validation has answer labels, while the public test answers are blank/hidden. Test retrieval, ID-match, and routing metrics are calculable from released annotations; EM/F1 or answer correctness is not. Never interpret empty-reference test scores as zero model quality.
- **Local data and vision privacy:** the M3SciQA corpus/images are local. When Ollama vision is enabled, image bytes go to the configured local Ollama service; arXiv collection requires network access. No remote model API is used by the default deterministic agent.
