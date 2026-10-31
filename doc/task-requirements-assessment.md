# Scientific Research Agent Task Assessment

**Assessment date:** 2026-10-31  
**Overall:** A credible prototype that demonstrates the requested architecture and components, but it does not yet demonstrate reliable scientific answer quality. Its strengths are automated paper search, a bounded LLM tool-calling loop, PDF/image processing, and reproducible evaluation. Its main weaknesses are low answer scores, limited independent-corpus retrieval evaluation, and no tested policy for conflicting or insufficient evidence.

## Requirement Review

| Requirement | Status | Assessment |
| --- | --- | --- |
| Scientific paper collection | Mostly met | The arXiv collector searches by query and saves metadata and abstracts. The local battery corpus has 20 automated candidates requiring relevance review. It does not automatically download and fully process every returned PDF. |
| Paper processing and indexing | Mostly met | PDF ingestion extracts selectable text and embedded images; optional page rendering captures vector figures/tables; BM25 ranks indexed records. Scanned PDFs have no OCR, and collection and PDF ingestion are separate steps. |
| Evidence-based question answering | Partially met | The configured LLM can call retrieval/analysis tools and compose a response from observations. The complete GPT validation run shows low answer overlap with references, and answer claims/citations are not independently verified. |
| Research tools | Met at prototype level | arXiv collection, BM25 retrieval, calculator, publication-trend analysis, PDF figure inspection, and optional vision analysis are available. |
| ML or scientific analysis | Minimally met | Annual paper counts and a least-squares linear trend provide a small analysis component. This is not a trained model or a battery-science experiment; no battery lifetime prediction is claimed. |
| Evaluation and testing | Partially met | There are automated tests and M3SciQA answer/retrieval evaluations. Retrieval evaluation ranks within annotation-provided evidence, not an independent open corpus; paper-ID match is not semantic entailment. Test answers are hidden, and the full test prediction run is incomplete. |
| Adaptive agent/tool choice | Implemented, but measured quality is weak | Ollama and OpenAI LLMs receive tool schemas. The bounded agent selects one next action, observes its result, and then asks the model to re-plan; batched calls are rejected. The fresh GPT validation run's strict first-tool selection accuracy is 0.0996 with zero tool-call errors. This evaluator measures the first choice, not subsequent replanning; see [fresh tool-calling metrics](../results/tool-calling/validation/agentic-run-2026-10-31/metrics.json). |
| Multimodal handling | Met at prototype level | PDF figures and M3SciQA images can be sent to a vision model. The complete GPT validation run exercised all 452 validation images. Predictions indicate substantial remaining visual/table reasoning errors. |
| Robustness to missing/conflicting evidence | Improved, not demonstrated as reliable | Retrieval now returns no documents when token-overlap filtering finds no match, rather than falling back to the full corpus. Prompts ask the LLM to state when evidence is missing or contradictory. There is no explicit contradiction detector, calibrated abstention, or dedicated missing/conflicting-evidence evaluation. |
| Use of coding/AI agents during development | Not evidenced in repository | The source and docs do not record how development agents were used. This can be explained in an interview, but the repository currently provides no process log or representative agent-assisted workflow. |
| Working code and run instructions | Met | Automated tests cover tool orchestration and provider adapters. See the [README](../README.md) and [tests](../tests/test_research.py). The README documents local Ollama and hosted OpenAI agent/vision setup. |
| Architecture, examples, and results | Met | Component/sequence diagrams are in the [architecture document](system-architecture.md); examples are in the README; fresh GPT outputs are linked below. |

## Evaluation Evidence

The OpenAI GPT vision-only validation baseline covers all **452** labeled M3SciQA validation examples. With `gpt-4.1-mini`, exact match is **0.0** and token F1 is **0.1503**; figure F1 is **0.1517** over 142 examples and table F1 is **0.1496** over 310 examples. The fresh GPT RAG run on the same split/model, with BM25 top-3 annotation passages added to prompts, also has exact match **0.0** and token F1 **0.2387** (figure **0.2340**, table **0.2408**). RAG evidence hit@3 is **0.8319**. This descriptive comparison is not significance-tested, and retrieval uses M3SciQA-provided evidence rather than an independent paper corpus. See the [vision-only metrics](../results/gpt-qa-rerun-2026-10-31/m3sciqa-validation-gpt-4.1-mini-452-metrics.json), [fresh RAG metrics](../llm_agents/evaluation/validation/qa/metrics-agentic-run-2026-10-31.json), and [complete fresh RAG predictions](../results/qa/validation/agentic-run-2026-10-31/rag-predictions-complete.jsonl).

The latest saved GPT test file contains **122 of 1,000** predictions. The M3SciQA test split has hidden answers, so it cannot produce answer EM/F1; the remaining 878 test predictions have not been generated. See the [partial test predictions](../results/gpt-qa-rerun-2026-10-31/m3sciqa-test-gpt-4.1-mini-1000-predictions.jsonl).

The historical M3SciQA retrieval/routing scores (validation Recall@5 0.8533, MRR@5 0.7855, citation-ID match 0.7400, tool-selection accuracy 0.3407; test 0.8837, 0.8130, 0.7673, 0.3560) were produced by the former keyword router. The fresh GPT run has Recall@5 **0.8533**, MRR@5 **0.7855**, citation-ID match **0.7400**, and strict first-tool accuracy **0.0996** (zero tool-call errors). Retrieval ranks within annotation-provided passages, not an independent paper corpus. See the [historical validation metrics](../results/rerun-2026-10-31/m3sciqa-validation-metrics.json), [historical test metrics](../results/rerun-2026-10-31/m3sciqa-test-metrics.json), [fresh tool-calling metrics](../results/tool-calling/validation/agentic-run-2026-10-31/metrics.json), and [evaluation methodology](m3sciqa-evaluation-methodology.md).

## Main Gaps

1. **Answer quality:** The fresh full validation RAG score remains limited (EM 0.0, token F1 0.2387); avoid claiming reliable answers. The comparison with vision-only is descriptive, not significance-tested.
2. **Evidence robustness:** Empty retrieval no longer supplies an unrelated full corpus, and prompts request explicit handling of missing or conflicting support. Contradiction detection, calibrated abstention, and failure recovery are not systematically implemented or evaluated.
3. **Retrieval validity:** Existing full-split retrieval scores use benchmark-provided evidence candidates; they do not establish open-corpus RAG quality.
4. **Agent planning:** Tool choice is LLM-driven and each next decision receives preceding tool results. The fresh full-validation strict first-tool accuracy is low (0.0996); this benchmark does not measure later replanning, so multi-step planner quality remains unassessed.
5. **Scientific component:** The publication trend is the only implemented analysis experiment; there is no battery-specific ML experiment.
6. **Development process:** Coding-agent use is not recorded in project artifacts; prepare a truthful verbal account rather than imply the repository proves it.
