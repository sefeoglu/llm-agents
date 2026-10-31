# Scientific Research Agent Task Assessment

**Assessment date:** 2026-10-31  
**Overall:** Partially satisfies the task. The repository contains a working, tested research-agent prototype and most requested components, but answer generation, evidence reliability, and tool selection remain limited.

## Requirement Review

| Requirement | Status | Assessment |
| --- | --- | --- |
| Find and collect relevant scientific papers automatically | Mostly met | The arXiv collector searches by query and saves paper metadata and abstracts. The current battery corpus contains 20 automated search results; the README correctly describes them as candidates requiring relevance review. The collector does not automatically download and fully process every paper PDF. |
| Process and index papers, including useful figures/tables | Mostly met | PDF ingestion extracts selectable text and embedded images, with optional page rendering for vector figures and tables. BM25 ranks indexed paper content. Scanned PDFs have no OCR, and arXiv collection and PDF ingestion are separate steps. |
| Answer questions using evidence from collected papers | Partially met | The interactive path retrieves and returns a top-ranked source passage with its paper identifier and source. It does not generate a synthesized answer across multiple papers. The project documents this as a deterministic retrieval baseline rather than a generative chatbot. |
| Provide useful research tools | Met at prototype level | Tools include arXiv collection, BM25 retrieval, a safe calculator, publication-trend analysis, and optional local Ollama figure inspection. |
| Include a small ML or scientific data-analysis component | Minimally met | The publication-trend tool computes yearly paper counts and fits a least-squares linear slope. This is a small corpus analysis, not a trained model or a battery-science experiment. No battery lifetime regression result is claimed. |
| Evaluate evidence retrieval and answer reliability | Partially met | The M3SciQA harness reports retrieval/ranking, paper-ID match, tool-selection accuracy, and labeled-answer EM/F1. Retrieval candidates come from M3SciQA's annotated `evidence_reference` passages, so this measures ranking within annotated evidence rather than retrieval from an independent paper corpus. The citation score is an ID match, not a semantic entailment judgment. |
| Let the agent select tools rather than use one fixed pipeline | Partially met | A bounded agent loop chooses tools from question cues. Routing is deterministic keyword/regex logic rather than model-based planning, and the benchmark reports low figure/table tool-selection accuracy. It does not plan arbitrary multi-step research workflows. |
| Working code | Met | The current test suite passes: **16 passed**. See the [README run instructions](../README.md) and [tests](../tests/test_research.py). |
| Short README with run instructions | Partially met | The README provides installation, CLI, dataset, and evaluation instructions, but it is considerably more detailed than a short task README. |
| Architecture and design description | Met | The project has a separate [system architecture document](system-architecture.md) with component and sequence diagrams, design choices, and limitations. |
| Example questions and outputs | Met | The README contains retrieval, trend, calculator, and multimodal examples, including unsuccessful vision answers. |
| Evaluation results | Met, with important limitations | Saved result files are linked below. They should be interpreted as prototype measurements, not evidence of high answer quality or open-corpus retrieval quality. |

## Evaluation Evidence

The saved 20-example M3SciQA validation vision run with `llava` reports **exact match 0.0** and **token F1 0.1092**. The modality F1 values are 0.0503 for 3 figure examples and 0.1196 for 17 table examples. See [validation answer metrics](../results/rerun-2026-10-31/m3sciqa-validation-llava-20-metrics.json).

The 20-example M3SciQA test inference has hidden answer labels, so EM and F1 are correctly `null` rather than scored as zero. See [test answer metrics](../results/rerun-2026-10-31/m3sciqa-test-llava-20-metrics.json).

The full annotation-based M3SciQA validation evaluation reports Recall@5 **0.8533**, MRR@5 **0.7855**, citation-ID correctness **0.7400**, and tool-selection accuracy **0.3407** over 452 rows (450 retrieval-eligible). The corresponding test metrics report Recall@5 **0.8837**, MRR@5 **0.8130**, citation-ID correctness **0.7673**, and tool-selection accuracy **0.3560** over 1,000 rows (997 retrieval-eligible). These are annotation-based measures; test answer correctness is not scored. See [validation metrics](../results/rerun-2026-10-31/m3sciqa-validation-metrics.json), [test metrics](../results/rerun-2026-10-31/m3sciqa-test-metrics.json), and the [evaluation methodology](m3sciqa-evaluation-methodology.md).

## Main Gaps

1. **Answer synthesis and grounding:** The interactive agent returns a retrieved passage rather than composing a cross-paper answer with claim-level citations and support checks.
2. **Evidence reliability:** There is no explicit conflict-resolution policy. When no documents share query tokens, retrieval falls back to the full corpus; the system does not reliably abstain when evidence is weak or missing.
3. **Tool routing:** Tool choice is based on explicit keyword cues. Its measured figure/table selection accuracy is only about 34–36% under the benchmark's expected-tool rule.
4. **Scientific analysis depth:** The annual publication trend is a useful small analysis, but there is no battery-specific model, experiment, or evaluated scientific prediction component.
5. **Paper corpus quality and scope:** The local battery corpus is a small automated result set, not a curated or comprehensively indexed literature collection.
