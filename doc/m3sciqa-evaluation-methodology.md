# M3SciQA Agent Evaluation Methodology

## Purpose

This evaluation measures evidence retrieval, rank quality, citation-ID matching, and configured-LLM tool selection on Yale NLP's M3SciQA validation and test splits. Both splits expose question text, an image (figure or table), `reference_arxiv_id`, `evidence_reference`, and modality labels. Test answers are hidden, but these annotation fields are available for retrieval and routing metrics. No answer-quality metric (EM/F1) is computed for test.

## Data and Scope

- Source: [Yale NLP M3SciQA](https://huggingface.co/datasets/yale-nlp/M3SciQA), CC-BY-4.0.
- Local inputs: `data/m3sciqa/validation.jsonl` and `data/m3sciqa/test.jsonl`.
- Validation examples considered: 452.
- Test examples considered: 1,000.
- Validation retrieval/citation examples: 450 with a `reference_arxiv_id` and non-empty `evidence_reference`.
- Test retrieval/citation examples: 997 with a `reference_arxiv_id` and non-empty `evidence_reference`.
- Tool-routing examples: all 452 validation and 1,000 test rows with `modal` equal to `figure` or `table`.
- Test answer labels are hidden, so test EM/F1 is not computed. Retrieval, paper-ID match, and tool-choice metrics use the released evidence/reference-ID/modality annotations only.

The evaluator reads the existing dataset but does not edit it. It does not use `answer`, `answer_anchor`, or `explanation_reference` to build retrieval documents.

## Retrieval Corpus

For each row in the selected split with a reference ID and evidence text, `evidence_reference` is grouped under its `reference_arxiv_id`. Repeated evidence snippets for the same paper are deduplicated and joined. Each resulting paper entry has the form:

```text
[arXiv:<reference_arxiv_id>] <deduplicated evidence_reference passages>
```

The row's `question` is the retrieval query. The existing BM25 `Reranker` scores candidate entries and returns up to $k=5$ with positive scores. The gold relevant document is the row's `reference_arxiv_id`. Rows lacking usable reference evidence or ID are excluded from retrieval and citation metrics, not counted as failures.

The image-grounded QA benchmark also uses the selected split's public `evidence_reference` passages as RAG context. It groups passages by `reference_arxiv_id`, retrieves up to `--rag-top-k` positive-scoring passages for each question, and supplies passage text and paper IDs with the image to the answer model. Answer and explanation fields are not indexed. Because this QA corpus is derived from benchmark annotations, it evaluates answer generation with provided evidence rather than open-corpus document retrieval.

## Metrics

**Recall@5.** There is one gold reference paper per eligible row, so per-row recall is 1 if that paper appears in the positive-scoring top five and 0 otherwise. The reported metric is the mean over eligible rows.

**MRR@5.** For each eligible row, reciprocal rank is $1/r$ where $r$ is the rank of the first retrieved gold paper; it is 0 if that paper is not in the top five. The reported metric is the mean over eligible rows.

**Citation correctness.** This retrieval metric is 1 when the top-ranked paper ID equals the row's `reference_arxiv_id`, otherwise 0, averaged over rows with a citation. It is an ID-match proxy only; it does not score a generated answer or check whether a passage semantically entails the answer.

**Tool-selection accuracy.** The first decision from the configured Ollama or OpenAI LLM, given the question and registered tool schemas, is compared with the expected `inspect_figures` tool for each row whose modality is `figure` or `table`. Use `--limit` for a smaller run. This strict label is a simple visual-tool expectation, not a measure of multi-step planning quality. The interactive agent selects one action at a time and receives each result before its next decision. This evaluator measures only the initial selection for each row; it does not exercise subsequent replanning.

## Reproduction

From the repository root, with the local M3SciQA files present:

```bash
python main.py evaluate-agent --agent-provider openai --data-dir data/m3sciqa --split validation --top-k 5 --output results/tool-calling/validation/new-run/metrics.json --samples-output results/tool-calling/validation/new-run/samples.json
```

The command writes JSON metrics; `--samples-output` additionally writes one input/output record per sample. Inputs include question, modality, local image path, reference paper ID, and evidence annotation. Outputs include ranked paper IDs and BM25 scores, gold rank, Recall@k hit, reciprocal rank, top-citation ID match, and selected/expected tool. Hidden answer text is not copied into traces. OpenAI tool selection sends questions and registered tool schemas to the hosted API; do not commit `config/gpt.json`. Tests use synthetic fixtures and injected fake LLMs for deterministic expected values:

```bash
python -m pytest -q tests/test_research.py -k m3sciqa_agent
```

## Recorded Results

Historical results from the former rule-based router, measured on both complete splits:

| Split | Recall@5 | MRR@5 | Citation correctness | Tool-selection accuracy |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.8533 (450 rows) | 0.7855 (450 rows) | 0.7400 (450 citations) | 0.3407 (452 rows) |
| Test | 0.8837 (997 rows) | 0.8130 (997 rows) | 0.7673 (997 citations) | 0.3560 (1,000 rows) |

Historical GPT routing results (`gpt-4.1-mini`, before sequential replanning) had first-tool accuracy 0.1615 and 3 tool-call errors. The fresh 2026-10-31 run using the current one-action-per-decision implementation reports Recall@5 0.8533, MRR@5 0.7855, citation-ID match 0.7400, first-tool accuracy 0.0996, and zero tool-call errors. It selected `inspect_figures` for 45 of 452 rows; accuracy was 41/142 for figure questions and 4/310 for table questions. See [fresh metrics](../results/tool-calling/validation/agentic-run-2026-10-31/metrics.json) and [sample traces](../results/tool-calling/validation/agentic-run-2026-10-31/samples.json). All retrieval metrics rank within provided evidence annotations; citation-ID matching is not semantic support. Test answer labels are hidden, so this answer-generation and planner evaluation is validation-only.
