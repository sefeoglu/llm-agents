# M3SciQA Agent Evaluation Methodology

## Purpose

This evaluation measures evidence retrieval, rank quality, citation-ID correctness, and first-tool selection on Yale NLP's M3SciQA validation and test splits. Both splits expose question text, an image (figure or table), `reference_arxiv_id`, `evidence_reference`, and modality labels. Test answers are hidden, but these annotation fields are available and are sufficient for the four metrics in this harness. No answer-quality metric (EM/F1) is computed for test.

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

## Metrics

**Recall@5.** There is one gold reference paper per eligible row, so per-row recall is 1 if that paper appears in the positive-scoring top five and 0 otherwise. The reported metric is the mean over eligible rows.

**MRR@5.** For each eligible row, reciprocal rank is $1/r$ where $r$ is the rank of the first retrieved gold paper; it is 0 if that paper is not in the top five. The reported metric is the mean over eligible rows.

**Citation correctness.** The current deterministic answer path cites its top-ranked retrieved paper. This metric is 1 when that paper ID equals the row's `reference_arxiv_id`, otherwise 0, averaged over rows with a citation. This is a citation-ID match proxy only. It does not check whether the passage semantically entails or fully supports the answer; that requires human annotation or an independently validated entailment judge.

**Tool-selection accuracy.** The actual first decision from `ResearchLLM.decide(question, [])` is compared with the expected `inspect_figures` tool for every row whose modality is `figure` or `table`. Accuracy is the fraction of these decisions matching the expected tool. This evaluates the current router's visual-intent behavior; it does not claim that figure inspection alone is sufficient for every multi-document question.

## Reproduction

From the repository root, with the local M3SciQA files present:

```bash
python main.py evaluate-agent --data-dir data/m3sciqa --split validation --top-k 5 --output results/m3sciqa-agent-metrics-validation.json
python main.py evaluate-agent --data-dir data/m3sciqa --split test --top-k 5 --output results/m3sciqa-agent-metrics-test.json
python main.py evaluate-agent --data-dir data/m3sciqa --split validation --top-k 5 --samples-output results/m3sciqa-agent-samples-validation.json
python main.py evaluate-agent --data-dir data/m3sciqa --split test --top-k 5 --samples-output results/m3sciqa-agent-samples-test.json
```

The command writes JSON metrics; `--samples-output` additionally writes one input/output record per sample. Inputs include question, modality, local image path, reference paper ID, and evidence annotation. Outputs include ranked paper IDs and BM25 scores, gold rank, Recall@k hit, reciprocal rank, top-citation ID match, and selected/expected tool. Hidden answer text is not copied into either split's trace. The evaluator makes no model calls and does not modify dataset files. Tests use synthetic temporary fixtures for deterministic expected values:

```bash
python -m pytest -q tests/test_research.py -k m3sciqa_agent
```

## Recorded Results

Measured on both complete splits:

| Split | Recall@5 | MRR@5 | Citation correctness | Tool-selection accuracy |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.8533 (450 rows) | 0.7855 (450 rows) | 0.7400 (450 citations) | 0.3407 (452 rows) |
| Test | 0.8837 (997 rows) | 0.8130 (997 rows) | 0.7673 (997 citations) | 0.3560 (1,000 rows) |

The test metrics use public evidence passages and paper IDs; they do not use hidden answers, and no answer correctness is implied. Tool-selection accuracy is low because the lexical router often misses visual intent when a question does not explicitly mention a figure/table. Citation correctness is not semantic support accuracy. Aggregate results are saved in [validation metrics](../results/m3sciqa-agent-metrics-validation.json) and [test metrics](../results/m3sciqa-agent-metrics-test.json); per-sample I/O is in [validation samples](../results/m3sciqa-agent-samples-validation.json) and [test samples](../results/m3sciqa-agent-samples-test.json).
