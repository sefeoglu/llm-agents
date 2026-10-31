# M3SciQA Agent Evaluation Methodology

## Purpose

This evaluation measures four parts of the research assistant using the labeled validation split of Yale NLP's M3SciQA dataset: evidence retrieval, rank quality, citation-ID correctness, and first-tool selection. The dataset contains question text, an image (figure or table), `reference_arxiv_id`, `evidence_reference`, and modality labels. The test split has hidden answers and is not used for scored evaluation.

## Data and Scope

- Source: [Yale NLP M3SciQA](https://huggingface.co/datasets/yale-nlp/M3SciQA), CC-BY-4.0.
- Local input: `data/m3sciqa/validation.jsonl`.
- Validation examples considered: 452.
- Retrieval/citation examples: 450, those with a `reference_arxiv_id` and non-empty `evidence_reference`.
- Tool-routing examples: 452 with `modal` equal to `figure` or `table`.
- Test labels are hidden; test runs produce predictions only and have no EM/F1 or retrieval gold score in this harness.

The evaluator reads the existing dataset but does not edit it. It does not use `answer`, `answer_anchor`, or `explanation_reference` to build retrieval documents.

## Retrieval Corpus

For each validation row with a reference ID and evidence text, `evidence_reference` is grouped under its `reference_arxiv_id`. Repeated evidence snippets for the same paper are deduplicated and joined. Each resulting paper entry has the form:

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

From the repository root, with the local M3SciQA validation files present:

```bash
python main.py evaluate-agent --data-dir data/m3sciqa --split validation --top-k 5 --output results/m3sciqa-agent-metrics.json
```

The command writes JSON metrics. It does not make model calls or modify dataset files. Tests use synthetic temporary fixtures for deterministic expected values:

```bash
python -m pytest -q tests/test_research.py -k m3sciqa_agent
```

## Recorded Results

Measured on the full labeled validation split:

| Metric | Value | Denominator |
| --- | ---: | ---: |
| Recall@5 | 0.8533 | 450 retrieval-eligible rows |
| MRR@5 | 0.7855 | 450 retrieval-eligible rows |
| Citation correctness (top-1 ID match) | 0.7400 | 450 emitted top-1 citations |
| Tool-selection accuracy | 0.3407 | 452 figure/table rows |

The low tool-selection accuracy indicates the lexical router frequently fails to infer visual intent when the question does not explicitly mention a figure/table. Citation correctness is not semantic support accuracy. The machine-readable output is [results/m3sciqa-agent-metrics.json](../results/m3sciqa-agent-metrics.json).
