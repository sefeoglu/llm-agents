# Evaluation Results

This folder collects the completed **validation-split** evaluations. Original artifacts remain under `results/qa/` and `results/tool-calling/`.

## QA with RAG

The saved vision-only baseline and fresh agentic RAG run use OpenAI `gpt-4.1-mini` over the same 452 labeled validation examples:

| Metric | Vision only | BM25 RAG |
| --- | ---: | ---: |
| Exact match | 0.0000 | 0.0000 |
| Token F1 | 0.1503 | 0.2387 |
| ROUGE-L F1 | 0.1161 | 0.2044 |
| chrF | 0.2391 | 0.3220 |
| Smoothed sentence BLEU-4 | 0.0616 | 0.1125 |

Fresh RAG artifacts: [metrics](validation/qa/metrics-agentic-run-2026-10-31.json), [scored predictions](validation/qa/predictions-agentic-run-2026-10-31-scored.jsonl), and [complete raw predictions](../../results/qa/validation/agentic-run-2026-10-31/rag-predictions-complete.jsonl). The first 390 rows came from an interrupted run and the remaining 62 were resumed separately; the merged file was verified to cover all 452 row IDs exactly once. Earlier predictions and metrics remain unchanged under [`results/qa/validation/`](../../results/qa/validation/).

Both metric files include overall and modality-level exact match; token precision/recall/F1; ROUGE-1/2/L precision/recall/F1; smoothed sentence BLEU-1 through BLEU-4; chrF; and answer-length statistics. RAG adds evidence hit@k, MRR, top-citation ID match, and retrieved-paper coverage; these retrieval scores compare against annotation-provided paper IDs and do not establish semantic entailment. The RAG corpus is derived from the split's `evidence_reference`, so this evaluates answer generation with provided evidence, not open-corpus retrieval. Metrics are macro-averages over labeled examples.
```bash
python llm_agents/evaluation/evaluate_saved_qa.py \
	--predictions results/qa/validation/agentic-run-2026-10-31/rag-predictions-complete.jsonl \
	--output llm_agents/evaluation/validation/qa/metrics-recomputed.json \
	--scored-predictions llm_agents/evaluation/validation/qa/predictions-recomputed-scored.jsonl
```

The retrieval corpus is annotation-provided evidence, so this measures answer generation with supplied passages rather than open-corpus paper retrieval. GPT received the question, local image, and retrieved passages. No Ollama model was used. The fresh paired analysis and exported charts/CSV are in [the dated figure directory](validation/qa/figures-agentic-run-2026-10-31/); the source analysis is [rag-vs-vision-only.ipynb](rag-vs-vision-only.ipynb).

## Tool Calling and Retrieval

- Provider/model: OpenAI `gpt-4.1-mini` (fresh run using the current one-action-per-decision agent)
- Examples considered: 452
- Retrieval-eligible: 450
- Recall@5: 0.8533
- MRR@5: 0.7855
- Citation-ID correctness: 0.7400
- Tool-selection accuracy: 0.0996 (45 `inspect_figures` and 407 `search` first decisions)
- Tool-call errors: 0
- Artifacts: [fresh metrics](../../results/tool-calling/validation/agentic-run-2026-10-31/metrics.json), [fresh sample traces](../../results/tool-calling/validation/agentic-run-2026-10-31/samples.json), [tool-calling analysis notebook](tool_calling.ipynb)

Retrieval and citation scores use M3SciQA annotation-provided evidence. Citation correctness is an arXiv-ID match, not a semantic entailment score. Tool-selection accuracy compares only the first GPT decision with the benchmark's expected `inspect_figures` label for figure/table rows; it does not evaluate later replanning. The fresh run had 41/142 correct figure selections and 4/310 correct table selections, with no tool-call errors. Earlier outputs remain available at the original result paths.
