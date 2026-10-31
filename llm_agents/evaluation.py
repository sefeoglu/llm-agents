"""Evaluation metrics for retrieval, citations, and tool selection on M3SciQA."""
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .llm import LLM, OllamaToolCallingLLM, OpenAIToolCallingLLM
from .reranker import Reranker
from .research import ResearchAssistant


def _read_jsonl(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def evaluate_m3sciqa_agent(data_dir: str, split: str = "validation",
                           top_k: int = 5, limit: int = 0,
                           samples_output_path: str = "",
                           llm: Optional[LLM] = None,
                           model: Optional[str] = None,
                           provider: str = "ollama") -> dict:
    """Measure retrieval/citation/tool routing against M3SciQA annotations.

    Retrieval candidates are built only from the public ``evidence_reference``
    field, grouped by reference arXiv ID. The question's answer fields are not
    indexed. Citation correctness is an ID-match proxy, not human entailment.
    """
    if split not in {"validation", "test"}:
        raise ValueError("M3SciQA split must be validation or test")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    dataset_dir = Path(data_dir).expanduser().resolve()
    rows = _read_jsonl(dataset_dir / f"{split}.jsonl")
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No M3SciQA rows found in {dataset_dir / f'{split}.jsonl'}")

    evidence_by_paper: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        paper_id = str(row.get("reference_arxiv_id") or "").strip()
        evidence = str(row.get("evidence_reference") or "").strip()
        if paper_id and evidence:
            evidence_by_paper[paper_id].append(evidence)
    documents = [
        f"[arXiv:{paper_id}] " + "\n".join(dict.fromkeys(passages))
        for paper_id, passages in sorted(evidence_by_paper.items())
    ]
    paper_ids = [document.split("]", 1)[0].removeprefix("[arXiv:") for document in documents]
    reranker = Reranker()
    if llm:
        router = llm
    elif provider == "openai":
        router = OpenAIToolCallingLLM(model=model)
    elif provider == "ollama":
        router = OllamaToolCallingLLM(
            model or os.environ.get("OLLAMA_AGENT_MODEL", "qwen2.5:3b")
        )
    else:
        raise ValueError("provider must be 'ollama' or 'openai'")
    tool_specs = ResearchAssistant([], llm=router).agent.tools.describe()

    retrieval_recalls = []
    reciprocal_ranks = []
    citation_correct = 0
    citations_total = 0
    tool_correct = 0
    tool_total = 0
    tool_selection_errors = 0
    eligible_retrieval = 0
    sample_results = []

    for row in rows:
        question = str(row.get("question") or "")
        gold_id = str(row.get("reference_arxiv_id") or "").strip()
        ranked_ids = []
        ranked_results = []
        gold_rank = None
        recall_hit = None
        reciprocal_rank = None
        citation_match = None
        if gold_id and row.get("evidence_reference") and documents:
            ranked = reranker.rerank(question, documents, top_k=top_k)
            ranked_results = [
                {
                    "paper_id": paper_ids[documents.index(document)],
                    "score": round(float(score), 6),
                }
                for document, score in ranked if score > 0
            ]
            ranked_ids = [item["paper_id"] for item in ranked_results]
            eligible_retrieval += 1
            recall_hit = gold_id in ranked_ids
            retrieval_recalls.append(float(recall_hit))
            gold_rank = next(
                (position for position, paper_id in enumerate(ranked_ids, start=1)
                 if paper_id == gold_id),
                None,
            )
            reciprocal_rank = 1.0 / gold_rank if gold_rank else 0.0
            reciprocal_ranks.append(reciprocal_rank)
            if ranked_ids:
                citation_match = ranked_ids[0] == gold_id
                citation_correct += citation_match
                citations_total += 1

        modality = str(row.get("modal") or "").lower()
        selected_tool = "answer"
        expected_tool = "inspect_figures" if modality in {"figure", "table"} else None
        tool_selection_error = None
        if modality in {"figure", "table"}:
            try:
                decision = router.decide(question, [], tool_specs)
                selected_tool = decision.get("tool", "answer")
            except Exception as error:
                selected_tool = "error"
                tool_selection_error = f"{type(error).__name__}: {error}"
                tool_selection_errors += 1
            tool_correct += selected_tool == expected_tool
            tool_total += 1

        sample_results.append({
            "row_id": row.get("row_id"),
            "input": {
                "question": question,
                "modality": modality or None,
                "image_path": (row.get("figure") or {}).get("path"),
                "reference_paper_id": gold_id or None,
                "evidence_reference": row.get("evidence_reference") or None,
            },
            "output": {
                "retrieved_papers": ranked_results,
                "gold_rank": gold_rank,
                "recall_hit_at_k": recall_hit,
                "reciprocal_rank": round(reciprocal_rank, 6)
                if reciprocal_rank is not None else None,
                "top_citation_matches_reference": citation_match,
                "selected_tool": selected_tool,
                "expected_tool": expected_tool,
                "tool_selection_error": tool_selection_error,
            },
        })

    samples_file = None
    if samples_output_path:
        samples_file = Path(samples_output_path).expanduser().resolve()
        samples_file.parent.mkdir(parents=True, exist_ok=True)
        samples_file.write_text(
            json.dumps({
                "dataset": "M3SciQA",
                "split": split,
                "top_k": top_k,
                "tool_selection_provider": provider if llm is None else "injected",
                "tool_selection_model": getattr(router, "model", "injected evaluator"),
                "sample_count": len(sample_results),
                "note": "Answers are omitted; test answer labels are hidden.",
                "samples": sample_results,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return {
        "dataset": "M3SciQA",
        "split": split,
        "tool_selection_provider": provider if llm is None else "injected",
        "tool_selection_model": getattr(router, "model", "injected evaluator"),
        "examples_considered": len(rows),
        "retrieval_examples": eligible_retrieval,
        "retrieval_k": top_k,
        "recall_at_k": round(sum(retrieval_recalls) / eligible_retrieval, 4)
        if eligible_retrieval else None,
        "mrr_at_k": round(sum(reciprocal_ranks) / eligible_retrieval, 4)
        if eligible_retrieval else None,
        "citation_correctness": round(citation_correct / citations_total, 4)
        if citations_total else None,
        "citations_evaluated": citations_total,
        "tool_selection_examples": tool_total,
        "tool_selection_errors": tool_selection_errors,
        "tool_selection_accuracy": round(tool_correct / tool_total, 4)
        if tool_total else None,
        "citation_metric_note": (
            "Correctness is a paper-ID match against reference_arxiv_id; it does "
            "not establish semantic entailment."
        ),
        "tool_selection_note": (
            "Expected tool is inspect_figures for every row whose M3SciQA modality is "
            "figure or table; accuracy measures the configured LLM's first tool decision."
        ),
        "retrieval_corpus_note": (
            "Candidate text uses only public evidence_reference annotations grouped "
            "by reference_arxiv_id; answer/reference explanations are not indexed."
        ),
        "samples_output_file": str(samples_file) if samples_file else None,
    }