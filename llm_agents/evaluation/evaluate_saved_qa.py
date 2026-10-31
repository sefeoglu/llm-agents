"""Recompute QA metrics from saved prediction/reference JSONL without model calls."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from llm_agents.benchmark import QA_SCORE_KEYS, _qa_text_metrics


def _mean_metrics(rows: list[dict]) -> dict:
    means = {}
    for key in QA_SCORE_KEYS:
        values = [row[key] for row in rows if row.get(key) is not None]
        means[key] = round(sum(values) / len(values), 4) if values else None
    return means


def main() -> None:
    default_predictions = (
        REPOSITORY_ROOT
        / "results/qa/validation/rag-predictions.jsonl"
    )
    default_output = (
        REPOSITORY_ROOT / "llm_agents/evaluation/validation/qa/metrics.json"
    )
    parser = argparse.ArgumentParser(
        description="Compute EM, token F1, and ROUGE-L F1 for saved QA outputs."
    )
    parser.add_argument("--predictions", type=Path, default=default_predictions)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument(
        "--scored-predictions", type=Path,
        help="Optional JSONL output containing each row's recomputed metrics",
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        parser.error(f"No prediction rows found in {args.predictions}")

    scored = []
    by_modality = defaultdict(list)
    retrieval_hits = []
    retrieval_reciprocal_ranks = []
    top_citation_matches = []
    for row in rows:
        prediction = str(row.get("prediction") or "")
        reference = str(row.get("reference_answer") or "")
        if reference.strip():
            row.update(_qa_text_metrics(prediction, reference))
            scored.append(row)
            by_modality[str(row.get("modality") or "unknown")].append(row)
        else:
            row.update({key: None for key in QA_SCORE_KEYS})

        retrieved = row.get("retrieved_evidence") or []
        reference_paper_id = str(row.get("reference_arxiv_id") or "").strip()
        retrieved_paper_ids = [str(item.get("paper_id") or "") for item in retrieved]
        if reference_paper_id and retrieved_paper_ids:
            hit = reference_paper_id in retrieved_paper_ids
            retrieval_hits.append(float(hit))
            rank = retrieved_paper_ids.index(reference_paper_id) + 1 if hit else None
            retrieval_reciprocal_ranks.append(1 / rank if rank else 0.0)
            top_citation_matches.append(float(retrieved_paper_ids[0] == reference_paper_id))
            row["evidence_hit_at_k"] = hit
            row["evidence_reciprocal_rank"] = 1 / rank if rank else 0.0
            row["top_citation_matches_reference"] = retrieved_paper_ids[0] == reference_paper_id
        else:
            row["evidence_hit_at_k"] = None
            row["evidence_reciprocal_rank"] = None
            row["top_citation_matches_reference"] = None

    modality_metrics = {}
    for modality in sorted({str(row.get("modality") or "unknown") for row in rows}):
        all_rows = [
            row for row in rows
            if str(row.get("modality") or "unknown") == modality
        ]
        labeled_rows = by_modality[modality]
        modality_metrics[modality] = {
            "examples": len(all_rows),
            "labeled_examples": len(labeled_rows),
            **_mean_metrics(labeled_rows),
        }

    retrieved_evidence = [
        evidence
        for row in rows
        for evidence in row.get("retrieved_evidence", [])
    ]
    retrieved_papers = {
        evidence.get("paper_id")
        for evidence in retrieved_evidence
        if evidence.get("paper_id")
    }
    augmentation_rows = [
        row.get("augmentation_row_ids", [])
        for row in rows
    ]
    metrics = {
        "dataset": rows[0].get("dataset"),
        "split": rows[0].get("split"),
        "vision_provider": rows[0].get("vision_provider"),
        "vision_model": rows[0].get("vision_model"),
        "examples_evaluated": len(rows),
        "labeled_examples": len(scored),
        **_mean_metrics(scored),
        "image_examples": sum(bool(row.get("image_path")) for row in rows),
        "augmentation": {
            "source_split": "validation" if any(augmentation_rows) else None,
            "examples_per_test_case": max(
                (len(row_ids) for row_ids in augmentation_rows), default=0
            ),
        },
        "metric_definitions": {
            "normalization": "lowercase alphanumeric tokens; punctuation is discarded",
            "token_precision_recall_f1": "multiset unigram overlap",
            "rouge_1_2": "n-gram overlap precision, recall, and F1 for n=1 and n=2",
            "rouge_l": "longest-common-subsequence precision, recall, and F1 over normalized tokens",
            "bleu_1_to_4": "mean sentence-level cumulative BLEU over examples, add-one-smoothed n-gram precision, brevity penalty applied",
            "chrf": "mean sentence-level character n-gram F-beta, character orders 1-6, beta=2, normalized text",
            "length": "normalized whitespace token counts and prediction/reference count ratio",
        },
        "rag": {
            "enabled": bool(retrieved_evidence),
            "top_k": max(
                (len(row.get("retrieved_evidence", [])) for row in rows), default=0
            ),
            "examples_with_retrieved_evidence": sum(
                bool(row.get("retrieved_evidence")) for row in rows
            ),
            "unique_retrieved_papers": len(retrieved_papers),
            "corpus": (
                "saved retrieved_evidence per prediction"
                if retrieved_evidence else None
            ),
        },
        "evidence_retrieval": {
            "eligible_examples": len(retrieval_hits),
            "hit_at_k": round(sum(retrieval_hits) / len(retrieval_hits), 4)
            if retrieval_hits else None,
            "mrr_at_k": round(
                sum(retrieval_reciprocal_ranks) / len(retrieval_reciprocal_ranks), 4
            ) if retrieval_reciprocal_ranks else None,
            "top_citation_id_match": round(
                sum(top_citation_matches) / len(top_citation_matches), 4
            ) if top_citation_matches else None,
            "note": (
                "These metrics compare retrieved paper IDs against reference_arxiv_id; "
                "they do not assess semantic entailment."
            ),
        },
        "answer_length": {
            "mean_prediction_tokens": round(
                sum(row["prediction_token_count"] for row in scored) / len(scored), 2
            ) if scored else None,
            "mean_reference_tokens": round(
                sum(row["reference_token_count"] for row in scored) / len(scored), 2
            ) if scored else None,
            "mean_prediction_reference_length_ratio": round(
                sum(row["answer_length_ratio"] for row in scored
                    if row["answer_length_ratio"] is not None)
                / sum(row["answer_length_ratio"] is not None for row in scored), 4
            ) if any(row["answer_length_ratio"] is not None for row in scored) else None,
        },
        "by_modality": modality_metrics,
        "source_predictions": str(args.predictions.resolve()),
    }

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.scored_predictions:
        args.scored_predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.scored_predictions.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Metrics saved to {output_path}")
    if args.scored_predictions:
        print(f"Scored predictions saved to {args.scored_predictions}")


if __name__ == "__main__":
    main()
