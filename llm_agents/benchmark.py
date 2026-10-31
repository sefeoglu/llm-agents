"""Local scientific benchmark downloads and evaluation."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
import re
import shutil
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from .reranker import Reranker
from .vision import analyze_images_with_ollama

SCIFACT_ARCHIVE = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
M3SCIQA_DATASET = "yale-nlp/M3SciQA"
M3SCIQA_ROWS_API = "https://datasets-server.huggingface.co/rows"
M3SCIQA_SPLITS = {"validation": 452, "test": 1000}


def download_scifact(data_dir: str) -> None:
    target_dir = Path(data_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    wanted = {"corpus.jsonl", "claims_dev.jsonl"}
    existing = [target_dir / name for name in wanted if (target_dir / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing files: {existing}")
    request = urllib.request.Request(
        SCIFACT_ARCHIVE, headers={"User-Agent": "scientific-research-agent/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        archive_bytes = response.read()
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        for member in archive.getmembers():
            basename = Path(member.name).name
            if basename not in wanted or not member.isfile():
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            with source, (target_dir / basename).open("wb") as destination:
                shutil.copyfileobj(source, destination)
    missing = [name for name in wanted if not (target_dir / name).exists()]
    if missing:
        raise ValueError(f"SciFact archive is missing expected files: {missing}")


def _read_jsonl(path: str) -> List[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def download_m3sciqa(data_dir: str,
                     splits: Sequence[str] = ("validation", "test")) -> dict:
    """Download M3SciQA records and image files into a persistent local directory."""
    target_dir = Path(data_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    requested = list(dict.fromkeys(splits))
    unknown = set(requested) - set(M3SCIQA_SPLITS)
    if unknown:
        raise ValueError(f"Unsupported M3SciQA splits: {sorted(unknown)}")
    downloaded = {}
    for split in requested:
        output_path = target_dir / f"{split}.jsonl"
        if output_path.exists():
            records = _read_jsonl(str(output_path))
            missing_images = [
                row.get("figure", {}).get("path", "")
                for row in records
                if not row.get("figure", {}).get("path")
                or not (target_dir / row["figure"]["path"]).is_file()
            ]
            if missing_images:
                raise FileExistsError(
                    f"Local {split} split is incomplete; missing images: "
                    f"{missing_images[:5]}"
                )
            downloaded[split] = {
                "status": "already local",
                "records": len(records),
                "images": len(records) - len(missing_images),
                "jsonl": str(output_path),
                "image_dir": str(target_dir / "images" / split),
            }
            continue

        records = []
        offset = 0
        total = None
        pending_images = []
        image_dir = target_dir / "images" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        while total is None or offset < total:
            query = urllib.parse.urlencode({
                "dataset": M3SCIQA_DATASET,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": 100,
            })
            request = urllib.request.Request(
                f"{M3SCIQA_ROWS_API}?{query}",
                headers={"User-Agent": "scientific-research-agent/1.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                page = json.loads(response.read().decode("utf-8"))
            total = page["num_rows_total"]
            rows = page.get("rows", [])
            if not rows:
                break
            for item in rows:
                row = item["row"]
                row_index = item.get("row_idx", len(records))
                figure = row.get("figure") or {}
                image_url = figure.get("src")
                local_relative = f"images/{split}/{row_index:05}.png" if image_url else ""
                row["figure"] = {
                    "path": local_relative,
                    "width": figure.get("width"),
                    "height": figure.get("height"),
                }
                row["row_id"] = row_index
                row["split"] = split
                records.append(row)
                if image_url:
                    pending_images.append((image_url, target_dir / local_relative))
            offset += len(rows)

        if not records:
            raise ValueError(f"No records returned for M3SciQA split {split}")

        def save_image(task):
            image_url, destination = task
            request = urllib.request.Request(
                image_url, headers={"User-Agent": "scientific-research-agent/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                destination.write_bytes(response.read())

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(save_image, pending_images))

        with output_path.open("w", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        downloaded[split] = {
            "status": "downloaded",
            "records": len(records),
            "images": len(pending_images),
            "jsonl": str(output_path),
            "image_dir": str(image_dir),
        }
    return downloaded


def _normalize_answer(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def _token_f1(prediction: str, reference: str) -> float:
    predicted_tokens = Counter(_normalize_answer(prediction).split())
    reference_tokens = Counter(_normalize_answer(reference).split())
    overlap = sum((predicted_tokens & reference_tokens).values())
    if not overlap:
        return 1.0 if not predicted_tokens and not reference_tokens else 0.0
    precision = overlap / sum(predicted_tokens.values())
    recall = overlap / sum(reference_tokens.values())
    return 2 * precision * recall / (precision + recall)


def evaluate_m3sciqa(data_dir: str, model: Optional[str], split: str = "validation",
                     limit: int = 0,
                     answer_fn: Optional[Callable[[Sequence[str], str, str], str]] = None,
                     output_path: Optional[str] = None,
                     augmentation_k: int = 2) -> dict:
    """Evaluate local image-grounded QA outputs against M3SciQA answers."""
    if split not in M3SCIQA_SPLITS:
        raise ValueError(f"Unsupported M3SciQA split: {split}")
    if augmentation_k < 0:
        raise ValueError("augmentation_k cannot be negative")
    model = model or os.environ.get("OLLAMA_VISION_MODEL")
    if not model and answer_fn is None:
        raise ValueError("Set a local vision model with --vision-model or OLLAMA_VISION_MODEL")
    target_dir = Path(data_dir).expanduser().resolve()
    records = _read_jsonl(str(target_dir / f"{split}.jsonl"))
    if limit:
        records = records[:limit]
    validation_records = (
        _read_jsonl(str(target_dir / "validation.jsonl"))
        if split == "test" and augmentation_k
        else []
    )
    run_answer = answer_fn or analyze_images_with_ollama
    scored = []
    output_stream = None
    output_file = Path(output_path).expanduser().resolve() if output_path else None
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_stream = output_file.open("x", encoding="utf-8")
    try:
        for row in records:
            figure = row.get("figure") or {}
            relative_image_path = figure.get("path", "")
            image_path = target_dir / relative_image_path
            if not relative_image_path or not image_path.is_file():
                raise FileNotFoundError(
                    f"Local M3SciQA image missing for row {row.get('row_id')}: {image_path}"
                )
            augmentation_examples = []
            if validation_records:
                candidate_docs = []
                for index, example in enumerate(validation_records):
                    example_figure = example.get("figure") or {}
                    example_image = target_dir / example_figure.get("path", "")
                    if example_image.is_file():
                        candidate_docs.append((
                            index,
                            f"question: {example['question']} answer: {example['answer']}",
                        ))
                ranked = Reranker().rerank(
                    row["question"],
                    [f"[validation row {index}] {text}" for index, text in candidate_docs],
                    top_k=augmentation_k,
                )
                chosen_ids = {
                    int(document.split("]", 1)[0].split()[-1])
                    for document, _score in ranked
                }
                augmentation_examples = [
                    validation_records[index]
                    for index, _text in candidate_docs
                    if index in chosen_ids
                ]

            prompt_lines = [
                "Answer the target scientific question using its figure or table "
                "and the question. Use the validation examples only as demonstrations "
                "of how to read scientific visuals; do not copy their answers unless "
                "the target image supports them. Do not assume the target reference answer.",
            ]
            for example_number, example in enumerate(augmentation_examples, start=1):
                prompt_lines.extend([
                    f"Validation example {example_number} image is attached in image "
                    f"position {example_number}.",
                    f"Question: {example['question']}",
                    f"Answer: {example['answer']}",
                ])
            target_image_position = len(augmentation_examples) + 1
            prompt_lines.extend([
                f"Target image is attached in image position {target_image_position}.",
                f"Target modality: {row.get('modal', 'unknown')}",
                f"Target question: {row['question']}",
                "Return only a concise answer grounded in the target image and relevant "
                "scientific context.",
            ])
            prompt = "\n".join(prompt_lines)
            image_paths = [
                str(target_dir / (example.get("figure") or {}).get("path", ""))
                for example in augmentation_examples
            ] + [str(image_path)]
            prediction = run_answer(image_paths, prompt, model or "")
            reference = row["answer"]
            modality = row.get("modal", "unknown")
            item = {
                "prediction": prediction,
                "reference": reference,
                "modal": modality,
                "augmentation_examples": augmentation_examples,
            }
            scored.append(item)
            if output_stream:
                saved_record = {
                    "dataset": "M3SciQA",
                    "split": split,
                    "row_id": row.get("row_id"),
                    "vision_model": model or "injected evaluator",
                    "question": row["question"],
                    "prediction": prediction,
                    "reference_answer": reference or None,
                    "modality": modality,
                    "anchor_arxiv_id": row.get("anchor_arxiv_id"),
                    "reference_arxiv_id": row.get("reference_arxiv_id"),
                    "image_path": relative_image_path,
                    "augmentation_row_ids": [
                        example.get("row_id") for example in augmentation_examples
                    ],
                    "exact_match": (
                        int(_normalize_answer(prediction) == _normalize_answer(reference))
                        if reference.strip() else None
                    ),
                    "token_f1": (
                        round(_token_f1(prediction, reference), 4)
                        if reference.strip() else None
                    ),
                }
                output_stream.write(json.dumps(saved_record, ensure_ascii=False) + "\n")
                output_stream.flush()
    finally:
        if output_stream:
            output_stream.close()
    if not scored:
        raise ValueError(f"No M3SciQA records found for split {split}")

    labeled = [item for item in scored if item["reference"].strip()]
    modal_scores = defaultdict(list)
    for item in scored:
        modal_scores[item["modal"]].append(item)
    metrics = {
        "dataset": "M3SciQA",
        "split": split,
        "examples_evaluated": len(scored),
        "labeled_examples": len(labeled),
        "vision_model": model or "injected evaluator",
        "augmentation": {
            "source_split": "validation" if validation_records else None,
            "examples_per_test_case": augmentation_k if validation_records else 0,
        },
        "exact_match": round(sum(
            _normalize_answer(item["prediction"]) == _normalize_answer(item["reference"])
            for item in labeled
        ) / len(labeled), 4) if labeled else None,
        "token_f1": round(sum(
            _token_f1(item["prediction"], item["reference"]) for item in labeled
        ) / len(labeled), 4) if labeled else None,
        "image_examples": sum(bool(item) for item in scored),
        "by_modality": {
            modality: {
                "examples": len(items),
                "labeled_examples": sum(bool(item["reference"].strip()) for item in items),
                "token_f1": round(sum(
                    _token_f1(item["prediction"], item["reference"])
                    for item in items if item["reference"].strip()
                ) / sum(bool(item["reference"].strip()) for item in items), 4)
                if any(item["reference"].strip() for item in items) else None,
            }
            for modality, items in sorted(modal_scores.items())
        },
    }
    if not labeled:
        metrics["scoring_note"] = (
            "This split has no public answer labels; predictions are saved but "
            "exact match and token F1 are not computable."
        )
    if output_file:
        metrics["predictions_file"] = str(output_file)
    return metrics


def evaluate_scifact(corpus_path: str, claims_path: str, top_k: int = 5,
                     limit: int = 0) -> dict:
    corpus = _read_jsonl(corpus_path)
    claims = _read_jsonl(claims_path)
    documents: List[str] = []
    document_ids: List[str] = []
    for document in corpus:
        doc_id = str(document["doc_id"])
        title = document.get("title", "")
        abstract = " ".join(document.get("abstract", []))
        documents.append(f"[SciFact:{doc_id}] {title}\n{abstract}")
        document_ids.append(doc_id)
    id_by_document = dict(zip(documents, document_ids))
    reranker = Reranker()
    eligible = [claim for claim in claims if claim.get("cited_doc_ids")]
    if limit:
        eligible = eligible[:limit]
    if not eligible:
        raise ValueError("No evidence-bearing SciFact claims were found")

    hits = 0
    reciprocal_ranks = []
    recall_scores = []
    for claim in eligible:
        gold_ids = {str(doc_id) for doc_id in claim["cited_doc_ids"]}
        ranked = reranker.rerank(claim["claim"], documents, top_k)
        ranked_ids = [id_by_document[document] for document, _ in ranked]
        retrieved = set(ranked_ids)
        hits += bool(gold_ids & retrieved)
        recall_scores.append(len(gold_ids & retrieved) / len(gold_ids))
        reciprocal_ranks.append(
            1 / next((rank for rank, doc_id in enumerate(ranked_ids, 1)
                      if doc_id in gold_ids), float("inf"))
        )
    count = len(eligible)
    return {
        "dataset": "SciFact dev",
        "claims_evaluated": count,
        "top_k": top_k,
        "hit_rate_at_k": round(hits / count, 4),
        "recall_at_k": round(sum(recall_scores) / count, 4),
        "mrr_at_k": round(sum(reciprocal_ranks) / count, 4),
    }