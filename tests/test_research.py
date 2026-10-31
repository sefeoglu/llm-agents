import json

from llm_agents.benchmark import evaluate_m3sciqa, evaluate_scifact
from llm_agents.evaluation import evaluate_m3sciqa_agent
from llm_agents.papers import ResearchPaper, parse_arxiv_feed, sample_papers
from llm_agents.research import ResearchAssistant


def test_arxiv_feed_parser_normalizes_record():
    feed = """<feed xmlns="http://www.w3.org/2005/Atom">
      <entry><id>https://arxiv.org/abs/2005.11401v2</id>
      <published>2020-05-22T00:00:00Z</published>
      <title> Retrieval-Augmented Generation </title>
      <summary>Uses retrieved passages.</summary>
      <author><name>Author One</name></author></entry>
    </feed>"""
    paper = parse_arxiv_feed(feed)[0]
    assert paper.paper_id == "2005.11401v2"
    assert paper.year == 2020
    assert paper.authors == ["Author One"]


def test_agent_selects_retrieval_trend_and_calculator():
    assistant = ResearchAssistant(sample_papers())
    retrieval = assistant.run("What does RAG use for knowledge-intensive tasks?")
    assert retrieval["trace"][0]["tool"] == "search"
    assert "[arXiv:2005.11401]" in retrieval["answer"]

    trend = assistant.run("How has the literature changed over time?")
    assert trend["trace"][0]["tool"] == "publication_trend"
    assert "not the field as a whole" in trend["answer"]

    calculation = assistant.run("Compute 12 * (3 + 4)")
    assert calculation["answer"] == "84"


def test_scifact_retrieval_metrics_use_gold_document_ids(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    claims_path = tmp_path / "claims_dev.jsonl"
    corpus_path.write_text(
        json.dumps({"doc_id": 12, "title": "Evidence paper", "abstract": ["Drug X reduces inflammation."]}) + "\n"
        + json.dumps({"doc_id": 19, "title": "Other paper", "abstract": ["Unrelated astronomy result."]}) + "\n",
        encoding="utf-8",
    )
    claims_path.write_text(
        json.dumps({"id": 1, "claim": "Drug X reduces inflammation", "cited_doc_ids": [12]}) + "\n",
        encoding="utf-8",
    )

    metrics = evaluate_scifact(str(corpus_path), str(claims_path), top_k=1)
    assert metrics == {
        "dataset": "SciFact dev",
        "claims_evaluated": 1,
        "top_k": 1,
        "hit_rate_at_k": 1.0,
        "recall_at_k": 1.0,
        "mrr_at_k": 1.0,
    }


def test_trend_tool_reports_small_corpus_scope():
    papers = [
        ResearchPaper(str(year), "Paper", [], year, "text", "https://example.test")
        for year in (2020, 2021, 2021)
    ]
    trend = ResearchAssistant(papers).publication_trend()
    assert trend["year_counts"] == {2020: 1, 2021: 2}
    assert "collected corpus" in trend["summary"]


def test_m3sciqa_evaluation_uses_local_image_and_answer(tmp_path):
    import base64

    image_path = tmp_path / "images" / "validation" / "00000.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/"
        "5Z8AAAAASUVORK5CYII="
    ))
    row = {
        "row_id": 0,
        "question": "Which curve is higher?",
        "answer": "The blue curve is higher.",
        "figure": {"path": "images/validation/00000.png"},
        "modal": "figure",
        "anchor_arxiv_id": "1234.5678",
        "reference_arxiv_id": "2345.6789",
    }
    (tmp_path / "validation.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    calls = []

    def fake_answer(image, prompt, model):
        calls.append((image, prompt, model))
        return "The blue curve is higher."

    metrics = evaluate_m3sciqa(
        str(tmp_path), "fake-local-vlm", limit=1, answer_fn=fake_answer,
        output_path=str(tmp_path / "saved" / "predictions.jsonl"),
    )
    assert len(calls) == 1
    assert "Which curve is higher?" in calls[0][1]
    assert "blue curve" not in calls[0][1]
    assert metrics["examples_evaluated"] == 1
    assert metrics["image_examples"] == 1
    assert metrics["exact_match"] == 1.0
    assert metrics["token_f1"] == 1.0
    assert metrics["by_modality"]["figure"]["examples"] == 1
    assert metrics["predictions_file"] == str(
        (tmp_path / "saved" / "predictions.jsonl").resolve()
    )
    saved = json.loads((tmp_path / "saved" / "predictions.jsonl").read_text().strip())
    assert saved["question"] == "Which curve is higher?"
    assert saved["prediction"] == "The blue curve is higher."
    assert saved["reference_answer"] == "The blue curve is higher."
    assert saved["image_path"] == "images/validation/00000.png"
    assert saved["token_f1"] == 1.0


def test_m3sciqa_evaluator_uses_local_vision_model_environment(tmp_path, monkeypatch):
    import base64

    image_path = tmp_path / "images" / "validation" / "00000.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/"
        "5Z8AAAAASUVORK5CYII="
    ))
    row = {
        "row_id": 0,
        "question": "Read the chart.",
        "answer": "Increasing.",
        "figure": {"path": "images/validation/00000.png"},
        "modal": "figure",
    }
    (tmp_path / "validation.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    observed_models = []
    monkeypatch.setenv("OLLAMA_VISION_MODEL", "local-vision-model")

    metrics = evaluate_m3sciqa(
        str(tmp_path), None, limit=1,
        answer_fn=lambda _image, _prompt, model: observed_models.append(model) or "Increasing",
    )
    assert observed_models == ["local-vision-model"]
    assert metrics["vision_model"] == "local-vision-model"


def test_m3sciqa_test_uses_validation_examples_without_test_answer_leakage(tmp_path):
    import base64

    image_root = tmp_path / "images"
    for split, filename in (("validation", "val.png"), ("test", "test.png")):
        split_dir = image_root / split
        split_dir.mkdir(parents=True)
        (split_dir / filename).write_bytes(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/"
            "5Z8AAAAASUVORK5CYII="
        ))
    validation = {
        "row_id": 4,
        "question": "What does the chart show about temperature change?",
        "answer": "Temperature increases over time.",
        "figure": {"path": "images/validation/val.png"},
        "modal": "figure",
    }
    test = {
        "row_id": 8,
        "question": "What does this chart indicate about temperature change?",
        "answer": "Secret test gold answer.",
        "figure": {"path": "images/test/test.png"},
        "modal": "figure",
    }
    (tmp_path / "validation.jsonl").write_text(json.dumps(validation) + "\n", encoding="utf-8")
    (tmp_path / "test.jsonl").write_text(json.dumps(test) + "\n", encoding="utf-8")
    observed = {}

    def fake_answer(images, prompt, model):
        observed.update(images=images, prompt=prompt, model=model)
        return "Temperature rises."

    metrics = evaluate_m3sciqa(
        str(tmp_path), "fake-local-vlm", split="test", limit=1, answer_fn=fake_answer
    )
    assert len(observed["images"]) == 2
    assert observed["images"][0].endswith("val.png")
    assert observed["images"][1].endswith("test.png")
    assert validation["question"] in observed["prompt"]
    assert validation["answer"] in observed["prompt"]
    assert test["question"] in observed["prompt"]
    assert test["answer"] not in observed["prompt"]
    assert "Target image is attached in image position 2" in observed["prompt"]
    assert metrics["augmentation"] == {
        "source_split": "validation",
        "examples_per_test_case": 2,
    }


def test_unlabeled_m3sciqa_test_rows_are_not_scored(tmp_path):
    import base64

    for split, filename in (("validation", "val.png"), ("test", "test.png")):
        split_dir = tmp_path / "images" / split
        split_dir.mkdir(parents=True)
        (split_dir / filename).write_bytes(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/"
            "5Z8AAAAASUVORK5CYII="
        ))
    validation = {
        "row_id": 0,
        "question": "How does the plotted value change?",
        "answer": "It increases.",
        "figure": {"path": "images/validation/val.png"},
        "modal": "figure",
    }
    test = {
        "row_id": 1,
        "question": "What is shown in this test figure?",
        "answer": "",
        "figure": {"path": "images/test/test.png"},
        "modal": "figure",
    }
    (tmp_path / "validation.jsonl").write_text(json.dumps(validation) + "\n", encoding="utf-8")
    (tmp_path / "test.jsonl").write_text(json.dumps(test) + "\n", encoding="utf-8")
    output_path = tmp_path / "predictions.jsonl"

    metrics = evaluate_m3sciqa(
        str(tmp_path), "fake-local-vlm", split="test", limit=1,
        answer_fn=lambda _images, _prompt, _model: "A rising curve.",
        output_path=str(output_path),
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert metrics["examples_evaluated"] == 1
    assert metrics["labeled_examples"] == 0
    assert metrics["exact_match"] is None
    assert metrics["token_f1"] is None
    assert "no public answer labels" in metrics["scoring_note"]
    assert metrics["by_modality"]["figure"] == {
        "examples": 1,
        "labeled_examples": 0,
        "token_f1": None,
    }
    assert saved["reference_answer"] is None
    assert saved["exact_match"] is None
    assert saved["token_f1"] is None


def test_m3sciqa_agent_metrics_cover_retrieval_citations_and_tools(tmp_path):
    rows = [
        {
            "row_id": 0,
            "question": "What does the figure show about lithium capacity fade?",
            "answer": "SECRET ANSWER MUST NOT BE INDEXED",
            "evidence_reference": "Lithium capacity fade increases during rapid charging.",
            "reference_arxiv_id": "1111.00001",
            "anchor_arxiv_id": "1111.00002",
            "modal": "figure",
        },
        {
            "row_id": 1,
            "question": "What does the table show about temperature and battery degradation?",
            "answer": "ANOTHER SECRET ANSWER",
            "evidence_reference": "Higher temperature accelerates battery degradation and capacity loss.",
            "reference_arxiv_id": "2222.00001",
            "anchor_arxiv_id": "2222.00002",
            "modal": "table",
        },
    ]
    (tmp_path / "validation.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    samples_path = tmp_path / "results" / "samples.json"
    metrics = evaluate_m3sciqa_agent(
        str(tmp_path), top_k=5, samples_output_path=str(samples_path)
    )

    assert metrics["examples_considered"] == 2
    assert metrics["retrieval_examples"] == 2
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr_at_k"] == 1.0
    assert metrics["citation_correctness"] == 1.0
    assert metrics["tool_selection_examples"] == 2
    assert metrics["tool_selection_accuracy"] == 1.0
    assert "does not establish semantic entailment" in metrics["citation_metric_note"]
    saved = json.loads(samples_path.read_text(encoding="utf-8"))
    assert saved["sample_count"] == 2
    assert saved["samples"][0]["input"]["question"] == rows[0]["question"]
    assert saved["samples"][0]["output"]["selected_tool"] == "inspect_figures"
    assert saved["samples"][0]["output"]["retrieved_papers"][0]["paper_id"] == "1111.00001"
    assert "answer" not in saved["samples"][0]["input"]


def test_m3sciqa_agent_eval_allows_test_annotations_without_answer_labels(tmp_path):
    row = {
        "question": "What does the figure show about cell capacity fade?",
        "answer": "",
        "evidence_reference": "Cell capacity declines during aging.",
        "reference_arxiv_id": "1111.00001",
        "modal": "figure",
    }
    (tmp_path / "test.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    metrics = evaluate_m3sciqa_agent(str(tmp_path), split="test")

    assert metrics["split"] == "test"
    assert metrics["retrieval_examples"] == 1
    assert metrics["recall_at_k"] == 1.0
    assert metrics["tool_selection_examples"] == 1