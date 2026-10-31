import json
import pytest

from llm_agents.benchmark import (
    _chrf,
    _qa_text_metrics,
    _rouge_l_f1,
    _sentence_bleu_scores,
    evaluate_m3sciqa,
)
from llm_agents.evaluation import evaluate_m3sciqa_agent
from llm_agents.llm import LLM, OllamaToolCallingLLM, OpenAIToolCallingLLM
from llm_agents.papers import ResearchPaper, parse_arxiv_feed, sample_papers
from llm_agents.research import ResearchAssistant
from llm_agents.vision import analyze_images_with_openai


class ScriptedLLM(LLM):
    def __init__(self, decisions):
        self.decisions = iter(decisions)

    def decide(self, question, observations, tools=None):
        return next(self.decisions)


class FigureToolLLM(LLM):
    def decide(self, question, observations, tools=None):
        return {"tool": "inspect_figures", "arguments": {"query": question}}


class FailingLLM(LLM):
    def decide(self, question, observations, tools=None):
        raise ValueError("empty model response")


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
    assistant = ResearchAssistant(sample_papers(), llm=ScriptedLLM([
        {"tool": "search", "arguments": {"query": "RAG knowledge-intensive tasks"}},
        {"answer": "[arXiv:2005.11401] Retrieval-Augmented Generation evidence"},
        {"tool": "publication_trend", "arguments": {}},
        {"answer": "The trend describes this corpus, not the field as a whole."},
        {"tool": "calculator", "arguments": {"expression": "12 * (3 + 4)"}},
        {"answer": "84"},
    ]))
    retrieval = assistant.run("What does RAG use for knowledge-intensive tasks?")
    assert retrieval["trace"][0]["tool"] == "search"
    assert "[arXiv:2005.11401]" in retrieval["answer"]

    trend = assistant.run("How has the literature changed over time?")
    assert trend["trace"][0]["tool"] == "publication_trend"
    assert "not the field as a whole" in trend["answer"]

    calculation = assistant.run("Compute 12 * (3 + 4)")
    assert calculation["answer"] == "84"


def test_research_assistant_can_select_openai_agent_provider(monkeypatch):
    selected = {}

    class FakeOpenAI(LLM):
        def __init__(self, model=None):
            selected["model"] = model

        def decide(self, question, observations, tools=None):
            return {"answer": "test response"}

    monkeypatch.setattr("llm_agents.research.OpenAIToolCallingLLM", FakeOpenAI)
    assistant = ResearchAssistant(
        sample_papers(), agent_provider="openai", agent_model="configured-gpt"
    )

    assert isinstance(assistant.agent.llm, FakeOpenAI)
    assert selected["model"] == "configured-gpt"


def test_ollama_llm_sends_tool_schemas_and_parses_tool_calls(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"message": {"tool_calls": [{"function": {
                "name": "search",
                "arguments": {"query": "battery degradation"},
            }}]}}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("llm_agents.llm.urllib.request.urlopen", fake_urlopen)
    tool_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    decision = OllamaToolCallingLLM("test-model").decide(
        "Find battery papers", [], [{
            "name": "search",
            "description": "Search the paper corpus",
            "parameters": tool_schema,
        }],
    )

    assert decision == {
        "tool": "search", "arguments": {"query": "battery degradation"},
    }
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["payload"]["model"] == "test-model"
    assert captured["payload"]["tools"][0]["function"]["parameters"] == tool_schema
    assert captured["timeout"] == 120


def test_openai_agent_sends_tool_schemas_and_parses_function_call(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"tool_calls": [{
                "id": "call-search-1",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": '{"query":"battery aging"}',
                },
            }]}}]}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("llm_agents.llm.urllib.request.urlopen", fake_urlopen)
    spec = {
        "name": "search",
        "description": "Search research papers",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
    decision = OpenAIToolCallingLLM("gpt-4.1-mini", api_key="test-key").decide(
        "Find battery aging evidence", [], [spec]
    )

    assert decision == {
        "tool": "search",
        "arguments": {"query": "battery aging"},
        "tool_call_id": "call-search-1",
    }
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-key"
    assert captured["payload"]["model"] == "gpt-4.1-mini"
    assert captured["payload"]["tools"][0]["function"]["parameters"] == spec["parameters"]
    assert captured["payload"]["parallel_tool_calls"] is False
    assert captured["timeout"] == 120


def test_openai_agent_rejects_batched_calls(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"tool_calls": [
                {"id": "call-search", "function": {
                    "name": "search", "arguments": '{"query":"battery"}',
                }},
                {"id": "call-calc", "function": {
                    "name": "calculator", "arguments": '{"expression":"2 + 2"}',
                }},
            ]}}]}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("llm_agents.llm.urllib.request.urlopen", fake_urlopen)
    tools = [
        {"name": "search", "description": "Search", "parameters": {"type": "object", "properties": {}}},
        {"name": "calculator", "description": "Calculate", "parameters": {"type": "object", "properties": {}}},
    ]
    with pytest.raises(ValueError, match="one adaptive action"):
        OpenAIToolCallingLLM("gpt-4.1-mini", api_key="test-key").decide(
            "Search and calculate", [], tools
        )

    assert captured["payload"]["parallel_tool_calls"] is False


def test_rouge_l_f1_rewards_in_order_overlap():
    assert _rouge_l_f1("alpha beta gamma", "alpha x beta gamma") == pytest.approx(0.8571, abs=1e-4)
    assert _rouge_l_f1("alpha beta gamma", "gamma beta alpha") == pytest.approx(1 / 3)
    assert _rouge_l_f1("alpha", "omega") == 0.0
    assert _rouge_l_f1("", "") == 1.0


def test_qa_text_metrics_include_standard_overlap_and_length_scores():
    scores = _qa_text_metrics("alpha beta gamma", "alpha beta delta")
    assert scores["exact_match"] == 0
    assert scores["token_precision"] == pytest.approx(2 / 3)
    assert scores["token_recall"] == pytest.approx(2 / 3)
    assert scores["token_f1"] == pytest.approx(2 / 3)
    assert scores["rouge_1_f1"] == pytest.approx(scores["token_f1"])
    assert scores["rouge_2_f1"] == 0.5
    assert scores["rouge_l_f1"] == pytest.approx(2 / 3)
    assert 0 <= scores["bleu_4"] <= scores["bleu_1"] <= 1
    assert 0 <= scores["chrf"] <= 1
    assert scores["prediction_token_count"] == 3
    assert scores["reference_token_count"] == 3
    assert scores["answer_length_ratio"] == 1.0
    assert _chrf("same words", "same words") == 1.0
    assert set(_sentence_bleu_scores([], ["empty"])) == {
        "bleu_1", "bleu_2", "bleu_3", "bleu_4",
    }


def test_openai_vision_adapter_sends_local_image_and_parses_response(tmp_path, monkeypatch):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"png-bytes")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {
                "content": "The chart shows an upward trend."
            }}]}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("llm_agents.vision.urllib.request.urlopen", fake_urlopen)
    answer = analyze_images_with_openai([str(image_path)], "Read the chart", "gpt-4o-mini")

    assert answer == "The chart shows an upward trend."
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-key"
    assert captured["payload"]["model"] == "gpt-4o-mini"
    assert captured["payload"]["messages"][0]["content"][0] == {
        "type": "text", "text": "Read the chart",
    }
    assert captured["payload"]["messages"][0]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert captured["timeout"] == 120


def test_openai_vision_adapter_requires_api_key(tmp_path, monkeypatch):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"png-bytes")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        analyze_images_with_openai([str(image_path)], "Read the chart", "gpt-4o-mini")


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
    assert saved["rouge_l_f1"] == 1.0
    assert metrics["rouge_l_f1"] == 1.0
    assert metrics["by_modality"]["figure"]["rouge_l_f1"] == 1.0


def test_m3sciqa_evaluation_supports_openai_provider(tmp_path, monkeypatch):
    image_path = tmp_path / "images" / "validation" / "00000.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image-bytes")
    row = {
        "row_id": 0,
        "question": "What trend is shown?",
        "answer": "An upward trend.",
        "figure": {"path": "images/validation/00000.png"},
        "modal": "figure",
    }
    (tmp_path / "validation.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    observed = {}
    monkeypatch.delenv("OPENAI_VISION_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    metrics = evaluate_m3sciqa(
        str(tmp_path), None, limit=1, vision_provider="openai",
        answer_fn=lambda images, prompt, model: (
            observed.update(images=images, prompt=prompt, model=model)
            or "An upward trend."
        ),
        output_path=str(tmp_path / "predictions.jsonl"),
    )

    saved = json.loads((tmp_path / "predictions.jsonl").read_text(encoding="utf-8"))
    assert observed["model"] == "gpt-4o-mini"
    assert metrics["vision_provider"] == "openai"
    assert metrics["vision_model"] == "gpt-4o-mini"
    assert saved["vision_provider"] == "openai"


def test_m3sciqa_qa_rag_includes_retrieved_evidence_not_reference_answer(tmp_path):
    image_path = tmp_path / "images" / "validation" / "00000.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image-bytes")
    row = {
        "row_id": 0,
        "question": "How does elevated temperature affect cell capacity?",
        "answer": "SECRET GOLD ANSWER",
        "evidence_reference": "Elevated temperature accelerates cell capacity fade during cycling.",
        "reference_arxiv_id": "1234.56789",
        "figure": {"path": "images/validation/00000.png"},
        "modal": "figure",
    }
    (tmp_path / "validation.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    observed = {}

    def fake_answer(images, prompt, model):
        observed.update(images=images, prompt=prompt, model=model)
        return "Elevated temperature accelerates capacity fade."

    output_path = tmp_path / "predictions.jsonl"
    metrics = evaluate_m3sciqa(
        str(tmp_path), "fake-model", limit=1, answer_fn=fake_answer,
        output_path=str(output_path), rag_top_k=2,
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert "Elevated temperature accelerates cell capacity fade during cycling." in observed["prompt"]
    assert "[arXiv:1234.56789]" in observed["prompt"]
    assert "SECRET GOLD ANSWER" not in observed["prompt"]
    assert saved["retrieved_evidence"][0]["paper_id"] == "1234.56789"
    assert metrics["rag"] == {
        "enabled": True,
        "top_k": 2,
        "corpus": "split evidence_reference passages grouped by reference_arxiv_id",
        "unique_papers": 1,
    }


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
    figure_metrics = metrics["by_modality"]["figure"]
    assert figure_metrics["examples"] == 1
    assert figure_metrics["labeled_examples"] == 0
    assert all(value is None for key, value in figure_metrics.items()
               if key not in {"examples", "labeled_examples"})
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
        str(tmp_path), top_k=5, samples_output_path=str(samples_path),
        llm=FigureToolLLM(),
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

    metrics = evaluate_m3sciqa_agent(
        str(tmp_path), split="test", llm=FigureToolLLM()
    )

    assert metrics["split"] == "test"
    assert metrics["retrieval_examples"] == 1
    assert metrics["recall_at_k"] == 1.0
    assert metrics["tool_selection_examples"] == 1


def test_m3sciqa_agent_eval_counts_model_errors_as_incorrect_tool_choices(tmp_path):
    row = {
        "question": "What does the figure show?",
        "evidence_reference": "A rising trend is shown.",
        "reference_arxiv_id": "1111.00001",
        "modal": "figure",
    }
    (tmp_path / "validation.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    metrics = evaluate_m3sciqa_agent(
        str(tmp_path), llm=FailingLLM()
    )

    assert metrics["tool_selection_examples"] == 1
    assert metrics["tool_selection_errors"] == 1
    assert metrics["tool_selection_accuracy"] == 0.0