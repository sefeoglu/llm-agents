import json

import pymupdf

from llm_agents.papers import ingest_pdf, load_papers, save_papers
from llm_agents.research import ResearchAssistant


def test_pdf_ingestion_extracts_text_and_visual_assets(tmp_path):
    pdf_path = tmp_path / "study.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Figure 1. Treatment response by study group.")
    pixmap = pymupdf.Pixmap(
        pymupdf.csRGB, pymupdf.IRect(0, 0, 96, 80), 0
    )
    pixmap.clear_with(220)
    page.insert_image(
        pymupdf.Rect(72, 100, 240, 240), stream=pixmap.tobytes("png")
    )
    document.set_metadata({"title": "Treatment Study", "author": "Researcher"})
    document.save(pdf_path)
    document.close()

    paper = ingest_pdf(str(pdf_path), str(tmp_path / "figures"), render_pages=True)
    assert "Treatment response by study group" in paper.full_text
    assert paper.title == "Treatment Study"
    assert len(paper.figures) == 2
    assert all(tmp_path.joinpath("figures", "study").joinpath(
        asset["path"].split("/")[-1]
    ).is_file() for asset in paper.figures)

    corpus_path = tmp_path / "papers.jsonl"
    save_papers(str(corpus_path), [paper])
    loaded = load_papers(str(corpus_path))
    assert loaded[0].figures == paper.figures


def test_agent_routes_visual_question_to_figure_inspection(tmp_path):
    image_path = tmp_path / "figure.png"
    pixmap = pymupdf.Pixmap(
        pymupdf.csRGB, pymupdf.IRect(0, 0, 96, 80), 0
    )
    pixmap.clear_with(240)
    pixmap.save(image_path)
    pdf_path = tmp_path / "study.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Figure 1. Treatment response by study group.")
    page.insert_image(pymupdf.Rect(72, 100, 240, 240), filename=str(image_path))
    document.save(pdf_path)
    document.close()
    paper = ingest_pdf(str(pdf_path), str(tmp_path / "extracted"))
    calls = []

    def fake_vision(path, prompt, model):
        calls.append((path, model))
        return "The plotted treatment group is higher than control."

    assistant = ResearchAssistant(
        [paper], vision_model="test-vision", vision_analyzer=fake_vision
    )
    result = assistant.run("What does Figure 1 show?")

    assert result["trace"][0]["tool"] == "inspect_figures"
    assert "higher than control" in result["answer"]
    assert calls and calls[0][1] == "test-vision"


def test_figure_inspection_has_clear_no_assets_fallback():
    assistant = ResearchAssistant([])
    result = assistant.run("What does the chart show?")
    assert result["trace"][0]["tool"] == "inspect_figures"
    assert result["answer"] == "No indexed PDF figures or page images were found."