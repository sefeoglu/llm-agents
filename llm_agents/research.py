"""Research assistant with adaptive retrieval and corpus analysis tools."""
import re
import os
from collections import Counter
from pathlib import Path
from typing import Callable, List, Optional

from .llm import RuleBasedLLM
from .orchestrator import Agent
from .papers import ResearchPaper
from .reranker import Reranker, tokenize
from .tools import Tool, default_registry
from .vision import analyze_image_with_ollama


class ResearchLLM(RuleBasedLLM):
    """Choose analysis when requested; otherwise use retrieval or calculation."""

    def decide(self, question: str, observations: List[dict]) -> dict:
        used = {observation["tool"] for observation in observations}
        asks_for_visual = re.search(
            r"\b(fig(?:ure)?s?|chart|plot|diagram|graph|table|image|visual)\b\.?",
            question,
            re.I,
        )
        if asks_for_visual and "inspect_figures" not in used:
            return {"tool": "inspect_figures", "arguments": {"query": question}}
        if observations and observations[-1]["tool"] == "inspect_figures":
            return {"answer": observations[-1]["result"]["summary"]}
        asks_for_trend = re.search(
            r"\b(trend|over time|by year|annual|growth|changed over time)\b",
            question,
            re.I,
        )
        if asks_for_trend and "publication_trend" not in used:
            return {"tool": "publication_trend", "arguments": {}}
        if observations and observations[-1]["tool"] == "publication_trend":
            return {"answer": observations[-1]["result"]["summary"]}
        return super().decide(question, observations)


class ResearchAssistant:
    def __init__(self, papers: List[ResearchPaper], top_k: int = 3,
                 vision_model: Optional[str] = None,
                 vision_analyzer: Optional[Callable[[str, str, str], str]] = None):
        self.papers = papers
        self.documents = [paper.indexed_text() for paper in papers]
        self.reranker = Reranker()
        self.vision_model = vision_model or os.environ.get("OLLAMA_VISION_MODEL")
        self.vision_analyzer = vision_analyzer or analyze_image_with_ollama
        registry = default_registry(self._retrieve)
        registry.register(Tool(
            "publication_trend",
            "Count papers by year and estimate the linear trend in annual counts",
            self.publication_trend,
        ))
        registry.register(Tool(
            "inspect_figures",
            "Retrieve paper figures and inspect them with an optional vision model",
            self.inspect_figures,
        ))
        self.agent = Agent(ResearchLLM(), registry, self.reranker, top_k=top_k)

    def _retrieve(self, query: str) -> List[str]:
        query_terms = set(tokenize(query))
        matches = [
            document for document in self.documents
            if query_terms.intersection(tokenize(document))
        ]
        return matches or self.documents

    def publication_trend(self) -> dict:
        yearly_counts = Counter(paper.year for paper in self.papers if paper.year)
        if not yearly_counts:
            return {
                "year_counts": {},
                "slope_papers_per_year": 0.0,
                "summary": "No publication-year metadata is available.",
            }
        first_year, last_year = min(yearly_counts), max(yearly_counts)
        years = list(range(first_year, last_year + 1))
        counts = [yearly_counts.get(year, 0) for year in years]
        mean_year = sum(years) / len(years)
        mean_count = sum(counts) / len(counts)
        variance = sum((year - mean_year) ** 2 for year in years)
        slope = (
            sum((year - mean_year) * (count - mean_count)
                for year, count in zip(years, counts)) / variance
            if variance else 0.0
        )
        direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
        return {
            "year_counts": dict(sorted(yearly_counts.items())),
            "slope_papers_per_year": round(slope, 3),
            "summary": (
                f"The corpus contains {len(self.papers)} papers from {first_year} "
                f"to {last_year}. Annual counts are {dict(sorted(yearly_counts.items()))}; "
                f"the fitted linear trend is {direction} ({slope:+.2f} papers/year). "
                "This describes the collected corpus, not the field as a whole."
            ),
        }

    def inspect_figures(self, query: str, top_k: int = 2) -> dict:
        assets = [
            (paper, figure)
            for paper in self.papers
            for figure in paper.figures
            if Path(figure["path"]).is_file()
        ]
        if not assets:
            return {"figures": [], "summary": "No indexed PDF figures or page images were found."}
        descriptions = [
            f"[asset:{index}] {paper.title} {figure.get('kind', 'figure')} page {figure['page']} "
            f"{figure.get('caption', '')} {figure.get('context', '')}"
            for index, (paper, figure) in enumerate(assets)
        ]
        ranked = self.reranker.rerank(query, descriptions, top_k)
        results = []
        for description, score in ranked:
            index = descriptions.index(description)
            paper, figure = assets[index]
            result = {
                "paper": paper.title,
                "page": figure["page"],
                "kind": figure.get("kind", "figure"),
                "path": figure["path"],
                "context": figure.get("caption") or figure.get("context", ""),
                "score": score,
            }
            if self.vision_model:
                prompt = (
                    "Describe the visible scientific figure faithfully. Identify axes, "
                    "legend, trends, and labeled values when legible. Do not infer "
                    "causality or details that are not visible. Paper context: "
                    f"{result['context']} Question: {query}"
                )
                try:
                    result["visual_description"] = self.vision_analyzer(
                        figure["path"], prompt, self.vision_model
                    )
                except Exception as error:
                    result["visual_description"] = f"Vision analysis unavailable: {error}"
            else:
                result["visual_description"] = (
                    "Image extracted; set OLLAMA_VISION_MODEL and run a local Ollama "
                    "vision model to analyze its contents."
                )
            results.append(result)
        summary = "\n\n".join(
            f"{item['paper']}, {item['kind']} on page {item['page']}\n"
            f"Visual analysis: {item['visual_description']}\n"
            f"Nearby paper text: {item['context']}\nImage: {item['path']}"
            for item in results
        )
        return {"figures": results, "summary": summary}

    def run(self, question: str) -> dict:
        return self.agent.run(question)

    def search(self, query: str, top_k: int = 5):
        return self.reranker.rerank(query, self._retrieve(query), top_k)