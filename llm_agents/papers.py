"""Paper records, arXiv collection, and portable JSONL storage."""
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ResearchPaper:
    paper_id: str
    title: str
    authors: List[str]
    year: Optional[int]
    abstract: str
    source_url: str
    pdf_url: str = ""
    full_text: str = ""
    figures: List[dict] = field(default_factory=list)

    def indexed_text(self) -> str:
        year = f" ({self.year})" if self.year else ""
        body = self.full_text or self.abstract
        source_id = "arXiv" if self.source_url.startswith("https://arxiv.org/") else "paper"
        figure_text = "\n".join(
            f"Visual asset, page {figure['page']}: "
            f"{figure.get('caption') or figure.get('context', '')}"
            for figure in self.figures
        )
        return (
            f"[{source_id}:{self.paper_id}] {self.title}{year}\n"
            f"Authors: {', '.join(self.authors)}\n"
            f"Evidence text: {body}\n{figure_text}\nSource: {self.source_url}"
        )


def parse_arxiv_feed(xml_text: str) -> List[ResearchPaper]:
    root = ET.fromstring(xml_text)
    atom = "{http://www.w3.org/2005/Atom}"
    papers = []
    for entry in root.findall(f"{atom}entry"):
        entry_url = entry.findtext(f"{atom}id", default="").strip()
        paper_id = entry_url.rstrip("/").split("/")[-1]
        published = entry.findtext(f"{atom}published", default="")
        year_match = re.match(r"(\d{4})", published)
        title = " ".join(entry.findtext(f"{atom}title", default="").split())
        abstract = " ".join(entry.findtext(f"{atom}summary", default="").split())
        authors = [name.text or "" for name in entry.findall(f"{atom}author/{atom}name")]
        if not paper_id or not title:
            continue
        papers.append(ResearchPaper(
            paper_id=paper_id,
            title=title,
            authors=authors,
            year=int(year_match.group(1)) if year_match else None,
            abstract=abstract,
            source_url=f"https://arxiv.org/abs/{paper_id}",
            pdf_url=f"https://arxiv.org/pdf/{paper_id}",
        ))
    return papers


def collect_arxiv(query: str, max_results: int = 20) -> List[ResearchPaper]:
    if max_results < 1:
        raise ValueError("max_results must be at least 1")
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    request = urllib.request.Request(
        f"https://export.arxiv.org/api/query?{params}",
        headers={"User-Agent": "scientific-research-agent/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return parse_arxiv_feed(response.read().decode("utf-8"))


def save_papers(path: str, papers: List[ResearchPaper]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for paper in papers:
            stream.write(json.dumps(asdict(paper), ensure_ascii=False) + "\n")


def load_papers(path: str) -> List[ResearchPaper]:
    papers = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                papers.append(ResearchPaper(**json.loads(line)))
    return papers


def ingest_pdf(pdf_path: str, figure_dir: str = "data/figures",
               render_pages: bool = False) -> ResearchPaper:
    """Extract PDF text and embedded images, optionally rendering each page."""
    import pymupdf

    source = Path(pdf_path).expanduser().resolve()
    asset_dir = Path(figure_dir).expanduser().resolve() / source.stem
    asset_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    page_text = []

    with pymupdf.open(source) as document:
        metadata = document.metadata or {}
        title = metadata.get("title") or source.stem.replace("_", " ")
        authors = [metadata["author"]] if metadata.get("author") else []
        for page_number, page in enumerate(document, start=1):
            blocks = page.get_text("dict").get("blocks", [])
            text = "\n".join(
                " ".join(span["text"] for line in block.get("lines", [])
                         for span in line.get("spans", []))
                for block in blocks if block.get("type") == 0
            ).strip()
            if text:
                page_text.append(text)
            for image_number, block in enumerate(blocks, start=1):
                if block.get("type") != 1 or not block.get("image"):
                    continue
                width, height = block.get("width", 0), block.get("height", 0)
                if min(width, height) < 48:
                    continue
                extension = block.get("ext", "png")
                image_path = asset_dir / f"page_{page_number:03}_image_{image_number:02}.{extension}"
                image_path.write_bytes(block["image"])
                figures.append({
                    "path": str(image_path),
                    "page": page_number,
                    "kind": "embedded image",
                    "width": width,
                    "height": height,
                    "caption": "",
                    "context": text,
                })
            if render_pages:
                rendered = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
                image_path = asset_dir / f"page_{page_number:03}_render.png"
                rendered.save(image_path)
                figures.append({
                    "path": str(image_path),
                    "page": page_number,
                    "kind": "rendered page",
                    "width": rendered.width,
                    "height": rendered.height,
                    "caption": f"Rendered PDF page {page_number}",
                    "context": text,
                })
        created = metadata.get("creationDate", "")
        year_match = re.search(r"D:(\d{4})", created)

    full_text = "\n\n".join(page_text)
    return ResearchPaper(
        paper_id=f"local-{source.stem}",
        title=title,
        authors=authors,
        year=int(year_match.group(1)) if year_match else None,
        abstract=full_text[:1200],
        source_url=source.as_uri(),
        pdf_url=source.as_uri(),
        full_text=full_text,
        figures=figures,
    )


def sample_papers() -> List[ResearchPaper]:
    """Small offline corpus; summaries are paraphrases, not source abstracts."""
    return [
        ResearchPaper(
            "1706.03762", "Attention Is All You Need", ["Vaswani et al."], 2017,
            "Curated summary: the Transformer uses attention instead of recurrence "
            "or convolution as its main sequence-modeling mechanism.",
            "https://arxiv.org/abs/1706.03762",
        ),
        ResearchPaper(
            "1810.04805", "BERT: Pre-training of Deep Bidirectional Transformers "
            "for Language Understanding", ["Devlin et al."], 2018,
            "Curated summary: BERT pre-trains bidirectional language representations "
            "using masked-language modeling and next-sentence prediction.",
            "https://arxiv.org/abs/1810.04805",
        ),
        ResearchPaper(
            "2005.11401", "Retrieval-Augmented Generation for Knowledge-Intensive "
            "NLP Tasks", ["Lewis et al."], 2020,
            "Curated summary: retrieval-augmented generation combines a parametric "
            "sequence model with non-parametric retrieved passages and reports "
            "improvements on knowledge-intensive NLP tasks.",
            "https://arxiv.org/abs/2005.11401",
        ),
        ResearchPaper(
            "2312.10997", "Retrieval-Augmented Generation for Large Language Models: "
            "A Survey", ["Gao et al."], 2023,
            "Curated summary: this survey organizes RAG methods into retrieval, "
            "generation, and augmentation stages and discusses evaluation challenges.",
            "https://arxiv.org/abs/2312.10997",
        ),
    ]