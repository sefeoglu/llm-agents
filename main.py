"""CLI for collecting papers and running the scientific literature assistant."""
import argparse
import json
import os
from pathlib import Path

from llm_agents.benchmark import (
    download_m3sciqa,
    evaluate_m3sciqa,
)
from llm_agents.evaluation import evaluate_m3sciqa_agent
from llm_agents.papers import (
    collect_arxiv,
    ingest_pdf,
    load_papers,
    sample_papers,
    save_papers,
)
from llm_agents.research import ResearchAssistant


def _papers_for(path: str):
    return load_papers(path) if Path(path).exists() else sample_papers()


def main() -> None:
    parser = argparse.ArgumentParser(description="Scientific literature research agent")
    commands = parser.add_subparsers(dest="command")

    collect = commands.add_parser("collect", help="Collect abstracts from arXiv")
    collect.add_argument("--query", required=True)
    collect.add_argument("--max-results", type=int, default=20)
    collect.add_argument("--output", default="data/arxiv_papers.jsonl")

    ask = commands.add_parser("ask", help="Ask a question over a paper corpus")
    ask.add_argument("--corpus", default="data/arxiv_papers.jsonl")
    ask.add_argument("--question", required=True)
    ask.add_argument("--vision-model", help="Ollama vision model name; defaults to OLLAMA_VISION_MODEL")

    ingest = commands.add_parser("ingest-pdf", help="Index PDF text and extract visual assets")
    ingest.add_argument("--pdf", required=True)
    ingest.add_argument("--output", default="data/arxiv_papers.jsonl")
    ingest.add_argument("--figure-dir", default="data/figures")
    ingest.add_argument("--render-pages", action="store_true",
                         help="Also save page renders to capture vector charts and tables")

    dataset = commands.add_parser(
        "download-m3sciqa", help="Download M3SciQA text and figure images locally"
    )
    dataset.add_argument("--data-dir", default="data/m3sciqa")
    dataset.add_argument("--split", choices=("validation", "test", "all"), default="all")

    benchmark = commands.add_parser("benchmark", help="Evaluate multimodal answers on M3SciQA")
    benchmark.add_argument("--data-dir")
    benchmark.add_argument("--download", action="store_true")
    benchmark.add_argument("--split", choices=("validation", "test"), default="validation")
    benchmark.add_argument("--vision-model",
                           default=os.environ.get("OLLAMA_VISION_MODEL"),
                           help="Ollama vision model; defaults to OLLAMA_VISION_MODEL")
    benchmark.add_argument("--limit", type=int, default=0,
                           help="Evaluate only the first N examples")
    benchmark.add_argument("--output", help="Save per-example M3SciQA answers as JSONL")
    benchmark.add_argument("--augmentation-k", type=int, default=2,
                           help="Validation examples per test question; 0 disables augmentation")

    evaluate = commands.add_parser(
        "evaluate-agent", help="Evaluate M3SciQA retrieval, citations, and tool routing"
    )
    evaluate.add_argument("--data-dir", default="data/m3sciqa")
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.add_argument("--limit", type=int, default=0)
    evaluate.add_argument("--output", help="Optional path to save metric JSON")
    evaluate.add_argument("--samples-output",
                          help="Optional JSON path for each sample's inputs and system outputs")

    args = parser.parse_args()
    if args.command == "collect":
        papers = collect_arxiv(args.query, args.max_results)
        save_papers(args.output, papers)
        print(f"Saved {len(papers)} papers to {args.output}")
    elif args.command == "ask":
        assistant = ResearchAssistant(
            _papers_for(args.corpus), vision_model=args.vision_model
        )
        result = assistant.run(args.question)
        print(result["answer"])
        for observation in result["trace"]:
            print(f"\nTool used: {observation['tool']}")
    elif args.command == "ingest-pdf":
        paper = ingest_pdf(args.pdf, args.figure_dir, args.render_pages)
        papers = load_papers(args.output) if Path(args.output).exists() else []
        papers = [existing for existing in papers if existing.paper_id != paper.paper_id]
        save_papers(args.output, papers + [paper])
        print(
            f"Indexed {paper.title}: {len(paper.full_text)} text characters, "
            f"{len(paper.figures)} visual assets -> {args.output}"
        )
    elif args.command == "download-m3sciqa":
        splits = ("validation", "test") if args.split == "all" else (args.split,)
        print(json.dumps(download_m3sciqa(args.data_dir, splits), indent=2))
    elif args.command == "evaluate-agent":
        metrics = evaluate_m3sciqa_agent(
            args.data_dir,
            split=args.split,
            top_k=args.top_k,
            limit=args.limit,
            samples_output_path=args.samples_output or "",
        )
        rendered = json.dumps(metrics, indent=2)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    elif args.command == "benchmark":
        if not args.vision_model:
            parser.error(
                "M3SciQA evaluation needs --vision-model or OLLAMA_VISION_MODEL"
            )
        data_dir = Path(args.data_dir or "data/m3sciqa")
        if args.download and not (data_dir / f"{args.split}.jsonl").exists():
            download_m3sciqa(str(data_dir), (args.split,))
        metrics = evaluate_m3sciqa(
            str(data_dir),
            args.vision_model,
            split=args.split,
            limit=args.limit,
            output_path=args.output,
            augmentation_k=args.augmentation_k,
        )
        print(json.dumps(metrics, indent=2))
    else:
        assistant = ResearchAssistant(sample_papers())
        for question in (
            "What evidence does RAG use for knowledge-intensive tasks?",
            "How has the RAG literature changed over time?",
            "Compute 12 * (3 + 4)",
        ):
            result = assistant.run(question)
            print(f"Q: {question}\nA: {result['answer']}\n")


if __name__ == "__main__":
    main()
