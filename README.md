# Scientific Literature Research Agent

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Agent](https://img.shields.io/badge/Agent-LLM%20tool--calling-247A5A)
![Retrieval](https://img.shields.io/badge/Retrieval-BM25-00897B)
![Multimodal](https://img.shields.io/badge/Benchmark-M3SciQA-EA6A47)
[![Dataset](https://img.shields.io/badge/Dataset-M3SciQA-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/yale-nlp/M3SciQA)
![Tests](https://img.shields.io/badge/Tests-29%20passing-2E8B57)
![License](https://img.shields.io/badge/License-MIT-555555)

> **Note:** This is a small research-agent prototype, evaluated on limited data. Its orchestration design was inspired by [Chen et al. (2026)](#orchestration-reference).

> **Dataset note:** The repository includes M3SciQA split metadata and a small representative subset of images. Download the complete M3SciQA dataset from [Hugging Face](https://huggingface.co/datasets/yale-nlp/M3SciQA) to run the full multimodal evaluation.

**Scientific Literature Research Agent** is a small CLI prototype for scientific literature research. Its bounded agent loop uses an **Ollama or OpenAI LLM with native function calling** to select one next tool action, inspect its result, and then decide what to do next; `ToolRegistry` dispatches calls and the agent records an inspectable trace. Tools cover arXiv literature search, **BM25** paper retrieval, safe arithmetic, publication-trend analysis, and PDF figure/table inspection with optional local Ollama vision. Separate evaluation commands measure **M3SciQA** retrieval, **citation-ID** matching, **tool selection**, and **multimodal answers** where labels are available.

The interactive agent supports local Ollama or OpenAI GPT via `--agent-provider openai`. The reported M3SciQA tool-selection and image-grounded QA evaluations used OpenAI `gpt-4.1-mini`, including as the vision model for images; they did not use LLaVA or Ollama vision. The local arXiv corpus contains 20 automated search results about battery degradation/cycle life with abstracts; these are candidate records and require relevance review. The default `python main.py` examples use a small generic sample corpus, not battery analysis. The separate battery question command demonstrates retrieval from the local battery paper corpus, but this project does not currently run a battery lifetime regression experiment or claim a battery-model score.


## Run

Python 3.10+ is required. Interactive agent runs can use local Ollama or OpenAI GPT tool calling. Install the pinned package snapshot and pull a local model if using Ollama:

```bash
python -m pip install -r requirements.txt
ollama pull qwen2.5:3b
export OLLAMA_AGENT_MODEL=qwen2.5:3b
python main.py
python -m pytest -q
```

`OLLAMA_AGENT_MODEL` defaults to `qwen2.5:3b`; `ask` and `evaluate-agent` accept `--agent-provider ollama|openai` and `--agent-model`. OpenAI reads credentials from `OPENAI_API_KEY` or the untracked `config/gpt.json`. Tests use fake LLM responses and do not require a live model. `OLLAMA_VISION_MODEL` separately configures optional local figure-image analysis.

For OpenAI agent tool selection, the adapter reads `OPENAI_API_KEY` from the environment, or `api_key` from the local untracked `config/gpt.json`; the model comes from `--agent-model`, `OPENAI_AGENT_MODEL`, or that config file. This sends questions and tool results to the OpenAI API. Run the validation-split tool-selection evaluation and save metrics/sample traces separately from QA outputs:

```bash
python main.py evaluate-agent --agent-provider openai --agent-model gpt-4.1-mini --data-dir data/m3sciqa --split validation --top-k 5 --output results/tool-calling/validation/new-run/metrics.json --samples-output results/tool-calling/validation/new-run/samples.json
```

Collect up to 25 candidate records from arXiv and ask a battery research question against the local corpus:

```bash
python main.py collect --query "battery degradation machine learning cycle life prediction" --max-results 25
python main.py ask --agent-model qwen2.5:3b --corpus data/arxiv_papers.jsonl --question "What evidence links early-cycle measurements with battery cycle-life prediction?"
```

For GPT-driven interactive research, use `python main.py ask --agent-provider openai --corpus data/arxiv_papers.jsonl --question "..."`. OpenAI receives the question and tool results; local Ollama is not used for agent reasoning in that mode.

Index a downloaded paper PDF, including embedded figures, then ask a figure-specific question:

```bash
python main.py ingest-pdf --pdf papers/study.pdf --render-pages
OLLAMA_AGENT_MODEL=qwen2.5:3b OLLAMA_VISION_MODEL=your-vision-model python main.py ask --corpus data/arxiv_papers.jsonl --question "What does Figure 2 show?"
```

PDF ingestion extracts selectable text and embedded raster images. `--render-pages` also saves page images so vector charts and tables can be inspected visually. Figure questions use the optional local Ollama vision API at `127.0.0.1:11434`; set `OLLAMA_VISION_MODEL` to an installed vision-capable model. Images are sent only to that local Ollama service. Without a configured model, the tool still retrieves the image and nearby page text and states that visual interpretation is unavailable. Scanned PDF text is not OCR-processed.

The arXiv API call requires network access. The collector stores titles, authors, publication years, abstracts, source links, and PDF links as JSONL. Generated figure/page images and corpus data are kept under ignored `data/` paths. `data/arxiv_papers.jsonl` is present locally with 20 records; because automated search can return off-topic matches, inspect titles/abstracts and filter the corpus before using it as scientific evidence. If the file is absent in a fresh checkout, the demo uses its built-in sample corpus until papers are collected or a PDF is ingested.

## Architecture

- Detailed component relationships, UML class model, and request/evaluation sequence diagrams are documented in [doc/system-architecture.md](doc/system-architecture.md).
- `papers.py` collects arXiv Atom records and stores PDF text plus extracted visual assets in portable paper records.
- `research.py` exposes BM25 retrieval, figure inspection, a calculator, and an annual-publication trend tool. Ollama or OpenAI tool-calling adapters select tools and form answers from their observations.
- `vision.py` supports image analysis through OpenAI and local Ollama adapters. The reported M3SciQA image-grounded QA results used OpenAI `gpt-4.1-mini` as the vision model; local Ollama vision is an optional alternative for interactive figure inspection.
- `reranker.py` provides lightweight BM25 ranking. Retrieved records include paper identifiers and source locations; the LLM is prompted to cite them, but citation support is not independently verified.
- `benchmark.py` downloads M3SciQA records and images and evaluates image-grounded answers.
- `main.py` provides collection, PDF ingestion, question-answering, and benchmark commands.

The interactive agent uses an LLM for tool selection and answer synthesis, while retrieval and analysis tools remain deterministic. The optional vision model is separate from the text agent model. When lexical filtering finds no documents, retrieval returns no evidence rather than the full corpus; prompts instruct the LLM to identify missing or conflicting support. This is not calibrated abstention or verified conflict resolution, and answer entailment is not independently validated. A battery-focused regression experiment is not implemented.

### Orchestration Reference

The agent orchestration and tool-calling design is informed by the constraint-aware agent approach described in:

Chen, B., Bök, T.E., Rasti, B., Markl, V., & Demir, B. (2026). “Foundation Model Selection for Remote Sensing via a Constraint-Aware Agent.” In P. Favaro, Z. Kukelova, A. Maki, A. Rohrbach, K. Schindler, & F. Tombari (Eds.), *Computer Vision – ECCV 2026*, Lecture Notes in Computer Science, vol. 17064. Springer, Cham. [https://doi.org/10.1007/978-3-032-37196-6_34](https://doi.org/10.1007/978-3-032-37196-6_34)

```bibtex
@inproceedings{chen2026constraintAwareAgent,
	author    = {Chen, B. and B{\"o}k, T. E. and Rasti, B. and Markl, V. and Demir, B.},
	title     = {Foundation Model Selection for Remote Sensing via a Constraint-Aware Agent},
	booktitle = {Computer Vision -- ECCV 2026},
	series    = {Lecture Notes in Computer Science},
	volume    = {17064},
	publisher = {Springer},
	address   = {Cham},
	year      = {2026},
	doi       = {10.1007/978-3-032-37196-6_34}
}
```

This is an architecture reference for orchestration and tool calling, not a source for battery-science claims. This project uses an Ollama- or OpenAI-backed LLM and a bounded tool loop.

## Example

```text
Q: What evidence does RAG use for knowledge-intensive tasks?
A: [arXiv:2005.11401] Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020)
Authors: Lewis et al.
Evidence text: Curated summary: retrieval-augmented generation combines a parametric sequence model with non-parametric retrieved passages and reports improvements on knowledge-intensive NLP tasks.
Source: https://arxiv.org/abs/2005.11401

Q: How has the RAG literature changed over time?
A: The corpus contains 4 papers from 2017 to 2023. Annual counts are {2017: 1, 2018: 1, 2020: 1, 2023: 1}; the fitted linear trend is decreasing (-0.07 papers/year). This describes the collected corpus, not the field as a whole.

Q: Compute 12 * (3 + 4)
A: 84
```

These are illustrative outputs from the built-in sample corpus; its summaries are paraphrases, not published abstracts. LLM-generated wording can vary between runs.

## Multimodal Dataset and Evaluation

The benchmark dataset is the official [Yale NLP M3SciQA](https://huggingface.co/datasets/yale-nlp/M3SciQA) release (CC-BY-4.0): 1,452 expert-annotated scientific QA examples, with 452 validation and 1,000 test records. Records include question and answer text, a figure/table image, modality label, paper IDs, and anchor/reference evidence fields. Download both splits and image bytes to local storage:

```bash
python main.py download-m3sciqa --data-dir "$HOME/datasets/m3sciqa-full" --split all
```

The repository has `data/m3sciqa/validation.jsonl` and `data/m3sciqa/test.jsonl` plus a small representative image subset; it does not track all 1,452 images. The downloader refuses to overwrite these sample split files, so fetch the complete dataset into a separate directory:

```bash
python main.py download-m3sciqa --data-dir "$HOME/datasets/m3sciqa-full" --split all
```

Use `--data-dir "$HOME/datasets/m3sciqa-full"` for full-dataset benchmark commands.

For GPT-based image-grounded QA with RAG, configure `OPENAI_API_KEY` (or `api_key` in the local `config/gpt.json`) without sharing it in chat. The following evaluates the full labeled validation split with the configured GPT model, retrieving up to three public evidence passages per question:

```bash
mkdir -p results/qa/validation/new-run
python main.py benchmark --vision-provider openai --vision-model gpt-4.1-mini --data-dir data/m3sciqa --split validation --rag-top-k 3 --augmentation-k 0 --output results/qa/validation/new-run/rag-predictions.jsonl > results/qa/validation/new-run/metrics.json
```

For each question, the QA RAG path ranks the split's public `evidence_reference` passages with BM25 and includes the top three positive-scoring passages, labeled with their arXiv IDs, alongside the image and question. It does not index answer or explanation fields. Because this QA corpus is built from M3SciQA annotations, it evaluates answer generation with provided evidence rather than retrieval over an independent paper corpus. This GPT-only run sends local images, questions, and retrieved passages to OpenAI and may incur API charges; it does not use Ollama. `--rag-top-k 0` disables retrieval augmentation.

The latest complete 452-example GPT RAG validation run with `gpt-4.1-mini` scored exact match `0.0` and token F1 `0.2387` (figure F1 `0.2340`, table F1 `0.2408`). The vision-only validation baseline scored token F1 `0.1503` on the same split/model. QA evidence retrieval hit@3 was `0.8319` (MRR `0.7824`, top-citation ID match `0.7434`); these figures rank against M3SciQA annotation-provided passages, not an independent paper corpus. See [fresh QA metrics](llm_agents/evaluation/validation/qa/metrics-agentic-run-2026-10-31.json), [scored predictions](llm_agents/evaluation/validation/qa/predictions-agentic-run-2026-10-31-scored.jsonl), and [complete raw predictions](results/qa/validation/agentic-run-2026-10-31/rag-predictions-complete.jsonl). This descriptive comparison adds BM25 evidence context; it has no significance test or separate-seed analysis.

Validation answers are scored against public reference answers. Test answers are hidden, so this README reports answer-quality scores only for validation. For test runs, known validation examples may be used as few-shot demonstrations; target test answers are never included in the prompt.

## Agent Evaluation Metrics

Run the retrieval and citation metrics plus GPT tool-selection evaluation. The full validation split makes one GPT decision per image-labeled row. This sends questions and tool schemas to OpenAI and may incur API charges. M3SciQA test answers are hidden, and this project records validation-only agent metrics.

```bash
python main.py evaluate-agent --agent-provider openai --agent-model gpt-4.1-mini --data-dir data/m3sciqa --split validation --top-k 5 --output results/tool-calling/validation/new-run/metrics.json --samples-output results/tool-calling/validation/new-run/samples.json
```

Each JSON sample contains the question, modality, image path, public reference paper/evidence fields, ranked paper IDs and scores, retrieval rank, citation match, and selected/expected tool. Hidden test answer text is omitted.

The evaluation uses `evidence_reference` passages grouped by `reference_arxiv_id`; answer and explanation fields are not put in the retrieval corpus. Recall@5 and MRR rank within those annotation-provided passages; citation correctness is an ID-match proxy, not a semantic-entailment judgment. Tool selection is compared with `inspect_figures` for rows labeled figure/table. Test answers are hidden; this project reports current agent results on validation only.

Fresh GPT agent validation results (`gpt-4.1-mini`, 452 rows): Recall@5 `0.8533`, MRR@5 `0.7855`, citation-ID correctness `0.7400`, and strict first-tool selection accuracy `0.0996` (zero tool-call errors). The planner selected `inspect_figures` on 45 rows and `search` on 407. The saved evaluation measures only the first decision per row; it does not measure later replanning. See [fresh tool-calling metrics](results/tool-calling/validation/agentic-run-2026-10-31/metrics.json) and [per-example traces](results/tool-calling/validation/agentic-run-2026-10-31/samples.json). Retrieval metrics use annotated evidence, not an open corpus.
