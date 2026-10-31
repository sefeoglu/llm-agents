# Scientific Literature Research Agent

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Agent](https://img.shields.io/badge/Agent-Rule--based%20prototype-6A5ACD)
![Retrieval](https://img.shields.io/badge/Retrieval-BM25-00897B)
![Multimodal](https://img.shields.io/badge/Benchmark-M3SciQA-EA6A47)
[![Dataset](https://img.shields.io/badge/Dataset-M3SciQA-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/yale-nlp/M3SciQA)
![Tests](https://img.shields.io/badge/Tests-16%20passing-2E8B57)
![License](https://img.shields.io/badge/License-MIT-555555)

> **Note:** This is a basic prototype, not a fully agentic system, and has been evaluated on a small dataset. Its orchestration design was inspired by [Chen et al. (2026)](#orchestration-reference).

> **Dataset note:** The repository includes M3SciQA split metadata and a small representative subset of images. Download the complete M3SciQA dataset from [Hugging Face](https://huggingface.co/datasets/yale-nlp/M3SciQA) to run the full multimodal evaluation.

**Scientific Literature Research Agent** is a small CLI prototype for scientific literature research. Its **orchestration layer** runs a bounded agent loop: a rule-based router selects tools dynamically, `ToolRegistry` dispatches calls, and every call is recorded in an inspectable trace. Available tools cover arXiv literature search, **BM25** paper retrieval, safe arithmetic, publication-trend analysis, and PDF figure/table inspection with optional local **Ollama** vision. Separate evaluation commands measure **M3SciQA** retrieval, **citation-ID** matching, **tool selection**, and **multimodal answers** where labels are available.

This is a transparent deterministic prototype, not a hosted generative chatbot. The local arXiv corpus contains 20 automated search results about battery degradation/cycle life with abstracts; these are candidate records and require relevance review. The default `python main.py` examples are still generic RAG/calculator demonstrations, not battery analysis. The separate battery question command demonstrates retrieval from the local battery paper corpus, but this project does not currently run a battery lifetime regression experiment or claim a battery-model score.


## Run

Python 3.10+. Install the pinned package snapshot from the configured project environment and run the offline sample:

```bash
python -m pip install -r requirements.txt
python main.py
python -m pytest -q
```

Collect up to 25 candidate records from arXiv and ask a battery research question against the local corpus:

```bash
python main.py collect --query "battery degradation machine learning cycle life prediction" --max-results 25
python main.py ask --corpus data/arxiv_papers.jsonl --question "What evidence links early-cycle measurements with battery cycle-life prediction?"
```

Index a downloaded paper PDF, including embedded figures, then ask a figure-specific question:

```bash
python main.py ingest-pdf --pdf papers/study.pdf --render-pages
OLLAMA_VISION_MODEL=llava python main.py ask --corpus data/arxiv_papers.jsonl --question "What does Figure 2 show?"
```

PDF ingestion extracts selectable text and embedded raster images. `--render-pages` also saves page images so vector charts and tables can be inspected visually. Figure questions use the optional local Ollama vision API at `127.0.0.1:11434`; set `OLLAMA_VISION_MODEL` to an installed vision-capable model. Images are sent only to that local Ollama service. Without a configured model, the tool still retrieves the image and nearby page text and states that visual interpretation is unavailable. Scanned PDF text is not OCR-processed.

The arXiv API call requires network access. The collector stores titles, authors, publication years, abstracts, source links, and PDF links as JSONL. Generated figure/page images and corpus data are kept under ignored `data/` paths. `data/arxiv_papers.jsonl` is present locally with 20 records; because automated search can return off-topic matches, inspect titles/abstracts and filter the corpus before using it as scientific evidence. If the file is absent in a fresh checkout, the offline demo uses its built-in sample corpus until papers are collected or a PDF is ingested.

## Architecture

- Detailed component relationships, UML class model, and request/evaluation sequence diagrams are documented in [doc/system-architecture.md](doc/system-architecture.md).
- `papers.py` collects arXiv Atom records and stores PDF text plus extracted visual assets in portable paper records.
- `research.py` exposes BM25 retrieval, figure inspection, a calculator, and an annual-publication trend tool. `ResearchLLM` routes visual questions to the figure tool and chooses trend analysis when the question asks about time.
- `vision.py` sends an extracted image to an explicitly configured local Ollama vision model and returns its visual description.
- `reranker.py` provides lightweight BM25 ranking. Answers quote the highest-ranked evidence record, including its paper identifier and source location.
- `benchmark.py` downloads M3SciQA records and images and evaluates image-grounded answers.
- `main.py` provides collection, PDF ingestion, question-answering, and benchmark commands.

The interactive agent is a deterministic routing baseline, not a generative answer model. The optional Ollama model is currently used for image descriptions and M3SciQA inference. A battery-focused regression experiment and synthesized, evidence-grounded battery answers are not implemented yet.

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

This is an architecture reference for orchestration and tool calling, not a source for battery-science claims. This project uses its own deterministic rule-based router and bounded tool loop.

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

The offline sample summaries are paraphrases, not the published abstracts. Outputs can vary with rankings and query terms.

## Multimodal Examples

These are historical outputs from `llava:latest` on the first three M3SciQA validation examples. All three are table-image questions. Model responses below are shortened excerpts. The corresponding inputs, reference evidence, and retrieval/tool-routing outputs are available in [the dated validation sample trace](results/rerun-2026-10-31/m3sciqa-validation-samples.json). That artifact does not contain llava-generated answer text; the original three-answer JSONL file is not present in the current workspace. Fresh 20-example inference outputs are available in the dated rerun directory linked below.

```text
Q: How does the method that scores 28.62 in WQ-B address word-level and structure-level diversity?
Model: The image appears to be a table ... it does not provide information about a method for the WQ-B task ... I would need more context.
Reference: It uses a dual forward/backward model and selection strategies to incorporate diverse expressions from external natural questions.

Q: How does normalizing flow contribute to abstractive summarization for the method with ROUGE-L 41.39?
Model: The table shows a ROUGE-L score of 41.39 ... I cannot directly interpret the specific contribution ... without further context.
Reference: It better approximates the posterior distribution of global semantics, with contextualized gating to balance that information.

Q: What improvement does the model provide on ALFRED's unseen test split?
Model: The SR model outperforms MOCA ... joint training with synthetic instructions likely helps it generalize better.
Reference: 8.5% task success rates on unseen test splits.
```

On this three-example sample, exact match was 0.0 and token F1 was 0.2265. These examples illustrate a current limitation: the vision model can describe visible table content, but often misses cross-paper details or gives unsupported inference.

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

Install and start a local Ollama vision model, then evaluate on validation (held-out test is also selectable):

```bash
ollama pull llava
OLLAMA_VISION_MODEL=llava python main.py benchmark --data-dir "$HOME/datasets/m3sciqa-full" --split validation --limit 20 --output results/m3sciqa-validation-llava-20.jsonl
OLLAMA_VISION_MODEL=llava python main.py benchmark --data-dir "$HOME/datasets/m3sciqa-full" --split test --limit 20 --augmentation-k 2 --output results/m3sciqa-test-llava-20-validation-augmented-unscored.jsonl
```

For validation runs, normalized exact match and token F1 are measured against the public reference answers. For test runs, the evaluator uses up to two BM25-selected validation examples, including their images and known Q/A, as few-shot demonstrations; the test answer is never placed in the prompt. The official M3SciQA test split has blank/hidden answer labels, so test runs are prediction-only and report metrics as `null`, not zero. Validation examples are never added to validation prompts. The `--output` files shown above are created when those commands are run; they are not present in the current workspace. Available test inputs and retrieval/tool outputs are in [the dated test sample trace](results/rerun-2026-10-31/m3sciqa-test-samples.json). JSONL inference records include each question, model answer, available reference answer, modality, paper IDs, local image path, selected validation row IDs, and per-example metrics. Records are flushed as examples finish, so completed answers remain saved if a later example fails. Images stay on disk locally and are sent only to the configured local Ollama server.

Measured evaluation results are recorded in the dated rerun directory described below. The official M3SciQA test split has no public answer labels, so its predictions cannot be assigned EM/F1; earlier test-run `0.0` scores were artifacts of treating blank references as labels and must not be interpreted as model quality. Use validation for answer scoring; test can be used for annotation-based retrieval/routing metrics or saved predictions only.

## Agent Evaluation Metrics

Run the four requested agent/evidence metrics against both M3SciQA splits:

```bash
python main.py evaluate-agent --data-dir "$HOME/datasets/m3sciqa-full" --split validation --top-k 5 --output results/m3sciqa-agent-metrics-validation.json
python main.py evaluate-agent --data-dir "$HOME/datasets/m3sciqa-full" --split test --top-k 5 --output results/m3sciqa-agent-metrics-test.json
```

To save the inputs and system outputs for every sample, add `--samples-output`:

```bash
python main.py evaluate-agent --data-dir "$HOME/datasets/m3sciqa-full" --split validation --top-k 5 --samples-output results/m3sciqa-agent-samples-validation.json
python main.py evaluate-agent --data-dir "$HOME/datasets/m3sciqa-full" --split test --top-k 5 --samples-output results/m3sciqa-agent-samples-test.json
```

Each JSON sample contains the question, modality, image path, public reference paper/evidence fields, ranked paper IDs and scores, retrieval rank, citation match, and selected/expected tool. Hidden test answer text is omitted.

Representative validation sample I/O from the saved traces:

```text
Sample 0 input:
Question: How does the method that scores 28.62 in WQ-B address word-level and structure-level diversity?
Modality/image: table, images/validation/00000.png
Reference paper: arXiv:2309.14362

System output:
Top retrieved papers: 2108.04927, 2309.14362, 2104.08836, 2110.14168, 2012.03208
Gold rank / Recall@5: 2 / hit
Top citation match: false
Selected tool / expected tool: search / inspect_figures

Sample 1 input:
Question: How does normalizing flow contribute to abstractive summarization for the method with ROUGE-L 41.39?
Modality/image: table, images/validation/00001.png
Reference paper: arXiv:2109.10616

System output:
Top retrieved papers: 2109.10616, 2212.09233, 2203.14187, 2102.02779, 2108.04927
Gold rank / Recall@5: 1 / hit
Top citation match: true
Selected tool / expected tool: inspect_figures / inspect_figures
```

These are retrieval and routing traces, not generated final-answer transcripts. The full 452-row input/output JSON is [validation samples](results/rerun-2026-10-31/m3sciqa-validation-samples.json); all 1,000 test traces are in [test samples](results/rerun-2026-10-31/m3sciqa-test-samples.json).

The evaluation uses `evidence_reference` passages grouped by `reference_arxiv_id`; answer and explanation fields are not put in the retrieval corpus. The gold evidence ID is the row's `reference_arxiv_id`. Recall@5 and MRR use the first relevant reference paper; citation correctness checks whether the baseline's top-ranked paper ID matches that reference. This citation metric is an ID-match proxy, not a human semantic-entailment judgment. Tool selection is scored against `inspect_figures` for rows labeled figure/table, since each M3SciQA example includes an image. The test split has hidden answer labels, so these metrics assess only the released retrieval/reference annotations and modality; they do not score answer correctness.

Full split results:

| Split | Recall@5 | MRR@5 | Citation correctness | Tool-selection accuracy |
| --- | ---: | ---: | ---: | ---: |
| Validation (452 rows; 450 retrieval-eligible) | 0.8533 | 0.7855 | 0.7400 | 0.3407 |
| Test (1,000 rows; 997 retrieval-eligible) | 0.8837 | 0.8130 | 0.7673 | 0.3560 |

The tool-selection result exposes a current limitation: lexical routing misses many M3SciQA questions that require an image even when their wording does not explicitly mention a figure or table. Test results are not answer-quality scores because answer labels are hidden. Current aggregate metrics and sample traces are saved under `results/rerun-2026-10-31/`.

## Latest Rerun (2026-10-31)

The full test suite passed (**16 tests**; output was reported by the test runner and is not stored as a separate log file). M3SciQA annotation metrics were recomputed on both validation and test. Local `llava:latest` inference ran on 20 validation examples, with exact match `0.0` and token F1 `0.1092` (3 figure examples: F1 `0.0503`; 17 table examples: F1 `0.1196`). The 20 test answers are hidden, so only predictions were saved; EM/F1 are unavailable.

All rerun artifacts are in [results/rerun-2026-10-31/](results/rerun-2026-10-31/), indexed by [summary.json](results/rerun-2026-10-31/summary.json). Files include the [offline demo](results/rerun-2026-10-31/offline-demo.txt), [battery literature query output](results/rerun-2026-10-31/battery-question.txt), [M3SciQA validation metrics](results/rerun-2026-10-31/m3sciqa-validation-metrics.json), [M3SciQA test metrics](results/rerun-2026-10-31/m3sciqa-test-metrics.json), both M3SciQA sample traces, and both llava prediction files. The battery query is retrieval over paper abstracts, not a battery degradation regression experiment; no NASA cycle-level battery dataset or battery ML result is included in this project run.
