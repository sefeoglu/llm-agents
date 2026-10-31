# Scientific Literature Research Agent

A compact, offline-capable research assistant for retrieval-augmented generation (RAG) papers. The sample corpus is a few curated summaries; collect live arXiv abstracts to use real paper records. The deterministic baseline quotes retrieved evidence and keeps source IDs/URLs in the answer rather than inventing synthesis.

## Run

Python 3.10+. Install the PDF extraction dependency and run the offline sample:

```bash
python -m pip install -r requirements.txt
python main.py
python -m pytest -q
```

Collect up to 25 relevant records from arXiv and ask questions against them:

```bash
python main.py collect --query "retrieval augmented generation" --max-results 25
python main.py ask --corpus data/arxiv_papers.jsonl --question "What evidence supports RAG for knowledge-intensive tasks?"
```

Index a downloaded paper PDF, including embedded figures, then ask a figure-specific question:

```bash
python main.py ingest-pdf --pdf papers/study.pdf --render-pages
OLLAMA_VISION_MODEL=llava python main.py ask --corpus data/arxiv_papers.jsonl --question "What does Figure 2 show?"
```

PDF ingestion extracts selectable text and embedded raster images. `--render-pages` also saves page images so vector charts and tables can be inspected visually. Figure questions use the optional local Ollama vision API at `127.0.0.1:11434`; set `OLLAMA_VISION_MODEL` to an installed vision-capable model. Images are sent only to that local Ollama service. Without a configured model, the tool still retrieves the image and nearby page text and states that visual interpretation is unavailable. Scanned PDF text is not OCR-processed.

The arXiv API call requires network access. The collector stores titles, authors, publication years, abstracts, source links, and PDF links as JSONL. Generated figure/page images and corpus data are kept under ignored `data/` paths.

## Architecture

- Detailed component relationships, UML class model, and request/evaluation sequence diagrams are documented in [doc/system-architecture.md](doc/system-architecture.md).
- `papers.py` collects arXiv Atom records and stores PDF text plus extracted visual assets in portable paper records.
- `research.py` exposes BM25 retrieval, figure inspection, a calculator, and an annual-publication trend tool. `ResearchLLM` routes visual questions to the figure tool and chooses trend analysis when the question asks about time.
- `vision.py` sends an extracted image to an explicitly configured local Ollama vision model and returns its visual description.
- `reranker.py` provides lightweight BM25 ranking. Answers quote the highest-ranked evidence record, including its paper identifier and source location.
- `benchmark.py` downloads the official SciFact dev split and corpus on demand, then reports evidence-document hit rate, recall@k, and MRR@k.
- `main.py` provides collection, PDF ingestion, question-answering, and benchmark commands.

This is a deterministic agent baseline, not a generative model. Replace `ResearchLLM` with an implementation backed by a chosen model for synthesized answers; preserve cited evidence and abstain when retrieval returns no relevant documents.

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

These are real outputs from `llava:latest` on the first three M3SciQA validation examples. All three are table-image questions. Model responses below are shortened excerpts; full responses, reference answers, image paths, and per-example scores are saved in [results/m3sciqa-validation-llava-examples.jsonl](results/m3sciqa-validation-llava-examples.jsonl).

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
python main.py download-m3sciqa --data-dir data/m3sciqa --split all
```

This creates `data/m3sciqa/validation.jsonl`, `data/m3sciqa/test.jsonl`, and `data/m3sciqa/images/{validation,test}/`. The downloader replaces remote image URLs with local relative paths. Those files are ignored by Git and remain available locally.

Install and start a local Ollama vision model, then evaluate on validation (held-out test is also selectable):

```bash
ollama pull llava
OLLAMA_VISION_MODEL=llava python main.py benchmark --dataset m3sciqa --data-dir data/m3sciqa --split validation --limit 20 --output results/m3sciqa-validation-llava-20.jsonl
OLLAMA_VISION_MODEL=llava python main.py benchmark --dataset m3sciqa --data-dir data/m3sciqa --split test --limit 20 --augmentation-k 2 --output results/m3sciqa-test-llava-20-validation-augmented-unscored.jsonl
```

For validation runs, normalized exact match and token F1 are measured against the public reference answers. For test runs, the evaluator uses up to two BM25-selected validation examples, including their images and known Q/A, as few-shot demonstrations; the test answer is never placed in the prompt. The official M3SciQA test split has blank/hidden answer labels, so test runs are prediction-only and report metrics as `null`, not zero. Validation examples are never added to validation prompts. For example, the saved 20-case augmented test run is [results/m3sciqa-test-llava-20-validation-augmented-unscored.jsonl](results/m3sciqa-test-llava-20-validation-augmented-unscored.jsonl). JSONL records include each question, model answer, available reference answer, modality, paper IDs, local image path, selected validation row IDs, and per-example metrics. Records are flushed as examples finish, so completed answers remain saved if a later example fails. Images stay on disk locally and are sent only to the configured local Ollama server.

Measured evaluation results are recorded in [results/run-results.json](results/run-results.json). The BM25 SciFact dev baseline at k=5 reached hit rate 0.7433, recall 0.7209, and MRR 0.6257 over 300 claims. On 100 labeled M3SciQA validation examples with local `llava:latest`, normalized exact match was 0.0 and token F1 was 0.1191 (31 figure examples: 0.0838 F1; 69 table examples: 0.1349 F1). The official M3SciQA test split has no public answer labels, so its predictions cannot be assigned EM/F1; earlier test-run `0.0` scores were artifacts of treating blank references as labels and must not be interpreted as model quality. Use validation metrics for scoring and test only for saved predictions or external/hidden-label evaluation.

The previous text-only SciFact evidence retrieval evaluation remains available as a separate task:

```bash
python main.py benchmark --dataset scifact --download --data-dir data/scifact
```

It reports hit-rate@k, recall@k, and MRR@k over SciFact dev evidence document IDs; its previous full-split BM25 result was 0.7433, 0.7209, and 0.6257 respectively at k=5. This is a separate text retrieval benchmark and does not measure multimodal answer quality.

## Agent Evaluation Metrics

Run the four requested agent/evidence metrics against both M3SciQA splits:

```bash
python main.py evaluate-agent --data-dir data/m3sciqa --split validation --top-k 5 --output results/m3sciqa-agent-metrics-validation.json
python main.py evaluate-agent --data-dir data/m3sciqa --split test --top-k 5 --output results/m3sciqa-agent-metrics-test.json
```

To save the inputs and system outputs for every sample, add `--samples-output`:

```bash
python main.py evaluate-agent --data-dir data/m3sciqa --split validation --top-k 5 --samples-output results/m3sciqa-agent-samples-validation.json
python main.py evaluate-agent --data-dir data/m3sciqa --split test --top-k 5 --samples-output results/m3sciqa-agent-samples-test.json
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

These are retrieval and routing traces, not generated final-answer transcripts. The full 452-row input/output JSON is [results/m3sciqa-agent-samples-validation.json](results/m3sciqa-agent-samples-validation.json); all 1,000 test traces are in [results/m3sciqa-agent-samples-test.json](results/m3sciqa-agent-samples-test.json).

The evaluation uses `evidence_reference` passages grouped by `reference_arxiv_id`; answer and explanation fields are not put in the retrieval corpus. The gold evidence ID is the row's `reference_arxiv_id`. Recall@5 and MRR use the first relevant reference paper; citation correctness checks whether the baseline's top-ranked paper ID matches that reference. This citation metric is an ID-match proxy, not a human semantic-entailment judgment. Tool selection is scored against `inspect_figures` for rows labeled figure/table, since each M3SciQA example includes an image. The test split has hidden answer labels, so these metrics assess only the released retrieval/reference annotations and modality; they do not score answer correctness.

Full split results:

| Split | Recall@5 | MRR@5 | Citation correctness | Tool-selection accuracy |
| --- | ---: | ---: | ---: | ---: |
| Validation (452 rows; 450 retrieval-eligible) | 0.8533 | 0.7855 | 0.7400 | 0.3407 |
| Test (1,000 rows; 997 retrieval-eligible) | 0.8837 | 0.8130 | 0.7673 | 0.3560 |

The tool-selection result exposes a current limitation: lexical routing misses many M3SciQA questions that require an image even when their wording does not explicitly mention a figure or table. Test results are not answer-quality scores because answer labels are hidden. Results are saved in [validation metrics](results/m3sciqa-agent-metrics-validation.json), [test metrics](results/m3sciqa-agent-metrics-test.json), [validation sample I/O](results/m3sciqa-agent-samples-validation.json), and [test sample I/O](results/m3sciqa-agent-samples-test.json).
