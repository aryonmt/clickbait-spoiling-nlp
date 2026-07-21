# Clickbait Spoiling Pipeline

An end-to-end NLP pipeline that automatically **detects the type of a clickbait spoiler** and **generates the spoiler text** for a given clickbait social media post and its linked article — built around the Webis Clickbait Spoiling Corpus / Clickbait Challenge task formulation.

The pipeline is organized as two sequential sub-tasks:

- **Task 1 — Spoiler Type Classification**: predicts whether a post's spoiler is a `phrase`, `passage`, or `multi` (multi-part) spoiler.
- **Task 2 — Spoiler Generation**: given the predicted type, extracts or generates the actual spoiler text from the target article, using a dedicated model per spoiler type.

Both tasks are wired together into a single **integrated inference pipeline**, complete with post-processing, evaluation, and automated HTML/PNG reporting.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)
- [Reports & Visualizations](#reports--visualizations)
- [Tech Stack](#tech-stack)
- [Notes](#notes)

## Overview

Clickbait posts ("You won't believe what happened next...") intentionally withhold information that is answered inside the linked article. **Spoiling** a clickbait post means surfacing that withheld information directly, without requiring the user to read the full article. The task comes in two parts, each requiring a different type of model:

1. Classify the *kind* of information needed to spoil the post (a short phrase, a longer passage, or several disjoint pieces of text).
2. Produce the spoiler itself, using an approach suited to that type — extractive question answering for phrases and passages, and either sequence-to-sequence generation, iterative extraction, or lexical retrieval for multi-part spoilers.

This repository implements both stages as independently trainable transformer models, plus the glue code needed to combine them into a single, evaluable pipeline.

## Key Features

- **Exploratory Data Analysis (`main_eda.py`)** — sequence length distributions, class balance, spoiler/article lexical overlap, positional bias of the spoiler inside the article, tag-specific length statistics, and an oracle retrieval upper bound.
- **Task 1 classifier** — a fine-tuned `roberta-base` sequence classifier over `(post, article)` pairs, trained with inverse-frequency class weighting to handle label imbalance.
- **Task 2 extractive QA** — two independently fine-tuned `deepset/roberta-base-squad2` checkpoints (one for `phrase`, one for `passage` spoilers), trained with sliding-window tokenization (`doc_stride`) to handle long articles, and evaluated with exact match / token-level F1 on answerable spans.
- **Task 2 multi-spoiler generation** — pluggable strategies for spoilers that span multiple non-contiguous parts of the article:
  - `seq2seq`: a fine-tuned Flan-T5 generative model.
  - `extractive_iterative`: repeated extractive QA passes reusing the passage QA model.
  - `tfidf` / `jaccard`: lexical similarity-based retrieval, used as a training-free fallback.
- **Integrated pipeline (`main_integrated_pipeline.py`)** — runs Task 1 inference to predict a spoiler type per example, then dynamically routes each example to the corresponding Task 2 model, with confidence scores and safe fallbacks (e.g. falling back to the first paragraph if extraction returns an empty span).
- **Post-processing** — enforces the challenge's length conventions: phrase spoilers are trimmed to a short word limit, and passage spoilers shorter than expected are expanded back to their containing sentence.
- **Evaluation suite** — Task 1 accuracy/macro-F1 with per-type precision/recall/F1, and Task 2 BLEU, ROUGE-L, METEOR, and BERTScore, with a baseline comparison report.
- **Automated reporting (`main_generate_report.py`)** — generates confusion matrices, confidence distributions, error-by-length plots, and a self-contained HTML dashboard (`reports/report.html`).
- **Kaggle-ready notebook (`notebook.ipynb`)** — a sequential runner that installs dependencies, verifies GPU availability, and executes every stage of the pipeline end to end (EDA → Task 1 training → Task 2 training → integrated inference → reporting).

## Project Structure

```
clickbait-spoiling/
├── main_eda.py                              # Exploratory data analysis runner
├── main_transformer_task1.py                # Task 1: spoiler type classifier training
├── main_transformer_task2_train.py          # Task 2: unified phrase/passage/QA training
├── main_transformer_task2_multi_seq2seq.py  # Task 2: Flan-T5 multi-spoiler training
├── main_integrated_pipeline.py              # End-to-end Task 1 + Task 2 inference
├── main_generate_report.py                  # Visual/HTML evaluation report generator
├── notebook.ipynb                           # Kaggle-oriented end-to-end runner
├── requirements.txt
├── src/
│   ├── config.py                    # PipelineConfig: all hyperparameters & paths
│   ├── data_loader.py                # JSONL loading, sanity checks, tag filtering
│   ├── eda_analysis.py               # TextLengthAnalyzer used by main_eda.py
│   ├── transformer_classifier.py     # Task 1 model wrapper & training loop
│   ├── qa_preprocessor.py            # Feature preparation for extractive QA
│   ├── seq2seq_qa_preprocessor.py    # Feature preparation for Flan-T5 multi-spoiler
│   ├── qa_inference.py               # Best-span decoding for extractive QA
│   ├── multi_spoiler_extractor.py    # Iterative extractive multi-spoiler strategy
│   ├── retrieval_generator.py        # TF-IDF / Jaccard lexical fallback strategy
│   ├── postprocessor.py              # Length-constraint enforcement rules
│   ├── evaluation.py                 # AdvancedEvaluationSuite (metrics computation)
│   ├── visualization_suite.py        # EvaluationReportBuilder (charts + HTML report)
│   ├── logging_setup.py              # Shared logger configuration
│   └── utils.py                      # Shared helpers (tag/text normalization)
├── reports/
│   ├── report.html                   # Interactive evaluation dashboard
│   ├── metrics_summary.json          # Aggregated Task 1 / Task 2 metrics
│   ├── comparison_to_baseline.json   # Delta vs. a reference baseline run
│   ├── *.png                         # Confusion matrix, confidence, error plots, etc.
│   └── eda/*.png                     # EDA-specific charts
└── data/                              # (not included) place train/validation JSONL here
```

## Dataset

The pipeline expects data in the JSON Lines format used by the Webis Clickbait Spoiling Corpus, with each record containing at least:

| Field              | Type            | Description                                                  |
|--------------------|-----------------|----------------------------------------------------------------|
| `uuid`             | `string`        | Unique identifier for the example.                              |
| `postText`         | `list[string]`  | The clickbait social media post text.                          |
| `targetParagraphs` | `list[string]`  | The paragraphs of the linked article.                          |
| `tags`             | `list[string]`  | Ground-truth spoiler type (`phrase`, `passage`, or `multi`).    |
| `spoiler`          | `list[string]`  | Ground-truth spoiler text segment(s), used for Task 2 training/eval. |

Place your data as:

```
data/train.jsonl
data/validation.jsonl
```

The `data/` directory is intentionally excluded from version control (see `.gitignore`) — supply your own copy of the corpus (for example, the official Webis Clickbait Spoiling Corpus or an equivalent mirror).

## Installation

```bash
git clone <this-repository-url>
cd clickbait-spoiling

python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended for training; all training scripts default to `fp16` mixed precision. NLTK's sentence tokenizer data (`punkt`) is downloaded automatically on first use by the post-processing module.

## Usage

### 1. Exploratory Data Analysis

```bash
python main_eda.py
```

Reads `data/train.jsonl`, runs dataset sanity checks, and writes length/overlap/positional-bias charts to `reports/eda/`.

### 2. Train the Task 1 classifier

```bash
python main_transformer_task1.py \
  --train_path data/train.jsonl \
  --val_path data/validation.jsonl \
  --model_name roberta-base \
  --epochs 5 \
  --lr 2e-5 \
  --output_dir outputs/task1
```

### 3. Train the Task 2 extractive QA models

Two independent runs are needed — one per spoiler type:

```bash
# Phrase spoilers
python main_transformer_task2_train.py \
  --spoiler_type phrase \
  --train_path data/train.jsonl \
  --val_path data/validation.jsonl

# Passage spoilers
python main_transformer_task2_train.py \
  --spoiler_type passage \
  --train_path data/train.jsonl \
  --val_path data/validation.jsonl
```

A `qa` variant (trained on the unfiltered dataset) is also available via `--spoiler_type qa` for experimentation.

### 4. Train the multi-spoiler Seq2Seq generator (optional)

```bash
python main_transformer_task2_multi_seq2seq.py \
  --train_path data/train.jsonl \
  --val_path data/validation.jsonl \
  --model_name google/flan-t5-base \
  --epochs 3
```

If this model isn't trained, the integrated pipeline automatically falls back to the TF-IDF lexical retrieval strategy for multi-spoilers.

### 5. Run the integrated end-to-end pipeline

```bash
python main_integrated_pipeline.py \
  --val_path data/validation.jsonl \
  --t1_checkpoint outputs/task1 \
  --phrase_checkpoint results_qa_phrase \
  --passage_checkpoint results_qa_passage \
  --multi_strategy seq2seq \
  --output_path final_integrated_submission.jsonl
```

This classifies every example's spoiler type, routes it to the matching Task 2 model, applies post-processing, computes Task 1/Task 2 metrics, and writes a submission file with predicted tags, spoilers, and confidence scores.

### 6. Generate the evaluation report

```bash
python main_generate_report.py \
  --predictions_path final_integrated_submission.jsonl \
  --val_path data/validation.jsonl \
  --output_dir reports
```

Produces the PNG charts and the interactive `reports/report.html` dashboard.

## Configuration

All hyperparameters and paths are centralized in `src/config.py` as a single `PipelineConfig` dataclass — including model names, learning rates, batch sizes, epochs, max sequence/answer lengths, and output directories for every stage. Every training script accepts CLI flags that transparently override the matching config field, so you can adjust behavior without editing source code.

## Results

The figures below come from `reports/metrics_summary.json` and `reports/comparison_to_baseline.json` for the most recent evaluation run; your own numbers will depend on your training run and data split.

**Task 1 — Spoiler Type Classification**

| Metric              | Overall |
|----------------------|:-------:|
| Accuracy             | 72.25%  |
| Macro F1             | 71.77%  |

| Type    | Precision | Recall | F1     |
|---------|:---------:|:------:|:------:|
| Phrase  | 74.92%    | 74.03% | 74.47% |
| Passage | 73.11%    | 69.25% | 71.13% |
| Multi   | 65.24%    | 74.83% | 69.71% |

**Task 2 — Spoiler Generation**

| Metric      | Overall |
|-------------|:-------:|
| BLEU        | 0.193   |
| ROUGE-L     | 0.449   |
| METEOR      | 0.411   |
| BERTScore   | 0.904   |

| Type    | BLEU  | ROUGE-L | METEOR | BERTScore |
|---------|:-----:|:-------:|:------:|:---------:|
| Phrase  | 0.188 | 0.617   | 0.505  | 0.934     |
| Passage | 0.239 | 0.371   | 0.368  | 0.894     |
| Multi   | 0.105 | 0.233   | 0.285  | 0.858     |

Roughly 0.5% of examples fall back to a default prediction (e.g. when extraction returns an empty span).

**Comparison against a reference baseline run:**

| Metric           | Baseline | Current | Δ       |
|-------------------|:--------:|:-------:|:-------:|
| Task 1 Accuracy   | 73.63%   | 72.25%  | -1.38pp |
| Task 1 Macro F1   | 73.08%   | 71.77%  | -1.31pp |
| Task 2 BLEU       | 0.168    | 0.193   | +0.025  |
| Task 2 ROUGE-L    | 0.434    | 0.449   | +0.016  |

Passage-type spoilers are the hardest to classify correctly and multi-part spoilers remain the hardest to generate faithfully — both expected given their higher linguistic variability and reliance on multi-sentence reasoning.

## Reports & Visualizations

Running the pipeline populates `reports/` with:

- `report.html` — an interactive dashboard combining all metrics and charts.
- `confusion_matrix.png`, `classification_metrics.png`, `confidence_distribution.png` — Task 1 diagnostics.
- `length_comparison.png`, `error_by_length.png`, `task2_scores_by_type.png` — Task 2 diagnostics.
- `eda/sequence_length_distributions.png`, `eda/spoiler_length_by_tag.png`, `eda/spoiler_positional_bias.png` — dataset-level EDA charts.
- `metrics_summary.json`, `comparison_to_baseline.json` — raw metric values backing the tables above.

## Tech Stack

- **Modeling**: PyTorch, Hugging Face `transformers`, `datasets`, `accelerate`, `evaluate`
- **Classification backbone**: `roberta-base`
- **Extractive QA backbone**: `deepset/roberta-base-squad2`
- **Generative backbone**: `google/flan-t5-base`
- **Metrics**: `rouge-score`, `nltk` (METEOR), `bert-score`, `scikit-learn`
- **Data & utilities**: `pandas`, `numpy`, `thefuzz` / `python-Levenshtein`
- **Visualization**: `matplotlib`, `seaborn`

## Notes

- Training scripts prune intermediate checkpoints after training and keep only the best-performing model (`save_total_limit=1`) to conserve disk space.
- `main_integrated_pipeline.py` automatically falls back from the Seq2Seq multi-spoiler strategy to TF-IDF retrieval if no trained Seq2Seq checkpoint is found at the configured path.
- This project does not currently include a license file — add one (e.g. MIT, Apache-2.0) before distributing or open-sourcing it publicly.
