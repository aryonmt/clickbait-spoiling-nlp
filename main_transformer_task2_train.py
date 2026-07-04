"""Unified training entry point for Task 2 (Spoiler Text Extraction).

Replaces the three near-duplicate scripts:
    - main_transformer_task2_phrase.py
    - main_transformer_task2_passage.py
    - main_transformer_task2_qa.py

Select which variant to train with --spoiler_type {phrase,passage,qa}.
"phrase" and "passage" filter the dataset by tag before training; "qa"
trains on the full (unfiltered) dataset, matching the previous scripts'
behavior exactly.
"""

import argparse
import glob
import os
import shutil

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForQuestionAnswering,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.config import PipelineConfig
from src.data_loader import JSONLLoader
from src.qa_preprocessor import QAPreprocessor

# Per-spoiler_type mapping of which config attributes to read/override.
# This is the only place that needs to know about the differences between
# the phrase / passage / qa variants.
VARIANT_CONFIG = {
    "phrase": {
        "tag_filter": "phrase",
        "model_name_attr": "task2_phrase_model_name",
        "output_dir_attr": "task2_phrase_output_dir",
        "epochs_attr": "task2_phrase_epochs",
        "description": "Fine-tune a Transformer model for Phrase Spoilers Extraction",
    },
    "passage": {
        "tag_filter": "passage",
        "model_name_attr": "task2_passage_model_name",
        "output_dir_attr": "task2_passage_output_dir",
        "epochs_attr": "task2_passage_epochs",
        "description": "Fine-tune a Transformer model for Passage Spoilers Extraction",
    },
    "qa": {
        "tag_filter": None,  # no filtering: train on the full dataset
        "model_name_attr": "task2_model_name",
        "output_dir_attr": "output_dir",
        "epochs_attr": "task2_epochs",
        "description": "Fine-tune a Transformer model for Task 2 (Spoiler Text Extraction)",
    },
}


def compute_qa_metrics(eval_pred):
    """Computes exact match and token-level F1 on token positions directly,

    including an answerable-only F1 metrics breakdown to prevent dynamic watering down.

    Args:
        eval_pred: Evaluated predictions containing (logits, labels).

    Returns:
        dict: Calculated overall and answerable-only metrics.
    """
    start_logits, end_logits = eval_pred.predictions
    start_positions, end_positions = eval_pred.label_ids

    pred_start = np.argmax(start_logits, axis=-1)
    pred_end = np.argmax(end_logits, axis=-1)

    exact_match = 0
    f1_scores = []

    # Metrics breakdown for answerable context windows (true start position != 0)
    answerable_exact_match = 0
    answerable_f1_scores = []

    total = len(start_positions)

    for i in range(total):
        s_true, e_true = start_positions[i], end_positions[i]
        s_pred, e_pred = pred_start[i], pred_end[i]

        # Exact Match (Token boundaries alignment)
        is_em = s_true == s_pred and e_true == e_pred
        if is_em:
            exact_match += 1

        # Token-level F1 calculation
        true_span = set(range(s_true, e_true + 1))
        pred_span = set(range(s_pred, e_pred + 1)) if e_pred >= s_pred else set()

        if not true_span and not pred_span:
            f1 = 1.0
        elif not true_span or not pred_span:
            f1 = 0.0
        else:
            intersection = true_span.intersection(pred_span)
            if not intersection:
                f1 = 0.0
            else:
                precision = len(intersection) / len(pred_span)
                recall = len(intersection) / len(true_span)
                f1 = (2 * precision * recall) / (precision + recall)

        f1_scores.append(f1)

        # Track metrics strictly for answerable contexts (Phase 2.1)
        if s_true != 0:
            answerable_f1_scores.append(f1)
            if is_em:
                answerable_exact_match += 1

    mean_answerable_f1 = np.mean(answerable_f1_scores) if answerable_f1_scores else 0.0
    mean_answerable_em = (
        answerable_exact_match / len(answerable_f1_scores)
        if answerable_f1_scores
        else 0.0
    )

    return {
        "exact_match": exact_match / total,
        "token_f1": np.mean(f1_scores),
        "answerable_exact_match": mean_answerable_em,
        "answerable_token_f1": mean_answerable_f1,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a Transformer model for Task 2 (Spoiler Text Extraction)"
    )
    parser.add_argument(
        "--spoiler_type",
        type=str,
        required=True,
        choices=list(VARIANT_CONFIG.keys()),
        help="Which Task 2 variant to train: phrase, passage, or qa (unfiltered).",
    )
    parser.add_argument(
        "--train_path", type=str, required=True, help="Path to train JSONL file"
    )
    parser.add_argument(
        "--val_path", type=str, required=True, help="Path to validation JSONL file"
    )
    parser.add_argument("--model_name", type=str, help="Base QA model override")
    parser.add_argument("--output_dir", type=str, help="Output directory override")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--epochs", type=int, help="Training epochs")
    args = parser.parse_args()

    variant = VARIANT_CONFIG[args.spoiler_type]

    config = PipelineConfig()
    if args.model_name:
        setattr(config, variant["model_name_attr"], args.model_name)
    if args.output_dir:
        setattr(config, variant["output_dir_attr"], args.output_dir)
    if args.lr:
        config.task2_lr = args.lr
    if args.epochs:
        setattr(config, variant["epochs_attr"], args.epochs)

    model_name = getattr(config, variant["model_name_attr"])
    output_dir = getattr(config, variant["output_dir_attr"])
    epochs = getattr(config, variant["epochs_attr"])

    set_seed(config.seed)

    # Load (and optionally filter) datasets
    tag_filter = variant["tag_filter"]
    if tag_filter:
        print(f"Loading and filtering {tag_filter} datasets...")
        train_loader = JSONLLoader(args.train_path).filter_by_tag(tag_filter)
        val_loader = JSONLLoader(args.val_path).filter_by_tag(tag_filter)
        train_df, val_df = train_loader.df, val_loader.df
    else:
        print("Loading datasets for QA Extraction...")
        train_df = JSONLLoader(args.train_path).load_data()
        val_df = JSONLLoader(args.val_path).load_data()

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # Preprocess with type-specific max answer length
    print(f"Tokenizing and mapping text spans using: {model_name}")
    preprocessor = QAPreprocessor(
        model_name=model_name,
        max_length=config.task2_max_length,
        doc_stride=config.task2_doc_stride,
    )

    train_tokenized = train_dataset.map(
        preprocessor.prepare_train_features,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    val_tokenized = val_dataset.map(
        preprocessor.prepare_train_features,
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    preprocessor.export_stats(output_dir)

    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # Select best QA model using strictly answerable_token_f1 metric.
    # Keep only the single best checkpoint to save disk space.
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.task2_lr,
        per_device_train_batch_size=config.task2_batch_size,
        per_device_eval_batch_size=config.task2_batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="answerable_token_f1",
        greater_is_better=True,
        save_total_limit=1,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=DefaultDataCollator(),
        processing_class=preprocessor.tokenizer,
        compute_metrics=compute_qa_metrics,
    )

    print(
        f"\n=== LAUNCHING EXTRACTIVE TRANSFORMER {args.spoiler_type.upper()} QA TRAIN SEQUENCE ==="
    )
    trainer.train()

    print(f"Saving optimized {args.spoiler_type} QA model weights to {output_dir}...")
    trainer.save_model(output_dir)

    # Clean intermediate checkpoint folders
    for folder in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        shutil.rmtree(folder, ignore_errors=True)
        print(f"[CLEANUP] Pruned intermediate checkpoint: {folder}")

    print("[SUCCESS] Training execution routine complete.")


if __name__ == "__main__":
    main()
