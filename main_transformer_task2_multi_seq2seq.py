"""Execution runner for fine-tuning generative Seq2Seq models (e.g. T5) on multi-type spoilers.

Loads the training and validation datasets, filters for multi-spoilers exclusively,
preprocesses textual inputs, and launches sequence-to-sequence training.
"""

import argparse
import glob
import os
import shutil

from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from src.config import PipelineConfig
from src.data_loader import JSONLLoader
from src.seq2seq_qa_preprocessor import Seq2SeqQAPreprocessor


def main() -> None:
    """Parses training configuration args, preprocesses T5 targets, and runs the unified

    generative Seq2Seq training sequence for multi-spoilers.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune a Flan-T5 model for Generative Multi Spoiler Generation"
    )
    parser.add_argument(
        "--train_path", type=str, required=True, help="Path to train JSONL file"
    )
    parser.add_argument(
        "--val_path", type=str, required=True, help="Path to validation JSONL file"
    )
    parser.add_argument(
        "--model_name", type=str, default="google/flan-t5-base", help="Base T5 model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_multi_seq2seq",
        help="Output directory",
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    args = parser.parse_args()

    config = PipelineConfig()
    set_seed(config.seed)

    # Load and filter dataset for MULTI only.
    # Note: filter_by_tag automatically calls reset_index(drop=True) by default,
    # ensuring standard contiguous indices when wrapping inside HF Datasets.
    print("Loading and filtering multi-type datasets...")
    train_loader = JSONLLoader(args.train_path).filter_by_tag("multi")
    val_loader = JSONLLoader(args.val_path).filter_by_tag("multi")

    train_dataset = Dataset.from_pandas(train_loader.df)
    val_dataset = Dataset.from_pandas(val_loader.df)

    preprocessor = Seq2SeqQAPreprocessor(model_name=args.model_name)

    train_tokenized = train_dataset.map(
        preprocessor.prepare_features,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    val_tokenized = val_dataset.map(
        preprocessor.prepare_features,
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=10,
        load_best_model_at_end=True,
        predict_with_generate=True,
        save_total_limit=1,  # Keep only the single best checkpoint to save disk space
        fp16=True,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=preprocessor.tokenizer, model=model
        ),
        processing_class=preprocessor.tokenizer,
    )

    print("\n=== LAUNCHING GENERATIVE SEQ2SEQ MULTI-SPOILER TRAIN SEQUENCE ===")
    trainer.train()

    print(f"Saving optimized Seq2Seq model weights to {args.output_dir}...")
    trainer.save_model(args.output_dir)

    # Clean intermediate checkpoint folders safely ignoring file-locking errors
    for folder in glob.glob(os.path.join(args.output_dir, "checkpoint-*")):
        shutil.rmtree(folder, ignore_errors=True)
        print(f"[CLEANUP] Pruned intermediate checkpoint: {folder}")

    print("[SUCCESS] Training execution routine complete.")


if __name__ == "__main__":
    main()
