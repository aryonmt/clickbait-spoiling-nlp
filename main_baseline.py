"""Execution runner for training and evaluating the clickbait spoiling baseline pipeline.

This script trains a TF-IDF + Logistic Regression classifier for Task 1, generates baseline
rule-based heuristic spoilers for Task 2, and evaluates both results using advanced metrics.
"""

import os
import sys

import pandas as pd

from src.baseline_classifier import BaselineSpoilerClassifier
from src.baseline_heuristics import HeuristicSpoilerGenerator
from src.data_loader import JSONLLoader
from src.evaluation import AdvancedEvaluationSuite
from src.utils import extract_primary_tag


def main() -> None:
    """Orchestrates the entire baseline flow: data loading, training, inference,

    evaluation, and generating submission files.
    """
    train_path = os.path.join("data", "train.jsonl")
    val_path = os.path.join("data", "validation.jsonl")
    output_path = "run.jsonl"

    try:
        train_loader = JSONLLoader(train_path)
        train_df = train_loader.load_data()

        val_loader = JSONLLoader(val_path)
        val_df = val_loader.load_data()

        # Defensive reset: guards against IndexError if val_df is ever produced
        # by a filtering/sampling step upstream that leaves non-contiguous
        # row labels (predicted_tags/predicted_spoilers are plain 0-indexed lists).
        val_df = val_df.reset_index(drop=True)
    except FileNotFoundError as e:
        print(f"[ERROR] Required dataset not found: {e}")
        print(
            "Ensure both train.jsonl and validation.jsonl are placed in the 'data/' folder."
        )
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to parse JSONL data files: {e}")
        sys.exit(1)

    # Initialize and train Task 1 Baseline Classifier
    classifier = BaselineSpoilerClassifier()
    classifier.fit(train_df)
    predicted_tags = classifier.predict(val_df)

    # Generate rule-based Task 2 Spoilers using Heuristics
    generator = HeuristicSpoilerGenerator()
    predicted_spoilers = generator.generate(val_df, predicted_tags)

    y_true_tags = val_df["tags"].apply(extract_primary_tag).tolist()
    ground_truth_spoilers = val_df["spoiler"].tolist()

    # Evaluate performance for both tasks using the new formatted leaderboard metrics
    AdvancedEvaluationSuite.evaluate_task1(y_true_tags, predicted_tags)
    AdvancedEvaluationSuite.evaluate_task2(
        ground_truth_spoilers, predicted_spoilers, y_true_tags
    )

    # Save standardized submission artifact defensively using contiguous 0-based
    # enumeration rather than pandas row labels.
    print(f"\nSaving final baseline execution file to {output_path}...")
    output_records = []
    for i, (_, row) in enumerate(val_df.iterrows()):
        record = {
            "uuid": row["uuid"],
            "spoilerType": predicted_tags[i],
            "spoiler": predicted_spoilers[i],
        }
        output_records.append(record)

    output_df = pd.DataFrame(output_records)
    output_df.to_json(output_path, orient="records", lines=True)
    print("[SUCCESS] Baseline run accomplished safely.")


if __name__ == "__main__":
    main()
