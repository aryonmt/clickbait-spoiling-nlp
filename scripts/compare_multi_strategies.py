import argparse
import os
import sys

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction
from rouge_score import rouge_scorer

# Ensure the project root (parent of this scripts/ folder) is importable,
# so this file can be run directly (e.g. `python scripts/compare_multi_strategies.py`)
# as well as via `python -m scripts.compare_multi_strategies`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baseline_heuristics import HeuristicSpoilerGenerator
from src.data_loader import JSONLLoader
from src.evaluation import compute_pair_metrics
from src.retrieval_generator import RetrievalSpoilerGenerator
from src.utils import extract_primary_tag


def evaluate_predictions(ground_truths: list, predictions: list):
    """Computes BLEU and ROUGE-L mean scores."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smoother = SmoothingFunction().method1

    bleu_scores = []
    rouge_l_scores = []

    for gt, pred in zip(ground_truths, predictions):
        # include_meteor=False: this script only reports BLEU/ROUGE-L, matching
        # its original scope (skips METEOR's extra cost).
        metrics = compute_pair_metrics(gt, pred, scorer, smoother, include_meteor=False)
        bleu_scores.append(metrics["bleu"])
        rouge_l_scores.append(metrics["rouge_l"])

    return np.mean(bleu_scores), np.mean(rouge_l_scores)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-type Spoiler Generation A/B/C Test Suite"
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="data/validation.jsonl",
        help="Path to validation data",
    )
    args = parser.parse_args()

    if not os.path.exists(args.val_path):
        print(f"[ERROR] Validation file not found at: {args.val_path}")
        return

    loader = JSONLLoader(args.val_path)
    df = loader.load_data()

    multi_df = df[df["tags"].apply(extract_primary_tag) == "multi"].copy()
    total_cases = len(multi_df)
    print(f"\nEvaluating {total_cases} multi-type spoiler cases...")

    ground_truths = multi_df["spoiler"].tolist()

    # Strategy A: Baseline Heuristic (combines first sentences of top 3 paragraphs)
    predicted_spoilers_a = []
    for _, row in multi_df.iterrows():
        paragraphs = row["targetParagraphs"]
        spoilers_a = HeuristicSpoilerGenerator._extract_heuristic(paragraphs, "multi")
        predicted_spoilers_a.append(spoilers_a)
    bleu_a, rouge_a = evaluate_predictions(ground_truths, predicted_spoilers_a)

    print(
        "\n=========================================================================="
    )
    print("A/B/C TEST REPORT: MULTI-TYPE GENERATION STRATEGY")
    print("==========================================================================")
    print("Strategy A: Baseline Heuristics (First-sentence heuristics)")
    print(f"  - Mean BLEU:    {bleu_a:.4f}")
    print(f"  - Mean ROUGE-L: {rouge_a:.4f}")
    print("--------------------------------------------------------------------------")

    best_strategy = "Strategy A"
    best_bleu = bleu_a
    best_rouge = rouge_a
    best_k = 3

    # Compare Jaccard and TF-IDF across different k-values (2, 3, 4, 5)
    for k in [2, 3, 4, 5]:
        # Strategy B: Jaccard
        pred_jaccard = []
        for _, row in multi_df.iterrows():
            post_text = (
                " ".join(row["postText"])
                if isinstance(row["postText"], list)
                else str(row["postText"])
            )
            paragraphs = row["targetParagraphs"]
            spoilers = RetrievalSpoilerGenerator.generate_multi_spoiler(
                post_text, paragraphs, top_k=k, method="jaccard"
            )
            pred_jaccard.append(spoilers)
        bleu_j, rouge_j = evaluate_predictions(ground_truths, pred_jaccard)
        print(f"Strategy B: Jaccard Retrieval (top_k={k})")
        print(f"  - Mean BLEU:    {bleu_j:.4f} | ROUGE-L: {rouge_j:.4f}")

        if bleu_j > best_bleu:
            best_bleu = bleu_j
            best_rouge = rouge_j
            best_strategy = f"Strategy B (Jaccard, top_k={k})"
            best_k = k

        # Strategy C: TF-IDF
        pred_tfidf = []
        for _, row in multi_df.iterrows():
            post_text = (
                " ".join(row["postText"])
                if isinstance(row["postText"], list)
                else str(row["postText"])
            )
            paragraphs = row["targetParagraphs"]
            spoilers = RetrievalSpoilerGenerator.generate_multi_spoiler(
                post_text, paragraphs, top_k=k, method="tfidf"
            )
            pred_tfidf.append(spoilers)
        bleu_t, rouge_t = evaluate_predictions(ground_truths, pred_tfidf)
        print(f"Strategy C: TF-IDF Cosine Retrieval (top_k={k})")
        print(f"  - Mean BLEU:    {bleu_t:.4f}")
        print(f"  - Mean ROUGE-L: {rouge_t:.4f}")
        print(
            "--------------------------------------------------------------------------"
        )

        if bleu_t > best_bleu:
            best_bleu = bleu_t
            best_rouge = rouge_t
            best_strategy = f"Strategy C (TF-IDF, top_k={k})"
            best_k = k

    print(f"[SUCCESS] Winner identified: {best_strategy}")
    print(f"  - Optimal k:       {best_k}")
    print(f"  - Winning BLEU:    {best_bleu:.4f}")
    print(f"  - Winning ROUGE-L: {best_rouge:.4f}")
    print("==========================================================================")


if __name__ == "__main__":
    main()
