from typing import Dict, List, Optional

import nltk
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics import classification_report, confusion_matrix

from src.utils import SPOILER_LABELS, join_text_list

# Silent download of METEOR dependencies to ensure smooth execution
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


def compute_pair_metrics(
    gt: List[str],
    pred: List[str],
    scorer: rouge_scorer.RougeScorer,
    smoother: Optional[SmoothingFunction] = None,
    include_meteor: bool = True,
) -> Dict[str, float]:
    """Computes BLEU-4, ROUGE-L, and (optionally) METEOR for a single ground-truth/prediction pair.

    This centralizes evaluation logic to avoid redundant calculations across the codebase.

    Args:
        gt (List[str]): Ground truth spoiler segment(s).
        pred (List[str]): Predicted spoiler segment(s).
        scorer (rouge_scorer.RougeScorer): Pre-instantiated ROUGE scorer.
        smoother (SmoothingFunction, optional): BLEU smoothing strategy. Defaults to SmoothingFunction().method1.
        include_meteor (bool): Whether to calculate METEOR score. Defaults to True.

    Returns:
        Dict[str, float]: Calculated metrics containing 'bleu', 'rouge_l', and 'meteor'.
    """
    if smoother is None:
        smoother = SmoothingFunction().method1

    gt_text = join_text_list(gt).lower()
    pred_text = join_text_list(pred).lower()

    # Calculate ROUGE-L
    rouge_l = scorer.score(gt_text, pred_text)["rougeL"].fmeasure

    gt_tokens = gt_text.split()
    pred_tokens = pred_text.split()

    if not gt_tokens or not pred_tokens:
        return {"bleu": 0.0, "rouge_l": rouge_l, "meteor": 0.0}

    # Calculate BLEU-4
    bleu = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoother)

    # Calculate METEOR
    meteor = meteor_score([gt_tokens], pred_tokens) if include_meteor else 0.0

    return {"bleu": bleu, "rouge_l": rouge_l, "meteor": meteor}


class AdvancedEvaluationSuite:
    """Comprehensive evaluation metrics suite for both task classification

    and text generation evaluation.
    """

    @staticmethod
    def evaluate_task1(y_true: List[str], y_pred: List[str]) -> None:
        """Computes and prints detailed classification metrics matching Table 1.

        Args:
            y_true (List[str]): True categorical labels.
            y_pred (List[str]): Predicted categorical labels.
        """
        print("\n=== TASK 1: CLASSIFICATION PERFORMANCE ===")
        # Get standard classification report
        report = classification_report(
            y_true,
            y_pred,
            labels=SPOILER_LABELS,
            digits=4,
            zero_division=0.0,
            output_dict=True,
        )
        accuracy = report["accuracy"]

        # Print standard classification report
        print(
            classification_report(
                y_true, y_pred, labels=SPOILER_LABELS, digits=4, zero_division=0.0
            )
        )

        # Print a beautiful publication-style Table 1
        print("\n" + "=" * 80)
        print(
            f"{'TASK 1: CLASSIFICATION PERFORMANCE (Table 1 - Leaderboard Format)':^80}"
        )
        print("=" * 80)
        print(f"Overall Accuracy: {accuracy:.4f}")
        print("-" * 80)
        print(
            f"{'Spoiler Type':<20} | {'Precision (Pr.)':<15} | {'Recall (Rec.)':<15} | {'F1-Score':<15}"
        )
        print("-" * 80)
        for label in SPOILER_LABELS:
            metrics = report[label]
            print(
                f"{label.capitalize():<20} | {metrics['precision']:<15.4f} | {metrics['recall']:<15.4f} | {metrics['f1-score']:<15.4f}"
            )
        print("=" * 80)

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred, labels=SPOILER_LABELS))

    @staticmethod
    def evaluate_task2(
        ground_truths: List[List[str]],
        predictions: List[List[str]],
        y_true_tags: Optional[List[str]] = None,
    ) -> None:
        """Computes and displays aggregated and group-wise metrics matching Table 2.

        Args:
            ground_truths (List[List[str]]): List of target spoiler references.
            predictions (List[List[str]]): List of predicted spoiler candidates.
            y_true_tags (List[str], optional): List of true categorical labels for grouping.

        Raises:
            ValueError: If the lengths of ground_truths and predictions mismatch.
        """
        if len(ground_truths) != len(predictions):
            raise ValueError(
                f"[ERROR] Mismatched sequence counts: ground_truths has {len(ground_truths)} items, "
                f"while predictions has {len(predictions)} items."
            )

        print("\n=== TASK 2: TEXT GENERATION PERFORMANCE ===")

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        smoother = SmoothingFunction().method1

        bleu_scores: List[float] = []
        rouge_l_scores: List[float] = []
        meteor_scores: List[float] = []

        # Pre-compute semantic BERTScore if available
        bert_scores = None
        try:
            import bert_score

            references_str = [join_text_list(gt) for gt in ground_truths]
            candidates_str = [join_text_list(pred) for pred in predictions]

            if references_str and candidates_str:
                _, _, F1 = bert_score.score(
                    candidates_str, references_str, lang="en", verbose=False
                )
                bert_scores = F1.numpy().tolist()
        except ImportError:
            pass

        # Compute individual metrics
        for idx, (gt, pred) in enumerate(zip(ground_truths, predictions)):
            metrics = compute_pair_metrics(gt, pred, scorer, smoother)
            bleu_scores.append(metrics["bleu"])
            rouge_l_scores.append(metrics["rouge_l"])
            meteor_scores.append(metrics["meteor"])

        print(f"Mean BLEU Score:  {np.mean(bleu_scores):.4f}")
        print(f"Mean ROUGE-L Score: {np.mean(rouge_l_scores):.4f}")
        print(f"Mean METEOR Score:  {np.mean(meteor_scores):.4f}")
        if bert_scores is not None:
            print(f"Mean BERTScore:     {np.mean(bert_scores):.4f}")

        # If tags are provided, compute and output Table 2 metrics grouped by category
        if y_true_tags is not None:
            if len(y_true_tags) != len(ground_truths):
                print(
                    "[WARNING] y_true_tags length mismatch. Skipping Table 2 grouped metrics."
                )
                return

            groups = ["all"] + SPOILER_LABELS
            grouped_metrics = {
                g: {"bleu": [], "meteor": [], "bert": []} for g in groups
            }

            for idx, tag in enumerate(y_true_tags):
                tag_clean = str(tag).lower().strip()
                b_score = bert_scores[idx] if bert_scores is not None else 0.0

                grouped_metrics["all"]["bleu"].append(bleu_scores[idx])
                grouped_metrics["all"]["meteor"].append(meteor_scores[idx])
                grouped_metrics["all"]["bert"].append(b_score)

                if tag_clean in grouped_metrics:
                    grouped_metrics[tag_clean]["bleu"].append(bleu_scores[idx])
                    grouped_metrics[tag_clean]["meteor"].append(meteor_scores[idx])
                    grouped_metrics[tag_clean]["bert"].append(b_score)

            print("\n" + "=" * 80)
            print(f"{'TASK 2: TEXT GENERATION PERFORMANCE (Table 2 - Grouped)':^80}")
            print("=" * 80)
            print(
                f"{'Group / Tag':<15} | {'BLEU-4 (BL4)':<15} | {'BERTScore (BSc.)':<18} | {'METEOR (MET)':<15}"
            )
            print("-" * 80)
            for g in groups:
                m_bleu = (
                    np.mean(grouped_metrics[g]["bleu"])
                    if grouped_metrics[g]["bleu"]
                    else 0.0
                )
                m_met = (
                    np.mean(grouped_metrics[g]["meteor"])
                    if grouped_metrics[g]["meteor"]
                    else 0.0
                )
                m_bert = (
                    np.mean(grouped_metrics[g]["bert"])
                    if grouped_metrics[g]["bert"]
                    else 0.0
                )

                b_str = (
                    f"{m_bert:<18.4f}" if bert_scores is not None else f"{'N/A':<18}"
                )
                print(
                    f"{g.capitalize():<15} | {m_bleu:<15.4f} | {b_str} | {m_met:<15.4f}"
                )
            print("=" * 80)
