from typing import List

import nltk
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics import classification_report, confusion_matrix

from src.utils import SPOILER_LABELS

# Silent download of METEOR dependencies to ensure smooth execution
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

from nltk.translate.meteor_score import meteor_score


def compute_pair_metrics(
    gt: List[str],
    pred: List[str],
    scorer: rouge_scorer.RougeScorer,
    smoother=None,
    include_meteor: bool = True,
) -> dict:
    """Computes BLEU-4, ROUGE-L, and (optionally) METEOR for a single

    ground-truth/prediction pair. Centralizes logic that was previously
    duplicated across AdvancedEvaluationSuite.evaluate_task2,
    EvaluationReportBuilder._compute_generation_metrics (visualization_suite.py),
    and evaluate_predictions (scripts/compare_multi_strategies.py).

    Args:
        gt: Ground truth spoiler, as a list of strings or a raw string.
        pred: Predicted spoiler, as a list of strings or a raw string.
        scorer: A pre-built rouge_scorer.RougeScorer(["rougeL"]) instance.
        smoother: Optional BLEU SmoothingFunction method; defaults to method1.
        include_meteor: Whether to also compute METEOR (skip for callers that
          don't need it, to avoid its extra cost).

    Returns:
        dict: {"bleu": float, "rouge_l": float, "meteor": float} (meteor is
          0.0 and omitted from cost if include_meteor is False).
    """
    if smoother is None:
        smoother = SmoothingFunction().method1

    gt_text = " ".join(gt).lower() if isinstance(gt, list) else str(gt).lower()
    pred_text = " ".join(pred).lower() if isinstance(pred, list) else str(pred).lower()

    rouge_l = scorer.score(gt_text, pred_text)["rougeL"].fmeasure

    gt_tokens = gt_text.split()
    pred_tokens = pred_text.split()

    if not gt_tokens or not pred_tokens:
        return {"bleu": 0.0, "rouge_l": rouge_l, "meteor": 0.0}

    bleu = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoother)
    meteor = meteor_score([gt_tokens], pred_tokens) if include_meteor else 0.0

    return {"bleu": bleu, "rouge_l": rouge_l, "meteor": meteor}


class AdvancedEvaluationSuite:
    """Comprehensive evaluation metrics suite for both challenge tasks."""

    @staticmethod
    def evaluate_task1(y_true: List[str], y_pred: List[str]) -> None:
        """Computes and displays classification metrics for Task 1.

        Args:
            y_true (List[str]): Ground truth labels.
            y_pred (List[str]): Model predictions.
        """
        print("\n=== TASK 1: CLASSIFICATION PERFORMANCE ===")
        print(classification_report(y_true, y_pred, labels=SPOILER_LABELS, digits=4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred, labels=SPOILER_LABELS))

    @staticmethod
    def evaluate_task2(
        ground_truths: List[List[str]], predictions: List[List[str]]
    ) -> None:
        """Computes BLEU, ROUGE-L, and METEOR scores for the generated spoilers.

        Args:
            ground_truths (List[List[str]]): Target human spoilers.
            predictions (List[List[str]]): Generated baseline spoilers.
        """
        print("\n=== TASK 2: TEXT GENERATION PERFORMANCE ===")

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        smoother = SmoothingFunction().method1

        bleu_scores = []
        rouge_l_scores = []
        meteor_scores = []

        for gt, pred in zip(ground_truths, predictions):
            metrics = compute_pair_metrics(gt, pred, scorer, smoother)
            bleu_scores.append(metrics["bleu"])
            rouge_l_scores.append(metrics["rouge_l"])
            meteor_scores.append(metrics["meteor"])

        print(f"Mean BLEU Score: {np.mean(bleu_scores):.4f}")
        print(f"Mean ROUGE-L Score: {np.mean(rouge_l_scores):.4f}")
        print(f"Mean METEOR Score: {np.mean(meteor_scores):.4f}")

        # Compute semantic BERTScore if available
        try:
            import bert_score

            references_str = [
                " ".join(gt) if isinstance(gt, list) else str(gt)
                for gt in ground_truths
            ]
            candidates_str = [
                " ".join(pred) if isinstance(pred, list) else str(pred)
                for pred in predictions
            ]

            if references_str and candidates_str:
                _, _, F1 = bert_score.score(
                    candidates_str, references_str, lang="en", verbose=False
                )
                mean_bert = float(F1.mean().item())
                print(f"Mean BERTScore:    {mean_bert:.4f}")
        except ImportError:
            print(
                "[INFO] 'bert-score' library not installed. Skipping semantic BERTScore display."
            )
