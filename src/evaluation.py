from typing import Dict, List, Optional

import nltk
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics import classification_report, confusion_matrix

from src.utils import SPOILER_LABELS, join_text_list

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
        """Computes and prints detailed classification metrics for Task 1 (Spoiler Type).

        Args:
            y_true (List[str]): True categorical labels.
            y_pred (List[str]): Predicted categorical labels.
        """
        print("\n=== TASK 1: CLASSIFICATION PERFORMANCE ===")
        print(
            classification_report(
                y_true, y_pred, labels=SPOILER_LABELS, digits=4, zero_division=0.0
            )
        )
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred, labels=SPOILER_LABELS))

    @staticmethod
    def evaluate_task2(
        ground_truths: List[List[str]], predictions: List[List[str]]
    ) -> None:
        """Computes and displays aggregated BLEU-4, ROUGE-L, METEOR, and optional BERTScore.

        Args:
            ground_truths (List[List[str]]): List of target spoiler references.
            predictions (List[List[str]]): List of predicted spoiler candidates.

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

            references_str = [join_text_list(gt) for gt in ground_truths]
            candidates_str = [join_text_list(pred) for pred in predictions]

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
