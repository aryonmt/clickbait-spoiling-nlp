import json
import os
from datetime import datetime
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk.translate.bleu_score import SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import classification_report, confusion_matrix

from src.evaluation import compute_pair_metrics
from src.logging_setup import setup_logger
from src.utils import SPOILER_LABELS, extract_primary_tag, join_text_list

logger = setup_logger("visualization_suite")


class EvaluationReportBuilder:
    """Computes robust evaluation metrics and generates visual charts and HTML reports

    comparing predictions against validation ground truth.

    Attributes:
        predictions_path (str): Path to the predictions JSONL file.
        val_path (str): Path to the validation ground truth JSONL file.
        report_dir (str): Targeted unique directory created for exporting results.
    """

    def __init__(
        self, predictions_path: str, val_path: str, output_dir: str = "reports"
    ):
        """Initializes the EvaluationReportBuilder with target file paths.

        Args:
            predictions_path (str): Path to predictions JSONL file.
            val_path (str): Path to validation JSONL file.
            output_dir (str): Root directory for saving report assets. Defaults to "reports".
        """
        self.predictions_path: str = predictions_path
        self.val_path: str = val_path

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir: str = os.path.join(output_dir, timestamp)
        os.makedirs(self.report_dir, exist_ok=True)

    def _load_and_align_data(self) -> pd.DataFrame:
        """Loads both validation data and model predictions and aligns them on UUID.

        Returns:
            pd.DataFrame: Merged DataFrame containing aligned actual and predicted records.
        """
        logger.info(
            f"Loading files:\n - Ground truth: {self.val_path}\n - Predictions: {self.predictions_path}"
        )

        val_df = pd.read_json(self.val_path, lines=True)
        pred_df = pd.read_json(self.predictions_path, lines=True)

        merged_df = pd.merge(val_df, pred_df, on="uuid", suffixes=("_true", "_pred"))
        logger.info(f"Successfully aligned {len(merged_df)} validation samples.")
        return merged_df

    def _compute_generation_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes BLEU-4, ROUGE-L, METEOR, and optional semantic BERTScore per instance."""
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        smoother = SmoothingFunction().method1

        bleus: List[float] = []
        rouges: List[float] = []
        meteors: List[float] = []

        for _, row in df.iterrows():
            gt = row["spoiler_true"]
            pred = row["spoiler_pred"]

            metrics = compute_pair_metrics(gt, pred, scorer, smoother)
            bleus.append(metrics["bleu"])
            rouges.append(metrics["rouge_l"])
            meteors.append(metrics["meteor"])

        df["bleu"] = bleus
        df["rouge_l"] = rouges
        df["meteor"] = meteors

        # Calculate semantic BERTScore if available
        bert_scores = [0.0] * len(df)
        try:
            import bert_score

            references_str = [join_text_list(gt) for gt in df["spoiler_true"]]
            candidates_str = [join_text_list(pred) for pred in df["spoiler_pred"]]
            if references_str and candidates_str:
                _, _, F1 = bert_score.score(
                    candidates_str, references_str, lang="en", verbose=False
                )
                bert_scores = F1.numpy().tolist()
        except ImportError:
            pass

        df["bert_score"] = bert_scores
        df["true_tag"] = df["tags"].apply(extract_primary_tag)
        df["pred_tag"] = df["spoilerType"]

        return df

    def _plot_confusion_matrix(self, y_true: List[str], y_pred: List[str]) -> None:
        """Generates and saves the classification confusion matrix heatmap.

        Args:
            y_true (List[str]): List of true primary labels.
            y_pred (List[str]): List of predicted primary labels.
        """
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_true, y_pred, labels=SPOILER_LABELS)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=SPOILER_LABELS,
            yticklabels=SPOILER_LABELS,
        )
        plt.title("Task 1: Confusion Matrix Heatmap")
        plt.ylabel("Ground Truth Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        plt.savefig(os.path.join(self.report_dir, "confusion_matrix.png"), dpi=150)
        plt.close()

    def _plot_classification_metrics(self, report_dict: Dict[str, Any]) -> None:
        """Generates and saves bar plots of precision, recall, and f1-scores.

        Args:
            report_dict (Dict[str, Any]): Dictionary output from classification_report.
        """
        metrics_data: List[Dict[str, Any]] = []

        for cls in SPOILER_LABELS:
            if cls in report_dict:
                metrics_data.append(
                    {
                        "Class": cls,
                        "Precision": report_dict[cls]["precision"],
                        "Recall": report_dict[cls]["recall"],
                        "F1-Score": report_dict[cls]["f1-score"],
                    }
                )

        df_plot = pd.DataFrame(metrics_data).melt(
            id_vars="Class", var_name="Metric", value_name="Score"
        )

        plt.figure(figsize=(8, 5))
        sns.barplot(data=df_plot, x="Class", y="Score", hue="Metric", palette="Set2")
        plt.title("Task 1: Classification Metrics Breakdown")
        plt.ylim(0, 1.05)
        plt.tight_layout()

        plt.savefig(
            os.path.join(self.report_dir, "classification_metrics.png"), dpi=150
        )
        plt.close()

    def _plot_confidence_distribution(self, df: pd.DataFrame) -> None:
        """Generates a histogram comparing confidence scores for correct vs incorrect predictions.

        Args:
            df (pd.DataFrame): Dataframe containing 'confidence', 'true_tag', and 'pred_tag'.
        """
        if "confidence" not in df.columns:
            logger.warning(
                "Softmax confidence scores not found in predictions. Skipping histogram."
            )
            return

        df["is_correct"] = df["true_tag"] == df["pred_tag"]

        plt.figure(figsize=(7, 5))
        sns.histplot(
            data=df,
            x="confidence",
            hue="is_correct",
            multiple="stack",
            palette={True: "mediumseagreen", False: "salmon"},
            bins=20,
            alpha=0.8,
        )
        plt.title("Task 1: Prediction Confidence Distribution")
        plt.xlabel("Probability Confidence")
        plt.ylabel("Sample Count")
        plt.tight_layout()

        plt.savefig(
            os.path.join(self.report_dir, "confidence_distribution.png"), dpi=150
        )
        plt.close()

    def _plot_error_by_length(self, df: pd.DataFrame) -> None:
        """Generates a bar plot showing error rates categorized by the article length (word counts)."""
        df["article_len"] = df["targetParagraphs"].apply(
            lambda x: (
                len(join_text_list(x).split()) if isinstance(x, (list, str)) else 0
            )
        )
        df["is_correct"] = df["true_tag"] == df["pred_tag"]

        # Categorize length into bins
        def get_bin(length):
            if length < 200:
                return "Short (<200w)"
            elif length < 600:
                return "Medium (200w-600w)"
            return "Long (>600w)"

        df["length_bin"] = df["article_len"].apply(get_bin)

        # Calculate error rates per bin
        binned = (
            df.groupby("length_bin")["is_correct"]
            .value_counts(normalize=True)
            .unstack()
            .fillna(0)
        )
        if False in binned.columns:
            binned["error_rate"] = binned[False]
        else:
            binned["error_rate"] = 0.0

        # Standardize bin orders
        bin_order = ["Short (<200w)", "Medium (200w-600w)", "Long (>600w)"]
        binned = binned.reindex(bin_order).fillna(0)

        plt.figure(figsize=(7, 5))
        sns.barplot(
            x=binned.index,
            y=binned["error_rate"] * 100,
            palette="Oranges_r",
            hue=binned.index,
            legend=False,
        )
        plt.title("Task 1: Error Rate by Article Word Count")
        plt.xlabel("Article Length Group")
        plt.ylabel("Error Rate (%)")
        plt.ylim(0, 100)
        plt.tight_layout()

        plt.savefig(os.path.join(self.report_dir, "error_by_length.png"), dpi=150)
        plt.close()

    def _plot_task2_by_type(self, df: pd.DataFrame) -> None:
        """Generates and saves boxplots of text generation metrics per spoiler type."""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.boxplot(
            data=df,
            x="true_tag",
            y="bleu",
            hue="true_tag",
            legend=False,
            palette="Set3",
        )
        plt.title("BLEU Score Distribution by Spoiler Type")
        plt.xlabel("Spoiler Type")
        plt.ylabel("BLEU Score")
        plt.ylim(-0.05, 1.05)

        plt.subplot(1, 2, 2)
        sns.boxplot(
            data=df,
            x="true_tag",
            y="rouge_l",
            hue="true_tag",
            legend=False,
            palette="Set3",
        )
        plt.title("ROUGE-L Score Distribution by Spoiler Type")
        plt.xlabel("Spoiler Type")
        plt.ylabel("ROUGE-L Score")
        plt.ylim(-0.05, 1.05)

        plt.tight_layout()
        plt.savefig(os.path.join(self.report_dir, "task2_scores_by_type.png"), dpi=150)
        plt.close()

    def _plot_length_comparison(self, df: pd.DataFrame) -> None:
        """Generates and saves a predicted length vs actual length scatterplot."""
        df["gt_len"] = df["spoiler_true"].apply(
            lambda x: (
                len(join_text_list(x).split()) if isinstance(x, (list, str)) else 0
            )
        )
        df["pred_len"] = df["spoiler_pred"].apply(
            lambda x: (
                len(join_text_list(x).split()) if isinstance(x, (list, str)) else 0
            )
        )

        plt.figure(figsize=(7, 5))
        sns.scatterplot(
            data=df,
            x="gt_len",
            y="pred_len",
            hue="true_tag",
            alpha=0.7,
            palette="Dark2",
        )

        max_val = max(df["gt_len"].max(), df["pred_len"].max(), 10)
        plt.plot([0, max_val], [0, max_val], "k--", alpha=0.5)

        plt.title("Spoiler Length Comparison")
        plt.xlabel("Ground Truth Word Count")
        plt.ylabel("Predicted Word Count")
        plt.xlim(-2, max_val + 5)
        plt.ylim(-2, max_val + 5)
        plt.tight_layout()

        plt.savefig(os.path.join(self.report_dir, "length_comparison.png"), dpi=150)
        plt.close()

    def _generate_html_report(self, summary: dict, worst_cases: list):
        """Creates an integrated HTML page showcasing metrics and plots."""
        # Include confidence and length bin charts in HTML grid
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Clickbait Spoiling System Evaluation — Version 5</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f8f9fa; color: #333; }}
        h1, h2 {{ color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; margin-bottom: 15px; }}
        th, td {{ padding: 12px; border: 1px solid #ddd; text-align: left; }}
        th {{ background-color: #34495e; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        img {{ max-width: 100%; height: auto; border-radius: 4px; display: block; margin: 0 auto; }}
    </style>
</head>
<body>
    <h1>Clickbait Spoiling Evaluation Report (V5 Overhaul)</h1>
    <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

    <div class="card">
        <h2>System Performance Metrics (Overall Leaderboard Summary)</h2>
        <div class="grid" style="grid-template-columns: repeat(4, 1fr);">
            <div class="card" style="text-align: center;">
                <p>Task 1 Macro F1</p>
                <p class="metric-value">{summary["task1_macro_f1"]:.4f}</p>
            </div>
            <div class="card" style="text-align: center;">
                <p>Task 1 Accuracy</p>
                <p class="metric-value">{summary["task1_accuracy"]:.4f}</p>
            </div>
            <div class="card" style="text-align: center;">
                <p>Task 2 Mean BLEU-4</p>
                <p class="metric-value">{summary["task2_mean_bleu"]:.4f}</p>
            </div>
            <div class="card" style="text-align: center;">
                <p>Task 2 Mean BERTScore</p>
                <p class="metric-value">{summary["task2_mean_bert_score"]:.4f}</p>
            </div>
        </div>
    </div>

    <div class="card" style="margin-top: 20px;">
        <h2>Task 1: Classification Performance Table (Paper Format)</h2>
        <table>
            <thead>
                <tr>
                    <th>Spoiler Type (Class)</th>
                    <th>Precision (Pr.)</th>
                    <th>Recall (Rec.)</th>
                    <th>F1-Score</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><b>Phrase</b></td>
                    <td>{summary["task1_by_type"]["phrase"]["precision"]:.4f}</td>
                    <td>{summary["task1_by_type"]["phrase"]["recall"]:.4f}</td>
                    <td>{summary["task1_by_type"]["phrase"]["f1"]:.4f}</td>
                </tr>
                <tr>
                    <td><b>Passage</b></td>
                    <td>{summary["task1_by_type"]["passage"]["precision"]:.4f}</td>
                    <td>{summary["task1_by_type"]["passage"]["recall"]:.4f}</td>
                    <td>{summary["task1_by_type"]["passage"]["f1"]:.4f}</td>
                </tr>
                <tr>
                    <td><b>Multi</b></td>
                    <td>{summary["task1_by_type"]["multi"]["precision"]:.4f}</td>
                    <td>{summary["task1_by_type"]["multi"]["recall"]:.4f}</td>
                    <td>{summary["task1_by_type"]["multi"]["f1"]:.4f}</td>
                </tr>
                <tr style="background-color: #eaeded; font-weight: bold;">
                    <td>Overall Accuracy</td>
                    <td colspan="2" style="text-align: center;">-</td>
                    <td>{summary["task1_accuracy"]:.4f}</td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="card" style="margin-top: 20px;">
        <h2>Task 2: Text Generation Performance Table (Paper Format)</h2>
        <table>
            <thead>
                <tr>
                    <th>Group / Spoiler Tag</th>
                    <th>BLEU-4 (BL4)</th>
                    <th>BERTScore (BSc.)</th>
                    <th>METEOR (MET)</th>
                </tr>
            </thead>
            <tbody>
                <tr style="background-color: #ebf5fb; font-weight: bold;">
                    <td>All (Overall)</td>
                    <td>{summary["task2_mean_bleu"]:.4f}</td>
                    <td>{summary["task2_mean_bert_score"]:.4f}</td>
                    <td>{summary["task2_mean_meteor"]:.4f}</td>
                </tr>
                <tr>
                    <td><b>Phrase</b></td>
                    <td>{summary["task2_by_type"]["phrase"]["mean_bleu"]:.4f}</td>
                    <td>{summary["task2_by_type"]["phrase"]["mean_bert_score"]:.4f}</td>
                    <td>{summary["task2_by_type"]["phrase"]["mean_meteor"]:.4f}</td>
                </tr>
                <tr>
                    <td><b>Passage</b></td>
                    <td>{summary["task2_by_type"]["passage"]["mean_bleu"]:.4f}</td>
                    <td>{summary["task2_by_type"]["passage"]["mean_bert_score"]:.4f}</td>
                    <td>{summary["task2_by_type"]["passage"]["mean_meteor"]:.4f}</td>
                </tr>
                <tr>
                    <td><b>Multi</b></td>
                    <td>{summary["task2_by_type"]["multi"]["mean_bleu"]:.4f}</td>
                    <td>{summary["task2_by_type"]["multi"]["mean_bert_score"]:.4f}</td>
                    <td>{summary["task2_by_type"]["multi"]["mean_meteor"]:.4f}</td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="grid">
        <div class="card">
            <h2>Task 1: Confusion Matrix</h2>
            <img src="confusion_matrix.png" alt="Confusion Matrix">
        </div>
        <div class="card">
            <h2>Task 1: Metrics per Class</h2>
            <img src="classification_metrics.png" alt="Classification Metrics">
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h2>Task 1: Confidence Distribution</h2>
            <img src="confidence_distribution.png" alt="Confidence Distribution">
        </div>
        <div class="card">
            <h2>Task 1: Error Rate by Length</h2>
            <img src="error_by_length.png" alt="Error rate by Length">
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h2>Task 2: Performance by Type</h2>
            <img src="task2_scores_by_type.png" alt="Task 2 Metrics by Type">
        </div>
        <div class="card">
            <h2>Task 2: Length Correlation</h2>
            <img src="length_comparison.png" alt="Length Comparison">
        </div>
    </div>

    <div class="card" style="margin-top: 20px;">
        <h2>Worst Performing Prediction Samples (sorted by BLEU)</h2>
        <table>
            <thead>
                <tr>
                    <th>UUID</th>
                    <th>Type</th>
                    <th>Ground Truth Spoiler</th>
                    <th>Predicted Spoiler</th>
                    <th>BLEU Score</th>
                </tr>
            </thead>
            <tbody>
        """

        for case in worst_cases:
            html_content += f"""
                <tr>
                    <td>{case["uuid"]}</td>
                    <td><span style="font-weight:bold;">{case["true_tag"]}</span></td>
                    <td style="color: #27ae60;">{case["gt"]}</td>
                    <td style="color: #c0392b;">{case["pred"]}</td>
                    <td>{case["bleu"]:.4f}</td>
                </tr>
            """

        html_content += """
            </tbody>
        </table>
    </div>
</body>
</html>
        """

        with open(
            os.path.join(self.report_dir, "report.html"), "w", encoding="utf-8"
        ) as f:
            f.write(html_content)

    def build_report(self) -> None:
        """Orchestrates metric computations, plotting, and report emission."""
        df = self._load_and_align_data()
        df = self._compute_generation_metrics(df)

        # 1. Task 1 Metrics
        y_true = df["true_tag"].tolist()
        y_pred = df["pred_tag"].tolist()

        report_dict = classification_report(y_true, y_pred, output_dict=True)
        acc = report_dict["accuracy"]
        macro_f1 = report_dict["macro avg"]["f1-score"]

        self._plot_confusion_matrix(y_true, y_pred)
        self._plot_classification_metrics(report_dict)
        self._plot_confidence_distribution(df)
        self._plot_error_by_length(df)

        # 2. Task 2 Metrics
        mean_bleu = df["bleu"].mean()
        mean_rouge = df["rouge_l"].mean()

        self._plot_task2_by_type(df)
        self._plot_length_comparison(df)

        # Phase 3.2: Safely parse fallback rate from real label quality statistics if available
        fallback_rate = 0.0
        fallback_label = ""

        # Look for preprocessing statistics JSON inside predictions parallel folder
        stats_path = os.path.join(
            os.path.dirname(self.predictions_path), "label_quality_stats.json"
        )
        alt_stats_path = "results_qa/label_quality_stats.json"

        stats_file = (
            stats_path
            if os.path.exists(stats_path)
            else (alt_stats_path if os.path.exists(alt_stats_path) else None)
        )

        if stats_file:
            try:
                with open(stats_file, "r", encoding="utf-8") as f:
                    stats_data = json.load(f)
                fallback_rate = stats_data.get("fallback_rate", 0.0)
                fallback_label = f"{fallback_rate * 100:.2f}%"
                logger.info(f"Loaded exact CLS fallback rate from: {stats_file}")
            except Exception as e:
                logger.warning(
                    f"Error parsing stats JSON file: {e}. Reverting to estimation."
                )
                stats_file = None

        if not stats_file:
            # Revert back to estimated proxies if the label stats are missing
            fallback_cases = 0
            for _, row in df.iterrows():
                pred = row["spoiler_pred"]
                paragraphs = row["targetParagraphs"]
                first_para = (
                    paragraphs[0]
                    if isinstance(paragraphs, list) and len(paragraphs) > 0
                    else ""
                )
                if not pred or (
                    len(pred) == 1 and pred[0] == first_para and first_para != ""
                ):
                    fallback_cases += 1
            fallback_rate = fallback_cases / len(df)
            fallback_label = f"{fallback_rate * 100:.2f}% (تخمینی، نه دقیق)"

        # Group stats for JSON export
        summary_metrics = {
            "task1_accuracy": acc,
            "task1_macro_f1": macro_f1,
            "task2_mean_bleu": mean_bleu,
            "task2_mean_rouge_l": mean_rouge,
            "task2_mean_meteor": float(df["meteor"].mean()),
            "task2_mean_bert_score": float(df["bert_score"].mean()),
            "task2_cls_fallback_rate": fallback_rate,
            "task2_cls_fallback_label": fallback_label,
            "task1_by_type": {
                tag: {
                    "precision": report_dict[tag]["precision"],
                    "recall": report_dict[tag]["recall"],
                    "f1": report_dict[tag]["f1-score"],
                }
                for tag in SPOILER_LABELS
                if tag in report_dict
            },
            "task2_by_type": {
                tag: {
                    "mean_bleu": float(df[df["true_tag"] == tag]["bleu"].mean()),
                    "mean_rouge": float(df[df["true_tag"] == tag]["rouge_l"].mean()),
                    "mean_meteor": float(df[df["true_tag"] == tag]["meteor"].mean()),
                    "mean_bert_score": float(
                        df[df["true_tag"] == tag]["bert_score"].mean()
                    ),
                }
                for tag in SPOILER_LABELS
            },
        }

        # Save structured machine-readable JSON
        with open(
            os.path.join(self.report_dir, "metrics_summary.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(summary_metrics, f, indent=4)

        # Export delta-comparison report against Phase 0 baseline (Phase 5.1)
        baseline_metrics = {
            "task1_accuracy": 0.73625,
            "task1_macro_f1": 0.73082,
            "task2_mean_bleu": 0.16818,
            "task2_mean_rouge_l": 0.43362,
        }

        comparison = {
            "metrics": {
                "task1_accuracy": {
                    "baseline": baseline_metrics["task1_accuracy"],
                    "current": acc,
                    "delta": acc - baseline_metrics["task1_accuracy"],
                },
                "task1_macro_f1": {
                    "baseline": baseline_metrics["task1_macro_f1"],
                    "current": macro_f1,
                    "delta": macro_f1 - baseline_metrics["task1_macro_f1"],
                },
                "task2_mean_bleu": {
                    "baseline": baseline_metrics["task2_mean_bleu"],
                    "current": mean_bleu,
                    "delta": mean_bleu - baseline_metrics["task2_mean_bleu"],
                },
                "task2_mean_rouge_l": {
                    "baseline": baseline_metrics["task2_mean_rouge_l"],
                    "current": mean_rouge,
                    "delta": mean_rouge - baseline_metrics["task2_mean_rouge_l"],
                },
            }
        }

        with open(
            os.path.join(self.report_dir, "comparison_to_baseline.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(comparison, f, indent=4)
        logger.info(f"Comparison report exported successfully to: {self.report_dir}")

        # Get 5 worst performing predictions for error analysis
        worst_df = df.sort_values(by="bleu").head(5)
        worst_cases = []
        for _, row in worst_df.iterrows():
            worst_cases.append(
                {
                    "uuid": row["uuid"],
                    "true_tag": row["true_tag"],
                    "gt": " | ".join(row["spoiler_true"])
                    if isinstance(row["spoiler_true"], list)
                    else str(row["spoiler_true"]),
                    "pred": " | ".join(row["spoiler_pred"])
                    if isinstance(row["spoiler_pred"], list)
                    else str(row["spoiler_pred"]),
                    "bleu": float(row["bleu"]),
                }
            )

        # 3. Generate visual HTML report
        self._generate_html_report(summary_metrics, worst_cases)
        logger.info(
            f"Visual metrics report compiled successfully at: {self.report_dir}"
        )
