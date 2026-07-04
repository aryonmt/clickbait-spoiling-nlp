import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils import extract_primary_tag, join_text_list


class TextLengthAnalyzer:
    """Analyzer to calculate sequence lengths, lexical overlaps, positional biases,

    and statistical percentiles for text features in the dataset.

    Attributes:
        df (pd.DataFrame): Copy of the target dataset with analyzed metrics.
        output_dir (str): Directory path where generated plots are saved.
    """

    def __init__(self, df: pd.DataFrame, output_dir: str = "reports"):
        """Initializes the analyzer with a dataset and an output directory.

        Args:
            df (pd.DataFrame): The target dataset.
            output_dir (str): Directory where generated EDA plots will be saved.
        """
        self.df: pd.DataFrame = df.copy()
        self.output_dir: str = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _count_words(text_input) -> int:
        """Helper function to calculate total word count utilizing central utilities.

        Args:
            text_input (Union[str, List[str]]): String or list of strings to analyze.

        Returns:
            int: Total word count.
        """
        full_text = join_text_list(text_input)
        return len(full_text.split())

    def process_lengths(self) -> pd.DataFrame:
        """Computes word counts for clickbait posts and article bodies.

        Returns:
            pd.DataFrame: DataFrame containing computed sequence lengths.

        Raises:
            KeyError: If mandatory text columns are missing from the dataset.
        """
        required_cols = ["postText", "targetParagraphs"]
        for col in required_cols:
            if col not in self.df.columns:
                raise KeyError(
                    f"[ERROR] Mandatoy column '{col}' is missing from the dataset."
                )

        print("\nCalculating word counts for sequences...")
        self.df["post_word_count"] = self.df["postText"].apply(self._count_words)
        self.df["article_word_count"] = self.df["targetParagraphs"].apply(
            self._count_words
        )
        return self.df

    def display_statistics(self) -> None:
        """Computes and prints percentile statistics for sequence lengths to guide max_length choices."""
        print("\n=== Sequence Length Statistical Summary ===")
        columns_to_analyze = ["post_word_count", "article_word_count"]

        # Ensure processed columns exist
        if "post_word_count" not in self.df.columns:
            self.process_lengths()

        stats = self.df[columns_to_analyze].describe(
            percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]
        )
        print(stats)

    def plot_distributions(self) -> None:
        """Generates and saves distribution plots of sequence word counts."""
        if "post_word_count" not in self.df.columns:
            self.process_lengths()

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(self.df["post_word_count"], kde=True, color="skyblue")
        plt.title("Distribution of Clickbait Post Word Counts")
        plt.xlabel("Word Count")
        plt.ylabel("Frequency")

        plt.subplot(1, 2, 2)
        sns.histplot(self.df["article_word_count"], kde=True, color="salmon")
        plt.title("Distribution of Article Word Counts")
        plt.xlabel("Word Count")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "sequence_length_distributions.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\n[SUCCESS] Distribution plots saved to: {plot_path}")
        plt.close()

    def analyze_class_balance(self) -> pd.Series:
        """Analyzes and prints the distribution of spoiler classes (phrase/passage/multi).

        Returns:
            pd.Series: Distribution series of extract primary tags.
        """
        print("\n=== Class Distribution Analysis (Task 1) ===")
        class_series = self.df["tags"].apply(extract_primary_tag)

        counts = class_series.value_counts()
        percentages = class_series.value_counts(normalize=True) * 100

        distribution_df = pd.DataFrame({"Count": counts, "Percentage (%)": percentages})
        print(distribution_df)
        return class_series

    def analyze_spoiler_overlap(self) -> None:
        """Calculates Jaccard Similarity between target spoilers and complete article bodies

        to verify extractiveness rate.
        """
        print("\n=== Spoiler Lexical Overlap Analysis ===")

        overlaps: List[float] = []
        for _, row in self.df.iterrows():
            paragraphs = join_text_list(row.get("targetParagraphs", ""))
            spoilers = join_text_list(row.get("spoiler", ""))

            words_article = set(paragraphs.lower().split())
            words_spoiler = set(spoilers.lower().split())

            if not words_spoiler:
                overlaps.append(0.0)
                continue

            intersection = words_spoiler.intersection(words_article)
            overlap_ratio = len(intersection) / len(words_spoiler)
            overlaps.append(overlap_ratio)

        self.df["spoiler_overlap_ratio"] = overlaps
        print(self.df["spoiler_overlap_ratio"].describe())

        fully_extractive = (self.df["spoiler_overlap_ratio"] == 1.0).sum()
        print(
            f"\nFully Extractive Spoilers (100% word overlap):"
            f" {fully_extractive} out of {len(self.df)} "
            f"({fully_extractive / len(self.df) * 100:.2f}%)"
        )

    def analyze_positional_bias(self) -> None:
        """Analyzes paragraph indices of the first segment of target spoilers to find positional bias."""
        print("\n=== Spoiler Positional Bias Analysis ===")

        spoiler_paragraphs: List[int] = []
        for _, row in self.df.iterrows():
            positions = row.get("spoilerPositions", [])
            if isinstance(positions, list) and len(positions) > 0:
                try:
                    first_segment = positions[0]
                    start_position = first_segment[0]
                    para_index = start_position[0]
                    spoiler_paragraphs.append(para_index)
                except (IndexError, TypeError):
                    continue

        if not spoiler_paragraphs:
            print("[WARNING] No valid spoiler positions found to analyze.")
            return

        pos_series = pd.Series(spoiler_paragraphs)
        print("Paragraph Index Distribution for Spoilers:")
        print(pos_series.describe(percentiles=[0.5, 0.75, 0.9, 0.95]))

        plt.figure(figsize=(8, 5))
        sns.histplot(pos_series, bins=20, kde=False, color="purple")
        plt.title("Distribution of Spoiler Locations (Paragraph Index)")
        plt.xlabel("Paragraph Index")
        plt.ylabel("Count")
        plt.tight_layout()

        plot_path = os.path.join(self.output_dir, "spoiler_positional_bias.png")
        plt.savefig(plot_path, dpi=150)
        print(f"[SUCCESS] Positional bias plot saved to: {plot_path}")
        plt.close()

    def analyze_tag_specific_lengths(self) -> None:
        """Computes and plots statistical spoiler length summaries grouped by their tag."""
        print("\n=== Tag-Specific Spoiler Length Analysis ===")

        self.df["spoiler_word_count"] = self.df["spoiler"].apply(self._count_words)
        self.df["clean_tag"] = self.df["tags"].apply(extract_primary_tag)

        grouped_stats = self.df.groupby("clean_tag")["spoiler_word_count"].describe(
            percentiles=[0.5, 0.75, 0.9, 0.95]
        )
        print(grouped_stats)

        plt.figure(figsize=(8, 5))
        sns.boxplot(
            x="clean_tag",
            y="spoiler_word_count",
            data=self.df,
            palette="Set2",
            hue="clean_tag",
            legend=False,
        )
        plt.title("Spoiler Word Count Distribution per Class")
        plt.xlabel("Spoiler Type (Tag)")
        plt.ylabel("Word Count")
        plt.yscale("log")
        plt.tight_layout()

        plot_path = os.path.join(self.output_dir, "spoiler_length_by_tag.png")
        plt.savefig(plot_path, dpi=150)
        print(f"[SUCCESS] Tag-specific length plot saved to: {plot_path}")
        plt.close()

    def analyze_oracle_retrieval_bound(self) -> None:
        """Evaluates Jaccard Similarity between clickbait posts and article paragraphs,

        measuring Recall@1, Recall@3, and Recall@5 for retrieving ground-truth spoiler paragraphs.
        """
        print("\n=== Oracle Retrieval Upper-Bound Analysis (IR Metrics) ===")

        recall_at_1 = 0
        recall_at_3 = 0
        recall_at_5 = 0
        total_valid_cases = 0

        for _, row in self.df.iterrows():
            post_words = set(join_text_list(row.get("postText", "")).lower().split())
            paragraphs = row.get("targetParagraphs", [])
            spoilers = join_text_list(row.get("spoiler", ""))

            if not paragraphs or not spoilers:
                continue

            total_valid_cases += 1

            gt_indices = [
                i for i, p in enumerate(paragraphs) if spoilers.lower() in p.lower()
            ]
            if not gt_indices:
                continue
            gt_index = gt_indices[0]

            scored_paragraphs = []
            for idx, para in enumerate(paragraphs):
                para_words = set(str(para).lower().split())
                intersection = post_words.intersection(para_words)
                union = post_words.union(para_words)
                jaccard_score = len(intersection) / len(union) if union else 0.0
                scored_paragraphs.append((idx, jaccard_score))

            scored_paragraphs.sort(key=lambda x: x[1], reverse=True)
            ranked_indices = [item[0] for item in scored_paragraphs]

            if gt_index in ranked_indices[:1]:
                recall_at_1 += 1
            if gt_index in ranked_indices[:3]:
                recall_at_3 += 1
            if gt_index in ranked_indices[:5]:
                recall_at_5 += 1

        if total_valid_cases == 0:
            print("[WARNING] No valid cases for Oracle Analysis.")
            return

        print(f"Total Evaluated Cases: {total_valid_cases}")
        print(f"Oracle Recall@1: {recall_at_1 / total_valid_cases * 100:.2f}%")
        print(f"Oracle Recall@3: {recall_at_3 / total_valid_cases * 100:.2f}%")
        print(f"Oracle Recall@5: {recall_at_5 / total_valid_cases * 100:.2f}%")
