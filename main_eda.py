"""Execution runner for performing Exploratory Data Analysis (EDA) on the Clickbait Spoiling dataset.

Loads training data, runs sanity checks, and processes textual statistics such as sequence lengths,
lexical overlaps, and positional biases. All results and charts are exported to reports.
"""

import os
import sys

from src.data_loader import JSONLLoader
from src.eda_analysis import TextLengthAnalyzer


def main() -> None:
    """Orchestrates the EDA process: data ingestion, sanity checks, length analysis,

    plotting distributions, class balance calculation, and positional bias evaluation.
    """
    train_path = os.path.join("data", "train.jsonl")

    try:
        loader = JSONLLoader(file_path=train_path)
        train_df = loader.load_data()
        loader.run_sanity_checks()
    except FileNotFoundError as e:
        print(f"[ERROR] Required dataset not found: {e}")
        print(
            "Please ensure you have placed 'train.jsonl' inside the 'data/' directory."
        )
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] EDA processing aborted due to ingestion failure: {e}")
        sys.exit(1)

    # Launch advanced Exploratory Data Analysis
    analyzer = TextLengthAnalyzer(df=train_df, output_dir="reports")
    analyzer.process_lengths()
    analyzer.display_statistics()
    analyzer.plot_distributions()
    analyzer.analyze_class_balance()
    analyzer.analyze_spoiler_overlap()
    analyzer.analyze_positional_bias()
    analyzer.analyze_tag_specific_lengths()
    analyzer.analyze_oracle_retrieval_bound()


if __name__ == "__main__":
    main()
