"""CLI execution runner for generating visual evaluation reports and HTML dashboards.

Aligns prediction outputs with ground-truth validation data to compute performance summaries
and save distribution plots to the specified output folder.
"""

import argparse
import os
import sys

from src.visualization_suite import EvaluationReportBuilder


def main() -> None:
    """Parses command-line arguments and orchestrates visual report generation."""
    parser = argparse.ArgumentParser(description="Generate Visual Evaluation Report")
    parser.add_argument(
        "--predictions_path",
        type=str,
        required=True,
        help="Path to predictions JSONL file",
    )
    parser.add_argument(
        "--val_path", type=str, required=True, help="Path to validation JSONL file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="reports", help="Root directory for reports"
    )
    args = parser.parse_args()

    # Defensive validation of target file existence before executing matplotlib routines
    if not os.path.exists(args.predictions_path):
        print(f"[ERROR] Predictions file not found: {args.predictions_path}")
        sys.exit(1)
    if not os.path.exists(args.val_path):
        print(f"[ERROR] Validation file not found: {args.val_path}")
        sys.exit(1)

    # Orchestrate visual reports generation
    builder = EvaluationReportBuilder(
        predictions_path=args.predictions_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
    )
    builder.build_report()


if __name__ == "__main__":
    main()
