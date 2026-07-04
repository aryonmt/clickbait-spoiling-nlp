import argparse
import os
import sys

# Ensure the project root (parent of this scripts/ folder) is importable,
# so this file can be run directly (e.g. `python scripts/measure_qa_label_quality.py`)
# as well as via `python -m scripts.measure_qa_label_quality`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import JSONLLoader
from src.qa_preprocessor import find_answer_span
from src.utils import extract_primary_tag


def measure_quality(file_path: str):
    """Measures the matching rate of spoilers in context using baseline and normalized matching."""
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return

    loader = JSONLLoader(file_path)
    df = loader.load_data()

    total = len(df)
    exact_match = 0
    norm_match = 0

    type_stats = {
        "phrase": {"total": 0, "exact": 0, "norm": 0},
        "passage": {"total": 0, "exact": 0, "norm": 0},
        "multi": {"total": 0, "exact": 0, "norm": 0},
    }

    for _, row in df.iterrows():
        tag = extract_primary_tag(row["tags"])
        paragraphs = row["targetParagraphs"]
        context = (
            " ".join(paragraphs) if isinstance(paragraphs, list) else str(paragraphs)
        )
        spoilers = row["spoiler"]

        # Check if we have spoilers
        if not isinstance(spoilers, list) or len(spoilers) == 0:
            continue

        type_stats[tag]["total"] += 1

        # Exact match (baseline: checks only spoilers[0] exactly)
        primary_spoiler = spoilers[0]
        is_exact = context.find(primary_spoiler) != -1
        if is_exact:
            exact_match += 1
            type_stats[tag]["exact"] += 1

        # Improved match (normalized, checks all segments)
        is_norm = False
        for segment in spoilers:
            if find_answer_span(context, segment) is not None:
                is_norm = True
                break

        if is_norm:
            norm_match += 1
            type_stats[tag]["norm"] += 1

    print("\n==========================================")
    print(f"QUALITY REPORT FOR: {file_path}")
    print("==========================================")
    print(f"Total Rows Evaluated: {total}")
    print(
        f"Baseline Match Rate (Exact on spoilers[0]): {exact_match}/{total} ({exact_match / total * 100:.2f}%)"
    )
    print(
        f"Improved Match Rate (Normalized on all):    {norm_match}/{total} ({norm_match / total * 100:.2f}%)"
    )
    print(
        f"Fallback Reduction: {exact_match} matches -> {norm_match} matches (+{norm_match - exact_match} recovered, {(norm_match - exact_match) / total * 100:.2f}% improvement)"
    )

    print("\nBreakdown by Spoiler Type:")
    for tag, stats in type_stats.items():
        t_total = stats["total"] if stats["total"] > 0 else 1
        print(f"  [{tag.upper()}]:")
        print(f"    - Total cases: {stats['total']}")
        print(
            f"    - Baseline match: {stats['exact']}/{stats['total']} ({stats['exact'] / t_total * 100:.2f}%)"
        )
        print(
            f"    - Improved match: {stats['norm']}/{stats['total']} ({stats['norm'] / t_total * 100:.2f}%)"
        )
        print(
            f"    - Fallback rate reduced from {100 - (stats['exact'] / t_total * 100):.2f}% to {100 - (stats['norm'] / t_total * 100):.2f}%"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure QA Label Quality and Fallback Rates"
    )
    parser.add_argument(
        "--file_path", type=str, required=True, help="Path to JSONL dataset"
    )
    args = parser.parse_args()
    measure_quality(args.file_path)
