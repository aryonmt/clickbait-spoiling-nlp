import os
import sys

import pandas as pd

# تنظیم مسیر برای ایمپورت متمرکز
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import extract_primary_tag, join_text_list


def analyze_optimal_thresholds(file_path: str):
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return

    print(f"Analyzing dataset from: {file_path}...")
    df = pd.read_json(file_path, lines=True)
    df["clean_tag"] = df["tags"].apply(extract_primary_tag)

    # ۱. تحلیل طول کلمات اسپویلرهای واقعی به تفکیک تگ
    df["spoiler_word_count"] = df["spoiler"].apply(
        lambda x: len(join_text_list(x).split()) if isinstance(x, (list, str)) else 0
    )

    print("\n=======================================================")
    print("SPOILER WORD COUNT DISTRIBUTION BY TAG (Percentiles)")
    print("=======================================================")
    grouped = df.groupby("clean_tag")["spoiler_word_count"].describe(
        percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]
    )
    print(grouped)

    # ۲. تحلیل توزیع تعداد آیتم‌ها (لیست) در اسپویلرهای چندگانه (Multi)
    multi_df = df[df["clean_tag"] == "multi"]
    if not multi_df.empty:
        multi_df["item_count"] = multi_df["spoiler"].apply(
            lambda x: len(x) if isinstance(x, list) else 1
        )
        print("\n=======================================================")
        print("MULTI-SPOILER ITEM COUNT DISTRIBUTION (How many items per list?)")
        print("=======================================================")
        print(multi_df["item_count"].describe(percentiles=[0.5, 0.75, 0.9, 0.95]))
    else:
        print("\n[INFO] No 'multi' spoilers found in this dataset split.")


if __name__ == "__main__":
    # فایل train.jsonl را تحلیل می‌کنیم
    analyze_optimal_thresholds("train.jsonl")
