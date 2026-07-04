import json
import os
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer

from src.logging_setup import setup_logger
from src.utils import extract_primary_tag, join_text_list


def find_answer_span(context: str, answer: str) -> Optional[Tuple[int, int]]:
    """Locates the start and end character indices of the answer inside context

    using strict normalization (whitespace and quotation alignment) while
    preserving raw character offsets.

    Args:
        context (str): The raw context string from target paragraphs.
        answer (str): The raw answer text to find.

    Returns:
        Optional[Tuple[int, int]]: A tuple containing (start_char, end_char)
            if matched successfully, else None.
    """
    if not answer or not context:
        return None

    # Try 1: Fast exact match
    start_char = context.find(answer)
    if start_char != -1:
        return start_char, start_char + len(answer)

    # Try 2: Normalized strip matching
    clean_answer = answer.strip()
    if clean_answer:
        start_char = context.find(clean_answer)
        if start_char != -1:
            return start_char, start_char + len(clean_answer)

    # Try 3: Character-by-character mapping of collapsed spaces and standardized quotes
    def normalize_char(c: str) -> str:
        if c in '“”“"':
            return '"'
        if c in "‘’'":
            return "'"
        if c.isspace():
            return " "
        return c.lower()

    # Build index mapping from normalized representation to original raw string
    norm_context: List[str] = []
    orig_indices: List[int] = []
    prev_was_space = False

    for idx, char in enumerate(context):
        norm_c = normalize_char(char)
        if norm_c == " ":
            if not prev_was_space:
                norm_context.append(" ")
                orig_indices.append(idx)
                prev_was_space = True
        else:
            norm_context.append(norm_c)
            orig_indices.append(idx)
            prev_was_space = False

    norm_context_str = "".join(norm_context)

    # Build normalized answer search term
    norm_answer: List[str] = []
    prev_was_space = False
    for char in clean_answer:
        norm_c = normalize_char(char)
        if norm_c == " ":
            if not prev_was_space:
                norm_answer.append(" ")
                prev_was_space = True
        else:
            norm_answer.append(norm_c)
            prev_was_space = False

    norm_answer_str = "".join(norm_answer)

    norm_start = norm_context_str.find(norm_answer_str)
    if norm_start != -1:
        norm_end = norm_start + len(norm_answer_str) - 1
        # Map indices back to the exact positions inside raw context
        orig_start = orig_indices[norm_start]
        orig_end = orig_indices[norm_end] + 1
        return orig_start, orig_end

    return None


class QAPreprocessor:
    """Preprocesses Clickbait Spoiling datasets into standardized extractive

    Question-Answering token spans with active fallback monitoring.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer matching the target QA model.
        max_length (int): Maximum sequence token constraint.
        doc_stride (int): Sliding window stride steps for long articles.
        logger (logging.Logger): Logger for tracking quality metrics.
        total_processed (int): Total context windows analyzed.
        fallback_count (int): Count of windows falling back to CLS index.
        tag_fallbacks (Dict[str, int]): Fallback counts per spoiler type.
        tag_totals (Dict[str, int]): Total counts per spoiler type.
    """

    def __init__(self, model_name: str, max_length: int = 512, doc_stride: int = 128):
        """Initializes the QAPreprocessor.

        Args:
            model_name (str): HuggingFace model identifier.
            max_length (int): Maximum sequence length constraint. Defaults to 512.
            doc_stride (int): Sliding window stride. Defaults to 128.
        """
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length: int = max_length
        self.doc_stride: int = doc_stride
        self.logger = setup_logger("qa_preprocessor")

        # Diagnostics metrics for span extraction analysis
        self.total_processed: int = 0
        self.fallback_count: int = 0
        self.tag_fallbacks: Dict[str, int] = {"phrase": 0, "passage": 0, "multi": 0}
        self.tag_totals: Dict[str, int] = {"phrase": 0, "passage": 0, "multi": 0}

    def prepare_train_features(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenizes inputs and maps character span answer targets back to token start/end positions.

        Note:
            If multiprocessing tokenization is used (`num_proc > 1` in `Dataset.map`),
            the stateful diagnostic metrics (like self.fallback_count) will not aggregate
            correctly across sub-processes. Keep `num_proc=1` if exact counts are required.

        Args:
            examples (Dict[str, Any]): Batched training examples from datasets.Dataset.

        Returns:
            Dict[str, Any]: Tokenized examples containing 'input_ids', 'attention_mask',
                'start_positions', and 'end_positions'.
        """
        # Flatten posts and target paragraphs into single strings
        questions = [join_text_list(q) for q in examples["postText"]]

        contexts = [join_text_list(paras) for paras in examples["targetParagraphs"]]

        # Tokenize with truncation and padding, keeping overflows via sliding window
        tokenized_examples = self.tokenizer(
            questions,
            contexts,
            truncation="only_second",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples["offset_mapping"]

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        # Safe extraction of diagnostic fields to handle missing test-time columns
        uuids = examples.get("uuid", ["unknown"] * len(questions))
        raw_tags = examples.get("tags", [["phrase"]] * len(questions))

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Get the original sample index
            sample_index = sample_mapping[i]
            context = contexts[sample_index]

            # Reconstruct target span using multi-segment search strategy
            spoilers = (
                examples["spoiler"][sample_index] if "spoiler" in examples else []
            )

            # Extract labels for type-specific monitoring
            tag = extract_primary_tag(raw_tags[sample_index])
            self.tag_totals[tag] = self.tag_totals.get(tag, 0) + 1
            self.total_processed += 1

            # Try finding a valid span from any of the spoiler segments
            span = None
            for answer_text in spoilers:
                span = find_answer_span(context, answer_text)
                if span is not None:
                    break

            if span is None:
                # Anchor to CLS token (unanswerable fallback)
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)

                self.fallback_count += 1
                self.tag_fallbacks[tag] = self.tag_fallbacks.get(tag, 0) + 1

                self.logger.debug(
                    f"Spoiler text not found in context window for uuid: {uuids[sample_index]}. "
                    f"Falling back to CLS index. Tag: {tag}"
                )
            else:
                start_char, end_char = span

                # Fetch sequence IDs to identify context tokens safely
                sequence_ids = [
                    1 if val is not None and val > 0 else 0
                    for val in tokenized_examples.sequence_ids(i)
                ]

                # Find token span limits
                token_start_index = 0
                while (
                    token_start_index < len(offsets)
                    and sequence_ids[token_start_index] != 1
                ):
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while token_end_index >= 0 and sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer window falls completely inside this token slice.
                # Defensive check: if no context tokens exist in sequence_ids, token_start_index
                # will equal len(offsets) and token_end_index will equal -1, which would
                # raise an IndexError when accessing offsets.
                if (
                    token_start_index >= len(offsets)
                    or token_end_index < 0
                    or not (
                        offsets[token_start_index][0] <= start_char
                        and offsets[token_end_index][1] >= end_char
                    )
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)

                    self.fallback_count += 1
                    self.tag_fallbacks[tag] = self.tag_fallbacks.get(tag, 0) + 1

                    self.logger.debug(
                        f"Answer span outside active token window for uuid: {uuids[sample_index]}. "
                        f"Falling back to CLS index. Tag: {tag}"
                    )
                else:
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)

                    while (
                        token_end_index >= 0 and offsets[token_end_index][1] >= end_char
                    ):
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        # Print structured statistical evaluation inside logs
        if self.total_processed > 0:
            phrase_rate = (
                self.tag_fallbacks["phrase"] / max(1, self.tag_totals.get("phrase", 0))
            ) * 100
            passage_rate = (
                self.tag_fallbacks["passage"]
                / max(1, self.tag_totals.get("passage", 0))
            ) * 100
            multi_rate = (
                self.tag_fallbacks["multi"] / max(1, self.tag_totals.get("multi", 0))
            ) * 100

            self.logger.info(
                f"Data preprocessing chunk complete. Statistics:\n"
                f"  - Total processed: {self.total_processed}\n"
                f"  - Overall CLS Fallbacks: {self.fallback_count} ({self.fallback_count / self.total_processed * 100:.2f}%)\n"
                f"  - Phrase fallbacks: {self.tag_fallbacks['phrase']}/{self.tag_totals.get('phrase', 0)} ({phrase_rate:.2f}%)\n"
                f"  - Passage fallbacks: {self.tag_fallbacks['passage']}/{self.tag_totals.get('passage', 0)} ({passage_rate:.2f}%)\n"
                f"  - Multi fallbacks: {self.tag_fallbacks['multi']}/{self.tag_totals.get('multi', 0)} ({multi_rate:.2f}%)"
            )

        return tokenized_examples

    def export_stats(self, path: str) -> None:
        """Exports preprocessing diagnostic fallback metrics to a JSON file.

        Args:
            path (str): Target directory or file path for JSON export.
        """
        # Robustly handle directory creation even if the folder does not exist yet
        if not path.endswith(".json"):
            os.makedirs(path, exist_ok=True)
            export_file = os.path.join(path, "label_quality_stats.json")
        else:
            parent_dir = os.path.dirname(path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            export_file = path

        stats = {
            "total_processed": self.total_processed,
            "fallback_count": self.fallback_count,
            "fallback_rate": self.fallback_count / max(1, self.total_processed),
            "tag_totals": self.tag_totals,
            "tag_fallbacks": self.tag_fallbacks,
            "tag_fallback_rates": {
                tag: self.tag_fallbacks[tag] / max(1, self.tag_totals[tag])
                for tag in self.tag_totals
            },
        }

        with open(export_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)
        self.logger.info(
            f"Successfully exported preprocessing quality stats to: {export_file}"
        )
