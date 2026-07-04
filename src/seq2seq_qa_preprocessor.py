from typing import Any, Dict, List

from transformers import AutoTokenizer

from src.utils import join_text_list


class Seq2SeqQAPreprocessor:
    """Preprocesses clickbait spoiling datasets for Seq2Seq (e.g., T5) model training.

    Handles concatenation of post and context text, formatting of target multi-spoilers,
    and target padding token masking for standard PyTorch loss functions.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer loaded from the specified checkpoint.
        max_source_length (int): Maximum input sequence token limit.
        max_target_length (int): Maximum output target sequence token limit.
    """

    def __init__(
        self,
        model_name: str,
        max_source_length: int = 512,
        max_target_length: int = 128,
    ):
        """Initializes the Seq2SeqQAPreprocessor with model and length constraints.

        Args:
            model_name (str): HuggingFace model architecture name or path.
            max_source_length (int): Maximum input sequence length constraint. Defaults to 512.
            max_target_length (int): Maximum target sequence length constraint. Defaults to 128.
        """
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_source_length: int = max_source_length
        self.max_target_length: int = max_target_length

    def prepare_features(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Processes and tokenizes raw clickbait post and context strings into Seq2Seq model inputs.

        Combines postText and targetParagraphs into a structured prompt format and links
        target multi-spoilers using a fixed ' | ' delimiter.

        Args:
            examples (Dict[str, Any]): Batched training examples from datasets.Dataset.

        Returns:
            Dict[str, Any]: Preprocessed features containing tokenized 'input_ids',
                'attention_mask', and padded 'labels'.
        """
        inputs: List[str] = []
        for q, paras in zip(examples["postText"], examples["targetParagraphs"]):
            post_str = join_text_list(q)
            context_str = join_text_list(paras)
            inputs.append(f"question: {post_str} context: {context_str}")

        # Link target multi spoilers using a fixed delimiter " | "
        targets: List[str] = []
        for spoilers in examples["spoiler"]:
            targets.append(" | ".join(spoilers))

        # Tokenize source inputs
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
        )

        # Tokenize targets
        labels = self.tokenizer(
            text_target=targets,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
        )

        # Map pad tokens to -100 to avoid loss scaling penalties
        labels_ids: List[List[int]] = []
        for label in labels["input_ids"]:
            labels_ids.append(
                [
                    (token if token != self.tokenizer.pad_token_id else -100)
                    for token in label
                ]
            )

        model_inputs["labels"] = labels_ids
        return model_inputs
