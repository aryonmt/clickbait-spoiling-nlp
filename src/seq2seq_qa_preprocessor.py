from transformers import AutoTokenizer

from src.utils import join_text_list


class Seq2SeqQAPreprocessor:
    """Preprocesses clickbait spoiling datasets for Seq2Seq T5 model training."""

    def __init__(
        self,
        model_name: str,
        max_source_length: int = 512,
        max_target_length: int = 128,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def prepare_features(self, examples):
        inputs = []
        for q, paras in zip(examples["postText"], examples["targetParagraphs"]):
            post_str = join_text_list(q)
            context_str = join_text_list(paras)
            inputs.append(f"question: {post_str} context: {context_str}")

        # Link target multi spoilers using a fixed delimiter " | "
        targets = []
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
        labels_ids = []
        for label in labels["input_ids"]:
            labels_ids.append(
                [
                    (token if token != self.tokenizer.pad_token_id else -100)
                    for token in label
                ]
            )

        model_inputs["labels"] = labels_ids
        return model_inputs
