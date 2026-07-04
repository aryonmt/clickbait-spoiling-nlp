from typing import List

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

from src.utils import join_text_list


class TransformerInferencePipeline:
    """Inference pipeline to load trained Transformer checkpoints and predict

    clickbait spoiler types (phrase, passage, multi).

    Attributes:
        checkpoint_dir (str): Path to the directory containing model weights and tokenizer configs.
        max_length (int): Maximum sequence token constraint for the model.
        tokenizer (AutoTokenizer): Tokenizer loaded from the specified checkpoint.
        model (AutoModelForSequenceClassification): Loaded model weights for sequence classification.
        id_to_label (dict): Internal mapping from integer IDs back to string spoiler classes.
        trainer (Trainer): HuggingFace Trainer instance configured for pure batch inference.
    """

    def __init__(self, checkpoint_dir: str, max_length: int = 512):
        """Initializes the inference pipeline with trained checkpoints.

        Args:
            checkpoint_dir (str): Directory containing model checkpoint weights.
            max_length (int): Tokenization truncation length. Defaults to 512.
        """
        self.checkpoint_dir: str = checkpoint_dir
        self.max_length: int = max_length
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.model: AutoModelForSequenceClassification = (
            AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        )
        self.id_to_label: dict = {0: "phrase", 1: "passage", 2: "multi"}
        self.trainer: Trainer = Trainer(
            model=self.model, processing_class=self.tokenizer
        )

    def predict(self, df: pd.DataFrame) -> List[str]:
        """Generates category string predictions for the given clickbait dataset.

        Args:
            df (pd.DataFrame): Input dataset containing 'postText' and 'targetParagraphs'.

        Returns:
            List[str]: List of predicted spoiler type labels ('phrase', 'passage', 'multi').
        """
        if df.empty:
            return []

        processed_df = pd.DataFrame()
        processed_df["post"] = (
            df["postText"].apply(join_text_list) if "postText" in df.columns else ""
        )

        # Safely join top-8 paragraphs defensively checking for NaN/None/List entries
        processed_df["article"] = (
            df["targetParagraphs"].apply(
                lambda x: (
                    join_text_list(x, limit=8) if isinstance(x, (list, str)) else ""
                )
            )
            if "targetParagraphs" in df.columns
            else ""
        )

        # Dummy label mapping to satisfy HF internal Dataset requirements
        processed_df["label"] = 0

        # Reset indexes to prevent PyArrow from creating polluted '__index_level_0__' metadata columns
        processed_df = processed_df.reset_index(drop=True)

        dataset = Dataset.from_pandas(processed_df)

        def tokenize_func(examples):
            return self.tokenizer(
                examples["post"],
                examples["article"],
                truncation=True,
                max_length=self.max_length,
            )

        tokenized_dataset = dataset.map(tokenize_func, batched=True)

        print("Running batch transformer inference...")
        predictions = self.trainer.predict(tokenized_dataset)
        predicted_ids = np.argmax(predictions.predictions, axis=-1)

        return [self.id_to_label[idx] for idx in predicted_ids]
