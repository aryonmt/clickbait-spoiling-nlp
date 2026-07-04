from typing import List

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

from src.utils import join_text_list


class TransformerInferencePipeline:
    """Inference pipeline to load trained transformer checkpoints and predict

    spoiler types.
    """

    def __init__(self, checkpoint_dir: str, max_length: int = 512):
        self.checkpoint_dir = checkpoint_dir
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        self.id_to_label = {0: "phrase", 1: "passage", 2: "multi"}
        self.trainer = Trainer(model=self.model, processing_class=self.tokenizer)

    def predict(self, df: pd.DataFrame) -> List[str]:
        """Generates scalar string predictions for the given challenge dataframe."""
        processed_df = pd.DataFrame()
        processed_df["post"] = df["postText"].apply(join_text_list)
        processed_df["article"] = df["targetParagraphs"].apply(
            lambda x: join_text_list(x, limit=8) if isinstance(x, list) else ""
        )

        # Dummy label mapping to satisfy internal dataset formats during validation evaluation
        processed_df["label"] = 0

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
