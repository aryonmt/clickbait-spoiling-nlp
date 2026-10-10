import glob
import os
import shutil
from typing import Any, Dict, List

import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.config import PipelineConfig
from src.utils import extract_primary_tag, join_text_list

# Enforce global reproducibility across all initializations
set_seed(42)


class ClickbaitWeightedTrainer(Trainer):
    """Custom Trainer overriding compute_loss to handle class imbalance via Weighted Cross-Entropy.

    Ensures robust deployment by dynamically mapping the class weights to the exact device
    and float precision of the logits during the forward pass.
    """

    def __init__(self, class_weights: List[float], *args, **kwargs):
        """Initializes the ClickbaitWeightedTrainer.

        Args:
            class_weights (List[float]): Inverse frequency weights calculated from the dataset.
        """
        super().__init__(*args, **kwargs)
        self.class_weights: torch.Tensor = torch.tensor(
            class_weights, dtype=torch.float32
        ).to(self.args.device)

    def compute_loss(
        self,
        model: Any,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        """Calculates cross-entropy loss applying the dynamically aligned class weights.

        Args:
            model: The sequence classification transformer model.
            inputs (Dict[str, torch.Tensor]): Model training inputs including labels.
            return_outputs (bool): If True, returns loss and prediction outputs. Defaults to False.
            num_items_in_batch (int, optional): Batch size normalization argument.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Any]]: Loss tensor, or (loss, outputs) tuple.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Dynamically cast class weights to match both logits dtype (Float vs Half precision)
        # AND computing device to prevent PyTorch device mismatch exceptions on multi-GPU setups.
        weights = self.class_weights.to(device=logits.device, dtype=logits.dtype)

        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


class TransformerSpoilerClassifier:
    """A fully modular, encapsulated pipeline for training and evaluating Transformer-based

    sequence classifiers for Task 1 (Spoiler Type Classification).

    Attributes:
        config (PipelineConfig): Structured system configurations.
        model_name (str): Transformer model architecture checkpoint.
        max_length (int): Maximum sequence token constraint.
        lr (float): Classifier fine-tuning learning rate.
        tokenizer (AutoTokenizer): Initialized tokenizer for text parsing.
        label_to_id (dict): String labels to integer ID mappings.
        id_to_label (dict): Integer IDs to string labels mappings.
    """

    def __init__(self, config: PipelineConfig):
        """Initializes the TransformerSpoilerClassifier.

        Args:
            config (PipelineConfig): Global pipeline configuration container.
        """
        self.config: PipelineConfig = config
        self.model_name: str = config.task1_model_name
        self.max_length: int = config.task1_max_length
        self.lr: float = config.task1_lr
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.label_to_id: Dict[str, int] = {"phrase": 0, "passage": 1, "multi": 2}
        self.id_to_label: Dict[int, str] = {0: "phrase", 1: "passage", 2: "multi"}

    def _prepare_dataset(self, df: pd.DataFrame) -> Dataset:
        """Processes dataframe fields into tokenized text-pairs using standard utilities.

        Args:
            df (pd.DataFrame): Ingestion dataset containing 'postText' and 'targetParagraphs'.

        Returns:
            Dataset: An aligned HuggingFace Dataset ready for sequence model training.
        """
        processed_df = pd.DataFrame()
        processed_df["post"] = (
            df["postText"].apply(join_text_list) if "postText" in df.columns else ""
        )
        processed_df["article"] = (
            df["targetParagraphs"].apply(
                lambda x: (
                    join_text_list(x, limit=8) if isinstance(x, (list, str)) else ""
                )
            )
            if "targetParagraphs" in df.columns
            else ""
        )
        processed_df["label"] = (
            df["tags"].apply(extract_primary_tag).map(self.label_to_id)
        )

        # Drop non-contiguous index labels to prevent PyArrow serialization metadata pollution
        processed_df = processed_df.reset_index(drop=True)

        dataset = Dataset.from_pandas(processed_df)

        def tokenize_func(examples: Dict[str, List[str]]) -> Dict[str, Any]:
            """Tokenize post/article pairs.

            Args:
                examples: Batch with ``post`` and ``article`` string lists.

            Returns:
                Dict with ``input_ids`` and ``attention_mask``. Dynamic padding
                is left to ``DataCollatorWithPadding``.
            """
            return self.tokenizer(
                examples["post"],
                examples["article"],
                truncation=True,
                max_length=self.max_length,
            )

        return dataset.map(tokenize_func, batched=True)

    def _calculate_class_weights(self, df: pd.DataFrame) -> List[float]:
        """Computes inverse frequency class weights to address training dataset imbalance.

        Features defensive checks to safely handle empty categories during testing.

        Args:
            df (pd.DataFrame): Training DataFrame with a 'tags' column.

        Returns:
            List[float]: Normalized inverse frequency weights for each class.
        """
        labels = df["tags"].apply(extract_primary_tag).map(self.label_to_id).values
        class_counts = np.bincount(labels, minlength=3)
        total_samples = len(labels)

        weights: List[float] = []
        for count in class_counts:
            if count > 0:
                weights.append(total_samples / (3.0 * count))
            else:
                weights.append(1.0)  # Safe default weight for zero-count categories
        return weights

    def train_pipeline(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Executes full tokenization, model initialization, and weighted training with checkpoint self-cleaning.

        Args:
            train_df (pd.DataFrame): Training dataset.
            val_df (pd.DataFrame): Validation evaluation dataset.

        Returns:
            Dict[str, float]: Evaluation convergence metrics resulting from trainer.evaluate().
        """
        output_dir = self.config.output_dir
        print(f"Mapping and processing partitions for: {self.model_name}")
        train_dataset = self._prepare_dataset(train_df)
        val_dataset = self._prepare_dataset(val_df)

        class_weights = self._calculate_class_weights(train_df)
        print(f"[INFO] Computed Dynamic Class Weights: {class_weights}")

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,
            id2label=self.id_to_label,
            label2id=self.label_to_id,
            ignore_mismatched_sizes=self.config.task1_ignore_mismatched_sizes,
        )

        # Pre-load evaluate metric once globally to prevent redundant disk/api load overhead
        # during intermediate evaluation loops.
        f1_metric = evaluate.load("f1")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return f1_metric.compute(
                predictions=predictions, references=labels, average="macro"
            )

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.lr,
            per_device_train_batch_size=self.config.task1_batch_size,
            per_device_eval_batch_size=self.config.task1_batch_size,
            num_train_epochs=self.config.task1_epochs,
            weight_decay=self.config.task1_weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=1,
            logging_steps=10,
            fp16=False,
            gradient_accumulation_steps=2,
            warmup_steps=self.config.task1_warmup_steps,
            adam_epsilon=1e-6,
            max_grad_norm=1.0,
            report_to="none",
        )

        trainer = ClickbaitWeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=compute_metrics,
        )

        print(f"\n=== LAUNCHING TRANSFORMER TRAIN SEQUENCE: {self.model_name} ===")
        trainer.train()

        print("\n=== RUNNING CONVERGENCE EVALUATION ===")
        eval_metrics = trainer.evaluate()
        print(eval_metrics)

        print(f"Saving optimized Task 1 classifier weights to {output_dir}...")
        trainer.save_model(output_dir)

        checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        for folder in checkpoint_dirs:
            try:
                shutil.rmtree(folder)
                print(f"Removed intermediate checkpoint folder: {folder}")
            except Exception as e:
                print(f"Error removing checkpoint folder {folder}: {e}")

        return eval_metrics
