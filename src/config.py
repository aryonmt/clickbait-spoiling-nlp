import argparse
import os
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Configuration storage for task classification and answer extraction pipelines.

    Attributes:
        seed (int): Random seed for ensuring reproducibility across models.
        output_dir (str): Root directory for saving general training outputs.
        log_dir (str): Target directory for saving system log files.
        task1_model_name (str): HuggingFace transformer model architecture for classification.
        task1_max_length (int): Maximum sequence token constraints for classification.
        task1_batch_size (int): Training batch size for sequence classification.
        task1_lr (float): Initial learning rate constraint for classification training.
        task1_epochs (int): Number of training epochs for Task 1 classifier.
        task1_weight_decay (float): Weight decay strength for Task 1 classification regularization.
        task1_ignore_mismatched_sizes (bool): Ignore shape mismatches when loading classification weights.
        task2_model_name (str): Base Question Answering model for extractive spoiling.
        task2_max_length (int): Maximum sequence window constraint for QA models.
        task2_doc_stride (int): Stride step for sliding-window tokenization of long articles.
        task2_batch_size (int): Batch size used for QA extraction training.
        task2_lr (float): Initial learning rate constraint for QA model training.
        task2_epochs (int): Default training epochs for task 2 extraction.
        task2_weight_decay (float): Weight decay strength for QA model training.
        task2_phrase_model_name (str): Fine-tuned QA checkpoint model specifically for phrase spoilers.
        task2_phrase_max_answer_length (int): Length boundary constraint for phrase answers.
        task2_phrase_output_dir (str): Checkpoint directory for saving phrase model weights.
        task2_phrase_epochs (int): Training epochs dedicated for phrase-type spoilers.
        task2_passage_model_name (str): Fine-tuned QA checkpoint model specifically for passage spoilers.
        task2_passage_max_answer_length (int): Length boundary constraint for passage answers.
        task2_passage_output_dir (str): Checkpoint directory for saving passage model weights.
        task2_passage_epochs (int): Training epochs dedicated for passage-type spoilers.
        task2_multi_strategy (str): Method used for multi-spoilers ('seq2seq', 'extractive_iterative', etc).
        task2_multi_method (str): Metric used for similarity checks ('jaccard' or 'tfidf').
        task2_multi_top_k (int): Number of highest ranking multi-sentence spoilers to extract.
        task2_multi_seq2seq_dir (str): Target directory containing the Seq2Seq T5 weights.
    """

    # General Configuration
    seed: int = 42
    output_dir: str = "outputs"
    log_dir: str = "logs"

    # Task 1 - Classification (Roberta/Deberta)
    task1_model_name: str = "microsoft/deberta-v3-base"
    task1_max_length: int = 512
    task1_batch_size: int = 8
    task1_lr: float = 1e-5
    task1_epochs: int = 5
    task1_weight_decay: float = 0.01
    task1_ignore_mismatched_sizes: bool = False

    # Task 2 - QA / Extraction
    task2_model_name: str = "sjrhuschlee/deberta-v3-base-squad2"
    task2_max_length: int = 512
    task2_doc_stride: int = 128
    task2_batch_size: int = 8
    task2_lr: float = 3e-5
    task2_epochs: int = 3
    task2_weight_decay: float = 0.01

    # Task 2 - Phrase Extraction Config
    task2_phrase_model_name: str = "sjrhuschlee/deberta-v3-base-squad2"
    task2_phrase_max_answer_length: int = 10
    task2_phrase_output_dir: str = "results_qa_phrase"
    task2_phrase_epochs: int = 3

    # Task 2 - Passage Extraction Config
    task2_passage_model_name: str = "sjrhuschlee/deberta-v3-base-squad2"
    task2_passage_max_answer_length: int = 120
    task2_passage_output_dir: str = "results_qa_passage"
    task2_passage_epochs: int = 3

    # Multi-generator parameters (Phase 3 & 4)
    task2_multi_strategy: str = "extractive_iterative"
    task2_multi_method: str = "jaccard"
    task2_multi_top_k: int = 5
    task2_multi_seq2seq_dir: str = "results_multi_seq2seq"

    def __post_init__(self) -> None:
        """Create necessary output, multi-generator, and logging directories upon initialization."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.task2_phrase_output_dir, exist_ok=True)
        os.makedirs(self.task2_passage_output_dir, exist_ok=True)
        os.makedirs(self.task2_multi_seq2seq_dir, exist_ok=True)


def override_config_from_args(
    config: PipelineConfig, args: argparse.Namespace
) -> PipelineConfig:
    """Overrides default configuration parameters with parsed command-line arguments.

    Args:
        config (PipelineConfig): The default PipelineConfig instance.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        PipelineConfig: The updated PipelineConfig instance.
    """
    for key, value in vars(args).items():
        if value is not None:
            # Map arg overrides dynamically to correct dataclass attributes
            if key == "multi_strategy" and hasattr(config, "task2_multi_strategy"):
                setattr(config, "task2_multi_strategy", value)
            elif hasattr(config, key):
                setattr(config, key, value)
    return config
