import argparse
import os
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Configuration storage for task classification and answer extraction pipelines."""

    # General Configuration
    seed: int = 42
    output_dir: str = "outputs"
    log_dir: str = "logs"

    # Task 1 - Classification (Roberta/Deberta)
    task1_model_name: str = "roberta-base"
    task1_max_length: int = 512
    task1_batch_size: int = 8
    task1_lr: float = 2e-5
    task1_epochs: int = 5
    task1_weight_decay: float = 0.01
    task1_ignore_mismatched_sizes: bool = False

    # Task 2 - QA / Extraction
    task2_model_name: str = "deepset/roberta-base-squad2"
    task2_max_length: int = 512
    task2_doc_stride: int = 128
    task2_batch_size: int = 8
    task2_lr: float = 3e-5
    task2_epochs: int = 3
    task2_weight_decay: float = 0.01

    # Task 2 - Phrase Extraction Config
    task2_phrase_model_name: str = "deepset/roberta-base-squad2"
    task2_phrase_max_answer_length: int = 10
    task2_phrase_output_dir: str = "results_qa_phrase"
    task2_phrase_epochs: int = 3

    # Task 2 - Passage Extraction Config
    task2_passage_model_name: str = "deepset/roberta-base-squad2"
    task2_passage_max_answer_length: int = 150
    task2_passage_output_dir: str = "results_qa_passage"
    task2_passage_epochs: int = 3

    # Multi-generator parameters (Phase 3 & 4)
    task2_multi_strategy: str = (
        "seq2seq"  # Changed default strategy to seq2seq to automatically load T5 model
    )
    task2_multi_method: str = "jaccard"
    task2_multi_top_k: int = 3
    task2_multi_seq2seq_dir: str = (
        "results_multi_seq2seq"  # Moved directory from hardcoded to config
    )

    def __post_init__(self) -> None:
        """Create necessary output and logging directories."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.task2_phrase_output_dir, exist_ok=True)
        os.makedirs(self.task2_passage_output_dir, exist_ok=True)


def override_config_from_args(
    config: PipelineConfig, args: argparse.Namespace
) -> PipelineConfig:
    """Overrides default configuration parameters with parsed command-line arguments.

    Args:
        config: The default PipelineConfig instance.
        args: Parsed command-line arguments.

    Returns:
        The updated PipelineConfig instance.
    """
    for key, value in vars(args).items():
        if value is not None:
            # Map arg overrides dynamically to correct dataclass attributes
            if key == "multi_strategy" and hasattr(config, "task2_multi_strategy"):
                setattr(config, "task2_multi_strategy", value)
            elif hasattr(config, key):
                setattr(config, key, value)
    return config
