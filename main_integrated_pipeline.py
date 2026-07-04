import argparse
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,  # Promoted to module level to support robust unittest mocking
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
)

from src.config import PipelineConfig, override_config_from_args
from src.data_loader import JSONLLoader
from src.evaluation import AdvancedEvaluationSuite
from src.logging_setup import setup_logger
from src.qa_inference import predict_best_span
from src.retrieval_generator import RetrievalSpoilerGenerator
from src.utils import extract_primary_tag, join_text_list

# Set up integrated log tracking
logger = setup_logger("integrated_pipeline")


class ClickbaitIntegratedPipeline:
    """Orchestrates combined inference using sequential Task 1 sequence classification
    and Task 2 token span extraction across separate Phrase/Passage/Multi models.
    """

    def __init__(
        self, task1_dir: str, phrase_dir: str, passage_dir: str, config: PipelineConfig
    ):
        logger.info("Initializing Clickbait Spoiler Integrated Pipeline...")
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Task 1 Model
        logger.info(f"Loading Task 1 classification weights from: {task1_dir}")
        self.t1_tokenizer = AutoTokenizer.from_pretrained(task1_dir)
        self.t1_model = AutoModelForSequenceClassification.from_pretrained(task1_dir)
        self.t1_model.to(self.device)
        self.t1_model.eval()
        self.t1_id_to_label = {0: "phrase", 1: "passage", 2: "multi"}

        # Load Phrase QA Model
        logger.info(f"Loading Phrase QA weights from: {phrase_dir}")
        self.phrase_tokenizer = AutoTokenizer.from_pretrained(phrase_dir)
        self.phrase_model = AutoModelForQuestionAnswering.from_pretrained(phrase_dir)
        self.phrase_model.to(self.device)
        self.phrase_model.eval()

        # Load Passage QA Model
        logger.info(f"Loading Passage QA weights from: {passage_dir}")
        self.passage_tokenizer = AutoTokenizer.from_pretrained(passage_dir)
        self.passage_model = AutoModelForQuestionAnswering.from_pretrained(passage_dir)
        self.passage_model.to(self.device)
        self.passage_model.eval()

        # Configure dynamic multi spoilers strategy (Phase 3 & 4)
        self.multi_strategy = config.task2_multi_strategy
        self.multi_model = None
        self.multi_tokenizer = None

        if self.multi_strategy == "extractive_iterative":
            logger.info(
                "Iterative extraction selected. Binding multi model to passage weights."
            )
            self.multi_model = self.passage_model
            self.multi_tokenizer = self.passage_tokenizer
        elif self.multi_strategy == "seq2seq":
            multi_dir = config.task2_multi_seq2seq_dir
            if os.path.exists(multi_dir):
                logger.info(f"Loading Seq2Seq T5 weights from: {multi_dir}")
                self.multi_tokenizer = AutoTokenizer.from_pretrained(multi_dir)
                self.multi_model = AutoModelForSeq2SeqLM.from_pretrained(multi_dir)
                self.multi_model.to(self.device)
                self.multi_model.eval()
            else:
                logger.warning(
                    f"Seq2Seq directory {multi_dir} not found. Falling back to TF-IDF."
                )
                self.multi_strategy = "tfidf"

    def run_inference(self, df: pd.DataFrame):
        """Executes the complete integrated evaluation flow across the input dataframe.

        Args:
            df (pd.DataFrame): Input dataframe containing challenge samples.

        Returns:
            tuple: (predicted_tags, predicted_spoilers, confidence_scores) lists.
        """
        # --- Task 1: Predict Spoiler Types ---
        processed_t1 = pd.DataFrame()
        processed_t1["post"] = df["postText"].apply(join_text_list)
        processed_t1["article"] = df["targetParagraphs"].apply(
            lambda x: join_text_list(x, limit=8) if isinstance(x, list) else ""
        )
        processed_t1["label"] = 0

        t1_ds = Dataset.from_pandas(processed_t1)
        t1_tokenized = t1_ds.map(
            lambda x: self.t1_tokenizer(
                x["post"],
                x["article"],
                truncation=True,
                max_length=self.config.task1_max_length,
            ),
            batched=True,
        )

        logger.info("Running Task 1 Model Inference...")
        trainer = Trainer(model=self.t1_model, processing_class=self.t1_tokenizer)
        t1_preds = trainer.predict(t1_tokenized)

        # Calculate softmax probabilities to extract confidence metrics (Phase 4)
        logits_tensor = torch.tensor(t1_preds.predictions)
        probs = torch.nn.functional.softmax(logits_tensor, dim=-1).numpy()

        predicted_t1_ids = np.argmax(t1_preds.predictions, axis=-1)
        predicted_tags = [self.t1_id_to_label[idx] for idx in predicted_t1_ids]

        # Save confidence score of the predicted class for each prediction
        confidence_scores = [
            float(probs[i, idx]) for i, idx in enumerate(predicted_t1_ids)
        ]

        # --- Task 2: Robust Spoiler Span Extraction ---
        logger.info("Running Task 2 Model Inference with Safe Boundary Mapping...")
        predicted_spoilers = []

        for idx, row in df.iterrows():
            pred_tag = predicted_tags[idx]

            if pred_tag == "multi":
                post_text = join_text_list(row["postText"])
                paragraphs = row["targetParagraphs"]
                context_str = join_text_list(paragraphs)

                # Process dynamically based on multi strategy (Phase 3 & 5)
                if self.multi_strategy == "seq2seq" and self.multi_model:
                    inputs = self.multi_tokenizer(
                        f"question: {post_text} context: {context_str}",
                        max_length=512,
                        truncation=True,
                        return_tensors="pt",
                    ).to(self.device)
                    with torch.no_grad():
                        outputs = self.multi_model.generate(**inputs, max_length=128)
                    generated_text = self.multi_tokenizer.decode(
                        outputs[0], skip_special_tokens=True
                    ).strip()
                    # Splitting segment back based on fixed delimiter " | "
                    spoiler = [
                        s.strip() for s in generated_text.split("|") if s.strip()
                    ]
                    if not spoiler:
                        spoiler = [paragraphs[0]] if paragraphs else []
                elif self.multi_strategy == "extractive_iterative":
                    from src.multi_spoiler_extractor import (
                        generate_iterative_multi_spoiler,
                    )

                    spoiler = generate_iterative_multi_spoiler(
                        model=self.multi_model,
                        tokenizer=self.multi_tokenizer,
                        question=post_text,
                        context=context_str,
                        max_length=self.config.task2_max_length,
                        doc_stride=self.config.task2_doc_stride,
                        max_answer_length=30,  # Smaller answer length for sub-segments
                        device=self.device,
                    )
                    if not spoiler:
                        spoiler = [paragraphs[0]] if paragraphs else []
                else:
                    # Fallback to Jaccard or TF-IDF Lexical retrieval
                    spoiler = RetrievalSpoilerGenerator.generate_multi_spoiler(
                        post_text=post_text,
                        paragraphs=paragraphs,
                        top_k=self.config.task2_multi_top_k,
                        method=self.config.task2_multi_method,
                    )
            else:
                # Select correct model and parameters depending on predicted tag
                if pred_tag == "phrase":
                    tokenizer = self.phrase_tokenizer
                    model = self.phrase_model
                    max_ans_len = self.config.task2_phrase_max_answer_length
                else:  # passage
                    tokenizer = self.passage_tokenizer
                    model = self.passage_model
                    max_ans_len = self.config.task2_passage_max_answer_length

                paragraphs = row["targetParagraphs"]
                context_str = join_text_list(paragraphs)
                post_str = join_text_list(row["postText"])

                span_result = predict_best_span(
                    model=model,
                    tokenizer=tokenizer,
                    question=post_str,
                    context=context_str,
                    max_length=self.config.task2_max_length,
                    doc_stride=self.config.task2_doc_stride,
                    max_answer_length=max_ans_len,
                    device=self.device,
                )
                pred_text = span_result["text"]

                # Safe fallback to the first paragraph if the QA extraction returns empty
                if not pred_text and len(row["targetParagraphs"]) > 0:
                    pred_text = row["targetParagraphs"][0]

                spoiler = [pred_text] if pred_text else []

            predicted_spoilers.append(spoiler)

        return predicted_tags, predicted_spoilers, confidence_scores


def main():
    parser = argparse.ArgumentParser(
        description="Run integrated clickbait spoiler prediction pipeline"
    )
    parser.add_argument(
        "--val_path", type=str, required=True, help="Path to evaluation dataset"
    )
    parser.add_argument(
        "--t1_checkpoint",
        type=str,
        required=True,
        help="Path to Task 1 classifier weights",
    )
    parser.add_argument(
        "--phrase_checkpoint",
        type=str,
        required=True,
        help="Path to Phrase QA weights",
    )
    parser.add_argument(
        "--passage_checkpoint",
        type=str,
        required=True,
        help="Path to Passage QA weights",
    )
    parser.add_argument(
        "--multi_strategy",
        type=str,
        help="Strategy for multi-type spoilers (seq2seq, extractive_iterative, tfidf, jaccard)",
    )
    parser.add_argument(
        "--output_path", type=str, default="final_integrated_submission.jsonl"
    )
    args = parser.parse_args()

    logger.info("Loading validation data...")
    val_df = JSONLLoader(args.val_path).load_data()

    # Initialize configuration with dynamically defined parameters
    config = PipelineConfig()
    config = override_config_from_args(config, args)

    pipeline = ClickbaitIntegratedPipeline(
        task1_dir=args.t1_checkpoint,
        phrase_dir=args.phrase_checkpoint,
        passage_dir=args.passage_checkpoint,
        config=config,
    )

    pred_tags, pred_spoilers, confidence_scores = pipeline.run_inference(val_df)

    # Multi-metric baseline performance scoring
    y_true_tags = val_df["tags"].apply(extract_primary_tag).tolist()
    ground_truth_spoilers = val_df["spoiler"].tolist()

    logger.info("Computing integrated pipeline metrics...")
    AdvancedEvaluationSuite.evaluate_task1(y_true_tags, pred_tags)
    AdvancedEvaluationSuite.evaluate_task2(ground_truth_spoilers, pred_spoilers)

    # Save standardized submission artifact for evaluations with confidence metrics
    logger.info(f"Saving final pipeline output file to {args.output_path}...")
    output_records = [
        {
            "uuid": row["uuid"],
            "spoilerType": pred_tags[i],
            "spoiler": pred_spoilers[i],
            "confidence": confidence_scores[i],
        }
        for i, row in val_df.iterrows()
    ]
    pd.DataFrame(output_records).to_json(args.output_path, orient="records", lines=True)
    logger.info("Pipeline executed successfully.")


if __name__ == "__main__":
    main()
