"""
CLIN-LLM_runner.py
------------------
End-to-end pipeline runner for the CLIN-LLM framework.

Usage
-----
  python CLIN-LLM_runner.py --config configs/config.yaml --data data/raw/dataset.csv
  python CLIN-LLM_runner.py --config configs/config.yaml --infer \
         --symptoms "fever, dry cough, fatigue" --mode text
  python CLIN-LLM_runner.py --config configs/config.yaml --eval-only \
         --checkpoint outputs/models/biobert_best.pt

Pipeline stages
---------------
1. Preprocessing   : text cleaning, negation detection, SMOTE
2. Classification  : BioBERT fine-tuning with Focal Loss + MC Dropout
3. Retrieval       : Sentence-BERT semantic search over MedDialog
4. Generation      : FLAN-T5 treatment recommendation
5. Safety filters  : antibiotic stewardship, confidence triage, DDI check
6. Evaluation      : accuracy, F1, confusion matrix, ROC/PR curves
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# ------------------------------------------------------------------ #
# Logging setup
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("CLIN-LLM")


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration file and return as a nested dictionary."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_seed(seed: int) -> None:
    """Fix all relevant random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed set to %d.", seed)


# ------------------------------------------------------------------ #
# Pipeline stages
# ------------------------------------------------------------------ #

def run_training_pipeline(config: dict, data_path: str | Path) -> None:
    """
    Execute the complete training pipeline.

    1. Load and preprocess the symptom-disease dataset.
    2. Apply SMOTE for class rebalancing.
    3. Fine-tune BioBERT with Focal Loss.
    4. Persist the best checkpoint.
    """
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader

    from src.preprocessing import (
        ClinicalTextPreprocessor,
        apply_smote,
        load_symptom_disease_dataset,
    )
    from src.classification import (
        BioBERTClassifier,
        ClassificationTrainer,
        SymptomDataset,
        build_classifier_from_config,
    )

    cls_cfg = config["classification"]
    seed = config["project"]["seed"]
    set_seed(seed)

    # --- Load and preprocess ---
    logger.info("Stage 1/3: Loading and preprocessing dataset from %s", data_path)
    texts, labels, label_encoder = load_symptom_disease_dataset(
        filepath=data_path,
        preprocessor=ClinicalTextPreprocessor(),
    )

    # --- Train / val / test split ---
    test_split = config["evaluation"]["test_split"]
    val_split = config["evaluation"]["val_split"]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        texts, labels, test_size=test_split, random_state=seed, stratify=labels
    )
    val_frac = val_split / (1.0 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac, random_state=seed, stratify=y_trainval
    )

    logger.info(
        "Split sizes: train=%d, val=%d, test=%d", len(X_train), len(X_val), len(X_test)
    )

    # --- Build model and tokenizer ---
    logger.info("Stage 2/3: Building BioBERT classifier.")
    model, tokenizer = build_classifier_from_config(cls_cfg)
    device = cls_cfg.get("device", "cuda")

    # --- Datasets and loaders ---
    max_length = cls_cfg.get("max_length", 256)
    batch_size = cls_cfg.get("batch_size", 32)

    train_ds = SymptomDataset(X_train, y_train, tokenizer, max_length)
    val_ds = SymptomDataset(X_val, y_val, tokenizer, max_length)
    test_ds = SymptomDataset(X_test, y_test, tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # --- Train ---
    logger.info("Stage 3/3: Training BioBERT with Focal Loss.")
    output_dir = Path(config["paths"]["model_output"])
    trainer = ClassificationTrainer(model=model, config=cls_cfg, device=device)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
    )

    # --- Evaluate on test set ---
    logger.info("Running test-set evaluation.")
    run_evaluation(
        config=config,
        model=model,
        test_loader=test_loader,
        label_encoder_classes=label_encoder.classes_.tolist(),
        history=history,
        device=device,
    )


def run_evaluation(
    config: dict,
    model,
    test_loader,
    label_encoder_classes: list,
    history: dict | None = None,
    device: str = "cuda",
) -> None:
    """Compute metrics on the test set and save reports and plots."""
    from src.classification import MCDropoutInference
    from src.evaluation import Evaluator

    mc = MCDropoutInference(
        model=model,
        n_samples=config["classification"]["mc_dropout_samples"],
        device=device,
    )

    all_preds, all_probs, all_labels = [], [], []
    for batch in test_loader:
        result = mc.predict(batch)
        all_preds.extend(result["predicted_class"].tolist())
        all_probs.append(result["mean_probs"])
        all_labels.extend(batch["labels"].numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.concatenate(all_probs, axis=0)

    evaluator = Evaluator(config=config["evaluation"])
    evaluator.run_classification_eval(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=label_encoder_classes,
        history=history,
    )


def run_inference(
    config: dict,
    symptoms: str,
    mode: str = "text",
    checkpoint: str | None = None,
) -> None:
    """
    Run single-sample inference through the full pipeline.

    Parameters
    ----------
    config : dict
    symptoms : str
        Free-text symptom description.
    mode : str
        ``'text'`` or ``'vitals'`` (future structured input).
    checkpoint : str, optional
        Path to a fine-tuned BioBERT checkpoint. Uses untrained model if None.
    """
    from src.preprocessing import ClinicalTextPreprocessor
    from src.classification import (
        BioBERTClassifier,
        MCDropoutInference,
        build_classifier_from_config,
    )
    from src.retrieval import MedDialogRetriever
    from src.generation import TreatmentGenerator, build_generator_from_config
    from src.safety_filters import SafetyEvaluator

    set_seed(config["project"]["seed"])
    device = config["classification"].get("device", "cuda")
    cls_cfg = config["classification"]

    # --- Preprocess input ---
    preprocessor = ClinicalTextPreprocessor()
    clean_symptoms = preprocessor.process(symptoms)
    logger.info("Preprocessed symptoms: %s", clean_symptoms)

    # --- Load classifier ---
    model, tokenizer = build_classifier_from_config(cls_cfg)
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)
            logger.info("Loaded checkpoint: %s", ckpt_path)
        else:
            logger.warning("Checkpoint not found at %s. Using untrained model.", ckpt_path)

    model.to(device)

    # --- Classify ---
    encoding = tokenizer(
        [clean_symptoms],
        truncation=True,
        padding="max_length",
        max_length=cls_cfg.get("max_length", 256),
        return_tensors="pt",
    )
    mc = MCDropoutInference(
        model=model,
        n_samples=cls_cfg.get("mc_dropout_samples", 30),
        device=device,
    )
    mc_result = mc.predict(encoding)
    predicted_idx = int(mc_result["predicted_class"][0])
    confidence = float(mc_result["confidence"][0])

    # Use a placeholder label mapping if no encoder available
    label_name = f"disease_{predicted_idx}"
    logger.info(
        "Predicted class index: %d | confidence: %.3f", predicted_idx, confidence
    )

    # --- Retrieve context ---
    ret_cfg = config["retrieval"]
    retriever = MedDialogRetriever(
        model_name=ret_cfg.get("model_name", "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"),
        top_k=ret_cfg.get("top_k", 5),
        corpus_path=ret_cfg.get("corpus_path"),
    )
    retrieved = retriever.retrieve(label_name)
    context_strings = [r["dialogue"] for r in retrieved]
    logger.info("Retrieved %d context dialogues.", len(context_strings))

    # --- Generate recommendation ---
    generator = build_generator_from_config(config)
    recommendation = generator.generate(
        patient_input=symptoms,
        predicted_disease=label_name,
        retrieved_contexts=context_strings,
    )
    logger.info("Generated recommendation:\n%s", recommendation)

    # --- Safety filter ---
    safety_cfg = config.get("safety", {})
    evaluator = SafetyEvaluator(config=safety_cfg)
    report = evaluator.evaluate(
        recommendation=recommendation,
        predicted_disease=label_name,
        confidence=confidence,
    )

    # --- Print results ---
    print("\n" + "=" * 60)
    print("CLIN-LLM INFERENCE RESULT")
    print("=" * 60)
    print(f"  Predicted Disease : {label_name}")
    print(f"  Confidence        : {confidence:.3f}")
    print(f"  Triage Outcome    : {report.triage_outcome.value}")
    print(f"\n  Recommendation:\n  {recommendation}\n")
    if report.notes:
        print("  Safety Notes:")
        for note in report.notes:
            print(f"    - {note}")
    print("=" * 60 + "\n")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLIN-LLM: Clinical AI Pipeline for Diagnosis and Treatment",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to the training CSV dataset.",
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Run single-sample inference instead of training.",
    )
    parser.add_argument(
        "--symptoms",
        type=str,
        default=None,
        help="Free-text symptom description for inference mode.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["text", "vitals"],
        default="text",
        help="Input modality for inference.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a BioBERT checkpoint for inference or eval-only mode.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation on a saved checkpoint without retraining.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["project"]["seed"])

    # Add a file handler for persistent logs
    log_dir = Path(config["paths"]["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "clin_llm_run.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    if args.infer:
        if not args.symptoms:
            parser.error("--symptoms is required when using --infer.")
        run_inference(
            config=config,
            symptoms=args.symptoms,
            mode=args.mode,
            checkpoint=args.checkpoint,
        )
    elif args.data:
        run_training_pipeline(config=config, data_path=args.data)
    else:
        parser.print_help()
        logger.error("Provide either --data for training or --infer for inference.")
        sys.exit(1)


if __name__ == "__main__":
    main()
