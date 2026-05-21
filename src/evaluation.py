"""
evaluation.py
-------------
Evaluation module for the CLIN-LLM framework.

Computes and persists:
- Classification metrics: accuracy, macro/weighted F1, per-class precision/recall
- Confusion matrix heatmap
- ROC curves (one-vs-rest, per class)
- Precision-Recall curves (per class)
- Training dynamics (loss and F1 over epochs)
- Retrieval precision@k for the RAG component

All plots are publication-quality and saved in the configured format (default: PDF).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend, safe for server environments

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Plotting configuration
# --------------------------------------------------------------------------- #

def configure_plot_style() -> None:
    """Apply consistent, publication-quality plot styling."""
    sns.set_theme(style="whitegrid", font_scale=1.3)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 150,
    })


configure_plot_style()


# --------------------------------------------------------------------------- #
# Classification evaluation
# --------------------------------------------------------------------------- #

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute standard classification metrics.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        True integer labels.
    y_pred : np.ndarray, shape (n,)
        Predicted integer labels.
    class_names : list of str, optional
        Human-readable class names for the report.

    Returns
    -------
    dict
        Contains ``accuracy``, ``macro_f1``, ``weighted_f1``, and the full
        ``classification_report`` string.
    """
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    report_str = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    logger.info(
        "Evaluation results: accuracy=%.4f, macro_f1=%.4f, weighted_f1=%.4f",
        accuracy,
        macro_f1,
        weighted_f1,
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report_str,
    }


# --------------------------------------------------------------------------- #
# Confusion matrix
# --------------------------------------------------------------------------- #

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_dir: str | Path,
    fmt: str = "pdf",
) -> Path:
    """
    Plot and save a normalised confusion matrix heatmap.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    class_names : list of str
    output_dir : str or Path
    fmt : str
        Output file format (``'pdf'``, ``'png'``, etc.).

    Returns
    -------
    Path
        Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    n = len(class_names)
    fig_size = max(12, n * 0.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        cm_norm,
        annot=(n <= 20),
        fmt=".2f" if n <= 20 else "",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.4,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Normalised Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = output_dir / f"confusion_matrix.{fmt}"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to %s.", save_path)
    return save_path


# --------------------------------------------------------------------------- #
# ROC curves
# --------------------------------------------------------------------------- #

def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    output_dir: str | Path,
    fmt: str = "pdf",
) -> Path:
    """
    Plot and save per-class ROC curves (one-vs-rest).

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
    y_proba : np.ndarray, shape (n, n_classes)
    class_names : list of str
    output_dir : str or Path
    fmt : str

    Returns
    -------
    Path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(12, 8))
    palette = sns.color_palette("husl", n_classes)

    for i, (name, color) in enumerate(zip(class_names, palette)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=1.5, label=f"{name} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.tight_layout()

    save_path = output_dir / f"roc_curves.{fmt}"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curves saved to %s.", save_path)
    return save_path


# --------------------------------------------------------------------------- #
# Precision-Recall curves
# --------------------------------------------------------------------------- #

def plot_pr_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    output_dir: str | Path,
    fmt: str = "pdf",
) -> Path:
    """
    Plot and save per-class Precision-Recall curves.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
    y_proba : np.ndarray, shape (n, n_classes)
    class_names : list of str
    output_dir : str or Path
    fmt : str

    Returns
    -------
    Path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(12, 8))
    palette = sns.color_palette("husl", n_classes)

    for i, (name, color) in enumerate(zip(class_names, palette)):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, color=color, lw=1.5, label=f"{name} (AUC={pr_auc:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title("Precision-Recall Curves")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.tight_layout()

    save_path = output_dir / f"pr_curves.{fmt}"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("PR curves saved to %s.", save_path)
    return save_path


# --------------------------------------------------------------------------- #
# Training dynamics
# --------------------------------------------------------------------------- #

def plot_training_dynamics(
    history: Dict[str, List[float]],
    output_dir: str | Path,
    fmt: str = "pdf",
) -> Path:
    """
    Plot training and validation loss alongside F1 score dynamics.

    Parameters
    ----------
    history : dict
        Must contain ``train_loss``, ``val_loss``, ``train_f1``, ``val_f1`` lists.
    output_dir : str or Path
    fmt : str

    Returns
    -------
    Path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], color="tab:blue", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], color="tab:orange", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(epochs, history["train_f1"], "--", color="tab:green", label="Train F1")
    ax2.plot(epochs, history["val_f1"], "--", color="tab:red", label="Val F1")
    ax2.set_ylabel("F1 Score")
    ax2.legend(loc="upper right")

    plt.title("Training Dynamics: Loss and F1 Score per Epoch")
    fig.tight_layout()

    save_path = output_dir / f"training_dynamics.{fmt}"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training dynamics plot saved to %s.", save_path)
    return save_path


# --------------------------------------------------------------------------- #
# Retrieval precision@k
# --------------------------------------------------------------------------- #

def compute_retrieval_precision_at_k(
    queries: List[str],
    ground_truth_diseases: List[str],
    retrieved_results: List[List[Dict]],
    k: int = 5,
) -> float:
    """
    Compute Precision@k for the retrieval component.

    A retrieved dialogue is considered relevant if the ``disease`` field in
    the result dict matches the query's ground-truth disease label (case-insensitive).

    Parameters
    ----------
    queries : list of str
    ground_truth_diseases : list of str
    retrieved_results : list of list of dict
        Each inner list contains top-k retrieved result dicts (with ``disease`` key).
    k : int

    Returns
    -------
    float
        Mean Precision@k across all queries.
    """
    precisions: List[float] = []
    for gt, results in zip(ground_truth_diseases, retrieved_results):
        top_k = results[:k]
        hits = sum(
            1 for r in top_k
            if r.get("disease", "").lower() == gt.lower()
        )
        precisions.append(hits / min(k, len(top_k)) if top_k else 0.0)

    p_at_k = float(np.mean(precisions)) if precisions else 0.0
    logger.info("Retrieval Precision@%d: %.4f", k, p_at_k)
    return p_at_k


# --------------------------------------------------------------------------- #
# Full evaluation runner
# --------------------------------------------------------------------------- #

class Evaluator:
    """
    Convenience wrapper that runs all evaluation routines and saves reports.

    Parameters
    ----------
    config : dict
        Evaluation configuration block from ``configs/config.yaml``.
    """

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.output_dir = Path(config.get("report_output_dir", "outputs/results"))
        self.fmt = config.get("plot_format", "pdf")
        self.generate_plots = config.get("generate_plots", True)

    def run_classification_eval(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        class_names: List[str],
        history: Optional[Dict[str, List[float]]] = None,
    ) -> Dict:
        """
        Run the full classification evaluation suite.

        Parameters
        ----------
        y_true : np.ndarray
        y_pred : np.ndarray
        y_proba : np.ndarray or None
            Required for ROC and PR curves.
        class_names : list of str
        history : dict, optional
            Training history for the dynamics plot.

        Returns
        -------
        dict
            All computed metrics.
        """
        metrics = compute_classification_metrics(y_true, y_pred, class_names)

        # Print the report to the logger
        logger.info("\n%s", metrics["classification_report"])

        # Save text report
        report_path = self.output_dir / "classification_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(metrics["classification_report"], encoding="utf-8")

        if self.generate_plots:
            plot_confusion_matrix(y_true, y_pred, class_names, self.output_dir, self.fmt)

            if y_proba is not None:
                plot_roc_curves(y_true, y_proba, class_names, self.output_dir, self.fmt)
                plot_pr_curves(y_true, y_proba, class_names, self.output_dir, self.fmt)

            if history:
                plot_training_dynamics(history, self.output_dir, self.fmt)

        return metrics
