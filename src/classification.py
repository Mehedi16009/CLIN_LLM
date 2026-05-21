"""
classification.py
-----------------
BioBERT-based disease classification module with Monte Carlo Dropout for
uncertainty estimation and Focal Loss optimisation.

Key components
--------------
- BioBERTClassifier    : HuggingFace wrapper with dropout-enabled inference
- FocalLoss            : class-imbalance-aware loss function
- MCDropoutInference   : uncertainty quantification via repeated stochastic passes
- ClassificationTrainer: end-to-end training orchestrator
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# PyTorch Dataset
# --------------------------------------------------------------------------- #

class SymptomDataset(Dataset):
    """
    Minimal PyTorch dataset for tokenised symptom texts.

    Parameters
    ----------
    texts : list of str
        Preprocessed symptom strings.
    labels : list of int or np.ndarray
        Integer class labels.
    tokenizer : HuggingFace tokenizer
    max_length : int
        Maximum token sequence length.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int] | np.ndarray,
        tokenizer,
        max_length: int = 256,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# --------------------------------------------------------------------------- #
# Focal Loss
# --------------------------------------------------------------------------- #

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in multi-class classification.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Parameters
    ----------
    gamma : float
        Focusing parameter. Larger values down-weight well-classified samples.
    alpha : float
        Class-weight balancing factor.
    reduction : str
        ``'mean'`` or ``'sum'``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor, shape (batch, num_classes)
        targets : torch.Tensor, shape (batch,), integer class indices

        Returns
        -------
        torch.Tensor
            Scalar focal loss value.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma
        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# --------------------------------------------------------------------------- #
# BioBERT Classifier
# --------------------------------------------------------------------------- #

class BioBERTClassifier(nn.Module):
    """
    Fine-tunable BioBERT classifier with MC Dropout support.

    Dropout is applied to the pooled [CLS] representation before the
    linear classification head. When ``training=False`` the standard
    behaviour keeps dropout disabled; use :py:func:`enable_mc_dropout` to
    re-enable it explicitly for uncertainty estimation at inference time.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (default: ``dmis-lab/biobert-v1.1``).
    num_labels : int
        Number of target disease classes.
    dropout_rate : float
        Dropout probability applied to the pooled representation.
    """

    def __init__(
        self,
        model_name: str = "dmis-lab/biobert-v1.1",
        num_labels: int = 41,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)
        logger.info(
            "BioBERTClassifier initialised: model=%s, classes=%d, dropout=%.2f",
            model_name,
            num_labels,
            dropout_rate,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : torch.Tensor, shape (batch, seq_len)
        attention_mask : torch.Tensor, shape (batch, seq_len)
        token_type_ids : torch.Tensor, optional

        Returns
        -------
        logits : torch.Tensor, shape (batch, num_labels)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.pooler_output          # (batch, hidden_size)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)        # (batch, num_labels)
        return logits

    def enable_mc_dropout(self) -> None:
        """Set all Dropout layers to training mode for MC Dropout inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def disable_mc_dropout(self) -> None:
        """Restore evaluation mode (dropout off) on all Dropout layers."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.eval()


# --------------------------------------------------------------------------- #
# Monte Carlo Dropout Inference
# --------------------------------------------------------------------------- #

class MCDropoutInference:
    """
    Runs T stochastic forward passes through a BioBERTClassifier to
    approximate predictive uncertainty via posterior sampling.

    Parameters
    ----------
    model : BioBERTClassifier
    n_samples : int
        Number of MC Dropout forward passes (T).
    device : str
        ``'cuda'`` or ``'cpu'``.
    """

    def __init__(
        self,
        model: BioBERTClassifier,
        n_samples: int = 30,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.n_samples = n_samples
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def predict(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, np.ndarray]:
        """
        Perform MC Dropout inference on a single batch.

        Parameters
        ----------
        batch : dict
            Must contain ``input_ids``, ``attention_mask``, and optionally
            ``token_type_ids``. Labels are ignored if present.

        Returns
        -------
        dict with keys:
            ``mean_probs``     : mean class probabilities, shape (batch, classes)
            ``pred_entropy``   : predictive entropy per sample, shape (batch,)
            ``predicted_class``: argmax of mean probs, shape (batch,)
            ``confidence``     : 1 - normalised entropy, shape (batch,)
        """
        self.model.eval()
        self.model.enable_mc_dropout()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        all_probs: List[np.ndarray] = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        self.model.disable_mc_dropout()

        stacked = np.stack(all_probs, axis=0)          # (T, batch, classes)
        mean_probs = stacked.mean(axis=0)               # (batch, classes)
        predicted_class = mean_probs.argmax(axis=-1)    # (batch,)

        # Predictive entropy: H = -sum(p * log(p + eps))
        eps = 1e-8
        entropy = -(mean_probs * np.log(mean_probs + eps)).sum(axis=-1)
        max_entropy = math.log(mean_probs.shape[-1])
        normalised_entropy = entropy / (max_entropy + eps)
        confidence = 1.0 - normalised_entropy

        return {
            "mean_probs": mean_probs,
            "pred_entropy": entropy,
            "predicted_class": predicted_class,
            "confidence": confidence,
        }


# --------------------------------------------------------------------------- #
# Training orchestrator
# --------------------------------------------------------------------------- #

class ClassificationTrainer:
    """
    End-to-end training wrapper for ``BioBERTClassifier``.

    Parameters
    ----------
    model : BioBERTClassifier
    config : dict
        Hyperparameter dictionary loaded from ``configs/config.yaml``.
    device : str
        ``'cuda'`` or ``'cpu'``.
    """

    def __init__(
        self,
        model: BioBERTClassifier,
        config: Dict,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = FocalLoss(
            gamma=config.get("focal_loss_gamma", 2.0),
            alpha=config.get("focal_loss_alpha", 0.25),
        )
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_f1": [],
            "val_f1": [],
        }

    # ------------------------------------------------------------------ #
    # Public methods
    # ------------------------------------------------------------------ #

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
        output_dir: Optional[str | Path] = None,
    ) -> Dict[str, List[float]]:
        """
        Run the full training loop.

        Parameters
        ----------
        train_loader : DataLoader
        val_loader : DataLoader
        epochs : int, optional
            Overrides ``config['epochs']`` if supplied.
        output_dir : str or Path, optional
            Directory to save the best model checkpoint.

        Returns
        -------
        dict
            Training history (losses and F1 scores per epoch).
        """
        from sklearn.metrics import f1_score as sk_f1

        epochs = epochs or self.config.get("epochs", 10)
        lr = float(self.config.get("learning_rate", 2e-5))
        weight_decay = float(self.config.get("weight_decay", 0.01))
        warmup_ratio = float(self.config.get("warmup_ratio", 0.1))

        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)

        optimiser = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = get_linear_schedule_with_warmup(
            optimiser, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        best_val_f1 = -1.0

        for epoch in range(1, epochs + 1):
            train_loss, train_f1 = self._run_epoch(
                train_loader, optimiser, scheduler, sk_f1, training=True
            )
            val_loss, val_f1 = self._run_epoch(
                val_loader, optimiser, scheduler, sk_f1, training=False
            )

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_f1"].append(train_f1)
            self.history["val_f1"].append(val_f1)

            logger.info(
                "Epoch %02d/%02d | train_loss=%.4f val_loss=%.4f "
                "train_f1=%.4f val_f1=%.4f",
                epoch,
                epochs,
                train_loss,
                val_loss,
                train_f1,
                val_f1,
            )

            if output_dir and val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self._save_checkpoint(output_dir)

        return self.history

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _run_epoch(
        self,
        loader: DataLoader,
        optimiser,
        scheduler,
        f1_fn,
        training: bool,
    ) -> Tuple[float, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)

                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                loss = self.criterion(logits, labels)

                if training:
                    optimiser.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimiser.step()
                    scheduler.step()

                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy().tolist())

        avg_loss = total_loss / len(loader)
        f1 = f1_fn(all_labels, all_preds, average="weighted", zero_division=0)
        return avg_loss, float(f1)

    def _save_checkpoint(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "biobert_best.pt"
        torch.save(self.model.state_dict(), checkpoint_path)
        logger.info("Best model checkpoint saved to %s", checkpoint_path)


# --------------------------------------------------------------------------- #
# Convenience factory
# --------------------------------------------------------------------------- #

def build_classifier_from_config(config: Dict) -> Tuple[BioBERTClassifier, AutoTokenizer]:
    """
    Instantiate a BioBERT classifier and its tokenizer from a config dict.

    Parameters
    ----------
    config : dict
        Expects keys ``model_name``, ``num_labels``, ``dropout_rate``.

    Returns
    -------
    (BioBERTClassifier, AutoTokenizer)
    """
    model_name = config.get("model_name", "dmis-lab/biobert-v1.1")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BioBERTClassifier(
        model_name=model_name,
        num_labels=config.get("num_labels", 41),
        dropout_rate=config.get("dropout_rate", 0.3),
    )
    return model, tokenizer
