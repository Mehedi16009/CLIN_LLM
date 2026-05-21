"""
generation.py
-------------
Treatment recommendation generation module using a fine-tuned FLAN-T5 model.

The module combines three information sources into a structured prompt:
  1. Raw patient input (symptoms as free text, or structured vitals)
  2. Predicted disease diagnosis from the classification stage
  3. Top-k retrieved dialogue snippets from the RAG retrieval stage

A FLAN-T5-Large model generates a treatment recommendation from this prompt,
conditioned on the clinical evidence provided.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)

# Lazy import to surface a clear error message if transformers is missing
try:
    from transformers import AutoTokenizer, T5ForConditionalGeneration
except ImportError as exc:
    raise ImportError(
        "The 'transformers' library is required. Install with: pip install transformers"
    ) from exc


# --------------------------------------------------------------------------- #
# Prompt builder
# --------------------------------------------------------------------------- #

def build_treatment_prompt(
    patient_input: Union[str, Dict[str, Any]],
    predicted_disease: str,
    retrieved_contexts: List[str],
) -> str:
    """
    Construct the generation prompt from heterogeneous input sources.

    Parameters
    ----------
    patient_input : str or dict
        Either a free-text symptom description (str) or a dictionary of
        structured vital measurements (e.g. ``{"temperature": 38.5, "spo2": 94}``).
    predicted_disease : str
        Disease label predicted by the classification stage.
    retrieved_contexts : list of str
        Top-k dialogue snippets returned by the retrieval stage, one per entry.

    Returns
    -------
    str
        A formatted T5-style instruction prompt.
    """
    # Normalise patient input to a single string
    if isinstance(patient_input, dict):
        vitals_str = ", ".join(f"{k}: {v}" for k, v in patient_input.items())
        patient_text = f"Patient vitals: {vitals_str}"
    else:
        patient_text = str(patient_input).strip()

    # Format retrieved dialogues as numbered references
    context_block = ""
    if retrieved_contexts:
        lines = [
            f"  Reference {i+1}: {ctx.strip()}"
            for i, ctx in enumerate(retrieved_contexts)
        ]
        context_block = "Relevant clinical dialogues:\n" + "\n".join(lines)

    prompt = textwrap.dedent(f"""
        You are a clinical decision support system. Based on the information below,
        provide a concise, evidence-based treatment recommendation. Include medication
        dosage where appropriate and note any monitoring requirements.

        Patient presentation: {patient_text}

        Predicted diagnosis: {predicted_disease}

        {context_block}

        Treatment recommendation:
    """).strip()

    return prompt


# --------------------------------------------------------------------------- #
# FLAN-T5 generation module
# --------------------------------------------------------------------------- #

class TreatmentGenerator:
    """
    FLAN-T5-based treatment recommendation generator.

    The model ingests a structured prompt (patient vitals, predicted diagnosis,
    and retrieved clinical dialogue context) and generates a free-text treatment
    recommendation.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Default is ``google/flan-t5-large``.
    device : str
        ``'cuda'`` or ``'cpu'``. Falls back to CPU if CUDA is unavailable.
    max_new_tokens : int
        Maximum number of tokens to generate.
    min_new_tokens : int
        Minimum generation length to prevent truncated outputs.
    num_beams : int
        Beam width for beam search decoding.
    temperature : float
        Sampling temperature (only applies when ``do_sample=True``).
    repetition_penalty : float
        Penalty applied to repeated token logits.
    length_penalty : float
        Exponential penalty applied to sequence length during beam search.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-large",
        device: str = "cuda",
        max_new_tokens: int = 256,
        min_new_tokens: int = 40,
        num_beams: int = 4,
        temperature: float = 0.7,
        repetition_penalty: float = 1.3,
        length_penalty: float = 1.0,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.num_beams = num_beams
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty

        logger.info("Loading FLAN-T5 tokenizer and model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info(
            "TreatmentGenerator ready on device: %s", str(self.device)
        )

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def generate(
        self,
        patient_input: Union[str, Dict[str, Any]],
        predicted_disease: str,
        retrieved_contexts: List[str],
    ) -> str:
        """
        Generate a treatment recommendation for a single patient case.

        Parameters
        ----------
        patient_input : str or dict
            Free-text symptoms or structured vitals dictionary.
        predicted_disease : str
            Predicted disease label from the classification stage.
        retrieved_contexts : list of str
            Top-k retrieved dialogue snippets from the retrieval stage.

        Returns
        -------
        str
            Generated treatment recommendation text.
        """
        prompt = build_treatment_prompt(
            patient_input, predicted_disease, retrieved_contexts
        )
        recommendation = self._decode(prompt)
        logger.debug(
            "Generated recommendation for '%s' (%d chars).",
            predicted_disease,
            len(recommendation),
        )
        return recommendation

    def generate_batch(
        self,
        patient_inputs: List[Union[str, Dict[str, Any]]],
        predicted_diseases: List[str],
        retrieved_contexts_batch: List[List[str]],
    ) -> List[str]:
        """
        Generate recommendations for a batch of patient cases.

        Parameters
        ----------
        patient_inputs : list
        predicted_diseases : list of str
        retrieved_contexts_batch : list of lists of str

        Returns
        -------
        list of str
        """
        if not (len(patient_inputs) == len(predicted_diseases) == len(retrieved_contexts_batch)):
            raise ValueError(
                "patient_inputs, predicted_diseases, and retrieved_contexts_batch "
                "must have equal lengths."
            )

        prompts = [
            build_treatment_prompt(pi, pd, rc)
            for pi, pd, rc in zip(patient_inputs, predicted_diseases, retrieved_contexts_batch)
        ]
        return [self._decode(p) for p in prompts]

    def load_fine_tuned_weights(self, checkpoint_path: str | Path) -> None:
        """
        Load fine-tuned model weights from a saved checkpoint.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to a ``.pt`` or HuggingFace directory checkpoint.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if checkpoint_path.is_dir():
            self.model = T5ForConditionalGeneration.from_pretrained(str(checkpoint_path))
            self.model.to(self.device)
        else:
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.eval()
        logger.info("Fine-tuned weights loaded from %s.", checkpoint_path)

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _decode(self, prompt: str) -> str:
        """
        Tokenise *prompt*, run beam-search decoding, and return cleaned output.

        Parameters
        ----------
        prompt : str
            Fully assembled generation prompt.

        Returns
        -------
        str
            Decoded and stripped output text.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=self.min_new_tokens,
                num_beams=self.num_beams,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
                length_penalty=self.length_penalty,
                early_stopping=True,
            )

        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return text.strip()


# --------------------------------------------------------------------------- #
# Convenience factory
# --------------------------------------------------------------------------- #

def build_generator_from_config(config: Dict) -> TreatmentGenerator:
    """
    Instantiate a ``TreatmentGenerator`` from a config dictionary.

    Parameters
    ----------
    config : dict
        Expects a ``generation`` sub-dictionary from ``configs/config.yaml``.

    Returns
    -------
    TreatmentGenerator
    """
    gen_cfg = config.get("generation", config)
    return TreatmentGenerator(
        model_name=gen_cfg.get("model_name", "google/flan-t5-large"),
        device=gen_cfg.get("device", "cuda"),
        max_new_tokens=gen_cfg.get("max_new_tokens", 256),
        min_new_tokens=gen_cfg.get("min_new_tokens", 40),
        num_beams=gen_cfg.get("num_beams", 4),
        temperature=gen_cfg.get("temperature", 0.7),
        repetition_penalty=gen_cfg.get("repetition_penalty", 1.3),
        length_penalty=gen_cfg.get("length_penalty", 1.0),
    )
