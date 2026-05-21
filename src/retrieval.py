"""
retrieval.py
------------
Semantic search module for the CLIN-LLM Retrieval-Augmented Generation (RAG)
stage.

Architecture
------------
1. A biomedical Sentence-BERT model encodes dialogue entries from the
   MedDialog corpus into a dense vector index.
2. At inference time, the predicted disease string is encoded into a query
   vector and compared against the index via cosine similarity.
3. The top-k most similar dialogue snippets are returned for inclusion in
   the FLAN-T5 generation prompt.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import to avoid hard dependency at module load time
_sentence_transformers_available = False
try:
    from sentence_transformers import SentenceTransformer
    _sentence_transformers_available = True
except ImportError:
    logger.warning(
        "sentence-transformers not installed. Run: pip install sentence-transformers"
    )


# --------------------------------------------------------------------------- #
# MedDialog corpus mock / loader
# --------------------------------------------------------------------------- #

_MOCK_CORPUS: List[Dict[str, str]] = [
    {
        "disease": "pneumonia",
        "dialogue": (
            "Patient: I have had a high fever, cough with yellow mucus, and chest "
            "pain for five days. Doctor: These symptoms are consistent with bacterial "
            "pneumonia. I recommend a chest X-ray and a course of amoxicillin-clavulanate."
        ),
    },
    {
        "disease": "type 2 diabetes",
        "dialogue": (
            "Patient: I feel excessively thirsty, urinate frequently, and have blurred "
            "vision. Doctor: Your fasting glucose was 210 mg/dL. We will start metformin "
            "500 mg twice daily and arrange a dietary consultation."
        ),
    },
    {
        "disease": "hypertension",
        "dialogue": (
            "Patient: I have persistent headaches and occasional dizziness. Blood pressure "
            "readings at home are around 155/95. Doctor: This confirms stage 2 hypertension. "
            "Lifestyle modification and amlodipine 5 mg daily are indicated."
        ),
    },
    {
        "disease": "migraine",
        "dialogue": (
            "Patient: I experience throbbing, one-sided headaches with nausea and light "
            "sensitivity lasting up to 18 hours. Doctor: This presentation is classic "
            "migraine with aura. Sumatriptan 50 mg at onset is the first-line abortive "
            "treatment."
        ),
    },
    {
        "disease": "urinary tract infection",
        "dialogue": (
            "Patient: I have burning during urination, frequent urgency, and cloudy urine. "
            "Doctor: Urinalysis shows positive nitrites and leukocyte esterase. A 7-day "
            "course of trimethoprim-sulfamethoxazole is appropriate."
        ),
    },
    {
        "disease": "gastroesophageal reflux disease",
        "dialogue": (
            "Patient: I have a burning sensation in my chest after meals and sometimes "
            "a sour taste in my mouth. Doctor: This is consistent with GERD. Omeprazole "
            "20 mg before breakfast daily is recommended along with dietary modifications."
        ),
    },
    {
        "disease": "asthma",
        "dialogue": (
            "Patient: I have wheezing, shortness of breath and chest tightness especially "
            "at night and in the morning. Doctor: Spirometry confirms reversible airflow "
            "obstruction. Inhaled salbutamol as a reliever and low-dose budesonide as a "
            "controller are prescribed."
        ),
    },
    {
        "disease": "iron deficiency anemia",
        "dialogue": (
            "Patient: I feel constantly tired, have pale skin, and get out of breath "
            "easily. Doctor: Your haemoglobin is 9.2 g/dL with low ferritin. Ferrous "
            "sulphate 325 mg three times daily is recommended for eight weeks."
        ),
    },
    {
        "disease": "hypothyroidism",
        "dialogue": (
            "Patient: I have been gaining weight, feeling cold all the time, and "
            "experiencing hair loss. Doctor: TSH is elevated at 12 mIU/L, confirming "
            "primary hypothyroidism. We will initiate levothyroxine 50 mcg daily."
        ),
    },
    {
        "disease": "anxiety disorder",
        "dialogue": (
            "Patient: I have persistent worry, racing heart, and difficulty sleeping. "
            "Doctor: You meet criteria for generalised anxiety disorder. Cognitive "
            "behavioural therapy is first-line; sertraline 50 mg daily may also be "
            "appropriate if symptoms persist."
        ),
    },
]


def load_meddialog_corpus(corpus_path: Optional[str | Path] = None) -> List[Dict[str, str]]:
    """
    Load the MedDialog corpus from a JSON file or fall back to the built-in mock.

    Parameters
    ----------
    corpus_path : str or Path, optional
        Path to a JSON file containing a list of ``{"disease": ..., "dialogue": ...}``
        records. If ``None`` or the file does not exist, the built-in mock is used.

    Returns
    -------
    list of dict
        Each entry contains at least ``"disease"`` and ``"dialogue"`` keys.
    """
    if corpus_path is not None:
        corpus_path = Path(corpus_path)
        if corpus_path.exists():
            with corpus_path.open("r", encoding="utf-8") as fh:
                corpus = json.load(fh)
            logger.info("Loaded %d dialogue records from %s.", len(corpus), corpus_path)
            return corpus
        else:
            logger.warning(
                "Corpus file not found at %s. Falling back to built-in mock corpus.",
                corpus_path,
            )
    logger.info("Using built-in mock corpus with %d records.", len(_MOCK_CORPUS))
    return _MOCK_CORPUS


# --------------------------------------------------------------------------- #
# Semantic retrieval index
# --------------------------------------------------------------------------- #

class MedDialogRetriever:
    """
    Dense semantic search index over MedDialog dialogue snippets.

    The index is built by encoding each dialogue string with a biomedical
    Sentence-BERT model. At query time, cosine similarity is used to rank
    entries against an encoded disease/query vector.

    Parameters
    ----------
    model_name : str
        HuggingFace Sentence-Transformer identifier.
    top_k : int
        Number of top similar dialogues to retrieve.
    corpus_path : str or Path, optional
        Path to a processed corpus JSON (see ``load_meddialog_corpus``).
    """

    def __init__(
        self,
        model_name: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        top_k: int = 5,
        corpus_path: Optional[str | Path] = None,
    ) -> None:
        if not _sentence_transformers_available:
            raise ImportError(
                "sentence-transformers is required. Install with: "
                "pip install sentence-transformers"
            )
        self.top_k = top_k
        self.model = SentenceTransformer(model_name)
        self.corpus = load_meddialog_corpus(corpus_path)
        self._index: Optional[np.ndarray] = None
        self._build_index()
        logger.info(
            "MedDialogRetriever ready: model=%s, corpus_size=%d, top_k=%d",
            model_name,
            len(self.corpus),
            top_k,
        )

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str) -> List[Dict[str, str]]:
        """
        Return the top-k most semantically similar dialogue records for *query*.

        Parameters
        ----------
        query : str
            A disease name or free-text symptom description.

        Returns
        -------
        list of dict
            Sorted list (most similar first) of corpus entries, each augmented
            with a ``"similarity"`` key.
        """
        if self._index is None:
            raise RuntimeError("Index has not been built. Call _build_index() first.")

        query_vec = self._encode([query])                   # (1, dim)
        similarities = self._cosine_similarity(query_vec, self._index)  # (n_corpus,)
        top_indices = np.argsort(similarities)[::-1][: self.top_k]

        results = []
        for idx in top_indices:
            entry = dict(self.corpus[idx])
            entry["similarity"] = float(similarities[idx])
            results.append(entry)

        logger.debug("Retrieved %d dialogues for query: '%s'", len(results), query[:60])
        return results

    def retrieve_batch(self, queries: List[str]) -> List[List[Dict[str, str]]]:
        """Apply :py:meth:`retrieve` to every query in *queries*."""
        return [self.retrieve(q) for q in queries]

    def format_context(self, results: List[Dict[str, str]]) -> str:
        """
        Concatenate retrieved dialogues into a single formatted context string
        suitable for inclusion in a generation prompt.

        Parameters
        ----------
        results : list of dict
            Output of :py:meth:`retrieve`.

        Returns
        -------
        str
        """
        lines = []
        for i, entry in enumerate(results, start=1):
            lines.append(f"[Reference {i}] {entry['dialogue']}")
        return "\n\n".join(lines)

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _build_index(self) -> None:
        """Encode all corpus dialogues and store as the dense index matrix."""
        dialogues = [entry["dialogue"] for entry in self.corpus]
        logger.info("Building semantic index over %d dialogues.", len(dialogues))
        self._index = self._encode(dialogues)           # (n_corpus, dim)
        logger.info("Index built, shape: %s.", str(self._index.shape))

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalised sentence embeddings, shape (n, dim)."""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings

    @staticmethod
    def _cosine_similarity(
        query: np.ndarray,
        index: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between a single query vector and all index rows.

        Since both vectors are L2-normalised, the dot product equals cosine similarity.

        Parameters
        ----------
        query : np.ndarray, shape (1, dim)
        index : np.ndarray, shape (n, dim)

        Returns
        -------
        np.ndarray, shape (n,)
        """
        return (index @ query.T).squeeze(axis=-1)

    def save_index(self, output_path: str | Path) -> None:
        """Persist the dense index to disk as a .npy file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, self._index)
        logger.info("Saved index to %s.", output_path)

    def load_index(self, index_path: str | Path) -> None:
        """Load a previously saved dense index from a .npy file."""
        self._index = np.load(str(index_path))
        logger.info("Loaded index from %s, shape: %s.", index_path, self._index.shape)
