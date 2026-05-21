"""
preprocessing.py
----------------
Text preprocessing pipeline for the CLIN-LLM framework.

Responsibilities
----------------
- Clinical text cleaning and normalisation
- Rule-based negation detection (NegEx-style)
- UMLS concept mapping interface
- SMOTE-based class rebalancing for structured/low-dim features
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# One-time NLTK downloads (idempotent)
# --------------------------------------------------------------------------- #

def _ensure_nltk_data() -> None:
    """Download required NLTK resources if not already present."""
    for resource in ("punkt", "wordnet", "stopwords", "omw-1.4"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk_data()

# --------------------------------------------------------------------------- #
# Negation cues (NegEx-inspired minimal set)
# --------------------------------------------------------------------------- #

NEGATION_CUES: List[str] = [
    "no", "not", "never", "without", "absence of", "deny", "denies",
    "denied", "negative for", "free of", "ruled out", "unremarkable",
    "none", "neither", "nor", "cannot", "doesn't", "didn't", "isn't",
    "wasn't", "hasn't", "haven't", "shouldn't", "no evidence of",
]

# --------------------------------------------------------------------------- #
# Core text cleaner
# --------------------------------------------------------------------------- #

class ClinicalTextPreprocessor:
    """
    Cleans, normalises, and tokenises free-text clinical symptom descriptions.

    Parameters
    ----------
    negation_window : int
        Number of tokens to the right of a negation cue that are marked negated.
    remove_stopwords : bool
        Whether to remove common English stopwords.
    lemmatize : bool
        Whether to apply WordNet lemmatisation.
    """

    def __init__(
        self,
        negation_window: int = 5,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
    ) -> None:
        self.negation_window = negation_window
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self._lemmatizer = WordNetLemmatizer()
        self._stopwords = set(stopwords.words("english"))

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def process(self, text: str) -> str:
        """
        Apply the full preprocessing pipeline to a single text string.

        Returns
        -------
        str
            Cleaned, lemmatised text with negation markers inserted.
        """
        text = self._clean(text)
        tokens = word_tokenize(text)
        tokens = self._mark_negations(tokens)
        if self.remove_stopwords:
            tokens = [t for t in tokens if t.lower() not in self._stopwords]
        if self.lemmatize:
            tokens = [self._lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)

    def process_batch(self, texts: List[str]) -> List[str]:
        """Apply `process` to every string in *texts*."""
        processed = [self.process(t) for t in texts]
        logger.info("Preprocessed %d clinical text records.", len(processed))
        return processed

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _clean(text: str) -> str:
        """Lowercase, remove non-alphanumeric characters, collapse whitespace."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _mark_negations(self, tokens: List[str]) -> List[str]:
        """
        Insert a ``NEG_`` prefix on tokens that follow a negation cue within
        the configured window size.

        Parameters
        ----------
        tokens : list of str
            Word-tokenised text.

        Returns
        -------
        list of str
            Tokens with negation markers applied.
        """
        marked: List[str] = list(tokens)
        n = len(tokens)

        for i, token in enumerate(tokens):
            # Handle both single-word and multi-word cues
            for cue in NEGATION_CUES:
                cue_tokens = cue.split()
                cue_len = len(cue_tokens)
                window_slice = tokens[i: i + cue_len]
                if [t.lower() for t in window_slice] == cue_tokens:
                    end = min(i + cue_len + self.negation_window, n)
                    for j in range(i + cue_len, end):
                        if not marked[j].startswith("NEG_"):
                            marked[j] = f"NEG_{marked[j]}"
                    break

        return marked


# --------------------------------------------------------------------------- #
# UMLS concept mapping interface
# --------------------------------------------------------------------------- #

class UMLSMapper:
    """
    Interface for mapping free-text clinical terms to UMLS Concept Unique
    Identifiers (CUIs).

    When ``umls_api_key`` is provided this class calls the UMLS REST API;
    otherwise it operates in a rule-based fallback mode using a small
    built-in seed dictionary that covers the most common symptom terms.

    Parameters
    ----------
    umls_api_key : str, optional
        UMLS Metathesaurus API key obtained from the NLM portal.
    """

    _SEED_MAP: Dict[str, str] = {
        "fever": "C0015967",
        "cough": "C0010200",
        "headache": "C0018681",
        "fatigue": "C0015672",
        "nausea": "C0027497",
        "vomiting": "C0042963",
        "diarrhea": "C0011991",
        "chest pain": "C0008031",
        "shortness of breath": "C0013404",
        "sore throat": "C0242429",
        "rash": "C0015230",
        "abdominal pain": "C0000737",
        "dizziness": "C0012晕",
        "joint pain": "C0003862",
        "back pain": "C0004604",
    }

    def __init__(self, umls_api_key: Optional[str] = None) -> None:
        self._api_key = umls_api_key
        if umls_api_key:
            logger.info("UMLSMapper initialised with live API key.")
        else:
            logger.warning(
                "No UMLS API key provided. Falling back to seed dictionary. "
                "Set umls_api_key for full concept coverage."
            )

    def map(self, term: str) -> Optional[str]:
        """
        Return the UMLS CUI for *term*, or ``None`` if not found.

        Parameters
        ----------
        term : str
            Clinical term to map (will be normalised to lowercase).

        Returns
        -------
        str or None
        """
        term_lower = term.lower().strip()
        if self._api_key:
            return self._query_api(term_lower)
        return self._SEED_MAP.get(term_lower)

    def map_batch(self, terms: List[str]) -> Dict[str, Optional[str]]:
        """Map a list of terms and return a ``{term: CUI}`` dictionary."""
        return {t: self.map(t) for t in terms}

    def _query_api(self, term: str) -> Optional[str]:
        """
        Perform a live UMLS REST API lookup.

        Notes
        -----
        Requires the ``requests`` library. Returns the top CUI from the first
        result set returned by the /search/current endpoint.
        """
        try:
            import requests

            auth_url = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
            resp = requests.post(auth_url, data={"apikey": self._api_key}, timeout=10)
            resp.raise_for_status()
            tgt = resp.headers.get("location", "")

            ticket_resp = requests.post(
                tgt, data={"service": "http://umlsks.nlm.nih.gov"}, timeout=10
            )
            ticket = ticket_resp.text

            search_url = "https://uts-ws.nlm.nih.gov/rest/search/current"
            params = {"string": term, "ticket": ticket, "pageSize": 1}
            result = requests.get(search_url, params=params, timeout=10).json()
            results = result.get("result", {}).get("results", [])
            if results and results[0].get("ui") != "NONE":
                return results[0]["ui"]
        except Exception as exc:
            logger.warning("UMLS API lookup failed for '%s': %s", term, exc)
        return None


# --------------------------------------------------------------------------- #
# SMOTE-based class rebalancing
# --------------------------------------------------------------------------- #

def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to oversample minority classes in a tabular or low-dimensional
    embedding feature matrix.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix. May be structured vitals or reduced-dim text embeddings.
    y : np.ndarray, shape (n_samples,)
        Integer class labels.
    k_neighbors : int
        Number of nearest neighbours used by SMOTE.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_resampled : np.ndarray
    y_resampled : np.ndarray
    """
    logger.info(
        "Applying SMOTE: input shape %s, class distribution %s",
        X.shape,
        dict(zip(*np.unique(y, return_counts=True))),
    )
    sm = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    logger.info(
        "Post-SMOTE shape %s, class distribution %s",
        X_res.shape,
        dict(zip(*np.unique(y_res, return_counts=True))),
    )
    return X_res, y_res


# --------------------------------------------------------------------------- #
# Structured vitals normaliser
# --------------------------------------------------------------------------- #

def normalise_vitals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a structured vitals DataFrame to z-scores column-wise.

    Parameters
    ----------
    df : pd.DataFrame
        Numeric vital-sign columns (e.g., heart_rate, temperature, spo2).

    Returns
    -------
    pd.DataFrame
        Z-score normalised frame with the same columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        logger.warning("No numeric columns found in vitals DataFrame.")
        return df
    df = df.copy()
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / (
        df[numeric_cols].std() + 1e-8
    )
    logger.info("Normalised %d numeric vital columns.", len(numeric_cols))
    return df


# --------------------------------------------------------------------------- #
# Dataset loader helper
# --------------------------------------------------------------------------- #

def load_symptom_disease_dataset(
    filepath: str | Path,
    text_col: str = "symptoms",
    label_col: str = "disease",
    preprocessor: Optional[ClinicalTextPreprocessor] = None,
) -> Tuple[List[str], np.ndarray, LabelEncoder]:
    """
    Load a symptom-disease CSV, apply text preprocessing, and encode labels.

    Parameters
    ----------
    filepath : str or Path
        Path to the raw dataset CSV.
    text_col : str
        Column name containing symptom descriptions.
    label_col : str
        Column name containing disease labels.
    preprocessor : ClinicalTextPreprocessor, optional
        If ``None``, a default preprocessor is instantiated.

    Returns
    -------
    texts : list of str
        Preprocessed symptom texts.
    labels : np.ndarray
        Integer-encoded disease labels.
    label_encoder : LabelEncoder
        Fitted encoder (keep for inverse transformation at inference time).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Expected columns '{text_col}' and '{label_col}'; "
            f"got {df.columns.tolist()}"
        )

    df = df.dropna(subset=[text_col, label_col])
    logger.info("Loaded %d records from %s.", len(df), filepath)

    if preprocessor is None:
        preprocessor = ClinicalTextPreprocessor()

    texts = preprocessor.process_batch(df[text_col].tolist())
    le = LabelEncoder()
    labels = le.fit_transform(df[label_col].tolist())
    return texts, labels, le
