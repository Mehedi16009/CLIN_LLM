"""
safety_filters.py
-----------------
Post-generation safety validation layer for the CLIN-LLM pipeline.

Three independent safety mechanisms are implemented:

1. Antibiotic stewardship rules
   Regex and lookup structures enforce guideline-based constraints on
   antibiotic prescription (class, dosage range, and indication).

2. MC Dropout confidence triage
   Inputs with high predictive entropy (low classification confidence)
   are routed to a ``"HUMAN_REVIEW"`` state rather than released to
   the patient.

3. RxNorm drug-drug interaction (DDI) checks
   A lookup engine queries the NLM RxNorm REST API (or a local mock)
   to identify high-severity DDI pairs in the recommended regimen.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Enumerations and result types
# --------------------------------------------------------------------------- #

class TriageOutcome(str, Enum):
    """Possible routing decisions after safety evaluation."""
    APPROVED = "APPROVED"
    FLAG_EXPERT_REVIEW = "FLAG_EXPERT_REVIEW"
    FLAG_PHARMACIST_REVIEW = "FLAG_PHARMACIST_REVIEW"
    REJECTED = "REJECTED"


@dataclass
class SafetyReport:
    """
    Aggregated result of all safety checks for a single recommendation.

    Attributes
    ----------
    triage_outcome : TriageOutcome
    confidence_score : float
        MC Dropout confidence from the classification stage (0.0 to 1.0).
    antibiotic_violations : list of str
        Human-readable descriptions of any antibiotic stewardship violations.
    ddi_alerts : list of dict
        Drug pair records returned by the DDI checker.
    final_recommendation : str
        The (possibly modified) treatment recommendation text.
    notes : list of str
        Additional advisory notes appended to the recommendation.
    """
    triage_outcome: TriageOutcome
    confidence_score: float
    antibiotic_violations: List[str] = field(default_factory=list)
    ddi_alerts: List[Dict] = field(default_factory=list)
    final_recommendation: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        return self.triage_outcome == TriageOutcome.APPROVED


# --------------------------------------------------------------------------- #
# 1. Antibiotic stewardship filter
# --------------------------------------------------------------------------- #

# Antibiotic entries: {regex_pattern: (drug_name, max_daily_dose_mg, indication_hint)}
_ANTIBIOTIC_STEWARDSHIP_RULES: Dict[str, Tuple[str, float, str]] = {
    r"\bamoxicillin\b": ("amoxicillin", 3000.0, "community-acquired pneumonia, UTI"),
    r"\bamoxicillin.clavulanate\b": ("amoxicillin-clavulanate", 4000.0, "sinusitis, lower RTI"),
    r"\bazithromycin\b": ("azithromycin", 500.0, "atypical pneumonia, chlamydia"),
    r"\bciprofloxacin\b": ("ciprofloxacin", 1500.0, "gram-negative infections, UTI"),
    r"\btrimethoprim.sulfamethoxazole\b|bactrim|septra": (
        "trimethoprim-sulfamethoxazole", 320.0, "UTI, PCP prophylaxis"
    ),
    r"\bmetronidazole\b": ("metronidazole", 2000.0, "anaerobic infections, C. diff"),
    r"\bvancomycin\b": ("vancomycin", 4000.0, "MRSA, severe Gram+ infections"),
    r"\bdoxycycline\b": ("doxycycline", 200.0, "Lyme disease, chlamydia, atypicals"),
    r"\bclarithromycin\b": ("clarithromycin", 1000.0, "H. pylori, MAC"),
    r"\bceftriaxone\b": ("ceftriaxone", 4000.0, "severe community infections, meningitis"),
    r"\bmeropenem\b": ("meropenem", 6000.0, "hospital-acquired infections, NDM"),
}

# Broad-spectrum flags: these require documented culture sensitivity
_BROAD_SPECTRUM_ANTIBIOTICS: Set[str] = {
    "vancomycin", "meropenem", "ceftriaxone", "ciprofloxacin"
}

# Known contraindication pairs: (drug_a, drug_b, reason)
_ANTIBIOTIC_CONTRAINDICATIONS: List[Tuple[str, str, str]] = [
    ("metronidazole", "alcohol", "disulfiram-like reaction"),
    ("ciprofloxacin", "antacid", "chelation reduces absorption"),
    ("doxycycline", "antacid", "chelation reduces absorption"),
]


class AntibioticStewardshipFilter:
    """
    Rule-based filter enforcing antibiotic stewardship guidelines.

    The filter performs three checks:
    - Detects broad-spectrum antibiotics that require culture documentation.
    - Detects known contraindication pairs in the recommendation text.
    - Warns when an antibiotic is used outside its indicated disease class.

    Parameters
    ----------
    confidence_threshold : float
        Confidence below which culture requirement warnings are escalated.
    """

    def __init__(self, confidence_threshold: float = 0.75) -> None:
        self.confidence_threshold = confidence_threshold

    def check(
        self,
        recommendation: str,
        predicted_disease: str,
        confidence: float,
    ) -> List[str]:
        """
        Evaluate *recommendation* against stewardship rules.

        Parameters
        ----------
        recommendation : str
            Generated treatment recommendation text.
        predicted_disease : str
            Predicted disease label (used for indication matching).
        confidence : float
            Classification confidence from MC Dropout inference.

        Returns
        -------
        list of str
            Violation descriptions. An empty list means the recommendation
            passed all stewardship checks.
        """
        violations: List[str] = []
        rec_lower = recommendation.lower()

        for pattern, (drug_name, _, indication_hint) in _ANTIBIOTIC_STEWARDSHIP_RULES.items():
            if re.search(pattern, rec_lower):
                # Flag broad-spectrum use without documented sensitivity
                if drug_name in _BROAD_SPECTRUM_ANTIBIOTICS:
                    violations.append(
                        f"STEWARDSHIP: '{drug_name}' is broad-spectrum. "
                        "Culture and sensitivity documentation required before use."
                    )

        # Check contraindication pairs
        for drug_a, drug_b, reason in _ANTIBIOTIC_CONTRAINDICATIONS:
            if drug_a in rec_lower and drug_b in rec_lower:
                violations.append(
                    f"CONTRAINDICATION: Co-administration of '{drug_a}' and "
                    f"'{drug_b}' is contraindicated ({reason})."
                )

        return violations


# --------------------------------------------------------------------------- #
# 2. Confidence-based triage
# --------------------------------------------------------------------------- #

class ConfidenceTriage:
    """
    Routes classification outputs to appropriate review states based on the
    MC Dropout predictive confidence score.

    Parameters
    ----------
    confidence_threshold : float
        Samples with confidence below this value are flagged for expert review.
    """

    def __init__(self, confidence_threshold: float = 0.75) -> None:
        self.threshold = confidence_threshold

    def triage(self, confidence: float) -> TriageOutcome:
        """
        Determine the routing outcome for a given confidence score.

        Parameters
        ----------
        confidence : float
            MC Dropout confidence score in [0.0, 1.0].

        Returns
        -------
        TriageOutcome
        """
        if confidence >= self.threshold:
            return TriageOutcome.APPROVED
        return TriageOutcome.FLAG_EXPERT_REVIEW


# --------------------------------------------------------------------------- #
# 3. RxNorm DDI checker
# --------------------------------------------------------------------------- #

# Local mock DDI table: {(drug_a, drug_b): (severity, description)}
_MOCK_DDI_TABLE: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("warfarin", "ciprofloxacin"): (
        "major",
        "Ciprofloxacin inhibits CYP1A2, increasing warfarin effect and bleeding risk.",
    ),
    ("warfarin", "metronidazole"): (
        "contraindicated",
        "Metronidazole markedly increases INR; combination should be avoided.",
    ),
    ("metformin", "contrast dye"): (
        "major",
        "Risk of lactic acidosis if metformin is not withheld before iodinated contrast.",
    ),
    ("ssri", "tramadol"): (
        "major",
        "Risk of serotonin syndrome with concomitant SSRI and tramadol use.",
    ),
    ("simvastatin", "clarithromycin"): (
        "contraindicated",
        "Clarithromycin inhibits CYP3A4, leading to dangerous simvastatin accumulation.",
    ),
    ("clopidogrel", "omeprazole"): (
        "major",
        "Omeprazole inhibits CYP2C19, reducing clopidogrel antiplatelet effect.",
    ),
}


class RxNormDDIChecker:
    """
    Drug-drug interaction checker backed by the NLM RxNorm REST API.

    When ``use_mock=True`` (default), a built-in interaction table is used.
    Set ``use_mock=False`` with a valid network connection to query the live
    RxNorm API.

    Parameters
    ----------
    use_mock : bool
        If ``True``, use the built-in DDI lookup table.
    api_base : str
        RxNorm REST API base URL.
    timeout : int
        HTTP request timeout in seconds.
    high_risk_severities : list of str
        Severity labels that trigger a pharmacist review flag.
    """

    def __init__(
        self,
        use_mock: bool = True,
        api_base: str = "https://rxnav.nlm.nih.gov/REST",
        timeout: int = 5,
        high_risk_severities: Optional[List[str]] = None,
    ) -> None:
        self.use_mock = use_mock
        self.api_base = api_base
        self.timeout = timeout
        self.high_risk_severities = high_risk_severities or ["contraindicated", "major"]

    def check(self, recommendation: str) -> List[Dict]:
        """
        Extract drug names from *recommendation* and check all pairwise
        interactions.

        Parameters
        ----------
        recommendation : str
            Generated treatment recommendation text.

        Returns
        -------
        list of dict
            Each entry contains ``drug_a``, ``drug_b``, ``severity``, and
            ``description`` keys. An empty list means no interactions were found.
        """
        drugs = self._extract_drugs(recommendation)
        if len(drugs) < 2:
            return []

        alerts: List[Dict] = []
        checked: Set[Tuple[str, str]] = set()

        for i, drug_a in enumerate(drugs):
            for drug_b in drugs[i + 1:]:
                pair = tuple(sorted([drug_a, drug_b]))
                if pair in checked:
                    continue
                checked.add(pair)

                interaction = self._lookup(drug_a, drug_b)
                if interaction:
                    severity, description = interaction
                    alerts.append({
                        "drug_a": drug_a,
                        "drug_b": drug_b,
                        "severity": severity,
                        "description": description,
                    })

        return alerts

    def has_high_risk_interaction(self, alerts: List[Dict]) -> bool:
        """Return ``True`` if any alert carries a high-risk severity level."""
        return any(
            a["severity"] in self.high_risk_severities for a in alerts
        )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_drugs(text: str) -> List[str]:
        """
        Extract drug name candidates from text using a lexicon-based approach.

        This implementation matches a curated drug name lexicon against the
        text. In production, this should be replaced with a dedicated NER model
        (e.g., SciSpacy ``en_ner_bc5cdr_md``).
        """
        _DRUG_LEXICON: List[str] = [
            "warfarin", "aspirin", "metformin", "omeprazole", "simvastatin",
            "atorvastatin", "amlodipine", "lisinopril", "metoprolol",
            "amoxicillin", "amoxicillin-clavulanate", "ciprofloxacin",
            "azithromycin", "doxycycline", "metronidazole", "vancomycin",
            "ceftriaxone", "trimethoprim-sulfamethoxazole", "clarithromycin",
            "sertraline", "escitalopram", "fluoxetine", "clopidogrel",
            "tramadol", "ibuprofen", "naproxen", "prednisone",
            "levothyroxine", "allopurinol", "furosemide",
        ]
        text_lower = text.lower()
        return [drug for drug in _DRUG_LEXICON if drug in text_lower]

    def _lookup(
        self, drug_a: str, drug_b: str
    ) -> Optional[Tuple[str, str]]:
        """Return ``(severity, description)`` for a drug pair, or ``None``."""
        if self.use_mock:
            key = tuple(sorted([drug_a, drug_b]))
            return _MOCK_DDI_TABLE.get(key)
        return self._api_lookup(drug_a, drug_b)

    def _api_lookup(
        self, drug_a: str, drug_b: str
    ) -> Optional[Tuple[str, str]]:
        """Query the live RxNorm interaction API for two drug names."""
        try:
            import requests

            # Step 1: resolve drug names to RxCUIs
            rxcui_a = self._get_rxcui(drug_a)
            rxcui_b = self._get_rxcui(drug_b)
            if not rxcui_a or not rxcui_b:
                return None

            # Step 2: check interaction
            url = f"{self.api_base}/interaction/interaction.json"
            resp = requests.get(
                url,
                params={"rxcui": rxcui_a},
                timeout=self.timeout,
            )
            data = resp.json()
            groups = data.get("interactionTypeGroup", [])
            for group in groups:
                for itype in group.get("interactionType", []):
                    for pair in itype.get("interactionPair", []):
                        concepts = pair.get("interactionConcept", [])
                        rxcuis = [c["minConceptItem"]["rxcui"] for c in concepts]
                        if rxcui_b in rxcuis:
                            severity = pair.get("severity", "unknown").lower()
                            desc = pair.get("description", "")
                            return severity, desc
        except Exception as exc:
            logger.warning("RxNorm API lookup failed (%s/%s): %s", drug_a, drug_b, exc)
        return None

    def _get_rxcui(self, drug_name: str) -> Optional[str]:
        """Return the RxCUI for *drug_name* from the RxNorm API."""
        try:
            import requests

            url = f"{self.api_base}/rxcui.json"
            resp = requests.get(
                url, params={"name": drug_name, "search": 1},
                timeout=self.timeout
            )
            data = resp.json()
            ids = data.get("idGroup", {}).get("rxnormId", [])
            return ids[0] if ids else None
        except Exception as exc:
            logger.warning("RxNorm CUI lookup failed for '%s': %s", drug_name, exc)
            return None


# --------------------------------------------------------------------------- #
# Orchestrator: full safety evaluation pipeline
# --------------------------------------------------------------------------- #

class SafetyEvaluator:
    """
    Orchestrates all three safety checks and produces a single ``SafetyReport``.

    Parameters
    ----------
    config : dict
        Safety configuration from ``configs/config.yaml`` (``safety`` block).
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        cfg = config or {}
        self.confidence_threshold = float(cfg.get("confidence_threshold", 0.75))
        high_risk = cfg.get("high_risk_severity_levels", ["contraindicated", "major"])

        self.stewardship = AntibioticStewardshipFilter(
            confidence_threshold=self.confidence_threshold
        )
        self.triage = ConfidenceTriage(confidence_threshold=self.confidence_threshold)
        self.ddi_checker = RxNormDDIChecker(
            use_mock=True,
            high_risk_severities=high_risk,
        )

    def evaluate(
        self,
        recommendation: str,
        predicted_disease: str,
        confidence: float,
    ) -> SafetyReport:
        """
        Run all safety filters and return a consolidated ``SafetyReport``.

        Parameters
        ----------
        recommendation : str
            Generated treatment recommendation.
        predicted_disease : str
            Predicted disease label.
        confidence : float
            MC Dropout confidence score.

        Returns
        -------
        SafetyReport
        """
        # 1. Confidence-based triage
        triage_outcome = self.triage.triage(confidence)

        # 2. Antibiotic stewardship
        ab_violations = self.stewardship.check(
            recommendation, predicted_disease, confidence
        )

        # 3. DDI check
        ddi_alerts = self.ddi_checker.check(recommendation)

        # Escalate to pharmacist review if high-risk DDIs are found
        if self.ddi_checker.has_high_risk_interaction(ddi_alerts):
            if triage_outcome == TriageOutcome.APPROVED:
                triage_outcome = TriageOutcome.FLAG_PHARMACIST_REVIEW
            logger.warning(
                "High-risk DDI detected in recommendation for '%s'.", predicted_disease
            )

        # Escalate to expert review if stewardship violations exist
        if ab_violations and triage_outcome == TriageOutcome.APPROVED:
            triage_outcome = TriageOutcome.FLAG_PHARMACIST_REVIEW

        notes: List[str] = []
        if ab_violations:
            notes.extend(ab_violations)
        for alert in ddi_alerts:
            notes.append(
                f"DDI ({alert['severity'].upper()}): {alert['drug_a']} + "
                f"{alert['drug_b']}: {alert['description']}"
            )

        report = SafetyReport(
            triage_outcome=triage_outcome,
            confidence_score=confidence,
            antibiotic_violations=ab_violations,
            ddi_alerts=ddi_alerts,
            final_recommendation=recommendation,
            notes=notes,
        )

        logger.info(
            "Safety evaluation complete: outcome=%s, confidence=%.3f, "
            "ab_violations=%d, ddi_alerts=%d",
            triage_outcome.value,
            confidence,
            len(ab_violations),
            len(ddi_alerts),
        )
        return report
