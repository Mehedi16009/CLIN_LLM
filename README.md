# CLIN-LLM: An AI Pipeline for Medical Diagnosis and Treatment Recommendation

[![PLOS ONE](https://img.shields.io/badge/Published-PLOS%20ONE-blue)](https://journals.plos.org/plosone/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## Framework Overview

CLIN-LLM is a multi-stage clinical AI pipeline that processes patient symptom input through four sequential components to produce a safe, evidence-grounded treatment recommendation.

**Stage 1: Disease Classification.** Free-text symptoms or structured vitals are preprocessed (lemmatisation, negation detection, UMLS mapping) and fed into a fine-tuned BioBERT model. Monte Carlo Dropout provides predictive entropy as a model confidence estimate. The framework achieves **98% classification accuracy** on the held-out test split.

**Stage 2: Retrieval-Augmented Generation.** High-confidence predictions are converted to a query vector using a Biomedical Sentence-BERT model. The top-k most similar dialogue snippets are retrieved from the MedDialog corpus via cosine similarity.

**Stage 3: Treatment Generation.** A fine-tuned FLAN-T5-Large model generates a treatment recommendation from a structured prompt combining the patient presentation, predicted diagnosis, and retrieved clinical dialogues.

**Stage 4: Safety Validation.** Post-generation filtering enforces antibiotic stewardship rules, checks drug-drug interactions via the RxNorm REST API, and routes low-confidence cases to a human-in-the-loop review state. This layer produces a **67% reduction in unsafe recommendations** compared to unfiltered generation.

<img width="750" height="600" alt="CLIN-LLM_Framework" src="https://github.com/user-attachments/assets/4dec2ffa-ba10-496b-849f-82a2ac644f5b" />

```
Patient Input
     │
     ▼
[Preprocessing] ─→ [BioBERT + MC Dropout]
                             │
              ┌──────────────┴──────────────┐
         Low confidence                High confidence
              │                             │
              ▼                             ▼
     [Expert Review Flag]        [Sentence-BERT Retrieval]
                                           │
                                           ▼
                                  [FLAN-T5 Generation]
                                           │
                                           ▼
                               [Safety Filters (DDI + Stewardship)]
                                           │
                              ┌────────────┴────────────┐
                         Pharmacist                 Final Treatment
                          Review                  Recommendation
```

## Repository Structure

```
CLIN-LLM-Replication/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── data/
│   ├── processed/           # Tokenised, SMOTE-rebalanced splits
│   └── raw/                 # Original symptom-disease CSV (not committed)
├── configs/
│   └── config.yaml          # All hyperparameters (model, training, safety)
├── notebooks/
│   └── clinical_llm_exploration.ipynb   # Exploratory notebook
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # Text cleaning, negation, UMLS, SMOTE
│   ├── classification.py    # BioBERT, FocalLoss, MCDropout, Trainer
│   ├── retrieval.py         # Sentence-BERT semantic search over MedDialog
│   ├── generation.py        # FLAN-T5 treatment recommendation
│   ├── safety_filters.py    # Stewardship rules, DDI, confidence triage
│   └── evaluation.py        # Metrics, confusion matrix, ROC/PR curves
├── wiki/
│   ├── 01_Data_Processing.md
│   ├── 02_Hyperparameter_Guide.md
│   └── 03_Deployment_Guide.md
└── CLIN-LLM_runner.py       # End-to-end CLI runner
```

## Installation

```bash
git clone https://github.com/your-org/CLIN-LLM-Replication.git
cd CLIN-LLM-Replication
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quickstart

### Full training pipeline

```bash
python CLIN-LLM_runner.py \
    --config configs/config.yaml \
    --data data/raw/symptom_disease.csv
```

This will preprocess the dataset, apply SMOTE rebalancing, fine-tune BioBERT with Focal Loss, and persist the best checkpoint to `outputs/models/biobert_best.pt`. Evaluation metrics and plots are saved to `outputs/results/`.

### Single-sample inference

```bash
python CLIN-LLM_runner.py \
    --config configs/config.yaml \
    --infer \
    --symptoms "persistent dry cough, high fever, shortness of breath for 4 days" \
    --checkpoint outputs/models/biobert_best.pt
```

### Evaluate a saved checkpoint

```bash
python CLIN-LLM_runner.py \
    --config configs/config.yaml \
    --eval-only \
    --data data/raw/symptom_disease.csv \
    --checkpoint outputs/models/biobert_best.pt
```

## Key Hyperparameters

| Component | Parameter | Value |
|---|---|---|
| BioBERT | Base model | `dmis-lab/biobert-v1.1` |
| BioBERT | Learning rate | 2e-5 |
| BioBERT | Epochs | 10 |
| BioBERT | Batch size | 32 |
| Focal Loss | Gamma | 2.0 |
| MC Dropout | Samples (T) | 30 |
| MC Dropout | Confidence threshold | 0.75 |
| Sentence-BERT | Base model | `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` |
| FLAN-T5 | Base model | `google/flan-t5-large` |
| FLAN-T5 | Beam width | 4 |

Full hyperparameter documentation is in [`wiki/02_Hyperparameter_Guide.md`](wiki/02_Hyperparameter_Guide.md).

## Results

| Metric | Value |
|---|---|
| Classification Accuracy | 98.0% |
| Weighted F1 | 0.978 |
| Reduction in Unsafe Recommendations | 67% |
| Retrieval Precision@5 | Reported in paper |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{clin-llm-2024,
  title   = {CLIN-LLM: An AI Pipeline for Medical Diagnosis and Treatment Recommendation},
  author  = {[Authors]},
  journal = {PLOS ONE},
  year    = {2024},
  doi     = {[DOI placeholder]}
}
```

## License

This repository is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

This work uses the MedDialog corpus, BioBERT (DMIS Lab, Korea University), and the Google FLAN-T5 model. Drug-drug interaction data is queried from the U.S. National Library of Medicine RxNorm REST API.
