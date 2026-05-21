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

<img width="900" height="550" alt="CLIN-LLM_Framework" src="https://github.com/user-attachments/assets/4dec2ffa-ba10-496b-849f-82a2ac644f5b" />


## 📊 Datasets & Open Access

This project utilizes two primary public datasets to train and evaluate the CLIN-LLM pipeline. To reproduce our findings, download the datasets from the official sources below and place them into your local `data/raw/` directory.

### 1. Symptom2Disease Dataset
* **Description:** Contains 1,200 patient records evenly distributed across 24 diagnostic classes (50 samples per class). Each record includes unstructured free-text symptom descriptions coupled with structured vital signs (temperature, heart rate, oxygen saturation).
* **Usage in Pipeline:** Ingested by `src/preprocessing.py` and used to train the `BioBERT` disease classification module in `src/classification.py`.
* **Data Access:** Available publicly via [Kaggle - Symptom2Disease Dataset](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease/data).

### 2. MedDialog Dataset
* **Description:** A large-scale medical discourse corpus consisting of approximately 260,000 real-world, English-language doctor-patient dialogues spanning diverse clinical contexts.
* **Usage in Pipeline:** Vectorized using `Biomedical Sentence-BERT` to form the dense semantic search index in `src/retrieval.py` for retrieval-augmented generation.
* **Official Publication:** Zeng G, Yang W, Ju Z, Yang Q, Wang S, Zhang R, et al. MedDialog: A Large-scale Medical Dialogue Dataset. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2020. p. 9241–9252. 
* **Data Access:** * [Official EMNLP Publication (ACL Anthology Link)](https://aclanthology.org/2020.emnlp-main.743/)
  * [Public Data Mirror (Hugging Face Datasets Hub)](https://huggingface.co/datasets/OpenMed/MedDialog)

---

### 🛡️ Data Processing & Ethical Safeguards
As detailed in the manuscript, the pipeline implements strict preprocessing workflows before executing model operations:
* **Anonymization:** MedDialog undergoes an automated Medical Named Entity Recognition (NER) pipeline to strip out geographic references, names, and temporal data to protect patient privacy.
* **Normalization:** Text records are systematically mapped to the Unified Medical Language System (UMLS) to align medical synonyms (e.g., matching "shortness of breath" to "dyspnea").
* **Class Balancing:** To eliminate downstream bias toward rare clinical conditions within the Symptom2Disease dataset, the Synthetic Minority Over-sampling Technique (SMOTE) is applied to lower-dimensional representations strictly within the training split.


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
git clone https://github.com/Mehedi16009/CLIN_LLM.git
cd CLIN_LLM
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

Full hyperparameter documentation is in [`wiki/02_Hyperparameter_Guide.md`](wiki/Hyperparameter_Guide.md).

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
@article{hasan2025clin,
  title={Clin-llm: A safety-constrained hybrid framework for clinical diagnosis and treatment generation},
  author={Hasan, Md Mehedi and Hossain, Md Abir and Sayem, Farman Hossain and Paul, Bikash Kumar and Rahman, Ziaur and Uddin, Mohammad Shorif and Mostafiz, Rafid},
  journal={arXiv preprint arXiv:2510.22609},
  year={2025}
}
```

## License

This repository is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

This work uses the MedDialog corpus, BioBERT (DMIS Lab, Korea University), and the Google FLAN-T5 model. Drug-drug interaction data is queried from the U.S. National Library of Medicine RxNorm REST API.

---

## Contact

Md Mehedi Hasan <br>
Mawlana Bhashani Science and Technology University <br>
- GitHub: [CLIN-LLM Repository](https://github.com/Mehedi16009/CLIN_LLM))
- Personal Website: [Portfolio](https://md-mehedi-hasan-resume.vercel.app/)
- Email: [mehedi.hasan.ict@mbstu.ac.bd](mehedi.hasan.ict@mbstu.ac.bd)
