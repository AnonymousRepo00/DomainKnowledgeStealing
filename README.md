# DomainKnowledgeStealing


# 🔍 Query-Efficient Domain Knowledge Stealing

This repository implements a query-efficient framework for black-box model extraction and knowledge Stealing
---

## 📁 Directory Overview

```
.
├── allresults.py              # Main script to evaluate all result files
├── figure2.py                 # Script to plot accuracy curves for selected datasets
├── requirements.txt           # Python dependencies
├── README.md                  # Project description (this file)

# ↓ Evaluation results from multiple experiments (Ours, EvoKD, Model Leeching, etc.)
├── results/                  
│   ├── llama_ours_medqa_generate_output_0.csv
│   ├── gemma_evokd_fpb_results.csv
│   └── ...

# ↓ Submodules and utility functions
├── coq/                       # Chunk-Oriented Questioning module
├── correct/                   # Correctness post-processing
├── data/                      # Data loading and utils
├── followup/                  # Follow-up prompting logic
├── student/                   # Student model definitions
├── teacher/                   # Teacher model API (e.g., GPT-4 API)
├── TeacherAns/                # Cached teacher answers for medical and financial domains

# ↓ Training and bash scripts
├── trainStudent.py           
├── train0.sh                 
├── train1.sh                 
```

---

## 📊 Datasets

The following QA datasets are supported:

| Dataset     | Domain     | Task Type             | Evaluation Metric |
|-------------|------------|-----------------------|-------------------|
| MedQA       | Medical    | Multiple-choice QA    | Accuracy          |
| PubMedQA    | Medical    | Yes/No QA             | Accuracy          |
| ChemProt    | Medical    | Relation classification | Accuracy       |
| FOMC        | Finance    | Sentiment classification | Accuracy       |
| HeadLine    | Finance    | Multilabel tagging    | Macro Accuracy    |
| FPB         | Finance    | Phrase-level sentiment | Accuracy         |

---

## 🧪 Methods

We evaluate the following model extraction strategies:

- **Ours**: Query-efficient framework (ours)
- **EvoKD**: Entropy-based query scheduling
- **Model Leeching**: Passive extraction using raw prompts sampled from unstructured QA data

- **Ablations**: `wocluster`, `wocoq`, `woFB` test the removal of each module

---

## 🚀 How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Evaluate all results:

```bash
python allresults.py
```


3. Train student (LoRA-finetuning):

```bash
bash train0.sh  # or bash train1.sh
```
