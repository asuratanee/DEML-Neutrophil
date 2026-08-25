# DEML-Neutrophil

**DEG-Enhanced Machine Learning (DEML) Framework for Cytokine-Primed Human Neutrophil Analysis**

This repository contains the analysis pipeline accompanying the manuscript:

> Suratanee et al. (2026). Complementary Transcriptional and Network-Level Functional Programs in Cytokine-Primed Human Neutrophils Revealed by DEG-Enhanced Machine Learning. (submitted).

---

## Overview

The DEML framework integrates differential gene expression (DEG) analysis with weighted gene co-expression network embeddings and ensemble machine learning to reveal network-level biological programs in cytokine-primed human neutrophils that are inaccessible to expression-based analysis alone.

**Pipeline steps:**

1. Label preparation (DESeq2 results -> gene labels)
2. WGCNA weighted co-expression network construction
3. Node2Vec network embedding (512-dimensional features)
4. Voting model training for whole-genome gene scoring (CNN, MLP, XGBoost, and LightGBM)
5. Whole-genome ensemble vote probability prediction (XGBoost)
6. Preranked GSEA (MSigDB Hallmark and KEGG)

---

## Important Note on Example Data

> The example data is provided in `data/example/` is a small subset of the original data. 
> The paths can be updated in the `config.py` file.

---

## Requirements

- Python 3.10+
- R 4.x with WGCNA package (v1.73, required for Step 2, called via rpy2)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install R dependencies (run in R):

```r
if (!require("BiocManager")) install.packages("BiocManager")
BiocManager::install("WGCNA")
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/asuratanee/DEML-Neutrophil.git
cd DEML-Neutrophil
```

### 2. Prepare example data

Place the following files in `data/example/Data/`:
- `Normalised_counts_GMT.csv`
- `GMCSFvUT_DEGs.csv`
- `TNFvUT_DEGs.csv`
- `GMTvUT_DEGs.csv`

### 3. Configure paths

Open `config.py` and set:

```python
DATA_BACKEND = "local"
LOCAL_ROOT   = "./data/example"
```

Or if using Google Cloud Storage:

```python
DATA_BACKEND = "gcs"
GCS_BUCKET   = "gs://your-bucket/your-project"
```

### 4. Run the pipeline

```bash
# Full pipeline (train all classifiers)
python main.py

# From a specific step
python main.py --step 3

# Single step only
python main.py --only 4

# Train a specific classifier only (cnn, mlp, xgboost, lightgbm)
python main.py --only 4 --classifier xgboost
```

## Citation

If you use this code, please cite:

> Suratanee et al. (2026). Complementary Transcriptional and Network-Level Functional Programs in Cytokine-Primed Human Neutrophils Revealed by DEG-Enhanced Machine Learning. (submitted).

---
