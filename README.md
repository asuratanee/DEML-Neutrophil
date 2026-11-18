# DEML-Neutrophil

**DEG-Enhanced Machine Learning (DEML) Framework for Cytokine-Primed Human Neutrophil Analysis**

This repository contains the analysis pipeline accompanying the manuscript:

> Apichat Suratanee, Steven Edwards, Helen Wright, and Kitiporn Plaimas. Complementary Transcriptional and Network-Level Functional Programs in Cytokine-Primed Human Neutrophils Revealed by DEG-Enhanced Machine Learning. *PLOS ONE* (under revision).

---

## Overview

The DEML framework integrates differential gene expression (DEG) analysis with signed co-expression network embeddings and machine learning. Classifiers are trained to separate up-regulated from down-regulated genes using each gene's position in the co-expression networks, and the resulting genome-wide scores are used as a ranking for preranked GSEA, in parallel with the conventional DEG ranking.

**Pipeline steps** (`python main.py`):

1. Label preparation (DESeq2 results → up / down labels)
2. Signed co-expression networks (WGCNA adjacency, computed in NumPy)
3. Node2Vec network embeddings (256 dimensions per network, 512 features per gene)
4. Nested cross-validation of seven classifiers (CNN, MLP, XGBoost, LightGBM, logistic regression, random forest, SVM)
5. Out-of-fold genome-wide scores, performance summary and pre-specified model selection
6. Preranked GSEA (MSigDB Hallmark and KEGG) for the DEG ranking and the DEML ranking of every classifier

The pipeline starts from DESeq2 results. The R script that produces them is provided separately as an optional step (see [Optional: differential expression with DESeq2](#optional-differential-expression-with-deseq2)).

---

## Evaluation design

- **Nested cross-validation.** Each condition uses 5 repeats × 5 stratified outer folds over the labeled genes. Optuna tunes hyperparameters on an inner 3-fold split of the outer training portion, and early stopping uses the inner folds only. The final model of each outer fold is refitted with a fixed number of trees or epochs, and the outer validation fold is used neither for fitting nor for tuning.
- **Undersampling** is applied to the outer training portion only, so validation folds keep the natural up/down ratio.
- **Out-of-fold scores.** Each labeled gene is scored only by models that did not see it during training. Unlabeled genes are scored by the mean of all outer-fold models.
- **Pre-specified model selection.** A classifier is eligible if its mean outer-fold AUC is within 0.03 of the best classifier in every condition. Among eligible classifiers, the one with the highest out-of-fold separation score (mean score of up-regulated minus down-regulated genes) is selected. With the full data this rule selected SVM.
- **Common gene universe.** The DEG and DEML rankings are tested by GSEA on the same set of genes.

The analysis configuration is written to `config_<version>.json` on the first run. If the configuration in `config.py` is later changed without changing `CONFIG["version"]`, the pipeline stops, so that results from different settings cannot be mixed.

---

## Important Note on Example Data

> **The example data provided in `data/example/` is a small subset of the original data (2,000 genes)** and is intended only to demonstrate that the pipeline runs correctly. Results obtained from example data will differ from those reported in the manuscript.
>
> To reproduce the exact results reported in the manuscript, download the full original data from figshare: **https://doi.org/10.6084/m9.figshare.33204360** and update the paths in `config.py` accordingly.

---

## Requirements

- Python 3.10+ (the manuscript used Python 3.13)
- R is not needed to run the pipeline. It is used only by the two optional scripts in `R/` (DESeq2, the manuscript used DESeq2 1.44, and WGCNA)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Optional R dependencies (run in R):

```r
if (!require("BiocManager")) install.packages("BiocManager")
BiocManager::install(c("DESeq2", "WGCNA"))
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/asuratanee/DEML-Neutrophil.git
cd DEML-Neutrophil
```

### 2. Example data

`data/example/Data/` contains:
- `Normalised_counts_GMT.csv`
- `GMCSFvUT_DEGs.csv`
- `TNFvUT_DEGs.csv`
- `GMTvUT_DEGs.csv`

These files are a representative subset of the original data available at **https://doi.org/10.6084/m9.figshare.33204360**.

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

The input files are read from `<root>/Data/`, and all outputs are written under `<root>/`.

### 4. Run the pipeline

```bash
# Quick check on the example data (about 15 minutes on a 2-core CPU without GPU)
python main.py --quick

# Full pipeline with the manuscript settings
python main.py

# From a specific step
python main.py --step 4

# Single step only
python main.py --only 4

# Train a specific classifier only
python main.py --only 4 --classifier SVM
```

`--quick` (QUICK_DEMO) uses 1 outer repeat instead of 5, 2 Optuna trials per fold, at most 5 epochs for the neural networks, smaller Node2Vec embeddings (32 dimensions, 2 walks per node, 2 epochs), 100 GSEA permutations and the Hallmark collection only. It writes to separate `*_quick_demo` folders and never overwrites results produced with the manuscript settings.

Steps can be interrupted and restarted. Each outer fold is saved as soon as it finishes, and finished folds, networks and embeddings are not recomputed.

---

## Repository Structure

```
DEML-Neutrophil/
├── README.md
├── requirements.txt
├── config.py                      # paths, locked analysis configuration, QUICK_DEMO
├── main.py                        # pipeline entry point
├── R/
│   ├── deseq2_DEA.R               # optional: differential expression (DESeq2)
│   └── pickSoftThreshold_UT.R     # optional: scale-free fit used to document beta
├── data/
│   └── example/
│       ├── Data/                  # example input files for the pipeline
│       └── DEA/                   # simulated raw counts and the study sample sheet for deseq2_DEA.R
└── src/
    ├── label_preparation.py       # Step 1: gene filter and up/down labels
    ├── wgcna_network.py           # Step 2: signed co-expression networks
    ├── node2vec_embedding.py      # Step 3: network embeddings
    ├── train.py                   # Step 4: nested cross-validation for each classifier
    ├── ensemble_prediction.py     # Step 5: out-of-fold genome-wide scores
    ├── model_selection.py         # Step 5: performance summary and model selection
    ├── gsea_analysis.py           # Step 6: preranked GSEA, DEG versus DEML
    ├── trainers/
    │   ├── nested_cv.py           # outer/inner CV loop, undersampling, metrics, fold records
    │   ├── models.py              # seven classifiers and their Optuna search spaces
    │   └── nn_worker.py           # runs one CNN/MLP fold in a fresh Python process
    └── utils/
        └── io_utils.py            # local and gs:// file access, seeds, configuration lock
```

### Main outputs

| File | Content |
|---|---|
| `Networks/network_stats_b15_t04.csv` | edges, density and isolated genes of each network |
| `Results/Fold_results_<model>_<condition>.csv` | metrics and selected hyperparameters of every outer fold |
| `Results/Summary_performance_all_models.csv` | mean ± SD of AUC, AUPR (up and down), F1 and MCC |
| `Results/Ensemble_OOF_<model>_<condition>.csv` | genome-wide out-of-fold scores (DEML ranking) |
| `Results/Separation_scores.csv`, `Results/Model_selection.json` | separation scores and the selected model |
| `GSEA/GSEA_all_rankings_full.csv` | GSEA results for the DEG ranking and every DEML ranking |
| `GSEA/GSEA_<model>_vs_DEG_comparison.csv` | shared and method-specific pathways |

---

## Hardware and Expected Runtimes (full data)

The manuscript analysis ran on Google Colab Enterprise. CNN and MLP used an NVIDIA T4 GPU (n1-standard-8). All other steps used CPU only (n2-standard-16, 16 vCPU, 64 GB RAM).

| Step | Description | Time (manuscript run) |
|------|-------------|----------------|
| 1 | Label preparation | < 1 min |
| 2 | Signed networks | 1 to 2 min per network |
| 3 | Node2Vec embeddings | about 7 min per network |
| 4 | Nested CV, CNN and MLP (GPU) | about 3 to 6 hours per model per condition |
| 4 | Nested CV, XGBoost and LightGBM (CPU) | 4 to 28 min per model per condition |
| 4 | Nested CV, LogReg, RandomForest and SVM (CPU) | shorter than the neural networks; not timed separately |
| 5 | Scores and model selection | a few minutes |
| 6 | GSEA (DEG + 7 DEML rankings × 3 conditions × gene set collections) | not timed separately |

Results of the tree models and the neural networks can differ in the last decimal places on other hardware or with a different number of CPU threads.

---

## Optional: differential expression with DESeq2

This step is not needed to run the pipeline, because `data/example/Data/` already contains the DESeq2 results. The script shows how those results were produced.

```bash
# Simulated example (2,000 genes, same 14-sample design), runs in under a minute
Rscript R/deseq2_DEA.R

# Real data: raw counts (GEO GSE288312 and GSE320016), sample sheet, output folder
Rscript R/deseq2_DEA.R Raw_counts_GMT.csv data/example/DEA/samples.csv DESeq2_output
```

The script fits one DESeq2 model to all 14 samples with `design = ~ Donor + Cytokine` and default settings, then tests each cytokine against untreated cells. It writes the three `*_DEGs.csv` files and `Normalised_counts_GMT.csv`, the input files of Step 1. The simulated counts are only a check that the script runs, and their output is written to `data/example/DEA/output/`, so the example inputs of the pipeline are not overwritten. `data/example/DEA/samples.csv` gives the sample-to-donor assignment of the study, with the same library names and donor codes as the GEO sample records. With the real counts the script gives 429/129 (GM-CSF), 374/67 (TNF-α) and 1,108/380 (GM+TNF) up/down-regulated genes at padj ≤ 0.05 and |log2FC| ≥ 1.

---

## Citation

If you use this code, please cite:

> Apichat Suratanee, Steven Edwards, Helen Wright, and Kitiporn Plaimas. Complementary Transcriptional and Network-Level Functional Programs in Cytokine-Primed Human Neutrophils Revealed by DEG-Enhanced Machine Learning. *PLOS ONE* (under revision).

---

## License

MIT License
