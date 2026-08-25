# =============================================================
# config.py — DEML-Neutrophil Pipeline Configuration
# =============================================================
# Set DATA_BACKEND to "gcs" if using Google Cloud Storage,
# or "local" if using downloaded data from figshare.
# =============================================================

DATA_BACKEND = "local"  # "gcs" or "local"

# ── GCS paths (used when DATA_BACKEND = "gcs") ───────────────
GCS_BUCKET   = "gs://your-bucket/your-project"
GCS_DATA     = f"{GCS_BUCKET}/Data"
GCS_FEATURES = f"{GCS_BUCKET}/Features"
GCS_NETWORKS = f"{GCS_BUCKET}/Networks"
GCS_MODELS   = f"{GCS_BUCKET}/Models"
GCS_RESULTS  = f"{GCS_BUCKET}/Results"

# ── Local paths (used when DATA_BACKEND = "local") ───────────
LOCAL_ROOT     = "./data/example"
LOCAL_DATA     = f"{LOCAL_ROOT}/Data"
LOCAL_FEATURES = f"{LOCAL_ROOT}/Features"
LOCAL_NETWORKS = f"{LOCAL_ROOT}/Networks"
LOCAL_MODELS   = f"{LOCAL_ROOT}/Models"
LOCAL_RESULTS  = f"{LOCAL_ROOT}/Results"

# ── Resolve paths based on backend ───────────────────────────
if DATA_BACKEND == "gcs":
    DATA_PATH     = GCS_DATA
    FEATURES_PATH = GCS_FEATURES
    NETWORKS_PATH = GCS_NETWORKS
    MODELS_PATH   = GCS_MODELS
    RESULTS_PATH  = GCS_RESULTS
else:
    DATA_PATH     = LOCAL_DATA
    FEATURES_PATH = LOCAL_FEATURES
    NETWORKS_PATH = LOCAL_NETWORKS
    MODELS_PATH   = LOCAL_MODELS
    RESULTS_PATH  = LOCAL_RESULTS

# ── Sample column names ───────────────────────────────────────
UT_COLS = ['UT_1', 'UT_2', 'UT_3', 'UT_5', 'UT_6']
CONDITIONS = {
    'GMCSF' : ['GMCSF_1', 'GMCSF_2', 'GMCSF_4'],
    'TNF'   : ['TNF_1',   'TNF_2',   'TNF_4'],
    'GM_TNF': ['GMT_1',   'GMT_2',   'GMT_3'],
}

# ── DESeq2 thresholds ─────────────────────────────────────────
PADJ_THRESHOLD   = 0.05
FC_THRESHOLD     = 1.0
STABLE_PADJ_MIN  = 0.5

# ── WGCNA parameters ─────────────────────────────────────────
WGCNA_BETA            = 15
WGCNA_WEIGHT_THRESHOLD = 0.4
WGCNA_NETWORK_TYPE    = "signed"

# ── Node2Vec parameters ───────────────────────────────────────
NODE2VEC_DIMENSIONS  = 256
NODE2VEC_WALK_LENGTH = 20
NODE2VEC_NUM_WALKS   = 10
NODE2VEC_WINDOW_SIZE = 5
NODE2VEC_P           = 1
NODE2VEC_Q           = 1
NODE2VEC_EPOCHS      = 10

# ── Training parameters ───────────────────────────────────────
RANDOM_SEED          = 42
N_REPEATS            = 5
N_FOLDS              = 5
N_TRIALS_TREE        = 50   # XGBoost, LightGBM
N_TRIALS_NN          = 10   # CNN, MLP
EARLY_STOPPING_ROUNDS = 20
EPOCHS               = 100
PATIENCE             = 10

# ── GSEA parameters ───────────────────────────────────────────
GSEA_MIN_SIZE        = 15
GSEA_MAX_SIZE        = 500
GSEA_PERMUTATIONS    = 1000
GSEA_FDR_THRESHOLD   = 0.25
GSEA_GENE_SETS = {
    'Hallmark': 'MSigDB_Hallmark_2020',
    'KEGG'    : 'KEGG_2021_Human',
}

# ── Candidate gene selection ──────────────────────────────────
CANDIDATE_QUANTILE = 0.05  # top and bottom 5%
