# =============================================================
# src/train.py — Step 4
# Nested cross-validation for each classifier and condition.
# =============================================================

import warnings

import config as C
from src.trainers.models import get_spec
from src.trainers.nested_cv import run_nested_cv

warnings.filterwarnings("ignore")


def run(classifier="all"):
    names = C.CONFIG["models"] if classifier == "all" else [classifier]
    for name in names:
        spec = get_spec(name)
        for cond in C.CONFIG["conditions"]:
            run_nested_cv(spec, cond)
