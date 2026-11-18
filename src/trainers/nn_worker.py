# =============================================================
# src/trainers/nn_worker.py
# Runs one outer fold of a TensorFlow model (CNN or MLP) in a fresh Python process.
# Called by src/trainers/nested_cv.py; not meant to be run by hand.
# Usage: python -m src.trainers.nn_worker <model> <condition> <repeat> <fold>
# =============================================================

import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from src.trainers.nested_cv import run_one_fold, load_condition
from src.trainers.models import get_spec

if __name__ == "__main__":
    model_name, cond, rep, fold = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    spec = get_spec(model_name)
    print("RESULT::" + run_one_fold(spec, model_name, cond, rep, fold, load_condition(cond)))
