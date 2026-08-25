# =============================================================
# main.py — DEML-Neutrophil Pipeline Entry Point
# =============================================================
# Usage:
#   python main.py                           # run full pipeline
#   python main.py --step 4                  # run from step 4
#   python main.py --only 4                  # run step 4 only
#   python main.py --only 4 --classifier xgboost
# =============================================================

import argparse
import sys
import os

from config import (FEATURES_PATH, NETWORKS_PATH, MODELS_PATH, RESULTS_PATH)
from src import label_preparation
from src import wgcna_network
from src import node2vec_embedding
from src import train as train_module
from src import ensemble_prediction
from src import gsea_analysis


def parse_args():
    parser = argparse.ArgumentParser(
        description="DEML-Neutrophil: DEG-Enhanced Machine Learning Pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--step", type=int, metavar="N",
        help="Run from step N to the end (e.g. --step 4)")
    group.add_argument(
        "--only", type=int, metavar="N",
        help="Run only step N (e.g. --only 4)")
    parser.add_argument(
        "--classifier", type=str, default="all",
        choices=["all", "cnn", "mlp", "xgboost", "lightgbm"],
        help="Classifier to train in step 4 (default: all)")
    return parser.parse_args()


def main():
    args = parse_args()

    STEPS = {
        1: ("Label Preparation",   label_preparation.run),
        2: ("WGCNA Network",       wgcna_network.run),
        3: ("Node2Vec Embedding",  node2vec_embedding.run),
        4: ("Classifier Training", lambda: train_module.run(classifier=args.classifier)),
        5: ("Ensemble Prediction", ensemble_prediction.run),
        6: ("GSEA Analysis",       gsea_analysis.run),
    }

    if args.only:
        steps_to_run = [args.only]
    elif args.step:
        steps_to_run = list(range(args.step, len(STEPS) + 1))
    else:
        steps_to_run = list(STEPS.keys())

    print("\n" + "=" * 55)
    print("  DEML-Neutrophil Pipeline")
    print("=" * 55)
    print(f"  Steps to run : {steps_to_run}")
    if 4 in steps_to_run:
        print(f"  Classifier   : {args.classifier}")
    print()

    # Auto-create local directories
    for path in [FEATURES_PATH, NETWORKS_PATH, MODELS_PATH, RESULTS_PATH]:
        if not path.startswith("gs://"):
            os.makedirs(path, exist_ok=True)
    for model_type in ['CNN', 'MLP', 'XGBoost', 'LightGBM']:
        subdir = os.path.join(MODELS_PATH, model_type)
        if not subdir.startswith("gs://"):
            os.makedirs(subdir, exist_ok=True)

    for step_num in steps_to_run:
        if step_num not in STEPS:
            print(f"⚠️  Step {step_num} does not exist. Skipping.")
            continue
        name, fn = STEPS[step_num]
        print(f"\n{'='*55}")
        print(f"  Step {step_num}: {name}")
        print(f"{'='*55}")
        try:
            fn()
        except Exception as e:
            print(f"\n❌ Step {step_num} failed: {e}")
            sys.exit(1)

    print("\n" + "=" * 55)
    print("  Pipeline complete.")
    print("=" * 55)


if __name__ == "__main__":
    main()
