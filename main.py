# =============================================================
# main.py — DEML-Neutrophil pipeline entry point
# =============================================================
# Usage:
#   python main.py                            # full pipeline, manuscript settings
#   python main.py --quick                    # QUICK_DEMO: fast check on the example data
#   python main.py --step 4                   # run from step 4
#   python main.py --only 4 --classifier SVM  # one step, one classifier
# =============================================================

import argparse
import os
import sys

MODELS = ["CNN", "MLP", "XGBoost", "LightGBM", "LogReg", "RandomForest", "SVM"]


def parse_args():
    parser = argparse.ArgumentParser(description="DEML-Neutrophil: DEG-Enhanced Machine Learning pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--step", type=int, metavar="N", help="Run from step N to the end")
    group.add_argument("--only", type=int, metavar="N", help="Run only step N")
    parser.add_argument("--classifier", type=str, default="all", choices=["all"] + MODELS,
                        help="Classifier to train in step 4 (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="QUICK_DEMO settings (fewer repeats, trials, epochs and permutations)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        os.environ["DEML_QUICK_DEMO"] = "1"      # read by config.py, also inherited by worker processes

    import config as C
    from src.utils.io_utils import check_config_lock
    from src import label_preparation, wgcna_network, node2vec_embedding, train as train_module
    from src import ensemble_prediction, model_selection, gsea_analysis

    steps = {
        1: ("Label preparation", label_preparation.run),
        2: ("WGCNA signed co-expression networks", wgcna_network.run),
        3: ("Node2Vec embeddings", node2vec_embedding.run),
        4: ("Nested cross-validation", lambda: train_module.run(classifier=args.classifier)),
        5: ("Out-of-fold scores and model selection", lambda: (ensemble_prediction.run(), model_selection.run())),
        6: ("GSEA (DEG and DEML rankings)", gsea_analysis.run),
    }
    if args.only:
        to_run = [args.only]
    elif args.step:
        to_run = list(range(args.step, len(steps) + 1))
    else:
        to_run = list(steps)

    print("\n" + "=" * 60)
    print("  DEML-Neutrophil pipeline")
    print("=" * 60)
    print(f"  Config version : {C.CONFIG['version']}" + ("  (QUICK_DEMO)" if C.QUICK_DEMO else ""))
    print(f"  Output root    : {C.OUT}")
    print(f"  Steps          : {to_run}")
    check_config_lock()

    for n in to_run:
        if n not in steps:
            print(f"Step {n} does not exist. Skipping.")
            continue
        name, fn = steps[n]
        print(f"\n{'=' * 60}\n  Step {n}: {name}\n{'=' * 60}")
        try:
            fn()
        except Exception as e:
            print(f"\nStep {n} failed: {e}")
            sys.exit(1)
    print("\n" + "=" * 60 + "\n  Pipeline complete.\n" + "=" * 60)


if __name__ == "__main__":
    main()
