# =============================================================
# src/ensemble_prediction.py
# Applies all 25 trained XGBoost models to the full gene set
# (including stable genes) to generate ensemble vote probabilities
# used as gene ranking scores for GSEA.
# =============================================================

import numpy as np
import pandas as pd
import xgboost as xgb
from config import (FEATURES_PATH, MODELS_PATH, RESULTS_PATH,
                    DATA_PATH, CONDITIONS)


def load_labels(features_path, cond, all_genes):
    """Load labels and assign stable category to unlabeled genes."""
    label_df = pd.read_csv(
        f"{features_path}/labels_UpvDown_{cond}.csv", index_col=0)
    full = pd.DataFrame({'gene': all_genes})
    full = full.merge(
        label_df.reset_index().rename(columns={'index': 'gene'}),
        on='gene', how='left')
    full['category'] = 'stable'
    full.loc[full['label'] == 1, 'category'] = 'up'
    full.loc[full['label'] == 0, 'category'] = 'down'
    return full.set_index('gene')


def predict_all_genes(cond, features_path, models_path, results_path,
                      all_genes, registry):
    """Apply 25 models to all genes and compute ensemble vote probability."""
    print(f"\n{'─'*55}\n  {cond}\n{'─'*55}")

    embed = pd.read_csv(
        f"{features_path}/Embeddings512_UT_{cond}_WGCNA_concat.csv",
        index_col=0)
    common = np.intersect1d(all_genes, embed.index)
    X      = embed.loc[common].values.astype(np.float32)

    labels = load_labels(features_path, cond, common)
    cond_models = registry[registry['condition'] == cond]

    all_probs = []
    for _, row in cond_models.iterrows():
        model = xgb.XGBClassifier()
        model_path = row["model_path"]
        if not model_path.endswith(".json"):
            model_path = model_path + ".json"
        model.load_model(model_path)
        probs = model.predict_proba(X)[:, 1]
        all_probs.append(probs)

    probs_arr  = np.array(all_probs)
    vote_prob  = probs_arr.mean(axis=0)
    vote_std   = probs_arr.std(axis=0)

    # Directional ranking score for GSEA: range [-1, +1]
    direction  = np.where(vote_prob >= 0.5, 1, -1)
    rank_score = direction * np.abs(vote_prob - 0.5) * 2

    result = pd.DataFrame({
        'gene'      : common,
        'vote_prob' : vote_prob,
        'vote_std'  : vote_std,
        'rank_score': rank_score,
        'category'  : labels.loc[common, 'category'].values,
    })

    out_path = f"{results_path}/Ensemble_ALL_genes_XGBoost_{cond}.csv"
    result.to_csv(out_path, index=False)

    n_up     = (result['category'] == 'up').sum()
    n_down   = (result['category'] == 'down').sum()
    n_stable = (result['category'] == 'stable').sum()
    print(f"  Genes: {len(result):,} "
          f"(Up={n_up}, Down={n_down}, Stable={n_stable})")
    print(f"  Saved: {out_path}")

    return result


def run(features_path=FEATURES_PATH, models_path=MODELS_PATH,
        results_path=RESULTS_PATH, data_path=DATA_PATH):
    print("=" * 55)
    print("  Step 5: XGBoost Ensemble Prediction (All Genes)")
    print("=" * 55)

    norm     = pd.read_csv(f"{data_path}/Normalised_counts_GMT.csv", index_col=0)
    all_genes = norm.index.values

    registry = pd.read_csv(f"{results_path}/Model_registry_XGBoost.csv")
    print(f"Models loaded: {len(registry)}")

    for cond in CONDITIONS:
        predict_all_genes(cond, features_path, models_path,
                          results_path, all_genes, registry)

    print("\n✅ Ensemble prediction complete.")


if __name__ == "__main__":
    run()
