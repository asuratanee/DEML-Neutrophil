# =============================================================
# src/node2vec_embedding.py — Step 3
# Node2Vec with p = q = 1 (first-order random walk, next node drawn with
# probability proportional to edge weight), skip-gram Word2Vec on the walks.
# All start nodes are walked at once on a CSR matrix. workers = 1 and a
# deterministic hash make the embeddings reproducible.
# The 256-dimensional UT and condition embeddings are concatenated (512 features).
# =============================================================

import time

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

import config as C
from src.utils.io_utils import exists, save_df, load_df, stable_hash
from src.label_preparation import load_expression
from src.wgcna_network import build_network, edges_to_csr, BETA, TAU, PRIMARY_TAG

N2V = C.CONFIG["node2vec"]


def weighted_walks(A, num_walks, walk_length, seed):
    """First-order weighted random walks (Node2Vec with p = q = 1) from every non-isolated node."""
    rng = np.random.default_rng(seed)
    indptr, indices = A.indptr, A.indices
    w = A.data.astype(np.float64)
    deg_ = np.diff(indptr)
    nodes = np.where(deg_ > 0)[0]
    cw = np.cumsum(w)
    start_cum = np.concatenate([[0.0], cw])[indptr[:-1]]
    row_sum = np.zeros(A.shape[0]); row_sum[nodes] = cw[indptr[nodes + 1] - 1] - start_cum[nodes]
    walks = np.empty((num_walks * len(nodes), walk_length), dtype=np.int32)
    for k in range(num_walks):
        cur = rng.permutation(nodes)
        blk = slice(k * len(nodes), (k + 1) * len(nodes))
        walks[blk, 0] = cur
        for t in range(1, walk_length):
            target = start_cum[cur] + rng.random(len(cur)) * row_sum[cur]
            pos = np.searchsorted(cw, target, side="right")
            pos = np.clip(pos, indptr[cur], indptr[cur + 1] - 1)
            cur = indices[pos]
            walks[blk, t] = cur
    return walks, nodes


def embed_graph(A, gene_names, n2v, seed=None, workers=None):
    seed = n2v["seed"] if seed is None else seed
    workers = n2v["workers"] if workers is None else workers
    walks, nodes = weighted_walks(A, n2v["num_walks"], n2v["walk_length"], seed)
    sentences = [[str(x) for x in row] for row in walks]
    model = Word2Vec(sentences, vector_size=n2v["dimensions"], window=n2v["window"],
                     min_count=n2v["min_count"], sg=n2v["sg"], negative=n2v["negative"],
                     workers=workers, epochs=n2v["epochs"], seed=seed, hashfxn=stable_hash)
    E = np.vstack([model.wv[str(i)] for i in nodes]).astype(np.float32)
    if n2v["l2_normalize"]:
        E /= np.linalg.norm(E, axis=1, keepdims=True)
    return pd.DataFrame(E, index=np.asarray(gene_names)[nodes],
                        columns=[f"dim_{i}" for i in range(n2v["dimensions"])])


def concat_embeddings(E_ut, E_cond):
    common = E_ut.index.intersection(E_cond.index)
    out = pd.concat([E_ut.loc[common], E_cond.loc[common]], axis=1)
    out.columns = [f"UT_{c}" for c in E_ut.columns] + [f"COND_{c}" for c in E_cond.columns]
    return out


def emb512_path(cond):
    return f"{C.FEAT_DIR}/Embeddings{2 * N2V['dimensions']}_UT_{cond}_{PRIMARY_TAG}_seed{N2V['seed']}.parquet"


def primary_embedding(group, genes):
    path = f"{C.FEAT_DIR}/Embeddings{N2V['dimensions']}_{group}_{PRIMARY_TAG}_seed{N2V['seed']}.parquet"
    if exists(path):
        return load_df(path)
    t0 = time.time()
    r_, c_, w_ = build_network(group, BETA, TAU, PRIMARY_TAG)
    E = embed_graph(edges_to_csr(len(genes), r_, c_, w_), genes, N2V)
    save_df(E, path, index=True)
    print(f"  {group}: {E.shape} ({(time.time() - t0) / 60:.1f} min) → {path}")
    return E


def run():
    _, master_genes, _ = load_expression()
    genes = np.array(master_genes, dtype=str)
    emb = {g: primary_embedding(g, genes) for g in ["UT"] + C.CONFIG["conditions"]}
    for cond in C.CONFIG["conditions"]:
        X512 = concat_embeddings(emb["UT"], emb[cond])
        save_df(X512, emb512_path(cond), index=True)
        lab = load_df(f"{C.FEAT_DIR}/labels_UpvDown_{cond}.csv", index_col=0)
        n_lab = lab.index.isin(X512.index).sum()
        print(f"  {cond:<7} {X512.shape} | labeled genes with embedding: {n_lab:,} of {len(lab):,}")
