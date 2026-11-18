# =============================================================
# src/wgcna_network.py — Step 2
# Signed co-expression network per group (UT and each cytokine condition).
# Adjacency a_ij = ((1 + r_ij) / 2)^beta with Pearson r on log2(normalized counts + 1),
# the same formula as WGCNA::adjacency(type = "signed"), computed in NumPy in blocks.
# Edges are kept when a_ij >= tau. With beta = 15 and tau = 0.4 this is r >= 0.882.
# The choice of beta is documented with R/pickSoftThreshold_UT.R (optional).
# =============================================================

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

import config as C
from src.utils.io_utils import exists, save_npz, load_npz, save_df
from src.label_preparation import load_expression

BETA, TAU = C.CONFIG["wgcna"]["beta"], C.CONFIG["wgcna"]["tau"]
PRIMARY_TAG = f"b{BETA}_t{str(TAU).replace('.', '')}"


def signed_edges(expr_gxs, beta, tau, block=2000):
    """Upper-triangle edges with signed adjacency ((1+r)/2)^beta >= tau.
    expr_gxs: genes × samples array. Genes with zero variance get no edges
    (WGCNA returns NA for their correlations)."""
    X = np.asarray(expr_gxs, dtype=np.float64)
    Xc = X - X.mean(axis=1, keepdims=True)
    nrm = np.sqrt((Xc ** 2).sum(axis=1))
    ok = nrm > 0
    Z = np.zeros_like(Xc)
    Z[ok] = Xc[ok] / nrm[ok, None]
    n = X.shape[0]
    rows, cols, wts = [], [], []
    for s in range(0, n, block):
        e = min(s + block, n)
        Cm = Z[s:e] @ Z.T                        # Pearson r for rows s..e-1
        np.clip(Cm, -1.0, 1.0, out=Cm)
        A = ((1.0 + Cm) / 2.0) ** beta
        ii, jj = np.nonzero(A >= tau)
        gi = ii + s
        m = (jj > gi) & ok[gi] & ok[jj]          # upper triangle, valid genes
        rows.append(gi[m].astype(np.int32)); cols.append(jj[m].astype(np.int32))
        wts.append(A[ii[m], jj[m]].astype(np.float32))
        del Cm, A
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)


def edges_to_csr(n, r_, c_, w_):
    A = sp.coo_matrix((np.concatenate([w_, w_]),
                       (np.concatenate([r_, c_]), np.concatenate([c_, r_]))), shape=(n, n)).tocsr()
    A.sum_duplicates()
    return A


def build_network(group, beta, tau, tag, log_norm=None):
    path = f"{C.NET_DIR}/{group}_{tag}.npz"
    if exists(path):
        d = load_npz(path)
        return d["rows"], d["cols"], d["weights"]
    if log_norm is None:
        log_norm = load_expression()[2]
    t0 = time.time()
    r_, c_, w_ = signed_edges(log_norm[C.CONFIG["samples"][group]].values, beta, tau)
    save_npz(path, rows=r_, cols=c_, weights=w_)
    print(f"  {group}: {len(w_):,} edges ({time.time() - t0:.0f}s) → {path}")
    return r_, c_, w_


def run():
    _, master_genes, log_norm = load_expression()
    n_genes = len(master_genes)
    r_cut = 2 * TAU ** (1 / BETA) - 1
    print(f"β = {BETA}, τ = {TAU} → edge if Pearson r ≥ {r_cut:.4f}")
    net_stats = []
    for group in ["UT"] + C.CONFIG["conditions"]:
        r_, c_, w_ = build_network(group, BETA, TAU, PRIMARY_TAG, log_norm)
        deg_ = np.bincount(np.concatenate([r_, c_]), minlength=n_genes)
        net_stats.append({"network": group, "n_samples": len(C.CONFIG["samples"][group]), "nodes": n_genes,
                          "edges": len(w_), "density": len(w_) / (n_genes * (n_genes - 1) / 2),
                          "isolated": int((deg_ == 0).sum()), "median_degree": float(np.median(deg_))})
    net_stats = pd.DataFrame(net_stats)
    save_df(net_stats, f"{C.NET_DIR}/network_stats_{PRIMARY_TAG}.csv")
    print(net_stats.to_string(index=False))
