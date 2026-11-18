# =============================================================
# src/utils/io_utils.py
# File I/O for local paths and gs:// paths (fsspec), seeds, config lock.
# Shared helpers used by all steps.
# =============================================================

import os
import json
import hashlib
import datetime

import numpy as np
import pandas as pd
import fsspec

import config as C


def _open(path, mode):
    if not path.startswith("gs://") and ("w" in mode):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return fsspec.open(path, mode)


def exists(path):
    fs, _, paths = fsspec.get_fs_token_paths(path)
    return fs.exists(paths[0])


def _json_default(o):
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)):    return bool(o)
    if isinstance(o, np.ndarray):     return o.tolist()
    raise TypeError(type(o))


def save_json(obj, path):
    with _open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)


def load_json(path):
    with _open(path, "r") as f:
        return json.load(f)


def save_npz(path, **arrays):
    with _open(path, "wb") as f:
        np.savez_compressed(f, **arrays)


def load_npz(path):
    with _open(path, "rb") as f:
        d = np.load(f, allow_pickle=False)
        return {k: d[k] for k in d.files}


def save_df(df, path, index=False):
    if path.endswith(".parquet"):
        with _open(path, "wb") as f:
            df.to_parquet(f, index=index)
    else:
        with _open(path, "w") as f:
            df.to_csv(f, index=index)


def load_df(path, **kw):
    if path.endswith(".parquet"):
        with _open(path, "rb") as f:
            return pd.read_parquet(f, **kw)
    with _open(path, "r") as f:
        return pd.read_csv(f, **kw)


def fold_seed(rep, fold):
    return C.CONFIG["cv"]["seed"] + 100 * rep + fold


def stable_hash(s):
    # deterministic replacement for Python's salted hash(); used by gensim to seed word vectors
    return int(hashlib.md5(str(s).encode()).hexdigest()[:8], 16)


def check_config_lock():
    """Write the configuration on the first run; on later runs refuse to continue if it was changed."""
    lock_path = f"{C.OUT}/config_{C.CONFIG['version']}.json"
    if exists(lock_path):
        saved = load_json(lock_path)
        saved_cfg = saved["config"]
        if json.dumps(saved_cfg, sort_keys=True) != json.dumps(json.loads(json.dumps(C.CONFIG)), sort_keys=True):
            raise RuntimeError(f"CONFIG differs from the locked version at {lock_path}. "
                               "Restore the locked values, or change CONFIG['version'] to start a new, separately reported analysis.")
        print(f"Config matches lock written {saved['locked_at']}")
    else:
        save_json({"locked_at": datetime.datetime.now().isoformat(timespec="seconds"), "config": C.CONFIG}, lock_path)
        print(f"Config locked → {lock_path}")
