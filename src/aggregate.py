"""Aggregate credit-history records to ONE ROW PER APPLICATION (id).

Input : train_data_*.pq (many rows per id, one row = one past credit of the client)
        train_target.csv (id, flag)
Output: one parquet file, one row per id, plus features_meta.json with the list of
        categorical columns (for CatBoost / LightGBM / TabPFN).

Notes on the data (see description.xlsx of the competition):
  * pre_* numeric columns are *binned with random bucket numbers*, so the order of the
    numbers carries no meaning. We therefore never take mean/max/min of them; we use
    only first / last value and the number of distinct values, and treat them as
    categorical.
  * enc_* columns are categorical codes.
  * enc_paym_11, enc_paym_20, enc_paym_24 are coded with a +1 shift (values 1..4
    instead of 0..3). We shift them back so that all enc_paym_* share one dictionary.
  * rn is the order of the credit in the client's history (larger = opened later).

Usage:
  python src/aggregate.py --input data/raw --target data/raw/train_target.csv \
      --out data/agg.parquet [--files 0 1 2 3]
"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd

PAYM_SHIFTED = (11, 20, 24)          # columns coded with +1 shift
PAYM_WINDOWS = {"recent": (0, 3), "mid": (3, 12), "old": (12, 25)}  # months 0..24
PAYM_CODES = (0, 1, 2, 3)


def load_file(path):
    df = pd.read_parquet(path)
    df = df.sort_values(["id", "rn"], kind="stable").reset_index(drop=True)
    # undo the +1 shift of three payment columns
    for k in PAYM_SHIFTED:
        c = f"enc_paym_{k}"
        if df[c].min() == 1 and df[c].max() == 4:
            df[c] = df[c] - 1
    # all payment columns must now share one dictionary 0..3 (fail loudly otherwise)
    for k in range(25):
        c = f"enc_paym_{k}"
        lo, hi = df[c].min(), df[c].max()
        if lo < 0 or hi > 3:
            raise ValueError(f"{os.path.basename(path)}: {c} has values {lo}..{hi}, "
                             f"expected 0..3 - check the +1 shift")
    return df


def aggregate_one(df):
    cols = list(df.columns)
    paym_cols = [f"enc_paym_{k}" for k in range(25)]
    pre_cols = [c for c in cols if c.startswith("pre_")]
    flag_cols = [c for c in cols if c.startswith("is_zero_")] + ["pclose_flag", "fclose_flag"]
    enc_loan_cols = [c for c in cols if c.startswith("enc_loans_")]

    g = df.groupby("id", sort=True)
    idx = g.size().index
    out = {}  # filled with numpy arrays, converted to DataFrame once (no fragmentation)
    out["n_loans"] = g.size().astype("int16").values

    # 1. flags: how many credits have the flag and the share
    s = g[flag_cols].sum().astype("int16")
    m = g[flag_cols].mean().astype("float32")
    for c in flag_cols:
        out[f"{c}__sum"] = s[c].values
        out[f"{c}__mean"] = m[c].values

    # 2. categorical codes of the credit: counts per code
    for c in enc_loan_cols:
        for v in sorted(df[c].unique()):
            out[f"{c}__cnt{v}"] = (
                (df[c] == v).groupby(df["id"]).sum().astype("int16").values
            )
        out[f"{c}__nunique"] = g[c].nunique().astype("int8").values

    # 3. payment statuses: counts of each code in three windows of months
    arr = df[paym_cols].to_numpy()
    tmp = pd.DataFrame({"id": df["id"].values})
    for wname, (lo, hi) in PAYM_WINDOWS.items():
        for v in PAYM_CODES:
            tmp[f"paym_{wname}__cnt{v}"] = (arr[:, lo:hi] == v).sum(axis=1).astype("int16")
    pa = tmp.groupby("id", sort=True).sum()
    for c in pa.columns:
        out[c] = pa[c].astype("int16").values
    for wname in PAYM_WINDOWS:
        tot = sum(pa[f"paym_{wname}__cnt{v}"] for v in PAYM_CODES).replace(0, np.nan)
        # share of non-zero statuses among all observed months of the window
        out[f"paym_{wname}__share_nonzero"] = (
            1 - pa[f"paym_{wname}__cnt0"] / tot
        ).astype("float32").values

    # 4. binned numeric columns: first, last and number of distinct values
    first = g[pre_cols].first()
    last = g[pre_cols].last()
    nun = g[pre_cols].nunique()
    for c in pre_cols:
        out[f"{c}__first"] = first[c].astype("int16").values
        out[f"{c}__last"] = last[c].astype("int16").values
        out[f"{c}__nunique"] = nun[c].astype("int8").values

    # 5. last (most recent) credit: its three latest payment statuses
    last_paym = g[paym_cols[:3]].last()
    for c in paym_cols[:3]:
        out[f"last_loan__{c}"] = last_paym[c].astype("int8").values

    out = pd.DataFrame(out, index=idx).reset_index()
    return out


def combine(parts):
    """Concatenate per-file results.

    A category code may appear only in later files (new product type, status ...).
    Its count column then exists only in some parts; for the other parts the true
    count is 0, not NaN. Any other column must be complete.
    """
    all_cols = list(dict.fromkeys(c for p in parts for c in p.columns))
    partial = [c for c in all_cols if any(c not in p.columns for p in parts)]
    agg = pd.concat(parts, ignore_index=True)[all_cols]
    for c in partial:
        if "__cnt" not in c:
            raise ValueError(f"column {c} is missing in some files and is not a count")
        agg[c] = agg[c].fillna(0).astype("int16")
    if partial:
        print("columns that appear only in some files (filled with 0 elsewhere):")
        for c in partial:
            print("   ", c)
    bad = agg.columns[agg.isna().any()].tolist()
    if bad:
        raise ValueError(f"unexpected NaN in columns: {bad}")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder with train_data_*.pq")
    ap.add_argument("--target", required=True, help="train_target.csv")
    ap.add_argument("--out", required=True, help="output parquet")
    ap.add_argument("--files", nargs="*", type=int, default=None,
                    help="numbers of train_data_N.pq to use (default: all found)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.input, "train_data_*.pq")),
                   key=lambda p: int(re.findall(r"(\d+)\.pq$", p)[0]))
    if args.files is not None:
        paths = [p for p in paths
                 if int(re.findall(r"(\d+)\.pq$", p)[0]) in set(args.files)]
    if not paths:
        raise SystemExit("no train_data_*.pq found")

    parts = []
    for p in paths:
        t0 = time.time()
        part = aggregate_one(load_file(p))
        parts.append(part)
        print(f"{os.path.basename(p)}: {len(part)} applications, "
              f"{part.shape[1]} columns, {time.time() - t0:.0f}s", flush=True)

    agg = combine(parts)
    target = pd.read_csv(args.target)
    agg = agg.merge(target, on="id", how="left")
    assert agg["flag"].notna().all(), "some ids have no target"

    # drop constant columns
    const = [c for c in agg.columns if c not in ("id", "flag") and agg[c].nunique() <= 1]
    agg = agg.drop(columns=const)

    cat_cols = [c for c in agg.columns
                if c.endswith("__first") or c.endswith("__last") or c.startswith("last_loan__")]
    meta = {
        "target": "flag",
        "id": "id",
        "categorical": cat_cols,
        "dropped_constant": const,
        "n_features": agg.shape[1] - 2,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    agg.to_parquet(args.out, index=False)
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nsaved {args.out}: {len(agg)} rows, {meta['n_features']} features, "
          f"default rate {agg['flag'].mean():.4f}")
    print(f"categorical: {len(cat_cols)}, dropped constant: {const}")


if __name__ == "__main__":
    main()
