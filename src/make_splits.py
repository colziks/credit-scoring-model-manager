"""Time-based split and subsamples for the experiments.

Input : agg.parquet (from aggregate.py; one row per application, ids ordered by time)
Output (in --out folder):
  val.parquet, test.parquet         natural class balance, later in time than train
                                    (capped at --eval-size objects, random inside the window)
  subsamples/train_n{N}_p{P}_s{S}.parquet   training subsamples
  feature_sets.json                 nested random feature subsets (k features)
  manifest.csv                      what is in every file

Split by time (ids are ordered by application date):
  first  60% -> train pool   (subsamples are drawn from here)
  next   20% -> val          (tune thresholds of the manager here)
  last   20% -> test         (final numbers only)

Subsamples vary the three conditions the manager reacts to:
  * size N                          (e.g. 2k ... 50k; TabPFN-2.5 works up to 50k objects)
  * share of defaults P in train    (natural = share in the train pool)
  * number of features k            (nested random subsets, see feature_sets.json)
Every combination (N, P) is repeated with several seeds S to measure noise.

Usage:
  python src/make_splits.py --agg data/agg.parquet --out data/splits
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def draw(pool, n, rate, rng):
    y = pool["flag"].to_numpy()
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if rate is None:                                  # natural balance
        idx = rng.choice(len(pool), size=n, replace=False)
    else:
        n_pos = int(round(n * rate))
        n_neg = n - n_pos
        if n_pos > len(pos) or n_neg > len(neg):
            return None
        idx = np.concatenate([rng.choice(pos, n_pos, replace=False),
                              rng.choice(neg, n_neg, replace=False)])
        rng.shuffle(idx)
    return pool.iloc[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--eval-size", type=int, default=200000,
                    help="max objects in val and test (random sample inside the time window; "
                         "0 = keep all). TabPFN inference on millions of rows is very slow")
    ap.add_argument("--sizes", nargs="+", type=int,
                    default=[2000, 5000, 10000, 25000, 50000])
    ap.add_argument("--rates", nargs="+", default=["0.01", "nat", "0.10", "0.25"],
                    help="share of defaults in train subsample; 'nat' = natural")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-features", nargs="+", type=int, default=[20, 50, 100],
                    help="sizes of nested feature subsets (full set is added)")
    args = ap.parse_args()

    df = pd.read_parquet(args.agg).sort_values("id").reset_index(drop=True)
    meta_path = os.path.splitext(args.agg)[0] + "_meta.json"
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}

    n = len(df)
    i1, i2 = int(n * args.train_frac), int(n * (args.train_frac + args.val_frac))
    pool, val, test = df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]
    if args.eval_size and args.eval_size > 0:
        r = np.random.default_rng(777)
        cap = lambda d: d if len(d) <= args.eval_size else d.iloc[
            np.sort(r.choice(len(d), args.eval_size, replace=False))]
        val, test = cap(val), cap(test)
    os.makedirs(os.path.join(args.out, "subsamples"), exist_ok=True)
    val.to_parquet(os.path.join(args.out, "val.parquet"), index=False)
    test.to_parquet(os.path.join(args.out, "test.parquet"), index=False)

    rows = [
        dict(file="val.parquet", part="val", n=len(val), default_rate=val.flag.mean()),
        dict(file="test.parquet", part="test", n=len(test), default_rate=test.flag.mean()),
    ]
    nat = pool.flag.mean()
    print(f"train pool {len(pool)} (default {nat:.4f}), "
          f"val {len(val)} ({val.flag.mean():.4f}), test {len(test)} ({test.flag.mean():.4f})")

    for size in args.sizes:
        for r in args.rates:
            rate = None if r == "nat" else float(r)
            for seed in args.seeds:
                rng = np.random.default_rng(seed * 100003 + size)
                sub = draw(pool, size, rate, rng)
                if sub is None:
                    print(f"skip n={size} p={r}: not enough objects")
                    continue
                tag = "nat" if rate is None else f"{rate:.2f}"
                name = f"train_n{size}_p{tag}_s{seed}.parquet"
                sub.to_parquet(os.path.join(args.out, "subsamples", name), index=False)
                rows.append(dict(file=f"subsamples/{name}", part="train", n=size,
                                 default_rate=sub.flag.mean(), seed=seed))

    # nested random feature subsets: smaller subset is contained in the larger one
    feats = [c for c in df.columns if c not in ("id", "flag")]
    order = list(np.random.default_rng(12345).permutation(feats))
    fsets = {str(k): order[:k] for k in sorted(args.n_features) if k < len(feats)}
    fsets["all"] = feats
    json.dump({"feature_sets": fsets, "categorical": meta.get("categorical", [])},
              open(os.path.join(args.out, "feature_sets.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    pd.DataFrame(rows).to_csv(os.path.join(args.out, "manifest.csv"), index=False)
    print(f"saved {len(rows)} files to {args.out}; feature sets: {list(fsets)}")


if __name__ == "__main__":
    main()
