"""Пересобирает таблицу результатов TabPFN из сохранённых вероятностей (results/preds/tabpfn25__*.npz).

Нужен, если results_tabpfn.csv потерян или неполон. Метрики считаются теми же функциями, что и
при обучении (common.compute_metrics) по меткам val/test из data/splits. Время обучения и
предсказания в .npz не хранится, поэтому столбцы t_fit и t_pred остаются пустыми.

Запуск:
  python src/rebuild_tabpfn_csv.py --splits data/splits --preds results/preds --out results/results_tabpfn_rebuilt.csv
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

from common import compute_metrics, subset_idx

PAT = re.compile(r"tabpfn25__train_n(\d+)_p([\d.]+|nat)_s(\d+)__fs(\w+?)__sub(\d+)_(\d+)\.npz$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    yv_all = pd.read_parquet(os.path.join(args.splits, "val.parquet"), columns=["flag"])["flag"].to_numpy()
    yt_all = pd.read_parquet(os.path.join(args.splits, "test.parquet"), columns=["flag"])["flag"].to_numpy()
    man = pd.read_csv(os.path.join(args.splits, "manifest.csv"))
    rate_of = {os.path.basename(f): r for f, r in zip(man["file"], man["default_rate"])}

    rows = []
    for path in sorted(glob.glob(os.path.join(args.preds, "tabpfn25__*.npz"))):
        m = PAT.search(os.path.basename(path))
        if not m:
            print("пропущен (имя не распознано):", os.path.basename(path))
            continue
        n, rate, seed, fs, vn, tn = int(m[1]), m[2], int(m[3]), m[4], int(m[5]), int(m[6])
        z = np.load(path)
        iv, it = subset_idx(len(yv_all), vn, "val"), subset_idx(len(yt_all), tn, "test")
        assert len(z["val"]) == len(iv) and len(z["test"]) == len(it), f"длины не совпали: {path}"
        met = compute_metrics(yv_all[iv], z["val"], yt_all[it], z["test"])
        tag = "nat" if rate == "nat" else rate
        train_rate = rate_of.get(f"train_n{n}_p{tag}_s{seed}.parquet", np.nan)
        rows.append(dict(run_id=f"tabpfn25|train_n{n}_p{rate}_s{seed}|fs{fs}|sub{vn}_{tn}", model="tabpfn25",
                         n=n, rate=rate, seed=seed, feature_set=fs, val_n=vn, test_n=tn,
                         train_default_rate=train_rate, status="ok", t_fit=np.nan, t_pred=np.nan, **met))
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"saved {args.out}: {len(df)} строк; подмножества: {sorted(df.groupby(['val_n','test_n']).size().to_dict().items())}")


if __name__ == "__main__":
    main()
