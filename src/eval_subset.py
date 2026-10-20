"""Пересчёт метрик бустингов на том же подмножестве val/test, что и у TabPFN (без переобучения).

Берёт results/results_gbm.csv и results/preds/*.npz (вероятности на полных val/test),
выбирает те же объекты функцией subset_idx и пишет results/results_gbm_sub{V}_{T}.csv.

Запуск (локально):
  python src/eval_subset.py --splits data/splits --out results --val-n 5000 --test-n 10000
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import compute_metrics, subset_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-n", type=int, default=None)
    ap.add_argument("--test-n", type=int, default=None)
    args = ap.parse_args()

    yv_all = pd.read_parquet(os.path.join(args.splits, "val.parquet"), columns=["flag"])["flag"].to_numpy()
    yt_all = pd.read_parquet(os.path.join(args.splits, "test.parquet"), columns=["flag"])["flag"].to_numpy()
    iv, it = subset_idx(len(yv_all), args.val_n, "val"), subset_idx(len(yt_all), args.test_n, "test")
    yv, yt = yv_all[iv], yt_all[it]
    print(f"val {len(iv)} (дефолтов {int(yv.sum())}), test {len(it)} (дефолтов {int(yt.sum())})")

    res = pd.read_csv(os.path.join(args.out, "results_gbm.csv"))
    rows = []
    for _, r in res.iterrows():
        path = os.path.join(args.out, "preds", r["run_id"].replace("|", "__") + ".npz")
        if not os.path.exists(path):
            print("нет файла предсказаний:", path)
            continue
        z = np.load(path)
        met = compute_metrics(yv, z["val"][iv], yt, z["test"][it])
        row = r.to_dict(); row.update(met); row["val_n"] = len(iv); row["test_n"] = len(it)
        rows.append(row)
    out = os.path.join(args.out, f"results_gbm_sub{len(iv)}_{len(it)}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("saved", out, len(rows), "строк")


if __name__ == "__main__":
    main()
