"""Парный бутстреп разности ROC AUC двух запусков на ОДНИХ И ТЕХ ЖЕ объектах теста.

Нужен, чтобы отличить реальное превосходство модели от шума: разность AUC двух
моделей на малом тесте может быть случайной. Объекты теста пересэмплируются
одинаково для обеих моделей (парный бутстреп), поэтому учитывается корреляция моделей.

Файлы предсказаний: results/preds/<run_id>.npz (поля val, test).
У бустингов массивы на ПОЛНОМ тесте (50 000), у TabPFN - на подмножестве; если длина
массива равна размеру полного теста, берётся то же подмножество (subset_idx),
что использовал TabPFN (--test-n).

Пример:
  python src/paired_bootstrap.py --splits data/splits --test-n 4000 \
     --a results/preds/tabpfn25__train_n10000_pnat_s0__fsall__sub4000_4000.npz \
     --b results/preds/catboost__train_n10000_pnat_s0__fsall.npz
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import subset_idx


def load_test_pred(path, n_full, test_n):
    p = np.load(path)["test"]
    if len(p) == n_full and test_n is not None and test_n < n_full:
        p = p[subset_idx(n_full, test_n, "test")]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--test-n", type=int, required=True, help="размер подмножества теста, на котором считался TabPFN")
    ap.add_argument("--a", required=True, help="npz первой модели (например, TabPFN)")
    ap.add_argument("--b", required=True, help="npz второй модели (например, CatBoost)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    y_full = pd.read_parquet(os.path.join(args.splits, "test.parquet"), columns=["flag"])["flag"].to_numpy()
    idx = subset_idx(len(y_full), args.test_n, "test")
    y = y_full[idx]
    pa = load_test_pred(args.a, len(y_full), args.test_n)
    pb = load_test_pred(args.b, len(y_full), args.test_n)
    assert len(pa) == len(pb) == len(y), f"длины не совпали: {len(pa)}, {len(pb)}, {len(y)}"

    auc_a, auc_b = roc_auc_score(y, pa), roc_auc_score(y, pb)
    rng = np.random.default_rng(args.seed)
    diffs = []
    for _ in range(args.boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        diffs.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"объектов {len(y)}, дефолтов {int(y.sum())}")
    print(f"AUC A = {auc_a:.4f}, AUC B = {auc_b:.4f}, разность A-B = {auc_a - auc_b:+.4f}")
    print(f"95% доверительный интервал разности (парный бутстреп, {len(diffs)} повторов): [{lo:+.4f}, {hi:+.4f}]")
    print("интервал не содержит 0: разность значима на уровне 5%" if lo > 0 or hi < 0
          else "интервал содержит 0: разность на этих данных не отличима от шума")
    print("Замечание: интервал учитывает только выбор объектов теста, но не разброс по обучающим подвыборкам (seed).")


if __name__ == "__main__":
    main()
