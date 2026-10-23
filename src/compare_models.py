"""Сравнение двух моделей по ячейкам сетки (N, доля дефолтов) на общем подмножестве теста.

Для каждой ячейки берутся все seed'ы, для которых есть файлы предсказаний обеих моделей.
Доверительный интервал разности ROC AUC - парный бутстреп по объектам теста: одна и та же
передвыборка объектов для всех seed'ов и обеих моделей, статистика - средняя по seed'ам
разность AUC. Интервал учитывает только выбор объектов теста; разброс по обучающим
подвыборкам показывает число seed'ов, где модель A лучше (столбец wins).

Пример:
  python src/compare_models.py --splits data/splits --preds results/preds --test-n 10000 \
      --a tabpfn25 --b catboost --out results/compare_tabpfn_catboost.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import subset_idx
from paired_bootstrap import load_test_pred

SIZES = [2000, 5000, 10000, 25000, 50000]
RATES = ["0.01", "nat", "0.10", "0.25"]
SEEDS = [0, 1, 2]


def fname(model, n, rate, seed, fs, sub):
    base = f"{model}__train_n{n}_p{rate}_s{seed}__fs{fs}"
    return base + (f"__sub{sub}" if model.startswith("tabpfn") else "") + ".npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--test-n", type=int, required=True)
    ap.add_argument("--val-n", type=int, default=None, help="для имени файла TabPFN (по умолчанию = --test-n)")
    ap.add_argument("--a", default="tabpfn25")
    ap.add_argument("--b", default="catboost")
    ap.add_argument("--fs", default="all")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    y_full = pd.read_parquet(os.path.join(args.splits, "test.parquet"), columns=["flag"])["flag"].to_numpy()
    y = y_full[subset_idx(len(y_full), args.test_n, "test")]
    sub = f"{args.val_n or args.test_n}_{args.test_n}"
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(y), len(y)) for _ in range(args.boot)]
    boots = [i for i in boots if y[i].min() != y[i].max()]

    rows = []
    for n in SIZES:
        for rate in RATES:
            pa, pb = [], []
            for s in SEEDS:
                fa = os.path.join(args.preds, fname(args.a, n, rate, s, args.fs, sub))
                fb = os.path.join(args.preds, fname(args.b, n, rate, s, args.fs, sub))
                if os.path.exists(fa) and os.path.exists(fb):
                    pa.append(load_test_pred(fa, len(y_full), args.test_n))
                    pb.append(load_test_pred(fb, len(y_full), args.test_n))
            if not pa:
                continue
            auc_a = np.array([roc_auc_score(y, p) for p in pa])
            auc_b = np.array([roc_auc_score(y, p) for p in pb])
            d = np.array([np.mean([roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]) for a, b in zip(pa, pb)])
                          for i in boots])
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append(dict(n=n, rate=rate, seeds=len(pa), auc_a=auc_a.mean(), auc_b=auc_b.mean(),
                             diff=(auc_a - auc_b).mean(), ci_lo=lo, ci_hi=hi,
                             wins=int((auc_a > auc_b).sum()), significant=bool(lo > 0 or hi < 0)))
            print(f"N={n:>6} доля={rate:>4}  A={auc_a.mean():.3f} B={auc_b.mean():.3f}  "
                  f"разность {rows[-1]['diff']:+.3f}  95% ДИ [{lo:+.3f}, {hi:+.3f}]  "
                  f"A лучше в {rows[-1]['wins']}/{len(pa)}", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("saved", args.out)


if __name__ == "__main__":
    main()
