"""Check of the E_II formula and the curve "E_I vs E_II" for TabPFN-2.5 and CatBoost.

Protocol (the same as in the article):
  * the threshold is found on validation (quantile of predicted probabilities so that a share q
    of the most risky applications is rejected) and applied on test;
  * both models are evaluated on the same 10 000 test objects (subset used for TabPFN-2.5);
  * E_I = FN/(TP+FN), E_II = FP/(FP+TN), defaults are the positive class.

Outputs:
  results/error_check.csv   E_I, E_II (measured) and E_II by the formula for q = 5, 10, 20 %
  figures/fig_err_curve.png curve E_I vs E_II (mean over seeds) for q from 1 % to 50 %

Usage (cmd):
  cd /d D:\\dev\\proj
  python src\\error_curve.py --n 5000 --rate nat
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUB = 10000


def subset_idx(n_total, k, part):
    """Same rule as in common.py: sorted first k of a seeded permutation (101 = val, 202 = test)."""
    seed = {"val": 101, "test": 202}[part]
    return np.sort(np.random.default_rng(seed).permutation(n_total)[:k])


def errors(p_val, p_test, y_test, q):
    t = np.quantile(p_val, 1 - q)            # reject the share q of the most risky on validation
    reject = p_test >= t
    pos, neg = y_test == 1, y_test == 0
    fn = np.sum(~reject & pos)
    fp = np.sum(reject & neg)
    return fn / pos.sum(), fp / neg.sum(), reject.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="training sample size N")
    ap.add_argument("--rate", default="nat", help="share of defaults tag: nat, 0.01, 0.10, 0.25")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--gbm-name", default="catboost", help="file prefix of the boosting model")
    ap.add_argument("--preds", default="results/preds")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    os.makedirs(args.figures, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)

    yv = pd.read_parquet(os.path.join(args.splits, "val.parquet"), columns=["flag"])["flag"].to_numpy()
    yt = pd.read_parquet(os.path.join(args.splits, "test.parquet"), columns=["flag"])["flag"].to_numpy()
    iv, it = subset_idx(len(yv), SUB, "val"), subset_idx(len(yt), SUB, "test")
    yv_s, yt_s = yv[iv], yt[it]
    pi = yt_s.mean()
    print("ПРОВЕРКА подмножества теста: объектов %d, дефолтов %d (в статье 355), доля %.4f"
          % (len(yt_s), yt_s.sum(), pi))

    tag = f"train_n{args.n}_p{args.rate}"
    files = {
        "TabPFN-2.5": lambda s: f"tabpfn25__{tag}_s{s}__fsall__sub{SUB}_{SUB}.npz",
        "CatBoost": lambda s: f"{args.gbm_name}__{tag}_s{s}__fsall.npz",
    }

    def load(model, s):
        path = os.path.join(args.preds, files[model](s))
        if not os.path.exists(path):
            print("НЕТ ФАЙЛА:", path)
            return None
        z = np.load(path)
        pv, pt = z["val"], z["test"]
        if len(pv) != SUB:                      # boosting files hold the full 50 000 objects
            pv, pt = pv[iv], pt[it]
        return pv, pt

    qs_table = [0.05, 0.10, 0.20]
    qs_curve = np.unique(np.concatenate([np.linspace(0.01, 0.5, 50), qs_table]))
    rows, curves = [], {}
    for model in files:
        per_seed = []
        for s in args.seeds:
            pr = load(model, s)
            if pr is None:
                continue
            pv, pt = pr
            for q in qs_table:
                e1, e2, share = errors(pv, pt, yt_s, q)
                e2_formula = (share - (1 - e1) * pi) / (1 - pi)   # share = actual rejected share on test
                rows.append(dict(model=model, N=args.n, rate=args.rate, seed=s, q=q,
                                 share_rejected=round(share, 4), E_I=round(e1, 4),
                                 E_II=round(e2, 4), E_II_formula=round(e2_formula, 4)))
            per_seed.append([errors(pv, pt, yt_s, q)[:2] for q in qs_curve])
        if per_seed:
            curves[model] = np.mean(np.array(per_seed), axis=0)   # mean over seeds, shape (len(q), 2)

    if not rows:
        raise SystemExit("Не найдено ни одного файла с вероятностями, проверьте --n, --rate, --preds")
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(args.results, "error_check.csv"), index=False, encoding="utf-8-sig")
    summ = tab.groupby(["model", "q"])[["E_I", "E_II", "E_II_formula"]].mean().round(4)
    print("\nСредние по начальным состояниям (N=%d, доля дефолтов: %s):" % (args.n, args.rate))
    print(summ.to_string())
    print("Максимальное расхождение E_II и формулы (по фактической доле отказов): %.6f"
          % (tab.E_II - tab.E_II_formula).abs().max())
    print("Фактическая доля отказов на тесте при заданной q (среднее, мин., макс.):")
    print(tab.groupby("q").share_rejected.agg(["mean", "min", "max"]).round(4).to_string())

    fig, ax = plt.subplots(figsize=(6, 3.6))
    styles = {"TabPFN-2.5": "-", "CatBoost": "--"}
    for model, c in curves.items():
        ax.plot(c[:, 1] * 100, c[:, 0] * 100, styles[model], color="black", lw=1.3, label=model)
        for q in qs_table:
            j = int(np.argmin(np.abs(qs_curve - q)))
            ax.plot(c[j, 1] * 100, c[j, 0] * 100, "o", color="black", ms=3.5)
    for q in qs_table:
        j = int(np.argmin(np.abs(qs_curve - q)))
        ax.annotate(f"q = {int(q * 100)}%", (list(curves.values())[0][j, 1] * 100, list(curves.values())[0][j, 0] * 100),
                    textcoords="offset points", xytext=(6, 6), fontsize=7)
    ax.set_xlabel("Ошибка II рода, %")
    ax.set_ylabel("Ошибка I рода, %")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.figures, "fig_err_curve.png"), dpi=200)
    plt.close(fig)
    print("\nСохранено: results\\error_check.csv, figures\\fig_err_curve.png")


if __name__ == "__main__":
    main()
