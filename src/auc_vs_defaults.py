"""ROC AUC versus the number of defaults D in the training sample (marker = sample size N).

Evidence for the statement "quality is determined by the number of defaults, not by the size of the sample".
AUC is computed from the saved probabilities (all features, set 'all'):
  panel (a): boosting models on the full test (50 000 objects), every training sample N x share x seed;
  panel (b): TabPFN-2.5 and CatBoost on the same 10 000 test objects (subset used for TabPFN-2.5).
D = N x share of defaults in the training sample (from data/splits/manifest.csv).
Statistics printed: share of variance of AUC explained by log D and by log N alone, and by both together.

Usage (cmd):
  cd /d D:\\dev\\proj
  python src\\auc_vs_defaults.py
Options: --gbm-names catboost lightgbm xgboost   (file prefixes in results\\preds)
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

SUB = 10000


def subset_idx(n_total, k, part):
    seed = {"val": 101, "test": 202}[part]
    return np.sort(np.random.default_rng(seed).permutation(n_total)[:k])


def r2(y, X):
    X = np.c_[np.ones(len(y)), X]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return 1 - res.var() / y.var(), beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbm-names", nargs="+", default=["catboost", "lightgbm", "xgboost"])
    ap.add_argument("--preds", default="results/preds")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    os.makedirs(args.figures, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)

    yt = pd.read_parquet(os.path.join(args.splits, "test.parquet"), columns=["flag"])["flag"].to_numpy()
    it = subset_idx(len(yt), SUB, "test")
    man = pd.read_csv(os.path.join(args.splits, "manifest.csv"))
    man = man[man.part == "train"]

    rows = []
    for _, r in man.iterrows():
        m = re.search(r"train_n(\d+)_p([^_]+)_s(\d+)", r["file"])
        N, tag, seed = int(m.group(1)), m.group(2), int(m.group(3))
        D = int(round(N * r["default_rate"]))
        base = dict(N=N, share=tag, seed=seed, D=D)
        for g in args.gbm_names:
            f = os.path.join(args.preds, f"{g}__train_n{N}_p{tag}_s{seed}__fsall.npz")
            if os.path.exists(f):
                pt = np.load(f)["test"]
                rows.append(dict(base, model=g, auc_full=roc_auc_score(yt, pt), auc_sub=roc_auc_score(yt[it], pt[it])))
        f = os.path.join(args.preds, f"tabpfn25__train_n{N}_p{tag}_s{seed}__fsall__sub{SUB}_{SUB}.npz")
        if os.path.exists(f):
            pt = np.load(f)["test"]
            rows.append(dict(base, model="tabpfn25", auc_full=np.nan, auc_sub=roc_auc_score(yt[it], pt)))
    d = pd.DataFrame(rows)
    if d.empty:
        raise SystemExit("не найдено ни одного файла с вероятностями; проверьте --preds и --gbm-names")
    d.to_csv(os.path.join(args.results, "auc_vs_defaults.csv"), index=False, encoding="utf-8-sig")

    print("Запусков: " + ", ".join("%s %d" % (m, n) for m, n in d.model.value_counts().items()))
    print("\nДоля дисперсии AUC, объяснённая числом дефолтов D и размером N (регрессия на логарифмы):")
    print("%-10s %6s %10s %10s %12s   %s" % ("модель", "запусков", "R2(logD)", "R2(logN)", "R2(оба)", "коэффициенты при оба: logD, logN"))
    stat_rows = []
    for m, g in d.groupby("model"):
        col = "auc_full" if m != "tabpfn25" else "auc_sub"
        y = g[col].to_numpy()
        lD, lN = np.log(g.D.to_numpy(float)), np.log(g.N.to_numpy(float))
        a, _ = r2(y, lD)
        b, _ = r2(y, lN)
        c, beta = r2(y, np.c_[lD, lN])
        print("%-10s %6d %10.3f %10.3f %12.3f   %.4f, %.4f" % (m, len(g), a, b, c, beta[1], beta[2]))
        stat_rows.append(dict(model=m, runs=len(g), R2_logD=round(a, 3), R2_logN=round(b, 3), R2_both=round(c, 3),
                              coef_logD=round(beta[1], 4), coef_logN=round(beta[2], 4)))
    pd.DataFrame(stat_rows).to_csv(os.path.join(args.results, "auc_vs_defaults_stats.csv"), index=False, encoding="utf-8-sig")

    # ---------- figure
    marks = {2000: "o", 5000: "s", 10000: "^", 25000: "D", 50000: "v"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    ax = axes[0]
    g = d[d.model == args.gbm_names[0]]
    for N, mk in marks.items():
        s = g[g.N == N]
        if len(s):
            ax.scatter(s.D, s.auc_full, marker=mk, s=22, facecolor="white", edgecolor="black", linewidths=0.8, label=f"N = {N:,}".replace(",", " "))
    ax.set_xscale("log")
    ax.set_xlabel("Число дефолтов в обучающей выборке $D$")
    ax.set_ylabel("ROC AUC на тесте")
    ax.set_title("(а) %s, тест 50 000" % args.gbm_names[0].replace("catboost", "CatBoost"), fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[1]
    for m, color, face, lab in (("tabpfn25", "black", "black", "TabPFN-2.5"), (args.gbm_names[0], "black", "white", "CatBoost")):
        s = d[d.model == m]
        for N, mk in marks.items():
            t = s[s.N == N]
            if len(t):
                ax.scatter(t.D, t.auc_sub, marker=mk, s=22, facecolor=face, edgecolor=color, linewidths=0.8,
                           label=lab if N == min(s.N) else None)
    ax.set_xscale("log")
    ax.set_xlabel("Число дефолтов в обучающей выборке $D$")
    ax.set_title("(б) тест 10 000 объектов", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.figures, "fig_auc_defaults.png"), dpi=200)
    print("\nСохранено: figures\\fig_auc_defaults.png, results\\auc_vs_defaults.csv, results\\auc_vs_defaults_stats.csv")


if __name__ == "__main__":
    main()
