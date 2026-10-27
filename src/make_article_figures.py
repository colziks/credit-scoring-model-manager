"""Три компактных рисунка для статьи (ширина одной колонки) из таблиц результатов.

  fig_auc_size.png   ROC AUC от размера обучающей выборки N (естественная доля дефолтов), 4 модели
  fig_err1_size.png  ошибка I рода при отказе 10% заявок от N (естественная доля), 4 модели
  fig_diff_defaults.png  разность AUC (TabPFN минус CatBoost) с 95% ДИ от числа дефолтов в обучении

Все таблицы должны относиться к одному подмножеству теста (по умолчанию 10 000 + 10 000).

Запуск:
  python src/make_article_figures.py --gbm results/results_gbm_sub10000_10000.csv \
     --tab results/results_tabpfn_rebuilt.csv --compare results/compare_tabpfn_catboost.csv --out figures
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"catboost": "tab:blue", "lightgbm": "tab:green", "xgboost": "tab:orange", "tabpfn25": "tab:red"}
NAMES = {"catboost": "CatBoost", "lightgbm": "LightGBM", "xgboost": "XGBoost", "tabpfn25": "TabPFN-2.5"}
SHARE = {"0.01": 0.01, "nat": 0.0361, "0.10": 0.10, "0.25": 0.25}
W, H = 3.4, 2.7
XT = [2000, 5000, 10000, 25000, 50000]


def prep(path, val_n, test_n):
    d = pd.read_csv(path)
    d["rate"] = d["rate"].astype(str)
    if "feature_set" in d:
        d = d[d["feature_set"].astype(str) == "all"]
    for col, v in (("val_n", val_n), ("test_n", test_n)):
        if col in d:
            d = d[d[col] == v]
    return d


def band(ax, d, col, ylabel):
    for m in ["xgboost", "lightgbm", "catboost", "tabpfn25"]:
        s = d[(d.model == m) & (d.rate == "nat")]
        if s.empty:
            continue
        g = s.groupby("n")[col].agg(["mean", "min", "max"]).reset_index()
        ax.plot(g["n"], g["mean"], marker="o", ms=3.5, lw=1.4, color=COLORS[m], label=NAMES[m])
        ax.fill_between(g["n"], g["min"], g["max"], color=COLORS[m], alpha=0.15)
    ax.set_xscale("log"); ax.set_xticks(XT); ax.set_xticklabels(["2k", "5k", "10k", "25k", "50k"])
    ax.minorticks_off(); ax.set_xlabel("размер обучающей выборки N"); ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, frameon=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbm", required=True)
    ap.add_argument("--tab", required=True)
    ap.add_argument("--compare", required=True)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--val-n", type=int, default=10000)
    ap.add_argument("--test-n", type=int, default=10000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plt.rcParams.update({"font.size": 8})
    d = pd.concat([prep(args.gbm, args.val_n, args.test_n), prep(args.tab, args.val_n, args.test_n)], ignore_index=True)

    fig, ax = plt.subplots(figsize=(W, H)); band(ax, d, "auc_test", "ROC AUC (тест)")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "fig_auc_size.png"), dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(W, H)); band(ax, d, "err1_q10", "ошибка I рода, отказ 10%")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "fig_err1_size.png"), dpi=220); plt.close(fig)

    c = pd.read_csv(args.compare); c["rate"] = c["rate"].astype(str)
    c["D"] = c["n"] * c["rate"].map(SHARE)
    c = c.sort_values("D")
    fig, ax = plt.subplots(figsize=(W, H))
    ax.axhline(0, color="gray", lw=0.8)
    ax.errorbar(c["D"], c["diff"], yerr=[c["diff"] - c["ci_lo"], c["ci_hi"] - c["diff"]],
                fmt="o", ms=3.5, lw=1.1, capsize=2, color="tab:red")
    ax.set_xscale("log"); ax.set_xlabel("число дефолтов в обучающей выборке")
    ax.set_ylabel("AUC: TabPFN-2.5 минус CatBoost"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "fig_diff_defaults.png"), dpi=220); plt.close(fig)
    print("saved:", sorted(f for f in os.listdir(args.out) if f.startswith("fig_")))


if __name__ == "__main__":
    main()
