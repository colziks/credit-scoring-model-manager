"""Профили моделей по результатам обучения.

Читает один или несколько файлов results_*.csv (колонка `model` различает модели),
строит графики в --figs и пишет сводные таблицы в --out.

ВАЖНО: все файлы должны быть посчитаны на ОДНИХ И ТЕХ ЖЕ объектах val/test.
Результаты TabPFN (подмножество) сравнивать только с results_gbm_sub{V}_{T}.csv,
который получается из eval_subset.py, а не с results_gbm.csv на полном test.

Запуск:
  python src/profiles.py --csv results/results_gbm.csv --out results --figs figures
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"catboost": "tab:blue", "lightgbm": "tab:green", "xgboost": "tab:orange",
          "tabpfn25": "tab:red"}
RATE_ORDER = ["0.01", "nat", "0.10", "0.25"]
Q = "q10"   # доля отказов для ошибок I/II рода в графиках


def load(paths):
    d = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    d["rate"] = d["rate"].astype(str)
    d["n_pos_train"] = (d["n"] * d["train_default_rate"]).round().astype(int)
    return d


def agg(d, keys, col):
    g = d.groupby(keys)[col]
    return pd.DataFrame({"mean": g.mean(), "min": g.min(), "max": g.max(), "std": g.std()}).reset_index()


def line_with_band(ax, a, x, model):
    a = a.sort_values(x)
    c = COLORS.get(model)
    ax.plot(a[x], a["mean"], marker="o", color=c, label=model)
    ax.fill_between(a[x], a["min"], a["max"], color=c, alpha=0.15)


def fig_auc_by_size(d, figs, fs):
    sub = d[d.feature_set == fs]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for ax, r in zip(axes, RATE_ORDER):
        for m in sorted(sub.model.unique()):
            a = agg(sub[(sub.model == m) & (sub.rate == r)], ["n"], "auc_test")
            line_with_band(ax, a, "n", m)
        ax.set_xscale("log"); ax.set_title("доля дефолтов в обучении: " + ("естеств." if r == "nat" else r))
        ax.set_xlabel("размер обучающей выборки N"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("ROC AUC (тест)"); axes[0].legend()
    fig.tight_layout(); fig.savefig(os.path.join(figs, f"auc_by_size_fs{fs}.png"), dpi=180); plt.close(fig)


def fig_auc_vs_positives(d, figs, fs):
    sub = d[d.feature_set == fs]
    models = sorted(sub.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        s = sub[sub.model == m]
        sc = ax.scatter(s["n_pos_train"], s["auc_test"], c=np.log10(s["n"]), cmap="viridis", s=25)
        ax.set_xscale("log"); ax.set_title(m); ax.grid(alpha=0.3)
        ax.set_xlabel("число дефолтов в обучающей выборке")
    axes[0][0].set_ylabel("ROC AUC (тест)")
    fig.colorbar(sc, ax=axes[0].tolist(), label="log10(N)")
    fig.savefig(os.path.join(figs, f"auc_vs_positives_fs{fs}.png"), dpi=180, bbox_inches="tight"); plt.close(fig)


def fig_auc_by_features(d, figs, n, rate):
    sub = d[(d.n == n) & (d.rate == rate)]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for m in sorted(sub.model.unique()):
        line_with_band(ax, agg(sub[sub.model == m], ["n_features"], "auc_test"), "n_features", m)
    ax.set_xlabel("число признаков k"); ax.set_ylabel("ROC AUC (тест)")
    ax.set_title(f"N={n}, доля дефолтов: {'естеств.' if rate == 'nat' else rate}"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(figs, f"auc_by_features_n{n}_p{rate}.png"), dpi=180); plt.close(fig)


def fig_errors(d, figs, fs):
    sub = d[(d.feature_set == fs) & (d.rate == "nat")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, col, ttl in zip(axes, [f"err1_{Q}", f"err2_{Q}"],
                            ["ошибка I рода (пропущенный дефолт)", "ошибка II рода (отказ хорошему клиенту)"]):
        for m in sorted(sub.model.unique()):
            line_with_band(ax, agg(sub[sub.model == m], ["n"], col), "n", m)
        ax.set_xscale("log"); ax.set_xlabel("N"); ax.set_title(ttl + f", отказ {Q[1:]}%"); ax.grid(alpha=0.3)
    axes[0].legend(); fig.tight_layout()
    fig.savefig(os.path.join(figs, f"errors_{Q}_fs{fs}.png"), dpi=180); plt.close(fig)


def fig_time(d, figs, fs):
    sub = d[(d.feature_set == fs) & (d.rate == "nat")]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for m in sorted(sub.model.unique()):
        line_with_band(ax, agg(sub[sub.model == m], ["n"], "t_fit"), "n", m)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("N"); ax.set_ylabel("время обучения, с")
    ax.grid(alpha=0.3, which="both"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(figs, f"time_fit_fs{fs}.png"), dpi=180); plt.close(fig)


def tables(d, out):
    keys = ["model", "feature_set", "n", "rate"]
    cols = ["auc_test", "pr_auc_test", f"err1_{Q}", f"err2_{Q}", "t_fit"]
    t = d.groupby(keys)[cols].agg(["mean", "std"])
    t.columns = ["_".join(c) for c in t.columns]
    t.reset_index().to_csv(os.path.join(out, "profile_table.csv"), index=False)
    # кто лучше по AUC в каждой ячейке (N, доля, k): основа для правил менеджера
    cell = d.groupby(["feature_set", "n", "rate", "model"]).auc_test.mean().reset_index()
    best = cell.loc[cell.groupby(["feature_set", "n", "rate"]).auc_test.idxmax()]
    best.to_csv(os.path.join(out, "best_by_cell.csv"), index=False)
    print("\nЛучшая по среднему AUC модель в ячейках (число ячеек):")
    print(best.model.value_counts().to_string())
    # проверка гипотезы: что объясняет AUC - число дефолтов или размер N (МНК на log-признаках)
    print("\nМНК: auc ~ log10(число дефолтов) + log10(N) + доля, по моделям (набор all)")
    for m, s in d[d.feature_set == "all"].groupby("model"):
        X = np.column_stack([np.ones(len(s)), np.log10(s.n_pos_train), np.log10(s.n)])
        beta, *_ = np.linalg.lstsq(X, s.auc_test.to_numpy(), rcond=None)
        r2 = 1 - ((s.auc_test - X @ beta) ** 2).sum() / ((s.auc_test - s.auc_test.mean()) ** 2).sum()
        print(f"  {m}: log10(дефолты) {beta[1]:+.3f}, log10(N) {beta[2]:+.3f}, R2 {r2:.3f}")
    print("  (коэффициенты сильно коррелированных признаков неустойчивы; это ориентир, не доказательство)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--figs", default="figures")
    args = ap.parse_args()
    os.makedirs(args.figs, exist_ok=True); os.makedirs(args.out, exist_ok=True)
    d = load(args.csv)
    d["feature_set"] = d["feature_set"].astype(str)
    fig_auc_by_size(d, args.figs, "all")
    fig_auc_vs_positives(d, args.figs, "all")
    fig_errors(d, args.figs, "all")
    fig_time(d, args.figs, "all")
    fig_auc_by_features(d, args.figs, int(d.n.max()), "nat")
    if 10000 in set(d.n):
        fig_auc_by_features(d, args.figs, 10000, "nat")
    tables(d, args.out)
    print("\nграфики:", sorted(os.listdir(args.figs)))


if __name__ == "__main__":
    main()
