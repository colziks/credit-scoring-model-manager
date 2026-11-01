"""Figures and checks for the "Data" section of the article.

Reads only a few columns of data/agg.parquet (low memory), writes
  figures/fig_default_time.png   default rate over the time-ordered applications
  figures/fig_nloans.png         distribution of the number of past credits per application
  results/feature_groups.csv     number of features in every group (for the table in the article)
and prints the numbers used in the text (min n_loans, missing values, ...).

Usage (cmd):
  cd /d D:\\dev\\proj
  pip install matplotlib
  python src\\data_figures.py
"""
import argparse
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GROUPS = [
    ("n_loans", lambda c: c == "n_loans"),
    ("флаги кредитов", lambda c: c.startswith(("is_zero_", "pclose_flag", "fclose_flag"))),
    ("коды кредита (enc_loans_*)", lambda c: c.startswith("enc_loans_")),
    ("статусы платежей, окна (paym_*)", lambda c: c.startswith("paym_")),
    ("интервальные признаки (pre_*)", lambda c: c.startswith("pre_")),
    ("последний кредит (last_loan__*)", lambda c: c.startswith("last_loan__")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default="data/agg.parquet")
    ap.add_argument("--meta", default="data/agg_meta.json")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--results", default="results")
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()
    os.makedirs(args.figures, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)

    # ---- 1. feature groups (schema only, nothing is loaded into memory)
    cols = [c for c in pq.read_schema(args.agg).names if c not in ("id", "flag")]
    cat = set()
    if os.path.exists(args.meta):
        import json
        cat = set(json.load(open(args.meta, encoding="utf-8")).get("categorical", []))
    rows, used = [], set()
    for name, rule in GROUPS:
        members = [c for c in cols if rule(c) and c not in used]
        used.update(members)
        rows.append(dict(group=name, n_features=len(members),
                         n_categorical=sum(c in cat for c in members)))
    rest = [c for c in cols if c not in used]
    if rest:
        rows.append(dict(group="не отнесены ни к одной группе", n_features=len(rest),
                         n_categorical=sum(c in cat for c in rest)))
        print("НЕ ОТНЕСЕНЫ К ГРУППАМ:", rest[:10])
    gtab = pd.DataFrame(rows)
    gtab.to_csv(os.path.join(args.results, "feature_groups.csv"), index=False, encoding="utf-8-sig")
    print("\nГруппы признаков (всего %d, категориальных %d):" % (len(cols), len(cat)))
    print(gtab.to_string(index=False))

    # ---- 2. only id, flag, n_loans are loaded
    df = pd.read_parquet(args.agg, columns=["id", "flag", "n_loans"]).sort_values("id").reset_index(drop=True)
    n = len(df)
    print("\nЗаявок: %d, доля дефолтов: %.4f" % (n, df.flag.mean()))
    print("n_loans: min=%d, медиана=%.0f, среднее=%.2f, 99%%-квантиль=%.0f, max=%d" % (
        df.n_loans.min(), df.n_loans.median(), df.n_loans.mean(),
        df.n_loans.quantile(0.99), df.n_loans.max()))
    print("Заявок с n_loans = 1: %.1f%%" % (100 * (df.n_loans == 1).mean()))
    print("ПРОВЕРКА утверждения в тексте «у каждой заявки есть хотя бы одна запись»: "
          + ("ВЕРНО" if df.n_loans.min() >= 1 else "НЕВЕРНО, править текст"))

    # missing values over all columns, parquet is read by row groups
    pf = pq.ParquetFile(args.agg)
    n_nan = 0
    for i in range(pf.num_row_groups):
        n_nan += int(pf.read_row_group(i).to_pandas().isna().sum().sum())
    print("ПРОВЕРКА «пропусков в агрегированной таблице нет»: пропусков = %d" % n_nan)

    # ---- 3. figure: default rate over time
    edges = np.linspace(0, n, args.bins + 1).astype(int)
    rate = np.array([df.flag.iloc[a:b].mean() for a, b in zip(edges[:-1], edges[1:])]) * 100
    centers = (edges[:-1] + edges[1:]) / 2 / n * 100
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(centers, rate, marker="o", ms=3, lw=1.2, color="black")
    for frac, lab in ((args.train_frac, "обучение | валидация"),
                      (args.train_frac + args.val_frac, "валидация | тест")):
        ax.axvline(frac * 100, color="gray", ls="--", lw=0.8)
        ax.text(frac * 100 + 0.5, rate.max(), lab, rotation=90, va="top", ha="left", fontsize=7, color="gray")
    ax.set_xlabel("Положение заявки в порядке id (по времени), %")
    ax.set_ylabel("Доля дефолтов, %")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.figures, "fig_default_time.png"), dpi=200)
    plt.close(fig)

    # ---- 4. figure: number of credits per application
    top = int(df.n_loans.quantile(0.99))
    vc = df.n_loans.clip(upper=top).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(vc.index, vc.values / n * 100, color="gray", edgecolor="black", lw=0.4)
    ax.set_xlabel("Число прошлых кредитов в заявке (значения выше %d объединены)" % top)
    ax.set_ylabel("Доля заявок, %")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(args.figures, "fig_nloans.png"), dpi=200)
    plt.close(fig)

    print("\nСохранено: figures\\fig_default_time.png, figures\\fig_nloans.png, results\\feature_groups.csv")
    print("Доли дефолтов по 5 группам времени:",
          [round(float(df.flag.iloc[a:b].mean()) * 100, 2)
           for a, b in zip(np.linspace(0, n, 6).astype(int)[:-1], np.linspace(0, n, 6).astype(int)[1:])])


if __name__ == "__main__":
    main()
