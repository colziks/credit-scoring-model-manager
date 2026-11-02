"""Weak spots of TabPFN-2.5 and CatBoost: analysis of missed defaults (part 1, no retraining).

Works with the saved probabilities and the features of the test applications.
For the chosen regime (N, share of defaults in training, q) and all seeds:
  1. overlap of missed defaults: caught by both / only TabPFN-2.5 / only CatBoost / missed by both;
  2. type-I error E_I by groups of clients: history length (n_loans) and share of non-zero payment statuses,
     with 95% CI of the difference (bootstrap over test defaults);
  3. which numeric features separate (a) stably missed defaults from stably caught defaults and
     (b) stably missed defaults from approved good clients (AUC of one feature; 0.5 = no separation).

"Missed" = default that was not rejected at the threshold chosen on validation (share of rejections q).
"Stably" = in all seeds. Categorical features (random numbering of intervals) are skipped.

Usage (cmd):
  cd /d D:\\dev\\proj
  python src\\weak_spots.py --n 5000 --rate nat --q 0.10
"""
import argparse
import json
import os

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


def feature_group(name):
    for pref, g in (("n_loans", "длина истории"), ("is_zero_", "флаги кредитов"), ("pclose_flag", "флаги кредитов"),
                    ("fclose_flag", "флаги кредитов"), ("enc_loans_", "коды кредита"), ("paym_", "статусы платежей"),
                    ("pre_", "интервальные"), ("last_loan__", "последний кредит")):
        if name.startswith(pref):
            return g
    return "другое"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--rate", default="nat")
    ap.add_argument("--q", type=float, default=0.10)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--gbm-name", default="catboost")
    ap.add_argument("--preds", default="results/preds")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--meta", default="data/agg_meta.json")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--results", default="results")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.figures, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)
    q, S = args.q, len(args.seeds)

    val = pd.read_parquet(os.path.join(args.splits, "val.parquet"), columns=["flag"])
    test = pd.read_parquet(os.path.join(args.splits, "test.parquet"))
    iv, it = subset_idx(len(val), SUB, "val"), subset_idx(len(test), SUB, "test")
    yv = val["flag"].to_numpy()[iv]
    X = test.iloc[it].reset_index(drop=True)
    y = X["flag"].to_numpy()
    cat = set(json.load(open(args.meta, encoding="utf-8")).get("categorical", [])) if os.path.exists(args.meta) else set()
    feats = [c for c in X.columns if c not in ("id", "flag") and c not in cat]
    d_idx = np.flatnonzero(y == 1)
    print("Режим: N=%d, доля дефолтов %s, q=%.0f%%; тест: %d объектов, дефолтов %d; числовых признаков для анализа: %d" % (
        args.n, args.rate, q * 100, len(y), len(d_idx), len(feats)))

    # ---- rejections per seed and model
    rej = {"TabPFN-2.5": [], "CatBoost": []}
    for s in args.seeds:
        tag = f"train_n{args.n}_p{args.rate}_s{s}"
        files = {"TabPFN-2.5": f"tabpfn25__{tag}__fsall__sub{SUB}_{SUB}.npz", "CatBoost": f"{args.gbm_name}__{tag}__fsall.npz"}
        for m, f in files.items():
            path = os.path.join(args.preds, f)
            if not os.path.exists(path):
                raise SystemExit("нет файла: " + path)
            z = np.load(path)
            pv, pt = z["val"], z["test"]
            if len(pv) != SUB:
                pv, pt = pv[iv], pt[it]
            rej[m].append(pt >= np.quantile(pv, 1 - q))
    rej = {m: np.array(v) for m, v in rej.items()}                  # (S, n_test)
    miss = {m: ~v[:, d_idx] for m, v in rej.items()}                 # (S, n_def): default not rejected
    print("E_I по начальным состояниям: TabPFN-2.5 %s; CatBoost %s" % (
        np.round(miss["TabPFN-2.5"].mean(1), 3), np.round(miss["CatBoost"].mean(1), 3)))

    # ---- 1. overlap
    mT, mC = miss["TabPFN-2.5"], miss["CatBoost"]
    parts = {"пойман обеими": ~mT & ~mC, "только TabPFN-2.5": ~mT & mC,
             "только CatBoost": mT & ~mC, "пропущен обеими": mT & mC}
    ov = pd.DataFrame([dict(категория=k, доля_дефолтов=round(v.mean(), 3),
                            мин_по_запускам=round(v.mean(1).min(), 3), макс_по_запускам=round(v.mean(1).max(), 3))
                       for k, v in parts.items()])
    ov.to_csv(os.path.join(args.results, "weak_overlap.csv"), index=False, encoding="utf-8-sig")
    print("\n1. Пересечение пропущенных дефолтов (среднее по запускам):")
    print(ov.to_string(index=False))
    stable_both = (mT.all(0) & mC.all(0)).sum()
    print("Дефолтов, пропущенных обеими моделями во всех запусках: %d из %d" % (stable_both, len(d_idx)))

    # ---- 2. E_I by groups
    rng = np.random.default_rng(0)
    boots = rng.integers(0, len(d_idx), size=(args.boot, len(d_idx)))

    groups = {}
    if "n_loans" in X:
        nl = X.loc[d_idx, "n_loans"].to_numpy()
        for lo, hi, lab in ((1, 1, "1"), (2, 3, "2–3"), (4, 6, "4–6"), (7, 10, "7–10"), (11, 10**6, "11+")):
            groups[("n_loans", lab)] = (nl >= lo) & (nl <= hi)
    if "paym_recent__share_nonzero" in X:
        sh = X.loc[d_idx, "paym_recent__share_nonzero"].to_numpy()
        for lo, hi, lab in ((-1, 0, "0 (нет ненулевых)"), (0, 1 / 3, "(0; 1/3]"), (1 / 3, 2 / 3, "(1/3; 2/3]"), (2 / 3, 2, ">2/3")):
            groups[("paym_recent__share_nonzero", lab)] = (sh > lo) & (sh <= hi)
    rows = []
    for (feat, lab), mask in groups.items():
        n = int(mask.sum())
        if n < 5:
            continue
        mt_, mc_ = mT[:, mask].mean(), mC[:, mask].mean()
        bt = np.array([mT[:, mask][:, rng.integers(0, n, n)].mean() for _ in range(args.boot)])
        bc = np.array([mC[:, mask][:, rng.integers(0, n, n)].mean() for _ in range(args.boot)])
        # paired bootstrap of the difference (same resampled defaults for both models)
        bd = []
        for _ in range(args.boot):
            ii = rng.integers(0, n, n)
            bd.append(mT[:, mask][:, ii].mean() - mC[:, mask][:, ii].mean())
        lo, hi = np.percentile(bd, [2.5, 97.5])
        rows.append(dict(признак=feat, группа=lab, дефолтов=n, E_I_TabPFN=round(mt_, 3), E_I_CatBoost=round(mc_, 3),
                         разность=round(mt_ - mc_, 3), ДИ_низ=round(lo, 3), ДИ_верх=round(hi, 3)))
    gt = pd.DataFrame(rows)
    gt.to_csv(os.path.join(args.results, "weak_groups.csv"), index=False, encoding="utf-8-sig")
    print("\n2. Ошибка I рода по группам клиентов (разность = TabPFN-2.5 минус CatBoost):")
    print(gt.to_string(index=False))

    # figure: E_I by history length
    sub = gt[gt["признак"] == "n_loans"]
    if len(sub):
        fig, ax = plt.subplots(figsize=(6, 3.4))
        x = np.arange(len(sub))
        ax.bar(x - 0.2, sub["E_I_TabPFN"] * 100, 0.4, color="lightgray", edgecolor="black", lw=0.6, label="TabPFN-2.5")
        ax.bar(x + 0.2, sub["E_I_CatBoost"] * 100, 0.4, color="white", edgecolor="black", lw=0.6, hatch="//", label="CatBoost")
        ax.set_xticks(x)
        ax.set_xticklabels(["%s\n(n=%d)" % (g, n) for g, n in zip(sub["группа"], sub["дефолтов"])], fontsize=8)
        ax.set_xlabel("Число прошлых кредитов в заявке (n — число дефолтов в группе)")
        ax.set_ylabel("Ошибка I рода, %")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(args.figures, "fig_weak_nloans.png"), dpi=200)
        plt.close(fig)

    # ---- 3. features separating missed defaults
    print("\n3. Признаки, отличающие стабильно пропущенные дефолты (пропущены во всех запусках):")
    good_accepted = {m: (~rej[m]).all(0) & (y == 0) for m in rej}
    for m in ("TabPFN-2.5", "CatBoost"):
        sm = miss[m].all(0)                      # stably missed among defaults
        sc = (~miss[m]).all(0)                   # stably caught
        print("   %s: стабильно пропущено %d, стабильно поймано %d дефолтов; одобренных добросовестных %d" % (
            m, sm.sum(), sc.sum(), good_accepted[m].sum()))
        rows = []
        for f in feats:
            v = X[f].to_numpy(dtype=float)
            vd = v[d_idx]
            r = dict(модель=m, признак=f, группа=feature_group(f))
            if sm.sum() >= 10 and sc.sum() >= 10 and np.ptp(vd) > 0:
                lab = np.r_[np.ones(sm.sum()), np.zeros(sc.sum())]
                r["AUC_пропущен_vs_пойман"] = round(roc_auc_score(lab, np.r_[vd[sm], vd[sc]]), 3)
            ga = good_accepted[m]
            if sm.sum() >= 10 and ga.sum() >= 10 and np.ptp(v) > 0:
                lab = np.r_[np.ones(sm.sum()), np.zeros(ga.sum())]
                r["AUC_пропущен_vs_одобренный_хороший"] = round(roc_auc_score(lab, np.r_[vd[sm], v[ga]]), 3)
            rows.append(r)
        ft = pd.DataFrame(rows)
        ft.to_csv(os.path.join(args.results, "weak_features_%s.csv" % m.replace(".", "").replace("-", "")), index=False, encoding="utf-8-sig")
        for col in ("AUC_пропущен_vs_пойман", "AUC_пропущен_vs_одобренный_хороший"):
            if col in ft:
                t = ft.dropna(subset=[col]).copy()
                t["отклонение"] = (t[col] - 0.5).abs()
                print("   %s, топ-8 по |AUC - 0,5| (AUC > 0,5: у пропущенных значение больше):" % col)
                print(t.sort_values("отклонение", ascending=False).head(8)[["признак", "группа", col]].to_string(index=False))
                print("   макс. |AUC - 0,5| = %.3f" % t["отклонение"].max())

    print("\nСохранено: results\\weak_*.csv, figures\\fig_weak_nloans.png")


if __name__ == "__main__":
    main()
