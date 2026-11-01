"""Model manager: choose between TabPFN-2.5 and CatBoost by the regime of the training sample.

Rule:   TabPFN-2.5 if D < tau and N <= N_max, else CatBoost
        (D = number of defaults in the training sample, N = its size).
Loss:   L(tau, N_max) = mean over regimes of the type-I error E_I at the fixed share of rejections q,
        computed on VALIDATION (the threshold of the decision is also taken on validation).
Search: brute force over the grid of (tau, N_max); the minimum is shown on the plot.
Check:  the chosen (tau, N_max) is applied on TEST (same 10 000 objects as for TabPFN-2.5);
        compared with CatBoost everywhere, TabPFN-2.5 everywhere, rank averaging and stacking.
        Paired bootstrap over test objects (95% CI of the difference of E_I).
        Leave-one-N-out: (tau, N_max) is chosen without the runs of one N and tested on that N.

Usage (cmd):
  cd /d D:\\dev\\proj
  python src\\manager.py --q 0.10
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

SUB = 10000


def subset_idx(n_total, k, part):
    seed = {"val": 101, "test": 202}[part]
    return np.sort(np.random.default_rng(seed).permutation(n_total)[:k])


def thr_reject(p_val, p_test, q):
    """reject the share q of the most risky on validation, apply the threshold on test"""
    t = np.quantile(p_val, 1 - q)
    return p_val >= t, p_test >= t


def e1(reject, y):
    pos = y == 1
    return np.sum(~reject & pos) / pos.sum()


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ecdf_map(ref, x):
    r = np.sort(ref)
    return np.searchsorted(r, x, side="right") / len(r)


def load_runs(args, iv, it):
    man = pd.read_csv(os.path.join(args.splits, "manifest.csv"))
    man = man[man.part == "train"]
    runs = []
    for _, r in man.iterrows():
        m = re.search(r"train_n(\d+)_p([^_]+)_s(\d+)", r["file"])
        N, tag, seed = int(m.group(1)), m.group(2), int(m.group(3))
        f_t = os.path.join(args.preds, f"tabpfn25__train_n{N}_p{tag}_s{seed}__fsall__sub{SUB}_{SUB}.npz")
        f_c = os.path.join(args.preds, f"{args.gbm_name}__train_n{N}_p{tag}_s{seed}__fsall.npz")
        if not (os.path.exists(f_t) and os.path.exists(f_c)):
            continue
        zt, zc = np.load(f_t), np.load(f_c)
        runs.append(dict(N=N, tag=tag, seed=seed, D=int(round(N * r["default_rate"])),
                         tv=zt["val"], tt=zt["test"], cv=zc["val"][iv], ct=zc["test"][it]))
    return runs


def stack_scores(r, yv, seed=0):
    """logistic regression on the predictions of both models, out-of-fold on validation"""
    Xv = np.c_[logit(r["tv"]), logit(r["cv"])]
    Xt = np.c_[logit(r["tt"]), logit(r["ct"])]
    oof = np.zeros(len(yv))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xv, yv):
        oof[b] = LogisticRegression(max_iter=1000).fit(Xv[a], yv[a]).decision_function(Xv[b])
    return oof, LogisticRegression(max_iter=1000).fit(Xv, yv).decision_function(Xt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, default=0.10, help="share of rejections")
    ap.add_argument("--gbm-name", default="catboost")
    ap.add_argument("--preds", default="results/preds")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--results", default="results")
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()
    os.makedirs(args.figures, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)
    q = args.q

    yv_full = pd.read_parquet(os.path.join(args.splits, "val.parquet"), columns=["flag"])["flag"].to_numpy()
    yt_full = pd.read_parquet(os.path.join(args.splits, "test.parquet"), columns=["flag"])["flag"].to_numpy()
    iv, it = subset_idx(len(yv_full), SUB, "val"), subset_idx(len(yt_full), SUB, "test")
    yv, yt = yv_full[iv], yt_full[it]
    runs = load_runs(args, iv, it)
    if not runs:
        raise SystemExit("нет запусков, у которых есть вероятности обеих моделей; проверьте пути и --gbm-name")
    print("Запусков с обеими моделями: %d; режимов (N, доля): %d; значений D: %s" % (
        len(runs), len({(r['N'], r['tag']) for r in runs}), sorted({r['D'] for r in runs})))
    print("Валидационное подмножество: %d объектов, дефолтов %d; тест: %d, дефолтов %d" % (
        len(yv), yv.sum(), len(yt), yt.sum()))

    # ---------- per-run decisions and errors of every method
    for r in runs:
        r["rej"] = {}
        for name, (pv, pt) in {"TabPFN-2.5": (r["tv"], r["tt"]), "CatBoost": (r["cv"], r["ct"])}.items():
            rv, rt = thr_reject(pv, pt, q)
            r["rej"][name] = (rv, rt)
        # rank averaging: map each model's scores to the validation ECDF of the same model
        uv = (ecdf_map(r["tv"], r["tv"]) + ecdf_map(r["cv"], r["cv"])) / 2
        ut = (ecdf_map(r["tv"], r["tt"]) + ecdf_map(r["cv"], r["ct"])) / 2
        r["rej"]["Усреднение по рангам"] = thr_reject(uv, ut, q)
        sv, st = stack_scores(r, yv)
        r["rej"]["Стекинг"] = thr_reject(sv, st, q)

    # ---------- grid search on validation
    Ds = np.array(sorted({r["D"] for r in runs}), dtype=float)
    taus = np.concatenate([[0.0], (Ds[:-1] + Ds[1:]) / 2, [Ds[-1] + 1]])
    nmaxs = sorted({r["N"] for r in runs})

    def pick(r, tau, nmax):
        return "TabPFN-2.5" if (r["D"] < tau and r["N"] <= nmax) else "CatBoost"

    def loss(rs, tau, nmax, part):
        return np.mean([e1(r["rej"][pick(r, tau, nmax)][0 if part == "val" else 1],
                           yv if part == "val" else yt) for r in rs])

    def best(rs):
        grid = [(tau, nm, loss(rs, tau, nm, "val")) for nm in nmaxs for tau in taus]
        lmin = min(g[2] for g in grid)
        winners = [g for g in grid if g[2] <= lmin + 1e-12]
        winners.sort(key=lambda g: (g[0], g[1]))
        return winners[len(winners) // 2], winners, grid

    (tau_s, nmax_s, lmin), winners, grid = best(runs)
    pd.DataFrame(grid, columns=["tau", "N_max", "L_val"]).to_csv(
        os.path.join(args.results, "manager_grid_q%02d.csv" % round(q * 100)), index=False)
    print("\nМинимум потерь на валидации L = %.4f; точек сетки с этим минимумом: %d" % (lmin, len(winners)))
    print("   диапазон tau среди них: %.0f ... %.0f; N_max: %s" % (
        min(w[0] for w in winners), max(w[0] for w in winners), sorted({w[1] for w in winners})))
    print("Выбрано: tau = %.0f, N_max = %d" % (tau_s, nmax_s))
    lc = loss(runs, 0.0, nmax_s, "val")
    lt = loss(runs, 1e12, max(nmaxs), "val")
    print("   L на валидации: CatBoost везде %.4f, TabPFN везде (до N_max=%d) %.4f" % (lc, max(nmaxs), lt))
    on_edge = tau_s in (taus[0], taus[-1]) or len(winners) > 1
    print("   ВНИМАНИЕ: минимум достигается на границе сетки или на плато, оптимум не единственный"
          if on_edge else "   минимум внутренний и единственный")

    # ---------- test: methods
    methods = ["CatBoost", "TabPFN-2.5", "Усреднение по рангам", "Стекинг", "Менеджер"]
    for r in runs:
        r["rej"]["Менеджер"] = r["rej"][pick(r, tau_s, nmax_s)]
    pos = (yt == 1).astype(float)
    res = {m: np.array([e1(r["rej"][m][1], yt) for r in runs]) for m in methods}

    # paired bootstrap over test objects
    rng = np.random.default_rng(0)
    counts = rng.multinomial(len(yt), np.ones(len(yt)) / len(yt), size=args.boot).astype(float)  # B x n
    denom = counts @ pos
    boot = {}
    for m in methods:
        miss = np.stack([(~r["rej"][m][1] & (yt == 1)).astype(float) for r in runs], axis=1)  # n x runs
        boot[m] = ((counts @ miss) / denom[:, None]).mean(axis=1)   # mean over runs for every resample
    rows = []
    for m in methods:
        d = boot[m] - boot["Менеджер"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append(dict(method=m, E_I_test=round(res[m].mean(), 4),
                         diff_vs_manager=round(res[m].mean() - res["Менеджер"].mean(), 4),
                         ci_low=round(lo, 4), ci_high=round(hi, 4)))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(args.results, "manager_test_q%02d.csv" % round(q * 100)), index=False, encoding="utf-8-sig")
    print("\nТест, E_I при отказе q=%.0f%% (среднее по %d запускам); разность = метод минус менеджер, 95%% ДИ:" % (q * 100, len(runs)))
    print(tab.to_string(index=False))
    print("Менеджер выбирает TabPFN-2.5 в %d из %d запусков" % (
        sum(pick(r, tau_s, nmax_s) == "TabPFN-2.5" for r in runs), len(runs)))

    # ---------- leave-one-N-out
    print("\nLeave-one-N-out (tau, N_max выбираются без запусков данного N):")
    lono = []
    for N in nmaxs:
        tr = [r for r in runs if r["N"] != N]
        te = [r for r in runs if r["N"] == N]
        if not tr:
            continue
        (tau_k, nm_k, _), _, _ = best(tr)
        em = np.mean([e1(r["rej"][pick(r, tau_k, nm_k)][1], yt) for r in te])
        ec = np.mean([e1(r["rej"]["CatBoost"][1], yt) for r in te])
        et = np.mean([e1(r["rej"]["TabPFN-2.5"][1], yt) for r in te])
        lono.append(dict(N=N, tau=tau_k, N_max=nm_k, manager=round(em, 4), catboost=round(ec, 4), tabpfn=round(et, 4)))
    lt_ = pd.DataFrame(lono)
    print(lt_.to_string(index=False))
    lt_.to_csv(os.path.join(args.results, "manager_lono_q%02d.csv" % round(q * 100)), index=False)

    # ---------- figure 1: loss curve L(tau)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for nm, ls in zip(nmaxs, ["-", "--", "-.", ":", (0, (5, 1))]):
        xs = [g for g in grid if g[1] == nm]
        ax.plot(range(len(taus)), [g[2] * 100 for g in xs], ls, color="black", lw=1.1, label="$N_{max}$ = %d" % nm)
    j = int(np.argmin(np.abs(taus - tau_s)))
    ax.plot(j, lmin * 100, "o", ms=7, mfc="none", mec="black")
    ax.set_xticks(range(len(taus)))
    ax.set_xticklabels(["0"] + ["%.0f" % t for t in taus[1:-1]] + [">%.0f" % Ds[-1]], rotation=90, fontsize=6)
    ax.set_xlabel("Порог $\\tau$ по числу дефолтов в обучающей выборке")
    ax.set_ylabel("Потери $L$ = ошибка I рода, %")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.figures, "fig_manager_loss.png"), dpi=200)
    plt.close(fig)

    # ---------- figure 2: E_I vs N, natural share, three methods on test
    nat = [r for r in runs if r["tag"] == "nat"]
    if nat:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        for m, mk, ls in (("TabPFN-2.5", "o", "-"), ("CatBoost", "s", "--"), ("Менеджер", "^", ":")):
            xs = sorted({r["N"] for r in nat})
            ys = [np.mean([e1(r["rej"][m][1], yt) for r in nat if r["N"] == n_]) * 100 for n_ in xs]
            ax.plot(xs, ys, ls, marker=mk, color="black", lw=1.1, ms=4, label=m)
        ax.set_xscale("log")
        ax.set_xlabel("Размер обучающей выборки $N$ (естественная доля дефолтов)")
        ax.set_ylabel("Ошибка I рода при $q$ = %d%%, %%" % round(q * 100))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.figures, "fig_manager_err1.png"), dpi=200)
        plt.close(fig)

    print("\nСохранено: results\\manager_*_q%02d.csv, figures\\fig_manager_loss.png, figures\\fig_manager_err1.png" % round(q * 100))


if __name__ == "__main__":
    main()
