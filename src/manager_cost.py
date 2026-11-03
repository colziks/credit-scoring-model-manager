"""Cost-aware model manager (variant with the price of computation).

Rule (as before):  TabPFN-2.5 if D < tau and N <= N_max, else CatBoost.
Loss on validation:
    J(tau, N_max; mu) = L(tau, N_max) + mu * C(tau, N_max)
where
    L = mean E_I at the fixed share of rejections q (type-I error, validation),
    C = relative cost = mean time of the chosen models / mean time of "TabPFN-2.5 everywhere".
mu is the price of computation in units of E_I: "how much E_I we are ready to lose to save all the time".
The manager is built for several mu (mu = 0 is the previous manager); the whole trade-off is shown as a Pareto plot.

Times (seconds, defaults = measured values from the experiments, can be changed with the options):
  TabPFN-2.5: inference on 20 000 objects (GPU Tesla T4) at N = 2000, 5000, 10000, 25000
  CatBoost  : training on the sample (CPU) at the same N
NOTE: different hardware and slightly different work, so the comparison is approximate.

Usage (cmd):
  cd /d D:\\dev\\proj
  python src\\manager_cost.py --q 0.10
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

from manager import subset_idx, thr_reject, e1, load_runs, SUB


def e2(reject, y):
    neg = y == 0
    return float(np.sum(reject & neg) / neg.sum())


def parse_times(s):
    return {int(a): float(b) for a, b in (p.split(":") for p in s.split(","))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, default=0.10)
    ap.add_argument("--mus", nargs="+", type=float, default=[0.0, 0.005, 0.01, 0.02, 0.05])
    ap.add_argument("--mu-plot", type=float, default=0.02, help="mu for the loss-curve figure")
    ap.add_argument("--t-tabpfn", default="2000:84,5000:160,10000:333,25000:1243")
    ap.add_argument("--t-cat", default="2000:41,5000:41,10000:64,25000:85")
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
    tT, tC = parse_times(args.t_tabpfn), parse_times(args.t_cat)

    yv_full = pd.read_parquet(os.path.join(args.splits, "val.parquet"), columns=["flag"])["flag"].to_numpy()
    yt_full = pd.read_parquet(os.path.join(args.splits, "test.parquet"), columns=["flag"])["flag"].to_numpy()
    iv, it = subset_idx(len(yv_full), SUB, "val"), subset_idx(len(yt_full), SUB, "test")
    yv, yt = yv_full[iv], yt_full[it]
    runs = load_runs(args, iv, it)
    if not runs:
        raise SystemExit("нет запусков с вероятностями обеих моделей; проверьте пути и --gbm-name")
    missing = sorted({r["N"] for r in runs} - set(tT) - set())
    if missing or sorted({r["N"] for r in runs} - set(tC)):
        raise SystemExit("нет времени для некоторых N: задайте --t-tabpfn и --t-cat")

    # E_I of both models per run (validation and test) and the time of both models
    for r in runs:
        r["ev"], r["et"], r["rej_t"], r["auc"], r["e2"] = {}, {}, {}, {}, {}
        for name, (pv, pt) in {"TabPFN-2.5": (r["tv"], r["tt"]), "CatBoost": (r["cv"], r["ct"])}.items():
            rv, rt = thr_reject(pv, pt, q)
            r["ev"][name], r["et"][name], r["rej_t"][name] = e1(rv, yv), e1(rt, yt), rt
            r["auc"][name], r["e2"][name] = roc_auc_score(yt, pt), e2(rt, yt)
        r["time"] = {"TabPFN-2.5": tT[r["N"]], "CatBoost": tC[r["N"]]}
    t_ref = np.mean([r["time"]["TabPFN-2.5"] for r in runs])        # "TabPFN everywhere"

    Ds = np.array(sorted({r["D"] for r in runs}), dtype=float)
    taus = np.concatenate([[0.0], (Ds[:-1] + Ds[1:]) / 2, [Ds[-1] + 1]])
    nmaxs = sorted({r["N"] for r in runs})

    def pick(r, tau, nm):
        return "TabPFN-2.5" if (r["D"] < tau and r["N"] <= nm) else "CatBoost"

    def point(rs, tau, nm, part):
        ch = [pick(r, tau, nm) for r in rs]
        L = np.mean([r["ev" if part == "val" else "et"][c] for r, c in zip(rs, ch)])
        C = np.mean([r["time"][c] for r, c in zip(rs, ch)]) / np.mean([r["time"]["TabPFN-2.5"] for r in rs])
        return L, C

    def best(rs, mu):
        grid = []
        for nm in nmaxs:
            for tau in taus:
                L, C = point(rs, tau, nm, "val")
                grid.append((tau, nm, L, C, L + mu * C))
        jmin = min(g[4] for g in grid)
        win = sorted([g for g in grid if g[4] <= jmin + 1e-12], key=lambda g: (g[0], g[1]))
        return win[len(win) // 2], win, grid

    # bootstrap over test objects (same scheme as in manager.py)
    rng = np.random.default_rng(0)
    counts = rng.multinomial(len(yt), np.ones(len(yt)) / len(yt), size=args.boot).astype(float)
    pos = (yt == 1).astype(float)
    denom = counts @ pos

    def boot_mean(rejs):
        miss = np.stack([(~rj & (yt == 1)).astype(float) for rj in rejs], axis=1)
        return ((counts @ miss) / denom[:, None]).mean(axis=1)

    b_tab = boot_mean([r["rej_t"]["TabPFN-2.5"] for r in runs])
    b_cat = boot_mean([r["rej_t"]["CatBoost"] for r in runs])
    e_tab = np.mean([r["et"]["TabPFN-2.5"] for r in runs])
    e_cat = np.mean([r["et"]["CatBoost"] for r in runs])

    rows, picks = [], {}
    print("Времена (с): TabPFN-2.5 %s; CatBoost %s" % (tT, tC))
    print("q = %.0f%%; E_I на тесте: TabPFN везде %.4f (стоимость 1,00), CatBoost везде %.4f (стоимость %.2f)" % (
        q * 100, e_tab, e_cat, np.mean([r["time"]["CatBoost"] for r in runs]) / t_ref))
    for name in ("TabPFN-2.5", "CatBoost"):
        print("   %-10s везде: ROC AUC %.4f, E_II %.4f" % (name, np.mean([r["auc"][name] for r in runs]),
                                                           np.mean([r["e2"][name] for r in runs])))
    print()
    for mu in args.mus:
        (tau_s, nm_s, L_v, C_v, J_v), win, grid = best(runs, mu)
        rej = [r["rej_t"][pick(r, tau_s, nm_s)] for r in runs]
        b_m = boot_mean(rej)
        L_t, C_t = point(runs, tau_s, nm_s, "test")
        d_tab = np.percentile(b_m - b_tab, [2.5, 97.5])
        n_tab = sum(pick(r, tau_s, nm_s) == "TabPFN-2.5" for r in runs)
        chosen = [pick(r, tau_s, nm_s) for r in runs]
        auc_m = np.mean([r["auc"][c] for r, c in zip(runs, chosen)])
        e2_m = np.mean([r["e2"][c] for r, c in zip(runs, chosen)])
        rows.append(dict(mu=mu, tau=tau_s, N_max=nm_s, n_minimum_points=len(win), J_val=round(J_v, 4),
                         E_I_test=round(L_t, 4), E_II_test=round(e2_m, 4), AUC_test=round(auc_m, 4),
                         cost=round(C_t, 3), runs_tabpfn=n_tab,
                         vs_tabpfn=round(L_t - e_tab, 4), ci_low=round(d_tab[0], 4), ci_high=round(d_tab[1], 4),
                         vs_catboost=round(L_t - e_cat, 4)))
        picks[mu] = (tau_s, nm_s)

        # leave-one-N-out for this mu
        lo = []
        for N in nmaxs:
            tr, te = [r for r in runs if r["N"] != N], [r for r in runs if r["N"] == N]
            (tk, nk, *_), _, _ = best(tr, mu)
            Lk, Ck = point(te, tk, nk, "test")
            lo.append((N, tk, nk, Lk, Ck))
        rows[-1]["lono_E_I"] = round(np.mean([x[3] for x in lo]), 4)
        rows[-1]["lono_cost"] = round(np.mean([x[4] for x in lo]), 3)
        rows[-1]["lono_choices"] = "; ".join("N=%d: tau=%.0f,Nmax=%d" % (x[0], x[1], x[2]) for x in lo)

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(args.results, "manager_cost_q%02d.csv" % round(q * 100)), index=False, encoding="utf-8-sig")
    show = tab[["mu", "tau", "N_max", "n_minimum_points", "E_I_test", "E_II_test", "AUC_test", "cost", "runs_tabpfn", "vs_tabpfn", "ci_low", "ci_high", "lono_E_I", "lono_cost"]]
    print(show.to_string(index=False))
    print("\n(vs_tabpfn: E_I менеджера минус E_I TabPFN везде, 95% ДИ; cost: относительное время, TabPFN везде = 1;")
    print(" lono_*: то же при выборе порогов без запусков проверяемого N, среднее по N)")
    print("\nВыбор по leave-one-N-out:")
    for _, r in tab.iterrows():
        print("  mu=%.3f  %s" % (r["mu"], r["lono_choices"]))

    # ---------- figure 1: Pareto plot (validation L vs cost), chosen points
    (_, _, _, _, _), _, grid0 = best(runs, 0.0)
    G = np.array([(g[3], g[2]) for g in grid0])        # (C, L_val)
    order = np.argsort(G[:, 0], kind="stable")
    front, bestL = [], np.inf
    for i in order:
        if G[i, 1] < bestL - 1e-12:
            front.append(G[i]); bestL = G[i, 1]
    front = np.array(front)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.scatter(G[:, 0], G[:, 1] * 100, s=10, color="lightgray", label="все пары $(\\tau, N_{max})$")
    ax.step(front[:, 0], front[:, 1] * 100, where="post", color="black", lw=1.2, label="граница компромисса")
    by_choice = {}
    for mu, ch in picks.items():
        by_choice.setdefault(ch, []).append(mu)        # equal choices share one label
    for (tau_s, nm_s), mus_ in by_choice.items():
        L_v, C_v = point(runs, tau_s, nm_s, "val")
        ax.plot(C_v, L_v * 100, "o", ms=6, mfc="white", mec="black")
        ax.annotate("$\\mu$=" + ", ".join("%g" % m for m in mus_), (C_v, L_v * 100),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xlabel("Относительная стоимость вычислений (TabPFN-2.5 везде = 1)")
    ax.set_ylabel("Ошибка I рода на валидации, %")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.figures, "fig_cost_pareto.png"), dpi=200)
    plt.close(fig)

    # ---------- figure 2: J(tau) with the minimum, for mu = mu_plot
    mu = args.mu_plot
    (tau_s, nm_s, _, _, J_s), _, grid = best(runs, mu)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for nm, ls in zip(nmaxs, ["-", "--", "-.", ":"]):
        xs = [g for g in grid if g[1] == nm]
        ax.plot(range(len(taus)), [g[4] * 100 for g in xs], ls, color="black", lw=1.1, label="$N_{max}$ = %d" % nm)
    j = int(np.argmin(np.abs(taus - tau_s)))
    ax.plot(j, J_s * 100, "o", ms=7, mfc="none", mec="black")
    ax.set_xticks(range(len(taus)))
    ax.set_xticklabels(["0"] + ["%.0f" % t for t in taus[1:-1]] + [">%.0f" % Ds[-1]], rotation=90, fontsize=6)
    ax.set_xlabel("Порог $\\tau$ по числу дефолтов в обучающей выборке")
    ax.set_ylabel("$J = L + \\mu C$, %% ($\\mu$ = %g)" % mu)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.figures, "fig_cost_loss.png"), dpi=200)
    plt.close(fig)
    print("\nСохранено: results\\manager_cost_q%02d.csv, figures\\fig_cost_pareto.png, figures\\fig_cost_loss.png" % round(q * 100))


if __name__ == "__main__":
    main()
