"""Градиентные бустинги (CatBoost, LightGBM, XGBoost) на всех подвыборках.

Гиперпараметры ФИКСИРОВАННЫЕ и одинаковые для всех запусков (без подбора на val):
TabPFN тоже работает без подбора, и val нужен менеджеру для порогов, поэтому
early stopping по val не используется (иначе val утекает в обучение моделей).

Запуск (из корня репозитория):
  python src/train_gbm.py --splits data/splits --out results --models catboost lightgbm xgboost
Пробный прогон:
  python src/train_gbm.py --splits data/splits --out results --sizes 2000 --seeds 0 --feature-sets all
Прерванный запуск можно продолжить той же командой: готовые запуски пропускаются.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from common import (append_row, compute_metrics, done_ids, load_all, parse_name,
                    save_preds)

PARAMS = dict(n_estimators=500, learning_rate=0.05, max_depth=6)


def to_category(train, others, cats):
    """Категориальные признаки -> dtype category с категориями из train (для LightGBM/XGBoost)."""
    out = [train.copy()] + [o.copy() for o in others]
    for c in cats:
        dt = pd.CategoricalDtype(sorted(train[c].unique()))
        for d in out:
            d[c] = d[c].astype(dt)
    return out


def fit_predict(name, Xtr, ytr, Xs, cats, seed):
    if name == "catboost":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=PARAMS["n_estimators"], learning_rate=PARAMS["learning_rate"],
                               depth=PARAMS["max_depth"], random_seed=seed, verbose=0,
                               cat_features=cats, thread_count=-1)
        t0 = time.time(); m.fit(Xtr, ytr); t_fit = time.time() - t0
        t0 = time.time(); ps = [m.predict_proba(X)[:, 1] for X in Xs]; t_pred = time.time() - t0
    else:
        Xtr, *Xs = to_category(Xtr, Xs, cats)
        if name == "lightgbm":
            from lightgbm import LGBMClassifier
            m = LGBMClassifier(n_estimators=PARAMS["n_estimators"], learning_rate=PARAMS["learning_rate"],
                               max_depth=PARAMS["max_depth"], num_leaves=31, random_state=seed,
                               verbose=-1, n_jobs=-1)
        elif name == "xgboost":
            from xgboost import XGBClassifier
            m = XGBClassifier(n_estimators=PARAMS["n_estimators"], learning_rate=PARAMS["learning_rate"],
                              max_depth=PARAMS["max_depth"], tree_method="hist",
                              enable_categorical=True, random_state=seed, n_jobs=-1)
        else:
            raise ValueError(name)
        t0 = time.time(); m.fit(Xtr, ytr); t_fit = time.time() - t0
        t0 = time.time(); ps = [m.predict_proba(X)[:, 1] for X in Xs]; t_pred = time.time() - t0
    return ps, t_fit, t_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=["catboost", "lightgbm", "xgboost"])
    ap.add_argument("--feature-sets", nargs="+", default=None, help="например: 20 50 100 all (по умолчанию все)")
    ap.add_argument("--sizes", nargs="+", type=int, default=None)
    ap.add_argument("--rates", nargs="+", default=None, help="например: 0.01 nat 0.10 0.25")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fsets, cat_all, val, test, train_files = load_all(args.splits)
    fs_names = args.feature_sets or list(fsets)
    res_path = os.path.join(args.out, "results_gbm.csv")
    done = done_ids(res_path)
    yv, yt = val["flag"].to_numpy(), test["flag"].to_numpy()

    for rel in train_files:
        n, p, s = parse_name(rel)
        if (args.sizes and n not in args.sizes) or (args.rates and p not in args.rates) \
                or (args.seeds and s not in args.seeds):
            continue
        train = pd.read_parquet(os.path.join(args.splits, rel))
        ytr = train["flag"].to_numpy()
        for fs in fs_names:
            cols = fsets[fs]
            cats = [c for c in cols if c in cat_all]
            for model in args.models:
                run_id = f"{model}|train_n{n}_p{p}_s{s}|fs{fs}"
                if run_id in done:
                    continue
                ps, t_fit, t_pred = fit_predict(model, train[cols], ytr,
                                                [val[cols], test[cols]], cats, s)
                met = compute_metrics(yv, ps[0], yt, ps[1])
                row = dict(run_id=run_id, model=model, n=n, rate=p, seed=s, feature_set=fs,
                           n_features=len(cols), train_default_rate=float(ytr.mean()),
                           t_fit=t_fit, t_pred=t_pred, **met)
                append_row(res_path, row)
                save_preds(args.out, run_id, ps[0], ps[1])
                print(f"{run_id}: AUC {met['auc_test']:.4f}  fit {t_fit:.1f}s", flush=True)


if __name__ == "__main__":
    main()
