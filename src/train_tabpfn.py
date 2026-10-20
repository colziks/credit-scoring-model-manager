"""TabPFN-2.5 на тех же подвыборках, том же val/test и с теми же метриками, что и train_gbm.py.

Запускать в Kaggle/Colab (GPU). Нужно: pip install tabpfn, аккаунт PriorLabs и принятая
лицензия TabPFN-2.5 (токен: переменная окружения TABPFN_TOKEN, если пакет её просит).

Версия задаётся ЯВНО (в новых версиях пакета по умолчанию другая модель).
ВНИМАНИЕ: имена API (ModelVersion.V2_5, create_default_for_version) записаны по памяти
и не проверены на твоей версии пакета; скрипт при старте печатает версию пакета и
падает с понятной ошибкой, если API другой. Сверься с README пакета tabpfn.

Запуск (Colab, данные скопированы в /content/data/splits):
  python src/train_tabpfn.py --splits /content/data/splits --out results --sizes 2000 5000
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from common import (append_row, compute_metrics, done_ids, load_all, parse_name,
                    save_preds, subset_idx)

CHUNK = 2000   # порции, чтобы не упереться в память GPU и видеть прогресс


def make_model(cat_idx, seed):
    import tabpfn
    print("tabpfn version:", getattr(tabpfn, "__version__", "?"))
    from tabpfn import TabPFNClassifier
    try:
        from tabpfn.constants import ModelVersion
        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V2_5, categorical_features_indices=cat_idx,
            random_state=seed, device="cuda")
    except Exception as e:                       # API не совпало: не молчим и не берём модель по умолчанию
        raise SystemExit(f"Не удалось явно создать TabPFN-2.5 ({e!r}). "
                         f"Проверь API в документации установленной версии tabpfn.")


def predict_chunks(model, X, label=""):
    out, t0 = [], time.time()
    for i in range(0, len(X), CHUNK):
        out.append(model.predict_proba(X.iloc[i:i + CHUNK])[:, 1])
        done = min(i + CHUNK, len(X))
        print(f"   {label} {done}/{len(X)}  {time.time() - t0:.0f}s", flush=True)
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--feature-sets", nargs="+", default=None)
    ap.add_argument("--sizes", nargs="+", type=int, default=None)
    ap.add_argument("--rates", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--val-n", type=int, default=None, help="предсказывать только на k объектах val")
    ap.add_argument("--test-n", type=int, default=None, help="предсказывать только на k объектах test")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fsets, cat_all, val, test, train_files = load_all(args.splits)
    fs_names = args.feature_sets or list(fsets)
    res_path = os.path.join(args.out, "results_tabpfn.csv")
    done = done_ids(res_path)
    iv = subset_idx(len(val), args.val_n, "val")
    it = subset_idx(len(test), args.test_n, "test")
    val, test = val.iloc[iv].reset_index(drop=True), test.iloc[it].reset_index(drop=True)
    yv, yt = val["flag"].to_numpy(), test["flag"].to_numpy()
    sub = f"|sub{len(val)}_{len(test)}"
    print(f"val {len(val)} (дефолтов {int(yv.sum())}), test {len(test)} (дефолтов {int(yt.sum())})")

    for rel in train_files:
        n, p, s = parse_name(rel)
        if (args.sizes and n not in args.sizes) or (args.rates and p not in args.rates) \
                or (args.seeds and s not in args.seeds):
            continue
        train = pd.read_parquet(os.path.join(args.splits, rel))
        ytr = train["flag"].to_numpy()
        for fs in fs_names:
            cols = fsets[fs]
            cat_idx = [i for i, c in enumerate(cols) if c in cat_all]
            run_id = f"tabpfn25|train_n{n}_p{p}_s{s}|fs{fs}{sub}"
            if run_id in done:
                continue
            model = make_model(cat_idx, s)
            t0 = time.time(); model.fit(train[cols], ytr); t_fit = time.time() - t0
            t0 = time.time()
            pv = predict_chunks(model, val[cols], "val"); pt = predict_chunks(model, test[cols], "test")
            t_pred = time.time() - t0
            met = compute_metrics(yv, pv, yt, pt)
            row = dict(run_id=run_id, model="tabpfn25", n=n, rate=p, seed=s, feature_set=fs,
                       n_features=len(cols), val_n=len(val), test_n=len(test), train_default_rate=float(ytr.mean()),
                       t_fit=t_fit, t_pred=t_pred, **met)
            append_row(res_path, row)
            save_preds(args.out, run_id, pv, pt)
            print(f"{run_id}: AUC {met['auc_test']:.4f}  fit {t_fit:.1f}s  pred {t_pred:.1f}s", flush=True)


if __name__ == "__main__":
    main()
