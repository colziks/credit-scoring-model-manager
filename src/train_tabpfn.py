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


def make_model(cat_idx, seed, fit_mode):
    import tabpfn
    print("tabpfn version:", getattr(tabpfn, "__version__", "?"))
    from tabpfn import TabPFNClassifier
    try:
        from tabpfn.constants import ModelVersion
        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V2_5, categorical_features_indices=cat_idx,
            random_state=seed, device="cuda", fit_mode=fit_mode)
    except Exception as e:                       # API не совпало: не молчим и не берём модель по умолчанию
        raise SystemExit(f"Не удалось явно создать TabPFN-2.5 ({e!r}). "
                         f"Проверь API в документации установленной версии tabpfn.")


def predict_chunks(model, X, label="", chunk=CHUNK):
    out, t0 = [], time.time()
    for i in range(0, len(X), chunk):
        out.append(model.predict_proba(X.iloc[i:i + chunk])[:, 1])
        done = min(i + chunk, len(X))
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
    ap.add_argument("--fit-mode", default="fit_preprocessors",
                    choices=["low_memory", "fit_preprocessors", "fit_with_cache"],
                    help="режим TabPFN; fit_with_cache должен ускорять предсказание (проверить замером)")
    ap.add_argument("--chunk", type=int, default=CHUNK, help="размер порции predict_proba (большая порция может резко ускорить)")
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
    sub = f"|sub{len(val)}_{len(test)}" + ("" if args.fit_mode == "fit_preprocessors" else f"|{args.fit_mode}")
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
            base = dict(run_id=run_id, model="tabpfn25", n=n, rate=p, seed=s, feature_set=fs,
                        n_features=len(cols), fit_mode=args.fit_mode, chunk=args.chunk, val_n=len(val),
                        test_n=len(test), train_default_rate=float(ytr.mean()))
            model = None
            try:
                model = make_model(cat_idx, s, args.fit_mode)
                t0 = time.time(); model.fit(train[cols], ytr); t_fit = time.time() - t0
                t0 = time.time()
                pv = predict_chunks(model, val[cols], "val", args.chunk); pt = predict_chunks(model, test[cols], "test", args.chunk)
                t_pred = time.time() - t0
            except Exception as e:
                # нехватка памяти GPU фиксируется как результат запуска, остальная сетка продолжается
                if "out of memory" not in str(e).lower():
                    raise
                print(f"{run_id}: OOM (нехватка памяти GPU)", flush=True)
                append_row(res_path, dict(**base, status="oom"))
                del model
                import gc, torch
                gc.collect(); torch.cuda.empty_cache()
                continue
            met = compute_metrics(yv, pv, yt, pt)
            append_row(res_path, dict(**base, status="ok", t_fit=t_fit, t_pred=t_pred, **met))
            save_preds(args.out, run_id, pv, pt)
            print(f"{run_id}: AUC {met['auc_test']:.4f}  fit {t_fit:.1f}s  pred {t_pred:.1f}s", flush=True)

if __name__ == "__main__":
    main()
