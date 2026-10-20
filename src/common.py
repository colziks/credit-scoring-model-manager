"""Общие функции для train_gbm.py и train_tabpfn.py.

Определения (как в статье):
  ошибка I рода  = выдан кредит клиенту, который ушёл в дефолт  -> FN / (все дефолты)
  ошибка II рода = отказ платёжеспособному клиенту              -> FP / (все недефолты)
Положительный класс (flag = 1) = дефолт; "отказать" = предсказание 1.

Порог. Подвыборки с долей дефолтов 10% или 25% сдвигают вероятности модели, поэтому
фиксированный порог вида 0.5 между запусками несопоставим. Правило одинаково для всех
моделей: на валидации берём порог, при котором отказывается ровно доле q самых рискованных
заявок (q = 5, 10, 20%), и применяем этот порог на тесте. Сами вероятности сохраняются,
так что любой другой порог можно посчитать позже, не переобучая модели.
"""
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

QS = (0.05, 0.10, 0.20)


def load_all(splits):
    with open(os.path.join(splits, "feature_sets.json"), encoding="utf-8") as f:
        fs = json.load(f)
    val = pd.read_parquet(os.path.join(splits, "val.parquet"))
    test = pd.read_parquet(os.path.join(splits, "test.parquet"))
    man = pd.read_csv(os.path.join(splits, "manifest.csv"))
    train_files = man[man["part"] == "train"]["file"].tolist()
    return fs["feature_sets"], set(fs["categorical"]), val, test, train_files


def parse_name(rel):
    """subsamples/train_n5000_p0.10_s1.parquet -> (5000, '0.10', 1)"""
    m = re.search(r"train_n(\d+)_p([\d.]+|nat)_s(\d+)\.parquet$", rel)
    return int(m.group(1)), m.group(2), int(m.group(3))


def _errors(y, pred):
    pos, neg = (y == 1), (y == 0)
    err1 = float(((pred == 0) & pos).sum() / max(pos.sum(), 1))   # пропущенный дефолт
    err2 = float(((pred == 1) & neg).sum() / max(neg.sum(), 1))   # отказ хорошему клиенту
    return err1, err2


def compute_metrics(y_val, p_val, y_te, p_te):
    out = {
        "auc_test": float(roc_auc_score(y_te, p_te)),
        "pr_auc_test": float(average_precision_score(y_te, p_te)),
        "auc_val": float(roc_auc_score(y_val, p_val)),
    }
    for q in QS:
        thr = float(np.quantile(p_val, 1 - q))
        pred = (p_te >= thr).astype(int)
        e1, e2 = _errors(y_te, pred)
        tag = f"q{int(q * 100):02d}"
        out[f"err1_{tag}"] = e1
        out[f"err2_{tag}"] = e2
        out[f"flagged_{tag}"] = float(pred.mean())   # на тесте из-за дрейфа может отличаться от q
    return out


def done_ids(path):
    if not os.path.exists(path):
        return set()
    return set(pd.read_csv(path)["run_id"])


def append_row(path, row):
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def save_preds(out_dir, run_id, p_val, p_te):
    os.makedirs(os.path.join(out_dir, "preds"), exist_ok=True)
    safe = run_id.replace("|", "__")
    np.savez_compressed(os.path.join(out_dir, "preds", safe + ".npz"),
                        val=p_val.astype("float32"), test=p_te.astype("float32"))


SUBSET_SEEDS = {"val": 101, "test": 202}


def subset_idx(n_total, k, part):
    """Фиксированное случайное подмножество из k объектов. Вложенное: подмножество
    меньшего k содержится в большем. k=None или k>=n_total -> все объекты.
    Одна и та же функция используется для TabPFN и для пересчёта метрик бустингов,
    поэтому все модели сравниваются на одних и тех же объектах."""
    if k is None or k >= n_total:
        return np.arange(n_total)
    perm = np.random.default_rng(SUBSET_SEEDS[part]).permutation(n_total)
    return np.sort(perm[:k])
