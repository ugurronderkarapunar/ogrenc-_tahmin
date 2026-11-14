"""Model zoo, CV karşılaştırma, hiperparametre ayarı ve ensemble kurulumu."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from . import config as cfg
from .ozellikler import KolonTipleri, pipeline_kur

log = logging.getLogger(__name__)

# CV'de ölçülen metrikler. 'brier' sklearn'de negatif tutulur, tabloda pozitife çevrilir.
OLCUTLER = {"f1": "f1", "pr_auc": "average_precision", "roc_auc": "roc_auc",
            "recall": "recall", "precision": "precision", "brier": "neg_brier_score"}


# ------------------------------------------------------------------ model tanımları
def taban_model() -> LogisticRegression:
    """Referans (baseline) model: düzenlileştirilmiş lojistik regresyon."""
    return LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=cfg.RANDOM_STATE)


def rakip_modeller(y_train: pd.Series) -> dict:
    """Lojistik regresyondan farklı 4 aile: bagging (RF), iki boosting (XGB, LGBM), mesafe tabanlı (KNN)."""
    ne = 60 if cfg.HIZLI else 300
    pos_agirlik = float((y_train == 0).sum() / (y_train == 1).sum())   # elle yazılmış 3 yerine veriden
    rs = cfg.RANDOM_STATE
    dorduncu = (
        {"KNN": KNeighborsClassifier(n_neighbors=25, weights="distance")} if cfg.KNN_DAHIL else
        {"ExtraTrees": ExtraTreesClassifier(n_estimators=ne, min_samples_leaf=3, class_weight="balanced",
                                            n_jobs=1, random_state=rs)})        # KNN yerine: eğitim verisi saklamaz
    return {
        "RandomForest": RandomForestClassifier(n_estimators=ne, class_weight="balanced", n_jobs=1, random_state=rs),
        "XGBoost": XGBClassifier(n_estimators=ne, learning_rate=0.05, max_depth=3, subsample=0.8,
                                 colsample_bytree=0.8, scale_pos_weight=pos_agirlik, eval_metric="logloss",
                                 tree_method="hist", n_jobs=1, verbosity=0, random_state=rs),
        "LightGBM": LGBMClassifier(n_estimators=ne, learning_rate=0.05, num_leaves=15, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, class_weight="balanced",
                                   n_jobs=1, verbose=-1, random_state=rs),
        **dorduncu,
    }


def param_uzaylari() -> dict:
    """Rastgele arama dağılımları (prefix 'model__': pipeline'ın son adımı)."""
    ne = randint(40, 100) if cfg.HIZLI else randint(150, 500)
    return {
        "RandomForest": {"model__n_estimators": ne, "model__max_depth": [None, 4, 6, 8, 12],
                         "model__min_samples_leaf": randint(1, 15), "model__max_features": ["sqrt", 0.5, 0.8]},
        "XGBoost": {"model__n_estimators": ne, "model__learning_rate": loguniform(0.02, 0.2),
                    "model__max_depth": randint(2, 6), "model__subsample": uniform(0.6, 0.4),
                    "model__colsample_bytree": uniform(0.6, 0.4), "model__min_child_weight": randint(1, 10),
                    "model__reg_lambda": loguniform(0.1, 10)},
        "LightGBM": {"model__n_estimators": ne, "model__learning_rate": loguniform(0.02, 0.2),
                     "model__num_leaves": randint(6, 40), "model__min_child_samples": randint(10, 60),
                     "model__subsample": uniform(0.6, 0.4), "model__colsample_bytree": uniform(0.6, 0.4),
                     "model__reg_lambda": loguniform(0.1, 10)},
        "KNN": {"model__n_neighbors": randint(5, 60), "model__weights": ["uniform", "distance"],
                "model__p": [1, 2]},
        "ExtraTrees": {"model__n_estimators": ne, "model__max_depth": [None, 4, 6, 8, 12],
                       "model__min_samples_leaf": randint(1, 15), "model__max_features": ["sqrt", 0.5, 0.8]},
    }


# ------------------------------------------------------------------ CV karşılaştırma
def cv_karsilastir(adaylar: dict[str, Pipeline], X, y, cv, sirala: str = "f1") -> tuple[pd.DataFrame, dict]:
    """Aynı fold'larda tüm adayları değerlendirir. Döner: (özet tablo, fold bazlı skorlar)."""
    satirlar, fold_skorlari = [], {}
    for ad, pipe in adaylar.items():
        r = cross_validate(pipe, X, y, cv=cv, scoring=OLCUTLER, n_jobs=cfg.N_JOBS, error_score="raise")
        skorlar = {k: (-r[f"test_{k}"] if k == "brier" else r[f"test_{k}"]) for k in OLCUTLER}
        fold_skorlari[ad] = skorlar
        satir = {"model": ad}
        for k, v in skorlar.items():
            satir[f"{k}"] = v.mean()
            satir[f"{k}_std"] = v.std()
        satirlar.append(satir)
        log.info("%-22s F1=%.3f PR-AUC=%.3f", ad, satir["f1"], satir["pr_auc"])
    tablo = pd.DataFrame(satirlar).set_index("model").sort_values(sirala, ascending=False)
    return tablo, fold_skorlari


def ozet_goster(tablo: pd.DataFrame) -> pd.DataFrame:
    """Ortalama ± std biçiminde okunaklı tablo."""
    cikti = pd.DataFrame(index=tablo.index)
    for k in OLCUTLER:
        cikti[k] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(tablo[k], tablo[f"{k}_std"])]
    return cikti


# ------------------------------------------------------------------ hiperparametre ayarı
def ayarla(ad: str, model, param: dict, tipler: KolonTipleri, X, y, cv) -> RandomizedSearchCV:
    pipe = pipeline_kur(model, tipler)
    arama = RandomizedSearchCV(pipe, param, n_iter=cfg.N_ITER, scoring=cfg.OPTIMIZASYON_METRIGI, cv=cv,
                               n_jobs=cfg.N_JOBS, random_state=cfg.RANDOM_STATE, refit=True, error_score="raise")
    arama.fit(X, y)
    log.info("%-14s en iyi CV %s=%.3f", ad, cfg.OPTIMIZASYON_METRIGI, arama.best_score_)
    return arama


def ayarli_model(taban, arama: RandomizedSearchCV):
    """Aramanın bulduğu parametrelerle yalın (pipeline'sız) model kopyası."""
    p = {k.removeprefix("model__"): v for k, v in arama.best_params_.items()}
    return clone(taban).set_params(**p)


# ------------------------------------------------------------------ ensemble
def ensemble_kur(tur: str, uyeler: dict, tipler: KolonTipleri) -> Pipeline:
    """Ortak ön-işleme + (soft voting | stacking). Üyeler ayarlanmış yalın modellerdir."""
    estimators = [(ad, clone(m)) for ad, m in uyeler.items()]
    if cfg.ENSEMBLE_LR_DAHIL:
        estimators.append(("LogReg", taban_model()))
    if tur == "voting":
        model = VotingClassifier(estimators, voting="soft", n_jobs=1)
    elif tur == "stacking":
        model = StackingClassifier(
            estimators, stack_method="predict_proba", n_jobs=1, passthrough=False,
            final_estimator=LogisticRegression(max_iter=3000, class_weight="balanced"),
            cv=StratifiedKFold(3 if cfg.HIZLI else 5, shuffle=True, random_state=cfg.RANDOM_STATE))
    else:
        raise ValueError(tur)
    return pipeline_kur(model, tipler)


def eslesik_fark(fold_a: np.ndarray, fold_b: np.ndarray) -> tuple[float, float]:
    """Aynı fold'lardaki skor farkı (a-b) ve standart hatası. |fark| < 2·SE ise anlamlı değil."""
    fark = np.asarray(fold_a) - np.asarray(fold_b)
    return float(fark.mean()), float(fark.std(ddof=1) / np.sqrt(len(fark)))


def f1_taban_cizgisi(y) -> float:
    """'Herkese Kaldı de' stratejisinin F1'i = 2p/(1+p). Pozitif sınıf çoğunluksa çok yüksektir."""
    p = float(np.mean(y))
    return 2 * p / (1 + p)
