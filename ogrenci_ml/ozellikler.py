"""Özellik mühendisliği, kolon tipi tespiti ve ön-işleme (hepsi pipeline içinde, sızıntısız).

Not: Bu modülde tanımlı fonksiyon/sınıflar joblib ile kaydedilen modelin parçasıdır; modül yolunu
(ogrenci_ml.ozellikler) değiştirirsen eski kayıtlı model yüklenemez.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config as cfg

FE_COLS = ["ortalama_akademik_puan", "calisma_uyku_orani", "basari_devam_skoru",
           "toplam_zaman_yuku", "akademik_risk_skoru"]
FE_GEREKLI_HAM = ["vize_puani", "quiz_puani", "odev_puani", "gunluk_calisma_saati", "gunluk_uyku_saati",
                  "onceki_ortalama", "devam_yuzdesi", "ders_disi_aktivite_saati", "alttan_ders_sayisi"]


def ogrenci_feature_engineering(d: pd.DataFrame) -> pd.DataFrame:
    """Satır bazlı türetilmiş özellikler. Hiçbir istatistik öğrenmez (sızıntı riski yok).

    skipna=False: girdi eksikse sonuç da NaN kalır; eksiklik 0 ile gizlenmez, medyan doldurma pipeline'da yapılır.
    """
    d = d.copy()
    d["ortalama_akademik_puan"] = d[["vize_puani", "quiz_puani", "odev_puani"]].mean(axis=1, skipna=False)
    d["calisma_uyku_orani"] = d["gunluk_calisma_saati"] / (d["gunluk_uyku_saati"] + 1)
    d["basari_devam_skoru"] = d["onceki_ortalama"] * (d["devam_yuzdesi"] / 100)
    d["toplam_zaman_yuku"] = d[["gunluk_calisma_saati", "ders_disi_aktivite_saati"]].sum(axis=1, skipna=False)
    d["akademik_risk_skoru"] = d["alttan_ders_sayisi"] / d["onceki_ortalama"].replace(0, 0.1)
    return d.replace([np.inf, -np.inf], np.nan)


@dataclass(frozen=True)
class KolonTipleri:
    """Ham sütunların tipe göre ayrımı (yalnızca eğitim verisinden çıkarılır)."""
    binary: tuple[str, ...]
    kategorik: tuple[str, ...]
    sayisal: tuple[str, ...]

    @property
    def sayisal_tum(self) -> list[str]:
        return list(self.sayisal) + FE_COLS

    @property
    def ham(self) -> list[str]:
        return list(self.sayisal) + list(self.binary) + list(self.kategorik)


def tipleri_tespit(X: pd.DataFrame) -> KolonTipleri:
    binary, kategorik, sayisal = [], [], []
    for col in X.columns:
        if X[col].nunique() == 2:
            binary.append(col)
        elif pd.api.types.is_numeric_dtype(X[col]):
            sayisal.append(col)
        else:                       # object / str / category: pandas sürümünden bağımsız
            kategorik.append(col)
    return KolonTipleri(tuple(binary), tuple(kategorik), tuple(sayisal))


def on_isleme_kur(tipler: KolonTipleri, eksiklik_gostergesi: bool = False) -> ColumnTransformer:
    """Sayısal: medyan doldur -> standartlaştır. Kategorik/binary: mod doldur -> one-hot."""
    sayisal = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=eksiklik_gostergesi)),
        ("scaler", StandardScaler()),
    ])
    ikili = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)),
    ])
    kategorik = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("sayisal", sayisal, tipler.sayisal_tum),
        ("binary", ikili, list(tipler.binary)),
        ("kategorik", kategorik, list(tipler.kategorik)),
    ], remainder="drop")


def pipeline_kur(model, tipler: KolonTipleri, onbellek: bool = True,
                 eksiklik_gostergesi: bool = False) -> Pipeline:
    """FE -> ön-işleme -> model. onbellek=True ise FE+ön-işleme aynı fold için bir kez hesaplanır."""
    memory = None
    if onbellek:
        cfg.CACHE.mkdir(parents=True, exist_ok=True)
        memory = Memory(location=str(cfg.CACHE), verbose=0)
    return Pipeline([
        ("fe", FunctionTransformer(ogrenci_feature_engineering, validate=False)),
        ("prep", on_isleme_kur(tipler, eksiklik_gostergesi)),
        ("model", model),
    ], memory=memory)


def onbellegi_temizle() -> None:
    """FE fonksiyonunu değiştirdiysen çağır: önbellek kod değişikliğini fark etmez."""
    import shutil
    shutil.rmtree(cfg.CACHE, ignore_errors=True)
