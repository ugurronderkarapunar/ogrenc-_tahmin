"""Veri yükleme, doğrulama, temizlik ve train/test bölme.

Bu modülde yalnızca istatistik ÖĞRENMEYEN işlemler vardır (sızıntı riski yok). Doldurma/ölçekleme pipeline'dadır.
"""
from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as cfg
from .ozellikler import FE_GEREKLI_HAM

log = logging.getLogger(__name__)


class VeriHatasi(ValueError):
    """Veri sözleşmesi ihlali (eksik sütun, bilinmeyen hedef değeri, imkânsız değer)."""


def yukle(yol: str = cfg.VERI_YOLU) -> pd.DataFrame:
    df = pd.read_excel(yol)
    log.info("Veri yüklendi: %d satır, %d sütun", *df.shape)
    return df


def hazirla(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Sızıntı sütunlarını at, hedefi 0/1 yap, imkânsız değerleri yakala, tekrarlı satırları sil."""
    rapor: dict = {"ham_satir": len(df)}
    df = df.copy()

    atilan = [c for c in cfg.SIZINTI_SUTUNLARI if c in df.columns]
    df = df.drop(columns=atilan)
    rapor["atilan_sizinti_sutunlari"] = atilan

    eksik = [c for c in [cfg.HEDEF] + FE_GEREKLI_HAM if c not in df.columns]
    if eksik:
        raise VeriHatasi(f"Beklenen sütunlar yok: {eksik}")

    kodlu = df[cfg.HEDEF].map(cfg.HEDEF_ESLEME)
    if kodlu.isna().any():
        bilinmeyen = df.loc[kodlu.isna(), cfg.HEDEF].unique().tolist()
        raise VeriHatasi(f"Hedefte eşleşmeyen değerler: {bilinmeyen}")
    df[cfg.HEDEF] = kodlu.astype(int)

    for col, lo, hi, kesin in cfg.ARALIK_KONTROLLERI:
        if col not in df.columns:
            continue
        ihlal = int((~df[col].dropna().between(lo, hi)).sum())
        if ihlal:
            mesaj = f"{col}: {ihlal} değer [{lo}, {hi}] dışında"
            if kesin:
                raise VeriHatasi(mesaj)
            log.warning("Uyarı: %s", mesaj)

    rapor["tekrar_satir"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)      # split'ten ÖNCE: aynı satır hem train hem test'e düşmesin
    rapor["temiz_satir"] = len(df)
    log.info("Temizlik: %s", rapor)
    return df, rapor


def bol(df: pd.DataFrame):
    """Katmanlı (stratified) train/test bölme. Test seti yalnızca final değerlendirmede kullanılır."""
    X, y = df.drop(columns=cfg.HEDEF), df[cfg.HEDEF]
    return train_test_split(X, y, test_size=cfg.TEST_ORANI, random_state=cfg.RANDOM_STATE, stratify=y)
