"""Kalibrasyon, eşik seçimi, test metrikleri, bootstrap güven aralıkları ve istatistiksel güvenilirlik raporu."""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, accuracy_score,
                             average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_predict

from . import config as cfg

log = logging.getLogger(__name__)


def logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------ nihai model (kalibre + eşikli)
class KalibreModel:
    """Eğitilmiş pipeline + Platt kalibratörü + karar eşiği. Tek joblib dosyası olarak saklanır."""

    def __init__(self, pipeline, kalibrator, esik: float, meta: dict | None = None):
        self.pipeline, self.kalibrator, self.esik, self.meta = pipeline, kalibrator, float(esik), meta or {}

    def ham_olasilik(self, X) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    def olasilik(self, X) -> np.ndarray:
        """Kalibre 'Kaldı' olasılığı."""
        z = logit(self.ham_olasilik(X)).reshape(-1, 1)
        return self.kalibrator.predict_proba(z)[:, 1]

    def predict_proba(self, X) -> np.ndarray:
        p = self.olasilik(X)
        return np.c_[1 - p, p]

    def predict(self, X) -> np.ndarray:
        return (self.olasilik(X) >= self.esik).astype(int)


def esik_sec(y, p, yontem: str = cfg.ESIK_YONTEMI) -> float:
    """Eşik YALNIZCA eğitim OOF tahminlerinden seçilir (test setine bakılmaz)."""
    y = np.asarray(y)
    esikler = np.arange(0.05, 0.951, 0.01)
    if yontem == "f1":
        skor = [f1_score(y, p >= t, zero_division=0) for t in esikler]
        return float(esikler[int(np.argmax(skor))])
    maliyet = [cfg.C_FN * ((p < t) & (y == 1)).sum() + cfg.C_FP * ((p >= t) & (y == 0)).sum() for t in esikler]
    return float(esikler[int(np.argmin(maliyet))])


def kalibre_et(pipeline, X, y, cv, meta: dict | None = None) -> tuple[KalibreModel, np.ndarray]:
    """OOF tahmin -> Platt kalibrasyonu -> eşik seçimi -> tüm train'e fit. Döner: (model, kalibre OOF olasılıklar)."""
    oof = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba", n_jobs=cfg.N_JOBS)[:, 1]
    kal = LogisticRegression(C=1e6, max_iter=1000).fit(logit(oof).reshape(-1, 1), y)
    oof_kal = kal.predict_proba(logit(oof).reshape(-1, 1))[:, 1]
    esik = esik_sec(y, oof_kal)
    son = clone(pipeline).fit(X, y)
    son.set_params(memory=None)                 # kaydedilen modelde önbellek yolu taşınmasın
    return KalibreModel(son, kal, esik, meta), oof_kal


# ------------------------------------------------------------------ metrikler
METRIK_FN = {
    "roc_auc": lambda y, p, t: roc_auc_score(y, p),
    "pr_auc": lambda y, p, t: average_precision_score(y, p),
    "f1": lambda y, p, t: f1_score(y, p >= t, zero_division=0),
    "recall": lambda y, p, t: recall_score(y, p >= t, zero_division=0),
    "precision": lambda y, p, t: precision_score(y, p >= t, zero_division=0),
}


def metrik_ozeti(y, p, esik: float) -> dict:
    y = np.asarray(y); tahmin = (p >= esik).astype(int)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "f1": f1_score(y, tahmin), "precision": precision_score(y, tahmin, zero_division=0),
            "recall": recall_score(y, tahmin), "accuracy": accuracy_score(y, tahmin),
            "balanced_acc": balanced_accuracy_score(y, tahmin), "mcc": matthews_corrcoef(y, tahmin)}


def bootstrap_ci(y, p, metrik: str, esik: float, B: int = cfg.BOOTSTRAP_B, seed: int = cfg.RANDOM_STATE):
    """Test örneğini yerine koyarak yeniden çekip metriğin %95 güven aralığını verir. Döner: (nokta, alt, üst)."""
    y = np.asarray(y); p = np.asarray(p); fn = METRIK_FN[metrik]
    rng = np.random.default_rng(seed); sk = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            sk.append(fn(y[i], p[i], esik))
    lo, hi = np.percentile(sk, [2.5, 97.5])
    return float(fn(y, p, esik)), float(lo), float(hi)


def eslesik_bootstrap_fark(y, p_a, p_b, esik_a: float, esik_b: float, metrik: str,
                           B: int = cfg.BOOTSTRAP_B, seed: int = cfg.RANDOM_STATE):
    """Aynı bootstrap örneklerinde (a − b) farkı. Döner: (fark, alt, üst, iki yönlü yaklaşık p)."""
    y = np.asarray(y); p_a = np.asarray(p_a); p_b = np.asarray(p_b); fn = METRIK_FN[metrik]
    rng = np.random.default_rng(seed); f = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            f.append(fn(y[i], p_a[i], esik_a) - fn(y[i], p_b[i], esik_b))
    f = np.array(f); lo, hi = np.percentile(f, [2.5, 97.5])
    p_deger = float(min(1.0, 2 * min((f <= 0).mean(), (f >= 0).mean())))
    return float(fn(y, p_a, esik_a) - fn(y, p_b, esik_b)), float(lo), float(hi), p_deger


def kalibrasyon_istatistikleri(y, p) -> dict:
    """Eğim ≈ 1 ve kesişim ≈ 0 ideal. Brier skill > 0: 'hep prevalansı söyleyen' modelden iyi."""
    y = np.asarray(y)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    prev = y.mean(); ref = prev * (1 - prev)
    return {"egim": float(lr.coef_[0, 0]), "kesisim": float(lr.intercept_[0]),
            "brier": float(brier_score_loss(y, p)), "brier_skill": float(1 - brier_score_loss(y, p) / ref)}


# ------------------------------------------------------------------ güvenilirlik raporu
def guvenilirlik_raporu(y_train, oof_p, esik: float, y_test, p_test, p_base_test, esik_base: float,
                        X_test: pd.DataFrame | None = None, grup_sutunlari: tuple = ()) -> pd.DataFrame:
    """Finalin istatistiksel güvenilirlik karnesi. Her satır bir kontrol: değer, kriter, durum."""
    y_test = np.asarray(y_test); satirlar = []

    def ekle(kontrol, deger, kriter, gecti):
        satirlar.append({"kontrol": kontrol, "deger": deger, "kriter": kriter, "durum": "✅" if gecti else "⚠️"})

    n_poz, n_neg = int((y_test == 1).sum()), int((y_test == 0).sum())
    ekle("Test örnek sayısı (Kaldı / Geçti)", f"{n_poz} / {n_neg}", f"her sınıf ≥ {cfg.MIN_SINIF_ORNEGI}",
         min(n_poz, n_neg) >= cfg.MIN_SINIF_ORNEGI)

    roc = bootstrap_ci(y_test, p_test, "roc_auc", esik); pr = bootstrap_ci(y_test, p_test, "pr_auc", esik)
    f1 = bootstrap_ci(y_test, p_test, "f1", esik); prev = float(y_test.mean())
    ekle("ROC-AUC %95 GA", f"{roc[0]:.3f} [{roc[1]:.3f}, {roc[2]:.3f}]", "alt sınır > 0.5", roc[1] > 0.5)
    ekle("PR-AUC %95 GA", f"{pr[0]:.3f} [{pr[1]:.3f}, {pr[2]:.3f}]", f"alt sınır > prevalans ({prev:.2f})", pr[1] > prev)
    ekle("F1 GA genişliği", f"{f1[2] - f1[1]:.3f}", "≤ 0.12 (sonuç kararlı)", (f1[2] - f1[1]) <= 0.12)

    f1_oof = f1_score(y_train, np.asarray(oof_p) >= esik); pr_oof = average_precision_score(y_train, oof_p)
    ekle("F1: CV(OOF) vs test farkı", f"{f1_oof:.3f} vs {f1[0]:.3f}", "|fark| ≤ 0.05 (aşırı uyum yok)", abs(f1_oof - f1[0]) <= 0.05)
    ekle("PR-AUC: CV(OOF) vs test farkı", f"{pr_oof:.3f} vs {pr[0]:.3f}", "|fark| ≤ 0.05", abs(pr_oof - pr[0]) <= 0.05)

    for m in ("pr_auc", "f1"):
        d, lo, hi, p = eslesik_bootstrap_fark(y_test, p_test, p_base_test, esik, esik_base, m)
        ekle(f"Final − lojistik taban ({m})", f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}], p≈{p:.3f}",
             "GA 0'ı dışlamalı (aksi halde basit model tercih edilir)", lo > 0)

    k = kalibrasyon_istatistikleri(y_test, p_test)
    ekle("Kalibrasyon eğimi", f"{k['egim']:.2f} (kesişim {k['kesisim']:+.2f})", "0.8 – 1.2", 0.8 <= k["egim"] <= 1.2)
    ekle("Brier skill skoru", f"{k['brier_skill']:.3f}", "> 0", k["brier_skill"] > 0)

    if X_test is not None:
        tahmin = (np.asarray(p_test) >= esik).astype(int)
        for g in grup_sutunlari:
            if g not in X_test.columns:
                continue
            recall = {}
            for deger, idx in X_test.reset_index(drop=True).groupby(g).groups.items():
                yy = y_test[list(idx)]
                if (yy == 1).sum() >= cfg.MIN_SINIF_ORNEGI:
                    recall[deger] = recall_score(yy, tahmin[list(idx)], zero_division=0)
            if len(recall) >= 2:
                fark = max(recall.values()) - min(recall.values())
                ekle(f"Alt grup recall farkı ({g})", f"{fark:.2f}", "≤ 0.15", fark <= 0.15)
    return pd.DataFrame(satirlar)


def grafik_ciz(y, p, esik: float, yol=None):
    """ROC, PR, kalibrasyon eğrisi ve karışıklık matrisi (tek şekil)."""
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
    RocCurveDisplay.from_predictions(y, p, ax=ax[0]); ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set_title("ROC")
    PrecisionRecallDisplay.from_predictions(y, p, ax=ax[1]); ax[1].axhline(np.mean(y), color="gray", ls=":")
    ax[1].set_title("Precision–Recall (kesik: rastgele)")
    CalibrationDisplay.from_predictions(y, p, n_bins=8, ax=ax[2]); ax[2].set_title("Kalibrasyon (köşegen = ideal)")
    ConfusionMatrixDisplay.from_predictions(y, (np.asarray(p) >= esik).astype(int), display_labels=["Geçti", "Kaldı"],
                                            cmap="Blues", ax=ax[3], colorbar=False)
    ax[3].set_title(f"Karışıklık matrisi (eşik {esik:.2f})"); ax[3].grid(False)
    fig.tight_layout()
    if yol:
        fig.savefig(yol, dpi=140, bbox_inches="tight")
    return fig
