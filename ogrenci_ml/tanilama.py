"""Tanılama (yalnızca TRAIN üzerinde): eksiklik, baskın değer, sızıntı şüphesi, çoklu doğrusallık, aykırılık."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from statsmodels.api import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .ozellikler import KolonTipleri, ogrenci_feature_engineering


def eksik_raporu(X: pd.DataFrame) -> pd.DataFrame:
    r = pd.DataFrame({"adet": X.isna().sum(), "yuzde": X.isna().mean() * 100})
    return r[r["adet"] > 0].round(2)


def baskin_deger_raporu(X: pd.DataFrame, tipler: KolonTipleri, esik: float = 95.0) -> pd.DataFrame:
    satirlar = []
    for c in list(tipler.binary) + list(tipler.kategorik):
        oran = X[c].value_counts(normalize=True).iloc[0] * 100
        satirlar.append({"sutun": c, "baskin_deger": X[c].value_counts().index[0],
                         "oran_%": round(oran, 1), "uyari": oran >= esik})
    return pd.DataFrame(satirlar)


def _fe_medyan(X: pd.DataFrame, tipler: KolonTipleri) -> pd.DataFrame:
    X_fe = ogrenci_feature_engineering(X)[tipler.sayisal_tum]
    imp = SimpleImputer(strategy="median")
    return pd.DataFrame(imp.fit_transform(X_fe), columns=tipler.sayisal_tum, index=X_fe.index)


def tek_degisken_auc(X: pd.DataFrame, y: pd.Series, tipler: KolonTipleri, esik: float = 0.95) -> pd.DataFrame:
    """Her sayısal değişkenin TEK BAŞINA AUC'si. > esik ise hedefe gizli sızıntı şüphesi."""
    X_imp = _fe_medyan(X, tipler)
    auc = {c: roc_auc_score(y, X_imp[c]) for c in X_imp.columns}
    tablo = pd.DataFrame({"auc": {c: max(a, 1 - a) for c, a in auc.items()}})
    tablo["yon"] = ["artar→kalır" if auc[c] > 0.5 else "artar→geçer" for c in tablo.index]
    tablo["suphe"] = tablo["auc"] > esik
    return tablo.sort_values("auc", ascending=False).round(3)


def vif_tablosu(X: pd.DataFrame, tipler: KolonTipleri) -> pd.Series:
    """VIF (sabit terimli). >5 dikkat, >10 ciddi; tam doğrusal kombinasyonlar ∞ çıkar (örn. toplam = a + b)."""
    X_imp = _fe_medyan(X, tipler)
    X_imp = X_imp.loc[:, X_imp.std() > 0]
    Xc = add_constant(X_imp)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")        # tam doğrusal kombinasyonlarda rank-deficient uyarısı beklenir
        v = [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])]
    return pd.Series(v, index=X_imp.columns, name="VIF").sort_values(ascending=False)


def lof_ozeti(X: pd.DataFrame, y: pd.Series, tipler: KolonTipleri) -> dict:
    """Ölçekli veride LOF. Aykırılar SİLİNMEZ (test setine aynı silme uygulanamaz), yalnızca raporlanır."""
    Z = StandardScaler().fit_transform(_fe_medyan(X, tipler))
    aykiri = LocalOutlierFactor(n_neighbors=20, contamination="auto").fit_predict(Z) == -1
    return {"aykiri_sayisi": int(aykiri.sum()), "toplam": len(Z),
            "aykirilarda_kalma_%": round(float(y.values[aykiri].mean() * 100), 1) if aykiri.any() else None,
            "normallerde_kalma_%": round(float(y.values[~aykiri].mean() * 100), 1)}
