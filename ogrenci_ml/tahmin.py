"""Kayıtlı modeli yükleyip yeni öğrenci için tahmin: programatik (tahmin_et) ve etkileşimli (input) kullanım."""
from __future__ import annotations

import json
import logging
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .ozellikler import KolonTipleri

log = logging.getLogger(__name__)
MODEL_DOSYASI, SEMA_DOSYASI = "final_model.joblib", "sema.json"
KUTUPHANELER = ["scikit-learn", "xgboost", "lightgbm", "numpy", "pandas", "scipy", "joblib"]


def surumler() -> dict:
    """Modelin eğitildiği ortamdaki sürümler. Pickle dosyaları sürüme bağlıdır."""
    return {k: metadata.version(k) for k in KUTUPHANELER}


def surum_uyumsuzlugu(sema: dict) -> list[str]:
    """Çalışma ortamı, eğitim ortamından farklıysa uyarı mesajları döner."""
    kayitli, simdi = sema.get("surumler", {}), surumler()
    return [f"{k}: modelin eğitildiği sürüm {v}, burada {simdi.get(k)}" for k, v in kayitli.items()
            if simdi.get(k) != v and k in ("scikit-learn", "xgboost", "lightgbm")]


def knn_iceriyor_mu(model) -> bool:
    """KNN, eğitim satırlarını modelin içinde saklar: model dosyası paylaşılırsa veri de paylaşılmış olur.
    (repr() çıktısı 700 karakterde kesildiği için metin araması yerine model ağacı gezilir.)"""
    from sklearn.neighbors import KNeighborsClassifier

    def bak(e) -> bool:
        if isinstance(e, KNeighborsClassifier):
            return True
        if hasattr(e, "steps"):                                   # Pipeline
            return any(bak(adim) for _, adim in e.steps)
        for nitelik in ("estimators_", "estimators"):             # Voting / Stacking
            uyeler = getattr(e, nitelik, None)
            if uyeler:
                return any(bak(u[1] if isinstance(u, tuple) else u) for u in uyeler)
        return False

    return bak(getattr(model, "pipeline", model))


# ------------------------------------------------------------------ şema (formun Excel'siz kurulması için)
def sema_cikar(X_train: pd.DataFrame, tipler: KolonTipleri, oof_kalibre: np.ndarray | None = None) -> dict:
    sema = {"sutun_sirasi": list(X_train.columns), "sayisal": {}, "secimli": {}}
    for c in tipler.sayisal:
        s = X_train[c]
        sema["sayisal"][c] = {"min": float(s.min()), "max": float(s.max()), "medyan": float(s.median()),
                              "tam_sayi": bool((s.dropna() % 1 == 0).all())}
    for c in list(tipler.binary) + list(tipler.kategorik):
        mod = X_train[c].mode(dropna=True).iloc[0]
        sema["secimli"][c] = {"secenekler": X_train[c].dropna().unique().tolist(),
                              "sayisal": bool(pd.api.types.is_numeric_dtype(X_train[c])),
                              "varsayilan": mod.item() if hasattr(mod, "item") else mod}
    if oof_kalibre is not None:
        sema["p_dagilim"] = np.percentile(oof_kalibre, np.arange(0, 101)).round(4).tolist()
    return sema


def kaydet(model, sema: dict, klasor: Path) -> None:
    klasor = Path(klasor); klasor.mkdir(parents=True, exist_ok=True)
    sema = {**sema, "surumler": surumler()}
    joblib.dump(model, klasor / MODEL_DOSYASI)
    # Yayın için: bu dosyayı repo köküne requirements.txt olarak kopyala (pickle sürüme bağlıdır)
    (klasor / "requirements.model.txt").write_text(
        "\n".join(f"{k}=={v}" for k, v in sema["surumler"].items()) + "\nstreamlit\nmatplotlib\n", encoding="utf-8")
    if knn_iceriyor_mu(model):
        log.warning("⚠️ GİZLİLİK: Model KNN içeriyor ve eğitim verisini saklar. Public GitHub'a koyma; "
                    "KNN_DAHIL=0 ile yeniden eğit ya da private repo kullan.")
    (klasor / SEMA_DOSYASI).write_text(json.dumps(sema, ensure_ascii=False, indent=2), encoding="utf-8")


def yukle(klasor: Path):
    klasor = Path(klasor)
    model = joblib.load(klasor / MODEL_DOSYASI)
    sema = json.loads((klasor / SEMA_DOSYASI).read_text(encoding="utf-8"))
    return model, sema


# ------------------------------------------------------------------ tahmin
def _satir(sema: dict, girdi: dict) -> pd.DataFrame:
    """Girdiden tek satırlık DataFrame. Eksik değer np.nan (None DEĞİL): imputer yalnızca NaN'ı eksik sayar."""
    kolonlar = {}
    for c in sema["sutun_sirasi"]:
        v = girdi.get(c, np.nan)
        v = np.nan if v is None else v
        if c in sema["sayisal"]:
            kolonlar[c] = pd.Series([v], dtype="float64")
        elif sema["secimli"][c]["sayisal"]:
            kolonlar[c] = pd.Series([v], dtype="float64")
        else:
            kolonlar[c] = pd.Series([v], dtype="object")
    return pd.DataFrame(kolonlar)


def tahmin_et(model, sema: dict, girdi: dict) -> dict:
    """Tek öğrenci için kalibre kalma olasılığı, karar ve uyarılar."""
    uyarilar = []
    for c, b in sema["sayisal"].items():
        v = girdi.get(c)
        if v is not None and not (b["min"] <= v <= b["max"]):
            uyarilar.append(f"{c}={v} eğitim aralığının ({b['min']:.4g}–{b['max']:.4g}) dışında: tahmin güvenilmez olabilir")
    bos = [c for c in sema["sutun_sirasi"] if girdi.get(c) is None]
    if bos:
        uyarilar.append(f"{len(bos)} alan bilinmiyor (medyan/mod ile dolduruldu): {bos}")
    p = float(model.olasilik(_satir(sema, girdi))[0])
    sonuc = {"olasilik": p, "esik": model.esik, "riskli": bool(p >= model.esik), "uyarilar": uyarilar}
    if "p_dagilim" in sema:
        sonuc["yuzdelik"] = float(np.searchsorted(np.array(sema["p_dagilim"]), p, side="right"))
    return sonuc


# ------------------------------------------------------------------ etkileşimli
def _sayi_oku(metin: str) -> float:
    return float(metin.strip().replace(",", "."))      # Türkçe ondalık virgülü


def interaktif_tahmin(model, sema: dict, giris=input, cikis=print) -> dict:
    """Her değişken için sorar. Boş bırakılırsa 'bilinmiyor' sayılır."""
    cikis("=" * 60); cikis("YENİ ÖĞRENCİ RİSK ANALİZİ  (boş bırak = bilinmiyor)"); cikis("=" * 60)
    girdi: dict = {}
    for c in sema["sutun_sirasi"]:
        if c in sema["sayisal"]:
            b = sema["sayisal"][c]
            while True:
                ham = giris(f"{c} [{b['min']:.4g} – {b['max']:.4g}]: ").strip()
                if ham == "":
                    girdi[c] = None; break
                try:
                    v = _sayi_oku(ham)
                    if b["tam_sayi"] and v != int(v):
                        cikis("  Bu alan tam sayı olmalı."); continue
                    girdi[c] = v; break
                except ValueError:
                    cikis("  Geçerli bir sayı gir.")
        else:
            sec = sema["secimli"][c]["secenekler"]
            while True:
                ham = giris(f"{c} {dict(enumerate(sec, 1))}: ").strip()
                if ham == "":
                    girdi[c] = None; break
                if ham.isdigit() and 1 <= int(ham) <= len(sec):
                    girdi[c] = sec[int(ham) - 1]; break
                cikis("  Listeden numara seç.")
    s = tahmin_et(model, sema, girdi)
    cikis("-" * 60)
    cikis(f"Kalma olasılığı (kalibre): %{s['olasilik'] * 100:.1f}   | karar eşiği: %{s['esik'] * 100:.0f}")
    if "yuzdelik" in s:
        cikis(f"Eğitim verisindeki öğrencilerin %{min(s['yuzdelik'], 100):.0f}'inden daha riskli")
    cikis("SONUÇ: " + ("RİSKLİ ÖĞRENCİ, müdahale planla" if s["riskli"] else "Risk düşük (eşiğin altında)"))
    for u in s["uyarilar"]:
        cikis("⚠️ " + u)
    cikis("=" * 60)
    return s


def tipik_girdi(sema: dict) -> dict:
    """'Tipik öğrenci': sayısal alanlarda medyan, kategorik alanlarda mod."""
    g = {c: b["medyan"] for c, b in sema["sayisal"].items()}
    g.update({c: v.get("varsayilan", v["secenekler"][0]) for c, v in sema["secimli"].items()})
    return g


def yerel_etki(model, sema: dict, girdi: dict) -> pd.Series:
    """Model-bağımsız açıklama (ceteris paribus): her alanı tek tek 'tipik' değerine çekince kalma olasılığı
    kaç puan değişir? Pozitif = bu değer riski, tipik öğrenciye göre ARTIRIYOR. Etkileşimleri tek tek ayrıştırmaz."""
    tipik = tipik_girdi(sema); alanlar = list(sema["sutun_sirasi"])
    satirlar = [_satir(sema, girdi)]
    for c in alanlar:
        g = dict(girdi); g[c] = tipik[c]
        satirlar.append(_satir(sema, g))
    p = model.olasilik(pd.concat(satirlar, ignore_index=True))
    etki = pd.Series(p[0] - p[1:], index=alanlar)
    for c in alanlar:                                   # bilinmeyen alanın etkisi tanımsız: 0
        if girdi.get(c) is None:
            etki[c] = 0.0
    return etki
