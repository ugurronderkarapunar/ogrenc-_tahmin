"""Öğrenci kalma riski: Streamlit uygulaması (Excel yüklemeden, kayıtlı modelle tahmin).

Çalıştırma:  streamlit run app.py
Gerekli:     cikti/final_model.joblib ve cikti/sema.json  (calistir.py ile üretilir)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

KOK = Path(__file__).resolve().parent
if str(KOK) not in sys.path:
    sys.path.insert(0, str(KOK))          # joblib, ogrenci_ml modülünü buradan bulur

from ogrenci_ml import tahmin  # noqa: E402

st.set_page_config(page_title="Öğrenci Risk Tahmini", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner="Model yükleniyor…")
def model_yukle():
    return tahmin.yukle(KOK / "cikti")


try:
    model, sema = model_yukle()
except FileNotFoundError:
    st.error("Model dosyası bulunamadı (`cikti/final_model.joblib`, `cikti/sema.json`).\n\n"
             "Önce `python calistir.py` ile modeli eğit, sonra bu klasörleri repoya ekle.")
    st.stop()

# ----------------------------------------------------------------------- sürüm uyumu
for uyari in tahmin.surum_uyumsuzlugu(sema):
    st.warning(f"Kütüphane sürümü uyuşmuyor ({uyari}). Sonuçlar güvenilmez olabilir; "
               "`requirements.txt` dosyasını modelin eğitildiği sürümlere sabitle.")

# ----------------------------------------------------------------------- girdi formu
st.sidebar.header("Öğrenci bilgileri")
girdi: dict = {}
for c in sema["sutun_sirasi"]:
    if c in sema["sayisal"]:
        b = sema["sayisal"][c]
        lo, hi = b["min"], b["max"]
        if b["tam_sayi"]:
            girdi[c] = st.sidebar.slider(c, int(lo), int(hi), int(round(b["medyan"])), step=1)
        else:
            adim = float(f"{(hi - lo) / 100:.2g}") or 0.01
            girdi[c] = st.sidebar.slider(c, float(lo), float(hi), float(min(max(b["medyan"], lo), hi)), step=adim)
    else:
        sec = sema["secimli"][c]
        varsayilan = sec["secenekler"].index(sec["varsayilan"]) if sec.get("varsayilan") in sec["secenekler"] else 0
        girdi[c] = st.sidebar.selectbox(c, sec["secenekler"], index=varsayilan)

bilinmeyen = st.sidebar.multiselect("Bilinmeyen alanlar (model tipik değerle doldurur)", sema["sutun_sirasi"])
for c in bilinmeyen:
    girdi[c] = None

# ----------------------------------------------------------------------- tahmin
sonuc = tahmin.tahmin_et(model, sema, girdi)
p = sonuc["olasilik"]

st.title("🎓 Öğrenci Kalma Riski")
st.caption("Soldaki değerleri değiştir, sonuç anında güncellenir. Karar destek aracıdır; kesin sonuç değildir.")

k1, k2, k3 = st.columns(3)
k1.metric("Kalma olasılığı (kalibre)", f"%{p * 100:.0f}")
k2.metric("Karar eşiği", f"%{sonuc['esik'] * 100:.0f}")
if "yuzdelik" in sonuc:
    k3.metric("Eğitim verisindeki sıralama", f"%{min(sonuc['yuzdelik'], 100):.0f}'inden daha riskli")
st.progress(min(max(p, 0.0), 1.0))

if sonuc["riskli"]:
    st.error("**Riskli öğrenci**: model eşiği aşıldı, müdahale planlanması önerilir.")
else:
    st.success("Risk eşiğin altında.")
for u in sonuc["uyarilar"]:
    st.warning(u)

# ----------------------------------------------------------------------- neden?
st.subheader("Bu tahmin neden böyle?")
etki = tahmin.yerel_etki(model, sema, girdi)
en_etkili = etki.reindex(etki.abs().sort_values(ascending=False).index).head(8).sort_values()
st.bar_chart((en_etkili * 100).rename("olasılık puanı"), horizontal=True)
st.caption("Her alanı tek tek 'tipik öğrenci' değerine (medyan/mod) çekince kalma olasılığı kaç **puan** değişir? "
           "Pozitif (sağ) = bu değer riski tipik öğrenciye göre artırıyor. Alanlar birbirine bağlı olabilir; "
           "bu, nedensellik değil model davranışıdır.")

with st.expander("Girilen değerler ve model bilgisi"):
    st.dataframe(pd.DataFrame({"değer": {c: str(v) for c, v in girdi.items()}}))
    st.json({"eğitim_sürümleri": sema.get("surumler", {}), "eşik": sonuc["esik"]})
