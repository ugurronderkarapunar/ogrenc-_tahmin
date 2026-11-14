"""Merkezi ayarlar. Koddaki tüm 'sihirli sayılar' burada tutulur."""
from __future__ import annotations

import os
from pathlib import Path

# ---- Çalışma modu ----------------------------------------------------------
HIZLI = os.environ.get("HIZLI", "0") == "1"           # küçük ayarlarla hızlı duman testi
INTERAKTIF = os.environ.get("INTERAKTIF", "1") == "1"   # son adımda input() ile tahmin sorulsun mu

# ---- Yollar ---------------------------------------------------------------
VERI_YOLU = os.environ.get("VERI_YOLU", "/content/duzenlenmis_ogrenci_verisi.xlsx")
CIKTI = Path(os.environ.get("CIKTI", "cikti"))
CACHE = CIKTI / "onisleme_cache"          # FE+ölçekleme çıktısı burada önbelleğe alınır (hız için)

# ---- Veri -----------------------------------------------------------------
HEDEF = "gecme_kalma_durumu"
HEDEF_ESLEME = {"Geçti": 0, "Kaldı": 1}   # pozitif sınıf = Kaldı (1)
SIZINTI_SUTUNLARI = ["final_puani", "genel_basari_yuzdesi", "harf_notu"]   # hedefi doğrudan veren sütunlar
# (sütun, alt, üst, kesin_mi): kesin kural ihlali süreci durdurur, diğerleri uyarır
ARALIK_KONTROLLERI = [
    ("devam_yuzdesi", 0, 100, True), ("gunluk_uyku_saati", 0, 24, True),
    ("gunluk_calisma_saati", 0, 24, True), ("ders_disi_aktivite_saati", 0, 24, True),
    ("yas", 10, 80, False), ("odev_puani", 0, 100, False), ("quiz_puani", 0, 100, False),
    ("vize_puani", 0, 100, False), ("onceki_ortalama", 0, 100, False),
    ("alttan_ders_sayisi", 0, 50, False),
]

# ---- Bölme ve doğrulama -----------------------------------------------------
RANDOM_STATE = 4
TEST_ORANI = 0.25
CV_KAT = 3 if HIZLI else 5
CV_TEKRAR = 1 if HIZLI else 2
N_ITER = 3 if HIZLI else 20                # her model için rastgele arama deneme sayısı
N_JOBS = int(os.environ.get("N_JOBS", "-1"))

# ---- Optimizasyon ve karar -------------------------------------------------
OPTIMIZASYON_METRIGI = "f1"                # tuning ve sıralama ölçütü (alternatif: 'average_precision')
ESIK_YONTEMI = "f1"                        # 'f1' | 'maliyet'
C_FN, C_FP = 5, 1                          # 'maliyet' yönteminde: kaçırılan "Kaldı" / gereksiz uyarı maliyeti
KNN_DAHIL = os.environ.get("KNN_DAHIL", "0") == "0"   # KNN eğitim verisini modelin İÇİNDE saklar: yayınlayacaksan 0 yap
ENSEMBLE_LR_DAHIL = False                  # lojistik taban modeli ensemble üyesi olsun mu

# ---- İstatistiksel güvenilirlik ---------------------------------------------
BOOTSTRAP_B = 300 if HIZLI else 2000
MIN_SINIF_ORNEGI = 30                      # test setinde sınıf başına asgari örnek
