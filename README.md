# 🎓 Öğrenci Kalma Riski: ML + Streamlit

Öğrencinin geçme/kalma riskini tahmin eden, sızıntısız (leakage-free) bir ML hattı ve Excel yüklemeden çalışan
Streamlit uygulaması.

## Klasör yapısı
```
app.py                  Streamlit uygulaması (tek öğrenci için tahmin + "neden?" açıklaması)
calistir.py             Uçtan uca eğitim (veri → taban LR → 4 rakip → ayar → ensemble → final → test → kayıt)
ogrenci_ml/             Modüller: config, veri, ozellikler, tanilama, modeller, degerlendirme, tahmin
cikti/                  Eğitimden çıkan model (final_model.joblib) ve şema (sema.json)
tests/                  Duman testleri (model yükleniyor mu, sürümler uyumlu mu)
.github/workflows/      CI: her push'ta testleri çalıştırır
```

## 1) Modeli eğit (kendi bilgisayarında / Colab'da)
```bash
pip install -r requirements-dev.txt
export VERI_YOLU="/yol/duzenlenmis_ogrenci_verisi.xlsx"      # Windows: set VERI_YOLU=...
python calistir.py          # hızlı deneme: HIZLI=1 INTERAKTIF=0 python calistir.py
```
Çıktı `cikti/` klasörüne yazılır: `final_model.joblib`, `sema.json`, `requirements.model.txt`, raporlar.

## 2) Uygulamayı yerelde çalıştır
```bash
streamlit run app.py
```

## 3) GitHub'a koymadan önce, 3 kritik kontrol
1. **Öğrenci verisi (Excel) repoya girmemeli.** `.gitignore` `*.xlsx` ve `*.csv` dosyalarını dışlar; yine de `git status` ile kontrol et.
2. **KNN modeli eğitim verisini içinde saklar.** Model dosyasını *public* repoya koyacaksan `KNN_DAHIL=0 python calistir.py`
   ile yeniden eğit (KNN yerine ExtraTrees kullanılır). Kod, KNN'li model kaydedilirken seni uyarır.
   Gerçek öğrenci verisiyse repoyu **private** tut.
3. **Sürümleri sabitle.** `cikti/requirements.model.txt` içeriğini `requirements.txt` olarak kopyala
   (scikit-learn, xgboost, lightgbm sürümleri modelle birebir aynı olmalı). Aksi halde `joblib` modeli yüklenemez
   ya da sessizce farklı sonuç verir. Uygulama ve CI sürüm uyuşmazlığını uyarır.

```bash
cp cikti/requirements.model.txt requirements.txt
git init && git add . && git commit -m "İlk sürüm"
git branch -M main && git remote add origin https://github.com/KULLANICI/ogrenci-risk.git && git push -u origin main
```

## 4) Streamlit Community Cloud'a yayınla
1. https://share.streamlit.io → **New app** → GitHub reposunu seç.
2. **Main file path:** `app.py`  ·  **Branch:** `main`
3. **Advanced settings → Python version:** modeli eğittiğin sürüme yakın seç (ör. 3.11).
4. Deploy. Repo private ise Streamlit hesabına GitHub erişimi ver.

> Not: Uygulama herkese açık bir adresse kimlik doğrulaması yoktur. Gerçek öğrenci tahminleri için erişimi
> Streamlit'in "Share → Only specific people" ayarıyla sınırla.

## Modeli yeniden eğitince
`calistir.py` → `cikti/` güncellenir → `requirements.model.txt` içeriğini `requirements.txt`'ye kopyala → commit → push.
CI testleri yeşil olmadan yayına alma.

## Sınırlılıklar
- Karar destek aracıdır; kesin hüküm değildir. Korelasyon nedensellik değildir.
- "Neden?" grafiği model-bağımsız bir duyarlılık analizidir (alanı tipik değere çekince olasılık değişimi).
- Eğitim aralığı dışındaki girdilerde uygulama uyarır; tahmin güvenilmez olabilir.
