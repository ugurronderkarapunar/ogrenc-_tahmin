# ============================================================================
# ÖĞRENCİ GEÇME/KALMA: UÇTAN UCA ÇALIŞTIRMA (modülleri sırayla çağırır)
# "# %%" hücreleriyle Colab / VS Code / Jupytext'te hücre hücre ya da tek seferde çalışır.
# Akış: veri -> split -> tanılama -> taban LR -> 4 rakip (F1 sıralı) -> ayar -> ensemble
#       -> final (kalibre+eşik) -> gizli test -> güvenilirlik -> kayıt -> kullanıcı girdisiyle tahmin
# ============================================================================

# %% MODÜL 0: Kurulum
import logging, os, sys, warnings
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

sys.path.insert(0, os.getcwd())
from ogrenci_ml import config as cfg
from ogrenci_ml import degerlendirme as dg, modeller as md, ozellikler as oz, tahmin, tanilama, veri

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
cfg.CIKTI.mkdir(parents=True, exist_ok=True)

def goster(x):
    try:
        display(x)          # Jupyter/Colab
    except NameError:
        print(x.to_string() if hasattr(x, "to_string") else x)


# %% MODÜL 1: Veri yükleme, doğrulama, temizlik (istatistik öğrenmeyen işlemler)
df, temizlik = veri.hazirla(veri.yukle())
print("Hedef dağılımı (%):\n", (df[cfg.HEDEF].value_counts(normalize=True) * 100).round(1).to_string())


# %% MODÜL 2: Train/Test bölme. Test seti bundan sonra MODÜL 10'a kadar kilitli.
X_train, X_test, y_train, y_test = veri.bol(df)
tipler = oz.tipleri_tespit(X_train)
print(f"Train {X_train.shape} | Test {X_test.shape}")
print("Binary:", tipler.binary, "| Kategorik:", tipler.kategorik, "\nSayısal:", tipler.sayisal)


# %% MODÜL 3: Tanılama (yalnızca TRAIN)
print("--- Eksik değerler ---");           goster(tanilama.eksik_raporu(X_train))
print("--- Baskın değer ---");             goster(tanilama.baskin_deger_raporu(X_train, tipler))
auc_tab = tanilama.tek_degisken_auc(X_train, y_train, tipler)
print("--- Tek değişkenli AUC (sızıntı testi, >0.95 şüpheli) ---"); goster(auc_tab)
if auc_tab["suphe"].any():
    print("⚠️ SIZINTI ŞÜPHESİ:", list(auc_tab.index[auc_tab["suphe"]]),
          "\n   Bu değişkenin tahmin anında gerçekten bilinip bilinmediğini kontrol et.")
print("--- VIF (tam doğrusal kombinasyonlar ∞ çıkar; ağaçlarda sorun değil, LR'de L2 ile yönetilir) ---")
goster(tanilama.vif_tablosu(X_train, tipler).round(1))
print("--- LOF ---", tanilama.lof_ozeti(X_train, y_train, tipler))


# %% MODÜL 4: Doğrulama şemaları (aynı fold'lar tüm karşılaştırmalarda kullanılır)
cv_ayar = StratifiedKFold(cfg.CV_KAT, shuffle=True, random_state=cfg.RANDOM_STATE)
cv_rep = RepeatedStratifiedKFold(n_splits=cfg.CV_KAT, n_repeats=cfg.CV_TEKRAR, random_state=cfg.RANDOM_STATE)


# %% MODÜL 5: Taban model = lojistik regresyon (+ Dummy referansı), medyan doldurma
from sklearn.dummy import DummyClassifier
taban_adaylar = {
    "Dummy (prior)": oz.pipeline_kur(DummyClassifier(strategy="prior"), tipler),
    "LogReg (taban)": oz.pipeline_kur(md.taban_model(), tipler),
}
taban_tablo, taban_fold = md.cv_karsilastir(taban_adaylar, X_train, y_train, cv_rep)
goster(md.ozet_goster(taban_tablo))
F1_TABAN = md.f1_taban_cizgisi(y_train)
print(f"'Herkese Kaldı de' stratejisinin F1'i: {F1_TABAN:.3f} (prevalans %{y_train.mean()*100:.0f})")
if y_train.mean() > 0.5:
    print("⚠️ Pozitif sınıf (Kaldı) çoğunluk: F1 (eşik 0.5) bu veride YANILTICI olabilir. Modelleri bu taban çizgisiyle,\n"
          "   ayrıca PR-AUC / MCC / balanced accuracy ile birlikte yorumla. Ölçüt olarak cfg.OPTIMIZASYON_METRIGI='average_precision' önerilir.")


# %% MODÜL 6: 4 rakip model (lojistikten farklı aileler), F1'e göre sıralı (henüz ayarsız)
rakipler = md.rakip_modeller(y_train)
rakip_adaylar = {ad: oz.pipeline_kur(m, tipler) for ad, m in rakipler.items()}
rakip_tablo, rakip_fold = md.cv_karsilastir(rakip_adaylar, X_train, y_train, cv_rep, sirala="f1")
print("F1'e göre sıralama (ayarsız):"); goster(md.ozet_goster(rakip_tablo))
kotu = [ad for ad in rakip_tablo.index if rakip_tablo.loc[ad, "f1"] <= F1_TABAN + 0.02]
if kotu:
    print(f"⚠️ Şu modellerin F1'i 'herkese Kaldı de' taban çizgisine ({F1_TABAN:.3f}) yakın/altında: {kotu}")


# %% MODÜL 7: Hiperparametre ayarı (RandomizedSearchCV, ölçüt: F1) ve ayarlı modellerin yeniden sıralanması
aramalar = {ad: md.ayarla(ad, rakipler[ad], md.param_uzaylari()[ad], tipler, X_train, y_train, cv_ayar)
            for ad in rakip_tablo.index}                     # F1 sırasıyla
ayarli = {ad: md.ayarli_model(rakipler[ad], a) for ad, a in aramalar.items()}
for ad, a in aramalar.items():
    print(f"{ad:13s} en iyi CV F1={a.best_score_:.3f} | {a.best_params_}")

ayarli_adaylar = {f"{ad} (ayarlı)": oz.pipeline_kur(m, tipler) for ad, m in ayarli.items()}
ayarli_tablo, ayarli_fold = md.cv_karsilastir(ayarli_adaylar, X_train, y_train, cv_rep)
print("Ayarlı modeller, F1'e göre (not: parametreler aynı CV'de seçildiği için hafif iyimser):")
goster(md.ozet_goster(ayarli_tablo))


# %% MODÜL 8: Ensemble (ayarlı 4 modelle soft voting ve stacking) ve final seçimi
ens_adaylar = {"Voting (soft)": md.ensemble_kur("voting", ayarli, tipler),
               "Stacking (LR meta)": md.ensemble_kur("stacking", ayarli, tipler)}
ens_tablo, ens_fold = md.cv_karsilastir(ens_adaylar, X_train, y_train, cv_rep)
print("Ensemble karşılaştırması (F1'e göre):"); goster(md.ozet_goster(ens_tablo))

FINAL_AD = ens_tablo.index[0]
en_iyi_tekil = ayarli_tablo.index[0]
tum_fold = {**taban_fold, **ayarli_fold, **ens_fold}
for kiyas in ["LogReg (taban)", en_iyi_tekil]:
    d, se = md.eslesik_fark(tum_fold[FINAL_AD]["f1"], tum_fold[kiyas]["f1"])
    print(f"{FINAL_AD} − {kiyas}: ΔF1 = {d:+.4f} (SE {se:.4f}) → "
          + ("anlamlı" if abs(d) > 2 * se else "anlamlı DEĞİL: basit model tercih edilebilir"))
print("Final model:", FINAL_AD)


# %% MODÜL 9: Final model = kalibrasyon (Platt) + eşik (yalnızca train OOF) + tüm train'e fit
final, oof_final = dg.kalibre_et(ens_adaylar[FINAL_AD], X_train, y_train, cv_ayar, meta={"ad": FINAL_AD})
taban, oof_taban = dg.kalibre_et(oz.pipeline_kur(md.taban_model(), tipler), X_train, y_train, cv_ayar,
                                 meta={"ad": "LogReg (taban)"})
print(f"Seçilen eşikler → final: {final.esik:.2f} | taban LR: {taban.esik:.2f}")


# %% MODÜL 10: GİZLİ TEST SETİ (ilk ve tek kullanım)
p_final, p_taban = final.olasilik(X_test), taban.olasilik(X_test)
test_tablo = pd.DataFrame({FINAL_AD: dg.metrik_ozeti(y_test, p_final, final.esik),
                           "LogReg (taban)": dg.metrik_ozeti(y_test, p_taban, taban.esik)}).T.round(3)
print("Test seti (modellerin hiç görmediği veri):"); goster(test_tablo)
fig = dg.grafik_ciz(y_test, p_final, final.esik, cfg.CIKTI / "final_test_grafikleri.png"); plt.show()


# %% MODÜL 11: İstatistiksel güvenilirlik karnesi
karne = dg.guvenilirlik_raporu(y_train, oof_final, final.esik, y_test, p_final, p_taban, taban.esik,
                               X_test=X_test, grup_sutunlari=("cinsiyet", "ebeveyn_egitim_durumu", "internet_erisimi"))
goster(karne)
gecen = int((karne["durum"] == "✅").sum())
print(f"\nGüvenilirlik karnesi: {gecen}/{len(karne)} kontrol geçti.")
if (karne["durum"] == "⚠️").any():
    print("⚠️ Dikkat edilecek kontroller:", list(karne.loc[karne["durum"] == "⚠️", "kontrol"]))


# %% MODÜL 12: Kayıt (model + şema + raporlar)
sema = tahmin.sema_cikar(X_train, tipler, oof_final)
tahmin.kaydet(final, sema, cfg.CIKTI)
test_tablo.to_csv(cfg.CIKTI / "test_metrikleri.csv"); karne.to_csv(cfg.CIKTI / "guvenilirlik_karnesi.csv", index=False)
ayarli_tablo.to_csv(cfg.CIKTI / "cv_ayarli_modeller.csv"); ens_tablo.to_csv(cfg.CIKTI / "cv_ensemble.csv")
print("Kaydedildi:", sorted(p.name for p in cfg.CIKTI.iterdir() if p.is_file()))


# %% MODÜL 13: Kullanıcıdan değer alıp kayıtlı modelle tahmin (Excel yok)
model, sema = tahmin.yukle(cfg.CIKTI)
if cfg.INTERAKTIF:
    tahmin.interaktif_tahmin(model, sema)
else:
    print("Etkileşimli tahmin kapalı (INTERAKTIF=0). Programatik örnek:")
    ornek = {c: b["medyan"] for c, b in sema["sayisal"].items()}
    ornek.update({c: v["secenekler"][0] for c, v in sema["secimli"].items()})
    print(tahmin.tahmin_et(model, sema, ornek))
