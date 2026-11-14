"""Duman testleri: kayıtlı model yükleniyor mu, tahmin üretiyor mu, sürümler uyumlu mu?
Model dosyası repoda yoksa testler atlanır."""
import sys
from pathlib import Path

import pytest

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK))
CIKTI = KOK / "cikti"
pytestmark = pytest.mark.skipif(not (CIKTI / "final_model.joblib").exists(), reason="model dosyası repoda yok")

from ogrenci_ml import tahmin  # noqa: E402


@pytest.fixture(scope="module")
def model_sema():
    return tahmin.yukle(CIKTI)


def test_tipik_ogrenci_icin_gecerli_olasilik(model_sema):
    model, sema = model_sema
    s = tahmin.tahmin_et(model, sema, tahmin.tipik_girdi(sema))
    assert 0.0 <= s["olasilik"] <= 1.0


def test_bilinmeyen_alanlarla_calisir(model_sema):
    model, sema = model_sema
    girdi = {c: None for c in sema["sutun_sirasi"]}
    assert 0.0 <= tahmin.tahmin_et(model, sema, girdi)["olasilik"] <= 1.0


def test_yerel_etki_tum_alanlari_kapsar(model_sema):
    model, sema = model_sema
    etki = tahmin.yerel_etki(model, sema, tahmin.tipik_girdi(sema))
    assert set(etki.index) == set(sema["sutun_sirasi"])


def test_kutuphane_surumleri_uyumlu(model_sema):
    _, sema = model_sema
    assert tahmin.surum_uyumsuzlugu(sema) == []
