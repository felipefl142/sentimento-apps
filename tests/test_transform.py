"""Limpeza de texto, rotulos, silver (versao mais recente, edicoes, sem PII) e qualidade."""
import pandas as pd

from src.transform import quality
from src.transform import text as tx
from tests.conftest import EDITED_ID


def test_clean_idempotente_e_preserva_sinal():
    raw = "  Muuuuuito   BOM!!! 😍 veja https://x.com/a e mail a@b.com "
    c = tx.clean(raw)
    assert c == tx.clean(c)
    assert "😍" in c and "BOM!!!" in c and "<url>" in c and "<email>" in c
    assert "Muuuito" in c and "  " not in c
    assert tx.clean(None) == ""


def test_rotulo_pelas_estrelas():
    assert list(tx.label_from_score([1, 2, 3, 4, 5])) == ["neg", "neg", "neu", "pos", "pos"]


def test_tokens_sem_acento_e_sem_stopwords():
    assert tx.tokens("Não consigo PAGAR o Pix", {"nao", "o"}) == ["consigo", "pagar", "pix"]


def test_silver_versao_mais_recente(silver):
    df = pd.read_parquet(silver / "silver" / "reviews.parquet")
    assert not df.duplicated(["app", "review_id"]).any()
    ed = df[df["review_id"] == EDITED_ID].iloc[0]
    assert ed["score"] == 5 and ed["label"] == "pos" and "ótimo" in ed["text"]
    assert pd.Timestamp(ed["data"]) == pd.Timestamp("2026-05-03")
    assert "user_name" not in df.columns and "content" not in df.columns
    edicoes = pd.read_parquet(silver / "silver" / "reviews_edicoes.parquet")
    row = edicoes[edicoes["review_id"] == EDITED_ID].iloc[0]
    assert (row["score_inicial"], row["score_atual"]) == (1, 5)


def test_qualidade_ok(silver):
    from datetime import date
    out = quality.run(today=date(2026, 7, 1))
    assert out["ok"], [c for c in out["checks"] if not c["ok"]]
