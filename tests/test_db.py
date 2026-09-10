import pytest

import db


def test_url_normaliza_esquema_postgres_para_postgresql(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://user:pass@host:5432/nome")
    assert db.url() == "postgresql://user:pass@host:5432/nome"


def test_url_mantem_esquema_ja_postgresql(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/nome")
    assert db.url() == "postgresql://user:pass@host:5432/nome"


def test_url_levanta_erro_claro_quando_variavel_ausente(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        db.url()


def test_engine_e_singleton_por_processo():
    assert db.engine() is db.engine()
