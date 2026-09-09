"""
Acesso ao Postgres de produção (Railway).

Ponto único de conexão, usado por `acesso.autenticacao`, `acesso.auditoria` e pela
persistência de execuções de amostragem. Centralizar aqui significa que pool,
normalização da URL e tratamento de erro de configuração só existem em um lugar.

Por que normalizar o esquema da URL
------------------------------------
O Railway (como o Heroku antes dele) expõe a variável `DATABASE_URL` com o esquema
`postgres://`. O SQLAlchemy 2.x só aceita `postgresql://` — a diferença é só o nome
do dialeto, então a troca é uma substituição de prefixo, não uma reformatação da URL.

Por que `pool_pre_ping`
------------------------
Bancos gerenciados derrubam conexões ociosas sem avisar o cliente. Como o Streamlit
reexecuta o script a cada interação — às vezes com minutos de silêncio entre uma
e outra —, uma conexão do pool pode estar morta na próxima vez que for usada.
`pool_pre_ping=True` testa a conexão antes de entregá-la, trocando por uma nova
quando necessário, ao custo de um round-trip extra por checkout.
"""

from __future__ import annotations

import os

import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def url() -> str:
    """
    `DATABASE_URL` normalizada. Exposta (não só de uso interno) porque
    `migrations/env.py` também precisa dela — assim a URL só é lida e corrigida
    num lugar só, tanto para a engine do app quanto para o Alembic.
    """
    valor = os.environ.get("DATABASE_URL")
    if not valor:
        raise RuntimeError(
            "DATABASE_URL não definida. Em produção (Railway), referencie o serviço "
            "Postgres nas variáveis do app; em desenvolvimento local, aponte para um "
            "Postgres próprio antes de rodar `streamlit run app.py`."
        )
    if valor.startswith("postgres://"):
        valor = "postgresql://" + valor[len("postgres://"):]
    return valor


@st.cache_resource
def engine() -> Engine:
    """Engine única do processo, com pool de conexões compartilhado entre os módulos."""
    return create_engine(url(), pool_pre_ping=True)
