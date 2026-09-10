"""
Persistência da amostra sorteada em Postgres — `amostragem_execucoes` + `amostragem_pontos`.

Antes desta camada, o resultado de um sorteio vivia só em `st.session_state`: some
quando a sessão termina, e não fica registro de quais pontos foram excluídos por
revisão manual (Passo 3 — áreas especiais) nem de quais foram sorteados. As duas
tabelas já existiam desde a migration original (pensadas para isto), mas nada as
usava — esta é a primeira gravação.

O que entra em `parametros` (execução) é só o que não dá para recalcular depois:
`plano` (dimensionamento NBR 5426) e `config` (parâmetros do sorteio) são
`dataclasses` de tipos primitivos, então `dataclasses.asdict()` já serializa para
JSONB sem transformação. `cobertura_classes`, `cobertura_vias`, `abrangencia` e
`vias_principais` ficam de fora de propósito: são funções puras de
`base`/`vias_principais`, então recalculáveis a partir dos pontos — não há razão
para duplicar dado derivado no banco.

`base` (o cadastro preparado inteiro, milhares de linhas) NUNCA é gravada — só
`estrutural`/`qualidade` (a amostra de fato, dezenas a poucas centenas de pontos) e os
pontos excluídos manualmente em áreas especiais, que de outra forma desapareceriam
sem deixar rastro no momento em que `paginas/amostragem_campo.py` faz `base.drop(...)`.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd
from sqlalchemy import text

import db

COLUNAS_PONTO = ["_id", "_logradouro", "_bairro", "_classe", "_lat", "_lon", "_grupo"]


def _dumps(valor: dict) -> str:
    """`json.dumps` tolerante a tipos do numpy/pandas (`int64`, `float64`, ...) que os
    valores de `_lat`/`_lon`/`_id` carregam ao sair de um DataFrame — `.item()` os
    reduz ao tipo Python nativo equivalente; qualquer outro tipo inesperado vira str
    em vez de estourar a gravação de um sorteio que já terminou com sucesso."""
    def _conversor(o):
        return o.item() if hasattr(o, "item") else str(o)
    return json.dumps(valor, default=_conversor)


def _usuario_id(conexao, login: str | None) -> int | None:
    if not login:
        return None
    linha = conexao.execute(
        text("select id from users where login = :login"), {"login": login}
    ).first()
    return linha[0] if linha else None


def salvar_execucao(resultado, *, usuario_login: str | None,
                    pontos_excluidos: list[dict] | None = None) -> int | None:
    """
    Grava a execução + seus pontos. Devolve o `id` da execução, ou `None` em falha —
    nunca levanta: um sorteio bem-sucedido não pode virar tela de erro porque o
    Postgres estava fora do ar (mesmo princípio de `acesso/auditoria.py`). O chamador
    decide se avisa o usuário; aqui só se imprime no stdout.
    """
    try:
        parametros = {
            "plano": asdict(resultado.plano) if resultado.plano else None,
            "config": asdict(resultado.config),
            "ressalvas": list(resultado.ressalvas),
            "total_parque": resultado.total_parque,
            "total_amostra": resultado.total_amostra,
        }
        identificacao = f"{resultado.municipio}/{resultado.uf}".strip("/")

        linhas_pontos = []
        for _, linha in pd.concat([resultado.estrutural, resultado.qualidade]).iterrows():
            linhas_pontos.append({
                "dados": {c: (None if pd.isna(linha.get(c)) else linha.get(c))
                         for c in COLUNAS_PONTO},
                "selecionado": True,
                "revisao_manual": None,
            })
        for ponto in (pontos_excluidos or []):
            linhas_pontos.append({
                "dados": {c: ponto.get(c) for c in ("_id", "_logradouro", "_bairro", "_classe")},
                "selecionado": False,
                "revisao_manual": {
                    "motivo": "area_especial",
                    "categoria": ponto.get("categoria"),
                    "nome": ponto.get("nome"),
                    "osm_url": ponto.get("osm_url"),
                },
            })

        with db.engine().begin() as conexao:
            execucao_id = conexao.execute(
                text(
                    "insert into amostragem_execucoes "
                    "(usuario_id, identificacao, parametros, status) "
                    "values (:usuario_id, :identificacao, :parametros, 'concluida') "
                    "returning id"
                ),
                {
                    "usuario_id": _usuario_id(conexao, usuario_login),
                    "identificacao": identificacao,
                    "parametros": _dumps(parametros),
                },
            ).scalar_one()

            if linhas_pontos:
                # Um só `execute` com a lista inteira de parâmetros: o SQLAlchemy manda
                # como executemany, uma via de ida e volta ao Postgres em vez de uma por
                # ponto — uma amostra de algumas centenas de pontos não deveria custar
                # centenas de round-trips na mesma transação.
                conexao.execute(
                    text(
                        "insert into amostragem_pontos "
                        "(execucao_id, dados, selecionado, revisao_manual) "
                        "values (:execucao_id, :dados, :selecionado, :revisao_manual)"
                    ),
                    [
                        {
                            "execucao_id": execucao_id,
                            "dados": _dumps(linha["dados"]),
                            "selecionado": linha["selecionado"],
                            "revisao_manual": (
                                _dumps(linha["revisao_manual"])
                                if linha["revisao_manual"] is not None else None
                            ),
                        }
                        for linha in linhas_pontos
                    ],
                )
        return execucao_id
    except Exception as erro:                     # noqa: BLE001 — ver docstring do módulo
        print(f"[amostragem] falha ao persistir execução: {erro!r}", flush=True)
        return None


def listar_execucoes(limite: int = 50) -> pd.DataFrame:
    """Execuções mais recentes primeiro, com o nome de quem sorteou (se ainda existir)."""
    with db.engine().connect() as conexao:
        return pd.read_sql(
            text(
                "select e.id, e.criado_em, e.identificacao, e.status, "
                "       e.parametros->>'total_amostra' as total_amostra, "
                "       e.parametros->>'total_parque' as total_parque, "
                "       u.nome as usuario "
                "from amostragem_execucoes e left join users u on u.id = e.usuario_id "
                "order by e.criado_em desc limit :limite"
            ),
            conexao, params={"limite": limite},
        )


def carregar_pontos(execucao_id: int) -> pd.DataFrame:
    """Pontos de uma execução — sorteados e excluídos por revisão manual, juntos."""
    with db.engine().connect() as conexao:
        return pd.read_sql(
            text(
                "select dados, selecionado, revisao_manual from amostragem_pontos "
                "where execucao_id = :execucao_id order by id"
            ),
            conexao, params={"execucao_id": execucao_id},
        )
