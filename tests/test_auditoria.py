import uuid

from sqlalchemy import text

import db
from acesso import auditoria


def test_escrever_pg_mantem_sessao_id_vazio_mas_nulifica_segundos_sessao():
    """
    Regressão: `_escrever_pg` convertia TODO campo `""` para `None`, não só o numérico
    — isso fazia um evento anônimo (`sessao_id=""`) virar NULL no Postgres, e
    `paginas/administracao.py` (`eventos["sessao_id"].astype(str).str.len() > 0`) para
    de reconhecê-lo como sem sessão, já que `str(None)` também tem comprimento > 0.
    """
    assert auditoria.COLUNAS == [
        "timestamp_utc", "evento", "usuario", "nome", "perfil",
        "sessao_id", "segundos_sessao", "alvo", "detalhe",
    ]
    usuario_marcador = f"pytest_{uuid.uuid4().hex[:8]}"
    linha = [
        "2026-01-01T00:00:00+00:00", "login_falha", usuario_marcador, "", "",
        "", "", "", "teste de regressão",
    ]

    try:
        assert auditoria._escrever_pg([linha]) is True

        with db.engine().connect() as conexao:
            registro = conexao.execute(
                text("select sessao_id, segundos_sessao from audit_events "
                     "where usuario = :usuario"),
                {"usuario": usuario_marcador},
            ).mappings().first()

        assert registro is not None
        assert registro["sessao_id"] == ""
        assert registro["segundos_sessao"] is None
    finally:
        with db.engine().begin() as conexao:
            conexao.execute(
                text("delete from audit_events where usuario = :usuario"),
                {"usuario": usuario_marcador},
            )
