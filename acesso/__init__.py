"""
Controle de acesso e trilha de uso da Plataforma IP.

Dois serviços, deliberadamente acoplados: só faz sentido registrar quem usou o quê
depois de saber *quem* está usando. A ordem correta no `app.py` é sempre

    exigir_login()          # barra o anônimo; devolve o usuário autenticado
    registrar_pagina(...)   # a partir daqui todo evento tem dono

`exigir_login()` interrompe a execução do script quando não há sessão válida — ele
chama `st.stop()` internamente, então nada abaixo dele roda para quem não entrou.

Por que autenticação própria e não OIDC: a base de usuários é uma equipe pequena e
conhecida, e o custo de manter um provedor OAuth (projeto no Google Cloud, credencial,
URI de redirect, conta Google obrigatória para cada pessoa) não se paga nessa escala.
A superfície é pequena e está isolada aqui — migrar para `st.login` no dia em que o
portal for exposto a cliente externo mexe só neste pacote.

O que este pacote NÃO faz, e é importante ser explícito: ele protege o *portal*, não
os *dados*. Modelos `.pkl`, datasets e planilhas continuam legíveis por quem tiver
acesso ao repositório, independentemente de qualquer senha aqui.
"""

from .autenticacao import (
    Usuario,
    exigir_login,
    usuario_atual,
    encerrar_sessao,
    gerar_hash,
    verificar_senha,
)
from .auditoria import (
    registrar_evento,
    registrar_pagina,
    registrar_acao,
    backend_ativo,
)

__all__ = [
    "Usuario",
    "exigir_login",
    "usuario_atual",
    "encerrar_sessao",
    "gerar_hash",
    "verificar_senha",
    "registrar_evento",
    "registrar_pagina",
    "registrar_acao",
    "backend_ativo",
]
