"""
Trilha de uso do portal: quem entrou, o que abriu, o que gerou.

Dois backends, escolhidos automaticamente
-----------------------------------------
1. **Google Sheets**, quando `st.secrets` traz a service account e o id da planilha.
   É o backend de produção. Razão: o disco do Streamlit Community Cloud é efêmero — o
   container é recriado a cada redeploy e o app hiberna por inatividade —, então
   qualquer arquivo escrito lá dura dias, não meses. A planilha vive fora do container.
2. **CSV local**, quando o Sheets não está configurado. É o modo de desenvolvimento e
   o modo servidor próprio, onde o disco persiste de fato.

Princípio inegociável: **auditoria nunca derruba o app**. Toda falha de escrita é
engolida e reportada só no stdout (que no Cloud vira log de aplicação). Um portal que
cai porque a planilha de log ficou indisponível é pior do que um portal sem log.

Escrita assíncrona, e por que ela é obrigatória
-----------------------------------------------
O Streamlit reexecuta o script inteiro a cada interação. Escrever no Sheets de forma
síncrona somaria ~400 ms a cada clique e estouraria a quota da API (60 req/min por
usuário) numa navegação normal. Por isso os eventos entram numa fila em memória e uma
thread daemon os grava em lote. Duas consequências aceitas conscientemente:
  - eventos podem se perder se o container morrer com a fila cheia (é log de uso, não
    livro-razão);
  - a página só é registrada quando MUDA, não a cada rerun (ver `registrar_pagina`).

Duração de sessão
-----------------
Não há evento confiável de "saída": o usuário fecha a aba e o servidor não fica
sabendo. Em vez de inventar um heartbeat que polui o log, cada evento carrega
`sessao_id` e `segundos_sessao`. A duração de uma sessão é o maior `segundos_sessao`
entre os eventos daquele `sessao_id` — derivada na análise, não medida na hora.
"""

from __future__ import annotations

import atexit
import csv
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

COLUNAS = [
    "timestamp_utc", "evento", "usuario", "nome", "perfil",
    "sessao_id", "segundos_sessao", "alvo", "detalhe",
]

# `dados/` é irmão de `app/` e fica fora do repositório git — o CSV de uso não entra
# em commit por construção. `PLATAFORMA_IP_LOG` sobrescreve para outro disco.
PASTA_LOG = Path(os.environ.get(
    "PLATAFORMA_IP_LOG",
    Path(__file__).resolve().parents[2] / "dados" / "uso",
))
ARQUIVO_CSV = PASTA_LOG / "uso.csv"

INTERVALO_FLUSH_S = 4.0
LOTE_MAXIMO = 25
FILA_MAXIMA = 2000

_CHAVE_ULTIMA_PAGINA = "_acesso_ultima_pagina"


# ── Configuração do backend ──────────────────────────────────────────────────
def _conf_sheets() -> dict | None:
    """Bloco `[log_uso]` do secrets, se a planilha estiver configurada."""
    try:
        cfg = dict(st.secrets["log_uso"])
    except Exception:
        return None
    if not cfg.get("planilha_id"):
        return None
    return cfg


@st.cache_resource(show_spinner=False)
def _planilha():
    """
    Aba do Google Sheets pronta para append, ou None.

    Em cache de recurso porque a autenticação OAuth da service account custa uma
    chamada de rede: refazê-la a cada rerun do Streamlit seria absurdo. Cria o
    cabeçalho na primeira escrita, para que a planilha possa ser criada vazia.
    """
    cfg = _conf_sheets()
    if cfg is None:
        return None
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        cred = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]),
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        planilha = gspread.authorize(cred).open_by_key(str(cfg["planilha_id"]))
        nome_aba = str(cfg.get("aba", "uso"))
        try:
            aba = planilha.worksheet(nome_aba)
        except Exception:
            aba = planilha.add_worksheet(nome_aba, rows=1000, cols=len(COLUNAS))
        if not aba.acell("A1").value:
            aba.append_row(COLUNAS, value_input_option="RAW")
        return aba
    except Exception as erro:                     # noqa: BLE001 — ver docstring do módulo
        print(f"[acesso] Sheets indisponível, usando CSV local: {erro!r}", flush=True)
        return None


def backend_ativo() -> str:
    """Nome do backend em uso — exibido no rodapé da tela de administração."""
    return "Google Sheets" if _planilha() is not None else f"CSV local ({ARQUIVO_CSV})"


# ── Escritores ───────────────────────────────────────────────────────────────
def _escrever_csv(linhas: list[list]) -> None:
    PASTA_LOG.mkdir(parents=True, exist_ok=True)
    novo = not ARQUIVO_CSV.exists()
    with ARQUIVO_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(COLUNAS)
        w.writerows(linhas)


def _escrever(linhas: list[list]) -> None:
    """Grava o lote no backend disponível; o CSV é a rede de segurança do Sheets."""
    aba = _planilha()
    if aba is not None:
        try:
            aba.append_rows(linhas, value_input_option="RAW")
            return
        except Exception as erro:                 # noqa: BLE001
            print(f"[acesso] falha ao gravar no Sheets: {erro!r}", flush=True)
    try:
        _escrever_csv(linhas)
    except Exception as erro:                     # noqa: BLE001
        print(f"[acesso] falha ao gravar CSV: {erro!r}", flush=True)


# ── Fila e worker ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _fila() -> queue.Queue:
    """
    Fila única do processo, com a thread consumidora já em pé.

    `cache_resource` é o singleton do Streamlit: sobrevive aos reruns e é compartilhado
    entre as sessões do mesmo container, que é exatamente o escopo desejado — uma fila
    por processo, não por usuário.
    """
    f: queue.Queue = queue.Queue(maxsize=FILA_MAXIMA)

    def _worker() -> None:
        lote: list[list] = []
        ultimo = time.monotonic()
        while True:
            try:
                lote.append(f.get(timeout=INTERVALO_FLUSH_S))
            except queue.Empty:
                pass
            venceu = time.monotonic() - ultimo >= INTERVALO_FLUSH_S
            if lote and (len(lote) >= LOTE_MAXIMO or venceu):
                _escrever(lote)
                lote = []
                ultimo = time.monotonic()

    t = threading.Thread(target=_worker, name="acesso-auditoria", daemon=True)
    t.start()

    def _drenar() -> None:
        """Última chance de gravar o que ficou na fila num encerramento limpo."""
        restante = []
        while True:
            try:
                restante.append(f.get_nowait())
            except queue.Empty:
                break
        if restante:
            _escrever(restante)

    atexit.register(_drenar)
    return f


# ── API pública ──────────────────────────────────────────────────────────────
def registrar_evento(evento: str, *, usuario=None, alvo: str = "",
                     detalhe: str = "", login: str = "") -> None:
    """
    Enfileira um evento da trilha.

    `usuario` é o dataclass `Usuario`; `login` cobre o caso do login malsucedido, em que
    ainda não existe usuário autenticado. Nunca levanta: um erro aqui não pode
    interromper a ação que o usuário estava fazendo.
    """
    try:
        if usuario is None:
            from .autenticacao import usuario_atual
            usuario = usuario_atual()

        linha = [
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            evento,
            usuario.login if usuario else (login or "anonimo"),
            usuario.nome if usuario else "",
            usuario.perfil if usuario else "",
            usuario.sessao_id if usuario else "",
            round(usuario.segundos_de_sessao) if usuario else "",
            alvo,
            detalhe,
        ]
        try:
            _fila().put_nowait(linha)
        except queue.Full:
            # Fila cheia significa backend fora do ar há tempo. Descartar o evento novo
            # é melhor do que bloquear a thread da interface esperando espaço.
            print("[acesso] fila de auditoria cheia; evento descartado", flush=True)
    except Exception as erro:                     # noqa: BLE001
        print(f"[acesso] falha ao registrar evento: {erro!r}", flush=True)


def ler_eventos(limite: int = 5000):
    """
    Devolve a trilha registrada como DataFrame, mais recente primeiro.

    Lê do backend ativo. Devolve DataFrame vazio (com as colunas certas) quando ainda
    não há nada — a tela de administração precisa renderizar de qualquer jeito.

    O pandas é importado aqui dentro, e não no topo, porque este módulo é carregado no
    caminho crítico do login: quem só entra no portal não deve pagar o import.
    """
    import pandas as pd

    vazio = pd.DataFrame(columns=COLUNAS)
    try:
        aba = _planilha()
        if aba is not None:
            registros = aba.get_all_records()
            df = pd.DataFrame(registros) if registros else vazio
        elif ARQUIVO_CSV.exists():
            df = pd.read_csv(ARQUIVO_CSV)
        else:
            return vazio
    except Exception as erro:                     # noqa: BLE001
        print(f"[acesso] falha ao ler a trilha: {erro!r}", flush=True)
        return vazio

    if df.empty:
        return vazio
    return df.iloc[::-1].head(limite).reset_index(drop=True)


def registrar_pagina(pagina: str) -> None:
    """
    Registra a abertura de uma página, **só quando ela muda**.

    Sem essa guarda o Streamlit geraria um evento por rerun — cada clique em cada
    widget —, o que enche a planilha de ruído e estoura a quota da API do Sheets em
    minutos. O que interessa é a navegação, não o número de reruns.
    """
    if st.session_state.get(_CHAVE_ULTIMA_PAGINA) == pagina:
        return
    st.session_state[_CHAVE_ULTIMA_PAGINA] = pagina
    registrar_evento("pagina", alvo=pagina)


def registrar_acao(acao: str, alvo: str = "", detalhe: str = "") -> None:
    """
    Registra uma ação de peso: upload de cadastro, geração de planilha, download.

    Convenção de conteúdo: `alvo` recebe o objeto da ação (município, nome do arquivo),
    `detalhe` recebe o parâmetro relevante (nº de pontos, anos consultados). Não
    registre conteúdo de planilha de cliente aqui — a trilha responde "quem, quando, o
    quê", não guarda o dado em si.
    """
    registrar_evento(acao, alvo=alvo, detalhe=detalhe)
