"""
Autenticação por usuário nomeado, com senha guardada como hash.

Decisões e o porquê de cada uma
-------------------------------
**Usuário nomeado, não senha compartilhada.** Senha única e trilha de uso são
incompatíveis: o log registraria "alguém entrou", que é exatamente a informação
inútil que já existe hoje. Usuário nomeado também permite revogar um acesso isolado —
com senha única, cada desligamento obrigaria a redistribuir a senha a todo mundo, o
que na prática ninguém faz, e a senha vaza por acúmulo de ex-usuários.

**PBKDF2-HMAC-SHA256 da biblioteca padrão, não bcrypt/argon2.** Não por serem
melhores — argon2 é superior —, mas porque `hashlib` não adiciona dependência ao
`requirements.txt`, e o `requirements.txt` deste projeto tem uma restrição dura
(`scikit-learn==1.8.0`, casado com os `.pkl`). Menos roda no resolvedor de
dependências do deploy, menos chance de quebrar o carregamento dos modelos.
600.000 iterações é a recomendação corrente do OWASP para PBKDF2-SHA256; custa
~0,3 s por login nesta máquina, o que é irrelevante para o usuário e caro para quem
tenta força bruta.

**Os hashes vivem em `st.secrets`, nunca no repositório.** No Streamlit Cloud isso é
o painel de secrets; localmente é `.streamlit/secrets.toml`, que está no `.gitignore`.
Isso importa mais aqui do que no caso geral, porque o repositório é público.

**Mensagem de erro genérica.** "Usuário ou senha inválidos" não revela se o usuário
existe. E a verificação roda mesmo para usuário inexistente, contra um hash falso, para
que o tempo de resposta não denuncie a existência da conta (ataque por temporização).

Formato esperado em `st.secrets`
--------------------------------
    [auth]
    expiracao_horas = 12          # opcional (padrão 12); sessão inativa expira
    max_tentativas  = 5           # opcional (padrão 5); bloqueio temporário
    bloqueio_minutos = 15         # opcional (padrão 15)

    [auth.usuarios.jferreira]
    nome  = "Juliano Ferreira"
    senha_hash = "pbkdf2_sha256$600000$<salt>$<hash>"
    perfil = "admin"              # opcional; livre, hoje só informativo

Gere o hash com:  py -m acesso.gerar_hash
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets as _secrets
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import streamlit as st

ALGORITMO = "pbkdf2_sha256"
ITERACOES = 600_000
TAMANHO_SALT = 16

EXPIRACAO_HORAS_PADRAO = 12
MAX_TENTATIVAS_PADRAO = 5
BLOQUEIO_MINUTOS_PADRAO = 15

# Hash descartável usado só para gastar tempo quando o usuário não existe, de modo que
# login inexistente e senha errada custem o mesmo. Não corresponde a nenhuma senha.
_HASH_FALSO = None

_CHAVE_SESSAO = "_acesso_sessao"
_CHAVE_TENTATIVAS = "_acesso_tentativas"


@dataclass(frozen=True)
class Usuario:
    """Identidade autenticada. `sessao_id` correlaciona os eventos da trilha de uso."""

    login: str
    nome: str
    perfil: str
    sessao_id: str
    autenticado_em: datetime

    @property
    def segundos_de_sessao(self) -> float:
        return (datetime.now(timezone.utc) - self.autenticado_em).total_seconds()


# ── Hash de senha ────────────────────────────────────────────────────────────
def gerar_hash(senha: str, *, iteracoes: int = ITERACOES) -> str:
    """
    Deriva o hash de uma senha, no formato `algoritmo$iteracoes$salt$hash`.

    O salt é aleatório por senha: duas pessoas com a mesma senha produzem hashes
    diferentes, o que inutiliza tabelas pré-computadas. As iterações vão embutidas no
    próprio registro para que aumentá-las no futuro não invalide os hashes já emitidos.
    """
    if not senha:
        raise ValueError("senha vazia")
    salt = _secrets.token_bytes(TAMANHO_SALT)
    derivado = hashlib.pbkdf2_hmac("sha256", senha.encode("utf-8"), salt, iteracoes)
    return "$".join([
        ALGORITMO,
        str(iteracoes),
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(derivado).decode("ascii"),
    ])


def verificar_senha(senha: str, registro: str) -> bool:
    """
    Confere a senha contra o registro `algoritmo$iteracoes$salt$hash`.

    Compara com `hmac.compare_digest` (tempo constante): comparação byte a byte comum
    aborta no primeiro byte diferente, e a diferença de tempo permite descobrir o hash
    caractere por caractere. Registro malformado devolve False em vez de explodir —
    um erro de digitação no secrets não pode virar tela de erro para o usuário final.
    """
    try:
        algoritmo, iteracoes, salt_b64, hash_b64 = registro.split("$")
        if algoritmo != ALGORITMO:
            return False
        derivado = hashlib.pbkdf2_hmac(
            "sha256", senha.encode("utf-8"),
            base64.b64decode(salt_b64), int(iteracoes),
        )
        return hmac.compare_digest(derivado, base64.b64decode(hash_b64))
    except (ValueError, TypeError, base64.binascii.Error):
        return False


def _hash_falso() -> str:
    global _HASH_FALSO
    if _HASH_FALSO is None:
        _HASH_FALSO = gerar_hash(_secrets.token_urlsafe(32))
    return _HASH_FALSO


# ── Configuração ─────────────────────────────────────────────────────────────
def _cfg() -> dict:
    """Bloco `[auth]` do secrets, ou vazio quando não configurado."""
    try:
        return dict(st.secrets.get("auth", {}))
    except Exception:
        # st.secrets levanta quando não há nenhum secrets.toml — ausência não é erro.
        return {}


def _usuarios() -> dict:
    try:
        return dict(st.secrets["auth"]["usuarios"])
    except Exception:
        return {}


def esta_configurado() -> bool:
    return bool(_usuarios())


# ── Bloqueio por tentativas ──────────────────────────────────────────────────
# O contador vive no session_state, portanto é por aba do navegador. Isso NÃO impede
# força bruta distribuída — para isso seria preciso estado compartilhado entre sessões,
# que o Community Cloud não oferece de graça. O que ele faz é encarecer o ataque manual
# e, principalmente, gerar registro na trilha: cinco falhas seguidas viram cinco linhas
# no log, e é o log que denuncia a tentativa.
def _tentativas() -> dict:
    return st.session_state.setdefault(_CHAVE_TENTATIVAS, {"n": 0, "bloqueado_ate": 0.0})


def _segundos_de_bloqueio() -> float:
    return max(0.0, _tentativas()["bloqueado_ate"] - time.time())


def _registrar_falha() -> None:
    t = _tentativas()
    t["n"] += 1
    limite = int(_cfg().get("max_tentativas", MAX_TENTATIVAS_PADRAO))
    if t["n"] >= limite:
        minutos = float(_cfg().get("bloqueio_minutos", BLOQUEIO_MINUTOS_PADRAO))
        t["bloqueado_ate"] = time.time() + minutos * 60
        t["n"] = 0


# ── Sessão ───────────────────────────────────────────────────────────────────
def _expirada(u: Usuario) -> bool:
    horas = float(_cfg().get("expiracao_horas", EXPIRACAO_HORAS_PADRAO))
    if horas <= 0:
        return False
    return datetime.now(timezone.utc) - u.autenticado_em > timedelta(hours=horas)


def usuario_atual() -> Usuario | None:
    """Usuário da sessão, ou None se não autenticado ou expirado."""
    u = st.session_state.get(_CHAVE_SESSAO)
    if u is None:
        return None
    if _expirada(u):
        st.session_state.pop(_CHAVE_SESSAO, None)
        return None
    return u


def encerrar_sessao() -> Usuario | None:
    """Derruba a sessão e devolve quem estava logado, para o log registrar a saída."""
    return st.session_state.pop(_CHAVE_SESSAO, None)


def _autenticar(login: str, senha: str) -> Usuario | None:
    dados = _usuarios().get(login.strip().lower())
    if dados is None:
        verificar_senha(senha, _hash_falso())   # gasta o mesmo tempo; ver docstring
        return None
    if not verificar_senha(senha, str(dados.get("senha_hash", ""))):
        return None
    return Usuario(
        login=login.strip().lower(),
        nome=str(dados.get("nome") or login),
        perfil=str(dados.get("perfil") or "usuario"),
        sessao_id=uuid.uuid4().hex[:12],
        autenticado_em=datetime.now(timezone.utc),
    )


# ── Porta de entrada ─────────────────────────────────────────────────────────
def exigir_login() -> Usuario:
    """
    Garante que há um usuário autenticado; caso contrário desenha o formulário e
    interrompe o script com `st.stop()`.

    Chamar isto ANTES de qualquer `st.Page`/`st.navigation` no `app.py` é o que garante
    que nenhuma página renderiza para anônimo — o Streamlit executa o script de cima
    para baixo, e `st.stop()` impede tudo o que viria depois.
    """
    u = usuario_atual()
    if u is not None:
        return u

    if not esta_configurado():
        # Sem usuários cadastrados o portal ficaria aberto. Falhar fechado é o único
        # comportamento defensável: um secrets ausente no deploy não pode virar
        # "portal público" silenciosamente.
        _tela_nao_configurado()
        st.stop()

    _tela_login()
    st.stop()


# As telas abaixo NÃO chamam `st.set_page_config`: ele só pode ser executado uma vez por
# script e o `app.py` já o fez antes de chamar `exigir_login()`. Chamar de novo aqui
# levantaria StreamlitAPIException e a tela de login viraria tela de erro.
def _tela_nao_configurado() -> None:
    st.error("Controle de acesso não configurado.")
    st.markdown(
        "Nenhum usuário foi cadastrado em `st.secrets`, e por segurança o portal não "
        "abre sem controle de acesso.\n\n"
        "**Para configurar:** gere um hash com `py -m acesso.gerar_hash` e cole o bloco "
        "resultante em `.streamlit/secrets.toml` (local) ou no painel de secrets do "
        "Streamlit Cloud (publicado). Ver `app/acesso/README.md`."
    )


def _tela_login() -> None:
    st.markdown(
        """
        <style>
          /* A navegação lateral não existe antes do login; escondê-la evita o flash de
             uma sidebar vazia enquanto o formulário monta. */
          section[data-testid="stSidebar"] { display: none; }
          [data-testid="stForm"] { border: 1px solid #1f2937; border-radius: 14px;
              padding: 1.4rem 1.4rem 0.6rem 1.4rem; background: #12192b;
              max-width: 420px; margin: 0 auto; }
          .block-container { max-width: 560px; }

          /* Campos com contorno: sobre o fundo do cartão eles somem sem borda. */
          [data-testid="stForm"] input {
              background: #0b111e !important; border: 1px solid #24304a !important; }

          /* O botão primário sai vermelho porque `.streamlit/config.toml` omite
             `primaryColor` de propósito — definir o tema global mudaria a cor dos
             controles das quatro ferramentas. Por isso a cor é aplicada aqui, com
             escopo restrito ao formulário de login. */
          [data-testid="stForm"] [data-testid*="FormSubmit"] button,
          [data-testid="stForm"] button[data-testid*="FormSubmit"] {
              background: var(--marca-teal, #00A9E0) !important;
              border-color: var(--marca-teal, #00A9E0) !important;
              color: #06121d !important; font-weight: 700 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="text-align:center; margin: 2.2rem 0 1.4rem 0;">
          <div style="font-size:2.4rem; font-weight:800;
                      background: linear-gradient(90deg,#ffffff,#00A9E0);
                      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                      letter-spacing:-1px;">Plataforma IP</div>
          <div style="font-size:.78rem; color:#94a3b8; letter-spacing:.15em;
                      text-transform:uppercase;">Engenharia de Iluminação Pública</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    restante = _segundos_de_bloqueio()
    if restante > 0:
        st.error(f"Muitas tentativas. Novo acesso liberado em {int(restante // 60) + 1} min.")
        return

    with st.form("login", clear_on_submit=False):
        login = st.text_input("Usuário", autocomplete="username")
        senha = st.text_input("Senha", type="password", autocomplete="current-password")
        entrar = st.form_submit_button("Entrar", use_container_width=True, type="primary")

    if not entrar:
        return

    from . import auditoria   # import tardio: auditoria não pode ser exigência de import

    u = _autenticar(login, senha)
    if u is None:
        _registrar_falha()
        auditoria.registrar_evento(
            "login_falha", login=(login or "").strip().lower() or "(vazio)",
            detalhe="usuário ou senha inválidos",
        )
        st.error("Usuário ou senha inválidos.")
        return

    st.session_state[_CHAVE_SESSAO] = u
    st.session_state.pop(_CHAVE_TENTATIVAS, None)
    auditoria.registrar_evento("login", usuario=u)
    st.rerun()
