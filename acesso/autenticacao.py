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

**Os usuários vivem na tabela `users` do Postgres, nunca no repositório.** Até a
migração para hospedagem própria (Railway), viviam em `st.secrets` — mas um secrets
só existe por instância/redeploy, e o Community Cloud não oferecia banco nenhum.
Com um Postgres real disponível, a tabela substitui o bloco `[auth.usuarios]` inteiro;
cadastro/revogação de acesso passam a ser um INSERT/UPDATE, não uma edição de TOML
seguida de redeploy. `senha_hash` guarda exatamente o mesmo formato de antes.

**Mensagem de erro genérica.** "Usuário ou senha inválidos" não revela se o usuário
existe. E a verificação roda mesmo para usuário inexistente, contra um hash falso, para
que o tempo de resposta não denuncie a existência da conta (ataque por temporização).

Configuração restante em `st.secrets`
--------------------------------------
Só os parâmetros de sessão continuam em secrets — não são segredo, mas não há hoje
outro lugar de configuração no portal:

    [auth]
    expiracao_horas = 12          # opcional (padrão 12); sessão inativa expira
    max_tentativas  = 5           # opcional (padrão 5); bloqueio temporário
    bloqueio_minutos = 15         # opcional (padrão 15)

Cadastre um usuário com:  py -m acesso.gerar_hash --login jferreira --nome "Juliano Ferreira"
Grava direto na tabela `users` (requer `DATABASE_URL` no ambiente); rodar de novo para
um login existente atualiza nome/perfil/senha, então também serve para troca de senha.
Cadastro pela interface (perfil admin) vive em `paginas/administracao.py`.
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
from sqlalchemy import text

import db

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
    """Usuários ativos, por login. Consulta o Postgres a cada chamada: só é exercida
    na tela de login (não autenticado), então o volume é baixo e não justifica cache —
    cache aqui só atrasaria a revogação de um acesso."""
    try:
        with db.engine().connect() as conexao:
            linhas = conexao.execute(text(
                "select login, nome, senha_hash, perfil from users where ativo"
            )).mappings().all()
        return {linha["login"]: dict(linha) for linha in linhas}
    except Exception:
        # Falha de conexão não pode virar stack trace para o usuário final — vira
        # "controle de acesso não configurado", que é o mesmo caminho de falha fechada
        # que já existia para secrets ausente.
        return {}


def esta_configurado() -> bool:
    return bool(_usuarios())


# ── Administração (tela e CLI) ──────────────────────────────────────────────
def usuarios_cadastrados() -> dict:
    """Todos os usuários por login, ativos e inativos — para a tela de administração.
    Diferente de `_usuarios()` (só ativos), que é a que decide quem consegue logar."""
    try:
        with db.engine().connect() as conexao:
            linhas = conexao.execute(text(
                "select login, nome, perfil, ativo from users order by login"
            )).mappings().all()
        return {linha["login"]: dict(linha) for linha in linhas}
    except Exception:
        return {}


def criar_ou_atualizar_usuario(login: str, nome: str, perfil: str, senha: str) -> None:
    """
    Grava um login em `users`: cadastra se for novo, ou atualiza nome/perfil/senha (e
    reativa) se já existir. É a mesma operação por trás de `acesso/gerar_hash.py` e do
    formulário de `paginas/administracao.py` — uma serve quem tem acesso ao servidor,
    a outra quem só tem o perfil admin no portal; nenhuma duplica a lógica da outra.
    """
    registro = gerar_hash(senha)
    with db.engine().begin() as conexao:
        conexao.execute(
            text(
                "insert into users (login, nome, senha_hash, perfil, ativo) "
                "values (:login, :nome, :senha_hash, :perfil, true) "
                "on conflict (login) do update set "
                "nome = excluded.nome, senha_hash = excluded.senha_hash, "
                "perfil = excluded.perfil, ativo = true"
            ),
            {"login": login, "nome": nome, "senha_hash": registro, "perfil": perfil},
        )


def definir_ativo(login: str, ativo: bool) -> None:
    """Ativa/desativa um login sem apagar a linha. Não usa FK para `audit_events`
    (ver migration), então revogar um acesso nunca afeta o histórico já registrado."""
    with db.engine().begin() as conexao:
        conexao.execute(
            text("update users set ativo = :ativo where login = :login"),
            {"ativo": ativo, "login": login},
        )


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


_CHAVE_ULTIMA_VERIFICACAO = "_acesso_ultima_verificacao"
INTERVALO_VERIFICACAO_S = 60

# Revisão de 10/09/2026, achado em revisão de segurança: revogar ou rebaixar alguém pela
# tela de administração (`definir_ativo`/`criar_ou_atualizar_usuario`) só grava no
# Postgres — não existe mais o reinício de processo que o Streamlit Cloud fazia ao
# salvar o secrets (que derrubava toda sessão aberta de graça). Sem isto, uma conta
# desativada ou rebaixada de admin continuaria válida em qualquer aba já logada até a
# sessão expirar sozinha (`expiracao_horas`, padrão 12h).
def _ainda_valido(u: Usuario) -> bool:
    """
    Confere no banco se a conta segue ativa e com o mesmo perfil — no máximo uma vez a
    cada `INTERVALO_VERIFICACAO_S`, não a cada rerun (cada clique do Streamlit reexecuta
    o script inteiro; consultar o banco em todos eles custaria uma ida ao Postgres por
    interação). Falha de conexão não derruba quem já estava logado — mesma filosofia de
    "nunca derrubar o app" da auditoria: perder a reverificação por um instante é melhor
    que expulsar todo mundo porque o banco piscou.
    """
    agora = time.time()
    if agora - st.session_state.get(_CHAVE_ULTIMA_VERIFICACAO, 0.0) < INTERVALO_VERIFICACAO_S:
        return True
    st.session_state[_CHAVE_ULTIMA_VERIFICACAO] = agora
    try:
        with db.engine().connect() as conexao:
            linha = conexao.execute(
                text("select perfil, ativo from users where login = :login"),
                {"login": u.login},
            ).mappings().first()
    except Exception:
        return True
    return linha is not None and bool(linha["ativo"]) and linha["perfil"] == u.perfil


def usuario_atual() -> Usuario | None:
    """Usuário da sessão, ou None se não autenticado, expirado, ou revogado/alterado."""
    u = st.session_state.get(_CHAVE_SESSAO)
    if u is None:
        return None
    if _expirada(u) or not _ainda_valido(u):
        st.session_state.pop(_CHAVE_SESSAO, None)
        st.session_state.pop(_CHAVE_ULTIMA_VERIFICACAO, None)
        return None
    return u


def encerrar_sessao() -> Usuario | None:
    """Derruba a sessão e devolve quem estava logado, para o log registrar a saída."""
    st.session_state.pop(_CHAVE_ULTIMA_VERIFICACAO, None)
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
        "Nenhum usuário ativo foi encontrado na tabela `users`, e por segurança o "
        "portal não abre sem controle de acesso. Isso acontece tanto quando não há "
        "nenhum usuário cadastrado quanto quando o app não conseguiu consultar o "
        "Postgres (`DATABASE_URL` ausente ou inválida nesta instância).\n\n"
        "**Para configurar:** cadastre um usuário com `py -m acesso.gerar_hash "
        "--login <login> --nome \"<nome>\" --perfil admin` (requer `DATABASE_URL` no "
        "ambiente de onde o comando roda) ou confira se `DATABASE_URL` está definida "
        "nas variáveis deste serviço no Railway."
    )


def _tela_login() -> None:
    st.markdown(
        """
        <style>
          /* A navegação lateral não existe antes do login; escondê-la evita o flash de
             uma sidebar vazia enquanto o formulário monta. */
          section[data-testid="stSidebar"] { display: none; }
          [data-testid="stForm"] { border: 1px solid #E2E8F0; border-radius: 14px;
              padding: 1.4rem 1.4rem 0.6rem 1.4rem; background: #FFFFFF;
              max-width: 420px; margin: 0 auto; }
          .block-container { max-width: 560px; }

          /* O contorno dos campos e a cor do botão primário eram remendados aqui,
             porque o tema global não tinha `primaryColor` nem borda de widget. Desde
             que os dois passaram para `.streamlit/config.toml` (e o estilo de campo
             para `app.py`), esta tela herda o mesmo tratamento das outras — só
             sobrou o que é específico dela: esconder a sidebar e enquadrar o cartão. */
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="text-align:center; margin: 2.2rem 0 1.4rem 0;">
          <div style="font-size:2.4rem; font-weight:800;
                      background: linear-gradient(90deg,#1B3664,#00A9E0);
                      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                      letter-spacing:-1px;">Plataforma IP</div>
          <div style="font-size:.78rem; color:#5B6579; letter-spacing:.15em;
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
    st.session_state[_CHAVE_ULTIMA_VERIFICACAO] = time.time()
    st.session_state.pop(_CHAVE_TENTATIVAS, None)
    auditoria.registrar_evento("login", usuario=u)
    st.rerun()
