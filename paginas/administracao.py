"""
Administração de acessos — visível apenas para quem tem `perfil = "admin"`.

Duas funções: emitir credencial para uma pessoa nova e ler a trilha de uso.

Por que não existe "criar conta" na tela de login
--------------------------------------------------
Autocadastro anularia o controle de acesso: qualquer visitante criaria a própria conta
e entraria. Num portal cujo propósito é justamente conter modelos, bases e metodologia,
isso seria o mesmo que não ter senha. O modelo aqui é **provisionamento por
administrador** — alguém já autorizado emite a credencial de quem entra.

Esta página não escreve no `secrets` sozinha, e isso é limitação da plataforma, não
descuido: no Streamlit Cloud o `secrets` é gerenciado pelo painel e é somente leitura
em runtime; localmente, gravar um arquivo que o processo relê a quente daria margem a
uma condição de corrida entre o formulário e o próprio login. Então a página faz a
parte difícil (derivar o hash com o mesmo algoritmo do login, montar o TOML) e deixa
para o administrador o passo de colar — que leva dez segundos e é auditável.
"""

from __future__ import annotations

import secrets as _secrets

import pandas as pd
import streamlit as st

from acesso import autenticacao, auditoria, usuario_atual

st.markdown("## 🔐 Administração de acessos")

_eu = usuario_atual()
if _eu is None or _eu.perfil != "admin":
    # Defesa em profundidade: a página já não é registrada no menu para não-admin, mas
    # a navegação do Streamlit é por URL e `/administracao` é adivinhável.
    st.error("Esta área é restrita a administradores.")
    st.stop()

aba_usuarios, aba_trilha = st.tabs(["Usuários", "Trilha de uso"])

# ── Usuários ─────────────────────────────────────────────────────────────────
with aba_usuarios:
    cadastrados = autenticacao._usuarios()
    st.markdown("#### Quem tem acesso hoje")
    if cadastrados:
        st.dataframe(
            pd.DataFrame([
                {"Login": login,
                 "Nome": dados.get("nome", ""),
                 "Perfil": dados.get("perfil", "usuario")}
                for login, dados in sorted(cadastrados.items())
            ]),
            hide_index=True, use_container_width=True,
        )
    else:
        st.info("Nenhum usuário cadastrado.")

    st.caption(
        "Para **revogar** um acesso, apague o bloco `[auth.usuarios.<login>]` do "
        "secrets. Ninguém mais é afetado — é isso que uma senha compartilhada não "
        "permitiria fazer sem incomodar a equipe inteira."
    )

    st.divider()
    st.markdown("#### Emitir uma credencial nova")

    with st.form("nova_credencial"):
        c1, c2 = st.columns(2)
        novo_login = c1.text_input("Login", placeholder="sobrenome ou inicial+sobrenome",
                                   help="Minúsculas, sem espaços. É o identificador que "
                                        "aparece na trilha de uso.")
        novo_nome = c2.text_input("Nome exibido", placeholder="Nome Sobrenome")
        c3, c4 = st.columns(2)
        novo_perfil = c3.selectbox("Perfil", ["usuario", "admin"],
                                   help="`admin` enxerga esta página. Não há outra "
                                        "diferença de permissão hoje.")
        modo = c4.radio("Senha", ["Sortear uma forte", "Definir manualmente"],
                        horizontal=False)
        senha_manual = st.text_input("Senha (se manual)", type="password",
                                     help="Mínimo de 10 caracteres.")
        emitir = st.form_submit_button("Gerar bloco de credencial", type="primary")

    if emitir:
        erros = []
        login_norm = (novo_login or "").strip().lower()
        if not login_norm or " " in login_norm:
            erros.append("Login vazio ou com espaço.")
        if login_norm in cadastrados:
            erros.append(f"Já existe um usuário `{login_norm}`.")
        if modo == "Definir manualmente" and len(senha_manual or "") < 10:
            erros.append("Senha manual com menos de 10 caracteres.")

        if erros:
            for e in erros:
                st.error(e)
        else:
            senha = (_secrets.token_urlsafe(16) if modo == "Sortear uma forte"
                     else senha_manual)
            with st.spinner("Derivando o hash (PBKDF2, 600 mil iterações)..."):
                registro = autenticacao.gerar_hash(senha)

            st.success("Credencial gerada. Ela ainda **não** está ativa — falta colar.")
            st.markdown("**1. Envie a senha à pessoa** (por canal privado; ela não é "
                        "recuperável depois desta tela):")
            st.code(senha, language=None)
            st.markdown("**2. Cole este bloco** no `secrets` — painel do Streamlit Cloud "
                        "(*Settings → Secrets*) ou `app/.streamlit/secrets.toml` local. "
                        "No Cloud o app reinicia sozinho ao salvar:")
            st.code(
                f'[auth.usuarios.{login_norm}]\n'
                f'nome = "{(novo_nome or login_norm).strip()}"\n'
                f'perfil = "{novo_perfil}"\n'
                f'senha_hash = "{registro}"',
                language="toml",
            )
            st.caption("A senha em claro não é registrada na trilha nem gravada em disco "
                       "— só o hash sai daqui.")
            auditoria.registrar_acao("credencial_emitida", alvo=login_norm,
                                     detalhe=f"perfil {novo_perfil}")

# ── Trilha de uso ────────────────────────────────────────────────────────────
with aba_trilha:
    st.caption(f"Backend: {auditoria.backend_ativo()}")

    eventos = auditoria.ler_eventos()
    if eventos.empty:
        st.info("Nada registrado ainda.")
        st.stop()

    # Sessões e duração: a duração é o maior `segundos_sessao` de cada `sessao_id`,
    # porque não existe evento de saída confiável — ver acesso/README.md.
    com_sessao = eventos[eventos["sessao_id"].astype(str).str.len() > 0].copy()
    com_sessao["segundos_sessao"] = pd.to_numeric(
        com_sessao["segundos_sessao"], errors="coerce")
    duracoes = com_sessao.groupby("sessao_id")["segundos_sessao"].max()

    m = st.columns(4, border=True)
    m[0].metric("Eventos", f"{len(eventos):,}".replace(",", "."))
    m[1].metric("Sessões", f"{duracoes.size:,}".replace(",", "."))
    m[2].metric("Usuários distintos", eventos["usuario"].nunique())
    m[3].metric("Sessão mediana",
                f"{duracoes.median() / 60:.0f} min" if duracoes.size else "—",
                help="Mediana, não média: uma aba esquecida aberta distorce a média.")

    falhas = int((eventos["evento"] == "login_falha").sum())
    if falhas:
        st.warning(f"{falhas} tentativa(s) de login malsucedida(s) no período registrado. "
                   "Filtre por `login_falha` abaixo para ver quando e com qual login.")

    f1, f2 = st.columns(2)
    tipos = sorted(eventos["evento"].astype(str).unique())
    filtro_evento = f1.multiselect("Evento", tipos, default=[])
    filtro_usuario = f2.multiselect("Usuário",
                                    sorted(eventos["usuario"].astype(str).unique()),
                                    default=[])

    vista = eventos
    if filtro_evento:
        vista = vista[vista["evento"].isin(filtro_evento)]
    if filtro_usuario:
        vista = vista[vista["usuario"].isin(filtro_usuario)]

    st.dataframe(vista, hide_index=True, use_container_width=True, height=420)
    st.download_button("⬇️  Baixar trilha (.csv)",
                       vista.to_csv(index=False).encode("utf-8"),
                       file_name="trilha_uso.csv", mime="text/csv")
