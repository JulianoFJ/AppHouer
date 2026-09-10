"""
Administração de acessos — visível apenas para quem tem `perfil = "admin"`.

Duas funções: gerenciar quem tem credencial e ler a trilha de uso.

Por que não existe "criar conta" na tela de login
--------------------------------------------------
Autocadastro anularia o controle de acesso: qualquer visitante criaria a própria conta
e entraria. Num portal cujo propósito é justamente conter modelos, bases e metodologia,
isso seria o mesmo que não ter senha. O modelo aqui é **provisionamento por
administrador** — alguém já autorizado emite a credencial de quem entra.

Esta página grava direto na tabela `users` (via `acesso.autenticacao`), a mesma que o
login consulta — não existe mais um passo manual de colar TOML no meio: com Postgres
disponível, cadastrar ou revogar aqui já é o que vale na próxima tentativa de login,
sem redeploy nem edição de arquivo.
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
    cadastrados = autenticacao.usuarios_cadastrados()
    st.markdown("#### Quem tem acesso hoje")
    if cadastrados:
        st.dataframe(
            pd.DataFrame([
                {"Login": login,
                 "Nome": dados.get("nome", ""),
                 "Perfil": dados.get("perfil", "usuario"),
                 "Ativo": "Sim" if dados.get("ativo") else "Não"}
                for login, dados in sorted(cadastrados.items())
            ]),
            hide_index=True, use_container_width=True,
        )
    else:
        st.info("Nenhum usuário cadastrado.")

    st.divider()
    st.markdown("#### Cadastrar ou atualizar")
    st.caption(
        "Um login que já existe é **atualizado** (nome, perfil e senha), e reativado "
        "se estava desativado — é assim que se troca a senha de alguém, sem tela à parte."
    )

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
        confirmar_existente = st.checkbox(
            "Sei que este login pode já existir e quero sobrescrever nome/perfil/senha",
            help="Só precisa marcar se o login digitado já estiver na tabela acima — "
                 "evita trocar a senha de outra pessoa por um login digitado errado.",
        )
        emitir = st.form_submit_button("Salvar", type="primary")

    if emitir:
        erros = []
        login_norm = (novo_login or "").strip().lower()
        era_existente = login_norm in cadastrados
        if not login_norm or " " in login_norm:
            erros.append("Login vazio ou com espaço.")
        if modo == "Definir manualmente" and len(senha_manual or "") < 10:
            erros.append("Senha manual com menos de 10 caracteres.")
        if era_existente and not confirmar_existente:
            erros.append(
                f"O login `{login_norm}` já existe. Se é isso mesmo — troca de senha ou "
                "de perfil — marque a confirmação acima e clique em Salvar de novo."
            )

        if erros:
            for e in erros:
                st.error(e)
        else:
            senha = (_secrets.token_urlsafe(16) if modo == "Sortear uma forte"
                     else senha_manual)
            nome_norm = (novo_nome or login_norm).strip()
            try:
                with st.spinner("Derivando o hash (PBKDF2, 600 mil iterações) e gravando..."):
                    autenticacao.criar_ou_atualizar_usuario(
                        login_norm, nome_norm, novo_perfil, senha)
            except Exception as exc:
                st.error(f"Falha ao gravar no banco: {exc}")
                st.stop()

            st.success(
                f"Usuário `{login_norm}` {'atualizado' if era_existente else 'cadastrado'} "
                "e **já pode logar** — nenhum passo manual a mais."
            )
            st.markdown("**Envie a senha à pessoa** (por canal privado; ela não é "
                        "recuperável depois desta tela):")
            st.code(senha, language=None)
            st.caption("A senha em claro não é registrada na trilha nem gravada em disco "
                       "— só o hash vai para o banco.")
            auditoria.registrar_acao(
                "credencial_atualizada" if era_existente else "credencial_emitida",
                alvo=login_norm, detalhe=f"perfil {novo_perfil}",
            )
            st.rerun()

    st.divider()
    st.markdown("#### Ativar / desativar acesso")
    st.caption(
        "Desativar não apaga o login nem o histórico dele na trilha de uso — só "
        "impede novas entradas. Reative a qualquer momento, ou emita senha nova acima."
    )
    if cadastrados:
        alvo = st.selectbox("Login", sorted(cadastrados), key="alvo_status")
        ativo_hoje = bool(cadastrados[alvo].get("ativo"))
        c1, c2 = st.columns([1, 3])
        with c1:
            if ativo_hoje:
                if st.button("Desativar", type="secondary"):
                    try:
                        autenticacao.definir_ativo(alvo, False)
                    except Exception as exc:
                        st.error(f"Falha ao gravar no banco: {exc}")
                        st.stop()
                    auditoria.registrar_acao("acesso_revogado", alvo=alvo)
                    st.rerun()
            else:
                if st.button("Reativar", type="primary"):
                    try:
                        autenticacao.definir_ativo(alvo, True)
                    except Exception as exc:
                        st.error(f"Falha ao gravar no banco: {exc}")
                        st.stop()
                    auditoria.registrar_acao("acesso_reativado", alvo=alvo)
                    st.rerun()
        with c2:
            st.caption(f"Status atual: {'ativo' if ativo_hoje else 'desativado'}.")

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
