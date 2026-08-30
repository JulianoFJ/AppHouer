"""
Plataforma IP — entry point único para as aplicações de engenharia de iluminação pública.

Cada aplicação é uma página independente em `paginas/`. Para adicionar uma nova função
no futuro, crie `paginas/nova_app.py` (script Streamlit comum) e registre uma
entrada na lista `PAGINAS` abaixo.

Ordem de execução, que é o que garante o controle de acesso: `st.set_page_config` →
estilo → `exigir_login()` → navegação. Como o Streamlit roda o script de cima para
baixo e `exigir_login()` termina em `st.stop()` quando não há sessão, nenhuma página
chega a ser registrada — muito menos executada — para quem não está autenticado.
"""

from pathlib import Path

import streamlit as st

from acesso import encerrar_sessao, exigir_login, registrar_evento, registrar_pagina

st.set_page_config(
    page_title="Plataforma IP — Iluminação Pública",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilo global (Navy + Teal) ───────────────────────────────────────────────
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        :root {
            --marca-navy: #1B3664;
            --marca-teal: #00A9E0;
            --bg-dark: #0b111e;
            --card-bg: #12192b;
            --card-border: #1f2937;
        }

        .stApp {
            background:
                radial-gradient(circle at 0% 0%, #1B366433, transparent),
                radial-gradient(circle at 100% 100%, #00A9E011, transparent),
                #0b111e;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b111e 0%, #12192b 100%);
            border-right: 1px solid var(--card-border);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Porta de entrada ──────────────────────────────────────────────────────────
# Tudo abaixo desta linha só roda para usuário autenticado.
usuario = exigir_login()

# ── Identidade do portal no topo da sidebar ───────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 0.5rem 0 1.2rem 0; border-bottom: 1px solid #1f2937; margin-bottom: 0.5rem;">
            <div style="font-size: 1.5rem; font-weight: 800;
                        background: linear-gradient(90deg, #ffffff, #00A9E0);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                        letter-spacing: -0.5px;">PLATAFORMA IP</div>
            <div style="font-size: 0.72rem; color: #94a3b8; letter-spacing: 0.15em; text-transform: uppercase;">
                Iluminação Pública
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Registro de páginas ───────────────────────────────────────────────────────
# Para adicionar nova função no futuro, basta criar paginas/nome.py
# (script Streamlit comum) e registrar aqui.
BASE = Path(__file__).parent / "paginas"
PAGINAS = [
    st.Page(str(BASE / "home.py"),              title="Início",             icon="🏠", default=True, url_path="home"),
    st.Page(str(BASE / "simulacao_nbr.py"),     title="Simulação NBR 5101", icon="💡", url_path="simulacao"),
    st.Page(str(BASE / "analise_cadastro.py"),  title="Análise de Cadastro", icon="🗂️", url_path="cadastro"),
    st.Page(str(BASE / "hub_municipios.py"),    title="Hub de Municípios",  icon="🌎", url_path="municipios"),
]

# ── Desativado temporariamente em 30/08/2026 ─────────────────────────────────
# A Planilha de Engenharia IP saiu do menu a pedido do usuário. NADA foi removido:
# `paginas/premissas_ip.py`, o pacote `premissas_ip/` e os testes seguem intactos.
# Para reativar, basta descomentar a linha abaixo e o card correspondente em
# `paginas/home.py` (procure por "DESATIVADO" lá) — os dois precisam voltar juntos,
# porque `st.page_link` para uma página não registrada quebra a home.
#
# PAGINAS.insert(3, st.Page(str(BASE / "premissas_ip.py"),
#                           title="Planilha de Engenharia IP", icon="🧮",
#                           url_path="premissas"))

# A administração de acessos é registrada só para admin — a própria página revalida o
# perfil, porque a navegação do Streamlit é por URL e `/administracao` é adivinhável.
SECOES = {"Plataforma IP": PAGINAS}
if usuario.perfil == "admin":
    SECOES["Administração"] = [
        st.Page(str(BASE / "administracao.py"), title="Acessos e uso", icon="🔐",
                url_path="administracao"),
    ]

pg = st.navigation(SECOES, position="sidebar")

# ── Rodapé da sidebar: quem está logado e a saída ─────────────────────────────
# Depois de `st.navigation` para que apareça abaixo do menu, e antes de `pg.run()`
# para que o rodapé exista mesmo se a página levantar exceção.
with st.sidebar:
    st.divider()
    st.caption(f"Conectado como **{usuario.nome}**")
    if st.button("Sair", use_container_width=True):
        saindo = encerrar_sessao()
        registrar_evento("logout", usuario=saindo)
        st.rerun()

registrar_pagina(pg.title)
pg.run()
