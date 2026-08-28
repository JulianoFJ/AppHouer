"""
Hub Houer — entry point único para as aplicações de engenharia de iluminação pública.

Cada aplicação é uma página independente em `paginas/`. Para adicionar uma nova função
no futuro, crie `paginas/nova_app.py` (script Streamlit comum) e registre uma
entrada na lista `PAGINAS` abaixo.
"""

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Houer — Plataforma de Iluminação Pública",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilo global Houer (Navy + Teal) ─────────────────────────────────────────
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        :root {
            --houer-navy: #1B3664;
            --houer-teal: #00A9E0;
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

# ── Branding no topo da sidebar (renderizado antes da navegação) ──────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 0.5rem 0 1.2rem 0; border-bottom: 1px solid #1f2937; margin-bottom: 0.5rem;">
            <div style="font-size: 1.6rem; font-weight: 800;
                        background: linear-gradient(90deg, #ffffff, #00A9E0);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                        letter-spacing: -0.5px;">HOUER</div>
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
    st.Page(str(BASE / "premissas_ip.py"),      title="Planilha de Engenharia IP", icon="🧮", url_path="premissas"),
    st.Page(str(BASE / "hub_municipios.py"),    title="Hub de Municípios",  icon="🌎", url_path="municipios"),
]

pg = st.navigation({"Houer": PAGINAS}, position="sidebar")
pg.run()
