"""
Tela inicial — hub de aplicações Houer.

Mostra cards clicáveis para cada aplicação disponível. Para adicionar uma nova
aplicação ao hub, atualize a lista `PAGINAS` em `app.py` e (opcionalmente) adicione
uma entrada de card em `APPS_CARDS` abaixo.
"""

import streamlit as st

APPS_CARDS = [
    {
        "page_url": "simulacao",
        "icon": "💡",
        "title": "Simulação NBR 5101",
        "subtitle": "Predição de iluminação com ML",
        "description": (
            "Simulação individual e em lote de luminárias com base em modelos "
            "treinados (potência, lúmens, uniformidade, iluminância). "
            "Conformidade automática com a NBR 5101 e sugestões estruturais."
        ),
        "tags": ["ML", "NBR 5101", "Individual + Lote"],
    },
    {
        "page_url": "cadastro",
        "icon": "🗂️",
        "title": "Análise de Cadastro",
        "subtitle": "Tratamento e diagnóstico de cadastros municipais",
        "description": (
            "Recebe cadastro, inspeção de campo e bases IAE/ID de um município "
            "e produz as três planilhas padronizadas (Classificação, Análise de "
            "Cadastro, Quantitativo por Uso Final) + relatório de execução."
        ),
        "tags": ["Cadastro", "Inspeção", "IAE/ID", "ANEEL 2590/2019"],
    },
    {
        "page_url": "premissas",
        "page_file": "paginas/premissas_ip.py",
        "icon": "🧮",
        "title": "Planilha de Engenharia IP",
        "subtitle": "Premissas → inputs parametrizados + relatório",
        "description": (
            "Coleta as premissas do município e as proposições (IAE, ID, demanda "
            "reprimida, marcos, prazo). Importa DTO e planilhas (Extrapolação, "
            "Proposição IAE, InvBens) para auto-preencher e gera a planilha de inputs "
            "parametrizada por fórmulas + os blocos do relatório de engenharia."
        ),
        "tags": ["DTO + Planilhas", "Fórmulas vivas", "Inputs IP", "Relatório"],
    },
]


def _render_card(app: dict):
    tags_html = "".join(
        f'<span style="background: rgba(0,169,224,0.15); color: #00A9E0; '
        f'padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.72rem; '
        f'font-weight: 600; margin-right: 0.4rem;">{t}</span>'
        for t in app["tags"]
    )

    st.markdown(
        f"""
        <div style="
            background: rgba(18, 25, 43, 0.6);
            backdrop-filter: blur(10px);
            border: 1px solid #1f2937;
            border-top: 4px solid #00A9E0;
            border-radius: 20px;
            padding: 1.8rem;
            min-height: 290px;
            margin-bottom: 0.8rem;
        ">
            <div style="font-size: 2.8rem; margin-bottom: 0.5rem;">{app["icon"]}</div>
            <div style="font-size: 1.35rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.2rem;">
                {app["title"]}
            </div>
            <div style="font-size: 0.85rem; color: #00A9E0; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 1rem;">
                {app["subtitle"]}
            </div>
            <div style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 1.2rem;">
                {app["description"]}
            </div>
            <div>{tags_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page_file = app.get("page_file") or (
        f"paginas/{app['page_url'].replace('cadastro', 'analise_cadastro').replace('simulacao', 'simulacao_nbr')}.py"
    )
    st.page_link(
        page_file,
        label=f"Abrir {app['title']}  →",
        use_container_width=True,
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="margin-top: 1rem; margin-bottom: 2rem;">
        <div style="font-size: 3.2rem; font-weight: 800;
                    background: linear-gradient(90deg, #ffffff, #00A9E0);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    letter-spacing: -1px; line-height: 1.1;">
            Plataforma Houer
        </div>
        <div style="font-size: 1.1rem; color: #94a3b8; margin-top: 0.5rem;">
            Soluções de engenharia para iluminação pública municipal
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div style="font-size: 1.3rem; font-weight: 700; color: #f8fafc; '
    'margin-bottom: 1.2rem; padding-bottom: 0.6rem; border-bottom: 1px solid #1f2937;">'
    "Aplicações disponíveis</div>",
    unsafe_allow_html=True,
)

cols = st.columns(2, gap="large")
for idx, app in enumerate(APPS_CARDS):
    with cols[idx % 2]:
        _render_card(app)

st.markdown(
    """
    <div style="margin-top: 2.5rem; padding: 1.5rem; border-radius: 12px;
                background: rgba(27, 54, 100, 0.15); border-left: 3px solid #00A9E0;">
        <div style="font-size: 0.95rem; color: #cbd5e1;">
            <b style="color: #f8fafc;">Novas funcionalidades em breve.</b>
            A plataforma é modular — novas ferramentas podem ser adicionadas como páginas independentes.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
