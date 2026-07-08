"""
Análise de Cadastro de Iluminação Pública — pipeline de tratamento.

Implementa o agente descrito em `agente_ip_instrucoes_v1.4.md`:
recebe 4 inputs (cadastro, inspeção, IAE, ID), aplica tratamento técnico,
e gera 3 planilhas + relatório de execução.

Esta página é apenas a UI; a lógica do pipeline vive em `cadastro_ip/`.
"""

import pandas as pd
import streamlit as st

from cadastro_ip import aneel_2590, pipeline, relatorio
from cadastro_ip.saidas import analise_cadastro as saida_analise
from cadastro_ip.saidas import classificacao_pontos as saida_classificacao
from cadastro_ip.saidas import quantitativo_uso_final as saida_quantitativo


# ── Estado da sessão ──────────────────────────────────────────────────────────
SS_KEYS = {
    "cadastro": "ac_cadastro",
    "inspecao": "ac_inspecao",
    "iae": "ac_iae",
    "id": "ac_id",
    "municipio": "ac_municipio",
    "uf": "ac_uf",
    "tempo_operacao": "ac_tempo_operacao",
    "resultado": "ac_resultado",
}
for k in SS_KEYS.values():
    st.session_state.setdefault(k, None)


# ── Estilo local da página ────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .ac-hero-title {
            font-size: 2.6rem; font-weight: 800;
            background: linear-gradient(90deg, #ffffff, #00A9E0);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            letter-spacing: -1px; margin-bottom: 0.2rem;
        }
        .ac-hero-sub { font-size: 1rem; color: #94a3b8; margin-bottom: 1.5rem; }
        .ac-step {
            background: rgba(18, 25, 43, 0.6);
            border: 1px solid #1f2937;
            border-left: 4px solid #00A9E0;
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
        }
        .ac-step-title {
            font-size: 1.05rem; font-weight: 700; color: #f8fafc;
            margin-bottom: 0.4rem;
        }
        .ac-step-desc {
            font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.8rem;
        }
        .ac-status-ok    { color: #22c55e; font-weight: 600; }
        .ac-status-miss  { color: #f59e0b; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="ac-hero-title">🗂️ Análise de Cadastro</div>
    <div class="ac-hero-sub">
        Tratamento e diagnóstico técnico de cadastros municipais de iluminação pública —
        gera as 3 planilhas padronizadas + relatório de execução.
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Helpers de leitura de planilha ────────────────────────────────────────────
_TOKENS_HEADER = {
    "etiqueta", "id_ponto", "id pip", "identificador", "idg", "cod_id",
    "tecnologia", "tipo de lampada", "tipo lampada", "logradouro",
    "endereco", "latitude", "longitude",
}


def _detectar_linha_header(df_bruto: pd.DataFrame, max_linhas: int = 10) -> int:
    """
    Detecta em que linha está o cabeçalho real de uma planilha bruta.
    Procura a primeira linha que tenha múltiplos tokens conhecidos (etiqueta,
    tecnologia, latitude, etc.) — útil para arquivos com título em linhas 1-2.
    Retorna o índice da linha do cabeçalho (0-based). Default: 0.
    """
    import unicodedata, re

    def _slug(s):
        s = unicodedata.normalize("NFD", str(s))
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

    for r in range(min(max_linhas, len(df_bruto))):
        row = df_bruto.iloc[r].dropna().astype(str).tolist()
        slugs = {_slug(v) for v in row}
        # Contagem de matches contra os tokens conhecidos
        hits = sum(1 for token in _TOKENS_HEADER if any(token in s for s in slugs))
        if hits >= 2:
            return r
    return 0


def _ler_planilha(uploaded_file) -> pd.DataFrame | None:
    """
    Lê um arquivo xlsx/xls/csv enviado e retorna um DataFrame, ou None.
    Detecta automaticamente cabeçalho em multi-linha (ex: arquivos Q-E_*).
    """
    if uploaded_file is None:
        return None
    nome = uploaded_file.name.lower()
    try:
        if nome.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        # Lê sem header para detectar a linha correta
        df_bruto = pd.read_excel(uploaded_file, sheet_name=0, header=None)
        linha_header = _detectar_linha_header(df_bruto)
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file, sheet_name=0, header=linha_header)
    except Exception as exc:
        st.error(f"Erro ao ler **{uploaded_file.name}**: {exc}")
        return None


# ── Passo 1: Upload dos 4 inputs obrigatórios ─────────────────────────────────
st.markdown(
    '<div class="ac-step">'
    '<div class="ac-step-title">Passo 1 — Upload dos 4 inputs obrigatórios</div>'
    '<div class="ac-step-desc">'
    'Envie cadastro completo do município, base de inspeção amostral, base IAE '
    '(Iluminação de Áreas Especiais) e base ID (Iluminação Destacada). '
    'Aceita .xlsx, .xls e .csv.'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

col_a, col_b = st.columns(2)
with col_a:
    up_cad = st.file_uploader(
        "📋 Cadastro de iluminação pública",
        type=["xlsx", "xls", "csv"],
        key="up_cadastro",
        help="Planilha completa de todos os pontos do município.",
    )
    up_iae = st.file_uploader(
        "🏛️ Base IAE — Iluminação de Áreas Especiais",
        type=["xlsx", "xls", "csv"],
        key="up_iae",
        help="Planilha à parte com pontos classificados como IAE.",
    )
with col_b:
    up_insp = st.file_uploader(
        "🔍 Base de inspeção de campo (Base_HouerApp_IV)",
        type=["xlsx", "xls", "csv"],
        key="up_inspecao",
        help="Amostra inspecionada em campo.",
    )
    up_id = st.file_uploader(
        "💡 Base ID — Iluminação Destacada",
        type=["xlsx", "xls", "csv"],
        key="up_id",
        help="Planilha à parte com pontos classificados como ID.",
    )

# Lê os arquivos e armazena na sessão (mantém entre reruns)
if up_cad is not None:
    st.session_state[SS_KEYS["cadastro"]] = _ler_planilha(up_cad)
if up_insp is not None:
    st.session_state[SS_KEYS["inspecao"]] = _ler_planilha(up_insp)
if up_iae is not None:
    st.session_state[SS_KEYS["iae"]] = _ler_planilha(up_iae)
if up_id is not None:
    st.session_state[SS_KEYS["id"]] = _ler_planilha(up_id)


# ── Resumo de status dos uploads ──────────────────────────────────────────────
def _badge(df, label):
    if df is None:
        return f'<span class="ac-status-miss">○ {label} pendente</span>'
    return f'<span class="ac-status-ok">● {label}: {len(df):,} linhas</span>'.replace(",", ".")


cad_df = st.session_state[SS_KEYS["cadastro"]]
insp_df = st.session_state[SS_KEYS["inspecao"]]
iae_df = st.session_state[SS_KEYS["iae"]]
id_df = st.session_state[SS_KEYS["id"]]

status_html = " &nbsp;·&nbsp; ".join(
    [
        _badge(cad_df, "Cadastro"),
        _badge(insp_df, "Inspeção"),
        _badge(iae_df, "IAE"),
        _badge(id_df, "ID"),
    ]
)
st.markdown(f'<div style="margin: 0.5rem 0 1.5rem 0; font-size: 0.95rem;">{status_html}</div>', unsafe_allow_html=True)

todos_uploads_ok = all(df is not None for df in (cad_df, insp_df, iae_df, id_df))


# ── Passo 2: Identificação do município ───────────────────────────────────────
st.markdown(
    '<div class="ac-step">'
    '<div class="ac-step-title">Passo 2 — Identificação do município</div>'
    '<div class="ac-step-desc">'
    'Confirme município e UF — o tempo de operação diário será buscado na '
    'Resolução Homologatória ANEEL nº 2.590/2019.'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

col_m, col_uf = st.columns([3, 1])
with col_m:
    municipio_input = st.text_input(
        "Município",
        value=st.session_state[SS_KEYS["municipio"]] or "",
        placeholder="Ex: Sorocaba",
        disabled=not todos_uploads_ok,
    )
with col_uf:
    ufs = [
        "", "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG",
        "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO",
    ]
    uf_atual = st.session_state[SS_KEYS["uf"]] or ""
    uf_input = st.selectbox(
        "UF",
        ufs,
        index=ufs.index(uf_atual) if uf_atual in ufs else 0,
        disabled=not todos_uploads_ok,
    )

st.session_state[SS_KEYS["municipio"]] = municipio_input or None
st.session_state[SS_KEYS["uf"]] = uf_input or None


# ── Lookup ANEEL automático (com fallback manual) ────────────────────────────
horas_manual, minutos_manual = None, None
if municipio_input and uf_input:
    tempo_lookup = aneel_2590.buscar(municipio_input, uf_input)
    if tempo_lookup is not None:
        st.success(
            f"⏱️ Tempo de operação (ANEEL 2590/2019): **{tempo_lookup.formato_hhmm}** "
            f"para {tempo_lookup.municipio}/{tempo_lookup.uf}"
            + (f" (IBGE {tempo_lookup.codigo_ibge})" if tempo_lookup.codigo_ibge else "")
        )
    else:
        if not aneel_2590.base_disponivel():
            st.warning(
                "⚠️ Base ANEEL 2590/2019 ainda não foi extraída do PDF. "
                "Informe o tempo de operação manualmente abaixo (formato HHhMM ou decimal)."
            )
        else:
            st.warning(
                f"⚠️ Município **{municipio_input}/{uf_input}** não encontrado na base ANEEL. "
                "Informe manualmente abaixo (formato HHhMM ou decimal)."
            )
        col_h, col_m = st.columns(2)
        with col_h:
            horas_manual = st.number_input("Horas", min_value=0, max_value=24, value=11, step=1)
        with col_m:
            minutos_manual = st.number_input("Minutos", min_value=0, max_value=59, value=30, step=1)


# ── Passo 3: Processamento ────────────────────────────────────────────────────
st.markdown(
    '<div class="ac-step">'
    '<div class="ac-step-title">Passo 3 — Processar pipeline</div>'
    '<div class="ac-step-desc">'
    'Aplica normalização, roteamento (IAE / ID / LED IV / Convencional), '
    'fator de extrapolação, perdas de reator (Tabela 4 ABNT) e propagação de '
    'Classe Via. Gera os 3 arquivos .xlsx + relatório.'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

pronto = todos_uploads_ok and municipio_input and uf_input
if not pronto:
    faltando = []
    if not todos_uploads_ok:
        faltando.append("os 4 uploads")
    if not municipio_input or not uf_input:
        faltando.append("município + UF")
    st.info(f"⏳ Para processar, complete: {', '.join(faltando)}.")

btn = st.button("▶️ Processar pipeline", type="primary", disabled=not pronto, use_container_width=True)

if btn:
    try:
        with st.spinner("Processando cadastro..."):
            resultado_pipeline = pipeline.executar(
                cadastro=cad_df,
                inspecao=insp_df,
                iae=iae_df,
                id_=id_df,
                municipio=municipio_input,
                uf=uf_input,
                horas_operacao_manual=int(horas_manual) if horas_manual is not None else None,
                minutos_operacao_manual=int(minutos_manual) if minutos_manual is not None else None,
            )
            # Gera os 3 .xlsx em memória
            xlsx_classif = saida_classificacao.gerar(resultado_pipeline)
            xlsx_analise = saida_analise.gerar(resultado_pipeline)
            xlsx_quant = saida_quantitativo.gerar(resultado_pipeline)
            texto_relatorio = relatorio.gerar(resultado_pipeline)

            resultado_pipeline.xlsx_classificacao = xlsx_classif
            resultado_pipeline.xlsx_analise_cadastro = xlsx_analise
            resultado_pipeline.xlsx_quantitativo = xlsx_quant
            resultado_pipeline.relatorio_texto = texto_relatorio

            st.session_state[SS_KEYS["resultado"]] = resultado_pipeline
            st.success("✅ Pipeline executado com sucesso!")
    except Exception as exc:
        import traceback
        st.error(f"❌ Erro durante o processamento: {exc}")
        with st.expander("Detalhes do erro"):
            st.code(traceback.format_exc())


# ── Passo 4: Resultados e downloads ───────────────────────────────────────────
resultado = st.session_state[SS_KEYS["resultado"]]
if resultado is not None:
    st.markdown(
        '<div class="ac-step">'
        '<div class="ac-step-title">Passo 4 — Downloads e relatório</div>'
        '<div class="ac-step-desc">'
        'Os três arquivos .xlsx ficam disponíveis abaixo. Revise os alertas do '
        'relatório (coluna Executado, pontos de 1000W, divergências de classe, '
        'pontos sem classe atribuída).'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Métricas de topo ─────────────────────────────────────────────────────
    cm1, cm2, cm3, cm4 = st.columns(4)
    cm1.metric("Cadastro", f"{resultado.total_cadastro:,}".replace(",", "."))
    cm2.metric("Inspeção", f"{resultado.total_inspecao:,}".replace(",", "."))
    cm3.metric("Fator extrap.", resultado.fator_extrapolacao.fator)
    cm4.metric(
        "Tempo operação",
        resultado.tempo_operacao.formato_hhmm if resultado.tempo_operacao else "—",
    )

    # ── Botões de download ───────────────────────────────────────────────────
    municipio_slug = (resultado.municipio or "municipio").replace(" ", "_")
    cd1, cd2, cd3 = st.columns(3)
    with cd1:
        st.download_button(
            "📄 Classificação dos Pontos.xlsx",
            data=resultado.xlsx_classificacao,
            file_name=f"{municipio_slug} - Classificação dos Pontos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with cd2:
        st.download_button(
            "📊 Analise Cadastro.xlsx",
            data=resultado.xlsx_analise_cadastro,
            file_name=f"{municipio_slug} - Analise Cadastro.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with cd3:
        st.download_button(
            "📈 Quantitativo por Uso Final.xlsx",
            data=resultado.xlsx_quantitativo,
            file_name=f"{municipio_slug} - Quantitativo por Uso Final.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    # ── Relatório markdown + botão de download ───────────────────────────────
    st.markdown("---")
    st.markdown(resultado.relatorio_texto)
    st.download_button(
        "📝 Baixar relatório (.md)",
        data=resultado.relatorio_texto.encode("utf-8"),
        file_name=f"{municipio_slug} - Relatório de Execução.md",
        mime="text/markdown",
        use_container_width=False,
    )
