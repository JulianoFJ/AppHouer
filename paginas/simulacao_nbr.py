import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from geopy.geocoders import GoogleV3
import plotly.express as px
import plotly.graph_objects as go

def get_google_maps_api_key():
    """Lê a chave da API via Streamlit Secrets ou variável de ambiente."""
    try:
        if "GOOGLE_MAPS_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_MAPS_API_KEY"]
    except Exception:
        pass
    return os.getenv("GOOGLE_MAPS_API_KEY")

GOOGLE_MAPS_API_KEY = get_google_maps_api_key()

# ── Página renderizada via app.py (hub) ───────────────────────────────────────
# st.set_page_config é chamado pelo entry point `app.py`. Não duplicar aqui.

# Inicializa estado para Simulação em Lote
if 'df_lote' not in st.session_state:
    st.session_state.df_lote = None

# ── Estilo ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Paleta: Navy (#1B3664), Teal (#00A9E0) */
    :root {
        --marca-navy: #1B3664;
        --marca-teal: #00A9E0;
        --bg-claro: #F4F6FA;
        --card-bg: #FFFFFF;
        --card-border: #E2E8F0;
    }

    .stApp {
        background: radial-gradient(circle at 0% 0%, #1B36640D, transparent),
                    radial-gradient(circle at 100% 100%, #00A9E00D, transparent),
                    #F4F6FA;
    }

    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, var(--marca-navy), var(--marca-teal));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0rem;
        letter-spacing: -1px;
    }
    .hero-sub {
        font-size: 1.1rem;
        color: #5B6579;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    .forn-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-top: 4px solid var(--marca-teal);
        border-radius: 20px;
        padding: 1.8rem;
        box-shadow: 0 1px 3px rgba(27, 36, 52, 0.06);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .forn-card:hover {
        transform: translateY(-8px);
        border-color: var(--marca-teal);
        background: #F0F7FC;
        box-shadow: 0 20px 40px rgba(27, 36, 52, 0.12);
    }
    .forn-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1B2434;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #5B6579;
        margin-top: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #1B2434; /* Valor sempre no texto mais forte, para destaque */
    }
    .metric-unit {
        font-size: 0.9rem;
        font-weight: 400;
        color: #64748b;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1B2434;
        margin: 3rem 0 1.5rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .section-title::before {
        content: "";
        display: block;
        width: 6px;
        height: 28px;
        background: var(--marca-teal);
        border-radius: 3px;
    }

    div[data-testid="stSidebar"] {
        background: var(--card-bg);
        border-right: 1px solid var(--card-border);
    }
</style>
""", unsafe_allow_html=True)

# ── Previsão de iluminação (pacote de domínio) ────────────────────────────────
# Carregamento de modelos, regras de engenharia e geração de PDF vivem em
# `previsao_iluminacao/` (extraído de dentro desta página em 10/09/2026) — a página
# fica só com a UI. Cache (`st.cache_resource`/`st.cache_data`) continua aqui: o
# pacote não importa Streamlit, mesma convenção de `amostragem_ip`/`cadastro_ip`.
from previsao_iluminacao import constantes as _pi_const
from previsao_iluminacao import historico as _pi_historico
from previsao_iluminacao import modelos as _pi_modelos
from previsao_iluminacao import planilhas as _pi_planilhas
from previsao_iluminacao import relatorio as _pi_relatorio

FORNECEDORES = _pi_const.FORNECEDORES
CORES = _pi_const.CORES
TARGETS_MAP = _pi_const.TARGETS_MAP
UNITS_MAP = _pi_const.UNITS_MAP
BRACOS_PROJECAO = _pi_const.BRACOS_PROJECAO
BRACOS_ORDENADOS = _pi_const.BRACOS_ORDENADOS
projecao_para_braco = _pi_modelos.projecao_para_braco
TEMPLATE_COLUMNS = _pi_const.TEMPLATE_COLUMNS
MAPEAMENTO_COLS = _pi_const.MAPEAMENTO_COLS
NBR5101 = _pi_const.NBR5101

# ── Carrega modelos ───────────────────────────────────────────────────────────
@st.cache_resource
def carregar_modelos(suffix=""):
    return _pi_modelos.carregar_modelos(suffix)

@st.cache_resource
def carregar_classificadores(suffix=""):
    return _pi_modelos.carregar_classificadores(suffix)

prever_metricas_com_dependencia_w = _pi_modelos.prever_metricas_com_dependencia_w
analisar_melhorias = _pi_modelos.analisar_melhorias

@st.cache_data
def carregar_banco_luminarias():
    return _pi_historico.carregar_banco_luminarias()

buscar_custo = _pi_historico.buscar_custo
banco_luminarias = carregar_banco_luminarias()

@st.cache_data
def carregar_media_historica():
    return _pi_historico.carregar_media_historica()

medias_historicas = carregar_media_historica()

formatar_resultado_template = _pi_planilhas.formatar_resultado_template
formatar_tabela_resultado = _pi_planilhas.formatar_tabela_resultado
formatar_tabela_dinamica = _pi_planilhas.formatar_tabela_dinamica

gerar_pdf = _pi_relatorio.gerar_pdf


# ── Header ────────────────────────────────────────────────────────────────────
# Sem logotipo: a identidade do portal é textual e vive na sidebar do `app.py`, para
# que exista um único lugar a mexer se o nome mudar de novo.
#
# Estilo inline, e não as classes `hero-title`/`hero-sub`: aquelas foram calibradas
# para uma palavra curta (o antigo logotipo textual) e usam margem negativa para colar
# o subtítulo — com um título longo as duas linhas se sobrepõem.
st.markdown(
    """
    <div style="margin: 0.4rem 0 0.2rem 0;">
      <div style="font-size:2.6rem; font-weight:800; line-height:1.15;
                  background: linear-gradient(90deg,#1B3664,#00A9E0);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                  letter-spacing:-1px;">Simulação NBR 5101</div>
      <div style="font-size:1.02rem; color:#5B6579; margin-top:0.35rem;">
        Dimensionamento luminotécnico assistido por aprendizado de máquina
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('## 🏷️ Classificação da Via')
    st.markdown('**Selecione a subclasse NBR 5101:**')
    
    opcoes_via = (
        ['M1','M2','M3','M4','M5','M6'] +
        ['C0','C1','C2','C3','C4','C5'] +
        ['P1','P2','P3','P4','P5','P6']
    )
    subclasse = st.selectbox(
        'Subclasse da Via',
        opcoes_via,
        index=3,  # default C3
        format_func=lambda x: f"{x} — {'Via Motorizada' if x.startswith('M') else ('Área de Conflito' if x.startswith('C') else 'Via Pedonal/Ciclovia')}"
    )
    info_nbr = NBR5101.get(subclasse, {})
    
    # Mostra requisitos mínimos da subclasse selecionada
    st.markdown('**Requisitos NBR 5101:**')
    req_html = ""
    for k, v in info_nbr.items():
        if k == 'metricas': continue
        label = TARGETS_MAP.get(k, k)
        unit  = UNITS_MAP.get(k, '')
        req_html += f"<div style='font-size:0.8rem;color:#9ca3af;'>{label}: <b style='color:#FFD700;'>≥ {v} {unit}</b></div>"
    st.markdown(req_html, unsafe_allow_html=True)
    
    st.divider()
    st.markdown('## 🧹 Inteligência de Dados')
    modo_dados = st.radio(
        'Base de Treinamento:',
        ['Padrão (Com outliers)', 'Otimizada (Sem outliers)'],
        index=1,
        help="A opção Otimizada remove valores fisicamente impossíveis das planilhas originais para melhorar a precisão."
    )
    sufixo_modelo = "_limpo" if modo_dados == 'Otimizada (Sem outliers)' else ""
    
    modelos, meta = carregar_modelos(sufixo_modelo)
    clf_cpe, clf_braco = carregar_classificadores(sufixo_modelo)
    num_ok = meta.get('features_numericas', [])
    cat_ok = meta.get('features_categoricas', [])
    feature_w_col = meta.get('feature_w_col', 'Potencia simulada - IP Principal (W)')

    # Métricas cujos modelos são confiáveis o suficiente para verificar conformidade NBR
    # Threshold: R² >= 0.5. Modelos abaixo disso (ex: uo R²=-57.5, ul R²=-0.05) predizem
    # valores fisicamente impossíveis e causariam falsos 'Não Atende'.
    R2_MIN_CONFORMIDADE = 0.5
    metricas_confiaveis = {
        m for m in ['lmed', 'uo', 'ul', 'emed', 'emin', 'w']
        if meta.get(f'modelo_{m}', {}).get('r2', 0) >= R2_MIN_CONFORMIDADE
    }

    if not modelos:
        st.error(f'Modelos ({modo_dados}) não encontrados!')
        st.stop()

    st.divider()
    st.markdown('## ⚙️ Parâmetros (Individual)')
    
    st.markdown('### 🛣️ Geometria da Via')
    faixas         = st.slider('Faixas de Rodagem',            1, 6,    2, step=1)
    largura_via1   = st.slider('Largura Via 1 (m)',            4.0, 20.0, 7.0, step=0.5)
    largura_via2   = st.slider('Largura Via 2 (m)',            0.0, 20.0, 0.0, step=0.5)
    largura_passeio1 = st.slider('Largura Passeio 1 (m)',      0.0, 10.0, 2.0, step=0.5)
    largura_passeio2 = st.slider('Largura Passeio 2 (m)',      0.0, 10.0, 2.0, step=0.5)
    canteiro       = st.slider('Largura Canteiro Central (m)', 0.0, 10.0, 0.0, step=0.5)

    st.markdown('### 🏗️ Estrutura')
    altura_lum     = st.slider('Altura da Luminária (m)',      4.0, 16.0, 9.0,  step=0.5)
    projecao_braco = st.slider('Projeção do Braço (m)',        0.0,  4.0, 1.5,  step=0.25)
    dist_postes    = st.slider('Distância entre Postes (m)',  10.0, 60.0, 35.0, step=1.0)
    dist_poste_via = st.slider('Distância Poste à Via (m)',    0.0,  3.0,  0.5, step=0.25)
    altura_inst    = st.slider('Altura de Instalação (m)',     4.0, 16.0, 10.0, step=0.5)
    st.caption("ℹ️ Informacional — não integra o modelo preditivo atual.")

    st.markdown('### 🔩 Configurações')
    tipo_estrutura = st.selectbox('Tipo de Estrutura', ['Braço', 'Suporte'])
    posteacao      = st.selectbox('Posteação', ['Unilateral', 'Canteiro central', 'Bilateral alternada', 'Bilateral frontal'])
    # Braço atual derivado automaticamente da projeção informada
    braco_novo = projecao_para_braco(projecao_braco)
    st.caption(f"📏 Braço atual identificado: **{braco_novo}** ({projecao_braco:.2f}m) — sugestão de troca gerada pelo modelo ML de geometria histórica.")

    st.markdown('---')
    st.markdown('### 📍 Localização do Projeto')
    endereco_busca = st.text_input(
        'Endereço ou Rua', placeholder='Ex: Rua Joaquim Murtinho, Cuiabá',
        help='Digite e pressione Enter — a localização é feita na hora.')

    # Sem botão "Localizar": digitar e teclar Enter já localiza. A chamada continua
    # sendo uma por endereço, e não uma por rerun, porque o cache é a chave do endereço
    # — mover um slider da página não gasta cota da API do Google.
    @st.cache_data(show_spinner='Localizando endereço…', max_entries=64)
    def _geocodificar(endereco: str):
        location = GoogleV3(api_key=GOOGLE_MAPS_API_KEY).geocode(endereco, timeout=10)
        if location is None:
            return None
        return location.latitude, location.longitude, location.address

    endereco_busca = (endereco_busca or '').strip()
    if endereco_busca:
        if not GOOGLE_MAPS_API_KEY:
            st.warning("Defina `GOOGLE_MAPS_API_KEY` em `st.secrets` (deploy) ou variável de ambiente (local).")
        else:
            try:
                achado = _geocodificar(endereco_busca)
                if achado:
                    st.session_state.lat, st.session_state.lon, st.session_state.address = achado
                    st.success(f"📍 Localizado: {st.session_state.address}")
                else:
                    st.warning("Endereço não encontrado.")
            except Exception as e:
                st.error(f"Erro na geocodificação: {e}")

    st.markdown('### ⚡ Eficientização')
    potencia_atual = st.number_input('Potência Atual (W)', min_value=0.0, value=250.0, step=10.0, help="Potência da luminária instalada atualmente (ex: Sódio 250W, 400W)")

# Métricas ativas baseadas na subclasse NBR 5101
metricas_ativas = NBR5101.get(subclasse, {}).get('metricas', ['emed', 'w'])

# ── Tabs Principais ───────────────────────────────────────────────────────────
tab_individual, tab_lote, tab_dash = st.tabs(['🎯 Simulação Individual', '📂 Simulação em Lote', '📊 Dashboard de Lote'])

# ==============================================================================
# TAB 1: SIMULAÇÃO INDIVIDUAL
# ==============================================================================
with tab_individual:
    # ── Exibição do Mapa (se localizado)
    if 'lat' in st.session_state and 'lon' in st.session_state:
        fig_map = px.scatter_mapbox(
            lat=[st.session_state.lat], 
            lon=[st.session_state.lon],
            zoom=15, 
            height=300
        )
        fig_map.update_layout(
            mapbox_style="carto-darkmatter",
            margin={"r":0,"t":0,"l":0,"b":0},
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption(f"📌 {st.session_state.address}")

    def montar_entrada(fornecedor, dist_override=None):
        dados = {
            'Faixas de Rodagem':        faixas,
            'Largura Via 1':            largura_via1,
            'Largura Via 2':            largura_via2,
            'Largura Passeio 1':        largura_passeio1,
            'largura Passeio 2':        largura_passeio2,
            'largura Canteiro Central': canteiro,
            'altura da luminaria':      altura_lum,
            'projecao do braço':        projecao_braco,
            'projecao do brao':         projecao_braco,
            'distancia entre postes':   dist_override if dist_override is not None else dist_postes,
            'distancia Poste a via':    dist_poste_via,
            'Altura de Instalação':     altura_inst,
            'Altura de Instalao':       altura_inst,
            'Classificação viária':    subclasse,   # subclasse NBR 5101
            'Tipo de estrutura':        tipo_estrutura,
            'posteacao':                posteacao,
            'Braço Novo':               braco_novo,
            'Brao Novo':                braco_novo,
            'Fornecedor':               fornecedor,
        }
        colunas = list(dict.fromkeys(num_ok + cat_ok + [feature_w_col]))
        return pd.DataFrame([{k: dados.get(k, np.nan) for k in colunas}])

    # Roda as predições para todos os fornecedores
    resultados = {m: {} for m in metricas_ativas}
    config_dicts = {}

    for forn in FORNECEDORES:
        X_in = montar_entrada(forn)
        config_dicts[forn] = X_in.iloc[0].to_dict()
        preds_in = prever_metricas_com_dependencia_w(X_in, modelos, metricas_ativas, meta)
        for m in metricas_ativas:
            v = preds_in.get(m, np.array([np.nan]))[0]
            resultados[m][forn] = None if pd.isna(v) else float(v)

    # ── Ajuste de Potência pela Hierarquia NBR ───────────────────────────────
    # Classes M: o modelo lmed tem R²=0.28 e superestima luminância, então
    # o gatilho "pred < req" nunca dispara. Usa-se fator proporcional direto
    # em relação ao M3 (baseline), garantindo M1 > M3 > M6 em potência.
    # Classes C e P: o modelo emed é confiável (R²>0.8) — escalonamento
    # condicional apenas quando pred_ilum < req (mantém comportamento atual).
    fator_m = None
    metric_ref = 'emed'
    req_ilum_ref = None

    if 'w' in metricas_ativas:
        if subclasse.startswith('M'):
            req_m3_base = NBR5101['M3']['lmed']   # 1.0 cd/m²
            req_classe  = info_nbr.get('lmed', req_m3_base)
            fator_m     = req_classe / req_m3_base
            metric_ref  = 'lmed'
            for forn in FORNECEDORES:
                if resultados['w'].get(forn) is not None:
                    resultados['w'][forn] *= fator_m
        else:
            metric_ref   = 'emed'
            req_ilum_ref = info_nbr.get(metric_ref)
            if metric_ref in metricas_ativas and req_ilum_ref:
                for forn in FORNECEDORES:
                    pred_ilum = resultados[metric_ref].get(forn)
                    pred_pot  = resultados['w'].get(forn)
                    if (pred_ilum is not None and pred_ilum > 0
                            and pred_pot is not None and pred_ilum < req_ilum_ref):
                        resultados['w'][forn] = pred_pot * (req_ilum_ref / pred_ilum)

    # Verifica conformidade e gera sugestões estruturais
    sugestoes_por_forn = {}
    for forn in FORNECEDORES:
        falhou = any(
            resultados[m].get(forn) is not None
            and info_nbr.get(m) is not None
            and resultados[m][forn] < info_nbr[m]
            for m in metricas_ativas
            if m != 'w' and m in metricas_confiaveis
        )
        if falhou:
            sugestoes_por_forn[forn] = analisar_melhorias(
                forn, modelos, metricas_ativas, info_nbr, config_dicts[forn], num_ok, cat_ok)

    # ── Sugestão ML de braço (individual) ────────────────────────────────────
    BRACO_ML_CONF_MIN_IND = 0.50  # só exibe sugestão se confiança >= 50%
    braco_ml_sugestao = None
    braco_ml_prob = None
    if clf_braco is not None:
        try:
            row_braco_ind = pd.DataFrame([{
                'Faixas de Rodagem':        faixas,
                'Largura Via 1':            largura_via1,
                'Largura Via 2':            largura_via2,
                'Largura Passeio 1':        largura_passeio1,
                'largura Passeio 2':        largura_passeio2,
                'largura Canteiro Central': canteiro,
                'altura da luminaria':      altura_lum,
                'projecao do braço':        projecao_braco,
                'distancia entre postes':   dist_postes,
                'distancia Poste a via':    dist_poste_via,
                'Classificação viária':     subclasse,
                'Tipo de estrutura':        tipo_estrutura,
                'posteacao':                posteacao,
                'Fornecedor':               'LEDSTAR',
            }])
            _pred = clf_braco.predict(row_braco_ind)[0]
            _probs = clf_braco.predict_proba(row_braco_ind)[0]
            _prob = float(_probs[list(clf_braco.classes_).index(_pred)])
            if _prob >= BRACO_ML_CONF_MIN_IND:
                braco_ml_sugestao = _pred
                braco_ml_prob = _prob
        except Exception:
            pass

    # Exibe os Cards Dinâmicos
    st.markdown('<p class="section-title">📊 Resultados por Fornecedor</p>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, forn in enumerate(FORNECEDORES):
        cor = CORES[forn]
        with cols[i]:
            html_content = f'<div class="forn-card" style="border-top-color:{cor};"><div class="forn-name">{forn}</div>'
            for m in metricas_ativas:
                val = resultados[m].get(forn)
                val_str = f"{val:,.2f}" if val is not None else "—"
                unit = UNITS_MAP[m]
                label = TARGETS_MAP[m]
                req_min = info_nbr.get(m)
                badge_html = ""
                if val is not None and req_min is not None and m != 'w':
                    if m in metricas_confiaveis:
                        # Modelo confiável: badge verde/vermelho de conformidade
                        atende = val >= req_min
                        b_color = '#22c55e' if atende else '#ef4444'
                        b_txt   = '✔ Atende' if atende else '✘ Não Atende'
                        badge_html = f'<span style="font-size:.65rem;padding:2px 6px;border-radius:99px;background:{b_color};color:#fff;margin-left:6px;">{b_txt}</span>'
                    else:
                        # Modelo com baixa acurácia: apenas informa que é estimativa
                        badge_html = '<span style="font-size:.65rem;padding:2px 6px;border-radius:99px;background:#6b7280;color:#fff;margin-left:6px;">Estimado</span>'
                
                html_content += f'<div class="metric-label"><span>{label}</span>{badge_html}</div>'
                html_content += f'<div class="metric-value">{val_str} <span class="metric-unit">{unit}</span></div>'
            
            # Adiciona métricas de eficientização se 'w' foi previsto
            if 'w' in resultados and resultados['w'].get(forn) is not None and potencia_atual > 0:
                pot_prev = resultados['w'].get(forn)
                economia = potencia_atual - pot_prev
                reducao  = (economia / potencia_atual) * 100
                if economia > 0:
                    html_content += f'<div style="margin-top:15px; padding-top:15px; border-top:1px dashed #374151;">'
                    html_content += f'<div style="font-size:0.75rem; color:#22c55e; font-weight:700; text-transform:uppercase; margin-bottom:5px;">🍃 Economia de Energia</div>'
                    html_content += f'<div style="display:flex; justify-content:space-between; align-items:baseline;">'
                    html_content += f'<span style="font-size:1.2rem; font-weight:800; color:#22c55e;">{economia:,.1f}W</span>'
                    html_content += f'<span style="font-size:0.9rem; font-weight:600; color:#4ade80;">-{reducao:.1f}%</span>'
                    html_content += f'</div></div>'

            html_content += "</div>"
            st.markdown(html_content, unsafe_allow_html=True)
    
    # ── Sugestão ML de Braço ─────────────────────────────────────────────────
    if braco_ml_sugestao is not None:
        proj_ml = BRACOS_PROJECAO.get(braco_ml_sugestao, None)
        proj_atual_str = f"{projecao_braco:.2f}m"
        if str(braco_ml_sugestao) != str(braco_novo):
            cor_badge = '#f59e0b'
            icone = '🔄'
            msg = f"O modelo ML recomenda **{braco_ml_sugestao}** ({proj_ml:.1f}m) em vez do atual **{braco_novo}** ({proj_atual_str}) — baseado no padrão histórico desta geometria."
        else:
            cor_badge = '#22c55e'
            icone = '✔'
            msg = f"O modelo ML confirma o braço atual **{braco_novo}** ({proj_atual_str}) como adequado para esta geometria."
        conf_str = f"{braco_ml_prob*100:.0f}%" if braco_ml_prob is not None else "—"
        st.markdown(
            f'<div style="background:#FFFFFF; border:1px solid #E2E8F0; border-left:4px solid {cor_badge}; '
            f'border-radius:12px; padding:14px 18px; margin:12px 0;">'
            f'<span style="font-size:0.8rem; font-weight:700; color:{cor_badge}; text-transform:uppercase; letter-spacing:.05em;">'
            f'{icone} Braço — Recomendação ML</span>'
            f'<div style="font-size:0.9rem; color:#1B2434; margin-top:6px;">{msg}</div>'
            f'<div style="font-size:0.75rem; color:#64748b; margin-top:4px;">Confiança do modelo: {conf_str}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── Detecção Proativa de Ponto Escuro (CPE) ──────────────────────────────
    DIST_CPE_MIN = 45.0
    POT_DESVIO_FATOR = 1.40

    pot_previstas = [resultados['w'].get(f) for f in FORNECEDORES if 'w' in resultados and resultados['w'].get(f) is not None]
    pot_media_prev = np.mean(pot_previstas) if pot_previstas else None

    row_hist = medias_historicas[medias_historicas['Classe_Resumo'] == subclasse.upper()] if not medias_historicas.empty else pd.DataFrame()
    media_hist_w = row_hist['Média Histórica (W)'].iloc[0] if not row_hist.empty else None

    cpe_por_distancia = dist_postes >= DIST_CPE_MIN
    cpe_por_desvio = (
        media_hist_w is not None and pot_media_prev is not None
        and pot_media_prev > media_hist_w * POT_DESVIO_FATOR
    )
    cpe_acionado = cpe_por_distancia and (cpe_por_desvio or media_hist_w is None)

    # Classificador ML de CPE — sinal adicional ao conjunto de regras
    cpe_ml_pred = None
    cpe_ml_prob = None
    if clf_cpe is not None:
        try:
            row_cpe = pd.DataFrame([{
                'Faixas de Rodagem':        faixas,
                'Largura Via 1':            largura_via1,
                'Largura Via 2':            largura_via2,
                'Largura Passeio 1':        largura_passeio1,
                'largura Passeio 2':        largura_passeio2,
                'largura Canteiro Central': canteiro,
                'altura da luminaria':      altura_lum,
                'projecao do braço':        projecao_braco,
                'distancia entre postes':   dist_postes,
                'distancia Poste a via':    dist_poste_via,
                'Classificação viária':     subclasse,
                'Tipo de estrutura':        tipo_estrutura,
                'posteacao':                posteacao,
                'Fornecedor':               'LEDSTAR',
            }])
            cpe_ml_pred = int(clf_cpe.predict(row_cpe)[0])
            cpe_ml_prob = float(clf_cpe.predict_proba(row_cpe)[0][1])
        except Exception:
            pass

    # CPE acionado se as regras OR o modelo ML apontarem risco
    cpe_acionado = cpe_acionado or (cpe_ml_pred == 1)

    if 'w' in metricas_ativas and cpe_acionado:
        dist_cpe = dist_postes / 2
        resultados_cpe = {m: {} for m in metricas_ativas}

        for forn in FORNECEDORES:
            X_cpe = montar_entrada(forn, dist_override=dist_cpe)
            preds_cpe = prever_metricas_com_dependencia_w(X_cpe, modelos, metricas_ativas, meta)
            for m in metricas_ativas:
                v = preds_cpe.get(m, np.array([np.nan]))[0]
                resultados_cpe[m][forn] = None if pd.isna(v) else float(v)

        # Aplica ajuste NBR no cenário CPE — mesma lógica do cenário base
        if subclasse.startswith('M') and fator_m is not None:
            for forn in FORNECEDORES:
                if resultados_cpe['w'].get(forn) is not None:
                    resultados_cpe['w'][forn] *= fator_m
        elif metric_ref in metricas_ativas and req_ilum_ref:
            for forn in FORNECEDORES:
                p_ilum = resultados_cpe[metric_ref].get(forn)
                p_pot  = resultados_cpe['w'].get(forn)
                if (p_ilum is not None and p_ilum > 0
                        and p_pot is not None and p_ilum < req_ilum_ref):
                    resultados_cpe['w'][forn] = p_pot * (req_ilum_ref / p_ilum)

        motivo_parts = []
        if cpe_por_distancia:
            motivo_parts.append(f"distância de **{dist_postes:.0f}m** entre postes (limite recomendado: {DIST_CPE_MIN:.0f}m)")
        if cpe_por_desvio:
            motivo_parts.append(f"potência prevista **{pot_media_prev:.0f}W** acima da média histórica **{media_hist_w:.0f}W** (+{((pot_media_prev/media_hist_w)-1)*100:.0f}%)")
        if media_hist_w is None and not cpe_por_distancia:
            motivo_parts.append("classe sem média histórica disponível para validação de desvio")
        if cpe_ml_pred == 1 and not cpe_por_distancia and not cpe_por_desvio:
            motivo_parts.append(f"modelo ML detectou padrão geométrico de risco (probabilidade {cpe_ml_prob*100:.0f}%)")

        sinal_ml = ""
        if cpe_ml_prob is not None:
            cor_ml = '#22c55e' if cpe_ml_pred == 0 else '#f59e0b'
            sinal_ml = (
                f'\n\n🤖 **Classificador ML:** probabilidade de CPE = **{cpe_ml_prob*100:.0f}%** '
                f'<span style="color:{cor_ml};">{"⚠ Risco" if cpe_ml_pred == 1 else "✔ Baixo risco"}</span>'
            )

        st.markdown('<p class="section-title">⚠️ Correção de Ponto Escuro (CPE)</p>', unsafe_allow_html=True)
        st.warning(
            "**Risco de ponto escuro detectado** — " + " e ".join(motivo_parts) + ".\n\n"
            f"Cenário proposto: inserção de estrutura intermediária reduz a distância de **{dist_postes:.0f}m → {dist_cpe:.0f}m**."
            + sinal_ml
        )

        cpe_cols = st.columns(3)
        for i, forn in enumerate(FORNECEDORES):
            cor = CORES[forn]
            with cpe_cols[i]:
                cpe_html = f'<div class="forn-card" style="border-top-color:#f59e0b;">'
                cpe_html += f'<div class="forn-name" style="color:#f59e0b;">CPE — {forn}</div>'
                cpe_html += f'<div style="font-size:0.75rem;color:#5B6579;margin-bottom:10px;">Distância: {dist_postes:.0f}m → <b style="color:#f59e0b;">{dist_cpe:.0f}m</b></div>'

                for m in metricas_ativas:
                    val_orig = resultados[m].get(forn)
                    val_cpe  = resultados_cpe[m].get(forn)
                    if val_cpe is None:
                        continue
                    delta = val_cpe - val_orig if val_orig is not None else None
                    if delta is not None:
                        delta_color = '#22c55e' if delta >= 0 else '#ef4444'
                        delta_str = f' <span style="color:{delta_color};font-size:0.7rem;">({("+" if delta >= 0 else "")}{delta:.2f})</span>'
                    else:
                        delta_str = ''
                    cpe_html += f'<div class="metric-label"><span>{TARGETS_MAP[m]}</span></div>'
                    cpe_html += f'<div class="metric-value">{val_cpe:,.2f} <span class="metric-unit">{UNITS_MAP[m]}</span>{delta_str}</div>'

                cpe_html += '</div>'
                st.markdown(cpe_html, unsafe_allow_html=True)

    # ── Botão de Exportação PDF
    st.markdown("---")
    addr = st.session_state.get('address', 'Não informado')
    inputs_dict = {
        'Subclasse': subclasse,
        'Posteação': posteacao,
        'Altura (m)': altura_lum,
        'Distância (m)': dist_postes,
        'Largura (m)': largura_via1,
        'Projeção (m)': projecao_braco,
    }
    
    pdf_bytes = gerar_pdf(FORNECEDORES, resultados, info_nbr, inputs_dict, banco_luminarias,
                         sugestoes_por_forn, metricas_confiaveis, addr)
    
    st.download_button(
        label="📥 Baixar Relatório Técnico (PDF)",
        data=pdf_bytes,
        file_name=f"Relatorio_Simulacao_{subclasse}.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True
    )

    # ── Cards de Custo por Fornecedor
    if not banco_luminarias.empty and 'w' in resultados:
        st.markdown('<p class="section-title">💰 Luminária Mais Próxima e Custo Estimado</p>', unsafe_allow_html=True)
        custo_cols = st.columns(3)
        for i, forn in enumerate(FORNECEDORES):
            pot_prev = resultados['w'].get(forn)
            lum_nome, pot_real, custo = buscar_custo(banco_luminarias, forn, pot_prev)
            cor = CORES[forn]
            with custo_cols[i]:
                if custo is not None:
                    delta_str = f"+{pot_real - pot_prev:.0f}W" if pot_real > pot_prev else f"{pot_real - pot_prev:.0f}W"
                    c_html = f'<div style="background:#FFFFFF; border-radius:20px; padding:1.5rem; text-align:center; border:1px solid #E2E8F0; border-top:4px solid {cor}; height:100%; box-shadow: 0 1px 3px rgba(27, 36, 52, 0.06);">'
                    c_html += f'<div style="font-size:.75rem; color:#5B6579; text-transform:uppercase; letter-spacing:.1em; margin-bottom:1rem;">{forn}</div>'
                    c_html += f'<div style="font-size:.85rem; color:#1B2434; font-weight:600; margin-bottom:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="{lum_nome}">{lum_nome}</div>'
                    c_html += f'<div style="font-size:.9rem; color:#5B6579; margin-bottom:1rem;">{pot_real:.0f}W <span style="font-size:.7rem; color:#64748b;">({delta_str})</span></div>'
                    c_html += f'<div style="font-size:2.2rem; font-weight:800; color:#1B2434;">R$ {custo:,.2f}</div></div>'
                    st.markdown(c_html, unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="font-size:.75rem; color:#5B6579; text-transform:uppercase;">{forn}</div><div style="font-size:1.1rem; color:#64748b; margin-top:1rem;">Sem dados no banco</div></div>', unsafe_allow_html=True)

    # ── Seção de Sugestões (Compliance Assistant)
    if sugestoes_por_forn:
        st.markdown('<p class="section-title">💡 Assistente de Conformidade (Sugestões)</p>', unsafe_allow_html=True)
        sug_cols = st.columns(3)
        for i, forn in enumerate(FORNECEDORES):
            with sug_cols[i]:
                sugs = sugestoes_por_forn.get(forn, [])
                if sugs:
                    cor = CORES[forn]
                    st.markdown(f"""
                    <div style="background:rgba(255, 215, 0, 0.05); border-left:4px solid {cor}; border-radius:10px; padding:15px; height:100%;">
                        <div style="font-size:0.8rem; font-weight:700; color:{cor}; margin-bottom:8px;">{forn} - RECOMENDAÇÕES:</div>
                        {"".join([f'<div style="font-size:0.85rem; color:#1B2434; margin-bottom:5px;">{s}</div>' for s in sugs])}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="padding:15px; color:#22c55e; font-size:0.85rem; font-weight:600;">✔ Configuração atende os requisitos para {forn}.</div>', unsafe_allow_html=True)

    # Gráficos
    st.markdown('<p class="section-title">📈 Comparativo Gráfico</p>', unsafe_allow_html=True)
    tabs_graf = st.tabs([TARGETS_MAP[m] for m in metricas_ativas])
    
    def bar_chart(valores, unidade, titulo):
        fig = go.Figure()
        for forn, val in valores.items():
            if val is not None:
                fig.add_trace(go.Bar(
                    x=[forn], y=[val], name=forn,
                    marker_color=CORES[forn],
                    text=[f'{val:,.2f} {unidade}'], textposition='outside',
                    textfont=dict(size=13, color='white'),
                ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#d1d5db', family='Inter'),
            showlegend=False, bargap=0.35, height=360,
            margin=dict(t=30, b=10),
            yaxis=dict(title=titulo, gridcolor='#E2E8F0', zerolinecolor='#CBD5E1'),
            xaxis=dict(gridcolor='#E2E8F0'),
        )
        return fig

    for idx, m in enumerate(metricas_ativas):
        with tabs_graf[idx]:
            st.plotly_chart(bar_chart(resultados[m], UNITS_MAP[m], TARGETS_MAP[m]), use_container_width=True)

# ==============================================================================
# TAB 2: SIMULAÇÃO EM LOTE
# ==============================================================================
with tab_lote:
    st.markdown('### 📥 1. Baixe a Planilha Padrão')
    st.markdown('Preencha as características geométricas de cada instalação.')
    
    # Colunas necessárias para o modelo + rastreabilidade (em ordem do template)
    cols_template = [
        'ID', 'Padrão', 'Logradouro', 'latitude', 'longitude',
        'Classificação viária', 'Tipo de lâmpada', 'Potencia da lâmpada',
        'Faixas de Rodagem', 'Largura Passeio 1', 'Largura Via 1', 'Largura Via 2',
        'largura Passeio 2', 'largura Canteiro Central', 'posteacao', 'Tipo de estrutura',
        'distancia entre postes', 'altura da luminaria', 'qtd de Lampadas IP Princ',
        'distancia Poste a via', 'projecao do braço', 'Altura de Instalação',
        'Braço Antigo',                    # referência — tipo de braço atual instalado
        'Quantidade de pontos inspecionados IP Veic',
        'Quantidade de pontos inspecionados IP Sec',
    ]
    
    df_template = pd.DataFrame(columns=cols_template)
    # Linha de exemplo
    exemplo = {
        'ID': 1,
        'Padrão': 'V4',
        'Logradouro': 'Rua Exemplo, 100',
        'latitude': -23.5505,
        'longitude': -46.6333,
        'Classificação viária': 'M3',
        'Tipo de lâmpada': 'Sódio',
        'Potencia da lâmpada': 250.0,
        'Faixas de Rodagem': 2,
        'Largura Passeio 1': 2.0,
        'Largura Via 1': 7.0,
        'Largura Via 2': 0.0,
        'largura Passeio 2': 2.0,
        'largura Canteiro Central': 0.0,
        'posteacao': 'Unilateral',
        'Tipo de estrutura': 'Braço',
        'distancia entre postes': 35.0,
        'altura da luminaria': 10.0,
        'qtd de Lampadas IP Princ': 1,
        'distancia Poste a via': 0.5,
        'projecao do braço': 1.5,
        'Altura de Instalação': 10.0,
        'Braço Antigo': 'Médio II',
        'Quantidade de pontos inspecionados IP Veic': 2,
        'Quantidade de pontos inspecionados IP Sec': 0,
    }
    df_template = pd.concat([df_template, pd.DataFrame([exemplo])], ignore_index=True)
    
    buffer_template = io.BytesIO()
    with pd.ExcelWriter(buffer_template, engine='openpyxl') as writer:
        df_template.to_excel(writer, index=False)
    
    st.download_button(
        label='⬇️ Baixar Planilha Padrão (.xlsx)',
        data=buffer_template.getvalue(),
        file_name='template_simulacao.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    
    st.markdown('---')
    st.markdown('### 📤 2. Envie a Planilha Preenchida')
    arquivo_up = st.file_uploader('Selecione o arquivo modificado', type=['xlsx', 'csv'])
    
    if arquivo_up is not None:
        try:
            df_entrada = pd.read_csv(arquivo_up) if arquivo_up.name.endswith('.csv') else pd.read_excel(arquivo_up)
            
            # Tratamento robusto de decimais: aplica vírgula→ponto célula a célula antes de tentar converter
            def normalizar_coluna(serie):
                serie_norm = serie.apply(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
                convertida = pd.to_numeric(serie_norm, errors='coerce')
                # Se pelo menos 50% dos valores converteram bem, usa a versão numérica
                # Caso contrário (coluna de texto como Fornecedor), mantém original
                if convertida.notna().mean() >= 0.5:
                    return convertida
                return serie
            
            for col in df_entrada.columns:
                df_entrada[col] = normalizar_coluna(df_entrada[col])
            
            st.success(f'Arquivo lido com sucesso! ({len(df_entrada)} linhas)')
            
            with st.spinner('Realizando previsões...'):
                # Prepara DF de saída (preservando o ID se existir de forma case-insensitive)
                df_saida = df_entrada.copy()
                cols_upper = [c.upper() for c in df_saida.columns]
                if 'ID' not in cols_upper:
                    df_saida.insert(0, 'ID', range(1, len(df_saida) + 1))
                
                # Mapeia as colunas do arquivo para os nomes internos
                df_pipeline = df_entrada.rename(columns=MAPEAMENTO_COLS)
                
                # Se a planilha não tiver Classificacao, usamos a do sidebar
                # Procura por qualquer variação de 'Classificacao'
                col_classe = next((c for c in df_pipeline.columns if 'Classificação viária' == c), None)
                tem_classe = col_classe is not None

                metricas_lote = ['lmed', 'uo', 'ul', 'emed', 'emin', 'w']

                # ── Sugestão de Braço via Classificador ML ───────────────────────
                # Opção A: clf_braco aprende o padrão histórico de qual braço é
                # usado para cada geometria, independentemente da conformidade NBR.
                # Isso resolve o bloqueio das classes M (lmed R²<0.5) e o efeito
                # quase nulo da projeção sobre emed nas classes C/P.
                # Fallback: conformidade NBR (preservado para quando clf_braco=None).

                # 1. Identifica braço atual por projeção
                proj_series = pd.to_numeric(
                    df_pipeline.get('projecao do braço', pd.Series([1.8]*len(df_pipeline))),
                    errors='coerce'
                ).fillna(1.8).values
                bracos_atuais = [projecao_para_braco(p) for p in proj_series]

                classes_s = (
                    df_pipeline['Classificação viária'].values
                    if 'Classificação viária' in df_pipeline.columns
                    else [''] * len(df_pipeline)
                )

                bracos_pred = []

                BRACO_ML_CONF_MIN = 0.50  # só sugere troca se a confiança do modelo for >= 50%

                if clf_braco is not None:
                    # 2a. Predição vetorizada pelo classificador ML de braço
                    rows_ml = []
                    for i in range(len(df_pipeline)):
                        r = df_pipeline.iloc[i]
                        rows_ml.append({
                            'Faixas de Rodagem':        r.get('Faixas de Rodagem', 2),
                            'Largura Via 1':            r.get('Largura Via 1', 7),
                            'Largura Via 2':            r.get('Largura Via 2', 0),
                            'Largura Passeio 1':        r.get('Largura Passeio 1', 2),
                            'largura Passeio 2':        r.get('largura Passeio 2', 2),
                            'largura Canteiro Central': r.get('largura Canteiro Central', 0),
                            'altura da luminaria':      r.get('altura da luminaria', 9),
                            'projecao do braço':        r.get('projecao do braço', 1.5),
                            'distancia entre postes':   r.get('distancia entre postes', 35),
                            'distancia Poste a via':    r.get('distancia Poste a via', 0.5),
                            'Classificação viária':     r.get('Classificação viária', 'M3'),
                            'Tipo de estrutura':        r.get('Tipo de estrutura', 'Braço'),
                            'posteacao':                r.get('posteacao', 'Unilateral'),
                            'Fornecedor':               'LEDSTAR',
                        })
                    df_ml_input = pd.DataFrame(rows_ml)
                    try:
                        ml_preds = clf_braco.predict(df_ml_input)
                        ml_probs = clf_braco.predict_proba(df_ml_input).max(axis=1)
                        for ml_pred, ml_prob, atual in zip(ml_preds, ml_probs, bracos_atuais):
                            if str(ml_pred) != str(atual) and ml_prob >= BRACO_ML_CONF_MIN:
                                bracos_pred.append(ml_pred)
                            else:
                                bracos_pred.append(None)
                    except Exception:
                        bracos_pred = [None] * len(df_pipeline)

                else:
                    # 2b. Fallback: conformidade NBR (funciona apenas para C/P com emed confiável)
                    arm_preds_lote = {}
                    for arm_name, arm_proj in BRACOS_ORDENADOS:
                        df_arm = df_pipeline.copy()
                        df_arm['Braço Novo'] = arm_name
                        df_arm['projecao do braço'] = arm_proj
                        arm_preds_lote[arm_name] = {}
                        for forn in FORNECEDORES:
                            df_ra = df_arm.copy()
                            df_ra['Fornecedor'] = forn
                            for col in list(dict.fromkeys(num_ok + cat_ok + [feature_w_col])):
                                if col not in df_ra.columns:
                                    df_ra[col] = np.nan
                            arm_preds_lote[arm_name][forn] = prever_metricas_com_dependencia_w(
                                df_ra, modelos, metricas_lote, meta
                            )

                    def arm_atende_linha(arm_name, i, classe_i):
                        info_i = NBR5101.get(str(classe_i).upper(), {})
                        for m in info_i.get('metricas', []):
                            req = info_i.get(m)
                            if m == 'w' or req is None or m not in metricas_confiaveis or m not in modelos:
                                continue
                            for forn in FORNECEDORES:
                                arr = arm_preds_lote.get(arm_name, {}).get(forn, {}).get(m, np.array([]))
                                val = float(arr[i]) if i < len(arr) else np.nan
                                if pd.notna(val) and val < req:
                                    return False
                        return True

                    for i in range(len(df_pipeline)):
                        cl_i   = classes_s[i]
                        atual  = bracos_atuais[i]
                        info_i = NBR5101.get(str(cl_i).upper(), {})
                        tem_req = any(
                            info_i.get(m) is not None and m != 'w'
                            and m in metricas_confiaveis and m in modelos
                            for m in info_i.get('metricas', [])
                        )
                        if not tem_req or arm_atende_linha(atual, i, cl_i):
                            bracos_pred.append(None)
                        else:
                            rec = None
                            for arm_name, _ in BRACOS_ORDENADOS:
                                if arm_name == atual:
                                    continue
                                if arm_atende_linha(arm_name, i, cl_i):
                                    rec = arm_name
                                    break
                            bracos_pred.append(rec)

                # 3. Injeta braço final na pipeline (atual ou recomendado)
                bracos_final = [rec if rec else atual for rec, atual in zip(bracos_pred, bracos_atuais)]
                df_pipeline['Braço Novo'] = bracos_final
                df_saida['Braço Atual']         = bracos_atuais
                df_saida['Sugestão Braço Novo'] = [rec or '' for rec in bracos_pred]

                for forn in FORNECEDORES:
                    df_run = df_pipeline.copy()
                    df_run['Fornecedor'] = forn
                    for col in list(dict.fromkeys(num_ok + cat_ok + [feature_w_col])):
                        if col not in df_run.columns:
                            df_run[col] = np.nan

                    preds_all = prever_metricas_com_dependencia_w(df_run, modelos, metricas_lote, meta)

                    # Filtro dinâmico por linha baseada na classificação
                    classes = None
                    if tem_classe:
                        classes = df_pipeline['Classificação viária'].fillna('M').astype(str).str.upper().str[0]

                    for m in metricas_lote:
                        if m not in preds_all:
                            continue
                        preds = preds_all[m]
                        if tem_classe:
                            mask_valid = [False] * len(preds)
                            for i, c in enumerate(classes):
                                if c == 'M' and m in ['lmed', 'uo', 'ul', 'w']: mask_valid[i] = True
                                elif c == 'C' and m in ['emed', 'uo', 'w']: mask_valid[i] = True
                                elif c == 'P' and m in ['emed', 'emin', 'w']: mask_valid[i] = True
                            preds = [p if valid else np.nan for p, valid in zip(preds, mask_valid)]

                        df_saida[f'{TARGETS_MAP[m]} - {forn}'] = [max(p, 0) if pd.notna(p) else p for p in preds]

                # ── Pós-processamento: Ajuste de Potência pela Hierarquia NBR ──────────────
                # M: fator proporcional direto (lmed R²=0.28 não é confiável).
                # C/P: escalonamento condicional por iluminância (emed R²>0.8).
                if tem_classe:
                    classes_serie = df_pipeline['Classificação viária'].fillna('').astype(str).str.upper()
                    req_m3_lote = NBR5101['M3']['lmed']
                    for forn in FORNECEDORES:
                        pot_col = f'{TARGETS_MAP["w"]} - {forn}'
                        if pot_col not in df_saida.columns:
                            continue
                        for idx_loc, classe in zip(df_saida.index, classes_serie):
                            info_v = NBR5101.get(classe, {})
                            if not info_v:
                                continue
                            pred_pot = df_saida.loc[idx_loc, pot_col]
                            if pd.isna(pred_pot):
                                continue
                            if classe.startswith('M'):
                                fator_lote = info_v.get('lmed', req_m3_lote) / req_m3_lote
                                df_saida.loc[idx_loc, pot_col] = pred_pot * fator_lote
                            else:
                                ilum_col = f'{TARGETS_MAP["emed"]} - {forn}'
                                req = info_v.get('emed')
                                if req is None or ilum_col not in df_saida.columns:
                                    continue
                                pred_ilum = df_saida.loc[idx_loc, ilum_col]
                                if (pd.notna(pred_ilum) and pred_ilum > 0 and pred_ilum < req):
                                    df_saida.loc[idx_loc, pot_col] = pred_pot * (req / pred_ilum)

                # Cálculo de Eficientização e Conformidade por linha no lote
                # Detecta a coluna de potência atual de forma robusta (suporta template novo e arquivos legados)
                col_potencia_atual = next(
                    (c for c in df_entrada.columns if c in ['Potencia da lâmpada', 'Potencia Atual (W)', 'Potência Atual (W)']),
                    None
                )
                if col_potencia_atual:
                    for forn in FORNECEDORES:
                        pot_prev_col = f'{TARGETS_MAP["w"]} - {forn}'
                        if pot_prev_col in df_saida.columns:
                            df_saida[f'Economia (W) - {forn}'] = df_saida[col_potencia_atual] - df_saida[pot_prev_col]
                            df_saida[f'Reducao (%) - {forn}'] = (df_saida[f'Economia (W) - {forn}'] / df_saida[col_potencia_atual]) * 100
                
                # Validação NBR 5101 por linha (Status)
                if tem_classe:
                    for forn in FORNECEDORES:
                        status_list = []
                        for idx, row in df_saida.iterrows():
                            # Busca classificação de forma flexível
                            classe_val = row.get('Classificação viária', 'M3')
                            classe = str(classe_val).upper()
                            info_v = NBR5101.get(classe, {})
                            atende_linha = True
                            for m in info_v.get('metricas', []):
                                if m == 'w': continue
                                # Ignora métricas cujo modelo não é confiável (R² < 0.5)
                                if m not in metricas_confiaveis: continue
                                col_res = f'{TARGETS_MAP[m]} - {forn}'
                                if col_res in df_saida.columns:
                                    val = row[col_res]
                                    req = info_v.get(m)
                                    if pd.notna(val) and req is not None and val < req:
                                        atende_linha = False
                                        break
                            status_list.append('✔ Atende' if atende_linha else '✘ Não Atende')
                        df_saida[f'Status NBR - {forn}'] = status_list

                # Sugestão Braço Novo já calculada e injetada antes do loop de regressão

                # ── CPE: Detecção, Cálculo e Preenchimento do Template ─────────────────
                # Regra revisada:
                # distância >= 45m + desvio de potência histórica relevante.
                # Sem média histórica da classe: fallback por distância.
                # Para cada linha flagada, roda predições com distância/2 e aplica
                # o mesmo ajuste NBR proporcional, garantindo consistência total.
                col_dist_lote = next(
                    (c for c in df_saida.columns if 'distancia entre' in c.lower()), None
                )
                if col_dist_lote:
                    cpe_sim, cpe_qtd_veic, cpe_obs_list = [], [], []
                    DIST_CPE_MIN_LOTE = 45.0
                    POT_DESVIO_FATOR_LOTE = 1.40
                    for _, row_c in df_saida.iterrows():
                        dist_v = pd.to_numeric(row_c.get(col_dist_lote, 0), errors='coerce') or 0
                        classe_v = str(row_c.get('Classificação viária', '')).upper()
                        row_hist_v = medias_historicas[medias_historicas['Classe_Resumo'] == classe_v] if not medias_historicas.empty else pd.DataFrame()
                        media_hist_v = row_hist_v['Média Histórica (W)'].iloc[0] if not row_hist_v.empty else None
                        pot_vals = [
                            row_c.get(f'{TARGETS_MAP["w"]} - {forn}')
                            for forn in FORNECEDORES
                            if pd.notna(row_c.get(f'{TARGETS_MAP["w"]} - {forn}'))
                        ]
                        pot_prev_row = np.mean(pot_vals) if pot_vals else None
                        desvio_ok = (
                            media_hist_v is not None and pot_prev_row is not None
                            and pot_prev_row > media_hist_v * POT_DESVIO_FATOR_LOTE
                        )
                        aciona_regra = dist_v >= DIST_CPE_MIN_LOTE and (desvio_ok or media_hist_v is None)

                        # Classificador ML como sinal adicional por linha
                        cpe_ml_linha = False
                        if clf_cpe is not None:
                            try:
                                row_ml = pd.DataFrame([{
                                    'Faixas de Rodagem':        pd.to_numeric(row_c.get('Faixas de Rodagem', 2), errors='coerce') or 2,
                                    'Largura Via 1':            pd.to_numeric(row_c.get('Largura Via 1', 7), errors='coerce') or 7,
                                    'Largura Via 2':            pd.to_numeric(row_c.get('Largura Via 2', 0), errors='coerce') or 0,
                                    'Largura Passeio 1':        pd.to_numeric(row_c.get('Largura Passeio 1', 2), errors='coerce') or 2,
                                    'largura Passeio 2':        pd.to_numeric(row_c.get('largura Passeio 2', 2), errors='coerce') or 2,
                                    'largura Canteiro Central': pd.to_numeric(row_c.get('largura Canteiro Central', 0), errors='coerce') or 0,
                                    'altura da luminaria':      pd.to_numeric(row_c.get('altura da luminaria', 9), errors='coerce') or 9,
                                    'projecao do braço':        pd.to_numeric(row_c.get('projecao do braço', 1.5), errors='coerce') or 1.5,
                                    'distancia entre postes':   dist_v,
                                    'distancia Poste a via':    pd.to_numeric(row_c.get('distancia Poste a via', 0.5), errors='coerce') or 0.5,
                                    'Classificação viária':     classe_v,
                                    'Tipo de estrutura':        row_c.get('Tipo de estrutura', 'Braço'),
                                    'posteacao':                row_c.get('posteacao', 'Unilateral'),
                                    'Fornecedor':               'LEDSTAR',
                                }])
                                cpe_ml_linha = int(clf_cpe.predict(row_ml)[0]) == 1
                            except Exception:
                                pass

                        aciona_cpe = aciona_regra or cpe_ml_linha
                        if aciona_cpe:
                            cpe_sim.append('Sim')
                            cpe_qtd_veic.append(1)
                            obs = f"Redução {dist_v:.0f}m → {dist_v/2:.0f}m"
                            if desvio_ok and media_hist_v is not None and pot_prev_row is not None:
                                obs += f" | Pot {pot_prev_row:.0f}W > hist {media_hist_v:.0f}W"
                            if cpe_ml_linha and not aciona_regra:
                                obs += " | Detectado por ML"
                            cpe_obs_list.append(obs)
                        else:
                            cpe_sim.append('Não')
                            cpe_qtd_veic.append(0)
                            cpe_obs_list.append('')
                    df_saida['Correção de Ponto Escuro (CPE)'] = cpe_sim
                    df_saida['Quantidade de pontos adicionados para via de veículo'] = cpe_qtd_veic
                    df_saida['Observação CPE  (Reduçao entre postes e/ou Tipo de Posteação)'] = cpe_obs_list

                    # Predições CPE para linhas flagadas
                    cpe_indices = df_saida[df_saida['Correção de Ponto Escuro (CPE)'] == 'Sim'].index
                    if len(cpe_indices) > 0:
                        df_pipeline_cpe = df_pipeline.loc[cpe_indices].copy()
                        if 'distancia entre postes' in df_pipeline_cpe.columns:
                            df_pipeline_cpe['distancia entre postes'] = (
                                df_pipeline_cpe['distancia entre postes'] / 2
                            )

                        for forn in FORNECEDORES:
                            df_run_cpe = df_pipeline_cpe.copy()
                            df_run_cpe['Fornecedor'] = forn
                            for col in list(dict.fromkeys(num_ok + cat_ok + [feature_w_col])):
                                if col not in df_run_cpe.columns:
                                    df_run_cpe[col] = np.nan

                            metricas_lote = ['lmed', 'uo', 'ul', 'emed', 'emin', 'w']
                            preds_c_all = prever_metricas_com_dependencia_w(df_run_cpe, modelos, metricas_lote, meta)
                            for m in metricas_lote:
                                if m in preds_c_all:
                                    df_saida.loc[cpe_indices, f'CPE {TARGETS_MAP[m]} - {forn}'] = [
                                        max(p, 0) if pd.notna(p) else np.nan for p in preds_c_all[m]
                                    ]

                        # Ajuste NBR nas predições CPE — idêntico ao ajuste do cenário base
                        if tem_classe and 'Classificação viária' in df_pipeline_cpe.columns:
                            req_m3_cpe = NBR5101['M3']['lmed']
                            for forn in FORNECEDORES:
                                cpe_pot_col = f'CPE {TARGETS_MAP["w"]} - {forn}'
                                if cpe_pot_col not in df_saida.columns:
                                    continue
                                for idx_loc in cpe_indices:
                                    cl = str(df_pipeline.loc[idx_loc, 'Classificação viária']).upper()
                                    info_cpe = NBR5101.get(cl, {})
                                    if not info_cpe:
                                        continue
                                    p_pot = df_saida.loc[idx_loc, cpe_pot_col]
                                    if pd.isna(p_pot):
                                        continue
                                    if cl.startswith('M'):
                                        fator_cpe = info_cpe.get('lmed', req_m3_cpe) / req_m3_cpe
                                        df_saida.loc[idx_loc, cpe_pot_col] = p_pot * fator_cpe
                                    else:
                                        cpe_ilum_col = f'CPE {TARGETS_MAP["emed"]} - {forn}'
                                        req_cpe = info_cpe.get('emed')
                                        if req_cpe is None or cpe_ilum_col not in df_saida.columns:
                                            continue
                                        p_ilum = df_saida.loc[idx_loc, cpe_ilum_col]
                                        if (pd.notna(p_ilum) and p_ilum > 0 and p_ilum < req_cpe):
                                            df_saida.loc[idx_loc, cpe_pot_col] = p_pot * (req_cpe / p_ilum)

                # Busca de Custo e Modelo no Banco de Dados para o Lote
                if not banco_luminarias.empty:
                    for forn in FORNECEDORES:
                        pot_prev_col = f'{TARGETS_MAP["w"]} - {forn}'
                        if pot_prev_col in df_saida.columns:
                            modelos_sug = []
                            pots_reais = []
                            custos_unit = []
                            for p_prev in df_saida[pot_prev_col]:
                                lum_n, p_real, v_unit = buscar_custo(banco_luminarias, forn, p_prev)
                                modelos_sug.append(lum_n)
                                pots_reais.append(p_real)
                                custos_unit.append(v_unit)
                            
                            df_saida[f'Modelo Sugerido - {forn}'] = modelos_sug
                            df_saida[f'Potencia Real (W) - {forn}'] = pots_reais
                            df_saida[f'Custo Unitario (R$) - {forn}'] = custos_unit
                
                # Formata para o template de exportação (Columns D-DM)
                df_export = formatar_resultado_template(df_saida)
                st.session_state.df_lote = df_saida
                st.session_state.df_export = df_export

                st.markdown('### ✨ Resultados (Preview)')
                # Exibe preview sem as colunas CPE intermediárias (ficaria muito largo)
                cols_preview = [c for c in df_saida.columns if not c.startswith('CPE ')]
                st.dataframe(df_saida[cols_preview].head(10))

                # ── Seção CPE visual — mesma lógica do modo individual ───────────────
                if 'Correção de Ponto Escuro (CPE)' in df_saida.columns:
                    df_cpe_vis = df_saida[df_saida['Correção de Ponto Escuro (CPE)'] == 'Sim']
                    if not df_cpe_vis.empty:
                        st.markdown('<p class="section-title">⚠️ Correção de Ponto Escuro (CPE)</p>', unsafe_allow_html=True)
                        st.warning(
                            f"**{len(df_cpe_vis)} instalação(ões)** com risco de ponto escuro detectado(s) "
                            f"por regra de distância/potência ou pelo classificador ML 🤖. "
                            "Recomenda-se inserção de estrutura intermediária. "
                            "Veja a coluna **Observação CPE** para o motivo por linha."
                        )
                        cpe_rows_display = []
                        for _, row in df_cpe_vis.iterrows():
                            dist_orig = pd.to_numeric(row.get(col_dist_lote, 0), errors='coerce') or 0
                            entry = {
                                'ID': row.get('ID', ''),
                                'Logradouro': str(row.get('Logradouro', ''))[:35],
                                'Classe': row.get('Classificação viária', ''),
                                'Dist. Atual (m)': f"{dist_orig:.0f}",
                                'Dist. CPE (m)': f"{dist_orig/2:.0f}",
                            }
                            for forn in FORNECEDORES:
                                pot_orig = row.get(f'{TARGETS_MAP["w"]} - {forn}')
                                pot_cpe  = row.get(f'CPE {TARGETS_MAP["w"]} - {forn}')
                                entry[f'{forn[:8]} Orig (W)'] = f"{pot_orig:.0f}" if pd.notna(pot_orig) else '-'
                                entry[f'{forn[:8]} CPE (W)']  = f"{pot_cpe:.0f}"  if pd.notna(pot_cpe)  else '-'
                            cpe_rows_display.append(entry)
                        st.dataframe(
                            pd.DataFrame(cpe_rows_display),
                            use_container_width=True,
                            hide_index=True
                        )

                # Gera o resultado no formato da tabela dinâmica (linha por ponto × fornecedor)
                df_tabela = formatar_tabela_resultado(df_saida)
                df_tabela_din = formatar_tabela_dinamica(df_saida)
                df_preview_export = df_saida[cols_preview].copy()

                buffer_resultado = io.BytesIO()
                with pd.ExcelWriter(buffer_resultado, engine='openpyxl') as writer:
                    df_tabela.to_excel(writer, sheet_name='Resultado', index=False)
                    df_tabela_din.to_excel(writer, sheet_name='Tabela Dinâmica', index=False)
                    df_preview_export.to_excel(writer, sheet_name='Dados Completos', index=False)
                    df_export.to_excel(writer, sheet_name='Template', index=False)
                    writer.sheets['Template'].sheet_state = 'hidden'
                st.download_button(
                    '✅ Baixar Resultados (.xlsx)',
                    data=buffer_resultado.getvalue(),
                    file_name='resultados_simulacao.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    type='primary',
                )

        except Exception as e:
            st.error(f'Erro ao processar: {str(e)}')

# ==============================================================================
# TAB 3: DASHBOARD DE LOTE
# ==============================================================================
with tab_dash:
    if st.session_state.df_lote is None:
        st.info('👉 Realize uma simulação na aba "Simulação em Lote" para habilitar o Dashboard.')
    else:
        df = st.session_state.df_lote
        st.markdown('<p class="hero-title" style="font-size:2rem;">Dashboard Executivo</p>', unsafe_allow_html=True)
        
        # Filtro de fornecedor principal para o Dash
        forn_dash = st.selectbox('Selecione o Fornecedor para análise detalhada:', FORNECEDORES)
        
        st.divider()
        
        # ── KPIs Superiores
        m1, m2, m3, m4 = st.columns(4)
        
        col_custo = f'Custo Unitario (R$) - {forn_dash}'
        col_eco   = f'Economia (W) - {forn_dash}'
        col_red   = f'Reducao (%) - {forn_dash}'
        col_status = f'Status NBR - {forn_dash}'
        
        total_capex = df[col_custo].sum() if col_custo in df.columns else 0
        total_eco_kw = (df[col_eco].sum() / 1000) if col_eco in df.columns else 0
        media_red = df[col_red].mean() if col_red in df.columns else 0
        
        # % que atende
        if col_status in df.columns:
            atende_count = (df[col_status] == '✔ Atende').sum()
            perc_atende = (atende_count / len(df)) * 100
        else:
            perc_atende = 0

        with m1:
            st.metric("CAPEX Total Est.", f"R$ {total_capex:,.2f}")
        with m2:
            st.metric("Economia Total", f"{total_eco_kw:,.1f} kW")
        with m3:
            st.metric("Eficiência Média", f"{media_red:.1f}%")
        with m4:
            st.metric("Conformidade NBR", f"{perc_atende:.1f}%")

        st.markdown('---')
        
        # ── Gráficos
        g1, g2 = st.columns(2)
        
        with g1:
            st.markdown('### 💰 CAPEX Total por Fornecedor')
            custos_forn = {}
            for f in FORNECEDORES:
                c_col = f'Custo Unitario (R$) - {f}'
                if c_col in df.columns:
                    custos_forn[f] = df[c_col].sum()
            
            fig_custo = go.Figure(go.Bar(
                x=list(custos_forn.keys()),
                y=list(custos_forn.values()),
                marker_color=[CORES[f] for f in custos_forn.keys()],
                text=[f'R$ {v:,.0f}' for v in custos_forn.values()],
                textposition='auto'
            ))
            fig_custo.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig_custo, use_container_width=True)

        with g2:
            st.markdown('### ✅ Conformidade por Fornecedor')
            conf_forn = {}
            for f in FORNECEDORES:
                s_col = f'Status NBR - {f}'
                if s_col in df.columns:
                    conf_forn[f] = (df[s_col] == '✔ Atende').mean() * 100
            
            fig_conf = go.Figure(go.Bar(
                x=list(conf_forn.keys()),
                y=list(conf_forn.values()),
                marker_color=[CORES[f] for f in conf_forn.keys()],
                text=[f'{v:.1f}%' for v in conf_forn.values()],
                textposition='auto'
            ))
            fig_conf.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), yaxis=dict(range=[0, 105]))
            st.plotly_chart(fig_conf, use_container_width=True)

        st.markdown('---')
        st.markdown(f'### 🛣️ Análise por Classificação de Via ({forn_dash})')
        
        col_classe_dash = next((c for c in df.columns if c in ['Classificação viária', 'Classificacao (M/C/P)']), None)
        if col_classe_dash:
            df['Classe_Resumo'] = df[col_classe_dash].fillna('N/A').astype(str).str.upper()
            
            # Agrupa dados
            col_pot_real = f'Potencia Real (W) - {forn_dash}'
            
            if col_pot_real in df.columns:
                # Prepara o dicionário de agregação de forma dinâmica
                agg_dict = {col_pot_real: ['mean', 'count']}
                if col_red in df.columns:
                    agg_dict[col_red] = 'mean'
                if col_custo in df.columns:
                    agg_dict[col_custo] = 'sum'
                
                analise_via = df.groupby('Classe_Resumo').agg(agg_dict).reset_index()
                
                # Ajusta nomes das colunas após o agg
                novas_cols = ['Classe', 'Potência Média (W)', 'Quantidade']
                if col_red in df.columns:
                    novas_cols.append('Economia Média (%)')
                if col_custo in df.columns:
                    novas_cols.append('CAPEX Total (R$)')
                analise_via.columns = novas_cols
                
                # Mescla a média histórica de treinamento
                medias_historicas = carregar_media_historica()
                if not medias_historicas.empty:
                    analise_via = analise_via.merge(
                        medias_historicas.rename(columns={'Classe_Resumo': 'Classe'}),
                        on='Classe', how='left'
                    )
                    # Reordena para ficar 'Média Histórica (W)' logo após 'Potência Média (W)'
                    cols = list(analise_via.columns)
                    if 'Média Histórica (W)' in cols:
                        cols.remove('Média Histórica (W)')
                        idx_pot = cols.index('Potência Média (W)')
                        cols.insert(idx_pot + 1, 'Média Histórica (W)')
                        analise_via = analise_via[cols]
                
                c1, c2, c3 = st.columns([1.5, 1.5, 1])
                with c1:
                    fig_via = go.Figure()
                    fig_via.add_trace(go.Bar(
                        x=analise_via['Classe'], y=analise_via['Potência Média (W)'],
                        name='Potência Média (W)', marker_color=CORES[forn_dash]
                    ))
                    fig_via.update_layout(title="Potência Média por Tipo de Via", height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                    st.plotly_chart(fig_via, use_container_width=True)
                
                with c2:
                    if 'CAPEX Total (R$)' in analise_via.columns:
                        fig_capex = go.Figure()
                        fig_capex.add_trace(go.Bar(
                            x=analise_via['Classe'], y=analise_via['CAPEX Total (R$)'],
                            name='CAPEX Total (R$)', marker_color='#00A9E0'
                        ))
                        fig_capex.update_layout(title="CAPEX Total por Tipo de Via", height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                        st.plotly_chart(fig_capex, use_container_width=True)
                
                with c3:
                    fig_pie = go.Figure(go.Pie(
                        labels=analise_via['Classe'], values=analise_via['Quantidade'],
                        hole=.4, marker=dict(colors=['#00A9E0', '#1B3664', '#64748b', '#334155'])
                    ))
                    fig_pie.update_layout(title="Mix de Vias", height=350, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                    st.plotly_chart(fig_pie, use_container_width=True)

                formato = {'Potência Média (W)': '{:.1f} W'}
                if 'Média Histórica (W)' in analise_via.columns:
                    formato['Média Histórica (W)'] = '{:.1f} W'
                if 'Economia Média (%)' in analise_via.columns:
                    formato['Economia Média (%)'] = '{:.1f} %'
                if 'CAPEX Total (R$)' in analise_via.columns:
                    formato['CAPEX Total (R$)'] = 'R$ {:,.2f}'
                
                st.dataframe(analise_via.style.format(formato, na_rep='-'), use_container_width=True, hide_index=True)
            else:
                st.warning(f'Dados de potência para {forn_dash} não encontrados na simulação.')
        else:
            st.warning('Coluna "Classificação viária" não encontrada para análise por via.')


# ── Info do modelo ─────────────────────────────────────────────────────────────
with st.expander('ℹ️ Métricas dos Modelos Treinados'):
    cols = st.columns(3)
    targets_disp = ['lmed', 'uo', 'ul', 'emed', 'emin', 'w']
    for i, m in enumerate(targets_disp):
        with cols[i % 3]:
            st.markdown(f"**{TARGETS_MAP[m]}**")
            info = meta.get(f'modelo_{m}')
            if info:
                r2_val = info.get('r2')
                mae_val = info.get('mae')
                st.metric('R² (teste)', f"{r2_val:.4f}" if r2_val is not None else "—")
                st.caption(f"Erro Médio (MAE): {mae_val:.2f}" if mae_val is not None else "")
                st.caption(f"Algoritmo: {info.get('type','—')}")
            else:
                st.info("Modelo não treinado para este alvo.")
