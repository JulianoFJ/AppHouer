import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import io
import joblib
import plotly.graph_objects as go

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title='Previsão de Iluminação — Houer',
    page_icon='💡',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Estilo ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Paleta Houer: Navy (#1B3664), Teal (#00A9E0) */
    :root {
        --houer-navy: #1B3664;
        --houer-teal: #00A9E0;
        --bg-dark: #0b111e;
        --card-bg: #12192b;
        --card-border: #1f2937;
    }
    
    .stApp {
        background: radial-gradient(circle at 0% 0%, #1B366433, transparent), 
                    radial-gradient(circle at 100% 100%, #00A9E011, transparent),
                    #0b111e;
    }
    
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ffffff, var(--houer-teal));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0rem;
        letter-spacing: -1px;
    }
    .hero-sub {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .forn-card {
        background: rgba(18, 25, 43, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid var(--card-border);
        border-top: 4px solid var(--houer-teal);
        border-radius: 20px;
        padding: 1.8rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .forn-card:hover {
        transform: translateY(-8px);
        border-color: var(--houer-teal);
        background: rgba(27, 54, 100, 0.2);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    }
    .forn-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff; /* Valor sempre branco para destaque */
    }
    .metric-unit {
        font-size: 0.9rem;
        font-weight: 400;
        color: #64748b;
    }
    
    .section-title { 
        font-size: 1.5rem; 
        font-weight: 700; 
        color: #f8fafc; 
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
        background: var(--houer-teal);
        border-radius: 3px;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--houer-navy), #0b111e);
        border-right: 1px solid #1f2937;
    }
</style>
""", unsafe_allow_html=True)

# ── Caminhos ──────────────────────────────────────────────────────────────────
PASTA = os.path.dirname(os.path.abspath(__file__))
FEATURES_PATH = os.path.join(PASTA, 'features.json')

FORNECEDORES = ['LEDSTAR', 'SX LIGHTING', 'TECNOWATT']
CORES = {'LEDSTAR': '#00A9E0', 'SX LIGHTING': '#1B3664', 'TECNOWATT': '#606060'}

TARGETS_MAP = {
    'lmed': 'Luminância Média',
    'uo': 'Fator de Uniformidade',
    'ul': 'Uniformidade Longitudinal',
    'emed': 'Iluminância Média',
    'emin': 'Iluminância mínima horizontal E (lux)',
    'w': 'Potência (W)'
}
UNITS_MAP = {
    'lmed': 'cd/m²', 'uo': '', 'ul': '', 'emed': 'lux', 'emin': 'lux', 'w': 'W'
}

# ── Tabela NBR 5101 – Requisitos Mínimos por Subclasse ─────────────────────────
# Fonte: ABNT NBR 5101:2024
# M = Luminância (cd/m²) | C/P = Iluminância (lux)
NBR5101 = {
    # Vias Motorizadas (Lmed em cd/m², Uo, Ul)
    'M1': {'metricas': ['lmed','uo','ul','w'], 'lmed': 2.0, 'uo': 0.40, 'ul': 0.70},
    'M2': {'metricas': ['lmed','uo','ul','w'], 'lmed': 1.5, 'uo': 0.40, 'ul': 0.70},
    'M3': {'metricas': ['lmed','uo','ul','w'], 'lmed': 1.0, 'uo': 0.40, 'ul': 0.60},
    'M4': {'metricas': ['lmed','uo','ul','w'], 'lmed': 0.75,'uo': 0.40, 'ul': 0.60},
    'M5': {'metricas': ['lmed','uo','ul','w'], 'lmed': 0.50,'uo': 0.35, 'ul': 0.40},
    'M6': {'metricas': ['lmed','uo','ul','w'], 'lmed': 0.30,'uo': 0.35, 'ul': 0.40},
    # Áreas de Conflito (Emed em lux, Uo)
    'C0': {'metricas': ['emed','uo','w'], 'emed': 50.0, 'uo': 0.40},
    'C1': {'metricas': ['emed','uo','w'], 'emed': 30.0, 'uo': 0.40},
    'C2': {'metricas': ['emed','uo','w'], 'emed': 20.0, 'uo': 0.40},
    'C3': {'metricas': ['emed','uo','w'], 'emed': 15.0, 'uo': 0.35},
    'C4': {'metricas': ['emed','uo','w'], 'emed': 10.0, 'uo': 0.35},
    'C5': {'metricas': ['emed','uo','w'], 'emed':  5.0, 'uo': 0.35},
    # Vias Pedonais/Ciclovias (Emed e Emin em lux)
    'P1': {'metricas': ['emed','emin','w'], 'emed': 20.0, 'emin': 7.5},
    'P2': {'metricas': ['emed','emin','w'], 'emed': 15.0, 'emin': 5.0},
    'P3': {'metricas': ['emed','emin','w'], 'emed': 10.0, 'emin': 3.0},
    'P4': {'metricas': ['emed','emin','w'], 'emed':  7.5, 'emin': 1.5},
    'P5': {'metricas': ['emed','emin','w'], 'emed':  5.0, 'emin': 1.0},
    'P6': {'metricas': ['emed','emin','w'], 'emed':  3.0, 'emin': 0.6},
}

# ── Carrega modelos ───────────────────────────────────────────────────────────
@st.cache_resource
def carregar_modelos(suffix=""):
    meta_path = os.path.join(PASTA, f'features{suffix}.json')
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)

    modelos = {}
    for key in ['lmed', 'uo', 'ul', 'emed', 'emin', 'w']:
        path = os.path.join(PASTA, f'modelo_{key}{suffix}.pkl')
        if os.path.exists(path):
            modelos[key] = joblib.load(path)
    return modelos, meta

# ── Carrega banco de dados de luminarias ────────────────────────────────────
@st.cache_data
def carregar_banco_luminarias():
    """Lê a aba 'Banco de dados' das planilhas e retorna DataFrame com Fornecedor, Potência e Valor."""
    dfs = []
    for arq in os.listdir(PASTA):
        if not arq.endswith('.xlsx'):
            continue
        try:
            xl = pd.ExcelFile(os.path.join(PASTA, arq), engine='openpyxl')
            aba = next((n for n in xl.sheet_names if 'banco' in n.lower()), None)
            if not aba:
                continue
            df_raw = pd.read_excel(os.path.join(PASTA, arq), sheet_name=aba, header=1)
            df_raw = df_raw.dropna(how='all').dropna(axis=1, how='all')
            # A primeira linha contém os nomes reais
            df_raw.columns = df_raw.iloc[0]
            df_raw = df_raw[1:].reset_index(drop=True)
            # Padroniza nomes das colunas relevantes
            col_forn = next((c for c in df_raw.columns if 'forn' in str(c).lower()), None)
            col_pot  = next((c for c in df_raw.columns if 'pot' in str(c).lower() and '[w]' in str(c).lower()), None)
            col_lum  = next((c for c in df_raw.columns if 'lumin' in str(c).lower() and 'cod' not in str(c).lower() and 'consider' not in str(c).lower()), None)
            col_val  = next((c for c in df_raw.columns if str(c).strip().lower() == 'valor'), None)
            if not all([col_forn, col_pot, col_val]):
                continue
            df_sel = df_raw[[col_forn, col_pot, col_val]].copy()
            if col_lum:
                df_sel['Luminaria'] = df_raw[col_lum]
            df_sel.columns = ['Fornecedor', 'Potencia_W', 'Valor_R$'] + (['Luminaria'] if col_lum else [])
            df_sel = df_sel[df_sel['Fornecedor'].isin(['LEDSTAR', 'SX LIGHTING', 'TECNOWATT'])]
            df_sel['Potencia_W'] = pd.to_numeric(df_sel['Potencia_W'], errors='coerce')
            df_sel['Valor_R$']   = pd.to_numeric(df_sel['Valor_R$'],   errors='coerce')
            df_sel = df_sel.dropna(subset=['Potencia_W', 'Valor_R$'])
            dfs.append(df_sel)
        except Exception:
            continue
    if dfs:
        return pd.concat(dfs).drop_duplicates().reset_index(drop=True)
    return pd.DataFrame(columns=['Fornecedor', 'Potencia_W', 'Valor_R$'])

def buscar_custo(banco: pd.DataFrame, fornecedor: str, potencia_w: float):
    """Retorna (luminaria, potencia_real, valor) da luminaria mais próxima em potência."""
    sub = banco[banco['Fornecedor'] == fornecedor].copy()
    if sub.empty or potencia_w is None:
        return None, None, None
    idx_min = (sub['Potencia_W'] - potencia_w).abs().idxmin()
    row = sub.loc[idx_min]
    lum = row.get('Luminaria', '') if 'Luminaria' in sub.columns else ''
    return lum, row['Potencia_W'], row['Valor_R$']

# ── Carrega banco de dados de luminarias ────────────────────────────────────
banco_luminarias = carregar_banco_luminarias()

# ── Header ────────────────────────────────────────────────────────────────────
col_l, col_r = st.columns([1, 3])
with col_l:
    logo_path = os.path.join(PASTA, 'images.png')
    if os.path.exists(logo_path):
        st.image(logo_path, width=180)
    else:
        st.markdown('<div style="background:white; border-radius:12px; padding:10px; display:flex; justify-content:center; align-items:center; width:80px; height:80px;"><span style="color:#1B3664; font-size:24px; font-weight:900;">H<span style="color:#00A9E0;">O</span></span></div>', unsafe_allow_html=True)

with col_r:
    st.markdown('<p class="hero-title" style="margin-top:10px;">Houer</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub" style="margin-top:-15px; font-weight:600; color:var(--houer-teal);">impactando gerações</p>', unsafe_allow_html=True)
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
    num_ok = meta.get('features_numericas', [])
    cat_ok = meta.get('features_categoricas', [])

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
    dist_postes    = st.slider('Distância entre Postes (m)',  20.0, 60.0, 35.0, step=1.0)
    dist_poste_via = st.slider('Distância Poste à Via (m)',    0.0,  3.0,  0.5, step=0.25)
    altura_inst    = st.slider('Altura de Instalação (m)',     4.0, 16.0, 10.0, step=0.5)

    st.markdown('### 🔩 Configurações')
    tipo_estrutura = st.selectbox('Tipo de Estrutura', ['Braço', 'Suporte'])
    posteacao      = st.selectbox('Posteação', ['Unilateral', 'Canteiro central', 'Bilateral alternada', 'Bilateral frontal'])
    braco_novo     = st.selectbox('Braço Novo', ['Longo II', 'Longo I', 'Médio I', 'Médio II', 'Curto II', 'Curto I'])

# Métricas ativas baseadas na subclasse NBR 5101
metricas_ativas = NBR5101.get(subclasse, {}).get('metricas', ['emed', 'w'])

# ── Tabs Principais ───────────────────────────────────────────────────────────
tab_individual, tab_lote = st.tabs(['🎯 Simulação Individual', '📂 Simulação em Lote'])

# ==============================================================================
# TAB 1: SIMULAÇÃO INDIVIDUAL
# ==============================================================================
with tab_individual:
    def montar_entrada(fornecedor):
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
            'distancia entre postes':   dist_postes,
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
        colunas = num_ok + cat_ok
        return pd.DataFrame([{k: dados.get(k, np.nan) for k in colunas}])

    # Roda as predições
    resultados = {m: {} for m in metricas_ativas}
    for forn in FORNECEDORES:
        X_in = montar_entrada(forn)
        for m in metricas_ativas:
            if m in modelos:
                try:
                    val = modelos[m].predict(X_in)[0]
                    resultados[m][forn] = max(val, 0)
                except Exception as e:
                    print(f"Erro ao prever {m}: {e}")
                    resultados[m][forn] = None
            else:
                resultados[m][forn] = None

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
                    atende = val >= req_min
                    b_color = '#22c55e' if atende else '#ef4444'
                    b_txt   = '✔ Atende' if atende else '✘ Não Atende'
                    badge_html = f'<span style="font-size:.65rem;padding:2px 6px;border-radius:99px;background:{b_color};color:#fff;margin-left:6px;">{b_txt}</span>'
                
                html_content += f'<div class="metric-label"><span>{label}</span>{badge_html}</div>'
                html_content += f'<div class="metric-value">{val_str} <span class="metric-unit">{unit}</span></div>'
            
            html_content += "</div>"
            st.markdown(html_content, unsafe_allow_html=True)

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
                    c_html = f'<div style="background:rgba(27,54,100,0.4); backdrop-filter:blur(10px); border-radius:20px; padding:1.5rem; text-align:center; border:1px solid #2d4060; border-top:4px solid {cor}; height:100%;">'
                    c_html += f'<div style="font-size:.75rem; color:#94a3b8; text-transform:uppercase; letter-spacing:.1em; margin-bottom:1rem;">{forn}</div>'
                    c_html += f'<div style="font-size:.85rem; color:#f8fafc; font-weight:600; margin-bottom:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="{lum_nome}">{lum_nome}</div>'
                    c_html += f'<div style="font-size:.9rem; color:#94a3b8; margin-bottom:1rem;">{pot_real:.0f}W <span style="font-size:.7rem; color:#64748b;">({delta_str})</span></div>'
                    c_html += f'<div style="font-size:2.2rem; font-weight:800; color:white;">R$ {custo:,.2f}</div></div>'
                    st.markdown(c_html, unsafe_allow_html=True)
                else:
                    e_html = f'<div style="background:rgba(18,25,43,0.6); border-radius:20px; padding:1.5rem; text-align:center; border:1px solid #1f2937; border-top:4px solid {cor}; height:100%;">'
                    e_html += f'<div style="font-size:.75rem; color:#94a3b8; text-transform:uppercase;">{forn}</div><div style="font-size:1.1rem; color:#64748b; margin-top:1rem;">Sem dados no banco</div></div>'
                    st.markdown(e_html, unsafe_allow_html=True)

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
            yaxis=dict(title=titulo, gridcolor='#1f2937', zerolinecolor='#374151'),
            xaxis=dict(gridcolor='#1f2937'),
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
    
    colunas_template = num_ok.copy() + [c for c in cat_ok if c != 'Fornecedor']
    mapeamento_encoding = {
        'projecao do brao': 'projecao do braço',
        'Altura de Instalao': 'Altura de Instalação',
        'Brao Novo': 'Braço Novo'
    }
    colunas_amigaveis = [mapeamento_encoding.get(c, c) for c in colunas_template]
    
    # Adicionar coluna de Classificação
    colunas_amigaveis.insert(0, 'Classificacao (M/C/P)')
    
    df_template = pd.DataFrame(columns=colunas_amigaveis)
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
                df_saida = df_entrada.copy()
                map_inverso = {v: k for k, v in mapeamento_encoding.items()}
                df_pipeline = df_entrada.rename(columns=map_inverso)
                
                # Se a planilha não tiver Classificacao, usamos a do sidebar
                tem_classe = 'Classificacao (M/C/P)' in df_entrada.columns
                
                for forn in FORNECEDORES:
                    df_run = df_pipeline.copy()
                    df_run['Fornecedor'] = forn
                    for col in num_ok + cat_ok:
                        if col not in df_run.columns:
                            df_run[col] = np.nan
                            
                    # Prever todas as métricas para a facilidade de extração, 
                    # ou apenas as métricas dependendo da linha. Para simplificar no Lote, calculamos tudo.
                    for m in ['lmed', 'uo', 'ul', 'emed', 'emin', 'w']:
                        if m in modelos:
                            preds = modelos[m].predict(df_run)
                            
                            # Filtro dinâmico por linha baseada na classificação
                            if tem_classe:
                                classes = df_entrada['Classificacao (M/C/P)'].fillna('M').astype(str).str.upper().str[0]
                                # Máscara de validade
                                mask_valid = [False]*len(preds)
                                for i, c in enumerate(classes):
                                    if c == 'M' and m in ['lmed', 'uo', 'ul', 'w']: mask_valid[i] = True
                                    elif c == 'C' and m in ['emed', 'uo', 'w']: mask_valid[i] = True
                                    elif c == 'P' and m in ['emed', 'emin', 'w']: mask_valid[i] = True
                                    
                                preds = [p if valid else np.nan for p, valid in zip(preds, mask_valid)]
                            
                            df_saida[f'{TARGETS_MAP[m]} - {forn}'] = [max(p, 0) if pd.notna(p) else p for p in preds]

                st.markdown('### ✨ Resultados')
                st.dataframe(df_saida)
                
                buffer_resultado = io.BytesIO()
                with pd.ExcelWriter(buffer_resultado, engine='openpyxl') as writer:
                    df_saida.to_excel(writer, index=False)
                st.download_button('✅ Baixar Resultados (.xlsx)', data=buffer_resultado.getvalue(), file_name='resultados_simulacao.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', type='primary')

        except Exception as e:
            st.error(f'Erro ao processar: {str(e)}')

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
